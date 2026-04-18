import os
import random
from typing import Dict, List, Tuple, Optional, Literal

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from contvar.config import ProjectConfig


class GOSemanticTripletDataset(Dataset):
    """
    Dataset for GO semantic similarity pretraining (phase 0).

    Loads protein graphs only from prebuilt PyG ``.pt`` files under
    ``prebuilt_graph_root`` (no CIF / on-the-fly graph construction).

    Each sample is an (anchor, positive, negative) triplet from semantic
    similarity TSVs:

    - positives: sim >= 0.8
    - negatives: sim <= 0.2, split into two bins for balanced hard negatives:
        * bin_low:   [0.0, 0.1)
        * bin_mid:   [0.1, 0.2]
    """

    _global_pt_path_index: Dict[str, Dict[str, str]] = {}

    def __init__(
        self,
        tsv_path: str,
        ontology: str,
        config: ProjectConfig,
        prebuilt_graph_root: str,
        sim_col: Optional[str] = None,
        pos_threshold: float = 0.8,
        neg_low: float = 0.0,
        neg_mid: float = 0.1,
        neg_high: float = 0.2,
        phase0_split: Optional[Literal["train", "val", "test"]] = None,
        protein_to_split: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        if not prebuilt_graph_root or not os.path.isdir(prebuilt_graph_root):
            raise FileNotFoundError(
                f"[GO-{ontology}] prebuilt_graph_root must be an existing directory: "
                f"{prebuilt_graph_root!r}"
            )

        self.tsv_path = tsv_path
        self.ontology = ontology
        self.config = config
        self.prebuilt_graph_root = prebuilt_graph_root

        if sim_col is None:
            sim_col = f"sim_{ontology}"
        self.sim_col = sim_col

        self.pos_threshold = pos_threshold
        self.neg_low = neg_low
        self.neg_mid = neg_mid
        self.neg_high = neg_high
        self.phase0_split = phase0_split
        self.protein_to_split = protein_to_split

        print(
            f"[GO-{ontology}] Phase-0 graphs: prebuilt .pt only from {prebuilt_graph_root}"
        )

        self.triplets: List[Tuple[str, str, str]] = []
        self.graph_cache: Dict[str, Data] = {}

        self._parse_tsv()

        if self.phase0_split and self.protein_to_split:
            from contvar.go_identity_split import filter_triplets_by_split

            before = len(self.triplets)
            self.triplets = filter_triplets_by_split(
                self.triplets, self.protein_to_split, self.phase0_split
            )
            print(
                f"[GO-{self.ontology}] Split={self.phase0_split}: "
                f"{before:,} -> {len(self.triplets):,} triplets (identity filter)"
            )

        before = len(self.triplets)
        available_ids = self._get_available_prebuilt_ids()
        self.triplets = [
            (a, p, n)
            for (a, p, n) in self.triplets
            if a.lower() in available_ids
            and p.lower() in available_ids
            and n.lower() in available_ids
        ]
        print(
            f"[GO-{self.ontology}] Prebuilt .pt filter: "
            f"{before:,} -> {len(self.triplets):,} triplets "
            f"(available proteins={len(available_ids):,})"
        )

        max_triplets = getattr(self.config, "go_max_triplets_per_ontology", None)
        if max_triplets is not None and len(self.triplets) > max_triplets:
            original_len = len(self.triplets)
            seed = int(getattr(self.config, "go_split_seed", 42))
            if self.phase0_split:
                seed += {"train": 0, "val": 1, "test": 2}[self.phase0_split]
            rng = random.Random(seed)
            self.triplets = rng.sample(self.triplets, max_triplets)
            print(
                f"[GO-{self.ontology}] Subsampled triplets: {original_len:,} -> "
                f"{len(self.triplets):,} (max={max_triplets})"
            )

    def _parse_tsv(self):
        if not os.path.exists(self.tsv_path):
            print(f"[GO-{self.ontology}] TSV not found: {self.tsv_path} (skipping)")
            return

        import csv

        anchors: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}

        with open(self.tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                anchor_id = row["anchor"]
                cand_id = row["candidate"]
                try:
                    sim_val = float(row[self.sim_col])
                except (KeyError, ValueError):
                    continue

                if anchor_id not in anchors:
                    anchors[anchor_id] = {
                        "pos": [],
                        "neg_low": [],
                        "neg_mid": [],
                    }

                if sim_val >= self.pos_threshold:
                    anchors[anchor_id]["pos"].append((cand_id, sim_val))
                elif self.neg_low <= sim_val < self.neg_mid:
                    anchors[anchor_id]["neg_low"].append((cand_id, sim_val))
                elif self.neg_mid <= sim_val <= self.neg_high:
                    anchors[anchor_id]["neg_mid"].append((cand_id, sim_val))

        for anchor_id, buckets in anchors.items():
            pos_list = buckets["pos"]
            neg_low = buckets["neg_low"]
            neg_mid = buckets["neg_mid"]

            if not pos_list:
                continue
            if not (neg_low or neg_mid):
                continue

            for pos_id, _ in pos_list:
                has_low = len(neg_low) > 0
                has_mid = len(neg_mid) > 0

                if has_low and has_mid:
                    if random.random() < 0.5:
                        neg_id, _ = random.choice(neg_low)
                    else:
                        neg_id, _ = random.choice(neg_mid)
                elif has_low:
                    neg_id, _ = random.choice(neg_low)
                elif has_mid:
                    neg_id, _ = random.choice(neg_mid)
                else:
                    continue

                self.triplets.append((anchor_id, pos_id, neg_id))

        print(
            f"[GO-{self.ontology}] Parsed {len(self.triplets):,} triplets from {self.tsv_path}"
        )

    def _id_to_prebuilt_graph_path(self, protein_id: str) -> Optional[str]:
        pid_lower = protein_id.lower()
        root_key = os.path.abspath(self.prebuilt_graph_root)
        index = self.__class__._global_pt_path_index.get(root_key)
        if index is None:
            index = {}
            for root, _, files in os.walk(self.prebuilt_graph_root):
                for fname in files:
                    if not fname.lower().endswith(".pt"):
                        continue
                    base = os.path.splitext(fname)[0].lower()
                    prefix = base.split("_", 1)[0]
                    if prefix:
                        index[prefix] = os.path.join(root, fname)
            self.__class__._global_pt_path_index[root_key] = index

        return index.get(pid_lower)

    def _get_available_prebuilt_ids(self) -> set:
        root_key = os.path.abspath(self.prebuilt_graph_root)
        index = self.__class__._global_pt_path_index.get(root_key)
        if index is None:
            self._id_to_prebuilt_graph_path("__index_warmup__")
            index = self.__class__._global_pt_path_index.get(root_key, {})
        return set(index.keys())

    def _load_prebuilt_graph(self, protein_id: str) -> Optional[Data]:
        prebuilt_path = self._id_to_prebuilt_graph_path(protein_id)
        if prebuilt_path is None:
            return None
        try:
            data = torch.load(prebuilt_path, weights_only=False)
            if isinstance(data, Data):
                return data
            return None
        except Exception:
            return None

    def _get_graph(self, protein_id: str) -> Optional[Data]:
        if protein_id in self.graph_cache:
            return self.graph_cache[protein_id]

        g = self._load_prebuilt_graph(protein_id)
        if g is not None:
            self.graph_cache[protein_id] = g
        return g

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        max_attempts = 20
        for _ in range(max_attempts):
            anchor_id, pos_id, neg_id = self.triplets[idx]

            g_a = self._get_graph(anchor_id)
            g_p = self._get_graph(pos_id)
            g_n = self._get_graph(neg_id)

            if g_a is not None and g_p is not None and g_n is not None:
                return g_a, g_p, g_n

            idx = random.randint(0, len(self.triplets) - 1)

        raise RuntimeError(
            f"[GO-{self.ontology}] Failed to load .pt graphs after {max_attempts} attempts. "
            f"Check prebuilt files for IDs in {self.tsv_path} under {self.prebuilt_graph_root}."
        )
