import os
import random
import tempfile
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from Bio.PDB import MMCIFParser, PDBIO
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph

from contvar.config import ProjectConfig
from contvar.edges import SaladStyleEdgeBuilder


class GOSemanticTripletDataset(Dataset):
    """
    Dataset for GO semantic similarity pretraining (phase 0).

    Each sample is an (anchor, positive, negative) triplet built from
    semantic similarity scores:

    - positives: sim >= 0.8
    - negatives: sim <= 0.2, split into two bins for balanced hard negatives:
        * bin_low:   [0.0, 0.1)
        * bin_mid:   [0.1, 0.2]
    """

    # Global (per-process) index: structure_root -> {protein_id_lower: cif_path}.
    # This avoids calling `os.walk(...)` for every protein ID.
    _global_cif_path_index: Dict[str, Dict[str, str]] = {}
    # Global (per-process) index: prebuilt_graph_root -> {protein_id_lower: pt_path}.
    _global_pt_path_index: Dict[str, Dict[str, str]] = {}

    @staticmethod
    def _normalize_embedding_key(raw_id: str) -> str:
        key = raw_id.lower()
        if key.endswith("_model"):
            key = key[:-6]
        return key

    @staticmethod
    def _build_name_to_embedding_index(g) -> Dict[str, int]:
        node_order = []
        for node_name, _ in g.nodes(data=True):
            parts = node_name.split(":")
            node_chain = parts[0]
            node_resseq = int(parts[2])
            node_order.append((node_chain, node_resseq, node_name))
        node_order.sort(key=lambda x: (x[0], x[1]))
        return {name: i for i, (_, _, name) in enumerate(node_order)}

    def __init__(
        self,
        tsv_path: str,
        ontology: str,
        config: ProjectConfig,
        structure_root: str,
        esm2_embeddings: Optional[Dict[str, np.ndarray]] = None,
        file_exts: Tuple[str, ...] = (".cif", ".pdb"),
        sim_col: Optional[str] = None,
        pos_threshold: float = 0.8,
        neg_low: float = 0.0,
        neg_mid: float = 0.1,
        neg_high: float = 0.2,
        phase0_split: Optional[Literal["train", "val", "test"]] = None,
        protein_to_split: Optional[Dict[str, str]] = None,
        prebuilt_graph_root: Optional[str] = None,
        build_graph_if_missing: bool = True,
    ):
        super().__init__()
        self.tsv_path = tsv_path
        self.ontology = ontology
        self.config = config
        self.structure_root = structure_root
        # Optional per-protein ESM2 embeddings, keyed by protein ID.
        # When provided, we will append these to the node feature vectors so
        # that the resulting feature dimension matches the main model's
        # expected input_dim (e.g. 20 AA one-hot + 1280-dim embedding).
        self.esm2_embeddings = esm2_embeddings or {}
        self.file_exts = file_exts

        if sim_col is None:
            sim_col = f"sim_{ontology}"
        self.sim_col = sim_col

        self.pos_threshold = pos_threshold
        self.neg_low = neg_low
        self.neg_mid = neg_mid
        self.neg_high = neg_high
        self.phase0_split = phase0_split
        self.protein_to_split = protein_to_split
        self.prebuilt_graph_root = prebuilt_graph_root
        self.build_graph_if_missing = build_graph_if_missing

        # Graph-building helpers (SALAD-style by default)
        self.node_metadata_funcs = self.config.get_active_node_metadata_funcs()
        self.node_attributes = self.config.get_node_attributes_list()
        self.salad_edge_builder: Optional[SaladStyleEdgeBuilder] = None
        self.edge_funcs = []

        if self.config.edge_mode == "salad":
            self.salad_edge_builder = self.config.get_salad_edge_builder()
            print(
                f"[GO-{ontology}] Using SALAD-style edges: index={config.salad_num_index}, "
                f"spatial={config.salad_num_spatial}, random={config.salad_num_random}"
            )
        else:
            self.edge_funcs = self.config.get_active_edge_funcs()
            print(
                f"[GO-{ontology}] Using Graphein edges: kNN={config.edge_knn} (k={config.knn_k}), "
                f"distance={config.edge_distance} (thresh={config.dist_threshold})"
            )

        # Parsed triplets (anchor_id, pos_id, neg_id)
        self.triplets: List[Tuple[str, str, str]] = []

        # Simple in-memory cache for graphs
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

        # In prebuilt-only mode, keep only triplets whose all proteins
        # (anchor/positive/negative) exist as prebuilt .pt graphs.
        if self.prebuilt_graph_root and not self.build_graph_if_missing:
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
                f"[GO-{self.ontology}] Prebuilt-only filter: "
                f"{before:,} -> {len(self.triplets):,} triplets "
                f"(available proteins={len(available_ids):,})"
            )

        # Optional subsampling to keep phase-0 manageable on limited hardware
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

    # ------------------------------------------------------------------
    # TSV parsing and triplet construction
    # ------------------------------------------------------------------
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

        # Build simple triplet index with balanced negatives
        for anchor_id, buckets in anchors.items():
            pos_list = buckets["pos"]
            neg_low = buckets["neg_low"]
            neg_mid = buckets["neg_mid"]

            if not pos_list:
                continue
            if not (neg_low or neg_mid):
                continue

            # For each positive, we can pair a balanced negative
            for pos_id, _ in pos_list:
                # Decide which bin to sample from (target: equal count)
                # We do this per-sample with simple availability checks.
                has_low = len(neg_low) > 0
                has_mid = len(neg_mid) > 0

                if has_low and has_mid:
                    # 50-50 choice between bins
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

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    def _id_to_path(self, protein_id: str) -> Optional[str]:
        """
        Resolve a protein ID to a structure file path.

        We look recursively under structure_root for files whose name starts
        with the ID and ends with .cif, e.g. O57562_wt_model.cif.
        """
        if not self.structure_root:
            return None

        # Many Alphafold/UniProt downloads use lowercase IDs in filenames,
        # while TSVs may contain uppercase IDs. Normalise to lowercase for
        # matching so that, e.g., "A0A009IHW8" matches "a0a009ihw8_wt_model.cif".
        pid_lower = protein_id.lower()

        root_key = os.path.abspath(self.structure_root)
        index = self.__class__._global_cif_path_index.get(root_key)
        if index is None:
            index = {}
            for root, _, files in os.walk(self.structure_root):
                for fname in files:
                    if not fname.lower().endswith(".cif"):
                        continue
                    # Expected pattern: "<id>_*.cif" (e.g. a0a009ihw8_wt_model.cif)
                    # Map by prefix before the first underscore.
                    base = os.path.splitext(fname)[0].lower()
                    prefix = base.split("_", 1)[0]
                    if prefix:
                        index[prefix] = os.path.join(root, fname)
            self.__class__._global_cif_path_index[root_key] = index

        return index.get(pid_lower)

    def _id_to_prebuilt_graph_path(self, protein_id: str) -> Optional[str]:
        if not self.prebuilt_graph_root:
            return None
        if not os.path.exists(self.prebuilt_graph_root):
            return None

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
        if not self.prebuilt_graph_root:
            return set()
        if not os.path.exists(self.prebuilt_graph_root):
            return set()

        root_key = os.path.abspath(self.prebuilt_graph_root)
        index = self.__class__._global_pt_path_index.get(root_key)
        if index is None:
            # Build index once via existing helper path.
            self._id_to_prebuilt_graph_path("__index_warmup__")
            index = self.__class__._global_pt_path_index.get(root_key, {})
        return set(index.keys())

    def _build_graph(self, protein_id: str) -> Optional[Data]:
        # First try prebuilt .pt graphs if a folder is configured.
        prebuilt_path = self._id_to_prebuilt_graph_path(protein_id)
        if prebuilt_path is not None:
            try:
                data = torch.load(prebuilt_path, weights_only=False)
                if isinstance(data, Data):
                    return data
                return None
            except Exception:
                return None

        # Optional fallback: build from CIF files when prebuilt graph is missing.
        if not self.build_graph_if_missing:
            return None

        path = self._id_to_path(protein_id)
        if path is None:
            return None

        protein_code = os.path.basename(path).split(".")[0]
        temp_pdb_path = None

        try:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(protein_code, path)

            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                temp_pdb_path = tmp.name

            pdb_io = PDBIO()
            pdb_io.set_structure(structure)
            pdb_io.save(temp_pdb_path)

            if self.config.edge_mode == "salad":
                edge_funcs = []
            else:
                edge_funcs = self.edge_funcs

            g_config = ProteinGraphConfig(
                edge_construction_functions=edge_funcs,
                node_metadata_functions=self.node_metadata_funcs,
                verbose=False,
            )
            g = construct_graph(config=g_config, path=temp_pdb_path, verbose=False)

            if g is None or len(g.nodes()) == 0:
                return None

            protein_embedding = None
            if self.esm2_embeddings:
                emb_key = self._normalize_embedding_key(protein_id)
                protein_embedding = self.esm2_embeddings.get(emb_key)
            if self.esm2_embeddings and protein_embedding is None:
                return None

            name_to_emb_idx = None
            if protein_embedding is not None:
                name_to_emb_idx = self._build_name_to_embedding_index(g)

            # Node features
            node_features = []
            coords_list = []
            for n, d in g.nodes(data=True):
                if protein_embedding is not None:
                    emb_idx = name_to_emb_idx.get(n) if name_to_emb_idx is not None else None
                    if emb_idx is None or emb_idx < 0 or emb_idx >= len(protein_embedding):
                        return None
                    d["embedding"] = protein_embedding[emb_idx]

                fv = []
                for k in self.node_attributes:
                    v = d.get(k)
                    if v is None:
                        continue
                    if isinstance(v, (list, np.ndarray)):
                        fv.extend(list(v))
                    else:
                        fv.append(v)

                node_features.append(fv)
                coords_list.append(d["coords"])

            x = torch.tensor(node_features, dtype=torch.float)
            pos = torch.tensor([c.tolist() for c in coords_list], dtype=torch.float)

            data = Data()
            data.x = x
            data.pos = pos

            # SALAD-style edges
            if self.config.edge_mode == "salad":
                if self.salad_edge_builder is None:
                    raise RuntimeError("SALAD edge mode selected but builder is None.")

                coords_array = np.stack(coords_list, axis=0)
                residue_indices = np.arange(len(coords_list), dtype=np.int32)
                chain_ids = np.zeros_like(residue_indices)

                edge_index, edge_attr = self.salad_edge_builder.build_edge_index_and_attr(
                    coords=coords_array,
                    residue_indices=residue_indices,
                    chain_ids=chain_ids,
                    use_rbf=True,
                    num_rbf=self.config.salad_num_rbf,
                    d_max=self.config.salad_d_max,
                )

                data.edge_index = edge_index
                data.edge_attr = edge_attr
            else:
                # Graphein-style edges are already in the graph; keep it simple and
                # only use node features in phase-0 if needed.
                data.edge_index = torch.empty((2, 0), dtype=torch.long)
                data.edge_attr = torch.empty((0, 0), dtype=torch.float)

            if data.edge_index.numel() > 0:
                data.edge_index, data.edge_attr = to_undirected(
                    data.edge_index, data.edge_attr
                )

            return data
        except Exception:
            return None
        finally:
            if temp_pdb_path and os.path.exists(temp_pdb_path):
                os.remove(temp_pdb_path)

    def _get_graph(self, protein_id: str) -> Optional[Data]:
        if protein_id in self.graph_cache:
            return self.graph_cache[protein_id]

        g = self._build_graph(protein_id)
        if g is not None:
            self.graph_cache[protein_id] = g
        return g

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        """
        Robust __getitem__ with limited resampling.

        Many semantic-similarity IDs may not have corresponding CIF/PDB
        structures. Instead of recursing indefinitely when graphs are
        missing, we try a fixed number of random resamples, then raise.
        """
        max_attempts = 20
        for _ in range(max_attempts):
            anchor_id, pos_id, neg_id = self.triplets[idx]

            g_a = self._get_graph(anchor_id)
            g_p = self._get_graph(pos_id)
            g_n = self._get_graph(neg_id)

            if g_a is not None and g_p is not None and g_n is not None:
                return g_a, g_p, g_n

            # Pick a new random index and try again
            idx = random.randint(0, len(self.triplets) - 1)

        raise RuntimeError(
            f"[GO-{self.ontology}] Failed to load graphs after {max_attempts} attempts. "
            f"Check that CIF/PDB files exist for the IDs in {self.tsv_path} under {self.structure_root}."
        )

