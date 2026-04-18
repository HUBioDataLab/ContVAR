import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from torch_geometric.data import Batch

from contvar.config import ProjectConfig
from contvar.data.go_dataset import GOSemanticTripletDataset
from contvar.go_identity_split import resolve_phase0_split

# Ontology order matches meeting pseudocode (MF → BP → CC).
_GO_ONTOLOGY_ORDER: Tuple[str, ...] = ("mf", "bp", "cc")


def _triplet_loss(anchor, positive, negative, margin: float):
    d_pos = F.pairwise_distance(anchor, positive, p=2)
    d_neg = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_pos - d_neg + margin).mean(), d_pos, d_neg


def _infinite_batches(loader: DataLoader):
    """Yield batches forever; each pass over the loader gets a fresh shuffle."""
    while True:
        for batch in loader:
            yield batch


def _normalize_sampling_ratio(
    raw_ratio, active_ontologies: List[str]
) -> Dict[str, float]:
    """
    Build a valid probability map over active ontologies.
    Supports dict-like {"mf": 0.6, "bp": 0.2, "cc": 0.2} or list/tuple in
    ontology order (mf, bp, cc). Falls back to uniform if invalid.
    """
    if not active_ontologies:
        return {}

    weights: Dict[str, float] = {ont: 0.0 for ont in active_ontologies}

    if isinstance(raw_ratio, dict):
        for ont in active_ontologies:
            try:
                weights[ont] = max(float(raw_ratio.get(ont, 0.0)), 0.0)
            except (TypeError, ValueError):
                weights[ont] = 0.0
    elif isinstance(raw_ratio, (tuple, list)):
        for ont, w in zip(_GO_ONTOLOGY_ORDER, raw_ratio):
            if ont not in weights:
                continue
            try:
                weights[ont] = max(float(w), 0.0)
            except (TypeError, ValueError):
                weights[ont] = 0.0

    total = sum(weights.values())
    if total <= 0:
        uniform = 1.0 / float(len(active_ontologies))
        return {ont: uniform for ont in active_ontologies}

    return {ont: weights[ont] / total for ont in active_ontologies}


def _weighted_pick_ontology(
    candidates: List[str], ratio_map: Dict[str, float], rng: random.Random
) -> str:
    """Pick one ontology from candidates using ratio_map weights."""
    if len(candidates) == 1:
        return candidates[0]

    weights = [max(float(ratio_map.get(ont, 0.0)), 0.0) for ont in candidates]
    total = sum(weights)
    if total <= 0:
        weights = [1.0 for _ in candidates]
        total = float(len(candidates))

    pick = rng.random() * total
    running = 0.0
    for ont, w in zip(candidates, weights):
        running += w
        if pick <= running:
            return ont
    return candidates[-1]


def _triplet_batch_size(batch_triplet: Tuple[Batch, Batch, Batch]) -> int:
    ba, _, _ = batch_triplet
    return int(getattr(ba, "num_graphs", 0)) or 0


def _collect_prebuilt_protein_ids(prebuilt_graph_root: str) -> List[str]:
    ids = set()
    if not prebuilt_graph_root or not os.path.exists(prebuilt_graph_root):
        return []
    for root, _, files in os.walk(prebuilt_graph_root):
        for fname in files:
            if not fname.lower().endswith(".pt"):
                continue
            base = os.path.splitext(fname)[0].lower()
            pid = base.split("_", 1)[0]
            if pid:
                ids.add(pid)
    return sorted(ids)


def _build_random_prebuilt_split(cfg: ProjectConfig, protein_ids: List[str]) -> Dict[str, str]:
    rng = random.Random(int(getattr(cfg, "go_split_seed", 42)))
    ids = list(protein_ids)
    rng.shuffle(ids)

    rt = float(getattr(cfg, "go_train_ratio", 0.8))
    rv = float(getattr(cfg, "go_val_ratio", 0.1))
    rte = float(getattr(cfg, "go_test_ratio", 0.1))
    total = rt + rv + rte
    if total <= 0:
        rt, rv, rte = 0.8, 0.1, 0.1
        total = 1.0
    rt, rv, rte = rt / total, rv / total, rte / total

    n = len(ids)
    n_train = int(n * rt)
    n_val = int(n * rv)
    n_test = n - n_train - n_val

    split_map: Dict[str, str] = {}
    for pid in ids[:n_train]:
        split_map[pid] = "train"
    for pid in ids[n_train:n_train + n_val]:
        split_map[pid] = "val"
    for pid in ids[n_train + n_val:n_train + n_val + n_test]:
        split_map[pid] = "test"
    return split_map


def _compute_go_phase0_loss(
    model,
    batch_dict: Dict[str, Tuple[Batch, Batch, Batch]],
    device: torch.device,
    margin: float,
    ontologies: List[str],
):
    """
    Pseudocode-style GO loss: average triplet losses over ontologies present
    in this step (each uses its own head via forward_go_head).
    """
    losses = []
    per_ont = {}

    for ont in ontologies:
        triplet = batch_dict.get(ont)
        if triplet is None:
            continue
        ba, bpos, bneg = triplet
        ba = ba.to(device)
        bpos = bpos.to(device)
        bneg = bneg.to(device)

        za = model.forward_go_head(ba, ont)
        zp = model.forward_go_head(bpos, ont)
        zn = model.forward_go_head(bneg, ont)

        loss_ont, d_pos, d_neg = _triplet_loss(za, zp, zn, margin=margin)
        losses.append(loss_ont)
        per_ont[ont] = {
            "loss": loss_ont.item(),
            "avg_pos_dist": d_pos.mean().item(),
            "avg_neg_dist": d_neg.mean().item(),
            "dist_margin": (d_neg.mean() - d_pos.mean()).item(),
        }

    if not losses:
        return None, per_ont

    total = sum(losses) / len(losses)
    return total, per_ont


def _go_collate(batch):
    """
    Collate function for GO phase-0 triplets.

    Each item coming from the dataset is a simple (anchor, positive, negative)
    tuple of torch_geometric.data.Data objects. We need to convert each column
    into a separate Batch so the model can process them.
    """
    # Filter out any None entries (in case dataset decides to skip samples)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    anchors, positives, negatives = zip(*batch)
    ba = Batch.from_data_list(list(anchors))
    bp = Batch.from_data_list(list(positives))
    bn = Batch.from_data_list(list(negatives))
    return ba, bp, bn


def _mean_eval_loss_for_loaders(
    model,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    margin: float,
    ontologies: List[str],
) -> Optional[float]:
    """Average per-batch loss across all batches and ontologies (eval mode)."""
    model.eval()
    batch_losses: List[float] = []
    with torch.no_grad():
        for ont in ontologies:
            loader = loaders.get(ont)
            if loader is None:
                continue
            for batch in loader:
                if batch is None:
                    continue
                batch_dict = {ont: batch}
                loss, _ = _compute_go_phase0_loss(
                    model, batch_dict, device, margin, [ont]
                )
                if loss is not None:
                    batch_losses.append(loss.item())
    if not batch_losses:
        return None
    return sum(batch_losses) / len(batch_losses)


def _build_go_loader(
    tsv_path: str,
    ontology: str,
    cfg: ProjectConfig,
    prebuilt_graph_root: str,
    shuffle: bool,
    phase0_split: Optional[str] = None,
    protein_to_split: Optional[dict] = None,
) -> Optional[DataLoader]:
    dataset = GOSemanticTripletDataset(
        tsv_path=tsv_path,
        ontology=ontology,
        config=cfg,
        prebuilt_graph_root=prebuilt_graph_root,
        phase0_split=phase0_split,
        protein_to_split=protein_to_split,
    )
    if len(dataset) == 0:
        return None

    loader = DataLoader(
        dataset,
        batch_size=cfg.go_batch_size,
        shuffle=shuffle,
        collate_fn=_go_collate,
        num_workers=getattr(cfg, "go_num_workers", 0),
    )
    return loader


def run_go_pretraining(model, cfg: ProjectConfig, device: torch.device):
    """
    Phase-0 GO semantic pretraining.

    Uses semantic similarity triplets to train MF/BP/CC heads on top of the
    shared encoder.
    """
    if cfg.go_phase0_epochs <= 0:
        return

    print("\n=== Phase 0: GO Semantic Similarity Pretraining ===")

    prebuilt_graph_root = getattr(cfg, "go_prebuilt_graph_root", None)
    if not prebuilt_graph_root or not os.path.isdir(prebuilt_graph_root):
        raise FileNotFoundError(
            "Phase 0 requires go_prebuilt_graph_root to be set to a directory of prebuilt "
            f"PyG graph .pt files (got {prebuilt_graph_root!r})."
        )
    print(f"[Phase0] Prebuilt GO graphs: {prebuilt_graph_root}")

    # Resolve TSV paths
    tsv_dir = cfg.go_tsv_dir
    mf_tsv = os.path.join(
        tsv_dir, "semantic_similarity_swissprot_filtered_low0.2_high0.8_mf.tsv"
    )
    bp_tsv = os.path.join(
        tsv_dir, "semantic_similarity_swissprot_filtered_low0.2_high0.8_bp.tsv"
    )
    cc_tsv = os.path.join(
        tsv_dir, "semantic_similarity_swissprot_filtered_low0.2_high0.8_cc.tsv"
    )

    protein_to_split: Optional[dict] = None
    if bool(getattr(cfg, "go_random_split_from_prebuilt", False)):
        prebuilt_ids = _collect_prebuilt_protein_ids(prebuilt_graph_root)
        protein_to_split = _build_random_prebuilt_split(cfg, prebuilt_ids)
        n_train = sum(1 for s in protein_to_split.values() if s == "train")
        n_val = sum(1 for s in protein_to_split.values() if s == "val")
        n_test = sum(1 for s in protein_to_split.values() if s == "test")
        print(
            f"[Phase0] Prebuilt random split from proteins: total={len(protein_to_split):,} "
            f"| train={n_train:,} val={n_val:,} test={n_test:,}"
        )
    elif getattr(cfg, "go_split_mode", "none") == "identity_grouped":
        protein_to_split, _ = resolve_phase0_split(cfg, mf_tsv, bp_tsv, cc_tsv)

    def make_loaders_for_split(split_name: Optional[str], shuffle: bool):
        out: Dict[str, DataLoader] = {}
        for ont, path in [("mf", mf_tsv), ("bp", bp_tsv), ("cc", cc_tsv)]:
            if not os.path.exists(path):
                continue
            ps = split_name if protein_to_split else None
            pt = protein_to_split if protein_to_split else None
            loader = _build_go_loader(
                tsv_path=path,
                ontology=ont,
                cfg=cfg,
                prebuilt_graph_root=prebuilt_graph_root,
                shuffle=shuffle,
                phase0_split=ps,
                protein_to_split=pt,
            )
            if loader is not None:
                out[ont] = loader
        return out

    if protein_to_split:
        train_loaders = make_loaders_for_split("train", shuffle=True)
        val_loaders = make_loaders_for_split("val", shuffle=False)
        test_loaders = make_loaders_for_split("test", shuffle=False)
        loaders = train_loaders
        for split_label, ld in (
            ("train", train_loaders),
            ("val", val_loaders),
            ("test", test_loaders),
        ):
            for ont in _GO_ONTOLOGY_ORDER:
                if ont not in ld:
                    continue
                n = len(ld[ont].dataset)
                print(f"[Phase0] {split_label} triplets [{ont}]: {n:,}")
                wandb.log({f"phase0/split/{split_label}_triplets_{ont}": n})
    else:
        loaders = {}
        for ont, path in [("mf", mf_tsv), ("bp", bp_tsv), ("cc", cc_tsv)]:
            if os.path.exists(path):
                loader = _build_go_loader(
                    tsv_path=path,
                    ontology=ont,
                    cfg=cfg,
                    prebuilt_graph_root=prebuilt_graph_root,
                    shuffle=True,
                )
                if loader is not None:
                    loaders[ont] = loader
        val_loaders = {}
        test_loaders = {}

    if not loaders:
        print("No GO loaders constructed for phase 0, skipping.")
        return

    # Stable order for averaging (only ontologies that have a loader).
    active_ontologies = [o for o in _GO_ONTOLOGY_ORDER if o in loaders]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.go_lr, weight_decay=cfg.weight_decay
    )

    model.to(device)

    sampling_enabled = bool(getattr(cfg, "go_sampling_enabled", False))
    ratio_map = _normalize_sampling_ratio(
        getattr(cfg, "go_sampling_ratio", None), active_ontologies
    )
    log_sampling_stats = bool(getattr(cfg, "go_log_sampling_stats", True))
    if sampling_enabled:
        ratio_txt = ", ".join(
            f"{ont}:{ratio_map.get(ont, 0.0):.2f}" for ont in active_ontologies
        )
        print(f"[Phase0] Sampling mode enabled | ratios={ratio_txt}")

    for epoch in range(cfg.go_phase0_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        gens: Dict[str, Iterator] = {
            ont: _infinite_batches(loaders[ont]) for ont in active_ontologies
        }
        if sampling_enabled:
            # Keep epoch throughput comparable to previous loop:
            # previously each step consumed all ontologies.
            n_steps = sum(len(loaders[ont]) for ont in active_ontologies)
        else:
            # Legacy behavior: one batch per ontology per step.
            n_steps = max(len(loaders[ont]) for ont in active_ontologies)
        rng = random.Random(int(getattr(cfg, "go_split_seed", 42)) + epoch)
        sampled_step_counts = {ont: 0 for ont in active_ontologies}
        sampled_batch_counts = {ont: 0 for ont in active_ontologies}

        pbar = tqdm(
            range(n_steps),
            desc=f"Phase0 Epoch {epoch+1}",
            leave=False,
        )
        for _ in pbar:
            batch_dict = {}
            if sampling_enabled:
                tried = set()
                while len(tried) < len(active_ontologies):
                    remaining = [o for o in active_ontologies if o not in tried]
                    ont = _weighted_pick_ontology(remaining, ratio_map, rng)
                    tried.add(ont)
                    b = next(gens[ont])
                    if b is None:
                        continue
                    batch_dict[ont] = b
                    sampled_step_counts[ont] += 1
                    sampled_batch_counts[ont] += _triplet_batch_size(b)
                    break
            else:
                for ont in active_ontologies:
                    b = next(gens[ont])
                    if b is not None:
                        batch_dict[ont] = b
                        sampled_step_counts[ont] += 1
                        sampled_batch_counts[ont] += _triplet_batch_size(b)

            loss, per_ont = _compute_go_phase0_loss(
                model,
                batch_dict,
                device,
                cfg.go_margin,
                active_ontologies,
            )
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1

            log_dict = {
                "phase0/combined_batch_loss": loss.item(),
                "phase0/n_ontologies_in_batch": len(per_ont),
                "phase0/epoch": epoch + 1,
            }
            for ont, st in per_ont.items():
                log_dict[f"phase0/{ont}/batch_loss"] = st["loss"]
                log_dict[f"phase0/{ont}/avg_pos_dist"] = st["avg_pos_dist"]
                log_dict[f"phase0/{ont}/avg_neg_dist"] = st["avg_neg_dist"]
                log_dict[f"phase0/{ont}/dist_margin"] = st["dist_margin"]
            wandb.log(log_dict)

        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
        else:
            avg_loss = 0.0

        wandb.log({"phase0/epoch_loss": avg_loss, "phase0/epoch": epoch + 1})
        if log_sampling_stats:
            total_sampled_steps = sum(sampled_step_counts.values())
            total_sampled_batches = sum(sampled_batch_counts.values())
            sampling_log = {"phase0/epoch": epoch + 1}
            for ont in active_ontologies:
                sampling_log[f"phase0/sampling/steps_{ont}"] = sampled_step_counts[ont]
                sampling_log[f"phase0/sampling/samples_{ont}"] = sampled_batch_counts[ont]
                sampling_log[f"phase0/sampling/target_ratio_{ont}"] = ratio_map.get(
                    ont, 0.0
                )
                sampling_log[f"phase0/sampling/actual_step_ratio_{ont}"] = (
                    (sampled_step_counts[ont] / total_sampled_steps)
                    if total_sampled_steps > 0
                    else 0.0
                )
                sampling_log[f"phase0/sampling/actual_sample_ratio_{ont}"] = (
                    (sampled_batch_counts[ont] / total_sampled_batches)
                    if total_sampled_batches > 0
                    else 0.0
                )
            wandb.log(sampling_log)
        print(
            f"[Phase0] Epoch {epoch+1}/{cfg.go_phase0_epochs} | "
            f"Avg Loss: {avg_loss:.4f}"
        )

        if protein_to_split and val_loaders:
            v_loss = _mean_eval_loss_for_loaders(
                model, val_loaders, device, cfg.go_margin, active_ontologies
            )
            if v_loss is not None:
                wandb.log(
                    {"phase0/val/mean_loss": v_loss, "phase0/epoch": epoch + 1}
                )
                print(f"[Phase0] Val mean loss: {v_loss:.4f}")

        if protein_to_split and test_loaders:
            t_loss = _mean_eval_loss_for_loaders(
                model, test_loaders, device, cfg.go_margin, active_ontologies
            )
            if t_loss is not None:
                wandb.log(
                    {"phase0/test/mean_loss": t_loss, "phase0/epoch": epoch + 1}
                )
                print(f"[Phase0] Test mean loss: {t_loss:.4f}")