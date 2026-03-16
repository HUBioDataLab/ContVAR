import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from contvar.config import ProjectConfig
from contvar.data.go_dataset import GOSemanticTripletDataset
from contvar.utils import load_all_embeddings


def _triplet_loss(anchor, positive, negative, margin: float):
    d_pos = F.pairwise_distance(anchor, positive, p=2)
    d_neg = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_pos - d_neg + margin).mean(), d_pos, d_neg


def _build_go_loader(
    tsv_path: str,
    ontology: str,
    cfg: ProjectConfig,
    structure_root: str,
    esm2_embeddings: Optional[dict],
) -> Optional[DataLoader]:
    dataset = GOSemanticTripletDataset(
        tsv_path=tsv_path,
        ontology=ontology,
        config=cfg,
        structure_root=structure_root,
        esm2_embeddings=esm2_embeddings,
    )
    if len(dataset) == 0:
        return None

    loader = DataLoader(
        dataset,
        batch_size=cfg.go_batch_size,
        shuffle=True,
        num_workers=getattr(cfg, "num_workers", 0),
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

    # If GO structures are provided as a ZIP archive, extract them lazily.
    # This is especially useful on Colab, where the zip lives on Drive.
    if getattr(cfg, "go_structures_zip", None):
        zip_path = cfg.go_structures_zip

        # Derive a default root folder from the zip name if not provided.
        if not getattr(cfg, "go_structure_root", None):
            base = os.path.splitext(os.path.basename(zip_path))[0]
            # On Colab we typically work under /content; user can still override.
            cfg.go_structure_root = os.path.join("/content/content", base)

        if not os.path.exists(cfg.go_structure_root):
            os.makedirs(cfg.go_structure_root, exist_ok=True)
            if os.path.exists(zip_path):
                import zipfile
                print(f"[Phase0] Extracting GO structures from ZIP: {zip_path}")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(cfg.go_structure_root)
                print(f"[Phase0] Extracted GO structures to: {cfg.go_structure_root}")
            else:
                print(f"[Phase0] WARNING: go_structures_zip={zip_path} does not exist; continuing without unzip.")

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

    # Optional embeddings for node features
    esm_embeddings = None
    if getattr(cfg, "go_use_esm_embeddings", True) and cfg.go_embeddings_path:
        esm_embeddings = load_all_embeddings(cfg.go_embeddings_path)

    # Build loaders
    loaders = {}
    for ont, path in [("mf", mf_tsv), ("bp", bp_tsv), ("cc", cc_tsv)]:
        if os.path.exists(path):
            loader = _build_go_loader(
                tsv_path=path,
                ontology=ont,
                cfg=cfg,
                structure_root=cfg.go_structure_root,
                esm2_embeddings=esm_embeddings,
            )
            if loader is not None:
                loaders[ont] = loader

    if not loaders:
        print("No GO loaders constructed for phase 0, skipping.")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.go_lr, weight_decay=cfg.weight_decay
    )

    model.to(device)

    for epoch in range(cfg.go_phase0_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for ontology, loader in loaders.items():
            for batch in tqdm(
                loader, desc=f"Phase0 Epoch {epoch+1} [{ontology.upper()}]", leave=False
            ):
                ba, bp, bn = batch
                ba = ba.to(device)
                bp = bp.to(device)
                bn = bn.to(device)

                za = model.forward_go_head(ba, ontology)
                zp = model.forward_go_head(bp, ontology)
                zn = model.forward_go_head(bn, ontology)

                loss, d_pos, d_neg = _triplet_loss(
                    za, zp, zn, margin=cfg.go_margin
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1

                log_dict = {
                    f"phase0/{ontology}/batch_loss": loss.item(),
                    f"phase0/{ontology}/avg_pos_dist": d_pos.mean().item(),
                    f"phase0/{ontology}/avg_neg_dist": d_neg.mean().item(),
                    f"phase0/{ontology}/dist_margin": (
                        d_neg.mean() - d_pos.mean()
                    ).item(),
                    "phase0/ontology": ontology,
                    "phase0/epoch": epoch + 1,
                }
                wandb.log(log_dict)

        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
        else:
            avg_loss = 0.0

        wandb.log({"phase0/epoch_loss": avg_loss, "phase0/epoch": epoch + 1})
        print(
            f"[Phase0] Epoch {epoch+1}/{cfg.go_phase0_epochs} | "
            f"Avg Loss: {avg_loss:.4f}"
        )

