import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE

from contvar.config import ProjectConfig
from contvar.data.mapper import TripletDataPathMapper
from contvar.data.collate import parse_mut_pos_from_path
from contvar.mining import load_processed_graph_static
from contvar.model import DeepProteinGAT


def _extract_baseline_pooled(data, device, mut_pos_val=None):
    """
    Extract baseline embeddings from raw node features (no GNN).
    """
    batch_data = Batch.from_data_list([data]).to(device)
    x = batch_data.x.float()
    batch_vec = batch_data.batch

    g = global_mean_pool(x, batch_vec)

    if (mut_pos_val is not None and mut_pos_val >= 0
            and hasattr(batch_data, 'residue_number')
            and batch_data.residue_number is not None):
        res_num = batch_data.residue_number.to(device)
        mask = res_num == mut_pos_val
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0][0]
            l = x[idx].unsqueeze(0)
        else:
            l = g
    else:
        l = g

    return g.cpu().numpy(), l.cpu().numpy()


def extract_embeddings_for_tsne(model, mapper, split, processed_dir,
                                 device, max_families=20,
                                 max_variants_per_role=5,
                                 include_baseline=True):
    """
    Extract global and local embeddings for t-SNE visualization.
    """
    triplets = mapper.get_split(split)
    selected = triplets[:max_families]

    global_embs, local_embs, meta = [], [], []
    bl_global, bl_local = [], []

    model.eval()
    with torch.no_grad():
        for t in tqdm(selected, desc=f"Extracting embeddings ({split})"):
            pid = t["protein_id"]

            # --- Anchor ---
            a_data = load_processed_graph_static(t["anchor"], processed_dir)
            if a_data is None:
                continue
            batch_a = Batch.from_data_list([a_data]).to(device)
            ea_g, ea_l = model(batch_a)
            global_embs.append(ea_g.cpu().numpy())
            local_embs.append(ea_l.cpu().numpy())
            meta.append({"protein": pid, "role": "anchor"})
            if include_baseline:
                bg, bl = _extract_baseline_pooled(a_data, device)
                bl_global.append(bg); bl_local.append(bl)

            # --- Positives ---
            for pos_path in t["positives"][:max_variants_per_role]:
                p_data = load_processed_graph_static(pos_path, processed_dir)
                if p_data is None:
                    continue
                mut_pos_p = parse_mut_pos_from_path(pos_path)
                mut_tensor = torch.tensor(
                    [mut_pos_p if mut_pos_p is not None else -1], dtype=torch.long
                ).to(device)
                batch_p = Batch.from_data_list([p_data]).to(device)
                ep_g, ep_l = model(batch_p, mut_pos=mut_tensor)
                global_embs.append(ep_g.cpu().numpy())
                local_embs.append(ep_l.cpu().numpy())
                meta.append({"protein": pid, "role": "positive"})
                if include_baseline:
                    bg, bl = _extract_baseline_pooled(p_data, device, mut_pos_p)
                    bl_global.append(bg); bl_local.append(bl)

            # --- Negatives ---
            for neg_path in t["negatives"][:max_variants_per_role]:
                n_data = load_processed_graph_static(neg_path, processed_dir)
                if n_data is None:
                    continue
                mut_pos_n = parse_mut_pos_from_path(neg_path)
                mut_tensor = torch.tensor(
                    [mut_pos_n if mut_pos_n is not None else -1], dtype=torch.long
                ).to(device)
                batch_n = Batch.from_data_list([n_data]).to(device)
                en_g, en_l = model(batch_n, mut_pos=mut_tensor)
                global_embs.append(en_g.cpu().numpy())
                local_embs.append(en_l.cpu().numpy())
                meta.append({"protein": pid, "role": "negative"})
                if include_baseline:
                    bg, bl = _extract_baseline_pooled(n_data, device, mut_pos_n)
                    bl_global.append(bg); bl_local.append(bl)

    global_embs = np.vstack(global_embs)
    local_embs = np.vstack(local_embs)
    if include_baseline:
        return (global_embs, local_embs, meta,
                np.vstack(bl_global), np.vstack(bl_local))
    return global_embs, local_embs, meta, None, None


_ROLE_SIZE = {"negative": 30, "positive": 30, "anchor": 120}


def _build_protein_color_map(meta):
    """Assign each protein a unique green tone (positive) and red tone (negative)."""
    proteins = list(dict.fromkeys(m["protein"] for m in meta))
    n = max(len(proteins), 1)

    # Generate distinct tones across green and red hue ranges
    green_cmap = plt.cm.Greens
    red_cmap = plt.cm.Reds
    # Sample from 0.35..0.9 to avoid too-light and too-dark extremes
    positions = np.linspace(0.35, 0.9, n)

    color_map = {}  # (protein, role) -> color
    for i, pid in enumerate(proteins):
        color_map[(pid, "positive")] = green_cmap(positions[i])
        color_map[(pid, "negative")] = red_cmap(positions[i])
        color_map[(pid, "anchor")] = "gold"

    return proteins, color_map


def _plot_panel(ax, coords_2d, meta, title):
    """Draw a single t-SNE scatter panel on *ax* with per-protein coloring."""
    proteins, color_map = _build_protein_color_map(meta)

    # Draw negatives first, then positives, then anchors (z-order)
    for role, zorder in [("negative", 1), ("positive", 2), ("anchor", 3)]:
        marker = "*" if role == "anchor" else "o"
        for pid in proteins:
            idxs = [i for i, m in enumerate(meta)
                    if m["role"] == role and m["protein"] == pid]
            if not idxs:
                continue
            ax.scatter(
                coords_2d[idxs, 0],
                coords_2d[idxs, 1],
                c=[color_map[(pid, role)]],
                marker=marker,
                s=_ROLE_SIZE[role],
                alpha=0.7,
                edgecolors="none",
                zorder=zorder,
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, linewidth=0.3, alpha=0.5)


def _make_legend_handles(meta):
    """Build legend handles: one entry per protein with green/red tones, plus WT."""
    proteins, color_map = _build_protein_color_map(meta)
    roles_present = {m["role"] for m in meta}
    handles = []

    for pid in proteins:
        if "positive" in roles_present:
            handles.append(
                mlines.Line2D([], [], color=color_map[(pid, "positive")],
                              marker="o", linestyle="None", markersize=8,
                              label=f"{pid} (good)"))
        if "negative" in roles_present:
            handles.append(
                mlines.Line2D([], [], color=color_map[(pid, "negative")],
                              marker="o", linestyle="None", markersize=8,
                              label=f"{pid} (bad)"))

    if "anchor" in roles_present:
        handles.append(
            mlines.Line2D([], [], color="gold", marker="*",
                          linestyle="None", markersize=12, label="WT"))

    return handles


def plot_tsne_side_by_side(bl_2d, proj_2d, meta, suptitle,
                           left_title="Baseline (Raw ESM)",
                           right_title="Projected"):
    """Side-by-side baseline vs projected t-SNE (matplotlib)."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    _plot_panel(ax_l, bl_2d, meta, left_title)
    _plot_panel(ax_r, proj_2d, meta, right_title)

    handles = _make_legend_handles(meta)
    ax_r.legend(handles=handles, loc="upper right", fontsize=7,
                framealpha=0.8, ncol=2)

    fig.tight_layout()
    plt.show()


def plot_tsne(coords_2d, meta, title):
    """Single-panel t-SNE plot (matplotlib)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    _plot_panel(ax, coords_2d, meta, title)
    handles = _make_legend_handles(meta)
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              framealpha=0.8, ncol=2)
    fig.tight_layout()
    plt.show()


def visualize_tsne(model=None,
                   model_path="model_last.pt",
                   splits=None,
                   max_families=20,
                   max_variants_per_role=5,
                   perplexity=30,
                   random_state=42,
                   show_baseline=True,
                   data_root=None,
                   device=None):
    """
    Run t-SNE on global and local embeddings and display interactive plots.
    """
    if splits is None:
        splits = ["val"]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if data_root is None:
        from contvar.config import setup_environment
        data_root = setup_environment()['data_root']

    cfg = ProjectConfig()

    if model is None:
        model = DeepProteinGAT(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim,
            heads=cfg.heads,
            edge_dim=cfg.edge_attr_dim,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    split_json = os.path.join(data_root, "split.json")
    mapper = TripletDataPathMapper(
        data_root, val_pos=2, val_neg=2, seed=42,
        split_path=split_json if os.path.exists(split_json) else None,
    )
    processed_dir = os.path.join(data_root, "processed")

    for split in splits:
        print(f"\n{'='*50}")
        print(f"  t-SNE Visualization — {split} split")
        print(f"{'='*50}")

        global_embs, local_embs, meta, bl_global, bl_local = \
            extract_embeddings_for_tsne(
                model, mapper, split, processed_dir, device,
                max_families=max_families,
                max_variants_per_role=max_variants_per_role,
                include_baseline=show_baseline,
            )

        n_points = len(meta)
        effective_perplexity = min(perplexity, n_points - 1)
        print(f"Total points: {n_points} | Perplexity: {effective_perplexity}")

        tsne_kwargs = dict(
            n_components=2, perplexity=effective_perplexity,
            random_state=random_state, n_iter=1000,
        )

        print("Computing t-SNE projections...")
        global_2d = TSNE(**tsne_kwargs).fit_transform(global_embs)

        # For local plots, exclude anchors (they have no mutation position,
        # so their "local" embedding is just the global one — misleading).
        local_idxs = [i for i, m in enumerate(meta) if m["role"] != "anchor"]
        local_meta = [meta[i] for i in local_idxs]
        local_embs_filtered = local_embs[local_idxs]

        if show_baseline and bl_global is not None:
            bl_global_2d = TSNE(**tsne_kwargs).fit_transform(bl_global)
            bl_local_filtered = bl_local[local_idxs]

            # Recompute t-SNE for local with filtered points
            local_perp = min(perplexity, len(local_meta) - 1)
            local_tsne_kwargs = dict(
                n_components=2, perplexity=local_perp,
                random_state=random_state, n_iter=1000,
            )
            local_2d = TSNE(**local_tsne_kwargs).fit_transform(local_embs_filtered)
            bl_local_2d = TSNE(**local_tsne_kwargs).fit_transform(bl_local_filtered)

            plot_tsne_side_by_side(
                bl_global_2d, global_2d, meta,
                suptitle=f"t-SNE Embedding Visualization — Global ({split})",
                left_title="Baseline Global (Raw ESM)",
                right_title="Projected Global",
            )
            plot_tsne_side_by_side(
                bl_local_2d, local_2d, local_meta,
                suptitle=f"t-SNE Embedding Visualization — Local ({split})",
                left_title="Baseline Local (Raw ESM)",
                right_title="Projected Local",
            )
        else:
            local_2d = TSNE(**tsne_kwargs).fit_transform(local_embs_filtered)
            plot_tsne(global_2d, meta,
                      f"Projected Global t-SNE ({split})")
            plot_tsne(local_2d, local_meta,
                      f"Projected Local t-SNE ({split})")
