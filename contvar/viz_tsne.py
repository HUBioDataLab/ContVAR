import os

import numpy as np
import torch
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
    """Extract baseline embeddings from raw node features (no GNN)."""
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


def _extract_variant(model, path, processed_dir, device, pid, role,
                     global_embs, local_embs, meta,
                     bl_global, bl_local, mut_pos_val=None):
    """Load one variant, run model + baseline, append to lists."""
    data = load_processed_graph_static(path, processed_dir)
    if data is None:
        return

    mut_tensor = None
    if mut_pos_val is not None:
        mut_tensor = torch.tensor(
            [mut_pos_val if mut_pos_val is not None else -1], dtype=torch.long
        ).to(device)

    batch = Batch.from_data_list([data]).to(device)
    eg, el = model(batch) if mut_tensor is None else model(batch, mut_pos=mut_tensor)

    global_embs.append(eg.cpu().numpy())
    local_embs.append(el.cpu().numpy())
    meta.append({"protein": pid, "role": role})

    bg, bl = _extract_baseline_pooled(data, device, mut_pos_val)
    bl_global.append(bg)
    bl_local.append(bl)


def extract_embeddings_for_tsne(model, mapper, split, processed_dir,
                                 device, max_families=20,
                                 max_variants_per_role=5):
    """Extract global/local + baseline embeddings for t-SNE."""
    triplets = mapper.get_split(split)
    selected = triplets[:max_families]

    global_embs, local_embs, meta = [], [], []
    bl_global, bl_local = [], []

    model.eval()
    with torch.no_grad():
        for t in tqdm(selected, desc=f"Extracting embeddings ({split})"):
            pid = t["protein_id"]

            _extract_variant(model, t["anchor"], processed_dir, device,
                             pid, "anchor", global_embs, local_embs, meta,
                             bl_global, bl_local)

            for pos_path in t["positives"][:max_variants_per_role]:
                mut_pos = parse_mut_pos_from_path(pos_path)
                _extract_variant(model, pos_path, processed_dir, device,
                                 pid, "positive", global_embs, local_embs, meta,
                                 bl_global, bl_local, mut_pos)

            for neg_path in t["negatives"][:max_variants_per_role]:
                mut_pos = parse_mut_pos_from_path(neg_path)
                _extract_variant(model, neg_path, processed_dir, device,
                                 pid, "negative", global_embs, local_embs, meta,
                                 bl_global, bl_local, mut_pos)

    return (np.vstack(global_embs), np.vstack(local_embs), meta,
            np.vstack(bl_global), np.vstack(bl_local))


# =============================================
# Plotting
# =============================================

_ROLE_SIZE = {"negative": 12, "positive": 12, "anchor": 120}


def _build_protein_color_map(meta):
    """Assign each protein a unique green tone (positive) and red tone (negative)."""
    proteins = list(dict.fromkeys(m["protein"] for m in meta))
    n = max(len(proteins), 1)

    green_cmap = plt.cm.Greens
    red_cmap = plt.cm.Reds
    positions = np.linspace(0.35, 0.9, n)

    color_map = {}
    for i, pid in enumerate(proteins):
        color_map[(pid, "positive")] = green_cmap(positions[i])
        color_map[(pid, "negative")] = red_cmap(positions[i])
        color_map[(pid, "anchor")] = "gold"

    return proteins, color_map


def _plot_panel(ax, coords_2d, meta, title):
    """Draw a single t-SNE scatter panel on *ax* with per-protein coloring."""
    proteins, color_map = _build_protein_color_map(meta)

    for role, zorder in [("negative", 1), ("positive", 2), ("anchor", 3)]:
        marker = "*" if role == "anchor" else "o"
        edge = "black" if role == "anchor" else "none"
        for pid in proteins:
            idxs = [i for i, m in enumerate(meta)
                    if m["role"] == role and m["protein"] == pid]
            if not idxs:
                continue
            ax.scatter(
                coords_2d[idxs, 0], coords_2d[idxs, 1],
                c=[color_map[(pid, role)]], marker=marker,
                s=_ROLE_SIZE[role], alpha=0.5,
                edgecolors=edge, linewidth=0.5, zorder=zorder,
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)


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
                              label=f"{pid} (positive)"))
        if "negative" in roles_present:
            handles.append(
                mlines.Line2D([], [], color=color_map[(pid, "negative")],
                              marker="o", linestyle="None", markersize=8,
                              label=f"{pid} (negative)"))

    if "anchor" in roles_present:
        handles.append(
            mlines.Line2D([], [], color="gold", marker="*",
                          markeredgecolor="black", markeredgewidth=0.5,
                          linestyle="None", markersize=12, label="WT"))

    return handles


def _plot_side_by_side(bl_2d, proj_2d, meta, suptitle, filename):
    """1x2 figure: baseline (left) vs projected (right), legend outside."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")

    _plot_panel(ax_l, bl_2d, meta, "Baseline (Raw ESM)")
    _plot_panel(ax_r, proj_2d, meta, "Projected")

    handles = _make_legend_handles(meta)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=9, framealpha=0.8, ncol=1)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[t-SNE] Saved → {filename}")
    return filename


# =============================================
# Main entry point
# =============================================

def visualize_tsne(model=None,
                   model_path="model_best_loss.pt",
                   splits=None,
                   max_families=20,
                   max_variants_per_role=5,
                   perplexity=30,
                   random_state=42,
                   data_root=None,
                   device=None):
    """Run t-SNE on global and local embeddings, save 2x2 comparison plot."""
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
            input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim, heads=cfg.heads,
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
            )

        n_points = len(meta)
        effective_perplexity = min(perplexity, n_points - 1)
        print(f"Total points: {n_points} | Perplexity: {effective_perplexity}")

        tsne_kwargs = dict(
            n_components=2, perplexity=effective_perplexity,
            random_state=random_state, max_iter=1000,
            init="pca", learning_rate="auto",
        )

        print("Computing t-SNE projections...")
        global_2d = TSNE(**tsne_kwargs).fit_transform(global_embs)
        bl_global_2d = TSNE(**tsne_kwargs).fit_transform(bl_global)

        # Exclude anchors from local plots (no mutation position)
        local_idxs = [i for i, m in enumerate(meta) if m["role"] != "anchor"]
        local_meta = [meta[i] for i in local_idxs]

        local_perp = min(perplexity, len(local_meta) - 1)
        local_tsne_kwargs = dict(
            n_components=2, perplexity=local_perp,
            random_state=random_state, max_iter=1000,
            init="pca", learning_rate="auto",
        )
        local_2d = TSNE(**local_tsne_kwargs).fit_transform(local_embs[local_idxs])
        bl_local_2d = TSNE(**local_tsne_kwargs).fit_transform(bl_local[local_idxs])

        _plot_side_by_side(
            bl_global_2d, global_2d, meta,
            suptitle=f"t-SNE Global Embedding — {split}",
            filename=f"tsne_global_{split}.png",
        )
        _plot_side_by_side(
            bl_local_2d, local_2d, local_meta,
            suptitle=f"t-SNE Local Embedding — {split}",
            filename=f"tsne_local_{split}.png",
        )
