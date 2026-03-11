import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import plotly.express as px
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


def plot_tsne(coords_2d, meta, title):
    """Create a plotly scatter with color=protein, symbol=role."""
    df = pd.DataFrame({
        "t-SNE 1": coords_2d[:, 0],
        "t-SNE 2": coords_2d[:, 1],
        "Protein": [m["protein"] for m in meta],
        "Role": [m["role"] for m in meta],
    })

    role_order = {"negative": 0, "positive": 1, "anchor": 2}
    df["_order"] = df["Role"].map(role_order)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    symbol_map = {"anchor": "diamond", "positive": "circle", "negative": "x"}

    fig = px.scatter(
        df,
        x="t-SNE 1",
        y="t-SNE 2",
        color="Protein",
        symbol="Role",
        symbol_map=symbol_map,
        title=title,
        hover_data=["Protein", "Role"],
        category_orders={"Role": ["negative", "positive", "anchor"]},
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(width=1100, height=800, legend=dict(font=dict(size=9)))
    fig.show()


def visualize_tsne(model=None,
                   model_path="model_best_loss.pt",
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
            dropout=cfg.dropout,
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

        if show_baseline and bl_global is not None:
            print("Computing baseline t-SNE (raw pooled features, no GNN)...")
            bl_global_2d = TSNE(**tsne_kwargs).fit_transform(bl_global)
            plot_tsne(bl_global_2d, meta,
                      f"BASELINE — Global Pooled Features t-SNE ({split})")

            bl_local_2d = TSNE(**tsne_kwargs).fit_transform(bl_local)
            plot_tsne(bl_local_2d, meta,
                      f"BASELINE — Local Pooled Features t-SNE ({split})")

        print("Computing trained-model t-SNE...")
        global_2d = TSNE(**tsne_kwargs).fit_transform(global_embs)
        plot_tsne(global_2d, meta,
                  f"TRAINED — Global Embeddings t-SNE ({split})")

        local_2d = TSNE(**tsne_kwargs).fit_transform(local_embs)
        plot_tsne(local_2d, meta, f"TRAINED — Local Embeddings t-SNE ({split})")
