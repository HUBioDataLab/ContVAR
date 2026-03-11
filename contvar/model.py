import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class DeepProteinGAT(nn.Module):
    """3-layer GATv2 model with edge features, dropout, normalization and strong projection head.

    This model supports both edge construction modes:
    - SALAD-style edges: RBF distance encoding + neighbor type + sequence distance (edge_dim=20)
    - Graphein edges: edge type one-hot + euclidean distance (edge_dim=9)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.0,
                 edge_dim=20, projection_hidden_dim=None):
        super().__init__()

        self.dropout = dropout
        self.edge_dim = edge_dim

        if projection_hidden_dim is None:
            projection_hidden_dim = output_dim * 2

        # Edge feature embedding
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Layer 1
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True,
                               dropout=0.0, edge_dim=hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        # Layer 2
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=True,
                               dropout=0.0, edge_dim=hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim * heads)

        # Layer 3
        self.conv3 = GATv2Conv(hidden_dim * heads, output_dim, heads=1, concat=False,
                               dropout=0.0, edge_dim=hidden_dim)
        self.norm3 = nn.LayerNorm(output_dim)

        # Strong MLP Projection Head for Metric Learning (global)
        self.projection = nn.Sequential(
            nn.Linear(output_dim, projection_hidden_dim),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(projection_hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        # Projection head for local (mutation-position) embedding
        self.projection_local = nn.Sequential(
            nn.Linear(output_dim, projection_hidden_dim),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(projection_hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, data, mut_pos=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        x = x.float()

        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = edge_attr.float()
            edge_embed = self.edge_encoder(edge_attr)
        else:
            edge_embed = None

        x = self.conv1(x, edge_index, edge_attr=edge_embed)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr=edge_embed)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index, edge_attr=edge_embed)
        x = self.norm3(x)

        x_global = global_mean_pool(x, batch)
        x_global = self.projection(x_global)
        x_global = F.normalize(x_global, p=2, dim=1)

        # Local: embedding at mut_pos per graph (fallback to graph mean if position missing)
        if mut_pos is not None and hasattr(data, 'residue_number') and data.residue_number is not None:
            B = batch.max().item() + 1
            device = x.device
            res_num = data.residue_number.to(device)
            mut_pos = mut_pos.to(device)
            x_local = torch.zeros(B, x.size(1), device=device, dtype=x.dtype)
            for i in range(B):
                mask = (batch == i) & (res_num == mut_pos[i])
                if mask.any() and mut_pos[i] >= 0:
                    idx = mask.nonzero(as_tuple=True)[0][0]
                    x_local[i] = x[idx]
                else:
                    graph_mask = (batch == i)
                    x_local[i] = x[graph_mask].mean(0)
            x_local = self.projection_local(x_local)
            x_local = F.normalize(x_local, p=2, dim=1)
        else:
            x_local = x_global

        return x_global, x_local
