import os
import sys
from functools import partial

import torch

from graphein.protein.edges.distance import (
    add_aromatic_interactions, add_disulfide_interactions,
    add_hydrogen_bond_interactions, add_peptide_bonds,
    add_hydrophobic_interactions, add_ionic_interactions,
    add_k_nn_edges, add_distance_threshold
)
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.features.nodes import asa, rsa
from graphein.protein.features.nodes.dssp import secondary_structure

from contvar.edges import SaladStyleEdgeBuilder


class ProjectConfig:
    """Centralized configuration for features, edges, and hyperparameters"""

    def __init__(self):
        # Node Features
        self.use_coords = False
        self.use_b_factor = False
        self.use_amino_acid = True
        self.use_asa = False
        self.use_rsa = False
        self.use_ss = False
        self.use_backbone_dh = False
        self.use_sidechain_dh = False
        self.use_embedding = True
        self.esm_dim = 1280

        # Edge Construction Mode: "salad" or "graphein"
        self.edge_mode = "salad"

        # SALAD-style edge configuration
        self.salad_num_index = 16
        self.salad_num_spatial = 16
        self.salad_num_random = 16
        self.salad_num_rbf = 16
        self.salad_d_max = 22.0

        # Graphein edge configuration
        self.edge_peptide = False
        self.edge_aromatic = False
        self.edge_disulfide = False
        self.edge_hydrogen = False
        self.edge_hydrophobic = False
        self.edge_ionic = False
        self.edge_knn = True
        self.edge_distance = False
        self.knn_k = 32
        self.dist_threshold = 8.0

        # Model Hyperparameters
        self.hidden_dim = 128
        self.output_dim = 256
        self.heads = 4
        self.lr = 1e-4
        self.weight_decay = 0.01
        self.batch_size = 8
        self.epochs = 200
        self.margin = 0.3

        # Loss Function Configuration
        self.loss_type = "semi_hard"
        self.max_negatives = 10

        # Streaming Negative Mining Configuration
        self.mining_chunk_size = 10

        # Curriculum Learning Configuration
        self.curriculum_warmup_epochs = 0
        self.warmup_batch_size = 32
        self.mining_batch_size = 8
        self.grad_accumulation_steps = 4
        self.num_workers = 8

        # Phase 1 Intra-Epoch Early Stopping
        self.phase1_early_stop = True
        self.phase1_es_threshold = 0.01
        self.phase1_es_window = 20
        self.phase1_es_patience = 3
        self.phase1_es_min_batches = 50

        # Phase 0: GO semantic similarity pretraining
        self.go_phase0_epochs = 2  # set >0 to enable (override in Colab if needed)
        self.go_margin = 0.2
        self.go_batch_size = 8
        self.go_lr = 1e-4
        # Ontology sampling in phase-0 (GOAL2): use ratio-driven ontology picks.
        # Disable to preserve legacy behavior (one batch from each ontology per step).
        self.go_sampling_enabled = True
        self.go_sampling_ratio = {"mf": 0.6, "bp": 0.2, "cc": 0.2}
        self.go_log_sampling_stats = True
        # Limit and loader settings for GO pretraining
        self.go_max_triplets_per_ontology = 1000  # subsample for Colab; can be overridden
        self.go_num_workers = 0  # DataLoader workers for GO (0 is safer on Colab)

        # Paths for GO pretraining
        # Directory containing semantic similarity TSVs
        self.go_tsv_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "semantic_similarity"
        )
        # Phase 0 loads protein graphs only from this directory (PyG Data .pt files).
        # Required when go_phase0_epochs > 0.
        self.go_prebuilt_graph_root = None
        # If True, ignore identity-grouped split and create train/val/test directly
        # from prebuilt protein IDs using go_train_ratio/go_val_ratio/go_test_ratio.
        self.go_random_split_from_prebuilt = False

        # Phase-0: identity-aware train/val/test split (sequence groups, e.g. UniRef50)
        # "none" = use all triplets (legacy). "identity_grouped" = filter by cluster split.
        self.go_split_mode = "none"
        self.go_split_seed = 42
        self.go_train_ratio = 0.8
        self.go_val_ratio = 0.1
        self.go_test_ratio = 0.1
        # Tab-separated mapping: protein_id -> group_id (e.g. UniRef50 cluster).
        self.go_cluster_map_path = None
        # JSON with group_id -> "train"|"val"|"test"; share this file across teams.
        self.go_split_json_path = None
        # If set and go_split_json_path is missing, write a new split here after assignment.
        self.go_save_split_json_path = None

    @property
    def input_dim(self):
        """Calculate input dimension based on active features"""
        dim = 0
        if self.use_coords: dim += 3
        if self.use_b_factor: dim += 1
        if self.use_amino_acid: dim += 20
        if self.use_asa: dim += 1
        if self.use_rsa: dim += 1
        if self.use_ss: dim += 8
        if self.use_backbone_dh: dim += 3
        if self.use_sidechain_dh: dim += 5
        if self.use_embedding: dim += self.esm_dim
        return dim

    @property
    def edge_attr_dim(self):
        """Calculate edge attribute dimension based on edge mode"""
        if self.edge_mode == "salad":
            return self.salad_num_rbf + 3 + 1  # 16 + 3 + 1 = 20
        else:  # graphein mode
            return 8 + 1  # 9

    def get_salad_edge_builder(self):
        """Return a SaladStyleEdgeBuilder instance with current config"""
        return SaladStyleEdgeBuilder(
            num_index=self.salad_num_index,
            num_spatial=self.salad_num_spatial,
            num_random=self.salad_num_random
        )

    def get_active_edge_funcs(self):
        """Return list of active edge construction functions (for graphein mode)"""
        edge_funcs = []
        if self.edge_peptide: edge_funcs.append(add_peptide_bonds)
        if self.edge_aromatic: edge_funcs.append(add_aromatic_interactions)
        if self.edge_disulfide: edge_funcs.append(add_disulfide_interactions)
        if self.edge_hydrogen: edge_funcs.append(add_hydrogen_bond_interactions)
        if self.edge_hydrophobic: edge_funcs.append(add_hydrophobic_interactions)
        if self.edge_ionic: edge_funcs.append(add_ionic_interactions)
        if self.edge_knn: edge_funcs.append(partial(add_k_nn_edges, k=self.knn_k))
        if self.edge_distance:
            edge_funcs.append(partial(add_distance_threshold, long_interaction_threshold=self.dist_threshold))
        return edge_funcs

    def get_active_node_metadata_funcs(self):
        """Return active node feature functions"""
        node_funcs = []
        if self.use_amino_acid:
            node_funcs.append(amino_acid_one_hot)
        if self.use_asa:
            node_funcs.append(asa)
        if self.use_rsa:
            node_funcs.append(rsa)
        if self.use_ss:
            node_funcs.append(secondary_structure)
        return node_funcs

    def get_node_attributes_list(self):
        """Return list of active node attributes"""
        attrs = []
        if self.use_coords: attrs.append("coords")
        if self.use_b_factor: attrs.append("b_factor")
        if self.use_amino_acid: attrs.append("amino_acid_one_hot")
        if self.use_asa: attrs.append("asa")
        if self.use_rsa: attrs.append("rsa")
        if self.use_ss: attrs.append("ss")
        if self.use_backbone_dh: attrs.append("backbone_dihedral_radians")
        if self.use_sidechain_dh: attrs.append("sidechain_dihedral_radians")
        if self.use_embedding: attrs.append("embedding")
        return attrs


def setup_environment(data_root=None, embeddings_path=None, data_zip=None):
    """
    Auto-detect Colab vs. local and return configured paths.

    Returns:
        dict with keys: device, data_root, embeddings_path, data_zip
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    is_colab = 'google.colab' in sys.modules

    if is_colab:
        if data_root is None:
            data_root = "/content/content/content/protein_triplets_data"
        if embeddings_path is None:
            embeddings_path = "/content/drive/MyDrive/ContVAR/embeddings_variable.h5"
        if data_zip is None:
            data_zip = "/content/drive/MyDrive/ContVAR/protein_triplets_data_9march.zip"

        # Unzip data if needed
        if not os.path.exists(data_root) and data_zip and os.path.exists(data_zip):
            import zipfile
            print(f"Unzipping {data_zip} to /content/...")
            with zipfile.ZipFile(data_zip, 'r') as zip_ref:
                zip_ref.extractall("/content/")
            print("Unzipping complete.")
    else:
        if data_root is None:
            # Default local path: relative to project root
            data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protein_triplets_data")
        if embeddings_path is None:
            embeddings_path = os.path.join(data_root, "..", "embeddings_variable.h5")

    return {
        'device': device,
        'data_root': data_root,
        'embeddings_path': embeddings_path,
    }
