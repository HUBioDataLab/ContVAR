import os
import sys
from functools import partial

import torch

from graphein.protein.edges.distance import (
    add_k_nn_edges,
)
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot

from contvar.edges import SaladStyleEdgeBuilder


class ProjectConfig:
    """Centralized configuration for features, edges, and hyperparameters"""

    def __init__(self):
        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Node Features
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
        self.knn_k = 32

        # Model Hyperparameters
        self.hidden_dim = 128
        self.output_dim = 256
        self.heads = 4
        self.lr = 1e-4
        self.weight_decay = 0.01
        self.epochs = 200
        self.margin = 0.3

        # Loss Function Configuration
        self.loss_type = "semi_hard"
        self.max_negatives = 10

        # Streaming Negative Mining Configuration
        self.mining_chunk_size = 10

        # Stage-2 DMS training configuration
        self.mining_batch_size = 8
        self.eval_batch_size = 32
        self.grad_accumulation_steps = 4
        self.num_workers = 8

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
            _repo_root, "semantic_similarity"
        )
        # Phase 0 loads protein graphs only from this directory (PyG Data .pt files).
        # Required when go_phase0_epochs > 0.
        self.go_prebuilt_graph_root = None
        # Optional initialization checkpoint for reusing saved GO-pretrained weights.
        # If set, the model is loaded from this path before any phase-0 training.
        # To skip recomputing phase 0 entirely, combine this with go_phase0_epochs = 0.
        self.go_phase0_init_checkpoint_path = None
        self.go_phase0_best_model_path = os.path.join(
            _repo_root, "model_phase0_best_loss.pt"
        )
        self.go_phase0_last_model_path = os.path.join(
            _repo_root, "model_phase0_last.pt"
        )
        self.dms_protein_split_json_path = os.path.join(
            _repo_root, "local_splits", "dms_protein_split.json"
        )
        # Merged bundle: protein_id -> train|val|test (UniRef pipeline + optional graphless drop).
        self.go_protein_split_json_path = os.path.join(
            _repo_root, "local_splits", "phase0_protein_split_removed_graphless.json"
        )
        self.go_split_seed = 42

    @property
    def input_dim(self):
        """Calculate input dimension based on active features"""
        dim = 20
        if self.use_embedding: dim += self.esm_dim
        return dim

    @property
    def edge_attr_dim(self):
        """Calculate edge attribute dimension based on edge mode"""
        if self.edge_mode == "salad":
            return self.salad_num_rbf + 3 + 1  # 16 + 3 + 1 = 20
        else:  # graphein mode
            return 1  # Euclidean distance for kNN edges

    def get_salad_edge_builder(self):
        """Return a SaladStyleEdgeBuilder instance with current config"""
        return SaladStyleEdgeBuilder(
            num_index=self.salad_num_index,
            num_spatial=self.salad_num_spatial,
            num_random=self.salad_num_random
        )

    def get_active_edge_funcs(self):
        """Return the active Graphein edge construction functions."""
        return [partial(add_k_nn_edges, k=self.knn_k)]

    def get_active_node_metadata_funcs(self):
        """Return active node feature functions"""
        return [amino_acid_one_hot]

    def get_node_attributes_list(self):
        """Return list of active node attributes"""
        attrs = ["amino_acid_one_hot"]
        if self.use_embedding: attrs.append("embedding")
        return attrs


# Colab defaults for DMS triplets (zip on Drive, extracted under /content/...)
_COLAB_DEFAULT_DATA_ROOT = "/content/content/content/protein_triplets_data"
_COLAB_DEFAULT_DATA_ZIP = "/content/drive/MyDrive/ContVAR/protein_triplets_data_9march.zip"
_COLAB_DEFAULT_EMBEDDINGS = "/content/drive/MyDrive/ContVAR/embeddings_variable.h5"


def ensure_dms_triplets_unzipped(data_root, data_zip=None):
    """
    Extract the DMS protein_triplets zip if the expected folder is missing.

    Intended for Colab: call from train_pipeline so Step 3 can use
    setup_environment(prepare_dms_triplets_zip=False) without a long unzip.

    No-op if data_root already exists, or zip path is missing / file not found.
    """
    if data_root and os.path.isdir(data_root):
        return

    is_colab = "google.colab" in sys.modules
    if data_zip is None and is_colab:
        data_zip = _COLAB_DEFAULT_DATA_ZIP

    if not data_zip or not os.path.isfile(data_zip):
        if data_root:
            print(
                f"Warning: DMS folder not found ({data_root!r}) and no zip to extract "
                f"(data_zip={data_zip!r})."
            )
        return

    import zipfile

    print(f"Unzipping {data_zip} to /content/...")
    with zipfile.ZipFile(data_zip, "r") as zip_ref:
        zip_ref.extractall("/content/")
    print("Unzipping complete.")


def setup_environment(
    data_root=None,
    embeddings_path=None,
    data_zip=None,
    prepare_dms_triplets_zip=True,
):
    """
    Auto-detect Colab vs. local and return configured paths.

    Args:
        prepare_dms_triplets_zip: If True (default), unzip DMS data on Colab when
            ``data_root`` is missing. Set False for a fast path (e.g. wandb login only);
            then call ``ensure_dms_triplets_unzipped`` from ``train_pipeline`` before
            loading triplets.

    Returns:
        dict with keys: device, data_root, embeddings_path, data_zip (Colab zip path
        or None locally)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    is_colab = 'google.colab' in sys.modules

    if is_colab:
        if data_root is None:
            data_root = _COLAB_DEFAULT_DATA_ROOT
        if embeddings_path is None:
            embeddings_path = _COLAB_DEFAULT_EMBEDDINGS
        if data_zip is None:
            data_zip = _COLAB_DEFAULT_DATA_ZIP

        # Unzip data if needed (optional: defer to train_pipeline via prepare_dms_triplets_zip=False)
        if prepare_dms_triplets_zip and not os.path.exists(data_root) and data_zip and os.path.exists(data_zip):
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
        data_zip = None

    return {
        'device': device,
        'data_root': data_root,
        'embeddings_path': embeddings_path,
        'data_zip': data_zip,
    }
