# ContVAR

ContVAR learns structure-aware protein representations for single amino acid variants (SAVs). A graph neural network (GATv2) encodes each variant structure as:

- a **global** graph embedding (whole-protein representation), and
- a **local** embedding at the mutation site.

Training uses metric learning on triplets from the same protein family:

| Role | Structure |
|------|-----------|
| Anchor | Wild-type (WT) |
| Positive | Benign variant |
| Negative | Pathogenic variant |

The model is trained to pull benign variants toward the WT and push pathogenic variants away, both globally and at the mutation position.

A downstream **Phase 2 decoder** trains feed-forward networks on top of frozen protein embeddings to predict Gene Ontology (GO) annotations and to score variant-induced functional changes.

## Web demo

Try ContVAR without installing anything:

**[https://huggingface.co/spaces/fd55/contvarspace](https://huggingface.co/spaces/fd55/contvarspace)**

The demo supports two analysis modes:

| Mode | Input | Output |
|------|--------|--------|
| **Protein structure only** | One PDB or CIF file | Thresholded GO term predictions (MF, BP, CC) with confidence scores |
| **WT + variant** | Wild-type and variant PDB/CIF files | Per-term WT vs variant scores, delta, and gained / lost / stable functional changes |

Both modes accept `.pdb` and `.cif` structures and include example files for quick testing. Predictions are reported when scores reach the model threshold (≥ 0.60); the WT vs variant mode additionally classifies terms as gained or lost using a score-delta threshold.

---

## Pipeline overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 0 — GO semantic-similarity pretraining                             │
│   semantic_similarity/*.tsv + prebuilt GO graph .pt + protein split JSON │
│   → model_phase0_best_loss.pt                                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ initialize Phase 1 from Phase 0 weights
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1 — DMS triplet metric learning                                    │
│   protein_triplets_data/ + ESM2 H5 + dms_protein_split.json             │
│   → model_best_loss.pt, model_last.pt                                    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   Embedding export        t-SNE plots         Frozen inference
   (exports/*.h5)      (visualizations/)    (contvar.inference)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 2 decoder — GO term prediction & variant function scoring          │
│   ESM / ContVAR H5 + GOA + UniRef split → decoder_best_*.pt              │
└─────────────────────────────────────────────────────────────────────────┘
```

The encoder is trained in two steps: **Phase 0** learns from GO semantic-similarity triplets on Swiss-Prot proteins, then **Phase 1** fine-tunes on DMS variant triplets. Phase 1 always starts from Phase 0 weights — either by training Phase 0 yourself, or by loading the published `model_phase0_best_loss.pt` checkpoint.

**End-to-end local entry point:** `python starter.py` runs Phase 0 (when configured to train) and Phase 1, then exports embeddings and generates t-SNE plots.

**Colab notebook:** `run.ipynb` walks through the same encoder pipeline with Google Drive data paths.

---

## Repository layout

```text
ContVAR/
├── starter.py                  # Main CLI — edit STARTER_PATHS here
├── run.ipynb                   # Colab reproduction notebook
├── setup.py
├── local_splits/
│   └── dms_protein_split.json  # Fixed protein-family train/val/test split
├── contvar/                    # Encoder: model, training, graph building, inference
│   ├── training.py             # train_pipeline()
│   ├── go_pretraining.py       # Phase 0
│   ├── prebuild_graphs.py      # Build PyG graphs from mmCIF + ESM2
│   ├── inference.py            # Frozen-checkpoint embedding export
│   └── export_embeddings.py    # Post-training H5 export
└── phase2_decoder/             # Phase 2: GO decoder training, evaluation, variant prediction
    ├── train.py
    ├── test_eval.py
    ├── benchmark_eval.py
    ├── predict_variants.py
    └── grid_search.py
```

This repository contains the training and evaluation code, together with fixed data splits. Datasets, prebuilt graphs, and trained model checkpoints are distributed with the project release (see [Data and model artifacts](#data-and-model-artifacts)).

---

## Requirements

- Python ≥ 3.8
- CUDA GPU recommended for training (CPU works for small smoke tests)
- [Weights & Biases](https://wandb.ai/) account (training logs experiments automatically)

### Installation

```bash
git clone https://github.com/HUBioDataLab/ContVAR.git
cd ContVAR
pip install -e .
```

Install PyTorch and PyG for your platform first, then the remaining dependencies:

```bash
pip install torch torch-geometric graphein wandb biopython h5py \
            scikit-learn matplotlib tqdm pandas numpy networkx
```

On Linux, Graphein may require DSSP for some graph features:

```bash
sudo apt-get install dssp
```

Log in to Weights & Biases before training:

```bash
wandb login
```

---

## Data and model artifacts

Download the release bundle and place files according to the layouts below. The release includes trained encoder and decoder weights; exact download location will be listed in the project release page.

### Published model checkpoints

| Checkpoint | Stage |
|------------|-------|
| `model_phase0_best_loss.pt` | Phase 0 encoder (best validation loss) |
| `model_phase0_last.pt` | Phase 0 encoder (final epoch) |
| `model_best_loss.pt` | Phase 1 encoder (best validation loss) |
| `model_last.pt` | Phase 1 encoder (final epoch) |
| `decoder_best_{embedding}_{aspect}.pt` | Phase 2 decoder |

To reproduce Phase 1 without re-running Phase 0 training, set `go_phase0_epochs` to `0` and point `go_phase0_init_checkpoint_path` at `model_phase0_best_loss.pt`. This is the workflow used in `run.ipynb`.

### Phase 0 — GO pretraining data

| Path | Purpose |
|------|---------|
| `semantic_similarity/semantic_similarity_swissprot_filtered_low0.2_high0.8_mf.tsv` | MF semantic-similarity triplets |
| `semantic_similarity/semantic_similarity_swissprot_filtered_low0.2_high0.8_bp.tsv` | BP triplets |
| `semantic_similarity/semantic_similarity_swissprot_filtered_low0.2_high0.8_cc.tsv` | CC triplets |
| `<go_prebuilt_graph_root>/` | Directory tree of prebuilt PyG `.pt` graph files (one per Swiss-Prot protein) |
| `local_splits/phase0_protein_split_removed_graphless.json` | `protein_to_split` mapping for GO proteins |

### Phase 1 — DMS variant triplets

```text
protein_triplets_data/
├── originals/          # WT mmCIF files, one per protein family
│   └── <family>_wt_model.cif
├── positives/          # Benign variant structures
│   └── <family>_wt_model/
│       └── *.cif
├── negatives/          # Pathogenic variant structures
│   └── <family>_wt_model/
│       └── *.cif
└── processed/          # Auto-generated PyG graph cache (created on first run)
```

Additional files:

| File | Purpose |
|------|---------|
| `embeddings_variable.h5` | Per-residue ESM-2 embeddings keyed by structure filename stem |
| `local_splits/dms_protein_split.json` | Protein-family → `train` / `val` / `test` mapping (included in repo) |

The split JSON uses `family_to_split` keys that match WT filenames without extension (e.g. `blat_ecolx_stiffler_2015_p62593_wt_model`).

### Phase 2 decoder data

Place these files in the working directory (or update paths in `phase2_decoder/config.py`):

| File | Purpose |
|------|---------|
| `goa_2025-12-04_swissprot_noiea.tsv` | Swiss-Prot GO annotations |
| `esm2_t33_650M_UR50D_protein_embedding.h5` | ESM-2 protein embeddings |
| `go_pretraining_contvar_embeddings.h5` | ContVAR embeddings after Phase 0 |
| `stage2_best_pretraining_protein_embeddings.h5` | ContVAR embeddings after Phase 1 (`contvar_full`) |
| `protein_uniref50.tsv` | Protein → UniRef50 cluster mapping |
| `phase0_go_split.json` | UniRef50 cluster → split assignment |
| `go.obo` | GO hierarchy (for optional propagation) |
| `variant_specific_go_benchmark.tsv` | LOF/GOF benchmark (for `benchmark_eval`) |

---

## Configuration

All local file paths for the encoder pipeline are centralized in `STARTER_PATHS` at the top of [`starter.py`](starter.py). Edit this block before running:

```python
STARTER_PATHS = {
    "data_root": "protein_triplets_data",
    "embeddings_path": "embeddings_variable.h5",
    "dms_protein_split_json_path": "local_splits/dms_protein_split.json",
    "go_protein_split_json_path": "local_splits/phase0_protein_split_removed_graphless.json",
    "go_tsv_dir": "semantic_similarity",
    "go_prebuilt_graph_root": "/path/to/prebuilt_go_graphs",   # directory of .pt graphs
    "go_phase0_init_checkpoint_path": "model_phase0_best_loss.pt",
    # ... checkpoint and export paths ...
}
```

Important notes:

- `go_prebuilt_graph_root` must be a **directory** of `.pt` graph files, not a model checkpoint.
- `starter.py` sets `go_phase0_epochs: 200` to train Phase 0 from scratch. Set `go_phase0_epochs: 0` and provide `go_phase0_init_checkpoint_path` to start Phase 1 from the published Phase 0 weights without retraining.
- Hyperparameters (learning rate, margin, epochs, etc.) live in [`contvar/config.py`](contvar/config.py) (`ProjectConfig`).
- Decoder paths and hyperparameters live in [`phase2_decoder/config.py`](phase2_decoder/config.py) (`DecoderConfig`).

Note: internal code still uses `stage2_*` variable names for Phase 1 checkpoint paths — these refer to the same files (`model_best_loss.pt`, etc.).

---

## Running the pipeline

### 1. Full encoder training (recommended entry point)

```bash
python starter.py
```

This will:

1. Train Phase 0 on GO semantic-similarity triplets (`go_phase0_epochs: 200` by default in `starter.py`).
2. Initialize Phase 1 from the resulting Phase 0 weights.
3. Build or reuse cached graphs under `protein_triplets_data/processed/`.
4. Train Phase 1 with streaming semi-hard negative mining (300 epochs by default).
5. Save `model_best_loss.pt` (lowest validation loss) and `model_last.pt`.
6. Export embeddings to `exports/` and write t-SNE plots to `visualizations/`.

Rebuild all processed graphs from scratch:

```bash
python starter.py --force
```

### 2. Phase 1 from published Phase 0 checkpoint

To skip Phase 0 training and fine-tune directly from the released weights (as in `run.ipynb`):

```python
# In starter.py _build_config_overrides or via train_pipeline config dict:
"go_phase0_epochs": 0,
"go_phase0_init_checkpoint_path": "model_phase0_best_loss.pt",
```

Then run `python starter.py` or follow `run.ipynb`.

### 3. Build graphs from mmCIF structures

To prebuild PyG graphs (for GO proteins or custom structures) without training:

```bash
python -m contvar.prebuild_graphs \
  --structure-dir path/to/cif_files \
  --output-dir path/to/output_graphs \
  --embeddings-h5 embeddings_variable.h5
```

Build graphs and stream them through one or more checkpoints in a single pass:

```bash
python -m contvar.prebuild_graphs \
  --structure-dir path/to/cif_files \
  --output-dir path/to/output_graphs \
  --embeddings-h5 embeddings_variable.h5 \
  --checkpoint model_best_loss.pt \
  --inference-output-dir exports/prebuild_inference
```

### 4. Frozen-checkpoint inference

Export normalized global embeddings from existing `.pt` graph files:

```bash
python -m contvar.inference \
  --checkpoint model_best_loss.pt \
  --graph-root protein_triplets_data/processed \
  --out exports/inference_contvar_embeddings.h5 \
  --batch-size 32
```

### 5. Phase 2 decoder — train and evaluate

Train a GO decoder (aspect: `F`=MF, `P`=BP, `C`=CC):

```bash
python -m phase2_decoder.train --aspect F --embedding concat
```

Embedding modes: `esm`, `contvar`, `contvar_full`, `concat`, `concat_full`.

Evaluate the best saved checkpoint on the held-out test split:

```bash
python -m phase2_decoder.test_eval --aspect F --embedding concat
```

Run the variant-specific LOF/GOF benchmark:

```bash
python -m phase2_decoder.benchmark_eval --aspect F --embedding concat
python -m phase2_decoder.benchmark_eval --all
```

Predict functional changes for variants in an H5 file:

```bash
python -m phase2_decoder.predict_variants \
  --var_h5 embeddings_variable.h5 \
  --out predictions.csv
```

Hyperparameter search:

```bash
python -m phase2_decoder.grid_search --aspect F --embedding concat
```

---

## Outputs

### Encoder (`starter.py` / `train_pipeline`)

| Artifact | Description |
|----------|-------------|
| `model_phase0_best_loss.pt` | Best Phase 0 encoder weights |
| `model_phase0_last.pt` | Final Phase 0 encoder weights |
| `model_best_loss.pt` | Best Phase 1 encoder (validation loss) |
| `model_last.pt` | Final Phase 1 encoder |
| `model_epoch_{N}.pt` | Periodic Phase 1 snapshots (epoch 80, then every 100) |
| `exports/phase0_contvar_embeddings.h5` | Global embeddings for GO proteins |
| `exports/dms_variant_contvar_embeddings.h5` | Global embeddings for DMS variants |
| `visualizations/best/`, `visualizations/last/` | t-SNE plots (global vs local, baseline vs projected) |

### Decoder

| Artifact | Description |
|----------|-------------|
| `decoder_best_{embedding}_{aspect}.pt` | Best decoder checkpoint |
| `go_vocab_{aspect}.json` | GO term vocabulary |

Metrics logged to Weights & Biases include triplet loss, MRR, alignment, uniformity (encoder) and mAP, F1, MCC (decoder).

---

## Reproducibility

| Item | Location |
|------|----------|
| DMS protein-family split | `local_splits/dms_protein_split.json` (version 2) |
| GO protein split | `local_splits/phase0_protein_split_removed_graphless.json` |
| Decoder UniRef50 split | `phase0_go_split.json` |
| Random seed (GO split) | `go_split_seed = 42` in `ProjectConfig` |
| Decoder seed | `seed = 42` in `DecoderConfig` |

Training is deterministic given fixed splits and seeds, but exact GPU numerics may vary slightly across hardware.

### Verification checklist

Run these after preparing data and downloading the release artifacts:

```bash
# 1. Package imports
python -c "from contvar import train_pipeline, ProjectConfig; print('contvar OK')"

# 2. DMS split loads (requires protein_triplets_data/)
python -c "
from contvar.data.mapper import TripletDataPathMapper
m = TripletDataPathMapper('protein_triplets_data', 'local_splits/dms_protein_split.json')
print(f'Families: {len(m.triplets)} | train={len(m.train_triplets)} val={len(m.val_triplets)} test={len(m.test_triplets)}')
"

# 3. Decoder evaluation (requires decoder data + trained checkpoint)
python -m phase2_decoder.test_eval --aspect F --embedding esm
```

---

## Model architecture (summary)

**Encoder (`DeepProteinGAT`):**

- Input nodes: amino-acid one-hot (20-d) + ESM-2 residue embedding (1280-d)
- Edges: SALAD-style hybrid connectivity (default) or Graphein kNN
- Two GATv2 layers with residual connections and edge features
- Projection heads → 256-d global and local embeddings
- Ontology-specific heads for Phase 0 (MF / BP / CC)

**Decoder (`FFNDecoder`):**

- Multi-layer feed-forward network with dropout
- Multi-label sigmoid output over GO vocabulary
- Class-weighted BCE loss with extra weight on rare terms

---

## Citation

If you use this code, please cite the ContVAR paper (add citation when available) and acknowledge the [HUBioDataLab/ContVAR](https://github.com/HUBioDataLab/ContVAR) repository.

---

## License

See repository license file for terms of use.
