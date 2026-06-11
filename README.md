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

A second stage trains feed-forward **decoders** on top of frozen protein embeddings to predict Gene Ontology (GO) annotations and to score variant-induced functional changes.

---

## Pipeline overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 0 (optional) — GO semantic-similarity pretraining                   │
│   semantic_similarity/*.tsv + prebuilt GO graph .pt + protein split JSON  │
│   → model_phase0_best_loss.pt                                             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ warm-start (optional)
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 2 — DMS triplet metric learning                                     │
│   protein_triplets_data/ + ESM2 H5 + dms_protein_split.json              │
│   → model_best_loss.pt, model_last.pt                                     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   Embedding export        t-SNE plots         Frozen inference
   (exports/*.h5)      (visualizations/)    (contvar.inference)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 2 decoder — GO term prediction & variant function scoring           │
│   ESM / ContVAR H5 + GOA + UniRef split → decoder_best_*.pt               │
└─────────────────────────────────────────────────────────────────────────┘
```

**End-to-end local entry point:** `python starter.py` runs Stage 2 training (and optionally Phase 0), then exports embeddings and generates t-SNE plots.

**Colab notebook:** `run.ipynb` walks through the same encoder pipeline with Google Drive data paths.

---

## Repository layout

```text
ContVAR/
├── starter.py                  # Main CLI — edit STARTER_PATHS here
├── run.ipynb                   # Colab reproduction notebook
├── setup.py
├── local_splits/
│   └── dms_protein_split.json  # Fixed protein-family train/val/test split (tracked)
├── contvar/                    # Encoder: model, training, graph building, inference
│   ├── training.py             # train_pipeline()
│   ├── go_pretraining.py       # Phase 0
│   ├── prebuild_graphs.py      # Build PyG graphs from mmCIF + ESM2
│   ├── inference.py            # Frozen-checkpoint embedding export
│   └── export_embeddings.py    # Post-training H5 export
└── phase2_decoder/             # GO decoder training, evaluation, variant prediction
    ├── train.py
    ├── test_eval.py
    ├── benchmark_eval.py
    ├── predict_variants.py
    └── grid_search.py
```

Large data files and trained weights are **not** committed to git (see [Data requirements](#data-requirements)).

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

## Data requirements

Place external files according to the layouts below. Obtain the DMS structures, ESM2 embeddings, GO resources, and prebuilt graphs from your project data release or data-preparation pipeline.

### Stage 2 — DMS variant triplets (required for encoder training)

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

### Phase 0 — GO pretraining (optional)

Only needed when `go_phase0_epochs > 0`.

| Path | Purpose |
|------|---------|
| `semantic_similarity/semantic_similarity_swissprot_filtered_low0.2_high0.8_mf.tsv` | MF semantic-similarity triplets |
| `semantic_similarity/semantic_similarity_swissprot_filtered_low0.2_high0.8_bp.tsv` | BP triplets |
| `semantic_similarity/semantic_similarity_swissprot_filtered_low0.2_high0.8_cc.tsv` | CC triplets |
| `<go_prebuilt_graph_root>/` | Directory tree of prebuilt PyG `.pt` graph files (one per Swiss-Prot protein) |
| `local_splits/phase0_protein_split_removed_graphless.json` | `protein_to_split` mapping for GO proteins |

To **skip Phase 0 training** but still warm-start Stage 2, set `go_phase0_epochs` to `0` and point `go_phase0_init_checkpoint_path` at a saved Phase 0 checkpoint (see [Configuration](#configuration)).

### Phase 2 decoder (optional downstream evaluation)

Place these files in the working directory (or update paths in `phase2_decoder/config.py`):

| File | Purpose |
|------|---------|
| `goa_2025-12-04_swissprot_noiea.tsv` | Swiss-Prot GO annotations |
| `esm2_t33_650M_UR50D_protein_embedding.h5` | ESM-2 protein embeddings |
| `go_pretraining_contvar_embeddings.h5` | ContVAR embeddings after Phase 0 |
| `stage2_best_pretraining_protein_embeddings.h5` | ContVAR embeddings after Stage 2 (`contvar_full`) |
| `protein_uniref50.tsv` | Protein → UniRef50 cluster mapping |
| `phase0_go_split.json` | UniRef50 cluster → split assignment |
| `go.obo` | GO hierarchy (optional propagation) |
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
    "go_prebuilt_graph_root": "/path/to/prebuilt_go_graphs",   # must be a directory
    "go_phase0_init_checkpoint_path": None,                      # or path to .pt checkpoint
    # ... checkpoint and export paths ...
}
```

Important notes:

- `go_prebuilt_graph_root` must be a **directory** of `.pt` graph files, not a model checkpoint.
- `starter.py` sets `go_phase0_epochs: 200` by default. Set this to `0` for DMS-only training, or provide all Phase 0 data files above.
- Hyperparameters (learning rate, margin, epochs, etc.) live in [`contvar/config.py`](contvar/config.py) (`ProjectConfig`).
- Decoder paths and hyperparameters live in [`phase2_decoder/config.py`](phase2_decoder/config.py) (`DecoderConfig`).

---

## Running the pipeline

### 1. Encoder training (recommended entry point)

```bash
python starter.py
```

This will:

1. Run Phase 0 GO pretraining when `go_phase0_epochs > 0` and data paths are valid.
2. Build or reuse cached graphs under `protein_triplets_data/processed/`.
3. Train Stage 2 with streaming semi-hard negative mining (300 epochs by default).
4. Save `model_best_loss.pt` (lowest validation loss) and `model_last.pt`.
5. Export embeddings to `exports/` and write t-SNE plots to `visualizations/`.

Rebuild all processed graphs from scratch:

```bash
python starter.py --force
```

### 2. DMS-only training (no Phase 0)

In `starter.py`, configure:

```python
"go_prebuilt_graph_root": None,
"go_phase0_init_checkpoint_path": "model_go_pretraining_best_loss.pt",  # optional warm-start
```

And in `_build_config_overrides`, set `"go_phase0_epochs": 0`.

Or use `run.ipynb` / call `train_pipeline` directly with the same overrides (see notebook Step 4).

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
| `model_best_loss.pt` | Best Stage 2 encoder (validation loss) |
| `model_last.pt` | Final Stage 2 encoder |
| `model_epoch_{N}.pt` | Periodic snapshots (epoch 80, then every 100) |
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
| DMS protein-family split | `local_splits/dms_protein_split.json` (version 2, committed) |
| GO protein split | `local_splits/phase0_protein_split_removed_graphless.json` (provide locally) |
| Decoder UniRef50 split | `phase0_go_split.json` (provide locally) |
| Random seed (GO split) | `go_split_seed = 42` in `ProjectConfig` |
| Decoder seed | `seed = 42` in `DecoderConfig` |

Training is deterministic given fixed splits and seeds, but exact GPU numerics may vary slightly across hardware.

### Verification checklist

Run these after preparing data to confirm the installation:

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
- Optional ontology-specific heads for Phase 0 (MF / BP / CC)

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
