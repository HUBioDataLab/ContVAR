# ContVAR

ContVAR trains a graph-based metric learning model for single amino acid variants (SAVs).
It uses triplets from the same protein:

- `anchor`: wild-type structure
- `positive`: benign variant
- `negative`: pathogenic variant

The model learns to pull benign variants closer to the anchor and push pathogenic variants farther away.

## Installation

```bash
pip install -e .
pip install torch torch-geometric graphein wandb biopython h5py scikit-learn matplotlib tqdm numpy networkx
```

## Expected data layout

Local defaults assume this repository layout:

```text
ContVAR/
|- starter.py
|- train.py
|- embeddings_variable.h5 (not needed if graphs for dms dataset already exist)
|- local_splits/
|  |- dms_protein_split.json
|  |- phase0_protein_split_removed_graphless.json
|- protein_triplets_data/
|  |- originals/
|  |- positives/
|  |- negatives/
|  ` - processed/
|-
`- semantic_similarity/
   |- semantic_similarity_swissprot_filtered_low0.2_high0.8_mf.tsv
   |- semantic_similarity_swissprot_filtered_low0.2_high0.8_bp.tsv
   `- semantic_similarity_swissprot_filtered_low0.2_high0.8_cc.tsv
```

## Starter CLI

`starter.py` is the main entry point for local runs.
It centralizes the runtime paths in one place through the `STARTER_PATHS` block near the top of the file.

If you prefer the old command, `train.py` still works and forwards to the same starter CLI.

### 1. Review or edit the default paths

Open [starter.py](starter.py) and update `STARTER_PATHS` if you want machine-specific defaults in one place.

The block already uses the same repo-local paths that the previous version was using:

- `protein_triplets_data/`
- `embeddings_variable.h5`
- `local_splits/dms_protein_split.json`
- `local_splits/phase0_protein_split_removed_graphless.json`
- `semantic_similarity/`
- checkpoints in the repo root

You can inspect the resolved paths without starting training:

```bash
python starter.py --print-paths
```

### 2. Run a DMS-only training job

If `STARTER_PATHS["go_prebuilt_graph_root"]` is left as `None`, the starter script automatically disables GO phase-0 pretraining and runs stage-2 DMS training only.

```bash
python starter.py
```

### 3. Run full training with GO phase-0 enabled

Set `STARTER_PATHS["go_prebuilt_graph_root"]` in `starter.py`, then run:

```bash
python starter.py --go-phase0-epochs 2
```

## Common CLI options

- `--force`: rebuild processed protein graphs from scratch
- `--go-phase0-epochs`: override GO phase-0 epochs, use `0` to skip it
- `--wandb-key`: log in to Weights and Biases from the CLI
- `--visualize`: generate t-SNE plots after training
- `--print-paths`: print the resolved path configuration and exit

## Output files

By default, local runs write:

- `model_phase0_best_loss.pt`
- `model_phase0_last.pt`
- `model_best_loss.pt`
- `model_last.pt`
- `visualizations/`

## Notes

- GO phase-0 requires all of the following:
  `STARTER_PATHS["go_prebuilt_graph_root"]`, the GO TSV directory, and the GO split JSON.
- Stage-2 DMS training uses `protein_triplets_data`, `embeddings_variable.h5`, and the DMS split JSON.
- The training loop now reads checkpoint paths from configuration, and `starter.py` is the intended single place to edit local file paths.
