"""CLI entry point for ContVAR training."""
import argparse
import wandb
from contvar.config import setup_environment
from contvar.training import train_pipeline
from contvar.viz_tsne import visualize_tsne


def main():
    parser = argparse.ArgumentParser(description="ContVAR Training")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to protein_triplets_data directory")
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to ESM2 embeddings h5 file")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess all protein graphs from scratch")
    parser.add_argument("--go-phase0-load", type=str, default=None,
                        help="Initialize from a saved GO phase-0 checkpoint")
    parser.add_argument("--go-phase0-epochs", type=int, default=None,
                        help="Override the number of GO phase-0 epochs")
    parser.add_argument("--wandb-key", type=str, default=None,
                        help="WandB API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--visualize", action="store_true",
                        help="Run t-SNE visualization after training")
    args = parser.parse_args()

    env = setup_environment(
        data_root=args.data_root,
        embeddings_path=args.embeddings,
    )

    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    config_overrides = {}
    if args.go_phase0_load:
        config_overrides["go_phase0_init_checkpoint_path"] = args.go_phase0_load
    if args.go_phase0_epochs is not None:
        config_overrides["go_phase0_epochs"] = args.go_phase0_epochs

    model, mapper, processed_dir = train_pipeline(
        config=config_overrides or None,
        force=args.force,
        data_root=env['data_root'],
        embeddings_path=env['embeddings_path'],
        device=env['device'],
        data_zip=env.get('data_zip'),
    )

    if args.visualize and model is not None:
        visualize_tsne(
            model=model,
            mapper=mapper,
            processed_dir=processed_dir,
            split="val",
            device=env['device'],
        )


if __name__ == "__main__":
    main()
