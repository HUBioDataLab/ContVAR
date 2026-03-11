import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from contvar.config import ProjectConfig
from contvar.data.mapper import TripletDataPathMapper
from contvar.data.dataset import TripletProteinGraphDataset, ExhaustiveTripletDataset
from contvar.data.collate import triplet_collate
from contvar.model import DeepProteinGAT
from contvar.losses import StandardTripletLoss, SemiHardMiningTripletLoss
from contvar.mining import streaming_mining_batch_iterator
from contvar.metrics import compute_detailed_metrics, compute_embedding_stats
from contvar.utils import load_all_embeddings


def evaluate(model, loader, criterion, device, margin=0.3):
    """Evaluate model on a given dataloader with both global and local loss"""
    model.eval()
    total_loss = 0
    total_loss_g = 0
    total_loss_l = 0
    valid_batches = 0
    all_metrics = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            ba, bp, bn, neg_counts, mut_pos_positive, mut_pos_negatives = batch
            ba = ba.to(device)
            bp = bp.to(device)
            bn = bn.to(device)
            neg_counts = neg_counts.to(device)
            mut_pos_positive = mut_pos_positive.to(device)
            mut_pos_negatives = mut_pos_negatives.to(device)

            # Global embeddings
            ea_g, _ = model(ba)
            ep_g, ep_l = model(bp, mut_pos=mut_pos_positive)
            en_g, en_l = model(bn, mut_pos=mut_pos_negatives)

            # Global loss
            loss_g, neg_dist, en_neg, mining_stats = criterion(ea_g, ep_g, en_g, neg_counts)

            # Local loss
            hardest_indices = mining_stats["hardest_indices"]
            cumsum = torch.cat([torch.tensor([0], device=device), neg_counts.cumsum(0)[:-1]])
            flat_idx = cumsum + hardest_indices
            mut_pos_neg_selected = mut_pos_negatives[flat_idx]

            _, la_at_pos = model(ba, mut_pos=mut_pos_positive)
            _, la_at_neg = model(ba, mut_pos=mut_pos_neg_selected)
            zn_l_selected = en_l[flat_idx]

            B = la_at_pos.size(0)
            z_wt_l = torch.cat([la_at_pos, la_at_neg], dim=0)
            z_mut_l = torch.cat([ep_l, zn_l_selected], dim=0)
            lbl = torch.cat([
                torch.ones(B, device=device),
                torch.zeros(B, device=device)
            ])

            d_local = F.pairwise_distance(z_wt_l, z_mut_l, p=2)
            loss_attract = lbl * (d_local ** 2)
            loss_repel = (1.0 - lbl) * (F.relu(margin - d_local) ** 2)
            loss_l = (loss_attract + loss_repel).mean()

            # Combined loss
            loss = (loss_g + loss_l) / 2

            total_loss += loss.item()
            total_loss_g += loss_g.item()
            total_loss_l += loss_l.item()
            valid_batches += 1

            k_vals = [1, 5] if len(ea_g) >= 5 else [1]
            batch_metrics = compute_detailed_metrics(ea_g, ep_g, en_neg, top_k=k_vals)
            batch_metrics["loss"] = loss.item()
            batch_metrics["loss_g"] = loss_g.item()
            batch_metrics["loss_l"] = loss_l.item()
            all_metrics.append(batch_metrics)

    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0

    aggregated = {"loss": avg_loss}
    if all_metrics:
        metric_keys = [k for k in all_metrics[0].keys() if k != "loss"]
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregated[key] = np.mean(values) if values else 0.0

    return aggregated


def train_pipeline(config=None, force=False, split_path=None,
                   data_root=None, embeddings_path=None, device=None):
    """
    Main training pipeline with CURRICULUM LEARNING support.

    Args:
        config: dict of config overrides (e.g. from wandb sweep)
        force: If True, reprocess all protein graphs from scratch
        split_path: Path to existing split JSON for reproducibility
        data_root: Path to protein_triplets_data directory
        embeddings_path: Path to ESM2 embeddings h5 file
        device: torch device (auto-detected if None)
    """
    # Initialize config
    cfg = ProjectConfig()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize WandB
    run = wandb.init(
        project="ContVAR-Project",
        config=vars(cfg),
        reinit=True
    )

    # Update config from wandb if sweep is running
    if config:
        for key, value in config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    # Resolve paths
    if data_root is None:
        from contvar.config import setup_environment
        env = setup_environment()
        data_root = env['data_root']
        if embeddings_path is None:
            embeddings_path = env['embeddings_path']

    print(f"Training with LR: {cfg.lr}, Hidden: {cfg.hidden_dim}, Heads: {cfg.heads}")
    print(f"Curriculum Learning: {cfg.curriculum_warmup_epochs} warm-up epochs with exhaustive sampling")
    print(f"Streaming Mining: chunk_size={cfg.mining_chunk_size}, max_negatives={cfg.max_negatives}")
    print(f"Local Loss: Contrastive (attract good / repel bad at mutation position)")
    if cfg.phase1_early_stop:
        print(f"Phase 1 Early Stopping: ON (threshold={cfg.phase1_es_threshold}, "
              f"window={cfg.phase1_es_window}, patience={cfg.phase1_es_patience}, "
              f"min_batches={cfg.phase1_es_min_batches})")
    else:
        print(f"Phase 1 Early Stopping: OFF")

    # Load data with per-family hold-out split
    mapper = TripletDataPathMapper(data_root, val_pos=2, val_neg=2, seed=42, split_path=split_path)
    if not mapper.triplets:
        print("No data found!")
        wandb.finish()
        return

    mapper.save_split()

    shared_embeddings = None
    if force and embeddings_path:
        shared_embeddings = load_all_embeddings(embeddings_path)

    # =========================================================================
    # CREATE DATASETS FOR BOTH PHASES
    # =========================================================================
    print("\n=== Phase 1 Setup: Exhaustive Sampling ===")

    main_train_dataset = TripletProteinGraphDataset(
        mapper, root=data_root, config=cfg, split='train',
        esm2_embedding_path=embeddings_path,
        force=force, preloaded_embeddings=shared_embeddings
    )

    exhaustive_train_dataset = ExhaustiveTripletDataset(
        mapper, root=data_root, config=cfg, split='train',
        preloaded_embeddings=shared_embeddings
    )
    exhaustive_val_dataset = ExhaustiveTripletDataset(
        mapper, root=data_root, config=cfg, split='val',
        preloaded_embeddings=shared_embeddings
    )

    print("\n=== Phase 2 Setup: Streaming Semi-Hard Mining ===")

    # =========================================================================
    # CREATE DATALOADERS
    # =========================================================================
    phase1_train_loader = DataLoader(
        exhaustive_train_dataset,
        batch_size=cfg.warmup_batch_size,
        shuffle=True,
        collate_fn=triplet_collate,
        num_workers=getattr(cfg, "num_workers", 0)
    )
    phase1_val_loader = DataLoader(
        exhaustive_val_dataset,
        batch_size=cfg.warmup_batch_size,
        shuffle=False,
        collate_fn=triplet_collate,
        num_workers=getattr(cfg, "num_workers", 0)
    )

    processed_dir = os.path.join(data_root, "processed")
    num_phase2_train_batches = (len(main_train_dataset.triplets) + cfg.mining_batch_size - 1) // cfg.mining_batch_size

    print(f"\nDataset sizes:")
    print(f"  Phase 1 Train (exhaustive): {len(exhaustive_train_dataset):,} triplets")
    print(f"  Val (exhaustive, both phases): {len(exhaustive_val_dataset):,} triplets")
    print(f"  Phase 2 Train (streaming mining): {len(main_train_dataset.triplets)} families x train negatives")

    # =========================================================================
    # INITIALIZE MODEL
    # =========================================================================
    model = DeepProteinGAT(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        heads=cfg.heads,
        edge_dim=cfg.edge_attr_dim
    ).to(device)

    wandb.watch(model, log="gradients", log_freq=50)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    standard_criterion = StandardTripletLoss(margin=cfg.margin)
    semihard_criterion = SemiHardMiningTripletLoss(margin=cfg.margin)

    val_criterion = standard_criterion

    print("\n" + "="*60)
    print("STARTING CURRICULUM LEARNING")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(cfg.epochs):
        # =====================================================================
        # PHASE SELECTION
        # =====================================================================
        is_warmup_phase = epoch < cfg.curriculum_warmup_epochs

        if is_warmup_phase:
            train_loader = phase1_train_loader
            criterion = standard_criterion
            phase_name = "WARMUP (Exhaustive + Standard Loss)"
            current_batch_size = cfg.warmup_batch_size

            if epoch > 0:
                exhaustive_train_dataset.reshuffle()
        else:
            train_loader = streaming_mining_batch_iterator(
                model, main_train_dataset.triplets, processed_dir, device, cfg
            )
            criterion = semihard_criterion
            phase_name = "MINING (Streaming Semi-Hard)"
            current_batch_size = cfg.mining_batch_size

        if is_warmup_phase:
            num_train_batches = len(train_loader)
        else:
            num_train_batches = num_phase2_train_batches

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{cfg.epochs} | Phase: {phase_name}")
        print(f"Batch size: {current_batch_size} | Train batches: ~{num_train_batches}")
        print(f"{'='*60}")

        # =====================================================================
        # TRAINING
        # =====================================================================
        epoch_start = time.time()
        model.train()
        total_loss = 0
        total_loss_g = 0
        total_loss_l = 0
        valid_batches = 0
        train_metrics_list = []

        epoch_streaming_hard_total = 0
        epoch_streaming_semi_hard_total = 0
        epoch_streaming_evaluated_total = 0
        epoch_streaming_qualifying_total = 0

        epoch_local_easy_total = 0
        epoch_local_semi_hard_total = 0
        epoch_local_hard_total = 0

        phase1_early_stopped = False
        if is_warmup_phase and cfg.phase1_early_stop:
            es_loss_buffer = deque(maxlen=cfg.phase1_es_window)
            es_patience_counter = 0

        mining_batch_table = wandb.Table(columns=[
            "batch_idx", "hard_count", "semi_hard_count", "total_evaluated", "total_qualifying"
        ])

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, total=num_train_batches)

        for batch in pbar:
            if batch is None:
                continue

            if not is_warmup_phase:
                ba, bp, bn, neg_counts, mut_pos_positive, mut_pos_negatives, streaming_info = batch
            else:
                ba, bp, bn, neg_counts, mut_pos_positive, mut_pos_negatives = batch
                streaming_info = None

            if ba.num_graphs < 2:
                continue

            ba = ba.to(device)
            bp = bp.to(device)
            bn = bn.to(device)
            neg_counts = neg_counts.to(device)
            mut_pos_positive = mut_pos_positive.to(device)
            mut_pos_negatives = mut_pos_negatives.to(device)

            optimizer.zero_grad()

            ea_g, _ = model(ba)
            ep_g, ep_l = model(bp, mut_pos=mut_pos_positive)
            en_g, en_l = model(bn, mut_pos=mut_pos_negatives)

            loss_g, neg_dist, en_neg, mining_stats = criterion(ea_g, ep_g, en_g, neg_counts)
            hardest_indices = mining_stats["hardest_indices"]
            cumsum = torch.cat([torch.tensor([0], device=device), neg_counts.cumsum(0)[:-1]])
            flat_idx = cumsum + hardest_indices
            mut_pos_neg_selected = mut_pos_negatives[flat_idx]

            # =================================================================
            # LOCAL CONTRASTIVE LOSS
            # =================================================================
            _, la_at_pos = model(ba, mut_pos=mut_pos_positive)
            _, la_at_neg = model(ba, mut_pos=mut_pos_neg_selected)
            zn_l_selected = en_l[flat_idx]

            B = la_at_pos.size(0)

            z_wt_l = torch.cat([la_at_pos, la_at_neg], dim=0)
            z_mut_l = torch.cat([ep_l, zn_l_selected], dim=0)
            lbl = torch.cat([
                torch.ones(B, device=device),
                torch.zeros(B, device=device)
            ])

            d_local = F.pairwise_distance(z_wt_l, z_mut_l, p=2)

            loss_attract = lbl * (d_local ** 2)
            loss_repel = (1.0 - lbl) * (F.relu(cfg.margin - d_local) ** 2)
            loss_l = (loss_attract + loss_repel).mean()

            d_pos_l = d_local[:B]
            d_neg_l = d_local[B:]

            # Local contrastive mining stats
            with torch.no_grad():
                neg_mask_l = (lbl == 0)
                if neg_mask_l.sum() > 0:
                    d_neg_stats = d_local[neg_mask_l]
                    batch_local_easy = int((d_neg_stats > cfg.margin).sum().item())
                    batch_local_hard = int((d_neg_stats < cfg.margin * 0.5).sum().item())
                    batch_local_semi = int(((d_neg_stats >= cfg.margin * 0.5) & (d_neg_stats <= cfg.margin)).sum().item())
                else:
                    batch_local_easy = batch_local_hard = batch_local_semi = 0

                epoch_local_easy_total += batch_local_easy
                epoch_local_hard_total += batch_local_hard
                epoch_local_semi_hard_total += batch_local_semi

            # =================================================================
            # COMBINED LOSS
            # =================================================================
            loss = (loss_g + loss_l) / 2
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                k_vals = [1, 5] if current_batch_size >= 5 else [1]
                batch_metrics = compute_detailed_metrics(ea_g, ep_g, en_neg, top_k=k_vals)
                dist_pos = F.pairwise_distance(ea_g, ep_g)
                train_metrics_list.append(batch_metrics)

            log_payload = {
                "train/batch_loss": loss.item(),
                "train/batch_loss_g": loss_g.item(),
                "train/batch_loss_l": loss_l.item(),
                "train/Alignment": batch_metrics["Alignment"],
                "train/Uniformity": batch_metrics["Uniformity"],
                "train/avg_pos_dist": dist_pos.mean().item(),
                "train/avg_neg_dist": neg_dist.mean().item(),
                "train/dist_margin": (neg_dist.mean() - dist_pos.mean()).item(),
                "train/AUROC": batch_metrics["AUROC"],
                "train/Simple_Acc": batch_metrics["Simple_Acc"],
                "train/MRR": batch_metrics["MRR"],
                "train/R@1": batch_metrics["R@1"],
                "train/total_negatives": len(en_g),
                "train/phase": 1 if is_warmup_phase else 2,
                "train/batch_size": current_batch_size,
                "train/avg_pos_dist_l": d_pos_l.mean().item(),
                "train/avg_neg_dist_l": d_neg_l.mean().item(),
                "train/dist_margin_l": (d_neg_l.mean() - d_pos_l.mean()).item(),
                "local/batch_easy": batch_local_easy,
                "local/batch_semi_hard": batch_local_semi,
                "local/batch_hard": batch_local_hard,
            }

            if "R@5" in batch_metrics:
                log_payload["train/R@5"] = batch_metrics["R@5"]

            if streaming_info is not None:
                log_payload["streaming/batch_hard_count"] = streaming_info["streaming_hard"]
                log_payload["streaming/batch_semi_hard_count"] = streaming_info["streaming_semi_hard"]
                log_payload["streaming/batch_total_evaluated"] = streaming_info["total_evaluated"]
                log_payload["streaming/batch_total_qualifying"] = streaming_info["total_qualifying"]
                log_payload["streaming/batch_qualifying_ratio"] = (
                    streaming_info["total_qualifying"] / streaming_info["total_evaluated"]
                    if streaming_info["total_evaluated"] > 0 else 0
                )

            if epoch < 2:
                log_payload.update({
                    "step/batch_loss": loss.item(),
                    "step/Alignment": batch_metrics["Alignment"],
                    "step/Uniformity": batch_metrics["Uniformity"],
                    "step/avg_pos_dist": dist_pos.mean().item(),
                    "step/avg_neg_dist": neg_dist.mean().item(),
                    "step/dist_margin": (neg_dist.mean() - dist_pos.mean()).item(),
                    "step/AUROC": batch_metrics["AUROC"],
                    "step/Simple_Acc": batch_metrics["Simple_Acc"],
                    "step/MRR": batch_metrics["MRR"],
                    "step/R@1": batch_metrics["R@1"],
                    "step/epoch": epoch + 1,
                    "step/batch_idx": valid_batches,
                })
                if "R@5" in batch_metrics:
                    log_payload["step/R@5"] = batch_metrics["R@5"]

            wandb.log(log_payload)

            if streaming_info is not None:
                epoch_streaming_hard_total += streaming_info["streaming_hard"]
                epoch_streaming_semi_hard_total += streaming_info["streaming_semi_hard"]
                epoch_streaming_evaluated_total += streaming_info["total_evaluated"]
                epoch_streaming_qualifying_total += streaming_info["total_qualifying"]

            if streaming_info is not None:
                mining_batch_table.add_data(
                    valid_batches,
                    streaming_info["streaming_hard"],
                    streaming_info["streaming_semi_hard"],
                    streaming_info["total_evaluated"],
                    streaming_info["total_qualifying"],
                )

            total_loss += loss.item()
            total_loss_g += loss_g.item()
            total_loss_l += loss_l.item()
            valid_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'auc': f'{batch_metrics["AUROC"]:.3f}',
                'phase': 1 if is_warmup_phase else 2
            })

            # =================================================================
            # PHASE 1 INTRA-EPOCH EARLY STOPPING CHECK
            # =================================================================
            if is_warmup_phase and cfg.phase1_early_stop:
                es_loss_buffer.append(loss.item())

                if (valid_batches >= cfg.phase1_es_min_batches
                        and len(es_loss_buffer) == cfg.phase1_es_window):
                    window_mean = sum(es_loss_buffer) / len(es_loss_buffer)

                    if window_mean < cfg.phase1_es_threshold:
                        es_patience_counter += 1
                    else:
                        es_patience_counter = 0

                    if es_patience_counter >= cfg.phase1_es_patience:
                        phase1_early_stopped = True
                        print(f"\n  >>> Phase 1 early stop triggered at batch {valid_batches}/{num_train_batches} "
                              f"(window_mean_loss={window_mean:.6f} < {cfg.phase1_es_threshold} "
                              f"for {cfg.phase1_es_patience} consecutive windows)")
                        pbar.close()
                        break

        avg_train_loss = total_loss / valid_batches if valid_batches > 0 else 0
        avg_train_loss_g = total_loss_g / valid_batches if valid_batches > 0 else 0
        avg_train_loss_l = total_loss_l / valid_batches if valid_batches > 0 else 0
        epoch_duration_sec = time.time() - epoch_start

        train_epoch_metrics = {}
        if train_metrics_list:
            for key in train_metrics_list[0].keys():
                values = [m[key] for m in train_metrics_list]
                train_epoch_metrics[key] = np.mean(values)

        # =====================================================================
        # VALIDATION
        # =====================================================================
        val_metrics = evaluate(model, phase1_val_loader, val_criterion, device, margin=cfg.margin)

        embedding_stats = compute_embedding_stats(
            model, phase1_val_loader, device, val_criterion, max_batches=20
        )
        log_dict = {
            "train/epoch_loss": avg_train_loss,
            "train/epoch_loss_g": avg_train_loss_g,
            "train/epoch_loss_l": avg_train_loss_l,
            "train/epoch_AUROC": train_epoch_metrics.get("AUROC", 0),
            "train/epoch_Simple_Acc": train_epoch_metrics.get("Simple_Acc", 0),
            "train/epoch_MRR": train_epoch_metrics.get("MRR", 0),
            "val/loss": val_metrics.get("loss", 0),
            "val/loss_g": val_metrics.get("loss_g", 0),
            "val/loss_l": val_metrics.get("loss_l", 0),
            "val/AUROC": val_metrics.get("AUROC", 0),
            "val/Simple_Acc": val_metrics.get("Simple_Acc", 0),
            "val/MRR": val_metrics.get("MRR", 0),
            "val/R@1": val_metrics.get("R@1", 0),
            "val/Alignment": val_metrics.get("Alignment", 0),
            "val/Uniformity": val_metrics.get("Uniformity", 0),
            "train/epoch_duration_sec": epoch_duration_sec,
            "train/epoch_overfit_gap": val_metrics.get("loss", 0) - avg_train_loss,
            "train/epoch_loss_ratio_g": avg_train_loss_g / avg_train_loss if avg_train_loss > 0 else 0,
            "train/epoch_loss_ratio_l": avg_train_loss_l / avg_train_loss if avg_train_loss > 0 else 0,
            "train/epoch_num_batches": valid_batches,
        }

        for k, v in embedding_stats.items():
            log_dict[f"embedding_stats/{k}"] = v

        log_dict["streaming/epoch_hard_total"] = epoch_streaming_hard_total
        log_dict["streaming/epoch_semi_hard_total"] = epoch_streaming_semi_hard_total
        log_dict["streaming/epoch_total_evaluated"] = epoch_streaming_evaluated_total
        log_dict["streaming/epoch_total_qualifying"] = epoch_streaming_qualifying_total
        log_dict["streaming/epoch_qualifying_ratio"] = (
            epoch_streaming_qualifying_total / epoch_streaming_evaluated_total
            if epoch_streaming_evaluated_total > 0 else 0
        )
        log_dict["streaming/epoch_easy_total"] = (
            epoch_streaming_evaluated_total - epoch_streaming_qualifying_total
        )

        log_dict["local/epoch_easy_total"] = epoch_local_easy_total
        log_dict["local/epoch_semi_hard_total"] = epoch_local_semi_hard_total
        log_dict["local/epoch_hard_total"] = epoch_local_hard_total

        # Determine which epochs get persistent bar plots
        middle_epoch = cfg.epochs // 2
        snapshot_epochs = {1, middle_epoch, cfg.epochs - 1}

        if epoch in snapshot_epochs:
            tag = {1: "first", middle_epoch: "middle", cfg.epochs - 1: "last"}[epoch]

            epoch_easy = epoch_streaming_evaluated_total - epoch_streaming_qualifying_total
            mining_table = wandb.Table(
                data=[
                    ("Hard", epoch_streaming_hard_total),
                    ("Semi-hard", epoch_streaming_semi_hard_total),
                    ("Easy (discarded)", epoch_easy),
                ],
                columns=["Type", "Count"]
            )
            mining_bar = wandb.plot.bar(
                mining_table, label="Type", value="Count",
                title=f"ALL-neg mining counts — {tag} (Epoch {epoch+1})"
            )
            log_dict[f"mining_counts/{tag}_mining_bar"] = mining_bar

            local_mining_table = wandb.Table(
                data=[
                    ("Hard", epoch_local_hard_total),
                    ("Semi-hard", epoch_local_semi_hard_total),
                    ("Easy", epoch_local_easy_total),
                ],
                columns=["Type", "Count"]
            )
            local_mining_bar = wandb.plot.bar(
                local_mining_table, label="Type", value="Count",
                title=f"LOCAL contrastive mining — {tag} (Epoch {epoch+1})"
            )
            log_dict[f"local_mining/{tag}_local_bar"] = local_mining_bar

            hard_bar_per_batch = wandb.plot.bar(
                mining_batch_table,
                label="batch_idx",
                value="hard_count",
                title=f"ALL-neg hard per batch — {tag} (Epoch {epoch+1})",
            )
            semi_hard_bar_per_batch = wandb.plot.bar(
                mining_batch_table,
                label="batch_idx",
                value="semi_hard_count",
                title=f"ALL-neg semi-hard per batch — {tag} (Epoch {epoch+1})",
            )
            log_dict[f"mining_counts/{tag}_hard_per_batch_bar"] = hard_bar_per_batch
            log_dict[f"mining_counts/{tag}_semi_hard_per_batch_bar"] = semi_hard_bar_per_batch

        # Save best model
        if val_metrics.get("loss", float('inf')) < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model_name = "model_best_loss.pt"
            torch.save(model.state_dict(), model_name)

            artifact = wandb.Artifact(
                name=f"ContVAR-Best-Model-{wandb.run.id}",
                type="model",
                description=f"Best model at epoch {epoch+1} (Phase {'1-Warmup' if is_warmup_phase else '2-Mining'}) with val_loss {best_val_loss:.4f}"
            )
            artifact.add_file(model_name)
            wandb.log_artifact(artifact)
            log_dict["best_model_saved"] = True
        log_dict["val/best_loss_so_far"] = best_val_loss

        wandb.log(log_dict)

        phase_str = "P1-Warmup" if is_warmup_phase else "P2-Mining"
        saved_str = "(Saved)" if log_dict.get("best_model_saved") else ""
        es_str = " [EARLY STOPPED]" if phase1_early_stopped else ""
        print(f"[{phase_str}] Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
              f"Val AUROC: {val_metrics.get('AUROC', 0):.4f} | "
              f"Local[E:{epoch_local_easy_total} S:{epoch_local_semi_hard_total} H:{epoch_local_hard_total}] "
              f"{saved_str}{es_str}")

    wandb.finish()
    print("\nTraining completed!")
    return model
