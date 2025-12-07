#!/usr/bin/env python3
"""
🧪 Federated Learning Baseline Runner
------------------------------------

This module orchestrates the execution of federated learning baseline experiments.
It serves as the central driver for training, evaluating, and aggregating models 
across simulated distributed clients using various federated algorithms.

🧠 Purpose:
    To provide a robust, reproducible, and instrumented runtime environment for 
    comparing federated learning strategies (e.g., FedAvg, FedProx, FedDyn). It 
    manages the lifecycle of an experiment, from data partitioning to metric 
    serialization.

🔧 Core Functionalities:
    • Efficient data loading and conversion of subsets to high-performance tensors
    • Dynamic management of client selection and local training epochs
    • Global aggregation of client model updates
    • Real-time evaluation on hold-out validation sets and final testing
    • Automated early stopping and checkpointing of best-performing models
    • Comprehensive metric tracking (loss, accuracy, cost)

🎯 Intended Use:
    • Academic research for benchmarking novel federated algorithms
    • Comparative analysis of model convergence and communication costs
    • Production-grade simulation of distributed learning environments

📁 Dependencies:
    • numpy
    • torch (PyTorch)
    • sklearn (Scikit-learn)
    • models.baselines (Internal)
    • metrics.costs (Internal)

📝 Notes:
    This module assumes that data shuffling and partitioning have been handled
    upstream. It relies on a specific directory structure for saving models
    and metrics.

Author: Andrea Moleri
File Location: src/jobs/baseline_runner.py
Last Modified: 21/11/2025
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from imports.data_management import DATASET_META

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset, Dataset

# Models & Metrics
from models.baselines import (
    FederatedBaseline,
    FedAvgBaseline,
    FedProxBaseline,
    FedDFBaseline,
    FedDynBaseline,
    ScaffoldBaseline
)
from metrics.costs import ExperimentCostTracker
from imports.data_management import get_dataset
from imports.data_augmentation import build_transform

logger = logging.getLogger(__name__)


class TransformSubset(Dataset):
    """
    A wrapper that overrides the transform of a subset or dataset.
    Essential for ensuring:
    1. Training data gets random augmentation.
    2. Validation/Test data gets deterministic formatting.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        # Applies the new transform (augmentation or deterministic crop)
        # regardless of what the underlying dataset might have done,
        # assuming compatibility (e.g. PIL -> Tensor).
        return self.transform(x), y

    def __len__(self):
        return len(self.subset)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _default_num_workers() -> int:
    """
    Determine the optimal number of data loading workers.
    Respects MAX_WORKERS env var if set, otherwise falls back to heuristics.
    """
    import os
    # 1. Check for explicit limit from Shell Script (Critical for Parallel Jobs)
    env_max = os.environ.get("MAX_WORKERS")
    if env_max is not None and env_max.strip() != "":
        try:
            return int(env_max)
        except ValueError:
            pass

    # 2. Fallback heuristic
    try:
        # Reserve one core for the main process
        return max(1, (os.cpu_count() or 2) - 1)
    except Exception:
        return 4


def subset_to_tensor(
        subset: torch.utils.data.Dataset,
        batch_size: int = 1024,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        limit: Optional[int] = None,
) -> torch.Tensor:
    """
    Efficiently converts a Dataset or Subset into a single contiguous Tensor.
    """
    if num_workers is None:
        num_workers = _default_num_workers()

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    xs: List[torch.Tensor] = []
    n_acc = 0

    for x, _ in loader:
        if limit is not None and n_acc >= limit:
            break
        if limit is not None and n_acc + x.size(0) > limit:
            xs.append(x[: limit - n_acc])
            n_acc = limit
            break
        xs.append(x)
        n_acc += x.size(0)

    if not xs:
        return torch.empty(0)

    return torch.cat(xs, dim=0)


def dataset_to_tensor(
        ds: torch.utils.data.Dataset,
        batch_size: int = 1024,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
) -> torch.Tensor:
    """
    Wrapper function to convert a full Dataset object into a single Tensor.
    """
    return subset_to_tensor(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def evaluate_single_classifier(
        model: torch.nn.Module, ld: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform inference using a specific model on a given DataLoader.
    """
    model.to(device).eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in ld:
            y_true.append(y)
            y_pred.append(model(x.to(device)).argmax(1).cpu())

    model.cpu()
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


# ---------------------------------------------------------------------------
# Main baseline runner
# ---------------------------------------------------------------------------

def run_federated_baseline(
        baseline: FederatedBaseline,
        train_subsets_dict: Dict[str, Dict[int, Subset]],
        test_subsets_dict: Dict[str, Dict[int, Subset]],
        base_train_set,
        args,
        device,
        P,  # PathRegistry instance
        tracker: Optional[ExperimentCostTracker] = None,
        # --- NEW ARGUMENTS ---
        train_transform_override: Optional[any] = None,
        eval_transform_override: Optional[any] = None,
) -> Tuple[float, Dict, Dict, np.ndarray, np.ndarray]:
    """
    Execute the complete federated learning baseline experiment lifecycle.
    """
    logger.info(f"[BASELINE] Starting {type(baseline).__name__} federated learning")

    # 1. Resolve key
    base_key = args.dataset.split("(", 1)[0].lower()
    if base_key.startswith("medmnist"): base_key = "medmnist"

    # 2. Find metadata pointer
    meta_ptr = DATASET_META.get(base_key) or DATASET_META.get(args.dataset)
    if meta_ptr is None:
        # Fallback search
        for k in DATASET_META:
            if k in args.dataset:
                meta_ptr = DATASET_META[k]
                break

    # 3. Apply override
    if meta_ptr and meta_ptr.get("input_size") != args.input_size:
        logger.info(f"⚠️ Overriding dataset default resolution to: {args.input_size}")
        meta_ptr["input_size"] = args.input_size

    # ---------------------------------------------------------------
    # 0. Initialize Augmentation Strategies (NeurIPS Correction)
    # ---------------------------------------------------------------
    # specific=True -> Random Augmentation (Crops/Flips)
    # specific=False -> Deterministic (Resize/CenterCrop)

    # CHECK: Did experiment_setup pass us a robust transform?
    if train_transform_override is not None:
        logger.info("[BASELINE] Using provided TRAIN transform (preserving robustness/noise settings)")
        train_transform = train_transform_override
    else:
        # Fallback: Build it here (Note: might miss robustness flag if not passed in args)
        use_robustness = getattr(args, "robustness", False)
        train_transform = build_transform(args.dataset, train=True, robustness=use_robustness)

    # CHECK: Did experiment_setup pass us a deterministic eval transform?
    if eval_transform_override is not None:
        eval_transform = eval_transform_override
    else:
        eval_transform = build_transform(args.dataset, train=False)

    # ---------------------------------------------------------------
    # Initialize models
    # ---------------------------------------------------------------
    client_names = list(train_subsets_dict.keys())
    baseline.initialize_models(client_names)

    all_client_samples: Dict[str, int] = {}
    for client_name, class_subsets in train_subsets_dict.items():
        total_samples = sum(len(subset) for subset in class_subsets.values())
        all_client_samples[client_name] = total_samples
        logger.info(f"[BASELINE] Client {client_name}: {total_samples} total samples")

    # ---------------------------------------------------------------
    # Clients per round Configuration
    # ---------------------------------------------------------------
    total_clients = len(client_names)

    if hasattr(args, "clients_per_round") and args.clients_per_round is not None:
        clients_per_round = min(args.clients_per_round, total_clients)
    else:
        client_fraction = getattr(args, "client_fraction", 1.0)
        clients_per_round = max(1, int(total_clients * client_fraction))

    clients_per_round = min(clients_per_round, total_clients)
    logger.info(
        f"[BASELINE] Total clients: {total_clients}, "
        f"Clients per round: {clients_per_round}"
    )

    # ---------------------------------------------------------------
    # Official Test Loader Construction
    # ---------------------------------------------------------------
    try:
        # Use deterministic eval_transform for the global test set
        test_set = get_dataset(
            args.dataset,
            args.data_dir,
            False,  # is_train=False
            eval_transform,
        )
        test_imgs = dataset_to_tensor(test_set)

        labels_src = None
        for attr in ("targets", "labels"):
            if hasattr(test_set, attr):
                labels_src = getattr(test_set, attr)
                break
        if labels_src is None and hasattr(test_set, "imgs"):
            labels_src = [t for _, t in test_set.imgs]

        if labels_src is None:
            test_lbls = torch.zeros(len(test_imgs), dtype=torch.long)
        else:
            test_lbls = torch.as_tensor(labels_src, dtype=torch.long).reshape(-1)

        test_loader = DataLoader(
            TensorDataset(test_imgs, test_lbls),
            batch_size=args.batch_size,
            shuffle=False,
        )
        logger.info(
            f"[BASELINE] Created test loader with {len(test_imgs)} samples using deterministic transform."
        )
    except Exception as e:
        logger.warning(f"[BASELINE] Could not create proper test loader: {e}")
        test_loader = DataLoader(
            TensorDataset(torch.tensor([]), torch.tensor([])),
            batch_size=args.batch_size,
        )

    # ---------------------------------------------------------------
    # Precompute Train/Val Splits for Each Client
    # ---------------------------------------------------------------
    # train_data is kept as Dataset/Subset to ensure Dynamic Augmentation
    # val_data is converted to Tensors for Deterministic Evaluation & Speed
    client_train_data: Dict[str, Dataset] = {}
    client_val_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for client_name in client_names:
        # 1. Consolidate all class-subsets for this client into one dataset
        client_subsets = list(train_subsets_dict[client_name].values())
        if not client_subsets:
            continue

        full_client_dataset = ConcatDataset(client_subsets)

        # 2. Handle Validation Split

        # Case A: Explicit Validation Set (from Partitioning)
        if client_name in test_subsets_dict and test_subsets_dict[client_name]:
            # Convert validation data to tensors immediately (Freeze)
            xs_val = []
            ys_val = []
            for class_id, val_subset in test_subsets_dict[client_name].items():
                # APPLIED CORRECTION: Wrap validation subset with deterministic transform
                val_wrapped = TransformSubset(val_subset, eval_transform)

                imgs_val = subset_to_tensor(val_wrapped)
                if imgs_val.numel() > 0:
                    xs_val.append(imgs_val)
                    ys_val.append(torch.full((len(imgs_val),), class_id, dtype=torch.long))

            if xs_val:
                X_val = torch.cat(xs_val)
                y_val = torch.cat(ys_val)
                logger.info(f"[BASELINE] Client {client_name}: using explicit validation set ({len(X_val)})")
            else:
                X_val, y_val = torch.empty(0), torch.empty(0)

            # Train data remains dynamic
            client_train_data[client_name] = full_client_dataset
            client_val_data[client_name] = (X_val, y_val)

        # Case B: Implicit Hold-out Split
        else:
            total_len = len(full_client_dataset)
            if total_len > 1:
                val_len = int(total_len * 0.2)
                train_len = total_len - val_len

                # Random split (labels not easily accessible for stratification without loading)
                train_ds, val_ds = torch.utils.data.random_split(
                    full_client_dataset,
                    [train_len, val_len],
                    generator=torch.Generator().manual_seed(int(getattr(args, "seed", 42)))
                )

                # APPLIED CORRECTION: Wrap implicit validation split with deterministic transform
                val_wrapped = TransformSubset(val_ds, eval_transform)

                # Freeze validation set
                X_val = subset_to_tensor(val_wrapped)

                # Extract labels for validation tensor
                # We reuse the wrapped loader to ensure consistency
                val_loader_temp = DataLoader(val_wrapped, batch_size=1024, shuffle=False)
                val_xs, val_ys = [], []
                for x, y in val_loader_temp:
                    # x is already transformed by TransformSubset/subset_to_tensor call implies similar
                    # But here we are iterating directly.
                    # Note: subset_to_tensor re-iterates. We optimize by capturing here if needed.
                    # But subset_to_tensor is efficient. We just need Ys.
                    val_ys.append(y)

                if X_val.numel() > 0:
                    y_val = torch.cat(val_ys)
                else:
                    X_val, y_val = torch.empty(0), torch.empty(0)

                client_train_data[client_name] = train_ds
                client_val_data[client_name] = (X_val, y_val)

                logger.info(
                    f"[BASELINE] Client {client_name}: created dynamic train ({train_len}) / static val ({val_len}) split")
            else:
                # Not enough data
                client_train_data[client_name] = full_client_dataset
                client_val_data[client_name] = (torch.empty(0), torch.empty(0))
                logger.warning(f"[BASELINE] Client {client_name}: insufficient data for split.")

    # ---------------------------------------------------------------
    # Global Validation Loader Assembly
    # ---------------------------------------------------------------
    val_imgs_list: List[torch.Tensor] = []
    val_lbls_list: List[torch.Tensor] = []

    for client_name, (Xv, yv) in client_val_data.items():
        if Xv.numel() == 0:
            continue
        val_imgs_list.append(Xv)
        val_lbls_list.append(yv)

    if val_imgs_list:
        global_val_imgs = torch.cat(val_imgs_list)
        global_val_lbls = torch.cat(val_lbls_list)
        val_loader_global = DataLoader(
            TensorDataset(global_val_imgs, global_val_lbls),
            batch_size=args.batch_size,
            shuffle=False,
        )
        logger.info(
            f"[BASELINE] Global validation set size: {len(global_val_imgs)} samples"
        )
    else:
        logger.warning(
            "[BASELINE] No validation data available across clients, "
            "falling back to test set as validation."
        )
        val_loader_global = test_loader

    # ---------------------------------------------------------------
    # Main Federated Learning Loop
    # ---------------------------------------------------------------
    best_val_accuracy = 0.0
    rounds_without_improvement = 0
    max_rounds = getattr(args, "baseline_max_rounds", 50)
    patience = getattr(args, "baseline_patience", 10)

    round_accuracies: List[float] = []
    round_losses: List[float] = []
    best_round = 0

    for round_num in range(max_rounds):
        logger.info(f"[BASELINE] Round {round_num + 1}/{max_rounds}")

        selected_clients = np.random.choice(
            client_names, size=clients_per_round, replace=False
        ).tolist()
        logger.info(
            f"[BASELINE] Selected clients for round {round_num + 1}: {selected_clients}"
        )

        client_updates: Dict[str, Tuple[dict, int]] = {}
        round_losses_per_client: Dict[str, list] = {}

        # --- Local Training Phase ---
        for client_name in selected_clients:
            if tracker is not None:
                tracker.start_phase(f"client_{client_name}_round_{round_num}")

            try:
                # Training Data: Dynamic Dataset/Subset
                train_ds = client_train_data[client_name]

                # Validation Data: Static Tensors
                X_val, y_val = client_val_data[client_name]

                if len(train_ds) == 0:
                    logger.warning(f"[BASELINE] No training data for client {client_name}")
                    continue

                # APPLIED CORRECTION: Wrap training set with Augmentation pipeline
                # This ensures every epoch yields different random crops/flips
                augmented_train_ds = TransformSubset(train_ds, train_transform)

                train_loader = DataLoader(
                    augmented_train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=_default_num_workers(),
                    pin_memory=True,
                    persistent_workers=True
                )

                if X_val.numel() > 0:
                    val_loader = DataLoader(
                        TensorDataset(X_val, y_val),
                        batch_size=args.batch_size,
                        shuffle=False,
                    )
                else:
                    val_loader = DataLoader(
                        TensorDataset(torch.empty(0), torch.empty(0)),
                        batch_size=args.batch_size,
                        shuffle=False,
                    )

                client_state_dict, num_samples = baseline.train_client(
                    client_name, train_loader, val_loader, round_num
                )
                client_updates[client_name] = (client_state_dict, num_samples)

                loss_key = f"{client_name}_round{round_num}"
                if loss_key in baseline.history.get("train_loss", {}):
                    round_losses_per_client[client_name] = baseline.history[
                        "train_loss"
                    ][loss_key]

                logger.info(
                    f"[BASELINE] Client {client_name} trained with {num_samples} samples"
                )

            except Exception as e:
                logger.error(f"[BASELINE] Error training client {client_name}: {e}")
                continue
            finally:
                if tracker is not None:
                    tracker.end_phase(f"client_{client_name}_round_{round_num}")

        # --- Aggregation Phase ---
        if tracker is not None:
            tracker.start_phase(f"aggregation_round_{round_num}")

        if client_updates:
            baseline.aggregate(round_num, client_updates)
            logger.info(
                f"[BASELINE] Successfully aggregated {len(client_updates)} client updates"
            )
        else:
            logger.warning("[BASELINE] No client updates to aggregate")

        if tracker is not None:
            tracker.end_phase(f"aggregation_round_{round_num}")

        # --- Evaluation Phase ---
        if tracker is not None:
            tracker.start_phase(f"evaluation_round_{round_num}")

        val_accuracy = baseline.evaluate(val_loader_global)
        round_accuracies.append(val_accuracy)

        if round_losses_per_client:
            all_client_losses: List[float] = []
            for client_losses in round_losses_per_client.values():
                if client_losses:
                    all_client_losses.append(client_losses[-1])
            if all_client_losses:
                avg_round_loss = float(
                    sum(all_client_losses) / max(len(all_client_losses), 1)
                )
                round_losses.append(avg_round_loss)

        if tracker is not None:
            tracker.end_phase(f"evaluation_round_{round_num}")

        logger.info(
            f"[BASELINE] Round {round_num + 1} - Validation accuracy: {val_accuracy:.4f}"
        )

        baseline.history.setdefault("val_acc", {})
        baseline.history["val_acc"][f"round_{round_num}"] = [val_accuracy]

        # --- Early Stopping & Model Checkpointing ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_round = round_num + 1
            rounds_without_improvement = 0

            if baseline.global_model is not None:
                model_path = P.root / "models" / "classifiers" / "central_best.pt"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(baseline.global_model.state_dict(), model_path)
                logger.info(
                    f"[BASELINE] New best model saved with val accuracy: "
                    f"{best_val_accuracy:.4f}"
                )
        else:
            rounds_without_improvement += 1

        if rounds_without_improvement >= patience:
            logger.info(f"[BASELINE] Early stopping at round {round_num + 1}")
            break

    # ---------------------------------------------------------------
    # Final Evaluation on Test Set
    # ---------------------------------------------------------------
    final_test_accuracy = best_val_accuracy
    model_path = P.root / "models" / "classifiers" / "central_best.pt"

    # Initialize empty arrays in case loading fails
    final_y_true = np.array([])
    final_y_pred = np.array([])

    if model_path.exists() and baseline.global_model is not None:
        try:
            baseline.global_model.load_state_dict(torch.load(model_path))
            final_test_accuracy = baseline.evaluate(test_loader)

            # CAPTURE PREDICTIONS for confusion matrix
            final_y_true, final_y_pred = evaluate_single_classifier(
                baseline.global_model, test_loader, device
            )

            logger.info(
                f"[BASELINE] Final best model test accuracy: {final_test_accuracy:.4f} "
                f"(best val from round {best_round})"
            )
        except Exception as e:
            logger.warning(f"[BASELINE] Could not load best model or evaluate predictions: {e}")

    # ---------------------------------------------------------------
    # Metrics Serialization
    # ---------------------------------------------------------------

    # Calculate detailed class distribution per client
    client_class_dist: Dict[str, Dict[str, int]] = {}
    for c_name, c_subsets in train_subsets_dict.items():
        # Convert class key to string for JSON compatibility
        dist = {str(k): len(v) for k, v in c_subsets.items()}
        client_class_dist[c_name] = dist

    metrics = {
        "model": type(baseline).__name__,
        "dataset": args.dataset,
        "partition": getattr(args, "partition", None),
        "alpha": getattr(args, "alpha", None),
        "aggregation": "flower_fedavg",
        "best_accuracy": float(best_val_accuracy),
        "final_accuracy": float(final_test_accuracy),
        "best_val_accuracy": float(best_val_accuracy),
        "final_test_accuracy": float(final_test_accuracy),
        "best_round": best_round,
        "total_rounds": round_num + 1,
        "early_stopped": rounds_without_improvement >= patience,
        "round_accuracies": [float(acc) for acc in round_accuracies],
        "round_losses": [float(loss) for loss in round_losses] if round_losses else [],
        "clients_per_round": clients_per_round,
        "total_clients": total_clients,
        "client_sample_counts": all_client_samples,
        "client_class_distribution": client_class_dist,
    }

    try:
        metrics_path = P.root / "metrics" / "baseline_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.warning(f"[BASELINE] Could not save metrics: {e}")

    logger.info(
        f"[BASELINE] Experiment completed: "
        f"Best val accuracy {best_val_accuracy:.4f}, "
        f"Final test accuracy {final_test_accuracy:.4f}"
    )

    return final_test_accuracy, baseline.history, metrics, final_y_true, final_y_pred