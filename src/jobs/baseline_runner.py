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
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset

# Models & Metrics
from models.baselines import (
    FederatedBaseline,
    FedAvgBaseline,
    FedProxBaseline,
    FedDFBaseline,
    FedDynBaseline,
)
from metrics.costs import ExperimentCostTracker
from imports.data_management import get_dataset
from imports.data_augmentation import build_transform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _default_num_workers() -> int:
    """
    Determine the optimal number of data loading workers based on CPU availability.

    Returns:
        int: The recommended number of worker processes. Defaults to 4 if 
        CPU count cannot be determined, or `cpu_count - 1` otherwise.
    """
    try:
        import os
        # Reserve one core for the main process to prevent deadlock or starvation
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

    This function bypasses standard Python loops for data accumulation where possible,
    utilizing a DataLoader to leverage multi-processing and memory pinning.

    Parameters:
        subset (torch.utils.data.Dataset): The source dataset or subset to convert.
        batch_size (int): The size of chunks to fetch during iteration. 
                          Larger values reduce overhead.
        num_workers (Optional[int]): Number of subprocesses for data loading. 
                                     If None, a default heuristic is used.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA 
                           pinned memory before returning them.
        limit (Optional[int]): A maximum number of samples to retrieve. 
                               Useful for debugging or quick previews.

    Returns:
        torch.Tensor: A single tensor containing the stacked data samples 
                      (e.g., shape [N, C, H, W]).
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

    Parameters:
        ds (torch.utils.data.Dataset): The dataset to convert.
        batch_size (int): Chunk size for data loading.
        num_workers (Optional[int]): Number of worker processes.
        pin_memory (bool): Whether to use pinned memory.

    Returns:
        torch.Tensor: The complete dataset represented as a tensor.
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

    Parameters:
        model (torch.nn.Module): The neural network model to evaluate.
        ld (DataLoader): The data loader containing the evaluation data.
        device (torch.device): The computation device (CPU or GPU).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Array of ground truth labels.
            - Array of predicted class indices.
    """
    model.to(device).eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in ld:
            y_true.append(y)
            y_pred.append(model(x.to(device)).argmax(1).cpu())

    model.cpu()
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def safe_train_val_split(
    X,
    y,
    val_size: float = 0.2,
    random_state: Optional[int] = None,
    client_name: Optional[str] = None,
    logger_: Optional[logging.Logger] = None,
):
    """
    Perform a train/validation split that gracefully handles rare classes.

    This wrapper around sklearn's train_test_split:
      - Uses stratified splitting when every class has at least 2 samples.
      - Falls back to a non-stratified split when some classes are too rare
        to support stratification (which would otherwise raise a ValueError).

    This prevents the error:
        "The least populated class in y has only 1 member, which is too few..."

    Args:
        X: Features (array-like, PyTorch Tensor, etc.).
        y: Labels (array-like or Tensor).
        val_size (float): Fraction of samples to allocate to the validation set.
        random_state (Optional[int]): Random seed.
        client_name (Optional[str]): Client identifier for logging context.
        logger_ (Optional[logging.Logger]): Logger to use (defaults to module logger).

    Returns:
        X_train, X_val, y_train, y_val
    """
    if logger_ is None:
        logger_ = logger

    # Convert labels to a NumPy array for statistics, but keep the original
    # object for the actual split to preserve types (e.g., PyTorch Tensor).
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)

    y_np = y_np.reshape(-1)
    unique, counts = np.unique(y_np, return_counts=True)

    # Default: no stratification
    stratify = None

    if len(unique) <= 1:
        logger_.warning(
            "[BASELINE] Client %s: not stratifying train/val split because only one "
            "class is present (classes=%s, counts=%s).",
            client_name,
            unique.tolist(),
            counts.tolist(),
        )
    else:
        min_count = counts.min()
        if min_count < 2:
            logger_.warning(
                "[BASELINE] Client %s: not stratifying train/val split because some "
                "classes have < 2 samples (classes=%s, counts=%s). "
                "Falling back to random (non-stratified) split.",
                client_name,
                unique.tolist(),
                counts.tolist(),
            )
        else:
            stratify = y
            logger_.info(
                "[BASELINE] Client %s: using stratified train/val split "
                "(classes=%s, counts=%s).",
                client_name,
                unique.tolist(),
                counts.tolist(),
            )

    return train_test_split(
        X,
        y,
        test_size=val_size,
        stratify=stratify,
        random_state=random_state,
    )


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
) -> Tuple[float, Dict, Dict, np.ndarray, np.ndarray]:
    """
    Execute the complete federated learning baseline experiment lifecycle.

    This function manages the iterative process of client selection, local training,
    server aggregation, and global evaluation. It handles data splitting, 
    model checkpointing, and metric logging.

    Parameters:
        baseline (FederatedBaseline): The specific federated algorithm instance 
                                      (e.g., FedAvg, FedProx).
        train_subsets_dict (Dict): Dictionary mapping client IDs to their training 
                                   subsets (partitioned by class).
        test_subsets_dict (Dict): Dictionary mapping client IDs to their testing 
                                  subsets.
        base_train_set: The original full training dataset object.
        args: Configuration namespace containing experiment hyperparameters.
        device: The torch device for computation.
        P: A PathRegistry object managing file system paths.
        tracker (Optional[ExperimentCostTracker]): Utility to track computational 
                                                   and communication costs.

    Returns:
        Tuple[float, Dict, Dict, np.ndarray, np.ndarray]:
            - The final accuracy on the test set.
            - The training history (losses, accuracies per round).
            - A dictionary of comprehensive experiment metrics.
            - y_true: Ground truth labels from the final evaluation (best model).
            - y_pred: Predicted labels from the final evaluation (best model).
    """
    logger.info(f"[BASELINE] Starting {type(baseline).__name__} federated learning")

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
    train_transform = getattr(base_train_set, "transform", None)
    if train_transform is None:
        logger.warning(
            "[BASELINE] base_train_set has no .transform attribute; "
            "falling back to build_transform(%s) for test set.",
            args.dataset,
        )

    try:
        test_set = get_dataset(
            args.dataset,
            args.data_dir,
            False,  # is_train=False
            train_transform or build_transform(args.dataset),
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
            f"[BASELINE] Created test loader with {len(test_imgs)} samples using "
            f"transform={train_transform if train_transform is not None else 'build_transform'}"
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
    client_train_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    client_val_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for client_name in client_names:
        # --- Process Training Data ---
        xs_train: List[torch.Tensor] = []
        ys_train: List[torch.Tensor] = []

        for class_id, train_subset in train_subsets_dict[client_name].items():
            imgs = subset_to_tensor(train_subset)
            if imgs.numel() == 0:
                continue
            xs_train.append(imgs)
            ys_train.append(torch.full((len(imgs),), class_id, dtype=torch.long))

        if xs_train:
            X_all = torch.cat(xs_train)
            y_all = torch.cat(ys_train)
        else:
            X_all = torch.empty(0)
            y_all = torch.empty(0, dtype=torch.long)

        # --- Process Validation Data (explicit subsets, if available) ---
        xs_val: List[torch.Tensor] = []
        ys_val: List[torch.Tensor] = []

        if client_name in test_subsets_dict:
            for class_id, val_subset in test_subsets_dict[client_name].items():
                imgs_val = subset_to_tensor(val_subset)
                if imgs_val.numel() == 0:
                    continue
                xs_val.append(imgs_val)
                ys_val.append(torch.full((len(imgs_val),), class_id, dtype=torch.long))

        if xs_val:
            # Case 1: Explicit validation set exists
            X_val = torch.cat(xs_val)
            y_val = torch.cat(ys_val)
            X_train = X_all
            y_train = y_all
            logger.info(
                f"[BASELINE] Client {client_name}: using explicit validation set "
                f"of size {len(X_val)}"
            )
        else:
            # Case 2: Create a hold-out split from the training data
            if len(X_all) > 1:
                X_train, X_val, y_train, y_val = safe_train_val_split(
                    X_all,
                    y_all,
                    val_size=0.2,
                    random_state=getattr(args, "seed", None),
                    client_name=client_name,
                    logger_=logger,
                )
                logger.info(
                    f"[BASELINE] Client {client_name}: created hold-out "
                    f"validation split (train={len(X_train)}, val={len(X_val)})"
                )
            else:
                # Case 3: Insufficient data for splitting
                X_train, y_train = X_all, y_all
                X_val, y_val = X_all, y_all
                logger.warning(
                    f"[BASELINE] Client {client_name}: not enough data to split, "
                    f"using same set for train and validation (size={len(X_all)})"
                )

        client_train_data[client_name] = (X_train, y_train)
        client_val_data[client_name] = (X_val, y_val)

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
                X_train, y_train = client_train_data[client_name]
                X_val, y_val = client_val_data[client_name]

                if X_train.numel() == 0:
                    logger.warning(
                        f"[BASELINE] No training data for client {client_name}"
                    )
                    continue

                train_loader = DataLoader(
                    TensorDataset(X_train, y_train),
                    batch_size=args.batch_size,
                    shuffle=True,
                )

                if X_val.numel() > 0:
                    val_loader = DataLoader(
                        TensorDataset(X_val, y_val),
                        batch_size=args.batch_size,
                        shuffle=False,
                    )
                else:
                    val_loader = DataLoader(
                        TensorDataset(
                            torch.empty(0, *X_train.shape[1:]),
                            torch.empty(0, dtype=torch.long),
                        ),
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
