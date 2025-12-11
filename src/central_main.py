#!/usr/bin/env python3

"""
▶️ Centralized Training Script
---------------------------

This module orchestrates the complete lifecycle of a centralised deep learning
training pipeline, encompassing data loading, model optimisation, and
artifact serialization.

🧠 Purpose:
    To provide a reproducible and robust reference implementation for image
    classification tasks, serving as a topological upper bound by utilizing
    the full pooled training data and the canonical test set.

🔧 Core Functionalities:
    • Dynamic dataset loading with automatic input channel and class detection
    • Utilization of full training set and canonical test set (Upper Bound)
    • Decoupled augmentation logic (Augmented Train vs Deterministic Test)
    • Deterministic seeding for reproducibility across worker processes
    • Cosine annealing learning rate scheduling for convergence stability

🎯 Intended Use:
    • Benchmarking deep learning architectures (e.g., SimpleCNN, ResNet)
    • Establishing Centralized Upper Bounds for Federated Learning experiments
    • Pedagogical demonstrations of PyTorch best practices

📁 Dependencies:
    • torch
    • numpy
    • models.cnn (local)
    • imports (local)

📝 Notes:
    This script assumes that the dataset is locally available or can be
    downloaded via the `data_management` interface.

Author: Andrea Moleri
File Location: src/central_main.py
Last Modified: 11/12/2025
"""

import argparse
import json
import logging
import os
import sys
import time
import random
from pathlib import Path
from typing import Optional, Callable, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

# --- Project Imports ---
from models.cnn import SimpleCNN
# NEURIPS FIX: Added DATASET_META to imports to enable resolution override
from imports.data_management import get_dataset, prime_dataset_meta_for_transform, DATASET_META
from imports.data_augmentation import build_transform
from utils import set_seed

# Initialize system-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [CENTRAL] %(message)s'
)
logger = logging.getLogger(__name__)


class TransformedSubset(Dataset):
    """
    A dataset wrapper designed to decouple data loading from data transformation.

    This class enables the application of distinct transformations (e.g., augmentation
    vs. deterministic normalization) to subsets of a dataset.
    """

    def __init__(self, subset: Subset, transform: Optional[Callable]):
        """
        Initialize the TransformedSubset.

        Parameters
        ----------
        subset : Subset
            The underlying PyTorch Subset containing indices of the original dataset.
        transform : Optional[Callable]
            The transformation function (pipeline) to apply to the data item
            upon retrieval.
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> Any:
        """
        Retrieve a sample from the subset and apply the assigned transform.

        Parameters
        ----------
        index : int
            The index of the item to retrieve relative to the subset.

        Returns
        -------
        Any
            The transformed data sample (tuple of input and target).
        """
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        """
        Return the total number of samples in the subset.

        Returns
        -------
        int
            Length of the subset.
        """
        return len(self.subset)


def seed_worker(worker_id: int):
    """
    Configuration function to ensure deterministic behavior in DataLoader workers.

    By default, PyTorch DataLoader workers generate random numbers using the
    system state. This function synchronizes the NumPy and Python random seeds
    within each worker process to the base PyTorch seed, ensuring reproducibility.

    Parameters
    ----------
    worker_id : int
        The unique identifier for the worker process (automatically assigned).
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_centralized(args: argparse.Namespace):
    """
    Execute the centralized training pipeline.

    This function orchestrates the end-to-end workflow:
    1. Environment setup and reproducibility seeding.
    2. Data loading (Full Train + Canonical Test).
    3. Model initialization and optimization configuration.
    4. Execution of the training loop with metric tracking.
    5. Final evaluation and artifact serialization.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration parameters such as
        dataset path, batch size, learning rate, and output directory.

    Raises
    ------
    RuntimeError
        If critical resources (e.g., GPU) are misconfigured or dataset paths are invalid.
    """
    # --- Phase 1: Environment Configuration ---
    # Enforce global seeding for the main process to ensure experiment reproducibility
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Establish the directory structure for experiment artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 2: Data Preparation Strategy ---
    logger.info(f"Loading raw dataset: {args.dataset}")

    # Initialize metadata registry required for subsequent transformation logic
    # This loads defaults (e.g., MedMNIST=28, DomainNet=224)
    prime_dataset_meta_for_transform(args.dataset, args.data_dir)

    # =========================================================================
    # NEURIPS FIX: RESOLUTION MISMATCH
    # We must explicitly update the DATASET_META registry with the CLI input size.
    # Otherwise, build_transform() will use defaults while SimpleCNN uses args.input_size.
    # =========================================================================

    # 1. Resolve the key used in the registry
    base_key = args.dataset.split("(", 1)[0].lower()
    if base_key.startswith("medmnist"): base_key = "medmnist"

    # 2. Find the metadata pointer
    meta_ptr = DATASET_META.get(base_key) or DATASET_META.get(args.dataset)

    # 3. Fallback search (matches build_transform logic)
    if meta_ptr is None:
        for k in DATASET_META:
            if k in args.dataset:
                meta_ptr = DATASET_META[k]
                break

    # 4. Apply Override
    if meta_ptr:
        if meta_ptr.get("input_size") != args.input_size:
            logger.info(
                f"⚠️ Overriding dataset default resolution ({meta_ptr.get('input_size')}) "
                f"with CLI argument: {args.input_size}"
            )
            meta_ptr["input_size"] = args.input_size
    else:
        logger.warning(f"Could not find metadata entry for {args.dataset}. Transforms may fail.")
    # =========================================================================

    # Step A: Load Raw Datasets (Train and Test)
    # We load both the full training pool (for the upper bound) and the
    # canonical test set (for valid comparison against FL results).
    logger.info("Loading full training pool and canonical test set...")

    train_dataset_raw = get_dataset(
        name=args.dataset,
        root=args.data_dir,
        train=True,
        transform=None
    )

    test_dataset_raw = get_dataset(
        name=args.dataset,
        root=args.data_dir,
        train=False,
        transform=None
    )

    # Step B: Dynamic Input Analysis
    # Analyze the training set to infer channels and class counts.
    try:
        sample_img, _ = train_dataset_raw[0]
        if isinstance(sample_img, torch.Tensor):
            in_ch = sample_img.shape[0]
        else:
            # Handle PIL Images or similar container objects
            in_ch = len(sample_img.getbands())
    except Exception as e:
        logger.warning(f"Could not auto-detect channels ({e}), defaulting to 3.")
        in_ch = 3

    # Target Extraction for Class Counting
    # We scan the training targets to dynamically determine the number of output neurons.
    if hasattr(train_dataset_raw, 'targets'):
        # Standard TorchVision convention (e.g., CIFAR, MNIST)
        targets = np.array(train_dataset_raw.targets)
    elif hasattr(train_dataset_raw, 'labels'):
        # Common convention in medical datasets (e.g., MedMNIST)
        targets = np.array(train_dataset_raw.labels)
    else:
        # Fallback: Iterative extraction
        logger.info("Extracting targets manually for class counting...")
        targets = np.array([y for _, y in train_dataset_raw])

    # Determine class cardinality dynamically
    num_classes = len(np.unique(targets))
    logger.info(f"Detected: {in_ch} Channels, {num_classes} Classes")

    # Step C: Prepare Subsets (Full Data)
    # Unlike federated splits, the centralized baseline uses 100% of the training data.
    # We wrap them in Subset objects covering the full range to maintain compatibility
    # with the TransformedSubset wrapper logic used downstream.
    train_idx = list(range(len(train_dataset_raw)))
    test_idx = list(range(len(test_dataset_raw)))

    train_subset_raw = Subset(train_dataset_raw, train_idx)
    test_subset_raw = Subset(test_dataset_raw, test_idx)

    # Step D: Pipeline Assembly
    logger.info("Building separate transforms for Train (Augmented) and Test (Deterministic)...")

    # Construct context-aware transformation pipelines
    # Train: Includes random augmentations (flips, noise) to improve generalization.
    train_transform = build_transform(args.dataset, train=True, robustness=True)
    # Test: Strictly deterministic (resize/crop + normalization) for valid evaluation.
    test_transform = build_transform(args.dataset, train=False, robustness=False)

    # Wrap raw subsets with the appropriate transform logic
    train_subset = TransformedSubset(train_subset_raw, train_transform)
    test_subset = TransformedSubset(test_subset_raw, test_transform)

    logger.info(
        f"Final Data Configuration: {len(train_subset)} Training (Full Pool) / {len(test_subset)} Test (Canonical)")

    # Initialize a Generator for the main process to govern DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Configure DataLoaders
    # `worker_init_fn` and `generator` are essential for reproducible multi-process loading.
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True
    )

    # --- Phase 3: Model Initialization ---
    model = SimpleCNN(in_ch=in_ch, num_classes=num_classes, input_resolution=args.input_size)
    model = model.to(device)

    # --- Phase 4: Optimization Configuration ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    # Apply Cosine Annealing to decay the learning rate smoothly over epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Phase 5: Training Execution ---
    logger.info(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets_batch in train_loader:
            inputs, targets_batch = inputs.to(device), targets_batch.to(device)

            # Standard optimization step
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()

            # Metric aggregation
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets_batch.size(0)
            correct += predicted.eq(targets_batch).sum().item()

        # Update learning rate schedule
        scheduler.step()

        # Calculate epoch-level statistics
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        # Log progress periodically
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{args.epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    training_time = time.time() - start_time
    logger.info(f"Training finished in {training_time:.2f}s")

    # --- Phase 6: Evaluation & Artifact Generation ---
    model.eval()

    # Containers for aggregate metrics
    all_targets = []
    all_preds = []
    all_probs = []

    # Execute inference on the test set without gradient tracking
    with torch.no_grad():
        for inputs, targets_batch in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # Move data to CPU for storage and analysis
            all_targets.append(targets_batch.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Flatten aggregated batches into single arrays
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    # Compute scalar accuracy
    final_acc = (y_pred == y_true).mean()
    logger.info(f"Final Test Accuracy (Canonical Test Set): {final_acc:.4f}")

    # --- Phase 7: Serialization ---
    # Save raw predictions and probabilities for detailed post-hoc analysis
    np.savez_compressed(
        out_dir / "predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs
    )
    logger.info(f"Saved predictions.npz to {out_dir}")

    # Persist the model weights
    torch.save(model.state_dict(), out_dir / "model.pt")

    # Serialize experiment metadata and high-level metrics
    metrics = {
        "dataset": args.dataset,
        "input_size": args.input_size,
        "accuracy": float(final_acc),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "training_time_sec": training_time,
        "train_size": len(train_idx),
        "test_size": len(test_idx)
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    # Define command-line interface arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=32, help="Resolution (e.g. 28, 32, 64, 224)")
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    run_centralized(args)