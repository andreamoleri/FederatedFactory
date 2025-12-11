"""
💾 Data Partitioning Utilities
------------------------------

This module provides specialized functions for partitioning datasets into
client-specific subsets, designed primarily for Federated Learning (FL)
simulations. It supports both manual deterministic skewing and probabilistic
partitioning based on the Dirichlet distribution.

🧠 Purpose:
    To simulate Non-IID (Non-Independent and Identically Distributed) data
    environments.

    [CORRECTION - UPPER BOUND]:
    This version allocates 100% of the assigned data to the training subset.
    It returns empty test subsets. This ensures that in the baseline runner,
    clients train on all available local data, while global evaluation happens
    on the canonical central test set.

🔧 Core Functionalities:
    • Parse complex configuration strings for manual data allocation
    • Create deterministic skewed partitions
    • Generate partitions based on the Dirichlet distribution
"""

import logging
import numpy as np
from typing import Dict, Tuple, List
from torch.utils.data import Subset

# Initialize module-level logger
logger = logging.getLogger(__name__)

def parse_client_config(config_str: str) -> Dict[str, Dict[int, int]]:
    """
    Parses a client configuration string into a structured dictionary.
    (Logic unchanged from original)
    """
    if not config_str:
        return {}

    logger.info(f"[SKEW] Parsing client configuration: {config_str}")
    config = {}
    clients = config_str.split(';')

    for client in clients:
        if not client.strip():
            continue

        parts = client.split(':', 1)
        if len(parts) < 2:
            continue

        client_name = parts[0].strip()
        class_config_str = parts[1].strip()
        class_config = {}

        class_parts = class_config_str.split(',')
        for cp in class_parts:
            if not cp.strip():
                continue
            try:
                cp_parts = cp.strip().split(':')
                if len(cp_parts) == 2:
                    class_id = int(cp_parts[0].strip())
                    sample_count = int(cp_parts[1].strip())
                    class_config[class_id] = sample_count
                elif len(cp_parts) == 1:
                    class_id = int(cp_parts[0].strip())
                    class_config[class_id] = -1
                else:
                    continue
            except (ValueError, IndexError):
                continue

        if class_config:
            config[client_name] = class_config

    return config


def create_skew_partition(
        base_dataset,
        client_config: Dict[str, Dict[int, int]],
        seed: int,
        num_classes: int
) -> Tuple[Dict[str, Dict[int, Subset]], Dict[str, Dict[int, Subset]]]:
    """
    Creates deterministic partitioning based on specific counts.

    NOTE: Assigns 100% of selected data to 'train_subsets'.
    'test_subsets' is returned empty to force the baseline runner
    to use all data for training.
    """
    logger.info(f"[SKEW] Creating skew partition with {len(client_config)} clients")

    # Extract targets
    if hasattr(base_dataset, 'targets'):
        targets = base_dataset.targets
    elif hasattr(base_dataset, 'labels'):
        targets = base_dataset.labels
    else:
        targets = [label for _, label in base_dataset]

    targets = np.array(targets)

    # Map class indices
    class_indices = {}
    class_counts = {}
    for class_id in range(num_classes):
        indices = np.where(targets == class_id)[0]
        class_indices[class_id] = indices
        class_counts[class_id] = len(indices)

    # Handle auto-allocation (-1)
    auto_alloc_classes = {}
    for client_name, class_samples in client_config.items():
        for class_id, sample_count in class_samples.items():
            if sample_count == -1:
                if class_id not in auto_alloc_classes:
                    auto_alloc_classes[class_id] = []
                auto_alloc_classes[class_id].append(client_name)

    for class_id, clients in auto_alloc_classes.items():
        if class_id in class_counts:
            available_samples = class_counts[class_id]
            samples_per_client = available_samples // len(clients)
            for client_name in clients:
                client_config[client_name][class_id] = samples_per_client

    train_subsets = {}
    test_subsets = {} # Will remain empty of indices, but structured keys exist

    for client_name, class_samples in client_config.items():
        client_train_subsets = {}
        client_test_subsets = {}

        for class_id, total_samples in class_samples.items():
            if class_id not in class_indices or len(class_indices[class_id]) == 0:
                continue

            available_indices = class_indices[class_id]
            if total_samples > len(available_indices):
                total_samples = len(available_indices)

            # Deterministic Sampling
            client_seed = abs(hash(f"{seed}_{client_name}_{class_id}")) % (2 ** 32)
            rng = np.random.RandomState(client_seed)
            selected_indices = rng.choice(available_indices, size=total_samples, replace=False)

            # Update global pool (remove selected)
            mask = np.isin(available_indices, selected_indices, invert=True)
            class_indices[class_id] = available_indices[mask]

            # --- FIX: NO SPLIT. USE 100% FOR TRAIN ---
            client_train_subsets[class_id] = Subset(base_dataset, selected_indices)
            # Empty subset for local test to signal "use everything for training"
            client_test_subsets[class_id] = Subset(base_dataset, [])

        train_subsets[client_name] = client_train_subsets
        test_subsets[client_name] = client_test_subsets

    return train_subsets, test_subsets


def create_dirichlet_partition(
        base_dataset,
        num_clients: int,
        alpha: float,
        seed: int,
        min_require_size: int = 10
) -> Tuple[Dict[str, Dict[int, Subset]], Dict[str, Dict[int, Subset]]]:
    """
    Robust Dirichlet partitioner.

    NOTE: Assigns 100% of data to 'train_subsets'.
    """
    logger.info(f"[PARTITION] Dirichlet partition (alpha={alpha})")

    if hasattr(base_dataset, 'targets'):
        y_train = np.array(base_dataset.targets)
    elif hasattr(base_dataset, 'labels'):
        y_train = np.array(base_dataset.labels)
    else:
        y_train = np.array([lbl for _, lbl in base_dataset])

    if y_train.ndim > 1:
        y_train = y_train.flatten()

    num_classes = len(np.unique(y_train))
    rng = np.random.RandomState(seed)

    min_size = 0
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]

        for k in range(num_classes):
            idx_k = np.where(y_train == k)[0]
            rng.shuffle(idx_k)

            try:
                proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            except ValueError:
                proportions = np.array([1.0/num_clients] * num_clients)

            # Handle NaN/Underflow
            if np.isnan(proportions).any():
                proportions = np.zeros(num_clients)
                winner = rng.randint(0, num_clients)
                proportions[winner] = 1.0

            # Fix Floating Point Rounding
            split_points = (np.cumsum(proportions) * len(idx_k)).round().astype(int)[:-1]
            split_points = np.clip(split_points, 0, len(idx_k)).astype(int)

            idx_batch_split = np.split(idx_k, split_points)

            for i in range(num_clients):
                idx_batch[i] += idx_batch_split[i].tolist()

        min_size = min([len(idx_j) for idx_j in idx_batch])

    train_subsets = {}
    test_subsets = {}

    for i in range(num_clients):
        client_name = f"client{i}"
        client_indices = np.array(idx_batch[i])
        rng.shuffle(client_indices)

        client_train_struct = {}
        client_test_struct = {}
        client_labels = y_train[client_indices]

        for class_id in range(num_classes):
            cls_mask = (client_labels == class_id)
            cls_indices = client_indices[cls_mask]

            if len(cls_indices) == 0:
                continue

            # --- FIX: NO SPLIT. USE 100% FOR TRAIN ---
            client_train_struct[class_id] = Subset(base_dataset, cls_indices)
            client_test_struct[class_id] = Subset(base_dataset, [])

        train_subsets[client_name] = client_train_struct
        test_subsets[client_name] = client_test_struct

    return train_subsets, test_subsets