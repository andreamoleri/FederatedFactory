"""
💾 Data Partitioning Utilities
------------------------------

This module provides specialized functions for partitioning datasets into
client-specific subsets, designed primarily for Federated Learning (FL)
simulations. It supports both manual deterministic skewing and probabilistic
partitioning based on the Dirichlet distribution.

🧠 Purpose:
    To simulate Non-IID (Non-Independent and Identically Distributed) data
    environments common in distributed machine learning research.

    This version allocates 100% of the assigned data to the training subset
    and returns empty test subsets. This design choice forces the baseline
    runner to utilize all available local data for model training, while global
    evaluation is deferred to a canonical central test set.

🔧 Core Functionalities:
    • Parse complex configuration strings for granular data allocation
    • Generate deterministic partitions with specific class-per-client counts
    • Generate probabilistic partitions using the Dirichlet distribution to
      simulate varying degrees of data heterogeneity

🎯 Intended Use:
    • Federated Learning research simulations
    • Benchmarking aggregation algorithms under data skew
    • Reproducible dataset splitting for distributed systems

📁 Dependencies:
    • numpy
    • torch (for Subset)
    • logging

📝 Notes:
    The module assumes the input `base_dataset` is array-like or compatible
    with PyTorch dataset interfaces (specifically exposing `targets` or `labels`).

Author: Andrea Moleri
File Location: src/imports/data_partitions.py
Last Modified: 23/04/2025
"""

import logging
import numpy as np
from typing import Dict, Tuple, List
from torch.utils.data import Subset

# Initialize module-level logger
logger = logging.getLogger(__name__)

def parse_client_config(config_str: str) -> Dict[str, Dict[int, int]]:
    """
    Parses a formatted client configuration string into a structured dictionary.

    This function interprets a specifically formatted string to determine how many
    samples of specific classes should be allocated to each client.

    Format: "client_name:class_id:count,class_id:count;..."

    Args:
        config_str (str): The configuration string defining data distribution.
                          Example: "c1:0:100,1:50;c2:0:50" means client 'c1' gets
                          100 samples of class 0 and 50 of class 1.

    Returns:
        Dict[str, Dict[int, int]]: A nested dictionary mapping client names to
                                   another dictionary of {class_id: sample_count}.
                                   Returns an empty dictionary if input is empty.

    Note:
        A count of -1 indicates that the remaining samples for that class should
        be auto-allocated evenly among clients requesting it.
    """
    if not config_str:
        return {}

    logger.info(f"[SKEW] Parsing client configuration: {config_str}")
    config = {}
    clients = config_str.split(';')

    for client in clients:
        # Skip empty segments resulting from trailing semicolons
        if not client.strip():
            continue

        # Split client definition into name and class configuration
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
                # Case: Explicit count provided (class_id:count)
                if len(cp_parts) == 2:
                    class_id = int(cp_parts[0].strip())
                    sample_count = int(cp_parts[1].strip())
                    class_config[class_id] = sample_count
                # Case: Auto-allocation requested (class_id implies -1)
                elif len(cp_parts) == 1:
                    class_id = int(cp_parts[0].strip())
                    class_config[class_id] = -1
                else:
                    continue
            except (ValueError, IndexError):
                # Gracefully skip malformed class definitions
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
    Creates a deterministic data partition based on specific class counts per client.

    This function extracts indices from the base dataset and assigns them to clients
    according to the provided configuration. It supports auto-allocation for counts
    marked as -1.

    Args:
        base_dataset: The source dataset containing data and labels/targets.
        client_config (Dict[str, Dict[int, int]]): The distribution map returned
                                                   by `parse_client_config`.
        seed (int): Random seed for reproducibility of index selection.
        num_classes (int): The total number of unique classes in the dataset.

    Returns:
        Tuple[Dict, Dict]:
            - train_subsets: Dictionary mapping client names to class-specific
              PyTorch Subsets containing 100% of the allocated data.
            - test_subsets: Dictionary mapping client names to empty Subsets.

    Note:
        This function strictly enforces a 100% training split. The test subsets
        are returned as empty objects to signal to the consuming framework that
        local testing should be skipped or handled globally.
    """
    logger.info(f"[SKEW] Creating skew partition with {len(client_config)} clients")

    # Extract targets from dataset using common attribute names (torchvision/custom)
    if hasattr(base_dataset, 'targets'):
        targets = base_dataset.targets
    elif hasattr(base_dataset, 'labels'):
        targets = base_dataset.labels
    else:
        # Fallback for datasets that yield (data, label) tuples
        targets = [label for _, label in base_dataset]

    targets = np.array(targets)

    # Map class indices: Pre-calculate the location of every sample per class
    class_indices = {}
    class_counts = {}
    for class_id in range(num_classes):
        indices = np.where(targets == class_id)[0]
        class_indices[class_id] = indices
        class_counts[class_id] = len(indices)

    # Handle auto-allocation (-1): Distribute available samples evenly among requesters
    auto_alloc_classes = {}
    for client_name, class_samples in client_config.items():
        for class_id, sample_count in class_samples.items():
            if sample_count == -1:
                if class_id not in auto_alloc_classes:
                    auto_alloc_classes[class_id] = []
                auto_alloc_classes[class_id].append(client_name)

    # Calculate fair share for auto-allocated classes based on remaining global count
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
            # Cap request at available data size to prevent IndexErrors
            if total_samples > len(available_indices):
                total_samples = len(available_indices)

            # Deterministic Sampling: Derive a unique seed per client/class pair
            # to ensure consistency regardless of client iteration order.
            client_seed = abs(hash(f"{seed}_{client_name}_{class_id}")) % (2 ** 32)
            rng = np.random.RandomState(client_seed)
            selected_indices = rng.choice(available_indices, size=total_samples, replace=False)

            # Update global pool: Remove selected indices so they aren't reused
            # using boolean indexing for efficiency.
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
    Generates a probabilistic Non-IID data partition using the Dirichlet distribution.

    This method simulates heterogeneous data distributions where the proportion
    of class labels varies across clients, controlled by the concentration
    parameter alpha.

    Args:
        base_dataset: The source dataset containing data and labels.
        num_clients (int): Total number of clients to partition data for.
        alpha (float): Concentration parameter for the Dirichlet distribution.
                       - alpha -> 0: High heterogeneity (one class dominates).
                       - alpha -> infinity: Uniform distribution (IID).
        seed (int): Random seed for reproducibility.
        min_require_size (int, optional): Minimum number of samples required per client.
                                          Defaults to 10.

    Returns:
        Tuple[Dict, Dict]:
            - train_subsets: Dictionary mapping 'client{i}' to class-specific Subsets.
            - test_subsets: Dictionary mapping 'client{i}' to empty Subsets.
    """
    logger.info(f"[PARTITION] Dirichlet partition (alpha={alpha})")

    # Standardize target extraction
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
    # Rejection sampling: Retry partitioning until every client meets the minimum size requirement
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]

        for k in range(num_classes):
            idx_k = np.where(y_train == k)[0]
            rng.shuffle(idx_k)

            try:
                # Sample proportions from Dirichlet distribution
                proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            except ValueError:
                # Fallback to uniform distribution if Dirichlet fails
                proportions = np.array([1.0/num_clients] * num_clients)

            # Handle NaN/Underflow: Assign all to a random winner if calculation fails
            if np.isnan(proportions).any():
                proportions = np.zeros(num_clients)
                winner = rng.randint(0, num_clients)
                proportions[winner] = 1.0

            # Convert floating point proportions into integer split indices.
            # cumsum() helps determine the cut points in the index array.
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
            # Filter indices belonging to the current class within the client's batch
            cls_mask = (client_labels == class_id)
            cls_indices = client_indices[cls_mask]

            if len(cls_indices) == 0:
                continue

            client_train_struct[class_id] = Subset(base_dataset, cls_indices)
            client_test_struct[class_id] = Subset(base_dataset, [])

        train_subsets[client_name] = client_train_struct
        test_subsets[client_name] = client_test_struct

    return train_subsets, test_subsets