"""
💾 Data Partitioning Utilities
------------------------------

This module provides specialized functions for partitioning datasets into
client-specific subsets, designed primarily for Federated Learning (FL)
simulations. It supports both manual deterministic skewing and probabilistic
partitioning based on the Dirichlet distribution.

🧠 Purpose:
    To simulate Non-IID (Non-Independent and Identically Distributed) data
    environments, which are characteristic of real-world federated networks.
    It allows researchers to control the degree of data heterogeneity across
    clients.

🔧 Core Functionalities:
    • Parse complex configuration strings for manual data allocation
    • Create deterministic skewed partitions (specific classes/counts per client)
    • Generate partitions based on the Dirichlet distribution ($Dir(\\alpha)$)
    • Split resulting partitions into local training and testing sets

🎯 Intended Use:
    • Academic experiments benchmarking FL algorithms under data skew
    • Reproducible dataset generation for distributed machine learning
    • Simulation of edge device data distributions

📁 Dependencies:
    • numpy
    • torch.utils.data (Subset)
    • logging

📝 Notes:
    The module assumes the base dataset provides access to labels via attributes
    such as `targets` or `labels`. It defaults to an 80/20 train/test split
    for the local subsets.

Author: Andrea Moleri
File Location: src/imports/data_partitions.py
Last Modified: 21/11/2025
"""

import logging
import numpy as np
from typing import Dict, Tuple, List
from torch.utils.data import Subset

# Initialize module-level logger
logger = logging.getLogger(__name__)

def parse_client_config(config_str: str) -> Dict[str, Dict[int, int]]:
    """
    Parses a client configuration string into a structured dictionary for data allocation.

    This function interprets a specific string grammar to determine how many samples
    of specific classes should be assigned to each client. It supports both explicit
    counts and automatic allocation markers.

    Grammar Format:
        "client_id:class_id:count,class_id:count;client_id:..."

    Examples:
        - Explicit: "client0:0:1000,1:500;client1:2:800"
          (Client0 gets 1000 of class 0, 500 of class 1)
        - Automatic: "client0:0,1;client1:2,3"
          (Client0 gets equal shares of available class 0 and 1)

    Args:
        config_str (str): The raw configuration string defining the partition topology.

    Returns:
        Dict[str, Dict[int, int]]: A nested dictionary mapping:
            client_name -> {class_id -> sample_count}.
            Note: A sample_count of -1 indicates automatic allocation.
    """
    if not config_str:
        return {}

    logger.info(f"[SKEW] Parsing client configuration: {config_str}")
    config = {}
    # Split the master string into individual client segments
    clients = config_str.split(';')

    for client in clients:
        if not client.strip():
            continue

        # Split client name from class configurations
        # The format is expected to be "client_name:config_data"
        parts = client.split(':', 1)  # Split only on first colon to preserve subsequent colons
        if len(parts) < 2:
            logger.warning(f"[SKEW] Invalid client configuration format: {client}")
            continue

        client_name = parts[0].strip()
        class_config_str = parts[1].strip()
        class_config = {}

        # Split class configurations within the client segment
        class_parts = class_config_str.split(',')
        for cp in class_parts:
            if not cp.strip():
                continue
            try:
                # Try to split class_id and sample_count
                # Expected forms: "class_id:count" or "class_id"
                cp_parts = cp.strip().split(':')
                if len(cp_parts) == 2:
                    # Format: class_id:sample_count (Explicit allocation)
                    class_id = int(cp_parts[0].strip())
                    sample_count = int(cp_parts[1].strip())
                    class_config[class_id] = sample_count
                elif len(cp_parts) == 1:
                    # Format: class_id only (Automatic allocation)
                    class_id = int(cp_parts[0].strip())
                    # Use -1 as a sentinel value to indicate automatic allocation logic later
                    class_config[class_id] = -1 
                else:
                    logger.warning(f"[SKEW] Invalid class configuration format: {cp}")
                    continue
            except (ValueError, IndexError) as e:
                logger.warning(f"[SKEW] Invalid class configuration '{cp}': {e}")
                continue

        if class_config:
            config[client_name] = class_config
            logger.info(f"[SKEW] Client {client_name} configured with classes: {class_config}")
        else:
            logger.warning(f"[SKEW] No valid class configurations found for client {client_name}")

    logger.info(f"[SKEW] Final parsed client config: {config}")
    return config


def create_skew_partition(
        base_dataset,
        client_config: Dict[str, Dict[int, int]],
        seed: int,
        num_classes: int
) -> Tuple[Dict[str, Dict[int, Subset]], Dict[str, Dict[int, Subset]]]:
    """
    Creates deterministic training and testing subsets based on a skew configuration.

    This function allocates data to clients according to the dictionary produced by
    `parse_client_config`. It handles the logic for extracting targets, resolving
    automatic allocations (sentinel -1), and performing deterministic sampling
    to ensure reproducibility.

    

    Args:
        base_dataset: The source dataset (must contain `targets` or `labels`).
        client_config (Dict[str, Dict[int, int]]): The mapping of clients to their
            required class counts.
        seed (int): Global random seed for reproducibility.
        num_classes (int): Total number of unique classes in the dataset.

    Returns:
        Tuple[Dict, Dict]: A tuple containing (train_subsets, test_subsets).
            Structure: {"client_name": {class_id: torch.utils.data.Subset, ...}}.
    """
    logger.info(f"[SKEW] Creating skew partition with {len(client_config)} clients and {num_classes} total classes")

    # Extract targets/labels from dataset using common attribute names
    if hasattr(base_dataset, 'targets'):
        targets = base_dataset.targets
    elif hasattr(base_dataset, 'labels'):
        targets = base_dataset.labels
    else:
        # Fallback: Iterative extraction if no direct attribute exists (computationally expensive)
        targets = [label for _, label in base_dataset]

    targets = np.array(targets)

    # Create class indices mapping and count available samples per class
    # This index map allows O(1) access to all sample indices for a given class
    class_indices = {}
    class_counts = {}
    for class_id in range(num_classes):
        indices = np.where(targets == class_id)[0]
        class_indices[class_id] = indices
        class_counts[class_id] = len(indices)
        logger.info(f"[SKEW] Class {class_id}: {len(indices)} available samples")

    # Calculate automatic allocation for classes with -1 sample count
    # Identifies which clients are competing for "all remaining" data of a class
    auto_alloc_classes = {}
    for client_name, class_samples in client_config.items():
        for class_id, sample_count in class_samples.items():
            if sample_count == -1:  # Automatic allocation sentinel
                if class_id not in auto_alloc_classes:
                    auto_alloc_classes[class_id] = []
                auto_alloc_classes[class_id].append(client_name)

    # For automatic allocation, distribute available samples equally among competing clients
    for class_id, clients in auto_alloc_classes.items():
        if class_id in class_counts:
            available_samples = class_counts[class_id]
            samples_per_client = available_samples // len(clients)
            for client_name in clients:
                client_config[client_name][class_id] = samples_per_client
            logger.info(
                f"[SKEW] Automatically allocated {samples_per_client} samples of class {class_id} to {len(clients)} clients")

    train_subsets = {}
    test_subsets = {}

    # For each client, sample the specified number of examples per class
    for client_name, class_samples in client_config.items():
        client_train_subsets = {}
        client_test_subsets = {}
        logger.info(f"[SKEW] Processing client {client_name} with classes: {class_samples}")

        for class_id, total_samples in class_samples.items():
            if class_id not in class_indices or len(class_indices[class_id]) == 0:
                logger.warning(f"[SKEW] Class {class_id} not found in dataset for client {client_name}")
                continue

            available_indices = class_indices[class_id]

            # Cap the requested samples at the maximum available
            if total_samples > len(available_indices):
                logger.warning(
                    f"[SKEW] Requested {total_samples} samples for class {class_id} but only {len(available_indices)} available. Using all available.")
                total_samples = len(available_indices)

            # Use deterministic sampling with client-specific seed
            # The string hashing ensures that the subset is consistent for a (client, class) pair
            # regardless of the order in which clients are processed.
            client_seed = abs(hash(f"{seed}_{client_name}_{class_id}")) % (2 ** 32)
            rng = np.random.RandomState(client_seed)

            # Sample without replacement to ensure unique data points
            selected_indices = rng.choice(available_indices, size=total_samples, replace=False)

            # Remove selected indices from the available global pool to prevent data leakage
            # or double-counting across clients (though this implementation implies
            # consumption of the pool, it depends on proper config to avoid overlapping demands).
            mask = np.isin(available_indices, selected_indices, invert=True)
            class_indices[class_id] = available_indices[mask]

            # Split 80/20 for train/test
            train_size = int(0.8 * len(selected_indices))
            train_indices = selected_indices[:train_size]
            test_indices = selected_indices[train_size:]

            client_train_subsets[class_id] = Subset(base_dataset, train_indices)
            client_test_subsets[class_id] = Subset(base_dataset, test_indices)

            logger.info(
                f"[SKEW] Client {client_name} class {class_id}: {train_size} train, {len(test_indices)} test samples")

        train_subsets[client_name] = client_train_subsets
        test_subsets[client_name] = client_test_subsets

    logger.info(f"[SKEW] Skew partition created successfully")
    return train_subsets, test_subsets


def create_dirichlet_partition(
        base_dataset,
        num_clients: int,
        alpha: float,
        seed: int,
        min_require_size: int = 10
) -> Tuple[Dict[str, Dict[int, Subset]], Dict[str, Dict[int, Subset]]]:
    """
    Robust Dirichlet partitioner that handles NaN underflows and float rounding errors.
    """
    logger.info(f"[PARTITION] Creating Dirichlet partition (alpha={alpha}) for {num_clients} clients")
    
    # 1. Get Targets
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
            
            # --- FIX STARTS HERE ---
            try:
                proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            except ValueError:
                # Fallback if alpha is invalid, though usually rng handles small alpha ok 
                # unless it is negative.
                proportions = np.array([1.0/num_clients] * num_clients)

            # 1. Handle NaN/Underflow (The specific crash fix)
            # If alpha is tiny, proportions might be NaN. 
            # In that case, assign the whole class to one random client (Winner Takes All).
            if np.isnan(proportions).any():
                logger.warning(f"[PARTITION] Dirichlet underflow (NaN) for class {k}. Using Winner-Takes-All.")
                proportions = np.zeros(num_clients)
                winner = rng.randint(0, num_clients)
                proportions[winner] = 1.0

            # 2. Fix Floating Point Rounding (Prevent sample loss)
            # Instead of simple casting (floor), we calculate cumulative indices and round
            # to the nearest integer to minimize drift.
            
            # Balance the split points
            split_points = (np.cumsum(proportions) * len(idx_k)).round().astype(int)[:-1]
            
            # Ensure split points are strictly increasing and within bounds
            # (numpy.split requires sorted indices)
            split_points = np.clip(split_points, 0, len(idx_k)).astype(int)
            
            # --- FIX ENDS HERE ---

            idx_batch_split = np.split(idx_k, split_points)
            
            for i in range(num_clients):
                idx_batch[i] += idx_batch_split[i].tolist()
                
        min_size = min([len(idx_j) for idx_j in idx_batch])

    # 4. Create Subsets (Train/Test Split)
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
                
            split_point = int(len(cls_indices) * 0.8)
            train_idx = cls_indices[:split_point]
            test_idx = cls_indices[split_point:]
            
            if len(train_idx) > 0:
                client_train_struct[class_id] = Subset(base_dataset, train_idx)
            if len(test_idx) > 0:
                client_test_struct[class_id] = Subset(base_dataset, test_idx)
                
        train_subsets[client_name] = client_train_struct
        test_subsets[client_name] = client_test_struct
        
        # Calculate totals for logging
        total_samples = sum(len(s) for s in client_train_struct.values()) + \
                        sum(len(s) for s in client_test_struct.values())
        logger.info(f"[PARTITION] {client_name}: {total_samples} samples across {len(client_train_struct)} classes")

    return train_subsets, test_subsets
