"""
🚀 Experiment Environment and Data Preparation Module
---------------------------------------------------

This module orchestrates the initialization of the experimental environment and 
the pre-processing of datasets for machine learning workflows, specifically tailored 
for Federated Learning or distribution shift studies.

🧠 Purpose:
    It serves as the foundational layer for experiment reproducibility and data management. 
    It creates structured directory trees for artifacts, snapshots the codebase, and 
    partitions datasets according to various heterogeneity strategies (e.g., Silos, Skew, Dirichlet).

🔧 Core Functionalities:
    • Construct hierarchical filesystem paths based on experiment hyperparameters.
    • Snapshot the execution script to ensure code reproducibility.
    • Configure image transformations (resizing, grayscale conversion).
    • Load and partition datasets (IID vs. non-IID) for simulated distributed learning.
    • Export data distribution statistics for auditing.

🎯 Intended Use:
    • Initialization phase of training pipelines (`main.py` or `runner.py`).
    • Academic benchmarking of Federated Learning algorithms under non-IID conditions.

📁 Dependencies:
    • torch, torchvision (Data loading and tensors)
    • numpy, sklearn (Data splitting and array manipulation)
    • imports.data_management (Dataset retrieval and metadata)
    • imports.data_partitions (Partitioning logic)

📝 Notes:
    The module handles complex return signatures containing dataset objects, 
    transformation pipelines, and tensors to minimize overhead during the training loop.

Author: Andrea Moleri
File Location: src/jobs/experiment_setup.py
Last Modified: 21/11/2025
"""

from __future__ import annotations
import logging
import shutil
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from imports.data_management import (
    DATASET_META, get_dataset, 
    prime_dataset_meta_for_transform, 
    set_dataset_options
)
from imports.data_partitions import (
    parse_client_config, 
    create_skew_partition, 
    create_dirichlet_partition
)
from imports.data_augmentation import build_transform
from jobs.baseline_runner import subset_to_tensor, dataset_to_tensor

logger = logging.getLogger(__name__)

def _utc_now_parts():
    """
    Generates synchronized ISO 8601 timestamp parts for file naming and logging.

    Returns:
        Tuple[str, str, str]:
            - Date string (YYYY-MM-DD).
            - Time string formatted for filesystem paths (T%H-%M-%SZ).
            - Full ISO 8601 timestamp string.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    date_str = now.date().isoformat()
    time_iso = now.isoformat().replace("+00:00", "Z")
    # Format time for safe use in directory names (avoid colons)
    time_for_path = "T" + now.strftime("%H-%M-%SZ")
    return date_str, time_for_path, time_iso

@dataclass
class PathRegistry:
    """
    A data class responsible for managing and creating the directory structure
    for experiment artifacts.

    Attributes:
        root (Path): The base directory for the current experiment run.
    """
    root: Path

    def ensure(self):
        """
        Creates the standard subdirectory hierarchy if it does not exist.

        The following directories are created:
        - environment: For code snapshots.
        - metrics: For logging numerical results.
        - models/*: For saving generator and classifier state dictionaries.
        - artifacts/*: For generated samples, pairwise plots, and t-SNE projections.
        - datasets/*: For caching real or synthetic data.
        - costs/distributions: For analysis of computational cost and class balance.

        Returns:
            PathRegistry: The instance itself (fluent interface).
        """
        for d in (
                "environment", "metrics", "models/generators", "models/classifiers",
                "artifacts/samples", "artifacts/pairwise", "artifacts/tsne",
                "datasets/real", "datasets/synthetic",
                "costs", "distributions"
        ):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        return self

def setup_experiment_env(args, run_id: int | None) -> Tuple[PathRegistry, str]:
    """
    Configures the execution environment, creating necessary directories and 
    validating dataset metadata.

    This function constructs a deterministic path for the experiment based on 
    arguments such as model type, dataset, seed, and partition strategy. It also 
    performs a code snapshot for reproducibility.

    Args:
        args (Namespace): Parsed command-line arguments containing experiment configuration.
        run_id (int | None): An optional identifier for the specific run (e.g., for 
                             batch execution). If None, the timestamp is used.

    Returns:
        Tuple[PathRegistry, str]:
            - An initialized PathRegistry object pointing to the run directory.
            - The ISO 8601 timestamp string of the experiment start time.

    Raises:
        ValueError: If the specified dataset is not defined in the `DATASET_META` dictionary.
    """
    # Prepare metadata
    base_key = args.dataset.split("(", 1)[0].lower()
    try:
        # Initialize or update dataset metadata (e.g., paths, properties)
        prime_dataset_meta_for_transform(args.dataset, args.data_dir)
    except Exception as e:
        logger.warning(f"[DATA] prime_dataset_meta_for_transform failed: {e}")

    if base_key not in DATASET_META:
        raise ValueError(f"Dataset '{args.dataset}' not present in DATASET_META.")

    # Handle input size override
    # If the user explicitly requests a different input size, update metadata globally
    if getattr(args, "input_size", 0):
        new_sz = int(args.input_size)
        # Update relevant keys including NICO variations
        for k in (base_key, "nico++", "nicopp", "nico"):
            if k in DATASET_META:
                DATASET_META[k]["input_size"] = new_sz
        logger.info(f"[DATA] Overriding input_size -> {new_sz} for dataset '{args.dataset}'")

    # Paths construction
    date_str, time_for_path, time_iso = _utc_now_parts()
    run_leaf = f"run{run_id}" if run_id is not None else time_for_path
    
    partition_mode = getattr(args, "partition", "silos")
    aggregation_mode = getattr(args, "aggregation", "simple")
    
    # Determine folder naming based on partition strategy
    aggregation_display = "standard" if partition_mode == "silos" else aggregation_mode
    model_name = args.model.replace("baseline:", "") if args.model.startswith("baseline:") else args.model

    # Construct the hierarchical path:
    # out_dir / model / dataset / seed / inference_mode / partition_config / date / run_id
    run_root = (
            Path(args.out_dir)
            / model_name
            / args.dataset
            / f"seed-{args.seed}"
            / f"{args.infer_mode.lower()}-L{args.latent_dim}"
            / (f"{partition_mode}-{aggregation_display}" if partition_mode in ["skew", "dirichlet"] else partition_mode)
            / date_str
            / run_leaf
    )
    P = PathRegistry(run_root).ensure()
    
    # Snapshot code
    # Copies the current script file to the environment directory for reproducibility
    try:
        src_self = Path(__file__)
        shutil.copy2(src_self, P.root / "environment" / src_self.name)
    except Exception:
        # Fail silently if __file__ is not accessible or copy fails (e.g., interactive mode)
        pass
        
    return P, time_iso

def prepare_data(args, device) -> Tuple:
    """
    Loads, transforms, and partitions the dataset according to the experimental configuration.

    This function handles:
    1. Construction of image transformation pipelines (resizing, grayscale).
    2. Loading of the base training and test datasets.
    3. Calculation of dataset statistics (number of classes, image shape).
    4. Partitioning of the training data into client subsets (Skew, Dirichlet, or Silos).
    5. Preparation of test sets tailored to the partitioning strategy.

    Args:
        args (Namespace): Parsed command-line arguments containing data configuration.
        device: The torch device (unused in this function but kept for signature compatibility).

    Returns:
        Tuple: A tuple containing twelve elements:
            1.  **base_train_set**: The complete training dataset object.
            2.  **test_set**: The complete test dataset object.
            3.  **tfm**: The composed torchvision transform.
            4.  **num_classes** (int): Total number of unique classes.
            5.  **img_shape** (Tuple[int, ...]): Shape of a single input image (C, H, W).
            6.  **chans** (int): Number of image channels.
            7.  **train_subsets_dict** (Dict): Map of client ID to dataset subset (for skew/dirichlet).
            8.  **train_subsets** (List): List of dataset subsets (for silos).
            9.  **present_classes** (List[int]): List of class indices present in the current partition.
            10. **reserved_test_ld** (DataLoader): DataLoader for the full test set.
            11. **reserved_test_imgs_list** (List[Tensor]): List of test tensors split by class or partition.
            12. **test_imgs_tensor** (Tensor): The full test set as a single tensor.
    """
    # Transforms
    base_key = args.dataset.split("(", 1)[0].lower()
    meta = DATASET_META[base_key]
    chans = meta["channels"]
    
    base_tfm = build_transform(args.dataset)
    
    # Apply explicit resize if input_size is provided
    if getattr(args, "input_size", 0):
        sz = int(args.input_size)
        tfm = T.Compose([T.Resize((sz, sz), antialias=True), base_tfm])
    else:
        tfm = base_tfm

    # Force grayscale conversion if requested and input is RGB
    if bool(args.grayscale) and chans == 3:
        tfm = T.Compose([T.Grayscale(num_output_channels=3), base_tfm])
    else:
        tfm = base_tfm

    # NICO++ filter
    # Filters the NICO dataset to only include specific classes if arguments are provided
    if args.dataset.lower() in ("nico++", "nicopp", "nico"):
        selected = [c.strip() for c in args.classes.split(",") if c.strip()] or None
        set_dataset_options("nico++", classes=selected)

    # Base Train Set
    base_train_set = get_dataset(args.dataset, args.data_dir, True, tfm)
    
    # Num Classes logic
    # Attempts to determine class count via attributes 'classes', 'targets', or exhaustive iteration
    if hasattr(base_train_set, "classes") and len(getattr(base_train_set, "classes")) > 0:
        num_classes = len(base_train_set.classes)
    elif hasattr(base_train_set, "targets"):
        num_classes = int(len(set(map(int, base_train_set.targets))))
    else:
        labels_tmp = [int(lbl) for _, lbl in base_train_set]
        num_classes = int(len(set(labels_tmp)))
    
    # Img shape
    sample_img, _ = base_train_set[0]
    img_shape = tuple(sample_img.shape)

    # Test Set
    test_set = get_dataset(args.dataset, args.data_dir, False, tfm)
    test_imgs_tensor = dataset_to_tensor(test_set)
    
    # Extract labels from the test set for processing
    labels_src = None
    for attr in ("targets", "labels"):
        if hasattr(test_set, attr):
            labels_src = getattr(test_set, attr)
            break
    if labels_src is None and hasattr(test_set, "imgs"):
        labels_src = [t for _, t in test_set.imgs]
    
    test_lbls_all = torch.as_tensor(labels_src, dtype=torch.long).reshape(-1)
    
    # Normalize EMNIST labels to be zero-indexed if necessary
    if args.dataset.startswith("emnist") and test_lbls_all.min().item() != 0:
        test_lbls_all = test_lbls_all - test_lbls_all.min()
        
    reserved_test_ld = DataLoader(TensorDataset(test_imgs_tensor, test_lbls_all), batch_size=args.batch_size)

    # Partition Logic
    partition_mode = getattr(args, "partition", "silos")
    train_subsets_dict = {} # For skew/dirichlet
    train_subsets = []      # For silos
    reserved_test_imgs_list = []
    reserved_test_lbls = []
    present_classes = []

    if partition_mode == "skew":
        # Create non-IID partitions based on client configuration
        client_config = parse_client_config(getattr(args, "client_config", ""))
        train_subsets_dict, _ = create_skew_partition(base_train_set, client_config, args.seed, num_classes)
        present_classes = list(range(num_classes))
        
        # Helper for test splitting (simplified)
        # Consolidate target extraction logic
        if hasattr(test_set, "targets"): targets_array = test_set.targets
        elif hasattr(test_set, "labels"): targets_array = test_set.labels
        else: targets_array = [lbl for _, lbl in test_set]
        
        targets_array = np.asarray(targets_array)
        if targets_array.ndim > 1: targets_array = targets_array[:, 0]
        
        # EMNIST label adjustment
        if args.dataset.startswith("emnist") and targets_array.min() != 0:
            targets_array = targets_array - targets_array.min()

        # Create a test tensor for each class individually
        for d in range(num_classes):
            idxs = np.flatnonzero(targets_array == d)
            if len(idxs) == 0:
                 reserved_test_imgs_list.append(torch.empty(0))
                 continue
            sub = Subset(test_set, idxs)
            reserved_test_imgs_list.append(subset_to_tensor(sub))

    elif partition_mode == "dirichlet":
        # Create partitions using Latent Dirichlet Allocation (LDA) sampling
        alpha = float(getattr(args, "alpha", 0.5))
        n_clients = int(getattr(args, "num_clients", 10))
        train_subsets_dict, _ = create_dirichlet_partition(base_train_set, n_clients, alpha, args.seed)
        present_classes = list(range(num_classes))
        
        # Same test split logic as skew (separating by class)
        if hasattr(test_set, "targets"): targets_array = test_set.targets
        elif hasattr(test_set, "labels"): targets_array = test_set.labels
        else: targets_array = [lbl for _, lbl in test_set]
        
        targets_array = np.asarray(targets_array)
        if targets_array.ndim > 1: targets_array = targets_array[:, 0]
        
        if args.dataset.startswith("emnist") and targets_array.min() != 0:
            targets_array = targets_array - targets_array.min()

        for d in range(num_classes):
            idxs = np.flatnonzero(targets_array == d)
            if len(idxs) == 0:
                 reserved_test_imgs_list.append(torch.empty(0))
                 continue
            sub = Subset(test_set, idxs)
            reserved_test_imgs_list.append(subset_to_tensor(sub))

    else: # silos
        # 'Silos' partition mode: typically for disjoint class splits
        if hasattr(base_train_set, "targets"): targets_array = base_train_set.targets
        elif hasattr(base_train_set, "labels"): targets_array = base_train_set.labels
        else: targets_array = [lbl for _, lbl in base_train_set]
        
        targets_array = np.asarray(targets_array)
        if targets_array.ndim > 1: targets_array = targets_array[:, 0]
        
        if args.dataset.startswith("emnist") and targets_array.min() != 0:
            targets_array = targets_array - targets_array.min()

        for d in range(num_classes):
            idxs = np.flatnonzero(targets_array == d)
            if len(idxs) == 0: continue
            present_classes.append(d)
            
            # Split the class-specific data into train (80%) and test (20%)
            train_idx, test_idx = train_test_split(idxs, test_size=0.20, random_state=args.seed, shuffle=True)
            train_subsets.append(Subset(base_train_set, train_idx))
            
            test_sub = Subset(base_train_set, test_idx)
            reserved_test_imgs_list.append(subset_to_tensor(test_sub))

    # Adjustment for Silos: if fewer classes are found than expected, update num_classes
    if len(present_classes) != num_classes and partition_mode == 'silos':
         num_classes = len(present_classes)

    return (base_train_set, test_set, tfm, num_classes, img_shape, chans, 
            train_subsets_dict, train_subsets, present_classes, 
            reserved_test_ld, reserved_test_imgs_list, test_imgs_tensor)

def _export_client_class_distribution(
        root: Path,
        client_sample_counts: Dict[str, Dict[int, int]],
        num_classes: int
) -> None:
    """
    Exports the distribution of classes per client to a CSV file.

    This is useful for visualizing the degree of data heterogeneity (skew)
    assigned to each client in the experiment.

    Args:
        root (Path): The root directory of the experiment.
        client_sample_counts (Dict[str, Dict[int, int]]): Mapping of client names 
                                                          to a dictionary of class counts.
        num_classes (int): The total number of classes in the dataset.

    Returns:
        None
    """
    distributions_dir = root / "distributions"
    distributions_dir.mkdir(parents=True, exist_ok=True)
    csv_file = distributions_dir / "client_class_distribution.csv"
    
    # Construct CSV header: client, class_0, class_1, ...
    header = ["client"] + [f"class_{i}" for i in range(num_classes)]
    client_names = sorted(client_sample_counts.keys())
    
    rows = []
    for client_name in client_names:
        row = [client_name]
        class_counts = client_sample_counts[client_name]
        for class_id in range(num_classes):
            # Fill count or 0 if class is missing for this client
            count = class_counts.get(class_id, 0)
            row.append(str(count))
        rows.append(row)
        
    with open(csv_file, 'w') as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
            
    logger.info(f"[EXPORT] Client class distribution exported to: {csv_file}")
