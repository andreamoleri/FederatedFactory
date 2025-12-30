"""
🚀 Experiment Environment and Data Preparation Module
---------------------------------------------------

This module orchestrates the initialization of the experimental environment and 
the pre-processing of datasets for machine learning workflows.

🧠 Purpose:
    Designed to establish a reproducible research environment by managing
    directory structures, standardising timestamps, and preparing partitioned
    datasets for federated or centralized learning experiments.

🔧 Core Functionalities:
    • Construct hierarchical directory paths for artifacts and models
    • Archive the execution context for reproducibility
    • Load, normalize, and transform image datasets (e.g., NICO++, EMNIST)
    • partition data according to specific distributions (Silos, Dirichlet, Skew)
    • Generate PyTorch DataLoaders and Tensors for training and evaluation

🎯 Intended Use:
    • Research pipelines requiring rigorous experiment tracking
    • Federated learning simulations with non-IID data distributions
    • Benchmarking machine learning models on diverse datasets

📁 Dependencies:
    • torch
    • torchvision
    • numpy
    • sklearn (compatibility utilities)
    • internal modules (imports.*)

📝 Notes:
    This module assumes the existence of a global DATASET_META dictionary
    defining dataset-specific properties (channels, classes, etc.).

Author: Andrea Moleri
File Location: src/jobs/experiment_setup.py
Last Modified: 23/04/2025
"""

from __future__ import annotations
import logging
import shutil
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as T
# Retention of train_test_split is required for legacy compatibility with specific external callers.
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

class TransformSubset(torch.utils.data.Dataset):
    """
    A custom dataset wrapper that applies a transformation lazily upon item retrieval.

    This class is essential for scenarios where the underlying subset (train or test)
    requires dynamic augmentation or normalization that differs from the base dataset's
    original configuration, ensuring deterministic evaluation when required.
    """

    def __init__(self, subset: Subset, transform: T.Compose):
        """
        Initialize the TransformSubset.

        Args:
            subset (Subset): The subset of the original dataset.
            transform (T.Compose): The torchvision transform pipeline to apply.
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieve a sample and its label, applying the transform to the data.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: The transformed image tensor and its corresponding label.
        """
        x, y = self.subset[index]
        return self.transform(x), y

    def __len__(self) -> int:
        """
        Return the total number of samples in the subset.

        Returns:
            int: Length of the subset.
        """
        return len(self.subset)

def _utc_now_parts() -> Tuple[str, str, str]:
    """
    Generate synchronized UTC timestamp components for file naming and logging.

    Returns:
        Tuple[str, str, str]:
            - date_str: ISO formatted date (YYYY-MM-DD).
            - time_for_path: Filesystem-safe time string (T%H-%M-%SZ).
            - time_iso: Full ISO 8601 timestamp with Z designator.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    date_str = now.date().isoformat()
    time_iso = now.isoformat().replace("+00:00", "Z")
    time_for_path = "T" + now.strftime("%H-%M-%SZ")
    return date_str, time_for_path, time_iso

@dataclass
class PathRegistry:
    """
    A registry for managing and enforcing the existence of experiment directory structures.

    Attributes:
        root (Path): The base root directory for the current experiment run.
    """
    root: Path

    def ensure(self) -> PathRegistry:
        """
        Create the standard directory hierarchy required for experiment artifacts.

        This method ensures that directories for metrics, models, distributions,
        and generated artifacts exist, creating parent directories as necessary.

        Returns:
            PathRegistry: The instance itself, allowing for method chaining.
        """
        for d in (
                "environment", "metrics", "models/generators", "models/classifiers",
                "artifacts/samples", "artifacts/pairwise", "artifacts/tsne",
                "datasets/real", "datasets/synthetic",
                "costs", "distributions", "checkpoints"
        ):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        return self

def setup_experiment_env(args: Any, run_id: int | None) -> Tuple[PathRegistry, str]:
    """
    Initialize the experiment environment, including directory creation and metadata preparation.

    This function constructs the output path based on model parameters, dataset details,
    and random seeds. It also archives the current script to ensure reproducibility.

    Args:
        args (Any): Parsed command-line arguments containing configuration parameters
                    (dataset, model, seed, partition, etc.).
        run_id (int | None): An explicit identifier for the run. If None, a timestamp is used.

    Returns:
        Tuple[PathRegistry, str]:
            - An initialized PathRegistry object pointing to the experiment root.
            - The ISO 8601 string representing the experiment start time.

    Raises:
        ValueError: If the specified dataset is not defined in the global DATASET_META.
    """
    # Isolate the base dataset key (e.g., remove parameters from 'dataset(param)')
    # FIXED: Split on both '(' and ':' to handle 'medmnist:retinamnist'
    base_key = args.dataset.split("(", 1)[0].split(":", 1)[0].lower()

    # Attempt to pre-calculate or load dataset metadata required for transforms
    try:
        prime_dataset_meta_for_transform(args.dataset, args.data_dir)
    except Exception as e:
        logger.warning(f"[DATA] prime_dataset_meta_for_transform failed: {e}")

    if base_key not in DATASET_META:
        raise ValueError(f"Dataset '{args.dataset}' not present in DATASET_META.")

    # Override the default input size in metadata if specified in arguments
    if getattr(args, "input_size", 0):
        new_sz = int(args.input_size)
        # Apply override to known dataset keys associated with this experiment
        for k in (base_key, "nico++", "nicopp", "nico"):
            if k in DATASET_META:
                DATASET_META[k]["input_size"] = new_sz
        logger.info(f"[DATA] Overriding input_size -> {new_sz} for dataset '{args.dataset}'")

    # Construct hierarchical path components
    date_str, time_for_path, time_iso = _utc_now_parts()
    run_leaf = f"run{run_id}" if run_id is not None else time_for_path

    partition_mode = getattr(args, "partition", "silos")
    aggregation_mode = getattr(args, "aggregation", "simple")
    # Determine display name for aggregation; "standard" implies basic FedAvg in Silos mode
    aggregation_display = "standard" if partition_mode == "silos" else aggregation_mode
    model_name = args.model.replace("baseline:", "") if args.model.startswith("baseline:") else args.model

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

    # Archive the current script file to the environment directory for reproducibility
    try:
        src_self = Path(__file__)
        shutil.copy2(src_self, P.root / "environment" / src_self.name)
    except Exception:
        # Failure to copy the script should not halt the experiment
        pass

    return P, time_iso


def prepare_data(args: Any, device: torch.device) -> Tuple:
    """
    Load, transform, and partition the dataset for the experiment.

    This function handles the end-to-end data pipeline:
    1. Configures transforms (resizing, grayscale, robustness noise).
    2. Loads the raw training and canonical testing datasets.
    3. Partitions the training data based on the selected mode (Silos, Skew, Dirichlet).
    4. Prepares evaluation tensors and loaders.

    Args:
        args (Any): Configuration namespace containing dataset and partition settings.
        device (torch.device): The target computation device (CPU/GPU).

    Returns:
        Tuple containing:
            - base_train_set (Dataset): The full, raw training dataset.
            - test_set (Dataset): The canonical test dataset with transforms.
            - tfm_train (T.Compose): Transform pipeline for training data.
            - num_classes (int): Total number of unique classes.
            - img_shape (Tuple[int, ...]): Dimensions of a single image sample.
            - chans (int): Number of image channels.
            - train_subsets_dict (Dict): Dictionary mapping clients to their data indices/subsets.
            - train_subsets (List[Dataset]): List of dataset objects for each client.
            - present_classes (List[int]): List of class indices present in the current partition.
            - reserved_test_ld (DataLoader): DataLoader for the full canonical test set.
            - reserved_test_imgs_list (List[Tensor]): List of test tensors per class.
            - test_imgs_tensor (Tensor): Full test set as a single tensor.
    """
    # FIXED: Correct parsing for colon-separated names (e.g. medmnist:retinamnist)
    base_key = args.dataset.split("(", 1)[0].split(":", 1)[0].lower()

    # DETECT BASELINE: If we are running a baseline, we can skip memory-heavy evaluation steps
    is_baseline = args.model.startswith("baseline:")

    meta = DATASET_META[base_key]
    chans = meta["channels"]

    use_robustness = getattr(args, "robustness", True)
    if use_robustness:
        logger.info("[DATA] 🛡️ Robustness Enabled: Injecting Gaussian Noise")

    # 1. Construct base transformation pipelines
    base_tfm_train = build_transform(args.dataset, train=True, robustness=use_robustness)
    base_tfm_test = build_transform(args.dataset, train=False, robustness=use_robustness)

    # 2. Apply explicit input size overrides if specified
    # MEMORY OPTIMIZATION CHECK: Large inputs (Camelyon/NICO)
    is_large_dataset = False
    if getattr(args, "input_size", 0):
        sz = int(args.input_size)
        if sz > 64:
            is_large_dataset = True
        resize_op = T.Resize((sz, sz), antialias=True)
        tfm_train = T.Compose([resize_op, base_tfm_train])
        tfm_test = T.Compose([resize_op, base_tfm_test])
    else:
        # Check meta default
        if meta.get("input_size", 32) > 64:
            is_large_dataset = True
        tfm_train = base_tfm_train
        tfm_test = base_tfm_test

    # 3. Apply Grayscale conversion if requested for 3-channel images
    if bool(args.grayscale) and chans == 3:
        # Enforce 3 output channels to maintain compatibility with model architectures
        gray_op = T.Grayscale(num_output_channels=3)
        tfm_train = T.Compose([gray_op, tfm_train])
        tfm_test = T.Compose([gray_op, tfm_test])

    # Handle NICO++ specific class filtering
    if args.dataset.lower() in ("nico++", "nicopp", "nico"):
        selected = [c.strip() for c in args.classes.split(",") if c.strip()] or None
        set_dataset_options("nico++", classes=selected)

    # --- LOAD RAW DATASETS ---
    # Load the full pool of training data; augmentation is applied via wrappers later
    base_train_set = get_dataset(args.dataset, args.data_dir, True, None)

    # Load the canonical test set (deterministic behavior required)
    test_set = get_dataset(args.dataset, args.data_dir, False, tfm_test)

    # MEMORY FIX: If baseline OR large dataset, do not load full test set into Tensor.
    if is_large_dataset or is_baseline:
        if is_large_dataset:
            logger.info("[DATA] ⚠️ Large dataset detected: Using Lazy Loading for Test Set to prevent OOM.")
        test_imgs_tensor = torch.empty(0)
    else:
        test_imgs_tensor = dataset_to_tensor(test_set)

    # Determine the number of classes via introspection of the dataset object
    if hasattr(base_train_set, "classes") and len(getattr(base_train_set, "classes")) > 0:
        num_classes = len(base_train_set.classes)
    elif hasattr(base_train_set, "targets"):
        num_classes = int(len(set(map(int, base_train_set.targets))))
    else:
        # Fallback: Iterate to find unique labels (computationally expensive but safe)
        labels_tmp = [int(lbl) for _, lbl in base_train_set]
        num_classes = int(len(set(labels_tmp)))

    # Grab a raw sample (PIL Image)
    raw_sample, _ = base_train_set[0]

    # Apply the training transform to convert it to a Tensor and resize it (if needed)
    # This ensures img_shape matches exactly what the model will receive (C, H, W)
    sample_tensor = tfm_train(raw_sample)

    img_shape = tuple(sample_tensor.shape)

    # --- PREPARE TEST LABELS ---
    # extract labels uniformly across different dataset implementations (targets vs labels)
    labels_src = None
    for attr in ("targets", "labels"):
        if hasattr(test_set, attr):
            labels_src = getattr(test_set, attr)
            break
    if labels_src is None and hasattr(test_set, "imgs"):
        labels_src = [t for _, t in test_set.imgs]

    test_lbls_all = torch.as_tensor(labels_src, dtype=torch.long).reshape(-1)

    # Normalize EMNIST labels to be 0-indexed if they are offset
    if args.dataset.startswith("emnist") and test_lbls_all.min().item() != 0:
        test_lbls_all = test_lbls_all - test_lbls_all.min()

    # MEMORY FIX: Use Lazy DataLoader for large datasets instead of TensorDataset.
    # Also set num_workers=0 to avoid fork overhead in concurrent environments.
    if is_large_dataset or is_baseline:
        reserved_test_ld = DataLoader(test_set, batch_size=args.batch_size, num_workers=0, shuffle=False)
    else:
        reserved_test_ld = DataLoader(TensorDataset(test_imgs_tensor, test_lbls_all), batch_size=args.batch_size)

    # --- PARTITION LOGIC ---
    partition_mode = getattr(args, "partition", "silos")
    train_subsets_dict = {}
    train_subsets = []
    reserved_test_imgs_list = []
    present_classes = []

    # Unify access to test targets as a numpy array
    if hasattr(test_set, "targets"):
        test_targets_arr = np.asarray(test_set.targets)
    elif hasattr(test_set, "labels"):
        test_targets_arr = np.asarray(test_set.labels)
    else:
        test_targets_arr = np.array([lbl for _, lbl in test_set])

    if test_targets_arr.ndim > 1: test_targets_arr = test_targets_arr[:, 0]

    # Apply EMNIST offset correction to numpy array targets
    if args.dataset.startswith("emnist") and test_targets_arr.min() != 0:
        test_targets_arr = test_targets_arr - test_targets_arr.min()

    # MEMORY FIX: Limit samples for Generative Metrics (FID/KID)
    max_eval_samples = getattr(args, "eval_samples_per_class", 2000)

    if partition_mode in ["skew", "dirichlet"]:
        # Logic for Synthetic Federated Partitions (Skew/Dirichlet)
        if partition_mode == "skew":
            client_config = parse_client_config(getattr(args, "client_config", ""))
            train_subsets_dict, _ = create_skew_partition(base_train_set, client_config, args.seed, num_classes)
        else:
            alpha = float(getattr(args, "alpha", 0.5))
            n_clients = int(getattr(args, "num_clients", 10))
            train_subsets_dict, _ = create_dirichlet_partition(base_train_set, n_clients, alpha, args.seed)

        present_classes = list(range(num_classes))

        # Create evaluation tensors for every class using the CANONICAL TEST SET
        for d in range(num_classes):
            # MEMORY FIX: If running a baseline, we DO NOT need these heavy tensors.
            # Skip loading them to save GBs of RAM.
            if is_baseline:
                reserved_test_imgs_list.append(torch.empty(0))
                continue

            idxs = np.flatnonzero(test_targets_arr == d)
            if len(idxs) == 0:
                reserved_test_imgs_list.append(torch.empty(0))
                continue

            # Slice indices if dataset is large to prevent OOM
            if is_large_dataset and len(idxs) > max_eval_samples:
                idxs = idxs[:max_eval_samples]

            sub = Subset(test_set, idxs)
            reserved_test_imgs_list.append(subset_to_tensor(sub))

    else:
        # Logic for "Silos" Mode (Natural/Domain Splits)
        if hasattr(base_train_set, "targets"):
            train_targets_arr = np.asarray(base_train_set.targets)
        elif hasattr(base_train_set, "labels"):
            train_targets_arr = np.asarray(base_train_set.labels)
        else:
            train_targets_arr = np.array([lbl for _, lbl in base_train_set])

        if train_targets_arr.ndim > 1: train_targets_arr = train_targets_arr[:, 0]
        if args.dataset.startswith("emnist") and train_targets_arr.min() != 0:
            train_targets_arr = train_targets_arr - train_targets_arr.min()

        for d in range(num_classes):
            # Identify all training samples belonging to this specific class
            idxs = np.flatnonzero(train_targets_arr == d)
            if len(idxs) == 0: continue
            present_classes.append(d)

            # 1. Assign 100% of these indices for Training
            raw_train_subset = Subset(base_train_set, idxs)
            train_subsets.append(raw_train_subset)

            # 2. Extract corresponding samples from the Canonical Test Set for evaluation
            # MEMORY FIX: If baseline, skip this heavy loading.
            if is_baseline:
                reserved_test_imgs_list.append(torch.empty(0))
                continue

            test_class_idxs = np.flatnonzero(test_targets_arr == d)
            if len(test_class_idxs) > 0:
                # Slice indices for generative metrics if dataset is large
                if is_large_dataset and len(test_class_idxs) > max_eval_samples:
                    test_class_idxs = test_class_idxs[:max_eval_samples]

                test_sub_final = Subset(test_set, test_class_idxs)
                reserved_test_imgs_list.append(subset_to_tensor(test_sub_final))
            else:
                reserved_test_imgs_list.append(torch.empty(0))

    # Adjust the number of classes if the partition resulted in a subset of the total classes
    if len(present_classes) != num_classes and partition_mode == 'silos':
        num_classes = len(present_classes)

    return (base_train_set, test_set, tfm_train, num_classes, img_shape, chans,
            train_subsets_dict, train_subsets, present_classes,
            reserved_test_ld, reserved_test_imgs_list, test_imgs_tensor)

def _export_client_class_distribution(
        root: Path,
        client_sample_counts: Dict[str, Dict[int, int]],
        num_classes: int
) -> None:
    """
    Export the distribution of classes per client to a CSV file.

    This function generates a matrix where rows correspond to clients and columns
    correspond to classes, with values indicating the sample count.

    Args:
        root (Path): The root directory where the 'distributions' folder resides.
        client_sample_counts (Dict[str, Dict[int, int]]): A mapping of client names to
                                                         dictionaries of class counts.
        num_classes (int): The total number of classes to include in the header.
    """
    distributions_dir = root / "distributions"
    distributions_dir.mkdir(parents=True, exist_ok=True)
    csv_file = distributions_dir / "client_class_distribution.csv"

    header = ["client"] + [f"class_{i}" for i in range(num_classes)]
    client_names = sorted(client_sample_counts.keys())

    rows = []
    for client_name in client_names:
        row = [client_name]
        class_counts = client_sample_counts[client_name]
        for class_id in range(num_classes):
            count = class_counts.get(class_id, 0)
            row.append(str(count))
        rows.append(row)

    with open(csv_file, 'w') as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")

    logger.info(f"[EXPORT] Client class distribution exported to: {csv_file}")