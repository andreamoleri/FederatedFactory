"""
💾 Data Management Module
-------------------------

This module provides a unified interface for loading, configuring, and managing diverse 
datasets used in machine learning pipelines. It abstracts the complexities of different 
backend sources—including TorchVision, Hugging Face (FLamby), and custom local 
directories (ImageFolder, .npz)—into a standardized retrieval mechanism.

🧠 Purpose:
    To serve as a centralized factory and registry for dataset ingestion, ensuring 
    consistency in data preprocessing, augmentation application, and metadata retrieval 
    across experimental setups.

🔧 Core Functionalities:
    • Unified `get_dataset` entry point for all supported data sources.
    • Dynamic registry (`DATASET_META`) for dataset metadata management.
    • Specialized logic for:
        - Hugging Face datasets (FLamby integration).
        - DomainNet (concatenation of domains with class intersection).
        - NICO++ (filtering and label remapping).
        - MedMNIST 2D (direct .npz ingestion).
    • Runtime parameterization via string parsing (e.g., specifying resolution or subsets).

🎯 Intended Use:
    • Academic research involving federated learning or domain adaptation.
    • Standardized benchmarking across multiple dataset types.

📁 Dependencies:
    • torch
    • torchvision
    • numpy
    • datasets (Hugging Face, optional/lazy-loaded)
    • huggingface_hub (optional/lazy-loaded)

📝 Notes:
    The module implements lazy loading for Hugging Face dependencies to minimize 
    overhead in environments where they are not required.

Author: Andrea Moleri
File Location: src/imports/data_management.py
Last Modified: 06/12/2025
"""

from __future__ import annotations

import re
import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Set

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms

# =============== FLAMBY (Hugging Face) setup ================================
# Attempt to import Hugging Face libraries. If missing, install them dynamically.
# This ensures the pipeline remains robust even in minimal environments.
try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub"]
    )
    from datasets import load_dataset  # type: ignore
    from huggingface_hub import hf_hub_download  # type: ignore

# --- Runtime options for datasets (configurable via setters) -----------------
_DATASET_OPTS: Dict[str, Dict[str, Any]] = {}


def set_dataset_options(name: str, **kwargs: Any) -> None:
    """
    Persists runtime configuration options for a specific dataset.

    These options can modify how a dataset is loaded later in the pipeline,
    such as filtering for specific classes in NICO++.

    Args:
        name (str): The name of the dataset (case-insensitive).
        **kwargs: Key-value pairs representing the options to store.

    Example:
        set_dataset_options('nico++', classes=['dog', 'cat', 'bus'])
    """
    name = name.lower()
    _DATASET_OPTS.setdefault(name, {}).update(kwargs)


def _get_dataset_opts(name: str) -> Dict[str, Any]:
    """
    Retrieves the stored runtime options for a given dataset.

    Args:
        name (str): The name of the dataset.

    Returns:
        Dict[str, Any]: A dictionary of configuration options, or an empty dict if none exist.
    """
    return _DATASET_OPTS.get(name.lower(), {})


# ------------------------- HF registry ---------------------------------------
# Metadata registry for FLamby/Hugging Face datasets.
_FLAMBY_INFO: Dict[str, Dict[str, Any]] = {
    "fed_camelyon16": dict(
        hf_repo="1aurent/PatchCamelyon",
        split_train="train",
        split_test="test",
        num_classes=2,
        channels=3,
    ),
    "fed_isic2019": dict(
        hf_repo="flwrlabs/fed-isic2019",
        split_train="train",
        split_test="test",
        num_classes=8,
        channels=3,
    ),
    "fed-isic2019": dict(  # Identical alias
        hf_repo="flwrlabs/fed-isic2019",
        split_train="train",
        split_test="test",
        num_classes=8,
        channels=3,
    ),
}


class _HFDatasetWrapper(Dataset):
    """
    Adapter class to wrap a Hugging Face `datasets.Dataset` into a `torch.utils.data.Dataset`.

    This wrapper bridges the gap between the dictionary-based access pattern of
    Hugging Face datasets and the index-based access pattern required by PyTorch
    DataLoaders. It also handles on-the-fly transformations.
    """

    def __init__(self, hf_ds: Any, transform: Optional[transforms.Compose]) -> None:
        """
        Initialize the wrapper.

        Args:
            hf_ds: The Hugging Face dataset object.
            transform: A torchvision transform pipeline to apply to images.
        """
        super().__init__()
        self.hf_ds = hf_ds
        self.transform = transform
        # Pre-load targets into a numpy array for efficient access
        self._targets = np.array(hf_ds["label"], dtype=np.int64)

    @property
    def targets(self) -> np.ndarray:
        """Access the label array directly."""
        return self._targets

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.hf_ds)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Retrieve a sample by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Any, int]: A tuple containing the transformed image and its integer label.
        """
        sample = self.hf_ds[int(idx)]
        img = sample["image"]  # Returns a PIL.Image
        lbl = int(sample["label"])

        if self.transform is not None:
            img = self.transform(img)

        return img, lbl


def _load_flamby(name: str, root: str, train: bool, transform: Optional[transforms.Compose]) -> _HFDatasetWrapper:
    """
    Loads a FLamby dataset from the Hugging Face Hub.

    Args:
        name (str): The key identifying the dataset in `_FLAMBY_INFO`.
        root (str): The root directory for caching the downloaded data.
        train (bool): If True, loads the training split; otherwise, loads the test split.
        transform: The transformation pipeline to apply.

    Returns:
        _HFDatasetWrapper: A PyTorch-compatible dataset instance.
    """
    info = _FLAMBY_INFO[name]
    cache_dir = Path(root) / "FLamby" / name
    cache_dir.mkdir(parents=True, exist_ok=True)

    split = info["split_train"] if train else info["split_test"]
    ds = load_dataset(info["hf_repo"], split=split, cache_dir=str(cache_dir))

    return _HFDatasetWrapper(ds, transform)


# --------------------- NICO++: Class Filtering and Label Remapping -----------
from torchvision.datasets import ImageFolder


class _FilterRemapImageFolder(torch.utils.data.Dataset):
    """
    A wrapper around `torchvision.datasets.ImageFolder` that filters specific classes
    and remaps their labels to a contiguous range [0, K-1].

    This is particularly useful for NICO++ experiments where only a subset of
    available classes constitutes the target domain.
    """

    def __init__(
            self,
            base: ImageFolder,
            allowed_classes: Optional[Union[List[str], List[int]]]
    ) -> None:
        """
        Initialize the filtered dataset.

        Args:
            base (ImageFolder): The original ImageFolder dataset containing all classes.
            allowed_classes: A list of class names (str) or indices (int) to retain.
                             If None or empty, all classes are retained.

        Raises:
            ValueError: If a requested class name does not exist in the source dataset.
        """
        super().__init__()
        self.base = base
        # Mapping: class_name -> original_idx (e.g., {'dog': 0, 'cat': 1})
        cls2idx = base.class_to_idx
        all_class_names = list(cls2idx.keys())

        if allowed_classes is None or len(allowed_classes) == 0:
            # Usage context: Use all classes; maintain native ordering.
            chosen_names = all_class_names
        else:
            # Normalize inputs: convert indices to names, or normalize case for names.
            if isinstance(allowed_classes[0], int):
                chosen_names = [all_class_names[i] for i in allowed_classes]  # type: ignore
            else:
                # Case-insensitive normalization map
                name_norm = {c.lower(): c for c in all_class_names}
                chosen_names = []
                for c in allowed_classes:
                    key = str(c).lower()
                    if key not in name_norm:
                        # Construct a helpful error message with available examples
                        available_preview = all_class_names[:8]
                        raise ValueError(
                            f"Class '{c}' not found in NICO++ (available examples: "
                            f"{available_preview} ... total={len(all_class_names)})"
                        )
                    chosen_names.append(name_norm[key])

        # Create mapping: original_idx -> new_idx
        # The new index corresponds to the position in the user-provided list.
        chosen_new_order = {cls2idx[name]: new_i for new_i, name in enumerate(chosen_names)}
        chosen_old_idxs = set(chosen_new_order.keys())

        # Filter samples and generate remapped targets
        self.samples = []
        self.targets = []
        for path, orig_t in base.samples:
            if orig_t in chosen_old_idxs:
                # Store path and the NEW label
                self.samples.append((path, chosen_new_order[orig_t]))
                self.targets.append(chosen_new_order[orig_t])

        # Mimic ImageFolder metadata for compatibility with other tools
        self.classes = chosen_names
        self.class_to_idx = {name: i for i, name in enumerate(chosen_names)}
        self.transform = base.transform
        self.loader = base.loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        path, target = self.samples[idx]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def _load_nicopp(root: str, train: bool, transform: Optional[transforms.Compose]) -> _FilterRemapImageFolder:
    """
    Loads the NICO++ dataset structured as an ImageFolder.

    Expected Structure: <root>/NICO++/{train|test}/<class>/*.jpg

    If `set_dataset_options` has been used to specify a subset of classes,
    this function applies the necessary filtering and remapping.

    Args:
        root (str): The root data directory.
        train (bool): True for training set, False for test set.
        transform: Image transformation pipeline.

    Returns:
        _FilterRemapImageFolder: The configured dataset object.
    """
    split_dir = "train" if train else "test"
    ds_root = Path(root) / "NICO++" / split_dir
    base = ImageFolder(str(ds_root), transform=transform)

    # Retrieve filtering options (list[str] or list[int])
    opts = _get_dataset_opts("nico++")
    allowed = opts.get("classes", None)
    return _FilterRemapImageFolder(base, allowed)


# ============================ Dataset Registry ===============================
"""
DATASET_META: Central configuration dictionary.
-----------------------------------------------
Each entry defines:
    • loader: Function or class to instantiate the dataset.
    • num_classes: Integer count of classes (or None if dynamic).
    • channels: Integer channel count (1 for grayscale, 3 for RGB).
    • default_split: Dictionary of extra arguments required by the loader.
    • input_size: Recommended input resolution (int).
"""
DATASET_META: Dict[str, Dict[str, Any]] = {
    # ---------- TorchVision Classics ---------------------------------------
    "mnist": dict(loader=datasets.MNIST, num_classes=10, channels=1,
                  default_split={}, input_size=32),
    "fashion": dict(loader=datasets.FashionMNIST, num_classes=10, channels=1,
                    default_split={}, input_size=32),
    "qmnist": dict(loader=datasets.QMNIST, num_classes=10, channels=1,
                   default_split={"what": "train"}, input_size=32),
    "kmnist": dict(loader=datasets.KMNIST, num_classes=10, channels=1,
                   default_split={}, input_size=32),
    "cifar": dict(loader=datasets.CIFAR10, num_classes=10, channels=3,
                  default_split={}, input_size=32),

    # ---------- EMNIST Variants --------------------------------------------
    "emnist-balanced": dict(loader=datasets.EMNIST, num_classes=47, channels=1,
                            default_split={"split": "balanced"}, input_size=32),
    "emnist-letters": dict(loader=datasets.EMNIST, num_classes=26, channels=1,
                           default_split={"split": "letters"}, input_size=32),
    "emnist-digits": dict(loader=datasets.EMNIST, num_classes=10, channels=1,
                          default_split={"split": "digits"}, input_size=32),

    # ---------- Omniglot ---------------------------------------------------
    "omniglot": dict(loader=datasets.Omniglot, num_classes=1623, channels=1,
                     default_split={}, input_size=32),

    # ---------- NotMNIST / QuickDraw (ImageFolder) -------------------------
    "notmnist": dict(loader=datasets.ImageFolder, num_classes=10, channels=1,
                     default_split={}, input_size=32),
    "quickdraw10": dict(loader=datasets.ImageFolder, num_classes=10, channels=1,
                        default_split={}, input_size=32),

    # ---------- NICO++ -----------------------------------------------------
    "nico++": dict(
        loader=None,  # Handled specially via _load_nicopp in get_dataset
        num_classes=None,  # Determined at runtime based on selected classes
        channels=3,
        default_split={},
        input_size=224,  # Standard for real-world RGB data
    ),
    "nicopp": dict(  # Convenience alias
        loader=None,
        num_classes=None,
        channels=3,
        default_split={},
        input_size=224,
    ),

    # ---------- MedMNIST (2D) ----------------------------------------------
    "medmnist": dict(
        loader=None,  # Handled specially via _load_medmnist
        num_classes=None,  # Set dynamically via manifest or .npz inspection
        channels=None,  # Set dynamically (1 or 3)
        default_split={},
        input_size=28,  # Default, updated at runtime if needed
    ),

    # DomainNet: Managed via parametric naming (see get_dataset logic)
    "domainnet": dict(loader="__DOMAINNET__", num_classes=-1, channels=3,
                      default_split={}, input_size=224),
}

# ------- Integrate FLAMBY datasets into the registry -------------------------
for _name, _info in _FLAMBY_INFO.items():
    if _name in DATASET_META:
        continue
    DATASET_META[_name] = dict(
        loader=None,
        num_classes=_info["num_classes"],
        channels=_info["channels"],
        default_split={},
        input_size=32,
    )


# ============================ Custom Helpers =================================
def _imagefolder_or_fail(path: Path, transform: Optional[transforms.Compose]) -> datasets.ImageFolder:
    """
    Attempts to instantiate an ImageFolder dataset.

    Raises:
        FileNotFoundError: If the directory does not exist or lacks the expected structure.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Directory not found: {path}. "
            "Expected ImageFolder layout: <root>/<dataset>/(train|test)/<class>/*.jpg"
        )
    return datasets.ImageFolder(str(path), transform=transform)


def _list_subdirs(path: Path) -> List[str]:
    """Returns a sorted list of subdirectory names within a given path."""
    if not path.exists():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_dir()])


def _domainnet_parse(name: str) -> Tuple[List[str], int]:
    """
    Parses a parametric DomainNet name string to extract domains and class limit.

    Format:
      "domainnet" -> defaults to 6 domains and 10 classes.
      "domainnet(d=clipart,real;c=10)" -> specified domains and class count.

    Args:
        name (str): The dataset identifier string.

    Returns:
        Tuple[List[str], int]: A list of domains to load and the number of classes (k).
    """
    default_domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    default_k = 10

    m = re.match(r"^domainnet(?:\((.*?)\))?$", name)
    if not m:
        return default_domains, default_k

    inside = m.group(1)
    if not inside:
        return default_domains, default_k

    # Parse key=value pairs separated by semicolons
    params = {}
    for part in inside.split(";"):
        if not part.strip():
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            params[k.strip().lower()] = v.strip()

    # Extract domains (supports 'd' or 'domains')
    doms = params.get("d") or params.get("domains")
    if doms:
        domains = [d.strip().lower() for d in doms.split(",") if d.strip()]
    else:
        domains = default_domains

    # Extract class count (supports 'c', 'classes', or 'k')
    k_str = params.get("c") or params.get("classes") or params.get("k")
    k = int(k_str) if k_str else default_k

    return domains, k


def _domainnet_build(
        root: str,
        train: bool,
        transform: Optional[transforms.Compose],
        domains: List[str],
        k_classes: int
) -> Tuple[Dataset, int]:
    """
    Constructs a unified DomainNet dataset by concatenating chosen domains.

    Crucially, this function ensures intersectional consistency: it identifies 
    classes common to ALL specified domains and retains only those, limited to 
    `k_classes`.

    Expected Layout:
      <root>/domainnet/(train|test)/<domain>/<class>/*.jpg

    Args:
        root (str): Root data directory.
        train (bool): Train/Test split selector.
        transform: Transformation pipeline.
        domains (List[str]): List of domain names (e.g., ['real', 'clipart']).
        k_classes (int): Maximum number of classes to include.

    Returns:
        Tuple[Dataset, int]: The concatenated dataset and the number of selected classes.
    """
    split = "train" if train else "test"
    base = Path(root) / "domainnet" / split

    # 1. Enumerate classes per domain to find the intersection
    classes_per_domain: Dict[str, List[str]] = {}
    for d in domains:
        dom_dir = base / d
        classes_per_domain[d] = _list_subdirs(dom_dir)

    # 2. Calculate common classes across all specified domains
    common: Optional[Set[str]] = None
    for d, cls in classes_per_domain.items():
        s = set(cls)
        common = s if common is None else (common & s)

    common_classes = sorted(list(common or []))
    if not common_classes:
        raise RuntimeError(
            f"No common classes found among domains {domains} in {base}. "
            "Please verify the directory structure."
        )

    # 3. Deterministically limit to k_classes
    selected = common_classes[: max(1, k_classes)]

    # 4. Helper to build a filtered ImageFolder for a specific domain
    def _filtered_imagefolder(domain: str) -> datasets.ImageFolder:
        dom_path = base / domain
        ds = datasets.ImageFolder(str(dom_path), transform=transform)

        # Identify indices of samples belonging to the selected classes
        keep_idx = [i for i, (p, y) in enumerate(ds.samples) if ds.classes[y] in selected]

        # Create a new mapping for the selected classes (sorted alphabetically)
        class_to_idx = {c: i for i, c in enumerate(selected)}

        # Filter samples and remap targets to the new range [0, len(selected)-1]
        samples = [(p, class_to_idx[ds.classes[y]]) for i, (p, y) in enumerate(ds.samples) if i in keep_idx]

        # Update dataset internal state
        ds.samples = samples
        ds.targets = [y for _, y in samples]
        ds.classes = selected
        ds.class_to_idx = class_to_idx
        return ds

    # 5. Build individual datasets and concatenate them
    per_domain_datasets = [_filtered_imagefolder(d) for d in domains]
    merged = ConcatDataset(per_domain_datasets)

    return merged, len(selected)


def _nicopp_build(root: str, train: bool, transform: Optional[transforms.Compose]) -> Tuple[Dataset, int, int]:
    """
    Loads the entire NICO++ dataset via ImageFolder without filtering.

    Expected Layout:
      <root>/nicopp/(train|test)/<class>/*.jpg

    Returns:
        Tuple[Dataset, int, int]: The dataset, number of classes, and channel count (3).
    """
    split = "train" if train else "test"
    base = Path(root) / "nicopp" / split
    if not base.exists():
        # Attempt fallback alias 'nico++'
        base = Path(root) / "nico++" / split

    ds = _imagefolder_or_fail(base, transform)
    num_classes = len(ds.classes)
    return ds, num_classes, 3


# ============================ Unified Data Loader ============================
def get_dataset(name: str, root: str, train: bool, transform: Optional[transforms.Compose]) -> Dataset:
    """
    Factory function to retrieve a dataset instance by name.

    This function acts as a dispatcher, handling specific initialization logic
    for TorchVision, FLamby, custom ImageFolders, and parametric datasets.

    Args:
        name (str): The identifier of the dataset (e.g., 'mnist', 'nico++', 'medmnist(organamnist)').
        root (str): The root path where data is stored or downloaded.
        train (bool): If True, loads the training set; otherwise, loads the test set.
        transform: The transformation pipeline to apply to the data.

    Returns:
        Dataset: A fully initialized PyTorch dataset.

    Raises:
        ValueError: If the dataset name is not recognized in the registry.
    """
    name = name.strip()
    key = name.lower()

    # ----- Case 1: FLAMBY / Hugging Face -----------------------------------
    if key in _FLAMBY_INFO:
        return _load_flamby(key, root, train, transform)

    # ----- Case 2: Parametric DomainNet ------------------------------------
    if key.startswith("domainnet"):
        domains, k = _domainnet_parse(key)
        ds, n_cls = _domainnet_build(root=root, train=train, transform=transform,
                                     domains=domains, k_classes=k)
        # Update metadata dynamically to reflect the intersected class count
        DATASET_META["domainnet"]["num_classes"] = n_cls
        DATASET_META["domainnet"]["channels"] = 3
        return ds

    # ----- Case 3: NICO++ (Special filtering) ------------------------------
    if name in ("nico++", "nicopp", "nico"):
        ds = _load_nicopp(root, train, transform)
        # Dynamically update registry num_classes based on the filtered dataset
        DATASET_META[name]["num_classes"] = len(getattr(ds, "classes", [])) or len(set(ds.targets))
        return ds

    # ----- Case 4: MedMNIST (2D) -------------------------------------------
    if key.startswith("medmnist"):
        subset, res = _medmnist_parse(key)
        ds = _load_medmnist(root, train, transform, subset=subset, res=res)
        return ds

    # ----- Case 5: Omniglot ------------------------------------------------
    if key == "omniglot":
        meta = DATASET_META[key]
        # Omniglot uses 'background' parameter instead of 'train'
        return meta["loader"](root, background=train, download=True, transform=transform, **meta["default_split"])

    # ----- Case 6: NotMNIST / QuickDraw (Standard ImageFolder) -------------
    if key.startswith("notmnist") or key.startswith("quickdraw"):
        split_dir = "train" if train else "test"
        path = Path(root) / key / split_dir
        return datasets.ImageFolder(path, transform=transform)

    # ----- Case 7: Default TorchVision Loaders -----------------------------
    if key not in DATASET_META:
        raise ValueError(f"Dataset '{name}' not found in DATASET_META registry.")

    meta = DATASET_META[key]
    return meta["loader"](root, train=train, download=True, transform=transform, **meta["default_split"])


# ============================ MedMNIST (2D) ==================================

def _medmnist_parse(name: str) -> Tuple[str, int]:
    """
    Parses MedMNIST dataset strings.

    Supported Formats:
      - medmnist:subset:res        -> Shell-safe (e.g., medmnist:organamnist:64)
      - medmnist(organamnist;res=64)
      - medmnist(organamnist,64)
      - medmnist(organamnist)      -> defaults to res=28

    Args:
        name (str): The dataset specifier.

    Returns:
        Tuple[str, int]: The subset name (e.g., 'organamnist') and the resolution.
    """
    key = name.strip().lower()

    # ==============================================================================
    # STRATEGY 1: COLON FORMAT (Shell-Safe)
    # Format: medmnist:subset[:resolution]
    # ==============================================================================
    if ":" in key:
        parts = key.split(":")

        # Validation: Must start with 'medmnist'
        if parts[0] != "medmnist":
            raise ValueError(f"Invalid medmnist spec (must start with 'medmnist'): {name}")

        # Validation: Must have at least the subset name
        if len(parts) < 2 or not parts[1].strip():
            raise ValueError("Must specify the MedMNIST subset (e.g., medmnist:organamnist:64)")

        subset = parts[1].strip()
        res = 28  # Default resolution

        # Parse resolution if provided
        if len(parts) >= 3 and parts[2].strip():
            try:
                res = int(parts[2])
            except ValueError:
                raise ValueError(f"Invalid resolution in spec: {name}")

        return subset, res

    # ==============================================================================
    # STRATEGY 2: LEGACY PARENTHESES FORMAT
    # Format: medmnist(subset, res)
    # ==============================================================================
    assert key.startswith("medmnist"), f"Not a medmnist spec: {name}"

    subset = None
    res = 28

    m = re.match(r"^medmnist\((.*?)\)$", key)
    if not m:
        # Updated error message to include the new colon format
        raise ValueError(
            "For MedMNIST use syntax 'medmnist:subset:res' (shell-safe) "
            "or 'medmnist(<subset>[;res=<int>] | ,<int>)'"
        )

    inside = m.group(1).strip()

    # Sub-Strategy A: Key-value pairs (key=val;key=val)
    if ";" in inside or "=" in inside:
        params = {}
        for part in inside.split(";"):
            if not part.strip():
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip().lower()] = v.strip()
            else:
                # If no '=', assume it's the subset name if not already set
                if subset is None:
                    subset = part.strip().lower()

        if "res" in params:
            try:
                res = int(params["res"])
            except Exception:
                raise ValueError(f"Invalid resolution: {params['res']}")

        if subset is None:
            subset = params.get("subset") or params.get("name")
            if subset is None:
                raise ValueError("Must specify the MedMNIST subset (e.g., organamnist)")

        subset = subset.lower()
        return subset, res

    # Sub-Strategy B: Positional format "<subset>[, <res>]"
    parts = [p.strip() for p in inside.split(",") if p.strip()]
    if len(parts) == 0:
        raise ValueError("Must specify the MedMNIST subset")

    subset = parts[0].lower()
    if len(parts) >= 2:
        res = int(parts[1])

    return subset, res


class _MedMNISTNPZ(torch.utils.data.Dataset):
    """
    Minimal dataset wrapper for MedMNIST (2D) .npz files.

    This class handles the loading of pre-split arrays stored in NumPy format,
    inferred channel dimensions, and PIL conversion for transform compatibility.
    """

    def __init__(self, npz_path: Path, split: str, transform: Optional[transforms.Compose]) -> None:
        """
        Initialize the MedMNIST loader.

        Args:
            npz_path (Path): Path to the .npz file.
            split (str): 'train', 'val', or 'test'.
            transform: Transformation pipeline.
        """
        super().__init__()
        self.transform = transform

        arr = np.load(npz_path, allow_pickle=True)
        split = split.lower()

        if split == "train":
            self.images = arr["train_images"]
            self.labels = arr["train_labels"]
        elif split == "test":
            self.images = arr["test_images"]
            self.labels = arr["test_labels"]
        elif split == "val":
            self.images = arr["val_images"]
            self.labels = arr["val_labels"]
        else:
            raise ValueError(f"Invalid split for MedMNIST: {split}")

        # Normalize labels: (N, 1) -> (N,)
        self.labels = self.labels.reshape(-1).astype(np.int64)

        # Infer channels from the array shape
        # Image shape: (N, H, W) -> 1 channel; (N, H, W, 3) -> 3 channels
        sample = self.images[0]
        if sample.ndim == 2:
            self._channels = 1
        elif sample.ndim == 3 and sample.shape[-1] == 3:
            self._channels = 3
        else:
            # Defensive fallback
            self._channels = 1

        # Ensure correct data type (uint8 is required for PIL)
        if self.images.dtype != np.uint8:
            self.images = np.clip(self.images, 0, 255).astype(np.uint8)

    @property
    def targets(self) -> np.ndarray:
        return self.labels

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img = self.images[idx]
        lbl = int(self.labels[idx])

        # Convert to PIL Image to ensure compatibility with standard torchvision transforms
        from PIL import Image
        if img.ndim == 2:
            pil = Image.fromarray(img, mode="L")  # Grayscale
        else:
            pil = Image.fromarray(img)  # RGB

        if self.transform is not None:
            pil = self.transform(pil)

        return pil, lbl


def _medmnist_manifest_meta(root: str, subset: str) -> Tuple[int, int]:
    """
    Retrieves metadata (channels, number of classes) for a MedMNIST subset.

    Logic:
    1. Try reading the official 'manifest_medmnist_2d.json'.
    2. Fallback: Load a sample .npz file (e.g., training split) and infer properties.
    """
    manifest = Path(root) / "MedMNIST" / "2D" / "manifest_medmnist_2d.json"

    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            entry = data.get(subset, {})
            n_ch = int(entry.get("n_channels", 1))
            n_cls = int(entry.get("n_classes", 2))
            return n_ch, n_cls
        except Exception:
            pass  # Fallback to inspection

    # Robust Fallback: Inspect .npz files at various resolutions
    possible = [
        Path(root) / "MedMNIST" / "2D" / subset / "28" / f"{subset}.npz",
        Path(root) / "MedMNIST" / "2D" / subset / "64" / f"{subset}_64.npz",
        Path(root) / "MedMNIST" / "2D" / subset / "128" / f"{subset}_128.npz",
        Path(root) / "MedMNIST" / "2D" / subset / "224" / f"{subset}_224.npz",
    ]
    for p in possible:
        if p.exists():
            arr = np.load(p, allow_pickle=True)
            imgs = arr["train_images"]
            lbls = arr["train_labels"].reshape(-1)

            # Infer channels and classes
            ch = 1 if imgs.ndim == 3 else (imgs.shape[-1] if imgs.ndim == 4 else 1)
            n_cls = int(len(np.unique(lbls)))
            return ch, n_cls

    # Prudent defaults if detection fails
    return 1, 2


def _load_medmnist(root: str, train: bool, transform: Optional[transforms.Compose], subset: str,
                   res: int) -> _MedMNISTNPZ:
    """
    Loads a MedMNIST 2D subset from a local .npz file at the specified resolution.

    Expected Path Structure:
      <root>/MedMNIST/2D/<subset>/<res>/<subset>_<res>.npz
    """
    base = Path(root) / "MedMNIST" / "2D" / subset / str(res)

    # Filename convention differs for default resolution (28)
    if res == 28:
        npz_path = base / f"{subset}.npz"
    else:
        npz_path = base / f"{subset}_{res}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"MedMNIST file not found: {npz_path}")

    split = "train" if train else "test"
    ds = _MedMNISTNPZ(npz_path, split=split, transform=transform)

    # Dynamically update registry to maintain state consistency across the pipeline
    n_ch, n_cls = _medmnist_manifest_meta(root, subset)
    DATASET_META["medmnist"]["channels"] = n_ch
    DATASET_META["medmnist"]["num_classes"] = n_cls
    DATASET_META["medmnist"]["input_size"] = int(res)

    return ds


def prime_dataset_meta_for_transform(dataset_name: str, root: str) -> None:
    """
    Pre-configures DATASET_META before the dataset is actually instantiated.

    This is crucial for parametric datasets (like MedMNIST or DomainNet) where
    metadata such as `input_size` or `channels` depends on the query string
    and is needed to build the transformation pipeline *before* loading the data.

    Args:
        dataset_name (str): The dataset specifier string.
        root (str): The root data directory.
    """
    key = dataset_name.strip().lower()
    base_key = re.split(r"\(", key)[0]

    if base_key == "medmnist":
        subset, res = _medmnist_parse(key)
        n_ch, n_cls = _medmnist_manifest_meta(root, subset)
        DATASET_META["medmnist"]["channels"] = n_ch
        DATASET_META["medmnist"]["num_classes"] = n_cls
        DATASET_META["medmnist"]["input_size"] = int(res)

    elif base_key == "domainnet":
        # Safe defaults: RGB images, 224x224 resolution
        DATASET_META["domainnet"]["channels"] = 3
        DATASET_META["domainnet"]["input_size"] = 224