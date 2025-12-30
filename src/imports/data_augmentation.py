"""
🌌 Data Augmentation and Transformation Module
----------------------------------------------

This module implements a scientifically rigorous factory for constructing image
transformation pipelines. It provides domain-specific augmentation strategies
tailored for disparate data modalities, including natural images, medical
imaging, and structural patterns (e.g., optical character recognition).

🧠 Purpose:
    To standardize data preprocessing across experimental conditions, ensuring
    statistical consistency during model training and evaluation. It handles
    resolution adjustments, geometric transformations, and intensity
    normalization based on empirical dataset statistics.

🔧 Core Functionalities:
    • Factory-based construction of `torchvision.transforms.Compose` pipelines
    • Domain-aware strategy selection (e.g., Isotropic vs. Oriented medical data)
    • Inverse normalization utilities for visualization and debugging
    • Robustness injection via additive Gaussian noise for simulated heterogeneity

🎯 Intended Use:
    • Deep Learning research pipelines
    • Federated Learning experiments requiring client-side data standardization
    • Benchmarking across diverse datasets (CIFAR, ImageNet, MedMNIST, etc.)

📁 Dependencies:
    • torch
    • torchvision
    • .data_management (Dataset metadata and parsing logic)

📝 Notes:
    The module assumes input images are RGB or Grayscale. Statistical
    normalization values (Mean/Std) are hardcoded based on standard
    literature benchmarks (e.g., ImageNet, CIFAR-10).

Author: Andrea Moleri
File Location: src/data/data_augmentation.py
Last Modified: 07/12/2025
"""

from __future__ import annotations
import logging
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from .data_management import DATASET_META, _medmnist_parse

logger = logging.getLogger(__name__)

# ============================ Scientific Constants ===============================

# Empirical mean and standard deviation for the ImageNet dataset (RGB channels).
# Used for normalizing high-resolution natural images to aid model convergence.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Empirical statistics for CIFAR-10/100 datasets.
# Precise normalization is critical for reproducing baseline accuracies in low-res domains.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

# Neutral statistics for mapping inputs to the range [-1, 1].
# Often used when domain-specific statistics are unavailable or for zero-centered data.
NEUTRAL_MEAN = (0.5, 0.5, 0.5)
NEUTRAL_STD = (0.5, 0.5, 0.5)
NEUTRAL_MEAN_1CH = (0.5,)
NEUTRAL_STD_1CH = (0.5,)


# ============================ Utility Functions ==================================

def denormalize(t: torch.Tensor, dataset_name: str = "imagenet") -> torch.Tensor:
    """
    Reverses the normalization operation to restore tensors to the [0, 1] range.

    This function applies the inverse transformation:
    $x_{orig} = x_{norm} \times \sigma + \mu$
    It is primarily used for visualization, logging, or debugging purposes where
    human-interpretable images are required.

    Args:
        t (torch.Tensor): The normalized input tensor, typically with shape
            (C, H, W) or (B, C, H, W).
        dataset_name (str): The identifier of the dataset (e.g., 'cifar10',
            'imagenet') used to resolve the correct statistical parameters.

    Returns:
        torch.Tensor: The denormalized tensor with values approximately in [0, 1].

    Raises:
        None: Safe defaults are applied if the dataset name is unrecognized.
    """
    device = t.device
    ds = dataset_name.lower()

    # Determine the appropriate statistical distribution based on the dataset identifier.
    # String matching is used to handle variants (e.g., 'cifar10' vs 'cifar100').
    if "cifar" in ds:
        mean, std = CIFAR_MEAN, CIFAR_STD
    elif any(x in ds for x in ["medmnist", "mnist", "omniglot", "emnist"]):
        # Handle channel differences for grayscale vs RGB variants of structural datasets.
        if t.shape[0] == 3:
            mean, std = NEUTRAL_MEAN, NEUTRAL_STD
        else:
            mean, std = NEUTRAL_MEAN_1CH, NEUTRAL_STD_1CH
    elif t.shape[0] == 3:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = NEUTRAL_MEAN_1CH, NEUTRAL_STD_1CH

    # Reshape statistics to (C, 1, 1) to enable broadcasting against the spatial dimensions (H, W).
    mean_t = torch.tensor(mean).view(-1, 1, 1).to(device)
    std_t = torch.tensor(std).view(-1, 1, 1).to(device)

    return t * std_t + mean_t


class AddGaussianNoise:
    """
    (Deactivated) A callable transformation that passes the tensor through unchanged.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        DEACTIVATED: Returns original tensor without noise.
        """
        # ORIGINAL CODE (Commented out):
        # return tensor + torch.randn_like(tensor) * self.std + self.mean
        return tensor


# ============================ Granular Pipeline Logic ============================

def _get_medmnist_strategy(name: str) -> str:
    """
    Determines the augmentation strategy for MedMNIST sub-datasets.

    Distinguishes between datasets where rotation preserves semantic meaning
    (isotropic) and those where orientation is diagnostically relevant.

    Args:
        name (str): The specific MedMNIST subset name (e.g., 'pathmnist').

    Returns:
        str: 'medical_isotropic' for rotation-invariant data, or
             'medical_oriented' otherwise.
    """
    subset, _ = _medmnist_parse(name)
    # Datasets like Pathology or Dermatology are often view-agnostic (microscope slides).
    isotropic = ["pathmnist", "bloodmnist", "tissuemnist", "dermamnist"]
    if subset in isotropic:
        return "medical_isotropic"
    return "medical_oriented"


def _is_sparse_domain(dataset_name: str) -> bool:
    """
    Identifies if a dataset belongs to a sparse or structural domain.

    Sparse domains typically consist of line drawings, sketches, or glyphs
    rather than dense natural photography.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        bool: True if the dataset represents a sparse domain, False otherwise.
    """
    name = dataset_name.lower()
    return any(x in name for x in ["sketch", "quickdraw", "infograph", "mnist", "omniglot"])


def build_transform(dataset_name: str, train: bool = False, robustness: bool = False):
    """
    Constructs a scientifically calibrated transformation pipeline for a given dataset.

    This factory method orchestrates the assembly of preprocessing steps including
    resizing, cropping, geometric augmentation, and normalization. It adapts
    dynamically to the dataset's resolution (low-res vs. high-res) and modality
    (natural vs. medical vs. structural).

    Args:
        dataset_name (str): The unique identifier for the target dataset.
        train (bool): If True, applies stochastic augmentations (e.g., random crops,
            flips) for training. If False, applies deterministic transforms for
            evaluation. Defaults to False.
        robustness (bool): If True, appends noise injection to the pipeline to
            simulate data corruption or enhance model robustness. Defaults to False.

    Returns:
        torchvision.transforms.Compose: A composed pipeline of transformations.

    Raises:
        ValueError: If the dataset metadata cannot be resolved from the global registry.
    """
    # force deactivate
    robustness = False

    # 1. Resolve Metadata
    # distinct MedMNIST subsets need to map back to the 'medmnist' base key for metadata lookup.
    base_key = dataset_name.split("(", 1)[0].lower()
    if base_key.startswith("medmnist"): base_key = "medmnist"

    # Retrieve dataset properties (input size, channel count) from the metadata registry.
    # Falls back to partial string matching if the exact key is not found.
    meta = DATASET_META.get(base_key) or DATASET_META.get(dataset_name)
    if meta is None:
        for k in DATASET_META:
            if k in dataset_name:
                meta = DATASET_META[k]
                break
        if meta is None: raise ValueError(f"Unknown dataset: {dataset_name}")

    target_size = meta["input_size"]
    channels = meta["channels"]

    # 2. Determine Strategy
    # Categorize the dataset to select the appropriate augmentation philosophy.
    structural_sets = ["mnist", "fashion", "kmnist", "qmnist", "emnist", "omniglot", "quickdraw"]

    if base_key in structural_sets:
        strategy = "structural"
    elif base_key == "medmnist":
        strategy = _get_medmnist_strategy(dataset_name)
    elif any(x in base_key for x in ["camelyon", "isic", "fed_camelyon", "fed_isic"]):
        strategy = "medical_isotropic"
    elif "cifar" in base_key:
        strategy = "cifar_optimized"
    elif target_size <= 64:
        strategy = "natural_low_res"
    else:
        strategy = "natural_high_res"

    ops = []

    # --- TRAIN TIME ---
    if train:
        if strategy == "structural":
            # For digits and characters, slight affine transformations simulate
            # handwriting variations without destroying legibility.
            ops.append(transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR))
            ops.append(transforms.Resize((target_size, target_size)))

        elif strategy == "medical_isotropic":
            # Isotropic medical data (e.g., cells) allows for aggressive geometric
            # invariances including 90-degree rotations.
            ops.append(transforms.RandomHorizontalFlip())
            ops.append(transforms.RandomVerticalFlip())
            ops.append(transforms.RandomChoice([
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
                transforms.Lambda(lambda x: x)
            ]))
            ops.append(transforms.Resize((target_size, target_size), interpolation=InterpolationMode.BICUBIC))

        elif strategy == "medical_oriented":
            # Oriented medical data (e.g., Chest X-Ray) must maintain verticality.
            # ColorJitter helps simulate varying acquisition equipment settings.
            ops.append(transforms.RandomAffine(
                degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR))
            ops.append(transforms.Resize((target_size, target_size), interpolation=InterpolationMode.BICUBIC))
            ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))

        elif strategy in ["natural_low_res", "cifar_optimized"]:
            # Standard optimization for low-res inputs (e.g., 32x32):
            # Resize guarantees dimensions, followed by padding and cropping to prevent information loss.
            ops.append(transforms.Resize((target_size, target_size)))
            ops.append(transforms.RandomCrop(target_size, padding=4, padding_mode='reflect'))
            ops.append(transforms.RandomHorizontalFlip())

        elif strategy == "natural_high_res":
            # High-resolution regimes (e.g., ImageNet) benefit from RandomResizedCrop (RRC).
            # The crop scale is adjusted based on domain sparsity.
            is_sparse = _is_sparse_domain(dataset_name)
            crop_scale = (0.6, 1.0) if is_sparse else (0.2, 1.0)

            ops.append(transforms.RandomResizedCrop(
                target_size, scale=crop_scale, interpolation=InterpolationMode.BICUBIC
            ))
            ops.append(transforms.RandomHorizontalFlip())

            if not is_sparse:
                # RandAugment is the current State-of-the-Art (SOTA) standard for
                # dense natural images, significantly improving generalization.
                ops.append(transforms.RandAugment(num_ops=2, magnitude=9))

    # --- TEST TIME ---
    else:
        if strategy in ["natural_low_res", "cifar_optimized", "structural"]:
            # Direct resizing for low-resolution or structural data where cropping
            # might remove essential features.
            ops.append(transforms.Resize((target_size, target_size)))
        else:
            # Standard evaluation protocol for high-res CNNs:
            # Resize the smaller edge to 256 (approx) and CenterCrop the target size.
            crop_scale = 256.0 / 224.0
            resize_dim = int(target_size * crop_scale)
            ops.append(transforms.Resize(resize_dim, interpolation=InterpolationMode.BICUBIC))
            ops.append(transforms.CenterCrop(target_size))

    # 4. Finalize
    # Convert PIL images or ndarrays to PyTorch Tensors [C, H, W] in range [0, 1].
    ops.append(transforms.ToTensor())

    # 5. Normalization (Scientifically Accurate)
    # Apply statistical normalization based on the dataset domain.
    # FIX: Use actual channel count, do not force 3 channels if input is 1.
    effective_channels = channels

    if "cifar" in dataset_name.lower():
        norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    elif strategy == "natural_high_res" and effective_channels == 3:
        norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    elif effective_channels == 3:
        # This catch-all applies to RGB datasets
        norm = transforms.Normalize(NEUTRAL_MEAN, NEUTRAL_STD)
    else:
        # This handles Grayscale datasets (MNIST, etc.)
        norm = transforms.Normalize(NEUTRAL_MEAN_1CH, NEUTRAL_STD_1CH)

    ops.append(norm)

    # 6. Robustness (Noise) - Applied AFTER Normalization
    # Noise is added in the normalized feature space to ensure consistent magnitude relative to the signal.
    if robustness:
        ops.append(AddGaussianNoise(std=0.05))

    return transforms.Compose(ops)