"""
🌌 Data Augmentation and Transformation Module
----------------------------------------------

This module provides a comprehensive suite of utilities for image data augmentation, 
tensor normalization, and the construction of adaptive transformation pipelines.

🧠 Purpose:
    Designed to enhance the robustness of machine learning models by systematically 
    introducing variations (noise, affine transformations, occlusions) into training data. 
    It also serves as a factory for standardizing input data dimensions and channel 
    configurations across diverse datasets.

🔧 Core Functionalities:
    • Implementation of additive Gaussian noise injection with range clamping.
    • A dataset wrapper (`NoisyCleanDataset`) for self-supervised or denoising tasks.
    • Dynamic construction of `torchvision` transformation pipelines based on dataset metadata.
    • Utility functions for tensor renormalization.

🎯 Intended Use:
    • Deep learning research pipelines requiring robust data preprocessing.
    • Denoising Autoencoder (DAE) training setups.
    • Standardization of heterogeneous image datasets (Grayscale/RGB).

📁 Dependencies:
    • numpy
    • torch
    • torchvision

📝 Notes:
    This module relies on a global registry (`DATASET_META`) imported from `.data_management` 
    to retrieve dataset-specific parameters such as input size and channel depth.

Author: Andrea Moleri
File Location: src/imports/data_augmentation.py
Last Modified: 06/12/2025
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from .data_management import DATASET_META  # Global dataset registry


# ============================ Utility Functions ==================================
def denormalize(t: torch.Tensor) -> torch.Tensor:
    """
    Rescales a tensor from the normalized range [-1, 1] back to the domain [0, 1].

    This operation is typically required when visualizing tensors that were previously 
    normalized using a mean of 0.5 and a standard deviation of 0.5 (Tanh-style normalization).

    Mathematically:
    $$ x_{new} = \\frac{x_{old} + 1}{2} $$

    Args:
        t (torch.Tensor): The input tensor with values in the range [-1, 1].

    Returns:
        torch.Tensor: The denormalized tensor with values in the range [0, 1].
    """
    return (t + 1) / 2


# ============================ Augmentation Logic ==================================
class AddGaussianNoise:
    """
    A callable class that injects additive Gaussian noise into a tensor.

    This class adds noise drawn from a normal distribution $\mathcal{N}(\mu, \sigma)$ 
    to the input tensor and ensures the resulting values are clamped within the 
    normalized range [-1, 1].
    """

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        """
        Initialize the noise generator.

        Args:
            mean (float): The mean ($\mu$) of the Gaussian distribution. Defaults to 0.0.
            std (float): The standard deviation ($\sigma$) of the Gaussian distribution. Defaults to 0.1.
        """
        self.mean, self.std = mean, std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise to the input tensor.

        Args:
            tensor (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The noisy tensor, clamped to the range [-1.0, 1.0].
        """
        # Generate noise matching the tensor's shape, add it, and clamp the result to maintain validity.
        return torch.clamp(
            tensor + torch.randn_like(tensor) * self.std + self.mean, -1.0, 1.0
        )


class NoisyCleanDataset(Dataset):
    """
    A Dataset wrapper designed to produce paired samples (noisy, clean) for training.

    This class applies "heavy" augmentations (affine transformations, random erasing) 
    to the source image to generate a 'clean' target, and then adds Gaussian noise 
    to a copy of that target to generate the 'noisy' input.



    This approach is particularly useful for training Denoising Autoencoders (DAEs) 
    where the model must learn to reconstruct the geometric and structural content 
    despite noise and occlusions.
    """

    def __init__(self, subset: Subset, noise_std: float, augment: bool = True):
        """
        Initialize the dataset wrapper.

        Args:
            subset (Subset): The underlying data subset containing source images.
            noise_std (float): The standard deviation of the Gaussian noise to be applied.
            augment (bool): If True, applies random affine and erasing transformations. 
                            If False, the transformation pipeline is an identity function.
        """
        self.subset = subset
        self.noise = AddGaussianNoise(0.0, noise_std)
        if augment:
            # Apply slight rotation/translation/scaling
            self.aug = transforms.RandomAffine(
                15, translate=(0.12, 0.12), scale=(0.9, 1.1)
            )
            # Randomly occlude parts of the image to force structure learning
            self.erase = transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)
            )
        else:
            # No-op: Identity function used when augmentation is disabled
            self.aug = self.erase = lambda x: x

    def __len__(self) -> int:  # type: ignore[override]
        """
        Retrieve the total number of samples in the dataset.

        Returns:
            int: The length of the underlying subset.
        """
        return len(self.subset)

    def __getitem__(self, idx: int):  # type: ignore[override]
        """
        Retrieve a sample pair at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - noisy (torch.Tensor): The augmented image with added Gaussian noise.
                - img (torch.Tensor): The augmented image (clean target).
        """
        img, _ = self.subset[idx]

        # Apply geometric and occlusion augmentations to the base image
        img = self.erase(self.aug(img))

        # Return a noisy copy as input, and the clean augmented version as target
        return self.noise(img.clone()), img


# top of file (optional – only if you want extra safety elsewhere)
# Note: The import below appears redundant as it was performed at the module level,
# but it is retained here to preserve the original code structure.
from .data_management import DATASET_META  # already present


def build_transform(dataset_name: str):
    """
    Constructs a transformation pipeline consistent with the dataset's channel count 
    and recommended input dimensions.

    This factory function retrieves metadata from the global registry and builds a 
    `torchvision.transforms.Compose` object. It handles resizing, cropping, tensor 
    conversion, and normalization.

    Args:
        dataset_name (str): The identifier for the dataset (e.g., "cifar10", "medmnist(s)").

    Returns:
        torchvision.transforms.Compose: The composed sequence of image transformations.

    Raises:
        AssertionError: If the resolved `input_size` is not a positive integer.
    """
    # Normalize the dataset key to handle parametric names (e.g., "medmnist(s)" -> "medmnist")
    base_key = dataset_name.split("(", 1)[0].lower()
    meta = DATASET_META.get(base_key, None)
    if meta is None:
        # Fallback: Attempt to use the original name if base key extraction failed
        meta = DATASET_META[dataset_name]

    size = meta["input_size"]
    # Ensure the input size is physically valid
    assert isinstance(size, int) and size > 0, "input_size must be > 0"

    import logging as _logging
    _logging.getLogger(__name__).info(
        f"[TFM] {base_key} input_size={size}, channels={meta['channels']}"
    )
    ch = meta["channels"]

    from torchvision import transforms

    # Define operations common to all image types (Resize -> Crop -> Tensor conversion)
    common_ops = [
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ]

    # Branch logic based on channel depth (RGB vs. Grayscale)
    if ch == 3:
        # For RGB images: Normalize 3 channels
        return transforms.Compose(common_ops + [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # For Grayscale images: Ensure single channel and normalize 1 channel
        return transforms.Compose(
            [transforms.Grayscale()] + common_ops + [
                transforms.Normalize((0.5,), (0.5,))
            ]
        )