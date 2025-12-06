"""
📦 Deep Learning Utilities and Evaluation Module
------------------------------------------------

This module acts as a comprehensive toolkit for deep learning experimentation, 
providing essential utilities for reproducibility, perceptual loss computation, 
data visualization, and model ensemble evaluation.

🧠 Purpose:
    Designed to support rigorous academic research and production-grade 
    pipeline development, this module centralises logic for random seeding, 
    VGG-based feature loss, and differential privacy noise injection.

🔧 Core Functionalities:
    • Deterministic seeding for reproducible stochastic processes
    • Perceptual and Style loss computation using a pretrained VGG-16 network
    • High-performance tensor conversion from PyTorch Datasets
    • Visualisation utilities for training curves and VAE latent grids
    • Ensemble prediction logic using Product of Experts (PoE) aggregation

🎯 Intended Use:
    • Model training pipelines requiring strict reproducibility
    • Generative model evaluation (VAE, GANs)
    • Hyperparameter grid search and configuration parsing
    • Post-training analysis and visualization

📁 Dependencies:
    • torch (PyTorch)
    • torchvision
    • numpy
    • matplotlib

📝 Notes:
    The VGGPerceptualLoss class relies on `torchvision.models` to download 
    pretrained weights. Ensure internet connectivity or local cache availability.

Author: Andrea Moleri
File Location: src/utils/helpers.py
Last Modified: 06/12/2025
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from models.cnn import SimpleCNN
from models.vae import VAE, Decoder


# ============================================================ SEED MANAGEMENT
def set_seed(seed: int) -> None:
    """
    Enforce determinism across random number generators to ensure reproducibility.

    This function sets the seed for Python's built-in `random`, `numpy`, and 
    `torch` (both CPU and GPU). It also configures cuDNN to use deterministic 
    algorithms, which may impact performance but guarantees consistent results.

    Args:
        seed (int): The integer seed value to be used for all RNGs.

    Returns:
        None
    """
    # Seed standard Python RNG and NumPy
    random.seed(seed)
    np.random.seed(seed)

    # Seed PyTorch for CPU and all available GPUs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms in cuDNN (disables benchmarking)
    # This trades computational speed for bit-exact reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================ VGG PERCEPTUAL LOSS
class VGGPerceptualLoss(torch.nn.Module):
    """
    Computes Perceptual (Feature) and Style losses using a pretrained VGG-16 network.

    This class extracts feature maps from specific layers of VGG-16 to compute 
    losses based on high-level semantic content rather than pixel-wise differences.
    It supports both content reconstruction loss and Gram-matrix based style loss.



    Attributes:
        blocks (torch.nn.ModuleList): Frozen VGG-16 blocks used for feature extraction.
        transform (callable): Function to resize inputs to VGG-compatible dimensions.
        resize (bool): Flag indicating whether input tensors should be resized.
        mean (torch.Tensor): ImageNet mean for normalization.
        std (torch.Tensor): ImageNet standard deviation for normalization.
    """

    def __init__(self, resize: bool = True):
        """
        Initialize the VGG perceptual loss module.

        Args:
            resize (bool): If True, resizes inputs to (224, 224) to match 
                           VGG-16 training conditions. Defaults to True.
        """
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        try:
            # Attempt to import and load the pretrained VGG-16 model
            from torchvision import models
            vgg = models.vgg16(pretrained=True).features
        except Exception:
            # Fallback if torchvision or the model is unavailable
            vgg = None

        if vgg is None:
            self.blocks = torch.nn.ModuleList([])
            return

        # Slice VGG-16 into distinct blocks to extract features at different depths.
        # These slices correspond to outputs after pooling layers.
        blocks.append(vgg[:4].eval())
        blocks.append(vgg[4:9].eval())
        blocks.append(vgg[9:16].eval())
        blocks.append(vgg[16:23].eval())

        # Freeze parameters to prevent gradients from flowing back into VGG
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize

        # Register ImageNet normalization statistics as non-trainable buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers: List[int] = [0, 1, 2, 3], style_layers: List[int] = []):
        """
        Compute the weighted sum of feature and style losses.

        Args:
            input (torch.Tensor): Generated image tensor.
            target (torch.Tensor): Target (ground truth) image tensor.
            feature_layers (List[int]): Indices of blocks to use for feature (L1) loss.
            style_layers (List[int]): Indices of blocks to use for style (Gram) loss.

        Returns:
            torch.Tensor: The scalar loss value representing the distance between inputs.
        """
        # Ensure input has 3 channels (RGB); repeat channels if grayscale
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Normalize inputs using ImageNet statistics
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        # Resize to VGG canonical size if requested
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        x = input
        y = target

        # Iterate through VGG blocks, accumulating loss at specified layers
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            # Content/Feature Loss: L1 distance between feature maps
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)

            # Style Loss: L1 distance between Gram matrices
            # The Gram matrix captures texture information: G = A * A^T
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.transpose(1, 2)
                gram_y = act_y @ act_y.transpose(1, 2)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)

        return loss


# ============================================================ SAMPLE GRID
@torch.no_grad()
def sample_grid(vae: VAE, latent_dim: int, out_path: Path, n: int = 64) -> None:
    """
    Generate a grid of synthetic images by sampling from the VAE latent space.



    Args:
        vae (VAE): The trained Variational Autoencoder model.
        latent_dim (int): Dimensionality of the latent vector z.
        out_path (Path): Filesystem path where the grid image will be saved.
        n (int): Number of samples to generate. Defaults to 64.

    Returns:
        None
    """
    vae.eval()
    # Infer device from model parameters to ensure compatibility
    device = next(vae.parameters()).device

    # Sample latent vectors from a standard normal distribution N(0, I)
    z = torch.randn(n, latent_dim, device=device)

    # Decode latent vectors into image space
    samples = vae.decoder(z).cpu()

    # Denormalize from Tanh range [-1, 1] to image range [0, 1]
    samples = (samples + 1.0) / 2.0

    # Create a visual grid of images
    grid = make_grid(samples, nrow=8, padding=2, normalize=False)

    # Convert tensor (C, H, W) to NumPy (H, W, C) for saving
    grid_np = grid.permute(1, 2, 0).numpy()
    plt.imsave(out_path, grid_np)


# ============================================================ DIFFERENTIAL PRIVACY
def _dp_add_noise_(params: List[torch.Tensor], clip: float, noise_mult: float) -> None:
    """
    Inject Gaussian noise into gradients or parameters for Differential Privacy.

    This operation is typically performed during DP-SGD updates. It adds noise 
    proportional to the clipping threshold and a noise multiplier.

    math:
        \theta \leftarrow \theta + \mathcal{N}(0, \sigma^2 I) 
        \text{ where } \sigma = \text{clip} \times \text{noise\_mult}

    Args:
        params (List[torch.Tensor]): List of model parameters to perturb.
        clip (float): The gradient clipping threshold (sensitivity).
        noise_mult (float): Multiplier controlling the privacy budget (epsilon).

    Returns:
        None: Modifies tensors in-place.
    """
    for p in params:
        # Only add noise to parameters that are being optimized
        if p.requires_grad:
            noise = torch.randn_like(p) * clip * noise_mult
            p.data.add_(noise)


# ============================================================ GRID FROM TENSORS
def grid_from_tensors(tensors: torch.Tensor, nrow: int = 8) -> np.ndarray:
    """
    Construct a visual grid from a batch of image tensors.

    Args:
        tensors (torch.Tensor): Input batch of images, range [-1, 1].
        nrow (int): Number of images per row in the grid. Defaults to 8.

    Returns:
        np.ndarray: The resulting grid image as a NumPy array (H, W, C).
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensors = (tensors + 1.0) / 2.0
    grid = make_grid(tensors, nrow=nrow, padding=2, normalize=False)
    return grid.permute(1, 2, 0).numpy()


# ============================================================ PLOT CURVES
def plot_curves(history: Dict[str, List[float]], out_path: Path) -> None:
    """
    Render and save training history plots (Loss and Accuracy).



[Image of deep learning training curves]


    Args:
        history (Dict[str, List[float]]): Dictionary mapping metric names to 
                                          lists of recorded values per epoch.
        out_path (Path): Filesystem path to save the generated plot.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ---------------- Plot Loss ----------------
    for key, values in history.items():
        if 'loss' in key:
            axes[0].plot(values, label=key)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # ---------------- Plot Accuracy ----------------
    for key, values in history.items():
        if 'acc' in key:
            axes[1].plot(values, label=key)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================ DECODER SIZE
def decoder_size_mb(decoder: Decoder) -> float:
    """
    Estimate the memory footprint of the Decoder model parameters.

    Args:
        decoder (Decoder): The decoder neural network module.

    Returns:
        float: Size of the model parameters in Megabytes (MB).
    """
    # Calculate total bytes: number of elements * size per element (usually 4 bytes for float32)
    return sum(p.numel() * p.element_size() for p in decoder.parameters()) / 1e6


# ============================================================ CSV PARSING
def parse_csv_or_single(value: str, dtype: type = str) -> List[Any]:
    """
    Parse a configuration string that may be a single value or a CSV list.

    Args:
        value (str): The input string (e.g., "1,2,3" or "1").
        dtype (type): The target type to cast each element to (e.g., int, float).

    Returns:
        List[Any]: A list of parsed values of type `dtype`.
    """
    if ',' in value:
        return [dtype(x.strip()) for x in value.split(',')]
    else:
        return [dtype(value.strip())]


def expand_grid(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate a Cartesian product of hyperparameters for grid search.

    Args:
        params (Dict[str, List[Any]]): Dictionary mapping parameter names to 
                                       lists of possible values.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a 
                              unique combination of parameters.
    """
    keys = params.keys()
    values = params.values()
    # Compute Cartesian product of all value lists
    combinations = list(itertools.product(*values))
    # Re-associate keys with combined values
    return [dict(zip(keys, combo)) for combo in combinations]


# ============================================================ CLASSIFIER EVALUATION
@torch.no_grad()
def evaluate_single_classifier(
        model: SimpleCNN, ld: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference with a single classifier to obtain ground truth and predictions.

    Args:
        model (SimpleCNN): The classifier model to evaluate.
        ld (DataLoader): The data loader containing the evaluation dataset.
        device (torch.device): Compute device (CPU/GPU).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Array of true labels.
            - Array of predicted labels (class indices).
    """
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in ld:
            y_true.append(y)
            # Compute argmax to get the most likely class index
            y_pred.append(model(x.to(device)).argmax(1).cpu())
    model.cpu()
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def ensemble_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the accuracy metric given aligned true and predicted labels.

    Args:
        y_true (np.ndarray): Array of ground truth labels.
        y_pred (np.ndarray): Array of predicted labels.

    Returns:
        float: Accuracy ratio (0.0 to 1.0).
    """
    return float((y_true == y_pred).mean())


# ============================================================ FAST I/O HELPERS
def _default_num_workers() -> int:
    """
    Determine an optimal number of DataLoader workers based on CPU core count.

    Returns:
        int: Recommended number of worker processes.
    """
    try:
        # Heuristic: Use all cores minus one to avoid starving the main process
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
    Efficiently convert a Dataset or Subset into a single contiguous Tensor.

    This function bypasses standard Python loops where possible by utilizing 
    a DataLoader with multi-processing to fetch batches, which are then concatenated.

    Args:
        subset (torch.utils.data.Dataset): The source dataset.
        batch_size (int): Batch size for internal loading. Defaults to 1024.
        num_workers (Optional[int]): Number of subprocesses. Defaults to auto-detect.
        pin_memory (bool): If True, pins memory for faster GPU transfer. Defaults to True.
        limit (Optional[int]): Maximum number of samples to load. Defaults to None.

    Returns:
        torch.Tensor: A single tensor containing the stacked data [N, ...].
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
    xs = []
    n_acc = 0
    for x, _ in loader:
        # Halt loading if the limit is reached
        if limit is not None and n_acc >= limit:
            break
        # Handle the case where the next batch exceeds the limit
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
    Convert an entire Dataset into a single tensor.

    Args:
        ds (torch.utils.data.Dataset): The source dataset.
        batch_size (int): Internal batch size for loading. Defaults to 1024.
        num_workers (Optional[int]): Number of worker processes.
        pin_memory (bool): Whether to pin memory.

    Returns:
        torch.Tensor: Contiguous tensor of the entire dataset.
    """
    return subset_to_tensor(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)


# ============================================================ ENSEMBLE PREDICTIONS
@torch.no_grad()
def ensemble_preds_poexp(
        classifiers: List[SimpleCNN], test_ld: DataLoader, device: torch.device
):
    """
    Perform Product of Experts (PoE) ensemble prediction.

    This method aggregates predictions by summing log-probabilities (equivalent 
    to multiplying probabilities) across multiple classifiers.

    math:
        \log P(y|x) \propto \sum_{i} \log P_i(y|x)

    Args:
        classifiers (List[SimpleCNN]): List of trained classifier models.
        test_ld (DataLoader): DataLoader for the test set.
        device (torch.device): Compute device.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Array of true labels.
            - Array of ensemble predicted labels.
    """
    for c in classifiers:
        c.to(device).eval()

    eps = 1e-12  # Numerical stability epsilon
    y_true, y_pred = [], []

    for x, y in test_ld:
        x, y = x.to(device), y.to(device)
        log_probs = None

        # Accumulate log-softmax probabilities from each model member
        for c in classifiers:
            # Clamp log probabilities to avoid -inf
            lp = torch.log_softmax(c(x), dim=1).clamp(min=np.log(eps))
            log_probs = lp if log_probs is None else log_probs + lp

        y_pred.append(log_probs.argmax(1).cpu())
        y_true.append(y.cpu())

    for c in classifiers:
        c.cpu()
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()