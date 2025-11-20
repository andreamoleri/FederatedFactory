"""
📊 Intensity Distribution Visualization Module
---------------------------------------------

This module provides a suite of functions to compute, analyze, and visualize the 
pixel intensity distributions of image datasets. It is specifically engineered to 
compare "real" (ground truth) data against "synthetic" (generated) data across 
different classes.

🧠 Purpose:
    Designed for evaluating the quality of generative models (e.g., GANs, Diffusion Models)
    by assessing how closely the statistical distribution of generated image intensities
    matches that of the training data.

🔧 Core Functionalities:
    • Convert multi-format image arrays (RGB, Grayscale, Tensors) to normalized luminance  
    • statistical aggregation of mean intensity per image across datasets  
    • Computation of Gaussian Probability Density Functions (PDF) for data fitting  
    • Generation of comparative histograms and PDF overlays using Matplotlib  

🎯 Intended Use:
    • Academic reporting on generative model fidelity  
    • Automated report generation pipelines  
    • Exploratory Data Analysis (EDA) of image datasets  

📁 Dependencies:
    • numpy  
    • matplotlib  
    • PIL (Pillow)  
    • torch  

📝 Notes:
    The module assumes input images can be converted to a [0, 1] float range. 
    It handles both folder-based storage (PNGs) and in-memory PyTorch datasets.

Author: Andrea Moleri  
File Location: src/reports/graphs/intensity_distributions.py  
Last Modified: 23/04/2025  
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is in the system path for module resolution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import math
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import torch

from imports.data_management import get_dataset
from logs.logger import get_logger

logger = get_logger(__name__)

_DEF_FIG_DPI = 300


def _to_gray01(arr: np.ndarray) -> np.ndarray:
    """
    Convert an input image array to a normalized grayscale float format in the range [0, 1].

    This function handles various input formats, including 2D (grayscale) and 3D (RGB/multichannel)
    arrays, as well as different data types (uint8, float). It applies standard luminance 
    weights for RGB to grayscale conversion.

    Parameters
    ----------
    arr : np.ndarray
        The input image array. Expected shapes are (H, W) or (H, W, C). 
        Supported types include uint8 and floating-point formats.

    Returns
    -------
    np.ndarray
        A 2D array representing the grayscale image with pixel values normalized 
        between 0.0 and 1.0.

    Notes
    -----
    - If the input has 3 channels, it uses the formula: Y = 0.299*R + 0.587*G + 0.114*B.
    - If the input is in the range [-1, 1], it is linearly scaled to [0, 1].
    """
    # Handle 2D arrays (already grayscale)
    if arr.ndim == 2:
        x = arr.astype(np.float32)
    # Handle 3D arrays with at least 3 channels (RGB)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        # Extract RGB channels and apply standard luminance coefficients
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        x = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        # Fallback for other 3D formats: average across the channel dimension
        x = arr.astype(np.float32)
        if arr.ndim == 3:
            x = x.mean(axis=2)

    # Normalize based on data type and range
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        x = np.asarray(x, dtype=np.float32)
        # Check if the data is in the range [-1, 1] (common in some ML preprocessing)
        if x.min() >= -1.001 and x.max() <= 1.001:
            x = (x + 1.0) * 0.5

        # Clip values to ensure strict [0, 1] compliance
        x = np.clip(x, 0.0, 1.0)
    return x


def _load_means_from_png_folder(folder: Path) -> np.ndarray:
    """
    Load all PNG images from a directory and compute the mean intensity per image.

    Parameters
    ----------
    folder : Path
        The directory path containing the PNG image files.

    Returns
    -------
    np.ndarray
        A 1D array containing the mean intensity value (float32) for each valid image found. 
        Returns an empty array if the folder does not exist or contains no valid images.
    """
    if not folder.exists():
        return np.empty((0,), dtype=np.float32)

    vals = []
    for p in sorted(folder.glob("*.png")):
        try:
            # Open image, ensure RGB mode, and convert to numpy array
            with Image.open(p) as im:
                im = im.convert("RGB")
                arr = np.asarray(im)
                # Convert to normalized grayscale
                gray = _to_gray01(arr)
                vals.append(float(gray.mean()))
        except Exception:
            # Gracefully skip unreadable files
            continue
    return np.asarray(vals, dtype=np.float32) if vals else np.empty((0,), dtype=np.float32)


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Calculate the Gaussian (Normal) Probability Density Function (PDF).

    Parameters
    ----------
    x : np.ndarray
        The input array of points where the PDF is evaluated.
    mu : float
        The mean (μ) of the distribution.
    sigma : float
        The standard deviation (σ) of the distribution.

    Returns
    -------
    np.ndarray
        The evaluated PDF values corresponding to the input `x`.

    Notes
    -----
    If `sigma` is non-positive, the function approximates a Dirac delta function 
    by returning a spike (1.0) at the closest value to `mu`.
    """
    if sigma <= 0:
        y = np.zeros_like(x, dtype=np.float64)
        y[np.argmin(np.abs(x - mu))] = 1.0
        return y

    # Standard Gaussian formula: (1 / (σ * sqrt(2π))) * exp(-0.5 * ((x - μ) / σ)^2)
    coef = 1.0 / (math.sqrt(2.0) * math.sqrt(math.pi) * sigma)
    return coef * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _summarize(values: np.ndarray) -> tuple[float, float, int]:
    """
    Compute descriptive statistics for an array of values.

    Parameters
    ----------
    values : np.ndarray
        The input array of numerical values.

    Returns
    -------
    tuple[float, float, int]
        A tuple containing:
        - Mean (μ)
        - Variance (σ²), calculated with Delta Degrees of Freedom = 1 (sample variance).
        - Count (N), the number of elements.
    """
    n = int(values.shape[0])
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = float(values.mean())
    var = float(values.var(ddof=1)) if n > 1 else 0.0
    return mu, var, n


def _class_dirs(root: Path, cls_id: int) -> List[Path]:
    """
    Locate subdirectories corresponding to a specific class ID.

    The function looks for directories matching the pattern "class-{cls_id}*".

    Parameters
    ----------
    root : Path
        The root directory to search within.
    cls_id : int
        The integer identifier of the class.

    Returns
    -------
    List[Path]
        A list of paths to directories matching the class pattern.
    """
    if not root.exists():
        return []
    pat = f"class-{cls_id:03d}*"
    return [p for p in root.glob(pat) if p.is_dir()]


def _real_means_from_dataset(test_set, tgt_all: np.ndarray, cls_id: int, max_imgs: int = 1000) -> np.ndarray:
    """
    Estimate the mean intensities for a specific class directly from a dataset object.

    This serves as a fallback when pre-extracted image folders are not available.

    Parameters
    ----------
    test_set : Dataset
        The dataset object (e.g., PyTorch Dataset) supporting indexing.
    tgt_all : np.ndarray
        An array containing labels/targets corresponding to the dataset indices.
    cls_id : int
        The class ID to filter for.
    max_imgs : int, optional
        The maximum number of images to process for this estimation (default is 1000).

    Returns
    -------
    np.ndarray
        A 1D array of mean intensity values for the selected class images.
    """
    # Filter indices that match the requested class ID, up to the limit
    idxs = [i for i, lbl in enumerate(tgt_all) if lbl == cls_id][:max_imgs]
    if not idxs:
        return np.empty((0,), dtype=np.float32)
    vals = []
    for i in idxs:
        try:
            # Access the dataset item; assumes tuple structure (image, label)
            x = test_set[i][0]
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()

            # Handle dimension ordering: (C, H, W) -> (H, W, C)
            if x.ndim == 3 and x.shape[0] in (1, 3):
                arr = np.transpose(x, (1, 2, 0))
            else:
                arr = x

            arr = np.asarray(arr, dtype=np.float32)

            # Normalize if values exceed standard range
            if arr.max() > 1.0:
                arr = arr / 255.0

            gray = _to_gray01(arr)
            vals.append(float(gray.mean()))
        except Exception:
            continue
    return np.asarray(vals, dtype=np.float32) if vals else np.empty((0,), dtype=np.float32)


def generate_intensity_distributions_pages(
        pdf: PdfPages,
        out: Path,
        classes_present: list,
        test_set,
        tgt_all: np.ndarray,
        figures_dir: Path = None
) -> int:
    """
    Generate and save per-class mean-intensity distribution pages to a PDF.

    This function orchestrates the comparison between real and synthetic image data.
    It computes histograms and Gaussian fits for image intensities and plots them.

    Logic:
    - Iterates through each class.
    - Loads synthetic data; if absent, the class is skipped (no plot).
    - Loads real data (from folders or dataset fallback).
    - Plots overlapping histograms and Gaussian curves.

    Parameters
    ----------
    pdf : PdfPages
        The Matplotlib PDF backend object to save the plots.
    out : Path
        The base output directory containing 'datasets/real' and 'datasets/synthetic'.
    classes_present : list
        A list of class IDs to process.
    test_set : Dataset
        The source dataset object (for fallback data loading).
    tgt_all : np.ndarray
        The array of target labels for the dataset.
    figures_dir : Path, optional
        If provided, individual PNG plots for each class will be saved here.

    Returns
    -------
    int
        The number of pages (plots) successfully added to the PDF.
    """
    pages_added = 0
    try:
        base_ds_dir = out / "datasets"
        real_root = base_ds_dir / "real"
        synth_root = base_ds_dir / "synthetic"

        for cls_id in classes_present:
            # --- 1) Calculate synthetic statistics FIRST; if absent, skip the class ---
            synth_dirs = _class_dirs(synth_root, cls_id)
            synth_vals_list = [_load_means_from_png_folder(d) for d in synth_dirs]
            # Flatten the list of arrays into a single array
            syn_vals = np.concatenate([v for v in synth_vals_list if v.size]) if synth_vals_list else np.empty((0,),
                                                                                                               dtype=np.float32)
            n_s = int(syn_vals.shape[0])

            if n_s == 0:
                logger.info(f"Skip class {cls_id}: no synthetic images found in {synth_root}")
                continue  # No graph generated without synthetic data

            # --- 2) Calculate real statistics (only now that we know we will draw) ---
            real_dirs = _class_dirs(real_root, cls_id)
            if real_dirs:
                real_vals_list = [_load_means_from_png_folder(d) for d in real_dirs]
                real_vals = np.concatenate([v for v in real_vals_list if v.size]) if real_vals_list else np.empty((0,),
                                                                                                                  dtype=np.float32)
            else:
                # Fallback to extraction from the dataset object if folders are missing
                real_vals = _real_means_from_dataset(test_set, tgt_all, cls_id)

            # If real data is also absent, plot anyway (showing synthetic only)
            n_r = int(real_vals.shape[0])

            # --- 3) Setup histograms and Gaussian curves ---
            bins = 20
            x_min, x_max = 0.0, 1.0
            edges = np.linspace(x_min, x_max, bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            bin_w = edges[1] - edges[0]

            # Calculate statistical moments (Mean and Variance)
            mu_r, var_r, _ = _summarize(real_vals)
            mu_s, var_s, _ = _summarize(syn_vals)

            # Compute histogram counts
            cnt_r, _ = np.histogram(real_vals, bins=edges) if n_r else (np.zeros(bins), edges)
            cnt_s, _ = np.histogram(syn_vals, bins=edges)

            # Compute Gaussian PDF curves for plotting
            xs = np.linspace(x_min, x_max, 800)
            # Scale PDF by count * bin_width to match histogram scale
            ys_r = _gaussian_pdf(xs, mu_r, math.sqrt(var_r)) * (n_r * bin_w) if n_r else np.zeros_like(xs)
            ys_s = _gaussian_pdf(xs, mu_s, math.sqrt(var_s)) * (n_s * bin_w)

            # --- 4) Plotting ---
            cmap = plt.get_cmap("tab10")
            base_color = cmap(int(cls_id) % 10)

            def _mix_with_white(color, mix=0.55):
                """Helper to lighten a color by mixing it with white."""
                r, g, b = plt.matplotlib.colors.to_rgb(color)
                v = np.array([r, g, b], dtype=float)
                out = (1.0 - mix) * v + mix * np.array([1.0, 1.0, 1.0])
                return tuple(out.tolist())

            synth_color = _mix_with_white(base_color, 0.55)

            fig, ax = plt.subplots(figsize=(9, 5.5), dpi=_DEF_FIG_DPI)

            if n_r:
                ax.bar(
                    centers, cnt_r, width=bin_w * 0.90, align="center",
                    color=base_color, edgecolor=base_color, linewidth=0.8, alpha=0.95,
                    label=f"Real (n={n_r})",
                )

            # Always present (we already checked n_s > 0)
            ax.bar(
                centers, cnt_s, width=bin_w * 0.65, align="center",
                color=synth_color, edgecolor=synth_color, linewidth=0.8, alpha=0.95,
                label=f"Synthetic (n={n_s})",
            )

            if n_r:
                ax.plot(xs, ys_r, linewidth=2.0, color=base_color, linestyle="-",
                        label=f"Real Gaussian μ={mu_r:.3f}, σ²={var_r:.4f}")

            ax.plot(xs, ys_s, linewidth=2.0, color=synth_color, linestyle="-",
                    label=f"Synth Gaussian μ={mu_s:.3f}, σ²={var_s:.4f}")

            ax.set_xlim(x_min, x_max)
            ax.set_xlabel("Per-image mean intensity (μ center, σ² width)")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"Class {cls_id} — Mean-intensity Distributions")
            ax.legend(frameon=False)
            ax.grid(alpha=0.25, linewidth=0.6)

            # Export individual figure if requested
            if figures_dir:
                fig.savefig(
                    figures_dir / f"intensity_distribution_class_{cls_id}.png",
                    dpi=_DEF_FIG_DPI, bbox_inches="tight",
                    facecolor='white', edgecolor='none'
                )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        return pages_added

    except Exception as e:
        logger.error(f"Error generating intensity distribution pages: {e}")
        return pages_added