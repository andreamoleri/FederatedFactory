"""
📊 Real vs. Synthetic Comparison Module
---------------------------------------

This module is responsible for generating comparative visualisations between
ground truth (real) images and synthetically generated samples. It facilitates
qualitative assessment of generative models by producing side-by-side grid
layouts organised by class label.

🧠 Purpose:
    To support the qualitative evaluation of generative adversarial networks (GANs)
    or diffusion models by rendering "Real vs. Synthetic" comparison charts
    into PDF reports and standalone image files.

🔧 Core Functionalities:
    • Ingest and normalise synthetic images from directory structures.
    • Retrieve corresponding real images from the test dataset.
    • Construct subplot grids aligning real and synthetic samples by class.
    • Handle image tensor permutations (HWC to CHW) for PyTorch compatibility.
    • Export visualisations to multi-page PDFs and individual PNG files.

🎯 Intended Use:
    • Evaluation pipelines for generative models.
    • Automated reporting systems in research workflows.
    • Visual debugging of mode collapse or class-conditional generation issues.

📁 Dependencies:
    • matplotlib
    • numpy
    • torch
    • PIL (Pillow)

📝 Notes:
    The module assumes specific directory naming conventions (e.g., 'class-xxx')
    for synthetic data ingestion. Images are normalised to the range [-1, 1].

Author: Andrea Moleri
File Location: src/reports/graphs/real_vs_synthetic.py
Last Modified: 23/04/2025
"""

from __future__ import annotations

import sys
from pathlib import Path

# Establish the root directory relative to this file's location
# to ensure consistent import resolution across different execution contexts.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from imports.data_management import get_dataset
from utils import grid_from_tensors
from logs.logger import get_logger
from .figure_export import save_current_figure, get_figure_name

logger = get_logger(__name__)

# Default Dots Per Inch (DPI) for high-resolution figure export.
_DEF_FIG_DPI = 300


def _load_images_from_class_folders(base_dir: Path, classes: list) -> tuple[list, list]:
    """
    Traverse directory structures to load and pre-process synthetic images.

    This function searches for subdirectories matching specific class patterns,
    loads PNG images, converts them to RGB, and normalises their pixel values
    to the range [-1, 1].

    Parameters
    ----------
    base_dir : Path
        The root directory containing class-specific subfolders.
    classes : list
        A list of class identifiers (integers) to search for within the `base_dir`.

    Returns
    -------
    tuple[list, list]
        A tuple containing:
        - A list of pre-processed image arrays (numpy.ndarray).
        - A corresponding list of class labels (int) for each image.
        Returns empty lists if the directory does not exist.
    """
    images = []
    labels = []

    if not base_dir.exists():
        return images, labels

    for class_label in classes:
        # Define patterns to match directory names like 'class-001' or variants.
        class_pattern = f"class-{class_label:03d}*"
        class_folders = list(base_dir.glob(class_pattern))

        # Fallback pattern if the primary naming convention is not found.
        if not class_folders:
            class_pattern_alt = f"*{class_label}*"
            class_folders = [f for f in base_dir.glob(class_pattern_alt) if f.is_dir()]

        for class_folder in class_folders:
            if class_folder.is_dir():
                png_files = list(class_folder.glob("*.png"))
                for img_path in png_files:
                    try:
                        with Image.open(img_path) as im:
                            # Ensure consistent channel depth (3 channels for RGB).
                            if im.mode != 'RGB':
                                im = im.convert('RGB')

                            # Convert to float32 and normalise to [-1, 1].
                            # Formula: (pixel / 255.0) * 2.0 - 1.0
                            arr = np.asarray(im, dtype=np.float32) / 255.0
                            arr = (arr * 2.0) - 1.0
                            images.append(arr)
                            labels.append(class_label)
                    except Exception as e:
                        logger.warning(f"Error loading {img_path}: {e}")
                        continue

    return images, labels


def generate_real_vs_synthetic_page(
        pdf: PdfPages,
        out: Path,
        args,
        classes_present: list,
        test_set,
        tgt_all: np.ndarray,
        figures_dir: Path = None
) -> int:
    """
    Generate and save a comparison page displaying Real vs. Synthetic images.

    This function constructs a Matplotlib figure where each row corresponds to a
    specific class. The left column displays real samples from the test set,
    while the right column displays generated synthetic samples. Rows are only
    created for classes where synthetic data is available.

    Parameters
    ----------
    pdf : PdfPages
        The PDF object to which the generated figure will be appended.
    out : Path
        The output directory path containing the 'datasets/synthetic' subdirectory.
    args : Any
        Configuration arguments object (structure depends on implementation).
    classes_present : list
        A list of class labels to be considered for the report.
    test_set : Any
        The dataset object containing real images (subscriptable).
    tgt_all : np.ndarray
        An array of labels corresponding to the `test_set`, used to locate
        indices of specific classes.
    figures_dir : Path, optional
        If provided, the generated figure is also saved as a standalone PNG
        in this directory. Defaults to None.

    Returns
    -------
    int
        Returns 1 if a page was successfully generated and added to the PDF,
        otherwise returns 0 (e.g., if no synthetic images were found).
    """
    try:
        # Construct path to synthetic data and load all relevant images into memory.
        synthetic_base_dir = out / "datasets" / "synthetic"
        synthetic_images, synthetic_labels = _load_images_from_class_folders(
            synthetic_base_dir, classes_present
        )
        logger.info(
            f"Loaded {len(synthetic_images)} synthetic images for {len(set(synthetic_labels))} classes"
        )

        # Filter the request: retain only classes that possess at least one synthetic sample.
        classes_with_synth = [
            d for d in classes_present
            if any(lbl == d for lbl in synthetic_labels)
        ]

        if not classes_with_synth:
            logger.warning(
                f"No synthetic images found for the requested classes in {synthetic_base_dir}"
            )
            return 0

        # Optimize lookup by grouping images by their class label.
        from collections import defaultdict
        synth_by_class = defaultdict(list)
        for img, lbl in zip(synthetic_images, synthetic_labels):
            synth_by_class[lbl].append(img)

        # Initialise the subplot grid based on the number of valid classes.
        # Layout: n_rows x 2 columns (Column 0: Real, Column 1: Synthetic).
        n_rows = len(classes_with_synth)
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(6, 3 * n_rows),
            constrained_layout=True,
            dpi=_DEF_FIG_DPI,
        )

        # Ensure 'axes' is always a 2D array, even when n_rows is 1.
        # This standardises indexing logic (axes[row, col]).
        if n_rows == 1:
            axes = np.array([axes])

        for row, d in enumerate(classes_with_synth):
            # --- Column 0: REAL Images ---
            # Identify indices in the test set matching the current class 'd'.
            real_idxs = [i for i, lbl in enumerate(tgt_all) if lbl == d][:64]
            if real_idxs:
                # Stack tensors to create a batch for the grid generator.
                real_imgs = torch.stack([test_set[i][0] for i in real_idxs])
                axes[row, 0].imshow(grid_from_tensors(real_imgs))
            else:
                axes[row, 0].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[row, 0].set_title(f"Real {d}")
            axes[row, 0].axis("off")

            # --- Column 1: SYNTHETIC Images ---
            # Retrieve pre-loaded synthetic images for the current class.
            class_synth_images = synth_by_class[d]
            selected_images = class_synth_images[:64]

            synth_tensors = []
            for img in selected_images:
                try:
                    # Convert numpy arrays to PyTorch tensors.
                    # Matplotlib/PIL usually provide HWC (Height, Width, Channel).
                    # PyTorch expects CHW (Channel, Height, Width).
                    if img.ndim == 3 and img.shape[2] == 3:
                        tensor = torch.tensor(img).permute(2, 0, 1)  # HWC -> CHW
                    elif img.ndim == 2:
                        tensor = torch.tensor(img).unsqueeze(0)  # HW -> 1HW (Grayscale)
                    else:
                        tensor = torch.tensor(img)  # Fallback logic
                        if tensor.dim() == 2:
                            tensor = tensor.unsqueeze(0)
                    synth_tensors.append(tensor)
                except Exception as e:
                    logger.warning(f"Tensor conversion failed for class {d}: {e}")

            if synth_tensors:
                batch = torch.stack(synth_tensors)
                grid_img = grid_from_tensors(batch)
                axes[row, 1].imshow(grid_img)
            else:
                # Handling the edge case where filtering succeeded but conversion failed.
                axes[row, 1].text(
                    0.5, 0.5, "Tensor conversion\nfailed",
                    ha="center", va="center", fontsize=8
                )

            axes[row, 1].set_title(f"Synthetic {d}")
            axes[row, 1].axis("off")

        # Optional: Export the figure as a standalone PNG file.
        if figures_dir:
            fig.savefig(
                figures_dir / "real_vs_synthetic_comparison.png",
                dpi=_DEF_FIG_DPI,
                bbox_inches="tight",
                facecolor='white',
                edgecolor='none'
            )

        # Commit the figure to the multi-page PDF report.
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return 1

    except Exception as e:
        logger.error(f"Error generating real vs synthetic page: {e}")
        return 0