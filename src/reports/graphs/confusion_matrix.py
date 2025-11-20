"""
📊 Confusion Matrix Visualization Module
----------------------------------------

This module facilitates the generation, rendering, and exportation of confusion
matrices for classification model evaluation. It integrates with the Matplotlib
backend to append visualizations to multipage PDF reports or save them as
standalone image files.

🧠 Purpose:
    To provide a visual assessment of classifier performance by comparing
    predicted labels against ground truth values across defined classes.

🔧 Core Functionalities:
    • Compute confusion matrices using Scikit-learn utilities
    • Render high-resolution heatmaps with customizable layouts
    • Append visualizations to an existing PdfPages object
    • Export individual figures to a specified directory

🎯 Intended Use:
    • Machine learning evaluation pipelines
    • Automated reporting systems
    • Academic research and performance analysis

📁 Dependencies:
    • matplotlib
    • numpy
    • sklearn

📝 Notes:
    This module assumes the input arrays (y_true, y_pred) are compatible
    with Scikit-learn's metric functions.

Author: Andrea Moleri
File Location: src/reports/graphs/confusion_matrix.py
Last Modified: 20/11/2025
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve the project root directory to facilitate absolute imports
# This allows the script to be executed as a standalone module or part of a package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from logs.logger import get_logger

# Initialize module-level logger
logger = get_logger(__name__)

# Default dots per inch for figure rendering
_DEF_FIG_DPI = 300


def generate_confusion_matrix_page(
        pdf: PdfPages,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
        metrics: dict,
        figures_dir: Path = None
) -> int:
    """
    Generates a confusion matrix visualization and appends it to a PDF report.

    This function computes the confusion matrix based on true and predicted
    labels, renders it as a heatmap using Matplotlib, and saves the result
    to a PDF object. Optionally, it saves a standalone PNG image if a
    directory path is provided.

    Parameters
    ----------
    pdf : PdfPages
        The open Matplotlib PdfPages object used to aggregate report pages.
    y_true : np.ndarray
        The array of ground truth (correct) target values.
    y_pred : np.ndarray
        The array of estimated targets as returned by the classifier.
    num_classes : int
        The total number of distinct classes in the classification problem.
        Used to define the dimensions of the matrix.
    metrics : dict
        A dictionary containing calculated performance metrics. Must include
        the key 'accuracy' for title annotation.
    figures_dir : Path, optional
        The directory path where the standalone image file should be saved.
        If None, no individual image is generated.

    Returns
    -------
    int
        Returns 1 if the page was successfully generated and added to the PDF.
        Returns 0 if an exception occurred during the process.

    Raises
    ------
    Exception
        Captures generic exceptions during matrix calculation or plotting
        to prevent pipeline failure. Errors are logged via the module logger.
    """

    try:
        # Compute the confusion matrix using Scikit-learn
        # The 'labels' parameter ensures the matrix is square even if some classes are absent in the batch
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
        )

        # Initialize the Matplotlib figure and axis
        fig, ax = plt.subplots(
            figsize=(6, 6),
            dpi=_DEF_FIG_DPI,
        )

        # Configure the display object for the confusion matrix
        disp = ConfusionMatrixDisplay(
            cm,
            display_labels=list(range(num_classes)),
        )

        # Render the plot onto the created axis
        # 'values_format="d"' ensures counts are displayed as integers
        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            xticks_rotation="vertical" if num_classes > 20 else "horizontal",
            values_format="d",
        )

        # Disable grid lines to ensure text readability within the heatmap cells
        ax.grid(False)

        # Enforce an equal aspect ratio to maintain square cells
        ax.set_aspect("equal", adjustable="box")

        # Annotate the figure with the accuracy score derived from the metrics dictionary
        ax.set_title(
            f"Test Confusion Matrix (Accuracy = {metrics['accuracy']:.4f})"
        )

        # Adjust subplot parameters to give specified padding
        fig.tight_layout()

        # Export individual figure if a destination directory is provided
        if figures_dir:
            fig.savefig(figures_dir / "confusion_matrix.png",
                        dpi=_DEF_FIG_DPI, bbox_inches="tight",
                        facecolor='white', edgecolor='none')

        # Save the figure to the multipage PDF report
        pdf.savefig(fig, bbox_inches="tight")

        # Explicitly close the figure to free memory resources
        plt.close(fig)
        return 1

    except Exception as e:
        # Log the error trace without interrupting the broader execution flow
        logger.error(f"Error generating confusion matrix page: {e}")
        return 0