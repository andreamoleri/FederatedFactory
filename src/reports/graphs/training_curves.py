"""
📊 Training Curves Visualization Module
---------------------------------------

This module facilitates the generation, rendering, and exportation of training 
and validation performance metrics for machine learning models. It is specifically 
designed to handle federated or multi-client training histories, plotting 
comparative curves for loss and accuracy.

🧠 Purpose:
    To provide a standardised mechanism for visualising model convergence and 
    performance stability across epochs, supporting both Variational Autoencoders (VAE) 
    and classification architectures.

🔧 Core Functionalities:
    • Generate multi-series line plots for specified metrics (e.g., VAE loss, accuracy)
    • Aggregate individual client training histories into a single figure
    • Export visualisations to a consolidated multi-page PDF report
    • Optionally persist individual plots as high-resolution PNG files for publication

🎯 Intended Use:
    • Post-training analysis and reporting
    • Academic publication figure generation
    • Monitoring federated learning convergence

📁 Dependencies:
    • matplotlib (pyplot, backend_pdf)
    • pathlib
    • typing

📝 Notes:
    The plotting logic assumes a nested dictionary structure for history data, 
    where metrics are keyed by metric name, then by client identifier.

Author: Andrea Moleri
File Location: src/reports/graphs/training_curves.py
Last Modified: 20/11/2025
"""

from pathlib import Path
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Note: The dependency on external utilities has been deprecated. 
# Internal implementations are now used to ensure module self-containment.
# from utils import plot_curves

_DEF_FIG_DPI = 300


def _plot_curves(
        hist: Dict[str, Dict[int | str, List[float]]],
        key: str,
        title: str,
        ylabel: str,
        ax: plt.Axes,
) -> None:
    """
    Render a generic line plot for a specific training metric on a given Axes.

    This helper function iterates through all client or series data associated 
    with a specific metric key and plots them onto a shared coordinate system. 
    It standardises the visual styling, including labels and legends.

    Parameters
    ----------
    hist : Dict[str, Dict[int | str, List[float]]]
        The training history data structure. It is expected to be a dictionary 
        where keys are metric names (e.g., 'vae_loss'), and values are 
        dictionaries mapping client IDs to lists of floating-point metric values.
    key : str
        The specific metric identifier to look up within the `hist` dictionary.
    title : str
        The text string to be assigned as the title of the subplot.
    ylabel : str
        The label text for the Y-axis, representing the unit or type of measurement 
        (e.g., "Loss", "Accuracy").
    ax : plt.Axes
        The Matplotlib Axes object upon which the curves will be drawn.

    Returns
    -------
    None
        This function operates in-place on the provided Axes object.
    """
    # Iterate through the inner dictionary to retrieve each client's data series.
    # 'cid' represents the Client ID, and 'series' is the list of metric values per epoch.
    for cid, series in hist[key].items():
        ax.plot(series, label=str(cid))

    # Configure visual properties of the axes for clarity.
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)

    # Create a legend to distinguish between different clients/series.
    # The layout is forced to 5 columns to accommodate a potentially high number of clients.
    ax.legend(ncol=5, fontsize=6)


def generate_training_curves_pages(
        pdf: PdfPages,
        hist: Dict,
        figures_dir: Optional[Path] = None,
):
    """
    Orchestrate the creation and export of training curve figures.

    This function iterates through a predefined specification of metrics, 
    generates a plot for each available metric, saves it to a multi-page PDF, 
    and optionally exports it as a standalone PNG image.



[Image of machine learning training curves]


    Parameters
    ----------
    pdf : PdfPages
        An open `matplotlib.backends.backend_pdf.PdfPages` object used to 
        append figures to a multipage PDF report.
    hist : Dict
        The comprehensive dictionary containing training history data for 
        all metrics and clients.
    figures_dir : Optional[Path]
        The filesystem path where individual PNG files should be saved. 
        If None, PNG export is skipped.

    Returns
    -------
    None
        The function performs I/O operations (plotting and saving files) 
        and does not return a value.

    Raises
    ------
    IOError
        If there are permissions issues writing to `figures_dir`.
    """
    # Define the configuration for the curves to be plotted.
    # Each tuple contains: (dictionary_key, plot_title, y_axis_label).
    curve_specs = [
        ("vae_loss", "Factory Training Loss", "Loss"),
        ("clf_train_acc", "Classifier Training Accuracy", "Accuracy"),
        ("clf_val_acc", "Classifier Validation Accuracy", "Accuracy"),
        ("clf_train_loss", "Classifier Training Loss", "Loss"),
        ("clf_val_loss", "Classifier Validation Loss", "Loss"),
    ]

    for key, title, yl in curve_specs:
        # Gracefully skip metrics defined in specs but missing from the provided data.
        if key not in hist:
            continue

        # 
        # Initialise the figure and axes with specific dimensions and resolution.
        fig, ax = plt.subplots(
            figsize=(8, 4),
            dpi=_DEF_FIG_DPI,
        )

        # Delegate the actual plotting logic to the helper function.
        _plot_curves(hist, key, title, yl, ax)

        # Persist the current figure to the provided PDF document.
        # 'bbox_inches="tight"' ensures that labels are not cut off.
        pdf.savefig(fig, bbox_inches="tight")

        # If a directory is provided, export the figure as a separate PNG file.
        if figures_dir is not None:
            fig_path = figures_dir / f"training_curve_{key}.png"
            fig.savefig(fig_path, bbox_inches="tight", dpi=_DEF_FIG_DPI)

        # Explicitly close the figure to release memory resources, preventing
        # memory leaks when processing a large number of plots.
        plt.close(fig)