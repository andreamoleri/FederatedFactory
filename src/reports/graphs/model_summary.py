"""
📊 Model Summary Report Module
-----------------------------

This module provides utilities for generating a concise, human-readable
graphical summary page of a model run and exporting it into a PDF report.

🧠 Purpose:
    • Aggregate key configuration parameters and evaluation metrics into a
      single, self-contained summary page.
    • Facilitate reproducibility and rapid inspection of experiment settings
      and performance outcomes.
    • Support automated reporting pipelines in research and production
      workflows.

🔧 Core Functionalities:
    • Construct a matplotlib figure containing textual information about a
      specific model run.
    • Persist the summary figure as a page in a multi-page PDF file.
    • Optionally export the summary figure as a standalone image file for
      downstream use.

🎯 Intended Use:
    • Academic and industrial experiments where each training or inference
      run should be documented in a report.
    • Batch reporting scripts that compile multiple model runs into a single
      PDF artifact.
    • Teaching materials and demonstration notebooks that illustrate how to
      programmatically generate experiment summaries.

📁 Dependencies:
    • matplotlib
    • logs.logger (project-specific logging utility)
    • pathlib
    • sys (for dynamic path manipulation)

📝 Notes:
    • The function defined in this module assumes that the `args` object
      exposes attributes such as `dataset`, `latent_dim`, `dp`, `infer_mode`,
      and `model`.
    • All exceptions raised during figure creation are caught and logged
      internally; the calling code receives an integer return code indicating
      success or failure.

Author: Andrea Moleri
File Location: src/reports/graphs/model_summary.py
Last Modified: 13/11/2025
"""

from __future__ import annotations

import sys
from pathlib import Path

# Dynamically set the project root directory one level above this file and
# ensure it is present in sys.path so that intra-project imports succeed
# even when the module is executed as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

from logs.logger import get_logger

# Module-level logger used for reporting errors during summary page generation.
logger = get_logger(__name__)

# Default dots-per-inch (DPI) value for figures exported by this module.
_DEF_FIG_DPI = 300


def generate_model_summary_page(
        pdf: PdfPages,
        args,
        metrics: dict,
        figures_dir: Path = None
) -> int:
    """
    Generate a single-page graphical summary of a model run and append it to
    an existing PDF report.

    The summary page encodes key configuration values (e.g., dataset, latent
    dimensionality, differential privacy flag, inference mode, and model
    identifier) together with an accuracy metric into a centered text block
    rendered on a matplotlib figure. The figure is then saved as a new page
    in the provided `PdfPages` object. Optionally, the same figure can also
    be exported as a standalone PNG file.

    Parameters
    ----------
    pdf : matplotlib.backends.backend_pdf.PdfPages
        An open `PdfPages` instance to which the summary page will be
        appended. The caller is responsible for managing the lifecycle of
        this object (i.e., opening and closing the underlying file).
    args :
        A configuration-like object (for example, an `argparse.Namespace` or
        any object with attribute access) providing the following attributes:
        - dataset: str
            Name of the dataset; will be converted to upper case for display.
        - latent_dim: int
            Dimensionality of the latent space.
        - dp: bool
            Flag indicating whether differential privacy (DP) was enabled.
        - infer_mode: str
            Inference mode identifier; displayed in upper case.
        - model: str
            Model name or identifier; displayed in upper case.
    metrics : dict
        Mapping of metric names to numeric values. This function expects an
        `"accuracy"` key whose value is a floating-point number representing
        the accuracy score to be displayed.
    figures_dir : pathlib.Path, optional
        Directory path in which to export an individual PNG file named
        `"model_summary.png"` containing the generated figure. If `None`,
        the standalone PNG export is skipped. The directory is assumed to
        exist; creation is not performed by this function.

    Returns
    -------
    int
        Number of pages appended to the `pdf` object:
        - `1` if the summary page is successfully generated and saved.
        - `0` if an exception occurs during processing; the error is logged.

    Exceptions
    ----------
    This function catches and logs all exceptions internally. Under normal
    circumstances, no exceptions are propagated to the caller. Instead, a
    return value of `0` indicates that an error occurred.

    """
    try:
        # Create a new figure and single axes object sized for a compact
        # summary page. DPI is fixed for consistency across reports.
        fig, ax = plt.subplots(
            figsize=(6, 4),
            dpi=_DEF_FIG_DPI,
        )
        # Hide axes spines, ticks, and background so only the text summary
        # is visible in the rendered figure.
        ax.axis("off")
        # Compose and render the summary text at the center of the figure.
        # The text includes dataset, latent dimension, accuracy, DP status,
        # inference mode, and model identifier.
        ax.text(
            0.5,
            0.5,
            (
                f"Dataset: {args.dataset.upper()}  |  Latent: {args.latent_dim}"
                f"\nAccuracy: {metrics['accuracy']:.4f}"
                f"\nDP: {'ON' if args.dp else 'OFF'}"
                f"  |  Mode: {args.infer_mode.upper()}"
                f"  |  Model: {args.model.upper()}"
            ),
            ha="center",
            va="center",
            fontsize=12,
        )

        # Optionally export the figure as an individual PNG artifact in
        # addition to embedding it into the PDF report.
        if figures_dir:
            fig.savefig(figures_dir / "model_summary.png",
                        dpi=_DEF_FIG_DPI, bbox_inches="tight",
                        facecolor='white', edgecolor='none')

        # Append the generated figure as a new page in the provided PDF and
        # immediately close the figure to release associated resources.
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return 1

    except Exception as e:
        # Log a generic error message together with the exception details;
        # the function reports failure to the caller via the return value.
        logger.error(f"Error generating model summary page: {e}")
        return 0
