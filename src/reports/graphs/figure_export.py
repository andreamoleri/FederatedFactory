"""
📊 Figure Export Utility Module
-----------------------------

This module serves as a dedicated utility for managing the filesystem export
of Matplotlib figures. It encapsulates the logic required to save visualizations
with consistent styling, resolution, and naming conventions.

🧠 Purpose:
    To provide a standardized interface for persisting graphical outputs generated
    during reporting or analysis pipelines, ensuring high-quality, print-ready
    artifacts.

🔧 Core Functionalities:
    • Management of a global singleton exporter instance for module-wide access
    • Automatic sequential naming of output files to preserve generation order
    • Configuration of export parameters (DPI, bounding boxes, background color)
    • Safe directory handling and path resolution

🎯 Intended Use:
    • Automated reporting pipelines
    • Data science research notebooks requiring reproducible outputs
    • Batch processing of analytical graphs

📁 Dependencies:
    • matplotlib
    • pathlib
    • sys

📝 Notes:
    The module utilizes a global state pattern (`_global_exporter`) to simplify
    access across different parts of the application without passing configuration
    objects explicitly.

Author: Andrea Moleri
File Location: src/reports/graphs/figure_export.py
Last Modified: 20/11/2025
"""

from __future__ import annotations

import sys
from pathlib import Path

# Dynamically resolve the project root directory to ensure local modules are
# importable regardless of the execution context. This manipulates the system
# path to include the parent directory structure.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional

# Default resolution for exported figures, set to 300 DPI to ensure
# high-quality output suitable for academic publications or print media.
_DEF_FIG_DPI = 300


class FigureExporter:
    """
    Helper class to export individual figures to files.

    This class manages the state related to the export directory and maintains
    a counter to ensure sequential naming of generated files.
    """

    def __init__(self, figures_dir: Optional[Path] = None):
        """
        Initialize the FigureExporter instance.

        Parameters
        ----------
        figures_dir : Optional[Path]
            The target directory where figure files will be saved. If None,
            export operations will be skipped.
        """
        self.figures_dir = figures_dir
        self.figure_counter = 0

    def save_figure(self, fig, filename: str, dpi: int = _DEF_FIG_DPI) -> None:
        """
        Save an individual Matplotlib figure to the file system.

        This method handles the construction of the full file path and applies
        standardized formatting settings (e.g., whitespace cropping, background
        color) during the save operation.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The Matplotlib figure object to be saved.
        filename : str
            The specific name of the file (including extension) to write.
        dpi : int, optional
            The resolution in dots per inch (default is 300).

        Returns
        -------
        None
        """
        # Proceed only if a valid directory is configured and exists physically on the disk.
        if self.figures_dir and self.figures_dir.exists():
            filepath = self.figures_dir / filename

            # Save the figure with specific parameters:
            # - bbox_inches="tight": Removes excess whitespace around the plot.
            # - facecolor='white': Ensures a white background instead of transparent.
            # - edgecolor='none': Removes the border around the figure canvas.
            fig.savefig(
                filepath,
                dpi=dpi,
                bbox_inches="tight",
                facecolor='white',
                edgecolor='none',
                format='png'  # Enforces PNG format for lossless raster compression.
            )

            # Increment the internal counter to track the number of exported figures.
            self.figure_counter += 1

    def get_next_figure_name(self, base_name: str) -> str:
        """
        Generate a sequential filename for a figure.

        Constructs a filename prefixed with a zero-padded counter (e.g.,
        "001_analysis.png"). This ensures that files sort correctly in
        alphabetical listings, reflecting their generation order.

        Parameters
        ----------
        base_name : str
            The semantic part of the filename (e.g., "confusion_matrix").

        Returns
        -------
        str
            The formatted filename string including the sequence prefix and extension.
        """
        return f"{self.figure_counter:03d}_{base_name}.png"


# Create a global exporter instance to be shared across the module context.
# This acts as a singleton placeholder, initialized via `init_figure_export`.
_global_exporter = None


def init_figure_export(figures_dir: Path) -> None:
    """
    Initialize the global figure exporter instance.

    This function sets up the module-level singleton, allowing subsequent calls
    to `save_current_figure` to function correctly with the specified directory.

    Parameters
    ----------
    figures_dir : Path
        The directory path where all figures should be exported.

    Returns
    -------
    None
    """
    global _global_exporter
    _global_exporter = FigureExporter(figures_dir)


def save_current_figure(fig, name: str) -> None:
    """
    Save the provided figure using the initialized global exporter.

    This is a convenience wrapper around the `FigureExporter.save_figure` method,
    utilizing the global singleton instance.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object to save.
    name : str
        The filename to assign to the figure.

    Returns
    -------
    None

    Notes
    -----
    If `init_figure_export` has not been called prior to this function,
    the operation is silently ignored.
    """
    if _global_exporter:
        _global_exporter.save_figure(fig, name)


def get_figure_name(base_name: str) -> str:
    """
    Retrieve the next sequential filename using the global exporter.

    If the global exporter is initialized, it returns a numbered filename.
    Otherwise, it falls back to returning the base name with a default extension.

    Parameters
    ----------
    base_name : str
        The descriptive name for the figure.

    Returns
    -------
    str
        The potentially prefixed and formatted filename.
    """
    if _global_exporter:
        return _global_exporter.get_next_figure_name(base_name)
    return f"{base_name}.png"