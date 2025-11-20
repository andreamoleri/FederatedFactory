"""
📊 PDF Report Graph Generation Package
-------------------------------------

This package aggregates and exposes the essential subroutines required to construct
the visual components of the final PDF report. It serves as the central entry point
for accessing distinct plotting modules, ensuring a modular and maintainable
architecture for report generation.

🧠 Purpose:
    To provide a unified interface for importing graph generation functions,
    abstracting the underlying file structure from the consumer of this package.

🔧 Core Functionalities:
    • Aggregation of disparate plotting modules (e.g., confusion matrices, training curves)
    • Exposure of a unified public API via the `__all__` directive
    • Management of figure export utilities for persistent storage
    • Provision of specific visualization routines for distinct report pages

🎯 Intended Use:
    • Master reporting scripts invoking page-specific generation functions
    • Analytical pipelines requiring specific metric visualizations
    • Automated documentation workflows

📁 Dependencies:
    • Internal submodules: `real_vs_synthetic`, `intensity_distributions`,
      `training_curves`, `confusion_matrix`, `model_summary`, `cost_metrics`,
      `generative_metrics`, `figure_export`, `client_class_distribution`,
      `classification_metrics`

Author: Andrea Moleri
File Location: src/reports/graphs/__init__.py
Last Modified: 20/11/2025
"""

# Import specific page generation functions from their respective submodules.
# These imports facilitate a cleaner namespace in the consuming scripts.

from .real_vs_synthetic import generate_real_vs_synthetic_page
from .intensity_distributions import generate_intensity_distributions_pages
from .training_curves import generate_training_curves_pages
from .confusion_matrix import generate_confusion_matrix_page
from .model_summary import generate_model_summary_page
from .cost_metrics import append_cost_pages
from .generative_metrics import generate_generative_metrics_pages

# Import figure export utilities to handle the saving and naming of generated plots.
from .figure_export import FigureExporter, init_figure_export, save_current_figure, get_figure_name
from .client_class_distribution import generate_client_class_distribution_pages

# Import the routine responsible for generating classification metric visualizations.
# This ensures that performance metrics are available for the final report.
from .classification_metrics import generate_classification_metrics_pages

# Define the public API of this package.
# This list restricts what is exported when `from src.reports.graphs import *` is used,
# preserving encapsulation and preventing namespace pollution.
__all__ = [
    'generate_real_vs_synthetic_page',
    'generate_intensity_distributions_pages',
    'generate_training_curves_pages',
    'generate_confusion_matrix_page',
    'generate_model_summary_page',
    'append_cost_pages',
    'generate_generative_metrics_pages',
    'FigureExporter',
    'init_figure_export',
    'save_current_figure',
    'get_figure_name',
    'generate_client_class_distribution_pages',
    'generate_classification_metrics_pages',
]