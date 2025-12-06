# src/utils/__init__.py
"""
📦 Utility Interface Module
---------------------------

This module serves as the public interface for the utilities package, strictly
controlling the namespace exposure of helper functions and classes. It centralizes
access to essential tools for reproducibility, visualization, loss calculation,
and model evaluation.

🧠 Purpose:
    To promote encapsulation and modularity by exposing only selected components
    from the internal `helpers` submodule, thereby simplifying the import structure
    for consuming modules.

🔧 Core Functionalities:
    • Reproducibility controls (seeding)
    • Perceptual loss mechanisms (VGG-based)
    • Visualization utilities (grid sampling, curve plotting)
    • Differential privacy primitives (noise injection)
    • Model evaluation metrics (accuracy, ensemble performance)

🎯 Intended Use:
    • Research experiments requiring consistent seeding and metrics
    • Model training loops needing perceptual loss
    • Post-processing and visualization pipelines

📁 Dependencies:
    • src.utils.helpers

Author: Andrea Moleri
File Location: src/utils/__init__.py
Last Modified: 06/12/2025
"""

from __future__ import annotations

# Import specific utilities from the internal helpers module to expose them
# as part of the public API of this package.
from .helpers import (
    set_seed,
    VGGPerceptualLoss,
    sample_grid,
    _dp_add_noise_,
    grid_from_tensors,
    plot_curves,
    decoder_size_mb,
    parse_csv_or_single,
    expand_grid,
    evaluate_single_classifier,  # Explicitly exposes the single classifier evaluation function
    ensemble_accuracy,           # Explicitly exposes the ensemble accuracy metric
)

# Define the public API of the module.
# This list restricts what symbols are exported when `from src.utils import *` is used,
# preventing namespace pollution with internal implementation details.
__all__ = [
    "set_seed",
    "VGGPerceptualLoss",
    "sample_grid",
    "_dp_add_noise_",
    "grid_from_tensors",
    "plot_curves",
    "decoder_size_mb",
    "parse_csv_or_single",
    "expand_grid",
    "evaluate_single_classifier",
    "ensemble_accuracy",
]