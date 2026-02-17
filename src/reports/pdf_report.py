"""
📊 PDF Report Generation Module
-------------------------------

This module provides the functionality to orchestrate the generation of comprehensive
PDF reports and individual figure exports for machine learning experiments. It aggregates
metrics, visualizations, and configuration data to produce a unified document summarizing
experiment performance.

🧠 Purpose:
    Serves as the presentation layer of the experimentation pipeline, automating the
    transformation of raw logs, CSVs, and model outputs into publication-ready
    visualizations and structured reports.

🔧 Core Functionalities:
    • Recursive discovery of experiment directories (permissive search)
    • Aggregation of training history, classification metrics, and generative metrics
    • Orchestration of matplotlib-based page generators
    • Production of multi-page PDF reports and extraction of raw PNG figures

🎯 Intended Use:
    • Post-training analysis to summarize results
    • Automated reporting for batch experiments
    • Visualization of model performance (confusion matrices, loss curves, etc.)

📁 Dependencies:
    • matplotlib (for plotting and PDF backend)
    • numpy (for numerical data handling)
    • torch/torchvision (for data transformations)
    • internal modules (graphs, logs, data_management)

📝 Notes:
    The module supports both a library mode (called by other scripts) and a
    standalone CLI mode (execution via `main`). It relies on a specific directory
    structure (metrics/, datasets/, args.json) to reconstruct experiment states.

Author: Andrea Moleri
File Location: src/reports/pdf_report.py
Last Modified: 24/01/2026
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------------------------
# Add project root (src/) to sys.path to enable absolute imports of sibling modules.
# This resolves the directory structure: .../src/reports/ -> .../src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import shutil
from typing import Dict, List, Tuple, Any, Set

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Logging Configuration
from logs.logger import get_logger
from logs import messages as logmsg

logger = get_logger(__name__)

# Project Dependencies
from imports.data_management import DATASET_META, get_dataset
from utils import grid_from_tensors

# Data Augmentation Imports
from imports.data_augmentation import build_transform

# Graph Generation Modules
# These functions are responsible for rendering specific pages within the PDF report.
from graphs import (
    generate_real_vs_synthetic_page,
    generate_intensity_distributions_pages,
    generate_training_curves_pages,
    generate_confusion_matrix_page,
    generate_model_summary_page,
    append_cost_pages,
    generate_generative_metrics_pages,
    generate_client_class_distribution_pages,
    generate_classification_metrics_pages,
)

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
_DEF_FIG_DPI = 300


def _get_meta_channels(ds_name: str) -> int | None:
    """
    Retrieve the number of image channels for a given dataset name.

    This function attempts to resolve the channel count by checking the
    global dataset metadata. It handles both exact matches and base name
    matches (e.g., stripping parentheses or colons).
    """
    # 1. Attempt exact match
    if ds_name in DATASET_META:
        val = DATASET_META[ds_name].get("channels")
        if val is not None:
            return int(val)

    # 2. Attempt base name match (e.g., "medmnist:bloodmnist" -> "medmnist")
    base = ds_name.split("(", 1)[0].split(":", 1)[0].strip().lower()
    
    if base in DATASET_META:
        val = DATASET_META[base].get("channels")
        if val is not None:
            return int(val)

    # 3. Heuristic Fallback
    # If the metadata explicitly says None (common in MedMNIST before init),
    # or if the key wasn't found, default to 3 for visualization purposes.
    if "mnist" in base and "med" in base:
         return 3 
    if "isic" in base or "camelyon" in base:
        return 3

    return 3 # Safe default for visualization


def _get_targets(test_set: Any) -> np.ndarray:
    """
    Extract target labels from a dataset object, normalizing diverse implementations.

    This function abstracts the differences between various PyTorch dataset
    implementations, which may store labels in attributes named 'targets'
    or 'labels', or require iteration to extract.

    Args:
        test_set (Any): The dataset object (e.g., torchvision Dataset or custom wrapper).

    Returns:
        np.ndarray: A 1-dimensional array of integer labels.
    """
    # Check for standard attribute names used in torchvision and custom datasets
    if hasattr(test_set, "targets"):
        tgt_all = test_set.targets
    elif hasattr(test_set, "labels"):
        tgt_all = test_set.labels
    else:
        # Fallback: Iterate through the dataset to collect labels (computationally expensive)
        tgt_all = [lbl for _, lbl in test_set]

    tgt_all = np.asarray(tgt_all)

    # Flatten dimensionality if necessary (e.g., convert [N, 1] to [N])
    if tgt_all.ndim > 1:
        tgt_all = tgt_all[:, 0]

    return tgt_all


def _adjust_emnist_labels(args: argparse.Namespace, tgt_all: np.ndarray) -> np.ndarray:
    """
    Normalize EMNIST labels to a zero-based index if required.

    Args:
        args (argparse.Namespace): Runtime arguments containing dataset information.
        tgt_all (np.ndarray): The array of target labels.

    Returns:
        np.ndarray: The adjusted array of labels.
    """
    # EMNIST letters class splits often start at index 1; shift to 0 for consistency
    if args.dataset.startswith("emnist") and tgt_all.min() != 0:
        tgt_all = tgt_all - int(tgt_all.min())
    return tgt_all


def export_individual_figures(
    out: Path,
    args: argparse.Namespace,
    hist: Dict,
    metrics: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> None:
    """
    Render and save individual figure files (PNG) to a dedicated directory.

    Args:
        out (Path): The root output directory for the experiment.
        args (argparse.Namespace): Experiment configuration arguments.
        hist (Dict): Training history dictionary (loss, accuracy, etc.).
        metrics (Dict): Dictionary containing computed metrics.
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        num_classes (int): Total number of classes in the dataset.
    """
    logger.info("Exporting individual figures to figures/ folder")

    # Ensure output directory structure exists
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Configure data transformations
    base_tfm = build_transform(args.dataset)
    channels = _get_meta_channels(args.dataset)

    # Fallback channel inference: Inspect the first element of the dataset
    if channels is None:
        try:
            tmp_ds = get_dataset(args.dataset, args.data_dir, False, base_tfm)
            x0 = tmp_ds[0][0]
            channels = int(getattr(x0, "shape", [None, None, None])[0] or 3)
        except Exception:
            channels = 3  # Safe default fallback

    # Apply grayscale conversion if requested and input is RGB
    if bool(getattr(args, "grayscale", False)) and channels == 3:
        tfm = T.Compose([T.Grayscale(num_output_channels=3), base_tfm])
    else:
        tfm = base_tfm

    gm = metrics.get("gen_metrics", {})

    # Enforce consistent scientific visualization styling
    from viz_style import use_nature_style
    use_nature_style()

    # Initialize a dummy PDF. This is necessary because the 'generate_*' functions
    # are designed to accept a PdfPages object, even if we are primarily saving PNGs.
    with PdfPages(out / "dummy.pdf") as pdf:

        # -------------------------------------------------------------
        # Data Preparation: Real vs Synthetic samples
        # -------------------------------------------------------------
        try:
            test_set = get_dataset(args.dataset, args.data_dir, False, tfm)
            tgt_all = _get_targets(test_set)
            tgt_all = _adjust_emnist_labels(args, tgt_all)
            classes_present = sorted(np.unique(tgt_all))

            generate_real_vs_synthetic_page(
                pdf, out, args, classes_present, test_set, tgt_all, figures_dir
            )
        except Exception as e:
            # MedMNIST often fails here if raw .npz not found or folders missing.
            # We catch it so we can still print the other charts.
            logger.warning(f"Could not load dataset/images for visuals ({e}). Skipping Real/Synth page.")
            test_set = None
            tgt_all = None
            classes_present = []

        # -------------------------------------------------------------
        # Classification Metrics
        # -------------------------------------------------------------
        try:
            generate_classification_metrics_pages(pdf, out, figures_dir)
        except Exception as e:
            logger.warning(f"Classification metrics pages skipped due to error: {e}")

        # -------------------------------------------------------------
        # Intensity Distributions
        # -------------------------------------------------------------
        if test_set is not None:
            generate_intensity_distributions_pages(
                pdf, out, classes_present, test_set, tgt_all, figures_dir
            )

        # -------------------------------------------------------------
        # Client Class Distribution
        # -------------------------------------------------------------
        generate_client_class_distribution_pages(
            pdf, out, figures_dir, csv_rel_path="distributions/client_class_distribution.csv"
        )

        # -------------------------------------------------------------
        # Training Dynamics (Loss/Accuracy)
        # -------------------------------------------------------------
        generate_training_curves_pages(pdf, hist, figures_dir)

        # -------------------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------------------
        generate_confusion_matrix_page(
            pdf, y_true, y_pred, num_classes, metrics, figures_dir
        )

        # -------------------------------------------------------------
        # Model Summary
        # -------------------------------------------------------------
        generate_model_summary_page(pdf, args, metrics, figures_dir)

        # -------------------------------------------------------------
        # Computational Costs
        # -------------------------------------------------------------
        try:
            append_cost_pages(pdf, out, figures_dir)
        except Exception as e:
            logger.warning(f"Cost/Compute pages skipped due to error: {e}")

        # -------------------------------------------------------------
        # Generative Metrics (Conditional)
        # -------------------------------------------------------------
        if gm:
            generate_generative_metrics_pages(pdf, gm, args, figures_dir)

    # Cleanup: Remove the temporary artifact
    (out / "dummy.pdf").unlink(missing_ok=True)

    logger.info(f"Individual figures exported to {figures_dir}")


def generate_pdf_report(
    out: Path,
    args: argparse.Namespace,
    hist: Dict,
    metrics: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    export_individual: bool = False
) -> None:
    """
    Orchestrate the generation of the complete PDF report.

    Args:
        out (Path): The output directory for the report.
        args (argparse.Namespace): Experiment arguments.
        hist (Dict): Training history data.
        metrics (Dict): Evaluated metrics.
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        num_classes (int): Number of classes.
        export_individual (bool): If True, also saves individual PNG figures alongside the PDF.
    """
    from viz_style import use_nature_style
    use_nature_style()

    logger.info(logmsg.GENERATING_PDF)

    # Configure transforms based on dataset metadata
    base_tfm = build_transform(args.dataset)
    channels = _get_meta_channels(args.dataset)

    # Fallback channel inference
    if channels is None:
        try:
            tmp_ds = get_dataset(args.dataset, args.data_dir, False, base_tfm)
            x0 = tmp_ds[0][0]
            channels = int(getattr(x0, "shape", [None, None, None])[0] or 3)
        except Exception:
            channels = 3  # Safe default fallback

    if bool(getattr(args, "grayscale", False)) and channels == 3:
        tfm = T.Compose([T.Grayscale(num_output_channels=3), base_tfm])
    else:
        tfm = base_tfm

    gm = metrics.get("gen_metrics", {})

    # Define output path
    pdf_path = out / "report.pdf"

    # Prepare directory for individual figures if requested
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the Multi-Page PDF Context
    with PdfPages(pdf_path) as pdf:

        # -------------------------------------------------------------
        # Dataset Preparation
        # -------------------------------------------------------------
        try:
            test_set = get_dataset(args.dataset, args.data_dir, False, tfm)
            tgt_all = _get_targets(test_set)
            tgt_all = _adjust_emnist_labels(args, tgt_all)
            classes_present = sorted(np.unique(tgt_all))

            # -------------------------------------------------------------
            # Visualisation: Real vs Synthetic Samples
            # -------------------------------------------------------------
            generate_real_vs_synthetic_page(
                pdf, out, args, classes_present, test_set, tgt_all, figures_dir if export_individual else None
            )
        except Exception as e:
            logger.warning(f"Could not load dataset for visuals ({e}). Skipping Real/Synth page.")
            test_set = None
            tgt_all = None
            classes_present = []

        # -------------------------------------------------------------
        # Visualisation: Classification Metrics
        # -------------------------------------------------------------
        try:
            generate_classification_metrics_pages(
                pdf, out, figures_dir if export_individual else None
            )
        except Exception as e:
            logger.warning(f"Classification metrics pages skipped due to error: {e}")

        # -------------------------------------------------------------
        # Visualisation: Intensity Distributions
        # -------------------------------------------------------------
        if test_set is not None:
            generate_intensity_distributions_pages(
                pdf, out, classes_present, test_set, tgt_all, figures_dir if export_individual else None
            )

        # -------------------------------------------------------------
        # Visualisation: Client Class Distribution
        # -------------------------------------------------------------
        generate_client_class_distribution_pages(
            pdf, out, figures_dir if export_individual else None,
            csv_rel_path="distributions/client_class_distribution.csv"
        )

        # -------------------------------------------------------------
        # Visualisation: Training Curves
        # -------------------------------------------------------------
        generate_training_curves_pages(
            pdf, hist, figures_dir if export_individual else None
        )

        # -------------------------------------------------------------
        # Visualisation: Confusion Matrix
        # -------------------------------------------------------------
        generate_confusion_matrix_page(
            pdf, y_true, y_pred, num_classes, metrics, figures_dir if export_individual else None
        )

        # -------------------------------------------------------------
        # Summary Page
        # -------------------------------------------------------------
        generate_model_summary_page(
            pdf, args, metrics, figures_dir if export_individual else None
        )

        # -------------------------------------------------------------
        # Visualisation: Cost & Compute Metrics
        # -------------------------------------------------------------
        try:
            append_cost_pages(
                pdf, out, figures_dir if export_individual else None
            )
        except Exception as e:
            logger.warning(f"Cost/Compute pages skipped due to error: {e}")

        # -------------------------------------------------------------
        # Visualisation: Generative Quality Metrics
        # -------------------------------------------------------------
        if gm:
            generate_generative_metrics_pages(
                pdf, gm, args, figures_dir if export_individual else None
            )

    logger.info(logmsg.PDF_GENERATED)

    # Redundant check: if export_individual was requested, call the standalone function
    # to ensure complete processing, though some logic overlaps with above.
    if export_individual:
        export_individual_figures(out, args, hist, metrics, y_true, y_pred, num_classes)


# =====================================================================
# Standalone Script Functionality
# =====================================================================

def find_experiment_folders(base_dirs: List[Path]) -> List[Path]:
    """
    Recursively scan provided paths to identify ALL valid experiment folders.
    
    A folder is considered an experiment if it contains a 'metrics' subdirectory.
    This function is permissive to ensure every subfolder is analyzed as requested.
    
    Args:
        base_dirs (List[Path]): A list of root directories to scan.
        
    Returns:
        List[Path]: A sorted list of unique experiment directories found.
    """
    experiment_folders = set()

    for base_dir in base_dirs:
        path = Path(base_dir)
        if not path.exists():
            logger.warning(f"⚠️ Search path not found: {path}")
            continue

        logger.info(f"🔍 Scanning: {path}")

        # 1. Direct check: Is the path itself an experiment?
        if (path / "metrics").exists():
            experiment_folders.add(path)

        # 2. Recursive search: Find all nested 'metrics' folders
        # This will catch everything inside H100 and L40 directories
        for metrics_dir in path.rglob("metrics"):
            if metrics_dir.is_dir():
                experiment_folders.add(metrics_dir.parent)

    return sorted(list(experiment_folders))


def load_experiment_data(experiment_dir: Path, override_data_dir: str | None = None) -> tuple | None:
    """
    Load and reconstruct experiment state from persistent storage (JSON/CSV).

    Args:
        experiment_dir (Path): The directory containing experiment artifacts.
        override_data_dir (str | None): Optional data directory path to override what's in args.json.

    Returns:
        tuple | None: (args, hist, metrics, y_true, y_pred, num_classes) on success,
                      or None if critical error.
    """
    try:
        # 1. Load Configuration (args.json)
        args_json_path = experiment_dir / "args.json"
        if args_json_path.exists():
            with open(args_json_path, "r") as f:
                args_dict = json.load(f)
            args = argparse.Namespace(**args_dict)
        else:
            # Handle baseline runs lacking explicit configuration
            logger.info(f"No args.json found in {experiment_dir}, creating minimal args for baseline")
            args = argparse.Namespace()
            
            # Infer dataset context from directory path structure
            path_parts = experiment_dir.parts
            if "mnist" in path_parts:
                if any("medmnist" in p for p in path_parts):
                    # Try to extract the full medmnist string from path (e.g. medmnist:pathmnist)
                    for part in path_parts:
                        if "medmnist" in part:
                            args.dataset = part
                            break
                    else:
                        args.dataset = "medmnist"
                else:
                    args.dataset = "mnist"
            elif "cifar" in path_parts:
                args.dataset = "cifar10"
            elif "isic" in path_parts:
                args.dataset = "fed_isic2019"
            elif "femnist" in path_parts or "emnist" in path_parts:
                args.dataset = "femnist"
            else:
                args.dataset = "unknown"

            args.data_dir = "../../data"  # Project default
            args.grayscale = False

        # OVERRIDE data_dir if provided via CLI
        if override_data_dir:
            args.data_dir = override_data_dir

        # 2. Load Classifier Metrics
        classifier_metrics_path = experiment_dir / "metrics" / "classifier.json"
        if classifier_metrics_path.exists():
            with open(classifier_metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        # 3. Load Training History (CSV)
        hist = {}
        hist_csv = experiment_dir / "metrics" / "history.csv"
        if hist_csv.exists():
            import csv
            with open(hist_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metric_name = row["metric"]
                    key = row["split_or_client"]
                    value = float(row["value"])

                    if metric_name not in hist:
                        hist[metric_name] = {}
                    if key not in hist[metric_name]:
                        hist[metric_name][key] = []

                    hist[metric_name][key].append(value)

        # 4. Load/Reconstruct Confusion Matrix Data
        cm_path = experiment_dir / "metrics" / "confusion-matrix.csv"
        if cm_path.exists():
            cm_data = np.loadtxt(cm_path, delimiter=",")
            # Reconstruct y_true and y_pred vectors from the aggregate confusion matrix
            y_true = []
            y_pred = []
            num_classes = cm_data.shape[0]

            for i in range(num_classes):
                for j in range(num_classes):
                    count = int(cm_data[i, j])
                    y_true.extend([i] * count)
                    y_pred.extend([j] * count)

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
        else:
            # Fallback: Generate random data to prevent report crash (use with caution)
            num_classes = metrics.get("num_classes", 10)
            y_true = np.array([0])
            y_pred = np.array([0])

        # 5. Load Generative Metrics
        gen_metrics_path = experiment_dir / "metrics" / "generative.json"
        if gen_metrics_path.exists():
            with open(gen_metrics_path, "r") as f:
                gen_metrics = json.load(f)
            metrics["gen_metrics"] = gen_metrics

        return args, hist, metrics, y_true, y_pred, num_classes

    except Exception as e:
        logger.error(f"Error loading experiment data from {experiment_dir}: {e}")
        return None


def main():
    """
    Execute the module as a standalone CLI script.
    """
    parser = argparse.ArgumentParser(description="Generate PDF reports for all experiments")
    # Updated: accept multiple directories via nargs='+'
    parser.add_argument("--base-dir", type=str, nargs='+', default=["../output"],
                        help="One or more base directories to scan for experiments")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override the data directory found in args.json (useful for MedMNIST)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments that already have reports")
    parser.add_argument("--export-figures", action="store_true",
                        help="Export individual figures to figures/ folder")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only export individual figures, don't create PDF")

    args = parser.parse_args()

    # Handle single string input compatibility
    raw_dirs = args.base_dir if isinstance(args.base_dir, list) else [args.base_dir]
    search_paths = [Path(p) for p in raw_dirs]

    # Locate targets
    experiment_folders = find_experiment_folders(search_paths)
    logger.info(f"Found {len(experiment_folders)} experiment folders")

    for exp_dir in experiment_folders:
        logger.info(f"Processing experiment: {exp_dir}")

        # Conditional Skip Logic
        report_pdf = exp_dir / "report.pdf"
        figures_dir = exp_dir / "figures"
        
        # Skip if report exists AND (we aren't exporting figures OR figures already exist)
        if args.skip_existing and report_pdf.exists():
            if not args.export_figures or (args.export_figures and figures_dir.exists()):
                logger.info(f"Skipping {exp_dir} - report already exists")
                continue

        # Data Loading Phase
        experiment_data = load_experiment_data(exp_dir, override_data_dir=args.data_dir)
        if experiment_data is None:
            logger.error(f"Failed to load data for {exp_dir}, skipping")
            continue

        exp_args, hist, metrics, y_true, y_pred, num_classes = experiment_data

        try:
            if args.figures_only:
                # Mode: Figure Export Only
                from viz_style import use_nature_style
                use_nature_style()
                export_individual_figures(exp_dir, exp_args, hist, metrics, y_true, y_pred, num_classes)
                logger.info(f"Successfully exported individual figures for {exp_dir}")
            else:
                # Mode: Full PDF Generation
                generate_pdf_report(exp_dir, exp_args, hist, metrics, y_true, y_pred, num_classes, args.export_figures)
                logger.info(f"Successfully generated report for {exp_dir}")

        except Exception as e:
            logger.error(f"Error generating report for {exp_dir}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("PDF report generation batch completed")


if __name__ == "__main__":
    main()
