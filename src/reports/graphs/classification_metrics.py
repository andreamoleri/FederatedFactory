"""
📊 Classification Metrics and Visualization Module
--------------------------------------------------

This module facilitates the computation and visualization of classification
performance metrics. It is designed to ingest raw prediction data, calculate
standard statistical indicators (e.g., Accuracy, F1-Score, Confusion Matrix),
and render publication-quality reports using Matplotlib.

🧠 Purpose:
    To provide a comprehensive, automated reporting pipeline for evaluating
    classifier performance in academic and research experiments. It supports
    detailed per-class analysis and high-level KPI summaries.

🔧 Core Functionalities:
    • Load and decode prediction artifacts from NumPy formats
    • Compute aggregate metrics (Accuracy, Macro/Micro/Weighted F1)
    • Generate normalized confusion matrices with thermal mapping
    • render detailed tabular reports and grouped bar charts per class
    • Export visualizations to multipage PDF documents

🎯 Intended Use:
    • Post-experiment evaluation pipelines
    • Research papers requiring standardized metric reporting
    • Model monitoring and validation dashboards

📁 Dependencies:
    • numpy
    • matplotlib
    • scikit-learn

📝 Notes:
    The module includes heuristics to handle both structured and unstructured
    NumPy arrays, automatically identifying ground truth and prediction columns
    based on common naming conventions.

Author: Andrea Moleri
File Location: src/reports/graphs/classification_metrics.py
Last Modified: 20/11/2025
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

_DEF_FIG_DPI = 300


# ------------------------- LOADING UTILITIES ------------------------- #

def decode_if_bytes(arr: np.ndarray) -> np.ndarray:
    """
    Converts a NumPy array of bytes or `np.bytes_` into standard Python strings.

    This utility ensures consistent string encoding handling, particularly when
    loading data saved by older NumPy versions or specific serialization protocols.

    Parameters
    ----------
    arr : np.ndarray
        The input array containing potential byte strings or mixed types.

    Returns
    -------
    np.ndarray
        A new array where all byte instances have been decoded to UTF-8 strings.
        Non-byte elements remain unchanged.
    """
    return np.array([
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x
        for x in arr
    ])


def load_predictions_from_experiment(experiment_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieves and parses test predictions from the experiment artifacts directory.

    This function is robust to different data structures, handling both structured
    NumPy arrays (with named fields) and standard 2D arrays. It employs field
    name heuristics to identify image identifiers, ground truth labels, and predictions.

    Parameters
    ----------
    experiment_dir : Path
        The root directory of the specific experiment containing an 'artifacts' subdirectory.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing three parallel arrays:
        - image_names: Identifiers for the samples.
        - y_true: The ground truth (actual) labels.
        - y_pred: The model's predicted labels.

    Raises
    ------
    FileNotFoundError
        If 'test_predictions.npy' does not exist in the expected location.
    ValueError
        If the data format is unstructured and does not match the expected (N, 3) shape.
    """
    predictions_path = experiment_dir / "artifacts" / "test_predictions.npy"

    if not predictions_path.exists():
        raise FileNotFoundError(f"File {predictions_path} not found")

    data = np.load(predictions_path, allow_pickle=True)

    # Case: Structured array (contains named fields)
    if isinstance(data, np.ndarray) and data.dtype.names is not None:
        names = data.dtype.names
        lower = [n.lower() for n in names]

        # Helper to locate a field based on a list of potential synonyms
        def get_field(candidates, default_idx):
            for c in candidates:
                if c in lower:
                    return data[names[lower.index(c)]]
            return data[names[default_idx]]

        # Attempt to resolve columns using common semantic names
        image_names = get_field(["image", "image_name", "filename", "file"], 0)
        y_true = get_field(["actual", "label", "y_true", "gt"], 1)
        y_pred = get_field(["pred", "prediction", "y_pred"], 2)
    else:
        # Case: Simple unstructured array, expecting shape (N, 3)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"Expected array (N,3), found {data.shape}")
        image_names = data[:, 0]
        y_true = data[:, 1]
        y_pred = data[:, 2]

    # Ensure uniform string encoding
    image_names = decode_if_bytes(image_names)
    y_true = decode_if_bytes(y_true)
    y_pred = decode_if_bytes(y_pred)
    return image_names, y_true, y_pred


# ------------------------- METRICS ------------------------- #

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculates a comprehensive suite of classification performance metrics.

    Computes global statistics (accuracy, F1-scores) and class-level details
    (confusion matrix, classification report). It also calculates a baseline
    metric based on the majority class in the ground truth.

    Parameters
    ----------
    y_true : np.ndarray
        An array of ground truth labels.
    y_pred : np.ndarray
        An array of predicted labels.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - 'labels': Unique class labels sorted alphanumerically.
        - 'accuracy': Overall accuracy score.
        - 'confusion_matrix': Raw confusion matrix (not normalized).
        - 'macro_f1', 'micro_f1', 'weighted_f1': Various F1 score averages.
        - 'report_dict': Detailed classification report (precision/recall/F1 per class).
        - 'majority_label': The label of the most frequent class.
        - 'majority_acc': Accuracy achievable by a naive majority-class classifier.
        - 'num_samples': Total number of evaluated samples.
        - 'num_classes': Total number of unique classes.
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate F1 scores using different averaging strategies
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    # Generate report as a dictionary for easier programmatic access later
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    # Determine baseline performance (Majority Class Classifier)
    uniq, counts = np.unique(y_true, return_counts=True)
    majority_idx = int(np.argmax(counts))
    majority_label = uniq[majority_idx]
    majority_acc = np.mean(y_pred == majority_label)

    return {
        "labels": labels,
        "accuracy": acc,
        "confusion_matrix": cm,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "report_dict": report_dict,
        "majority_label": majority_label,
        "majority_acc": majority_acc,
        "num_samples": len(y_true),
        "num_classes": len(labels),
    }


# ------------------------- PLOT STYLING ------------------------- #

def _ax_minimal(ax: plt.Axes) -> plt.Axes:
    """
    Applies a minimalist styling to a Matplotlib Axes object.

    Removes top and right spines and adjusts tick parameters for a cleaner look.

    Parameters
    ----------
    ax : plt.Axes
        The axes object to style.

    Returns
    -------
    plt.Axes
        The styled axes object.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.6)
    return ax


def _create_consistent_figure(
        title: str | None = None,
        subtitle: str | None = None,
        figsize: Tuple[float, float] = (8, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Initializes a Matplotlib Figure and Axes with consistent styling and typography.

    Parameters
    ----------
    title : str | None, optional
        The main title of the figure.
    subtitle : str | None, optional
        A secondary italicized subtitle.
    figsize : Tuple[float, float], optional
        The dimensions of the figure in inches. Default is (8, 8).

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        A tuple containing the created Figure and the primary Axes object.
    """
    fig = plt.figure(figsize=figsize, dpi=_DEF_FIG_DPI)

    if title:
        fig.text(
            0.5,
            0.98,
            title,
            ha="center",
            va="top",
            fontsize=12,
            weight="normal",
        )
    if subtitle:
        fig.text(
            0.5,
            0.955,
            subtitle,
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
        )

    # Define axes position manually to ensure consistent margins
    ax = fig.add_axes([0.1, 0.12, 0.85, 0.78])
    _ax_minimal(ax)
    return fig, ax


# ------------------------- FORMATTING HELPERS ------------------------- #

def _fmt_percentage(v: float) -> str:
    return f"{v * 100:.2f}%"


def _fmt_float(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}"


# ------------------------- PAGE 1: KPI CARDS ------------------------- #

def page_classification_kpi(pdf: PdfPages, metrics: Dict[str, Any],
                            figures_dir: Optional[Path] = None) -> None:
    """
    Generates a PDF page displaying high-level Key Performance Indicators (KPIs).

    Renders a grid of "cards", each containing a summary statistic such as
    accuracy, F1 scores, and dataset properties.

    Parameters
    ----------
    pdf : PdfPages
        The multipage PDF object to append the figure to.
    metrics : Dict[str, Any]
        The dictionary of calculated metrics.
    figures_dir : Optional[Path], optional
        If provided, the individual figure is saved as a PNG to this directory.
    """
    fig, ax = _create_consistent_figure(
        "Test Set Classification Overview",
        "Key performance indicators on test set",
        figsize=(8, 8),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    accuracy = metrics["accuracy"]
    macro_f1 = metrics["macro_f1"]
    micro_f1 = metrics["micro_f1"]
    weighted_f1 = metrics["weighted_f1"]
    majority_label = metrics["majority_label"]
    majority_acc = metrics["majority_acc"]
    num_samples = metrics["num_samples"]
    num_classes = metrics["num_classes"]

    kpis = [
        ("Test Accuracy", _fmt_percentage(accuracy)),
        ("F1 Macro", _fmt_percentage(macro_f1)),
        ("F1 Micro", _fmt_percentage(micro_f1)),
        ("F1 Weighted", _fmt_percentage(weighted_f1)),
        ("# Samples", f"{num_samples}"),
        ("# Classes", f"{num_classes}"),
        ("Majority Class", f"{majority_label}"),
        ("Maj. Class Accuracy", _fmt_percentage(majority_acc)),
    ]

    # Calculate grid layout for cards (2 columns)
    ncols = 2
    nrows = int(np.ceil(len(kpis) / ncols))
    x_pad, y_pad = 0.07, 0.07
    gap_x, gap_y = 0.04, 0.04
    card_w = (1 - 2 * x_pad - (ncols - 1) * gap_x) / ncols
    card_h = (1 - 2 * y_pad - (nrows - 1) * gap_y) / nrows

    cmap = plt.get_cmap("tab10")

    for i, (title, value) in enumerate(kpis):
        r = i // ncols
        c = i % ncols
        x_left = x_pad + c * (card_w + gap_x)
        y_top = 1 - y_pad - r * (card_h + gap_y)

        color = cmap(i % 10)

        # Draw a rounded rectangle "card" for the metric
        card = FancyBboxPatch(
            (x_left, y_top - card_h),
            card_w,
            card_h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=0.8,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
        )
        ax.add_patch(card)

        cx = x_left + card_w / 2
        cy = y_top - card_h / 2

        # Metric Title
        ax.text(
            cx,
            cy + card_h * 0.12,
            title,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
        )
        # Metric Value with background pill
        ax.text(
            cx,
            cy - card_h * 0.1,
            value,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=color,
                alpha=0.25,
                linewidth=0,
            ),
        )

    pdf.savefig(fig, bbox_inches="tight")

    # Save individual figure if requested
    if figures_dir:
        individual_path = figures_dir / "classification_kpi.png"
        fig.savefig(individual_path, dpi=_DEF_FIG_DPI, bbox_inches="tight")

    plt.close(fig)


# ------------------------- PAGE 2: CONFUSION MATRIX ------------------------- #

def page_classification_confusion_matrix(pdf: PdfPages, metrics: Dict[str, Any],
                                         figures_dir: Optional[Path] = None) -> None:
    """
    Generates a PDF page displaying the normalized Confusion Matrix.

    The matrix is row-normalized (values represent the percentage of the true label).
    Cells are annotated with absolute counts and percentage values.

    Parameters
    ----------
    pdf : PdfPages
        The multipage PDF object to append the figure to.
    metrics : Dict[str, Any]
        The dictionary of calculated metrics.
    figures_dir : Optional[Path], optional
        If provided, the individual figure is saved as a PNG to this directory.
    """
    labels = metrics["labels"]
    cm = metrics["confusion_matrix"].astype(float)

    # Row-wise normalization (True Class distribution)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = _create_consistent_figure(
        "Classification Confusion Matrix",
        "Rows: True Label — Columns: Predicted Label",
        figsize=(8, 8),
    )

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Percentage per true class")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)

    # Annotate cells with Count and Percentage
    # Dynamic text color based on cell background intensity
    thresh = cm_norm.max() / 2.0 if cm_norm.size > 0 else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(cm[i, j])
            perc = cm_norm[i, j] * 100
            txt = f"{count}\n{perc:.1f}%"
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )

    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)

    pdf.savefig(fig, bbox_inches="tight")

    # Save individual figure if requested
    if figures_dir:
        individual_path = figures_dir / "classification_confusion_matrix.png"
        fig.savefig(individual_path, dpi=_DEF_FIG_DPI, bbox_inches="tight")

    plt.close(fig)


# ------------------------- PAGE 3: PER-CLASS METRICS (BAR) ------------------------- #

def page_classification_per_class_bars(pdf: PdfPages, metrics: Dict[str, Any],
                                       figures_dir: Optional[Path] = None) -> None:
    """
    Generates a PDF page displaying grouped bar charts for per-class metrics.

    Visualizes Precision, Recall, and F1-score for each class side-by-side.

    Parameters
    ----------
    pdf : PdfPages
        The multipage PDF object to append the figure to.
    metrics : Dict[str, Any]
        The dictionary of calculated metrics.
    figures_dir : Optional[Path], optional
        If provided, the individual figure is saved as a PNG to this directory.
    """
    labels = metrics["labels"]
    report_dict = metrics["report_dict"]

    # Extract precision/recall/f1/support per label from the report dictionary
    per_class_prec = []
    per_class_rec = []
    per_class_f1 = []
    supports = []

    for label in labels:
        key = str(label)
        class_stats = report_dict.get(key, {})
        per_class_prec.append(class_stats.get("precision", 0.0))
        per_class_rec.append(class_stats.get("recall", 0.0))
        per_class_f1.append(class_stats.get("f1-score", 0.0))
        supports.append(class_stats.get("support", 0))

    per_class_prec = np.array(per_class_prec)
    per_class_rec = np.array(per_class_rec)
    per_class_f1 = np.array(per_class_f1)
    supports = np.array(supports)

    fig, ax = _create_consistent_figure(
        "Per-Class Classification Metrics",
        "Precision, recall, and F1-score per class",
        figsize=(9, 6),
    )

    x = np.arange(len(labels))
    width = 0.25

    # Plot bars with offsets to group them by class
    ax.bar(
        x - width,
        per_class_prec,
        width,
        label="Precision",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x,
        per_class_rec,
        width,
        label="Recall",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width,
        per_class_f1,
        width,
        label="F1",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    # Annotate sample support (n) above the bars
    for i, s in enumerate(supports):
        ax.text(
            x[i],
            1.03,
            f"n={s}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="gray",
        )

    pdf.savefig(fig, bbox_inches="tight")

    # Save individual figure if requested
    if figures_dir:
        individual_path = figures_dir / "classification_per_class_metrics.png"
        fig.savefig(individual_path, dpi=_DEF_FIG_DPI, bbox_inches="tight")

    plt.close(fig)


# ------------------------- PAGE 4: DETAILED TABLE ------------------------- #

def page_classification_detailed_metrics(pdf: PdfPages, metrics: Dict[str, Any],
                                         figures_dir: Optional[Path] = None) -> None:
    """
    Generates a PDF page containing a detailed tabular report of all metrics.

    Includes per-class metrics (Precision, Recall, F1, Support) followed by
    global aggregates (Micro, Macro, Weighted averages, and Accuracy).

    Parameters
    ----------
    pdf : PdfPages
        The multipage PDF object to append the figure to.
    metrics : Dict[str, Any]
        The dictionary of calculated metrics.
    figures_dir : Optional[Path], optional
        If provided, the individual figure is saved as a PNG to this directory.
    """
    labels = metrics["labels"]
    report_dict = metrics["report_dict"]

    header = ["Label", "Precision", "Recall", "F1-score", "Support"]
    table_rows = []

    # Populate per-class rows
    for label in labels:
        key = str(label)
        stats = report_dict.get(key, {})
        p = stats.get("precision", 0.0)
        r = stats.get("recall", 0.0)
        f1 = stats.get("f1-score", 0.0)
        s = stats.get("support", 0)
        table_rows.append([
            str(label),
            f"{p:.3f}",
            f"{r:.3f}",
            f"{f1:.3f}",
            str(s),
        ])

    # Retrieve global aggregates
    macro = report_dict.get("macro avg", {})
    weighted = report_dict.get("weighted avg", {})
    micro_f1 = metrics["micro_f1"]
    accuracy = metrics["accuracy"]
    num_samples = metrics["num_samples"]

    # Micro average row: (precision = recall = f1_micro for single-label multi-class)
    table_rows.append([
        "micro avg",
        f"{micro_f1:.3f}",
        f"{micro_f1:.3f}",
        f"{micro_f1:.3f}",
        str(num_samples),
    ])

    # Macro average row
    table_rows.append([
        "macro avg",
        f"{macro.get('precision', 0.0):.3f}",
        f"{macro.get('recall', 0.0):.3f}",
        f"{macro.get('f1-score', 0.0):.3f}",
        str(num_samples),
    ])

    # Weighted average row
    table_rows.append([
        "weighted avg",
        f"{weighted.get('precision', 0.0):.3f}",
        f"{weighted.get('recall', 0.0):.3f}",
        f"{weighted.get('f1-score', 0.0):.3f}",
        str(num_samples),
    ])

    # Accuracy row
    table_rows.append([
        "accuracy",
        f"{accuracy:.3f}",
        f"{accuracy:.3f}",
        f"{accuracy:.3f}",
        str(num_samples),
    ])

    fig, ax = _create_consistent_figure(
        "Detailed Classification Metrics",
        "Precision, recall and F1-score: per class and global",
        figsize=(8.5, 10),
    )
    ax.axis("off")

    # Construct the table
    col_labels = header
    cell_text = table_rows

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    # Style the table: distinct header and subtle grid
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)
            cell.set_text_props(weight="bold")
        else:
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.5)

    pdf.savefig(fig, bbox_inches="tight")

    # Save individual figure if requested
    if figures_dir:
        individual_path = figures_dir / "classification_detailed_metrics.png"
        fig.savefig(individual_path, dpi=_DEF_FIG_DPI, bbox_inches="tight")

    plt.close(fig)


# ------------------------- MAIN GENERATOR ------------------------- #

def generate_classification_metrics_pages(pdf: PdfPages, experiment_dir: Path,
                                          figures_dir: Optional[Path] = None) -> None:
    """
    Orchestrates the generation of all classification metric pages for a specific experiment.

    This function encapsulates the full pipeline:
    1. Loading prediction data.
    2. Computing metrics.
    3. Generating and saving plots to the PDF (and optionally as images).

    Parameters
    ----------
    pdf : PdfPages
        The multipage PDF object where the reports will be added.
    experiment_dir : Path
        The root directory of the experiment containing artifacts.
    figures_dir : Optional[Path], optional
        Directory to save individual PNG figures.
    """
    try:
        # Load predictions
        image_names, y_true, y_pred = load_predictions_from_experiment(experiment_dir)

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred)

        # Generate pages
        page_classification_kpi(pdf, metrics, figures_dir)
        page_classification_confusion_matrix(pdf, metrics, figures_dir)
        page_classification_per_class_bars(pdf, metrics, figures_dir)
        page_classification_detailed_metrics(pdf, metrics, figures_dir)

    except FileNotFoundError as e:
        print(f"Warning: {e}. Skipping classification pages.")
    except Exception as e:
        print(f"Error during classification metrics generation: {e}")
        import traceback
        print(traceback.format_exc())