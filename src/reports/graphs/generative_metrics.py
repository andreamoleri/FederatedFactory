"""
📊 Generative Metrics Visualization Module
------------------------------------------

This module provides a comprehensive suite of visualization functions designed
to generate and export high-quality charts and reports for evaluating generative
models. It handles a wide array of metrics, including Fréchet Inception Distance
(FID), Kernel Inception Distance (KID), Precision/Recall, and embedding
visualizations (t-SNE, PCA).

🧠 Purpose:
    To facilitate the rigorous analysis of generative model performance by
    converting raw metric dictionaries into interpretable, publication-ready
    visualizations (PDF reports and individual image files).

🔧 Core Functionalities:
    • Generate bar charts for class-wise metrics (FID, KID, Diversity)
    • Plot Precision-Recall scatter plots with iso-F1 contours
    • Visualize feature space embeddings using t-SNE (2D/3D) and PCA
    • Create distribution plots (Boxplots, Violin plots, Histograms) for distances
    • Render "scorecard" style summaries for reconstruction and diffusion metrics
    • Export all figures to a multipage PDF and optionally as individual PNGs

🎯 Intended Use:
    • Academic research pipelines for evaluating GANs, VAEs, and Diffusion models
    • Automated reporting systems in machine learning workflows
    • Comparative analysis of synthetic vs. real data distributions

📁 Dependencies:
    • matplotlib (pyplot, gridspec, 3D plotting)
    • numpy
    • path (pathlib)

📝 Notes:
    The module is designed to be resilient to missing keys in the input metric
    dictionary, skipping plots where data is unavailable rather than raising
    exceptions, ensuring robust report generation.

Author: Andrea Moleri
File Location: src/reports/graphs/generative_metrics.py
Last Modified: 20/11/2025
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve the root directory and ensure it is in the system path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import math
from typing import Dict, Optional, Any, List, Union
from pathlib import Path

from logs.logger import get_logger

logger = get_logger(__name__)

# Default dots per inch for figure rendering
_DEF_FIG_DPI = 300


# Helper functions from original file
def _ax_minimal(ax: plt.Axes) -> plt.Axes:
    """
    Apply a minimalist style to a Matplotlib Axes object.

    Removes the top and right spines and adjusts tick parameters for a cleaner,
    publication-quality look.

    Parameters
    ----------
    ax : plt.Axes
        The axes object to modify.

    Returns
    -------
    plt.Axes
        The styled axes object.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.6)
    return ax


def _label_bars(ax: plt.Axes, fmt: str = "{:.3f}", rotation: int = 0) -> None:
    """
    Annotate bar chart patches with their height values.

    Places a text label in the center of each bar in the provided axes.
    Useful for displaying exact metric values directly on the plot.

    Parameters
    ----------
    ax : plt.Axes
        The axes containing the bar chart.
    fmt : str, optional
        The format string for the numeric label (default is "{:.3f}").
    rotation : int, optional
        The rotation angle of the text (default is 0).
    """
    for p in ax.patches:
        h = p.get_height()
        x = p.get_x() + p.get_width() / 2
        y = p.get_y() + h / 2
        ax.text(
            x,
            y,
            fmt.format(h),
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            rotation=rotation,
        )


def _maybe_auto_ylim(ax: plt.Axes, values: List[float], pad: float = 0.3) -> None:
    """
    Automatically set the Y-axis limits based on the maximum value in the data.

    Adds a padding percentage above the maximum value to ensure bars or points
    are not cut off by the figure edge.

    Parameters
    ----------
    ax : plt.Axes
        The axes to adjust.
    values : List[float]
        The list of values plotted, used to determine the maximum.
    pad : float, optional
        The fraction of the maximum value to add as upper padding (default is 0.3).
    """
    if values is None or len(values) == 0:
        return
    vmax = float(np.nanmax(values))
    # Ensure vmax is positive to avoid issues with log scales or zero-height bars
    vmax = vmax if vmax > 0 else 1e-9
    ax.set_ylim(0, vmax * (1 + pad))


def _overlay_refs(
        ax: plt.Axes,
        classes: List[Any],
        arr_vals: List[float],
        rr_vals: Optional[Dict] = None,
        noise_vals: Optional[Dict] = None,
        sota: Optional[Dict] = None,
        ylabel: str = ""
) -> None:
    """
    Overlay reference lines for baseline comparisons on existing plots.

    Adds horizontal lines representing "Real-vs-Real" (RR) baselines or
    State-of-the-Art (SOTA) values if provided and applicable.

    Parameters
    ----------
    ax : plt.Axes
        The target axes.
    classes : List[Any]
        The list of class identifiers corresponding to the data.
    arr_vals : List[float]
        The main data values (used to check scale for appropriate scaling).
    rr_vals : Dict, optional
        Dictionary containing Real-vs-Real metric values per class.
    noise_vals : Dict, optional
        Dictionary containing noise baseline values (currently unused in logic).
    sota : Dict, optional
        Dictionary containing SOTA reference values (e.g., {'fid': 12.5}).
    ylabel : str, optional
        The label of the Y-axis, used to determine which SOTA key to look up
        (e.g., checks for 'FID' or 'KID').
    """
    if not arr_vals:
        return

    bar_max = float(np.nanmax(arr_vals))
    legend_handles = []
    legend_labels = []

    # 1. Overlay Real-vs-Real median if available
    if rr_vals is not None:
        available_rr_vals = [rr_vals[c] for c in classes if c in rr_vals]
        if available_rr_vals:
            med_rr = np.nanmedian(available_rr_vals)
            # Only plot if it falls within a reasonable visual range relative to the data
            if not (np.isnan(bar_max) or med_rr > 1.5 * bar_max):
                line = ax.axhline(
                    med_rr,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                )
                legend_handles.append(line)
                legend_labels.append("Real-vs-Real (median)")

    # 2. Overlay SOTA reference if available
    if isinstance(sota, dict):
        # Determine metric key based on y-label context
        key = "fid" if "FID" in ylabel.upper() else ("kid" if "KID" in ylabel.upper() else None)
        if key and key in sota:
            val = float(sota[key])
            # Only plot if it falls within a reasonable visual range
            if not (np.isnan(bar_max) or val > 2.0 * bar_max):
                line = ax.axhline(
                    val,
                    linestyle="-.",
                    linewidth=1.0,
                    alpha=0.9,
                )
                legend_handles.append(line)
                legend_labels.append(f"SOTA ref ({key.upper()})")

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=8)


def convert_keys_to_int(d: Any) -> Any:
    """
    Recursively convert dictionary keys to integers where possible.

    This is useful when parsing JSON inputs where integer keys (e.g., class IDs)
    are automatically converted to strings.

    Parameters
    ----------
    d : Any
        The input structure (usually a dict, list, or primitive).

    Returns
    -------
    Any
        The structure with digit-string keys converted to integers.
    """
    if not isinstance(d, dict):
        return d
    result = {}
    for k, v in d.items():
        try:
            # Attempt to convert string keys comprised of digits to integers
            new_key = int(k) if isinstance(k, str) and k.isdigit() else k
        except (ValueError, TypeError):
            new_key = k
        # Recursively process nested dictionaries
        result[new_key] = convert_keys_to_int(v) if isinstance(v, dict) else v
    return result


def generate_generative_metrics_pages(
        pdf: PdfPages,
        gm: Dict,
        args: Any,
        figures_dir: Path = None
) -> int:
    """
    Orchestrate the generation of all generative metric visualizations.

    Iterates through the provided metrics dictionary (`gm`) and generates
    plots for every available metric category (FID, KID, Precision/Recall,
    t-SNE, Reconstruction, etc.). Each plot is saved to the provided
    PDF object and optionally to individual image files.

    Parameters
    ----------
    pdf : PdfPages
        An open matplotlib PdfPages object where figures will be saved.
    gm : Dict
        The generative metrics dictionary containing raw data. Expected structure:
        {
            'fid_per_class': {class_id: value, ...},
            'kid_per_class': {class_id: value, ...},
            'precision_per_class': {...},
            'recall_per_class': {...},
            'tsne2': {'x': [[x,y], ...], 'labels': [...], 'domain': [...]},
            'reconstruction': {'mse_mean': float, ...},
            '...
        }
    args : Any
        Configuration namespace containing model parameters (e.g., `args.model`).
    figures_dir : Path, optional
        Directory path to export individual PNG files. If None, only the PDF
        is generated.

    Returns
    -------
    int
        The total number of pages (figures) added to the PDF.
    """
    pages_added = 0

    if not gm:
        return pages_added

    try:
        # Preprocess keys: JSON often loads integer class IDs as strings
        gm = convert_keys_to_int(gm)

        # Extract core metric dictionaries safely
        fid_per_class = gm.get("fid_per_class", {})
        kid_per_class = gm.get("kid_per_class", {})
        precision_per_class = gm.get("precision_per_class", {})
        recall_per_class = gm.get("recall_per_class", {})
        diversity_per_class = gm.get("diversity_nn_per_class", {})
        min_dist_per_class = gm.get("min_nn_dist_gen2real_per_class", {})

        if fid_per_class:
            classes = sorted(fid_per_class.keys())
        else:
            classes = []

        # -------------------------------------------------------------------------
        # 1. FID (Fréchet Inception Distance) per Class
        # -------------------------------------------------------------------------
        if classes:
            fid_vals = [fid_per_class[c] for c in classes]
            fig, ax = plt.subplots(
                figsize=(9, 4),
                dpi=_DEF_FIG_DPI,
            )

            # Use distinct colors for classes
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i % 10) for i in range(len(classes))]

            bars = ax.bar(range(len(classes)), fid_vals, color=colors)
            ax.set_title("Class-wise FID (lower is better)")
            ax.set_xlabel("Class")
            ax.set_ylabel("FID")
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels([str(c) for c in classes])

            _maybe_auto_ylim(ax, fid_vals, pad=0.3)
            _label_bars(ax, fmt="{:.3g}")

            _overlay_refs(
                ax,
                classes,
                fid_vals,
                rr_vals=None,
                noise_vals=gm.get("noise_fid"),
                sota=gm.get("sota"),
                ylabel="FID",
            )

            # Export individual figure if requested
            if figures_dir:
                fig.savefig(figures_dir / "generative_fid_per_class.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 2. KID (Kernel Inception Distance) per Class
        # -------------------------------------------------------------------------
        if kid_per_class and classes:
            kid_vals = [kid_per_class[c] for c in classes]
            fig, ax = plt.subplots(
                figsize=(9, 4),
                dpi=_DEF_FIG_DPI,
            )

            cmap = plt.get_cmap("tab10")
            colors = [cmap(i % 10) for i in range(len(classes))]

            bars = ax.bar(range(len(classes)), kid_vals, color=colors)
            ax.set_title("Class-wise KID (lower is better)")
            ax.set_xlabel("Class")
            ax.set_ylabel("KID")
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels([str(c) for c in classes])

            _maybe_auto_ylim(ax, kid_vals, pad=0.3)
            _label_bars(ax, fmt="{:.4g}")

            _overlay_refs(
                ax,
                classes,
                kid_vals,
                rr_vals=None,
                noise_vals=gm.get("noise_kid"),
                sota=gm.get("sota"),
                ylabel="KID",
            )

            if figures_dir:
                fig.savefig(figures_dir / "generative_kid_per_class.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 3. Precision and Recall Bars per Class
        # -------------------------------------------------------------------------
        if precision_per_class and recall_per_class and classes:
            pr_vals = np.array([precision_per_class[c] for c in classes])
            rc_vals = np.array([recall_per_class[c] for c in classes])

            fig, ax = plt.subplots(
                figsize=(10, 4),
                dpi=_DEF_FIG_DPI,
            )
            width = 0.35
            x = np.arange(len(classes))

            # Side-by-side bars for Precision vs Recall comparison
            b1 = ax.bar(
                x - width / 2,
                pr_vals,
                width,
                label="Precision",
            )
            b2 = ax.bar(
                x + width / 2,
                rc_vals,
                width,
                label="Recall",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in classes])
            ax.set_ylim(0, 1.0)
            ax.set_title(
                "Class-wise Precision and Recall (higher is better)"
            )
            ax.set_xlabel("Class")
            ax.set_ylabel("Score")
            ax.legend()

            # Annotate values on bars
            for bars_ in (b1, b2):
                for b in bars_:
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() / 2,
                        f"{b.get_height():.2f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

            if figures_dir:
                fig.savefig(figures_dir / "generative_precision_recall_bars.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 4. Precision-Recall Scatter with Iso-F1 Curves
        # -------------------------------------------------------------------------
        if precision_per_class and recall_per_class and classes:
            pr_vals = np.array([precision_per_class[c] for c in classes])
            rc_vals = np.array([recall_per_class[c] for c in classes])

            fig, ax = plt.subplots(
                figsize=(6, 6),
                dpi=_DEF_FIG_DPI,
            )
            classes_numeric = [int(c) if isinstance(c, str) and c.isdigit() else c for c in classes]

            # Scatter plot where color indicates class
            sc = ax.scatter(
                pr_vals,
                rc_vals,
                c=classes_numeric,
                cmap="tab10",
                edgecolors="none",
                linewidths=0,
            )

            # Handle legend or colorbar depending on the number of classes
            if len(classes) <= 10:
                handles = []
                labels = []
                cmap = plt.get_cmap("tab10")
                for i, c in enumerate(classes):
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="",
                            markerfacecolor=cmap(i % 10),
                            markeredgecolor="none",
                        )
                    )
                    labels.append(str(c))
                ax.legend(
                    handles=handles,
                    labels=labels,
                    title="Class",
                    fontsize=8,
                    ncol=2,
                )
            else:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label("Class")

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Precision")
            ax.set_ylabel("Recall")
            ax.set_title("PR Scatter with iso-F1 curves")

            # Plot Iso-F1 curves: R = (F1 * P) / (2 * P - F1)
            f1_levels = [0.2, 0.4, 0.6, 0.8, 0.9]
            P = np.linspace(0.01, 1.0, 200)
            for f1 in f1_levels:
                R = (f1 * P) / (2 * P - f1 + 1e-9)
                # Mask invalid values where calculation goes negative
                R[(2 * P - f1) <= 0] = np.nan
                ax.plot(P, R, linestyle="--", linewidth=0.8)

            if figures_dir:
                fig.savefig(figures_dir / "generative_pr_scatter.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 5. FID vs KID Scatter Plot
        # -------------------------------------------------------------------------
        if fid_per_class and kid_per_class and classes:
            fid_vals = [fid_per_class[c] for c in classes]
            kid_vals = [kid_per_class[c] for c in classes]

            fig, ax = plt.subplots(
                figsize=(6, 5),
                dpi=_DEF_FIG_DPI,
            )
            classes_numeric = [int(c) if isinstance(c, str) and c.isdigit() else c for c in classes]
            sc = ax.scatter(
                fid_vals,
                kid_vals,
                c=classes_numeric,
                cmap="tab10",
                edgecolors="none",
                linewidths=0,
            )
            ax.set_xlabel("FID (lower is better)")
            ax.set_ylabel("KID (lower is better)")
            ax.set_title("FID vs KID by Class")

            if len(classes) <= 10:
                handles = []
                labels = []
                cmap = plt.get_cmap("tab10")
                for i, c in enumerate(classes):
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="",
                            markerfacecolor=cmap(i % 10),
                            markeredgecolor="none",
                        )
                    )
                    labels.append(str(c))
                ax.legend(
                    handles=handles,
                    labels=labels,
                    title="Class",
                    fontsize=8,
                    ncol=2,
                )
            else:
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label("Class")

            if figures_dir:
                fig.savefig(figures_dir / "generative_fid_vs_kid_scatter.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 6. Intra-class Diversity (Feature Space Distance)
        # -------------------------------------------------------------------------
        if diversity_per_class and classes:
            div_vals = [diversity_per_class[c] for c in classes]
            fig, ax = plt.subplots(
                figsize=(9, 4),
                dpi=_DEF_FIG_DPI,
            )

            cmap = plt.get_cmap("tab10")
            colors = [cmap(i % 10) for i in range(len(classes))]

            bars = ax.bar(range(len(classes)), div_vals, color=colors)
            ax.set_title(
                "Intra-class Diversity (higher is better)"
            )
            ax.set_xlabel("Class")
            ax.set_ylabel(
                "Euclidean (L2) distance in feature space"
            )
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels([str(c) for c in classes])
            _label_bars(ax, fmt="{:.2f}")

            if figures_dir:
                fig.savefig(figures_dir / "generative_intra_class_diversity.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 7. Minimum Distance Distributions (Boxplots)
        # -------------------------------------------------------------------------
        if min_dist_per_class and classes:
            mins_all = [
                np.array(
                    min_dist_per_class[c],
                    dtype=np.float32,
                )
                for c in classes
            ]
            # Dynamic width based on class count
            fig, ax = plt.subplots(
                figsize=(max(10, len(classes) * 0.6), 4),
                dpi=_DEF_FIG_DPI,
            )

            cmap = plt.get_cmap("tab10")
            colors = [cmap(i % 10) for i in range(len(classes))]

            box_plot = ax.boxplot(
                mins_all,
                patch_artist=True,
                showfliers=False,
            )

            # Style the boxplot patches
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            for median in box_plot['medians']:
                median.set_color('darkred')
                median.set_linewidth(1.5)

            ax.set_title(
                "Distribution of min distance (generated → real) per class"
            )
            ax.set_xlabel("Class")
            ax.set_ylabel(
                "Euclidean (L2) distance in feature space"
            )
            ax.set_xticks(range(1, len(classes) + 1))
            ax.set_xticklabels(
                [str(c) for c in classes],
                rotation=0,
            )

            if figures_dir:
                fig.savefig(figures_dir / "generative_min_distance_boxplot.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 8. ECDF of Minimum Distances
        # -------------------------------------------------------------------------
        if min_dist_per_class and classes:
            fig, ax = plt.subplots(
                figsize=(8, 4),
                dpi=_DEF_FIG_DPI,
            )
            for c in classes:
                x = np.sort(
                    np.array(
                        gm["min_nn_dist_gen2real_per_class"][c],
                        dtype=np.float32,
                    )
                )
                if x.size == 0:
                    continue
                # Compute Empirical CDF
                y = np.linspace(0, 1, x.size)
                ax.plot(
                    x,
                    y,
                    label=str(c),
                    linewidth=1.0,
                )
            ax.set_title(
                "ECDF of min distance (generated → real), all classes"
            )
            ax.set_xlabel(
                "Distance (feature space)"
            )
            ax.set_ylabel("Cumulative probability")
            if len(classes) <= 12:
                ax.legend(ncol=2, fontsize=8)

            if figures_dir:
                fig.savefig(figures_dir / "generative_min_distance_ecdf.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 9. Copy-Detection / Memorization Check
        # -------------------------------------------------------------------------
        # Compare Real-Real nearest neighbors vs. Generated-Real nearest neighbors
        rr_all = np.concatenate(
            [
                np.array(
                    gm["min_rr_dist_per_class"][c],
                    dtype=np.float32,
                )
                for c in classes
                if len(gm["min_rr_dist_per_class"][c])
            ]
        )
        gr_all = np.concatenate(
            [
                np.array(
                    gm["min_nn_dist_gen2real_per_class"][c],
                    dtype=np.float32,
                )
                for c in classes
                if len(gm["min_nn_dist_gen2real_per_class"][c])
            ]
        )
        if rr_all.size and gr_all.size:
            fig, ax = plt.subplots(
                figsize=(8, 4),
                dpi=_DEF_FIG_DPI,
            )
            ax.hist(
                rr_all,
                bins=50,
                density=True,
                alpha=0.6,
                label="Real→Real (min NN)",
            )
            ax.hist(
                gr_all,
                bins=50,
                density=True,
                alpha=0.6,
                label="Gen→Real (min NN)",
            )
            ax.set_title(
                "Copy-detection: nearest-neighbor distance distributions"
            )
            ax.set_xlabel(
                "Euclidean (L2) distance in feature space"
            )
            ax.set_ylabel("Density")
            ax.legend()

            if figures_dir:
                fig.savefig(figures_dir / "generative_copy_detection_histogram.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 10. Histogram Overlay (Min Distance per Class)
        # -------------------------------------------------------------------------
        if min_dist_per_class and classes:
            fig, ax = plt.subplots(
                figsize=(10, 5),
                dpi=_DEF_FIG_DPI,
            )
            cmap = plt.get_cmap("tab10")
            for i, c in enumerate(classes):
                x = np.array(
                    gm["min_nn_dist_gen2real_per_class"][c],
                    dtype=np.float32,
                )
                if x.size == 0:
                    continue
                ax.hist(
                    x,
                    bins=30,
                    density=True,
                    alpha=0.35,
                    color=cmap(i % 10),
                    label=str(c),
                )
            ax.set_title(
                "Histogram of min distance (generated → real) in feature space — all classes"
            )
            ax.set_xlabel(
                "Euclidean (L2) distance in feature space"
            )
            ax.set_ylabel("Density")
            if len(classes) <= 12:
                ax.legend(ncol=2, fontsize=8)

            if figures_dir:
                fig.savefig(figures_dir / "generative_min_distance_histogram_overlay.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 11. t-SNE 2D Projection (Colored by Domain) - Nature Style
        # -------------------------------------------------------------------------
        if "tsne2" in gm:
            ts = gm["tsne2"]
            emb = np.array(ts["x"])
            y_cls = np.array(ts["labels"], dtype=int)
            y_dom = np.array(ts["domain"])
            
            # Create figure with extra width for external legend
            fig, ax = plt.subplots(
                figsize=(9, 6), 
                dpi=_DEF_FIG_DPI,
            )
            # Adjust subplot to leave room on the right for legend
            fig.subplots_adjust(right=0.75)

            # Nature Style Configuration
            # Soft ivory background
            ax.set_facecolor("#FDFCF0")
            # Soft dotted grid
            ax.grid(True, linestyle=":", alpha=0.6, color="#8c8c8c", zorder=0)

            # Nature colors: Moss Green (Real) vs Terracotta/Clay (Synthetic)
            colors = {
                "Real": "#3E6953",      # Moss/Forest Green
                "Synthetic": "#C76652"  # Terracotta/Clay Red
            }

            # Separate masks for Real vs Synthetic
            mask_real = y_dom == "Real"
            if np.any(mask_real):
                ax.scatter(
                    emb[mask_real, 0],
                    emb[mask_real, 1],
                    s=20,  # Slightly larger but transparent
                    alpha=0.6,
                    color=colors["Real"],
                    marker="o",
                    edgecolors="none",
                    linewidths=0,
                    label="Real",
                    zorder=2
                )

            mask_synth = y_dom == "Synthetic"
            if np.any(mask_synth):
                ax.scatter(
                    emb[mask_synth, 0],
                    emb[mask_synth, 1],
                    s=20,
                    alpha=0.6,
                    color=colors["Synthetic"],
                    marker="^",
                    edgecolors="none",
                    linewidths=0,
                    label="Synthetic",
                    zorder=3
                )

            ax.set_title("t-SNE (feature space): Real vs Synthetic (2D)", pad=15)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            
            # Clean Legend placed outside to right
            ax.legend(
                title="Domain", 
                loc="upper left", 
                bbox_to_anchor=(1.02, 1),
                frameon=False,
                borderaxespad=0.
            )

            if figures_dir:
                fig.savefig(figures_dir / "generative_tsne2_domain_colored.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 12. t-SNE 2D Projection (Colored by Class) - Nature Style
        # -------------------------------------------------------------------------
        if "tsne2" in gm:
            ts = gm["tsne2"]
            emb = np.array(ts["x"])
            y_cls = np.array(ts["labels"])
            y_dom = np.array(ts["domain"])
            
            fig, ax = plt.subplots(
                figsize=(9, 6),
                dpi=_DEF_FIG_DPI,
            )
            fig.subplots_adjust(right=0.75)

            # Nature Style Background
            ax.set_facecolor("#FDFCF0")
            ax.grid(True, linestyle=":", alpha=0.6, color="#8c8c8c", zorder=0)

            cmap = plt.get_cmap("tab10")
            uniq_cls = sorted(np.unique(y_cls))

            # Iterate over classes
            for i, cval in enumerate(uniq_cls):
                color = cmap(i % 10)
                mr = (y_cls == cval) & (y_dom == "Real")
                ms = (y_cls == cval) & (y_dom == "Synthetic")
                
                if np.any(mr):
                    ax.scatter(
                        emb[mr, 0],
                        emb[mr, 1],
                        s=18,
                        alpha=0.6,
                        color=color,
                        marker="o",
                        edgecolors="none",
                        linewidths=0,
                        zorder=2
                    )
                if np.any(ms):
                    ax.scatter(
                        emb[ms, 0],
                        emb[ms, 1],
                        s=18,
                        alpha=0.6,
                        color=color,
                        marker="^",
                        edgecolors="none",
                        linewidths=0,
                        zorder=3
                    )
            
            ax.set_title("t-SNE (feature space): by Class (2D)", pad=15)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

            # Custom Legend 1: Classes (Outside Right, Top)
            class_handles = []
            class_labels = []
            for i, cval in enumerate(uniq_cls):
                class_handles.append(
                    Line2D(
                        [0], [0],
                        marker="o", linestyle="",
                        markerfacecolor=cmap(i % 10),
                        markeredgecolor="none",
                        alpha=0.8
                    )
                )
                class_labels.append(str(cval))
            
            leg1 = ax.legend(
                handles=class_handles,
                labels=class_labels,
                title="Class (Color)",
                fontsize=9,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=False
            )
            ax.add_artist(leg1)

            # Custom Legend 2: Domain (Outside Right, Below Class Legend)
            domain_handles = [
                Line2D([0], [0], marker="o", linestyle="", color='gray', markerfacecolor="gray", 
                       markeredgecolor="none", label="Real", markersize=8),
                Line2D([0], [0], marker="^", linestyle="", color='gray', markerfacecolor="gray", 
                       markeredgecolor="none", label="Synthetic", markersize=8),
            ]
            
            ax.legend(
                handles=domain_handles,
                title="Domain (Marker)",
                fontsize=9,
                loc="upper left",
                bbox_to_anchor=(1.02, 0.4), # Positioned lower
                frameon=False
            )

            if figures_dir:
                fig.savefig(figures_dir / "generative_tsne2_class_colored.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 13. t-SNE 3D Projection (Colored by Domain) - Nature Style
        # -------------------------------------------------------------------------
        if "tsne3" in gm:
            ts3 = gm["tsne3"]
            emb3 = np.array(ts3["x"])
            y_dom3 = np.array(ts3["domain"])

            # Create figure with extra width
            fig = plt.figure(
                figsize=(10, 6),
                dpi=_DEF_FIG_DPI,
            )
            ax = fig.add_subplot(111, projection="3d")
            
            # Nature Style Colors
            colors = {
                "Real": "#3E6953",      # Moss/Forest Green
                "Synthetic": "#C76652"  # Terracotta/Clay Red
            }
            mdom = {"Real": "o", "Synthetic": "^"}
            
            # Soft background for 3D pane
            ax.set_facecolor("#FDFCF0") 
            # Make the panes transparent/white to not clash
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, linestyle=":", alpha=0.4)

            for dom in ("Real", "Synthetic"):
                mask = y_dom3 == dom
                ax.scatter(
                    emb3[mask, 0],
                    emb3[mask, 1],
                    emb3[mask, 2],
                    s=15, # Smaller points to avoid clutter in 3D
                    alpha=0.6,
                    marker=mdom[dom],
                    color=colors[dom],
                    edgecolors="none",
                    linewidths=0,
                    label=dom,
                    depthshade=True # Helps perception of depth
                )
            
            ax.set_title("t-SNE-like Embedding (3D): Real vs Synthetic", pad=20)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_zlabel("t-SNE 3")
            
            # Legend outside to the right
            ax.legend(
                title="Domain",
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),
                frameon=False
            )
            # Adjust layout to accommodate external legend
            plt.subplots_adjust(right=0.8)

            if figures_dir:
                fig.savefig(figures_dir / "generative_tsne3_domain_colored.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 14. t-SNE 3D Projection (Colored by Class) - Nature Style
        # -------------------------------------------------------------------------
        if "tsne3" in gm:
            ts3 = gm["tsne3"]
            emb3 = np.array(ts3["x"])
            y_cls3 = np.array(ts3["labels"])
            y_dom3 = np.array(ts3["domain"])

            fig = plt.figure(
                figsize=(10, 6),
                dpi=_DEF_FIG_DPI,
            )
            ax = fig.add_subplot(111, projection="3d")
            
            # Nature Style Background
            ax.set_facecolor("#FDFCF0")
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, linestyle=":", alpha=0.4)

            cmap = plt.get_cmap("tab10")
            uniq_cls3 = sorted(np.unique(y_cls3))

            for i, cval in enumerate(uniq_cls3):
                color = cmap(i % 10)
                mr = (y_cls3 == cval) & (y_dom3 == "Real")
                ms = (y_cls3 == cval) & (y_dom3 == "Synthetic")
                
                if np.any(mr):
                    ax.scatter(
                        emb3[mr, 0],
                        emb3[mr, 1],
                        emb3[mr, 2],
                        s=15,
                        alpha=0.6,
                        color=color,
                        marker="o",
                        edgecolors="none",
                        linewidths=0,
                    )
                if np.any(ms):
                    ax.scatter(
                        emb3[ms, 0],
                        emb3[ms, 1],
                        emb3[ms, 2],
                        s=15,
                        alpha=0.6,
                        color=color,
                        marker="^",
                        edgecolors="none",
                        linewidths=0,
                    )
            
            ax.set_title("t-SNE-like Embedding (3D): by Class", pad=20)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_zlabel("t-SNE 3")

            # Legends
            class_handles = []
            class_labels = []
            for i, cval in enumerate(uniq_cls3):
                class_handles.append(
                    Line2D(
                        [0], [0], marker="o", linestyle="",
                        markerfacecolor=cmap(i % 10),
                        markeredgecolor="none", alpha=0.8
                    )
                )
                class_labels.append(str(cval))
            
            leg1 = ax.legend(
                handles=class_handles,
                labels=class_labels,
                title="Class",
                loc="upper left",
                bbox_to_anchor=(1.05, 0.9),
                fontsize=9,
                frameon=False
            )
            ax.add_artist(leg1)

            domain_handles = [
                Line2D([0], [0], marker="o", linestyle="", color='gray', markerfacecolor="gray", 
                       markeredgecolor="none", label="Real"),
                Line2D([0], [0], marker="^", linestyle="", color='gray', markerfacecolor="gray", 
                       markeredgecolor="none", label="Synthetic"),
            ]
            ax.legend(
                handles=domain_handles,
                title="Domain",
                loc="upper left",
                bbox_to_anchor=(1.05, 0.4),
                fontsize=9,
                frameon=False
            )
            
            # Adjust layout
            plt.subplots_adjust(right=0.75)

            if figures_dir:
                fig.savefig(figures_dir / "generative_tsne3_class_colored.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 15. PCA Spectra (Cumulative Explained Variance)
        # -------------------------------------------------------------------------
        spectra = gm.get("pca_spectra", None)
        if isinstance(spectra, dict) and "real" in spectra and "gen" in spectra:
            ncols = min(5, len(classes))
            nrows = math.ceil(len(classes) / ncols) if classes else 1
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(min(14, 2.6 * len(classes)), 2.6 * nrows),
                squeeze=False,
                dpi=_DEF_FIG_DPI,
            )
            for i, c in enumerate(classes):
                r = i // ncols
                k = i % ncols
                ax = axes[r, k]
                evr_r = np.array(
                    spectra["real"].get(c, spectra["real"].get(str(c), [])),
                    dtype=float,
                )
                evr_g = np.array(
                    spectra["gen"].get(c, spectra["gen"].get(str(c), [])),
                    dtype=float,
                )

                # Plot cumulative sum of explained variance ratios
                if evr_r.size:
                    ax.plot(
                        np.arange(1, len(evr_r) + 1),
                        np.cumsum(evr_r),
                        label="Real",
                        linewidth=1.2,
                    )
                if evr_g.size:
                    ax.plot(
                        np.arange(1, len(evr_g) + 1),
                        np.cumsum(evr_g),
                        label="Synthetic",
                        linewidth=1.2,
                    )
                ax.set_title(f"Class {c}")
                ax.set_xlabel("Top-k PCs")
                ax.set_ylabel("Cumulative explained variance")
                ax.set_ylim(0, 1.01)
                if i == 0:
                    ax.legend(fontsize=8)
            fig.suptitle(
                "PCA spectra: Real vs Synthetic (per class)",
                fontsize=12,
                y=1.02,
            )
            fig.tight_layout()

            if figures_dir:
                fig.savefig(figures_dir / "generative_pca_spectra.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 16. Reconstruction Metrics (Text "Scorecards")
        # -------------------------------------------------------------------------
        if "reconstruction" in gm:
            rec = gm["reconstruction"]
            fig, axs = plt.subplots(
                2,
                2,
                figsize=(10, 6),
                dpi=_DEF_FIG_DPI,
            )
            axs = axs.ravel()

            # Tuple format: (Title, Value, Unit, Interpretation hint)
            metrics_cards = [
                ("MSE", rec["mse_mean"], "", "Lower is better"),
                ("PSNR", rec["psnr_mean"], "dB", "Higher is better"),
                ("SSIM", rec["ssim_mean"], "", "Higher is better"),
                (
                    "Perceptual (VGG L1)",
                    rec["perceptual_vgg_l1_mean"],
                    "",
                    "Lower is better",
                ),
            ]
            for ax, (title, val, unit, hint) in zip(axs, metrics_cards):
                ax.axis("off")
                # Draw border
                ax.add_patch(
                    plt.Rectangle(
                        (0.02, 0.1),
                        0.96,
                        0.8,
                        fill=False,
                        lw=1.0,
                        alpha=0.6,
                    )
                )
                # Render text stats
                ax.text(
                    0.5,
                    0.78,
                    title,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.text(
                    0.5,
                    0.47,
                    f"{val:.4f}{(' ' + unit) if unit else ''}",
                    ha="center",
                    va="center",
                    fontsize=24,
                )
                ax.text(
                    0.5,
                    0.22,
                    hint,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#555",
                )
            fig.suptitle(
                "Reconstruction metrics (reserved test, VAE)",
                fontsize=14,
                y=0.98,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            if figures_dir:
                fig.savefig(figures_dir / "generative_reconstruction_metrics.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 17. Reconstruction Metric Distributions (Violin Plots)
        # -------------------------------------------------------------------------
        if "reconstruction_dists" in gm:
            rd = gm["reconstruction_dists"]
            fig, axs = plt.subplots(
                2,
                2,
                figsize=(10, 6),
                dpi=_DEF_FIG_DPI,
            )
            axs = axs.ravel()

            def violin(ax, data, title, ylabel):
                v = ax.violinplot(
                    data,
                    showmeans=True,
                    showmedians=True,
                    showextrema=False,
                )
                # Style violin bodies
                for b in v["bodies"]:
                    b.set_facecolor("#d0d8f0")
                    b.set_edgecolor("#9aa9d6")
                    b.set_alpha(0.8)
                if "cmeans" in v:
                    v["cmeans"].set_color("#1f77b4")
                    v["cmeans"].set_linewidth(2.0)
                if "cmedians" in v:
                    v["cmedians"].set_color("#000000")
                    v["cmedians"].set_linestyle("--")
                    v["cmedians"].set_linewidth(1.6)

                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.set_xticks([])

            violin(axs[0], rd["mse"], "Reconstruction MSE", "MSE")
            violin(axs[1], rd["psnr"], "Reconstruction PSNR", "PSNR (dB)")
            violin(axs[2], rd["ssim"], "Reconstruction SSIM", "SSIM")
            violin(axs[3], rd["vgg_l1"], "Reconstruction Perceptual (VGG L1)", "VGG L1")

            handles = [
                Line2D(
                    [0],
                    [0],
                    color="#1f77b4",
                    lw=2.0,
                    label="Mean (blue line)",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#000000",
                    lw=1.6,
                    ls="--",
                    label="Median (black dashed)",
                ),
            ]
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=2,
                bbox_to_anchor=(0.5, 0.0),
            )
            fig.suptitle(
                "Reconstruction metric distributions (reserved test)",
                y=0.98,
            )
            fig.tight_layout(rect=[0, 0.04, 1, 0.95])

            if figures_dir:
                fig.savefig(figures_dir / "generative_reconstruction_violins.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

        # -------------------------------------------------------------------------
        # 18. Diffusion Model Specific Aggregates (Cards & Violins)
        # -------------------------------------------------------------------------
        if args.model.lower() == "diffusion":
            # Aggregate per-class metrics into flat arrays for distribution plotting
            fid_arr = (
                np.array([fid_per_class[c] for c in classes], dtype=float)
                if fid_per_class and classes
                else np.array([])
            )
            kid_arr = (
                np.array([kid_per_class[c] for c in classes], dtype=float)
                if kid_per_class and classes
                else np.array([])
            )
            pr_arr = (
                np.array([precision_per_class[c] for c in classes], dtype=float)
                if precision_per_class and classes
                else np.array([])
            )
            rc_arr = (
                np.array([recall_per_class[c] for c in classes], dtype=float)
                if recall_per_class and classes
                else np.array([])
            )

            def _safe_mean(a):
                return (
                    float(np.nanmean(a))
                    if a.size
                    else float("nan")
                )

            # Diffusion metrics cards
            fig, axs = plt.subplots(
                2,
                2,
                figsize=(10, 6),
                dpi=_DEF_FIG_DPI,
            )
            axs = axs.ravel()
            cards = [
                ("FID (mean)", _safe_mean(fid_arr), "", "Lower is better"),
                ("KID (mean)", _safe_mean(kid_arr), "", "Lower is better"),
                ("Precision (mean)", _safe_mean(pr_arr), "", "Higher is better"),
                ("Recall (mean)", _safe_mean(rc_arr), "", "Higher is better"),
            ]
            for ax, (title, val, unit, hint) in zip(axs, cards):
                ax.axis("off")
                ax.add_patch(
                    plt.Rectangle(
                        (0.02, 0.1),
                        0.96,
                        0.8,
                        fill=False,
                        lw=1.0,
                        alpha=0.6,
                    )
                )
                ax.text(
                    0.5,
                    0.78,
                    title,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.text(
                    0.5,
                    0.47,
                    f"{val:.4f}" if np.isfinite(val) else "N/A",
                    ha="center",
                    va="center",
                    fontsize=24,
                )
                ax.text(
                    0.5,
                    0.22,
                    hint,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#555",
                )
            fig.suptitle(
                "Diffusion quality metrics (reserved test, Diffusion)",
                fontsize=14,
                y=0.98,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            if figures_dir:
                fig.savefig(figures_dir / "generative_diffusion_metrics_cards.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

            # Diffusion metric distributions (Violin Plots)
            fig, axs = plt.subplots(
                2,
                2,
                figsize=(10, 6),
                dpi=_DEF_FIG_DPI,
            )
            axs = axs.ravel()

            def violin2(ax, data, title, ylabel):
                if len(data) == 0:
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        ha="center",
                        va="center",
                    )
                    return
                v = ax.violinplot(
                    data,
                    showmeans=True,
                    showmedians=True,
                    showextrema=False,
                )
                for b in v["bodies"]:
                    b.set_facecolor("#d0d8f0")
                    b.set_edgecolor("#9aa9d6")
                    b.set_alpha(0.8)
                if "cmeans" in v:
                    v["cmeans"].set_color("#1f77b4")
                    v["cmeans"].set_linewidth(2.0)
                if "cmedians" in v:
                    v["cmedians"].set_color("#000000")
                    v["cmedians"].set_linestyle = "--"
                    v["cmedians"].set_linewidth(1.6)

                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.set_xticks([])

            violin2(axs[0], fid_arr if fid_arr.size else [], "FID per class", "FID")
            violin2(axs[1], kid_arr if kid_arr.size else [], "KID per class", "KID")
            violin2(axs[2], pr_arr if pr_arr.size else [], "Precision per class", "Score")
            violin2(axs[3], rc_arr if rc_arr.size else [], "Recall per class", "Score")

            handles = [
                Line2D(
                    [0],
                    [0],
                    color="#1f77b4",
                    lw=2.0,
                    label="Mean (blue line)",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#000000",
                    lw=1.6,
                    ls="--",
                    label="Median (black dashed)",
                ),
            ]
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=2,
                bbox_to_anchor=(0.5, 0.0),
            )
            fig.suptitle(
                "Diffusion metric distributions (reserved test, Diffusion)",
                y=0.98,
            )
            fig.tight_layout(rect=[0, 0.04, 1, 0.95])

            if figures_dir:
                fig.savefig(figures_dir / "generative_diffusion_metric_distributions.png",
                            dpi=_DEF_FIG_DPI, bbox_inches="tight",
                            facecolor='white', edgecolor='none')

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pages_added += 1

    except Exception as e:
        logger.error(f"Error generating generative metrics pages: {e}")

    return pages_added
