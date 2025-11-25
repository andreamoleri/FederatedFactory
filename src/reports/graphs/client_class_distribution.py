"""
📊 Client Class Distribution Visualization
------------------------------------------

This module generates stacked bar charts to visualize the distribution of data classes
across different clients. It supports both absolute sample counts and normalized
percentage distributions.

🧠 Purpose:
    To provide a clear graphical representation of data heterogeneity or imbalance
    among clients in a federated or distributed dataset. This aids researchers in
    diagnosing non-IID (Independent and Identically Distributed) data issues.

🔧 Core Functionalities:
    • Load and validate client-class distribution data from CSV files
    • Generate stacked bar charts for absolute sample counts per client
    • Generate 100% stacked bar charts for relative class proportions
    • Export visualizations to an open Matplotlib PDF backend and local PNG files

🎯 Intended Use:
    • Automated reporting pipelines in machine learning experiments
    • Exploratory data analysis (EDA) for distributed systems
    • Academic publication of dataset characteristics

📁 Dependencies:
    • matplotlib
    • numpy
    • pandas

📝 Notes:
    The input CSV is expected to contain a 'client' column and multiple columns
    matching the regex pattern 'class_\\d+'.

Author: Andrea Moleri
File Location: src/reports/graphs/client_class_distribution.py
Last Modified: 20/11/2025
"""
from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

logger = logging.getLogger(__name__)

_DEF_FIG_DPI = 300


def _load_csv(exp_dir: Path, csv_rel_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the client class distribution CSV file.

    This function verifies the existence of the file and ensures that the
    required 'client' and 'class_*' columns are present. It sets the
    'client' column as the index for downstream plotting.

    Parameters
    ----------
    exp_dir : Path
        The base directory containing the experiment results.
    csv_rel_path : str
        The relative path to the CSV file from the experiment directory.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by 'client', containing only the columns
        corresponding to class counts (e.g., 'class_0', 'class_1').

    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist.
    ValueError
        If the 'client' column or any 'class_*' columns are missing.
    """
    csv_path = exp_dir / csv_rel_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "client" not in df.columns:
        raise ValueError("Column 'client' missing in CSV.")

    # Identify columns that match the pattern 'class_N' (e.g., class_0, class_10)
    class_cols: List[str] = [c for c in df.columns if re.match(r"^class_\d+$", c)]
    if not class_cols:
        raise ValueError("No 'class_*' columns found in CSV.")

    df = df.set_index("client")[class_cols]
    return df


def _load_alpha(exp_dir: Path) -> Optional[float]:
    """
    Attempts to retrieve the Dirichlet alpha parameter from experiment metadata.
    It returns the value ONLY IF the partition method is 'dirichlet'.

    Priority:
      1. args.json -> field "alpha" and "partition"
      2. manifest.json -> identity.dirichlet_alpha and identity.partition

    If not found or if partition is not 'dirichlet', returns None.
    """
    
    # 1) Try args.json
    args_path = exp_dir / "args.json"
    if args_path.exists():
        try:
            data = json.loads(args_path.read_text())
            
            partition_method = str(data.get("partition", "")).lower()
            alpha_value = data.get("alpha")
            
            # Restituisce alpha solo se il metodo di partizione è 'dirichlet'
            if partition_method == "dirichlet" and alpha_value is not None:
                return float(alpha_value)
            
        except Exception as e:
            logger.debug(f"Error reading or parsing args.json for alpha: {e}")
            pass

    # 2) Try manifest.json
    manifest_path = exp_dir / "manifest.json"
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            ident = data.get("identity", {})
            
            partition_method = str(ident.get("partition", "")).lower()
            alpha_value = ident.get("dirichlet_alpha")
            
            # Restituisce alpha solo se il metodo di partizione è 'dirichlet'
            if partition_method == "dirichlet" and alpha_value is not None:
                return float(alpha_value)
            
        except Exception as e:
            logger.debug(f"Error reading or parsing manifest.json for alpha: {e}")
            pass

    return None


def _present_class_cols(df: pd.DataFrame, class_cols: List[str]) -> List[str]:
    """
    Identifies class columns that contain at least one non-zero value.

    This utility prevents the plotting of empty classes (classes with zero samples
    across all clients) to maintain a clean visualization.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing class counts.
    class_cols : List[str]
        The list of all potential class column names.

    Returns
    -------
    List[str]
        A subset of `class_cols` where the column sum is greater than zero.
    """
    return [c for c in class_cols if (df[c] > 0).any()]


def _stacked_counts_figure(df: pd.DataFrame, alpha: Optional[float] = None) -> plt.Figure:
    """
    Generates a stacked bar chart representing absolute sample counts.

    The chart displays the number of samples per class for each client.
    Bars are stacked to show the total number of samples per client.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe indexed by client, containing absolute class counts.
    alpha : Optional[float]
        Dirichlet concentration parameter (if any), used for annotating
        the plot title.

    Returns
    -------
    plt.Figure
        A Matplotlib figure object containing the generated plot.
    """
    clients = df.index.tolist()
    class_cols = list(df.columns)
    present = _present_class_cols(df, class_cols)

    fig, ax = plt.subplots(figsize=(12, 6))
    # Handle the edge case where the dataframe is empty or contains only zeros
    if not present:
        ax.text(0.5, 0.5, "All values are zero", ha="center", va="center")
        ax.axis("off")
        return fig

    # Initialize the bottom baseline for the stacked bars
    bottoms = np.zeros(len(df), dtype=float)
    for col in present:
        heights = df[col].to_numpy(dtype=float)
        # Ensure strictly non-negative heights for plotting safety
        seg = np.where(heights > 0, heights, 0.0)
        ax.bar(clients, seg, bottom=bottoms, label=col)
        # Update the baseline for the next segment
        bottoms = bottoms + seg

    # Calculate the maximum total height to set the Y-axis limit with a buffer
    total_per_client = df[present].sum(axis=1).to_numpy(dtype=float)
    ymax = float(np.max(total_per_client)) * 1.10 if total_per_client.size else 1.0
    ax.set_ylim(0, ymax)

    ax.set_xlabel("Client")
    ax.set_ylabel("Number of samples")

    title = "Class distribution per client (stacked)"
    if alpha is not None:
        title += f" (alpha={alpha:g})"
    ax.set_title(title)

    # Place the legend outside the plot area to prevent occlusion
    ax.legend(title="Classes", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def _stacked_percent_figure(df: pd.DataFrame, alpha: Optional[float] = None) -> plt.Figure:
    """
    Generates a 100% stacked bar chart representing relative class proportions.

    Data is normalized row-wise so that the total height of every bar is 100.
    This visualization is useful for comparing class balance across clients
    regardless of the total number of samples.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe indexed by client, containing absolute class counts.
    alpha : Optional[float]
        Dirichlet concentration parameter (if any), used for annotating
        the plot title.

    Returns
    -------
    plt.Figure
        A Matplotlib figure object containing the generated plot.
    """
    # Normalize data row by row to percentage (0-100)
    # Replace 0 sum with NaN to avoid division by zero, then fill resulting NaNs
    row_sums = df.sum(axis=1).replace(0, np.nan)
    df_pct = (df.div(row_sums, axis=0) * 100.0).fillna(0.0)

    clients = df_pct.index.tolist()
    class_cols = list(df_pct.columns)
    present = _present_class_cols(df_pct, class_cols)

    fig, ax = plt.subplots(figsize=(12, 6))
    if not present:
        ax.text(0.5, 0.5, "All values are zero", ha="center", va="center")
        ax.axis("off")
        return fig

    bottoms = np.zeros(len(df_pct), dtype=float)
    for col in present:
        heights = df_pct[col].to_numpy(dtype=float)
        seg = np.where(heights > 0, heights, 0.0)
        ax.bar(clients, seg, bottom=bottoms, label=col)
        bottoms = bottoms + seg

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Client")

    title = "Class distribution per client (stacked 100%)"
    if alpha is not None:
        title += f" (alpha={alpha:g})"
    ax.set_title(title)

    ax.legend(title="Classes", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def generate_client_class_distribution_pages(
    pdf,                       # matplotlib.backends.backend_pdf.PdfPages
    exp_dir: Path,             # experiment folder
    figures_dir: Optional[Path] = None,
    csv_rel_path: str = "distributions/client_class_distribution.csv",
) -> None:
    """
    Orchestrates the generation and saving of class distribution plots.

    This function produces two visualizations:
    1. Absolute counts (stacked).
    2. Relative percentages (stacked 100%).

    The figures are appended to an open PDF report object and optionally
    saved as individual PNG files if a directory is provided.

    Parameters
    ----------
    pdf : matplotlib.backends.backend_pdf.PdfPages
        An open PdfPages object where figures will be saved.
    exp_dir : Path
        The root directory of the experiment.
    figures_dir : Optional[Path], optional
        Directory where individual PNG files should be saved. Defaults to None.
    csv_rel_path : str, optional
        Relative path to the source CSV file.
        Defaults to "distributions/client_class_distribution.csv".

    Returns
    -------
    None
    """
    try:
        df = _load_csv(exp_dir, csv_rel_path)
    except Exception as e:
        logger.warning(f"[ClientClassDistribution] skip: {e}")
        return

    # Try to load Dirichlet alpha from experiment metadata (if any)
    alpha = _load_alpha(exp_dir)

    # 1) Absolute counts visualization
    fig1 = _stacked_counts_figure(df, alpha=alpha)
    pdf.savefig(fig1, dpi=_DEF_FIG_DPI)
    if figures_dir is not None:
        (figures_dir / "client_class_distribution.png").parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(figures_dir / "client_class_distribution.png", dpi=_DEF_FIG_DPI)
    plt.close(fig1)

    # 2) Percentage visualization (0–100%)
    fig2 = _stacked_percent_figure(df, alpha=alpha)
    pdf.savefig(fig2, dpi=_DEF_FIG_DPI)
    if figures_dir is not None:
        fig2.savefig(figures_dir / "client_class_distribution_percent.png", dpi=_DEF_FIG_DPI)
    plt.close(fig2)

