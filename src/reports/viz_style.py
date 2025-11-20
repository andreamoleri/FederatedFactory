"""
📊 Visualization Styling Module
-----------------------------

This module configures the global plotting parameters for Matplotlib and Seaborn 
to adhere to the aesthetic guidelines typical of high-impact scientific journals 
(e.g., Nature).

🧠 Purpose:
    To provide a reproducible, publication-ready visualisation theme that prioritises 
    readability, vector graphics compatibility, and minimalist design. It standardises 
    font selection, axis formatting, and figure dimensions across the codebase.

🔧 Core Functionalities:
    • Registration of custom TrueType fonts (e.g., Arial) without requiring system-level installation
    • Configuration of Matplotlib `rcParams` for vector-friendly export (PDF Type 42)
    • Application of colour-blind-friendly palettes via Seaborn (if available)
    • Standardisation of tick sizes, line widths, and font sizes for print media

🎯 Intended Use:
    • Generating figures for academic papers and technical reports
    • Standardising visual output across collaborative research projects
    • Creating editable vector graphics (PDF/EPS) for post-processing in illustration software

📁 Dependencies:
    • os
    • matplotlib
    • seaborn (optional, but recommended for enhanced palettes)

📝 Notes:
    This module prefers "Arial" or "Helvetica" to match standard scientific formatting. 
    It explicitly disables LaTeX rendering to ensure font portability and easier 
    post-editing in tools like Adobe Illustrator.

Author: Andrea Moleri
File Location: src/reports/viz_style.py
Last Modified: 20/11/2025
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def _try_register_arial(local_paths):
    """
    Attempts to register custom TrueType fonts (specifically Arial variants) 
    from a list of provided file paths.

    This utility allows the usage of specific fonts without requiring root 
    privileges or system-wide installation, which is particularly useful 
    in restricted HPC or containerised environments.

    Parameters
    ----------
    local_paths : iterable of str
        A sequence of file paths (absolute or relative) pointing to candidate 
        `.ttf` font files.

    Returns
    -------
    None
        This function performs operations for side effects (updating the 
        font manager) and does not return a value.

    Notes
    -----
    - If a file at a given path does not exist, it is silently skipped.
    - If the font manager fails to add a font (e.g., corrupted file), the 
      exception is suppressed to ensure the pipeline continues without interruption.
    """
    # Lazy import to avoid overhead if the function is never invoked
    from matplotlib import font_manager as fm

    for p in local_paths:
        if os.path.exists(p):
            try:
                # Attempt to register the font with Matplotlib's font manager
                fm.fontManager.addfont(p)
            except Exception:
                # Fail silently to maintain robustness if a specific font file is invalid
                pass


def use_nature_style(
        prefer_arial=True,
        arial_candidates=(
                "fonts/Arial.ttf",
                "fonts/ArialMT.ttf",
                "fonts/Helvetica.ttf",
                "fonts/LiberationSans-Regular.ttf",
                "fonts/Arial Unicode.ttf",
        ),
):
    """
    Configures global Matplotlib and Seaborn settings to approximate the 
    'Nature' journal style guide.

    This function updates `matplotlib.rcParams` with specific values for 
    font sizes, line widths, and axis formatting. It also attempts to 
    apply a Seaborn theme for colour-blind accessibility.

    Parameters
    ----------
    prefer_arial : bool, optional
        If True, attempts to register Arial-like fonts from the `arial_candidates` 
        paths. Defaults to True.
    arial_candidates : tuple of str, optional
        A tuple of file paths to check for font files. These are prioritised 
        to ensure consistent rendering across different operating systems.

    Returns
    -------
    None
        Modifies the global state of the plotting library in-place.

    Raises
    ------
    No exceptions are explicitly raised; failures in font registration or 
    Seaborn importation are handled gracefully.
    """
    # Attempt to load local font files if the user prioritises Arial
    if prefer_arial:
        _try_register_arial(arial_candidates)

    # Update the global configuration dictionary (rcParams) for Matplotlib
    mpl.rcParams.update({
        # --- Font Configuration ---
        # Prioritise sans-serif fonts. The list acts as a fallback mechanism:
        # if Arial is missing, it tries Helvetica, then Liberation Sans, etc.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "Nimbus Sans"],

        # --- Output Format ---
        # Force generation of Type 42 fonts (TrueType) in PostScript/PDF.
        # This embeds the font subset, keeping text editable in vector editors 
        # (e.g., Illustrator, Inkscape) rather than converting it to paths.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Disable internal LaTeX rendering to avoid dependency on a local TeX distribution.
        # 'stixsans' is selected for math text to blend aesthetically with Arial.
        "text.usetex": False,
        "mathtext.fontset": "stixsans",

        # --- Figure Dimensions & Quality ---
        # Set base DPI for screen display (300) and high-resolution export (600).
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.transparent": True,
        "savefig.bbox": "tight",  # Automatically trim whitespace around figures

        # --- Axes & Spines ---
        # Configure axis aesthetics to match minimalist scientific standards.
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "axes.grid": False,  # Gridlines are typically discouraged in Nature-style plots
        "axes.titlesize": 9,
        "axes.labelsize": 8,

        # --- Ticks ---
        # Set ticks to point outwards to prevent data overlap.
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Define dimensions for major and minor ticks for visual hierarchy
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,

        # Tick label font sizes
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,

        # --- Legend ---
        # Remove the box frame around legends and adjust font size
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "legend.handlelength": 1.2,

        # --- Plot Elements ---
        # Set default line widths and marker sizes to be legible but fine
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,

        # Ensure patch edges (e.g., in histograms/bar charts) are thin but visible
        "patch.edgecolor": "black",
        "patch.linewidth": 0.5,
    })

    # --- Seaborn Integration ---
    # Attempt to apply Seaborn styling if the library is installed.
    try:
        import seaborn as sns

        # Set context to "paper" (scales elements down) and style to "ticks" (minimalist).
        # Use "colorblind" palette to ensure accessibility.
        sns.set_theme(
            context="paper",
            style="ticks",
            palette="colorblind",
        )

        # Explicitly remove the top and right spines for the classic "half-box" look.
        sns.set_style({"axes.spines.right": False, "axes.spines.top": False})

        # Re-enforce the paper context to ensure font scaling consistency.
        sns.set_context("paper")

    except Exception:
        # If Seaborn is unavailable, the Matplotlib rcParams set above provide a 
        # sufficient fallback, so we suppress the error.
        pass

    # Disable automatic layout adjustments globally.
    # This forces the user or calling code to explicitly handle layout 
    # (e.g., via `plt.tight_layout()` or `constrained_layout`), preventing 
    # unexpected shifting of axes during figure generation.
    mpl.rcParams["figure.autolayout"] = False