# src/reports/graphs/cost_metrics.py

"""
📊 Cost Metrics Visualization Module
------------------------------------

This module handles the generation of visualization artefacts related to computational
costs, energy consumption, and performance metrics for distributed machine learning experiments.

🧠 Purpose:
    To provide a comprehensive graphical analysis of the training lifecycle, enabling
    researchers to audit resource utilization, carbon footprint, and network overhead
    across client-server architectures.

🔧 Core Functionalities:
    • Parse and process raw cost data from JSON logs
    • Generate standardized Matplotlib figures for reporting
    • Visualize energy consumption versus time correlations
    • Analyze network traffic (RX/TX) and I/O operations
    • Compute and display cumulative cost timelines and KPIs

🎯 Intended Use:
    • Automated report generation pipelines
    • Performance analysis of Federated Learning systems
    • Environmental impact assessments of AI models

📁 Dependencies:
    • matplotlib
    • numpy
    • json
    • pathlib

📝 Notes:
    This module assumes the existence of a `costs.json` file within the experiment
    directory structure. It operates statelessly with respect to the broader
    application but relies on specific schema conventions within the JSON input.

Author: Andrea Moleri
File Location: src/reports/graphs/cost_metrics.py
Last Modified: 20/11/2025
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve the project root directory to ensure absolute imports work correctly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict

from logs.logger import get_logger

# Initialize module-level logger
logger = get_logger(__name__)

# Default dots per inch for high-resolution raster output
_DEF_FIG_DPI = 300


# Consistent styling with main report
def _ax_minimal(ax):
    """
    Applies a consistent, minimal aesthetic to a Matplotlib axis.

    Removes top and right spines and adjusts tick parameters to reduce visual clutter,
    aligning with the publication-quality style of the report.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to style.

    Returns
    -------
    matplotlib.axes.Axes
        The styled axis object.
    """
    """Consistent axis styling with main report"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.6)
    return ax


def _create_consistent_figure(title=None, subtitle=None, figsize=(8, 8)):
    """
    Initializes a Matplotlib figure with standardized dimensions and typography.

    This factory function ensures that all generated pages in the PDF report share
    a uniform layout, font weight, and title positioning.

    Parameters
    ----------
    title : str, optional
        The main title of the figure.
    subtitle : str, optional
        A secondary title or description placed below the main title.
    figsize : tuple[float, float], optional
        The width and height of the figure in inches. Defaults to (8, 8).

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        A tuple containing the generated figure and the primary axis.
    """
    """
    Create figure with consistent styling that matches other pages.
    Uses same approach as main report figures.
    """
    fig = plt.figure(figsize=figsize, dpi=_DEF_FIG_DPI)

    # Add titles with consistent styling (no bold, same font sizes)
    if title:
        # Use same styling as other pages - no bold, consistent font size
        fig.text(0.5, 0.98, title, ha='center', va='top', fontsize=12, weight='normal')  # Changed to normal weight
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha='center', va='top', fontsize=9, style='italic')

    # Consistent axes positioning with other pages
    ax = fig.add_axes([0.1, 0.12, 0.85, 0.78])
    _ax_minimal(ax)
    return fig, ax


def _load_costs_json(costs_json_path: Path) -> dict | None:
    """
    Safely loads the costs metrics from a JSON file.

    Parameters
    ----------
    costs_json_path : Path
        The filesystem path to the `costs.json` file.

    Returns
    -------
    dict | None
        The parsed dictionary if successful, or None if an error occurs
        (e.g., file not found, malformed JSON).
    """
    try:
        with open(costs_json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Cannot load costs JSON at {costs_json_path}: {e}")
        return None


def _fmt_hms_val(seconds: float) -> str:
    """
    Formats a time duration in seconds into a human-readable `HH:MM:SS` string.

    Parameters
    ----------
    seconds : float
        The duration in seconds.

    Returns
    -------
    str
        The formatted time string (e.g., "01:30:45").
    """
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600);
    m = int((seconds % 3600) // 60);
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_energy_kwh(v: float) -> str:
    """
    Formats energy values with appropriate unit scaling (kWh, Wh, mWh).

    Parameters
    ----------
    v : float
        Energy value in Kilowatt-hours (kWh).

    Returns
    -------
    str
        A string representing the formatted energy with the most appropriate unit.
    """
    v = float(v)
    if v >= 1e-3: return f"{v:.3f} kWh"
    wh = v * 1_000.0
    if wh >= 1e-3: return f"{wh:.3f} Wh"
    mwh = wh * 1_000.0
    return f"{mwh:.3f} mWh"


def _fmt_emissions(kg: float) -> str:
    """
    Formats carbon emission values with appropriate unit scaling (kg, g, mg).

    Parameters
    ----------
    kg : float
        Mass of CO2 equivalent in Kilograms.

    Returns
    -------
    str
        A string representing the formatted mass with LaTeX formatting for CO2.
    """
    kg = float(kg)
    if kg >= 1e-3: return f"{kg:.3f} kg CO$_2$e"
    g = kg * 1_000.0
    if g >= 1e-3: return f"{g:.3f} g CO$_2$e"
    mg = g * 1_000.0
    return f"{mg:.3f} mg CO$_2$e"


def _try_emissions(costs: dict, energy_data: dict) -> tuple[float | None, float | None]:
    """
    Attempts to derive carbon emissions and carbon intensity from cost data.

    This function employs a robust fallback strategy, checking multiple potential
    dictionary keys for emission data. It can also calculate missing values if
    sufficient partial data (e.g., Intensity and Total Energy) is present.

    Parameters
    ----------
    costs : dict
        The top-level costs dictionary containing general metrics.
    energy_data : dict
        The specific sub-dictionary containing energy metrics.

    Returns
    -------
    tuple[float | None, float | None]
        A tuple containing:
        - Total CO2 emissions (kg)
        - Carbon intensity (kgCO2/kWh)
    """
    co2_kg = None;
    intensity = None
    em = costs.get("emissions", {}) or {}

    # Attempt to extract from the top-level 'emissions' dictionary
    if isinstance(em, dict):
        co2_kg = (
                em.get("co2_kg") or em.get("co2e_kg") or
                (float(em.get("co2_tonnes")) * 1000.0 if em.get("co2_tonnes") else None) or
                em.get("emissions_kg_co2e")
        )
        intensity = (
                em.get("kgco2_per_kwh") or em.get("kg_co2_per_kwh") or
                em.get("intensity_kgco2_per_kwh") or em.get("emissions_factor_kg_per_kwh") or
                em.get("emissions_kgco2_per_kwh")
        )

    # Fallback: Attempt to extract from 'energy_data' if not found above
    if co2_kg is None:
        e = energy_data
        co2_kg = e.get("co2_kg") or (float(e.get("co2_tonnes")) * 1000.0 if e.get("co2_tonnes") else None) or e.get(
            "emissions_kg_co2e")
    if intensity is None:
        e = energy_data
        intensity = (
                e.get("kgco2_per_kwh") or e.get("kg_co2_per_kwh") or
                e.get("intensity_kgco2_per_kwh") or e.get("emissions_factor_kg_per_kwh") or
                (e.get("extra", {}) or {}).get("kgco2_per_kwh")
        )

    # Normalize types to float, handling potential casting errors gracefully
    try:
        co2_kg = float(co2_kg) if co2_kg is not None else None
    except:
        co2_kg = None
    try:
        intensity = float(intensity) if intensity is not None else None
    except:
        intensity = None
    try:
        kwh = float(energy_data.get("kwh")) if energy_data.get("kwh") is not None else None
    except:
        kwh = None

    # Mathematical inference: Calculate missing variable if other two are present
    if co2_kg is None and intensity is not None and kwh is not None:
        co2_kg = intensity * kwh
    if intensity is None and co2_kg is not None and kwh is not None and kwh > 0:
        intensity = co2_kg / kwh
    return co2_kg, intensity


def _get_phase_energy(phase_data):
    """
    Extracts energy metrics from a specific execution phase.

    Prioritizes direct hardware measurements (e.g., NVML) over estimated values
    derived from time-sharing ratios.

    Parameters
    ----------
    phase_data : dict
        Dictionary containing metrics for a specific phase.

    Returns
    -------
    tuple[float, float]
        A tuple containing (Energy in kWh, Emissions in kg CO2e).
        Returns (0.0, 0.0) if data is missing.
    """
    """Extract energy metrics from phase data, handling both measured and estimated values"""
    energy_data = phase_data.get('energy', {})

    # Try measured values first (direct NVML measurements)
    kwh = energy_data.get('kwh')
    emissions = energy_data.get('emissions_kg_co2e')

    # If measured values not available, try estimated values
    if kwh is None:
        kwh = energy_data.get('kwh_estimated_from_time_share', 0.0)
    if emissions is None:
        emissions = energy_data.get('emissions_kg_co2e_estimated', 0.0)

    return float(kwh or 0.0), float(emissions or 0.0)


def _extract_client_server_breakdown(phases: Dict) -> tuple[list, list, list, list]:
    """
    Separates performance metrics into client-side and server-side components.

    Parameters
    ----------
    phases : Dict
        Dictionary of phase data keyed by phase name.

    Returns
    -------
    tuple[list, list, list, list]
        A tuple containing four lists:
        - Client execution times
        - Server execution times
        - Client energy consumption
        - Server energy consumption
    """
    """Extract client vs server breakdown from phases"""
    client_times, server_times = [], []
    client_energy, server_energy = [], []

    for phase_name, phase_data in phases.items():
        time_sec = phase_data.get("wall_clock_sec", 0)
        kwh, _ = _get_phase_energy(phase_data)

        if "client" in phase_name.lower():
            client_times.append(time_sec)
            client_energy.append(kwh)
        elif "server" in phase_name.lower():
            server_times.append(time_sec)
            server_energy.append(kwh)

    return client_times, server_times, client_energy, server_energy


def _extract_phase_categories(phases: Dict) -> Dict[str, list]:
    """
    Classifies experiment phases into functional categories based on naming conventions.

    Categories include: 'generative', 'classifier', 'communication', 'evaluation', and 'other'.

    Parameters
    ----------
    phases : Dict
        Dictionary of phase data.

    Returns
    -------
    Dict[str, list]
        A dictionary mapping category names to lists of (phase_name, duration) tuples.
    """
    """Categorize phases by type (generative, classifier, etc.)"""
    categories = {
        "generative": [],
        "classifier": [],
        "communication": [],
        "evaluation": [],
        "other": []
    }

    for phase_name, phase_data in phases.items():
        time_sec = phase_data.get("wall_clock_sec", 0)
        # Filter out negligible phases to avoid noise in visualization
        if time_sec <= 0.1:
            continue

        phase_lower = phase_name.lower()
        if any(term in phase_lower for term in ["gen", "vae", "diffusion", "train_vae"]):
            categories["generative"].append((phase_name, time_sec))
        elif any(term in phase_lower for term in ["clf", "classif", "train_clf"]):
            categories["classifier"].append((phase_name, time_sec))
        elif any(term in phase_lower for term in ["broadcast", "ingest", "transfer", "network"]):
            categories["communication"].append((phase_name, time_sec))
        elif any(term in phase_lower for term in ["eval", "test", "metric"]):
            categories["evaluation"].append((phase_name, time_sec))
        else:
            categories["other"].append((phase_name, time_sec))

    return categories


# -------- NEW GRAPH FUNCTIONS --------

def _create_client_server_comparison(costs: Dict, phases: Dict, figures_dir: Path = None):
    """
    Generates a dual-pie-chart figure comparing Client vs. Server resource usage.

    Visualizes the proportion of Time and Energy consumed by client nodes versus
    the central server.

    Parameters
    ----------
    costs : Dict
        Global cost metrics.
    phases : Dict
        Phase-specific breakdown.
    figures_dir : Path, optional
        Directory to save the individual PNG file.

    Returns
    -------
    matplotlib.figure.Figure | None
        The generated figure, or None if insufficient data exists.
    """
    """Compare client vs server resource usage"""
    client_times, server_times, client_energy, server_energy = _extract_client_server_breakdown(phases)

    total_client_time = sum(client_times)
    total_server_time = sum(server_times)
    total_client_energy = sum(client_energy)
    total_server_energy = sum(server_energy)

    # Skip if no meaningful data
    if total_client_time + total_server_time < 0.1:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), dpi=_DEF_FIG_DPI)
    fig.suptitle("Client vs Server Resource Distribution", fontsize=12, weight='normal')
    fig.text(0.5, 0.95, "Comparison of computational load between clients and server",
             ha='center', va='top', fontsize=9, style='italic')

    # Time distribution
    labels = ['Client', 'Server']
    times = [total_client_time, total_server_time]
    colors = ['#3498DB', '#E74C3C']

    ax1.pie(times, labels=labels, autopct='%1.1f%%', colors=colors,
            startangle=90, textprops={'fontsize': 9})
    ax1.set_title('Time Distribution', fontsize=10, weight='normal')

    # Energy distribution
    energies = [total_client_energy, total_server_energy]
    if sum(energies) > 1e-6:
        ax2.pie(energies, labels=labels, autopct='%1.1f%%', colors=colors,
                startangle=90, textprops={'fontsize': 9})
    else:
        # Handle cases where energy monitoring was disabled or unavailable
        ax2.text(0.5, 0.5, "Energy data\nnot available",
                 ha='center', va='center', fontsize=10, color='gray')
        ax2.set_frame_on(False)
    ax2.set_title('Energy Distribution', fontsize=10, weight='normal')

    # Add absolute values as text annotations in a clean summary box
    time_text = f"Client: {_fmt_hms_val(total_client_time)}\nServer: {_fmt_hms_val(total_server_time)}"
    energy_text = f"Client: {_fmt_energy_kwh(total_client_energy)}\nServer: {_fmt_energy_kwh(total_server_energy)}"

    fig.text(0.25, 0.05, time_text, ha='center', va='bottom', fontsize=8,
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.7))
    fig.text(0.75, 0.05, energy_text, ha='center', va='bottom', fontsize=8,
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.7))

    plt.tight_layout(rect=[0, 0.1, 1, 0.93])

    if figures_dir:
        fig.savefig(figures_dir / "cost_client_server_comparison.png",
                    dpi=_DEF_FIG_DPI, bbox_inches="tight", facecolor='white', edgecolor='none')

    return fig


def _create_phase_categories_breakdown(phases: Dict, figures_dir: Path = None):
    """
    Generates a pie chart illustrating time distribution across functional categories.

    Aggregates phases into high-level concepts (e.g., Generative vs. Classification)
    to provide a macro view of the training workload.

    Parameters
    ----------
    phases : Dict
        Phase-specific breakdown.
    figures_dir : Path, optional
        Directory to save the individual PNG file.

    Returns
    -------
    matplotlib.figure.Figure | None
        The generated figure, or None if insufficient data exists.
    """
    """Show breakdown of time by phase categories"""
    categories = _extract_phase_categories(phases)

    category_totals = {}
    for category, phase_list in categories.items():
        category_totals[category] = sum(time for _, time in phase_list)

    # Filter out empty categories
    category_totals = {k: v for k, v in category_totals.items() if v > 0.1}

    if not category_totals:
        return None

    fig, ax = _create_consistent_figure(
        "Phase Categories Breakdown",
        "Time distribution across different types of operations"
    )

    labels = [cat.title() for cat in category_totals.keys()]
    times = list(category_totals.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(times, labels=labels, autopct='%1.1f%%',
                                      colors=colors, startangle=90)

    # Style the percentage labels inside the wedges
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)

    # Add absolute time values
    total_time = sum(times)
    ax.text(0, -1.3, f"Total Time: {_fmt_hms_val(total_time)}",
            ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    if figures_dir:
        fig.savefig(figures_dir / "cost_phase_categories.png",
                    dpi=_DEF_FIG_DPI, bbox_inches="tight", facecolor='white', edgecolor='none')

    return fig


def _create_network_traffic_breakdown(costs: Dict, phases: Dict, figures_dir: Path = None):
    """
    Generates a detailed analysis of network traffic (RX/TX).

    Creates two subplots:
    1. Total aggregate traffic (Download vs Upload).
    2. Breakdown of traffic per phase (if phase-level granularity is available).

    Parameters
    ----------
    costs : Dict
        Global cost metrics including aggregate byte counts.
    phases : Dict
        Phase-specific metrics.
    figures_dir : Path, optional
        Directory to save the individual PNG file.

    Returns
    -------
    matplotlib.figure.Figure | None
        The generated figure, or None if no network traffic was recorded.
    """
    """Show detailed network traffic breakdown by phase"""
    bytes_io = costs.get("bytes", {})
    network_rx = bytes_io.get("network_rx_bytes", 0)
    network_tx = bytes_io.get("network_tx_bytes", 0)

    if network_rx + network_tx == 0:
        return None

    # Extract network traffic by phase
    phase_network = {}
    for phase_name, phase_data in phases.items():
        rx = phase_data.get("network_rx_bytes", 0)
        tx = phase_data.get("network_tx_bytes", 0)
        if rx + tx > 0:
            phase_network[phase_name] = {"rx": rx, "tx": tx}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=_DEF_FIG_DPI)
    fig.suptitle("Network Traffic Analysis", fontsize=12, weight='normal')

    # Overall network traffic
    labels = ['Download (RX)', 'Upload (TX)']
    traffic = [network_rx, network_tx]
    colors = ['#2ECC71', '#E74C3C']

    ax1.bar(labels, [t / 1e9 for t in traffic], color=colors,
            edgecolor='black', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel('Gigabytes (GB)', fontsize=9)
    ax1.set_title('Total Network Traffic', fontsize=10, weight='normal')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    for i, v in enumerate(traffic):
        ax1.text(i, v / 1e9, f'{v / 1e9:.2f} GB',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Network traffic by phase (if available)
    if phase_network:
        phase_names = []
        phase_rx, phase_tx = [], []

        for phase_name, traffic_data in phase_network.items():
            phase_names.append(phase_name.replace('_', ' ').title())
            phase_rx.append(traffic_data['rx'] / 1e6)  # MB
            phase_tx.append(traffic_data['tx'] / 1e6)  # MB

        x = np.arange(len(phase_names))
        width = 0.35

        ax2.bar(x - width / 2, phase_rx, width, label='Download', color='#2ECC71', alpha=0.8)
        ax2.bar(x + width / 2, phase_tx, width, label='Upload', color='#E74C3C', alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(phase_names, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Megabytes (MB)', fontsize=9)
        ax2.set_title('Network Traffic by Phase', fontsize=10, weight='normal')
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, "Per-phase network data not available",
                 ha='center', va='center', fontsize=10, color='gray')
        ax2.set_frame_on(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if figures_dir:
        fig.savefig(figures_dir / "cost_network_traffic.png",
                    dpi=_DEF_FIG_DPI, bbox_inches="tight", facecolor='white', edgecolor='none')

    return fig



def _create_energy_time_correlation(phases: Dict, figures_dir: Path = None, top_k: int = 10):
    """
    Creates a scatter plot correlating Phase Duration (X-axis) vs. Energy Consumption (Y-axis).

    Features:
    - Color mapping based on Average Power (kW).
    - Smart labeling for the top-K most energy-intensive phases.
    - A force-directed label placement algorithm to prevent text overlap.

    Parameters
    ----------
    phases : Dict
        Phase-specific metrics.
    figures_dir : Path, optional
        Directory to save the individual PNG file.
    top_k : int, optional
        Number of top energy consumers to label. Defaults to 10.

    Returns
    -------
    matplotlib.figure.Figure | None
        The generated figure, or None if insufficient data points exist.
    """
    """
    Sleek scatter: Energy vs Time con colorbar per potenza media.
    - Etichette solo per i top-K punti (per energia) per evitare clutter.
    - Etichette con leader line e algoritmo di repulsione per ridurre le sovrapposizioni.
    """
    # ----- prepara dati
    phase_data = []
    for phase_name, phase_info in phases.items():
        time_sec = phase_info.get("wall_clock_sec", 0)
        kwh, _ = _get_phase_energy(phase_info)
        if time_sec > 0.1 and kwh > 1e-6:
            power_kw = (kwh * 3600) / time_sec  # kW
            phase_data.append({
                "name": phase_name,
                "time": float(time_sec),
                "energy": float(kwh),
                "power_kw": float(power_kw),
            })

    if len(phase_data) < 3:
        return None

    # Ordina per energia (decrescente) e seleziona top-K da etichettare
    phase_data.sort(key=lambda d: d["energy"], reverse=True)
    labels_data = phase_data[: max(1, min(top_k, len(phase_data)))]
    all_times = np.array([d["time"] for d in phase_data], dtype=float)
    all_energy = np.array([d["energy"] for d in phase_data], dtype=float)
    all_power = np.array([d["power_kw"] for d in phase_data], dtype=float)

    # ----- figura/assi
    fig, ax = _create_consistent_figure(
        "Energy vs Time Correlation",
        "Runtime vs energy consumption across phases (labels on top values)"
    )

    # Stile assi pulito
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.22, linestyle=":", linewidth=0.7)

    # ----- scatter
    pts = ax.scatter(
        all_times, all_energy,
        c=all_power, cmap="viridis",
        s=90, alpha=0.88,
        edgecolors="white", linewidth=0.6
    )

    # ----- formattazione assi
    from matplotlib.ticker import FuncFormatter

    def _fmt_hms(s: float) -> str:
        s = max(float(s), 0.0)
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:d}:{sec:02d}"

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _=None: _fmt_hms(x)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _=None: f"{y:.2f} kWh"))
    ax.set_xlabel("Time (hh:mm:ss)", fontsize=9)
    ax.set_ylabel("Energy (kWh)", fontsize=9)

    # ----- colorbar (potenza media in kW)
    cbar = plt.colorbar(pts, ax=ax, shrink=0.82, aspect=28, pad=0.02)
    cbar.set_label("Average Power (kW)", fontsize=8)

    # ----- trendline leggera (se sufficiente)
    if len(all_times) >= 4:
        order = np.argsort(all_times)
        z = np.polyfit(all_times, all_energy, 1)
        p = np.poly1d(z)
        ax.plot(all_times[order], p(all_times[order]),
                color="#666666", linestyle="--", linewidth=1, alpha=0.6)

    # ----- etichette "sleek" per top-K con repulsione
    import matplotlib.patheffects as pe
    anns = []
    # direzioni iniziali alternate per ridurre collisioni (offset in punti)
    base_offsets = [(10, 10), (-10, 10), (10, -10), (-10, -10), (14, 6), (-14, 6)]

    for i, d in enumerate(labels_data):
        name = d["name"].replace("_", " ").title()
        x, y = d["time"], d["energy"]
        dx, dy = base_offsets[i % len(base_offsets)]
        ann = ax.annotate(
            name, (x, y),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8, ha="left", va="bottom",
            arrowprops=dict(
                arrowstyle="-",
                color="0.3",
                shrinkA=0, shrinkB=0,
                lw=0.8, alpha=0.6
            ),
            bbox=None,
            path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.9)]
        )
        anns.append(ann)

    # Piccolo solver di repulsione tra bounding box (in display coords)
    def _adjust_labels(annotations, iterations=200, step=0.5):
        """
        Iterative algorithm to repel overlapping annotation bounding boxes.
        Moves overlapping labels in opposite Y directions.
        """
        fig.canvas.draw()  # necessario per avere i bbox corretti
        renderer = fig.canvas.get_renderer()
        for _ in range(iterations):
            moved = False
            bboxes = [a.get_window_extent(renderer=renderer).expanded(1.02, 1.08) for a in annotations]
            # sposta etichette che collidono
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    if bboxes[i].overlaps(bboxes[j]):
                        # spingi in direzioni opposte lungo y (più naturale visivamente)
                        pi = annotations[i].get_position()  # (xo, yo) in offset points
                        pj = annotations[j].get_position()
                        annotations[i].set_position((pi[0], pi[1] + step))
                        annotations[j].set_position((pj[0], pj[1] - step))
                        moved = True
            if not moved:
                break
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

    if len(anns) > 1:
        _adjust_labels(anns, iterations=250, step=0.8)

    # margini comodi
    ax.margins(x=0.05, y=0.08)

    if figures_dir:
        fig.savefig(
            figures_dir / "cost_energy_time_correlation.png",
            dpi=_DEF_FIG_DPI, bbox_inches="tight",
            facecolor="white", edgecolor="none"
        )

    return fig



def _create_cumulative_cost_timeline(phases: Dict, figures_dir: Path = None):
    """
    Generates a dual-axis chart showing the accumulation of Time and Energy over the experiment's lifecycle.

    Approximates the chronological order of phases by sorting them by duration
    (assuming duration correlates with sequence complexity in this context),
    then plotting the cumulative sum.

    Parameters
    ----------
    phases : Dict
        Phase-specific metrics.
    figures_dir : Path, optional
        Directory to save the individual PNG file.

    Returns
    -------
    matplotlib.figure.Figure | None
        The generated figure, or None if no phases exist.
    """
    """Show cumulative buildup of costs over time"""
    # Sort phases by start time (approximated)
    sorted_phases = sorted(
        [(name, data.get("wall_clock_sec", 0), _get_phase_energy(data)[0])
         for name, data in phases.items() if data.get("wall_clock_sec", 0) > 0.1],
        key=lambda x: x[1]  # Sort by duration as proxy for timing
    )

    if not sorted_phases:
        return None

    phase_names, durations, energies = zip(*sorted_phases)

    # Calculate cumulative values
    cumulative_time = np.cumsum(durations)
    cumulative_energy = np.cumsum(energies)

    fig, ax1 = _create_consistent_figure(
        "Cumulative Cost Timeline",
        "Progressive accumulation of time and energy costs"
    )

    # Plot cumulative time
    color1 = '#3498DB'
    ax1.fill_between(range(len(phase_names)), [0] + list(cumulative_time[:-1]), cumulative_time,
                     alpha=0.3, color=color1, step='post')
    ax1.step(range(len(phase_names) + 1), [0] + list(cumulative_time),
             where='post', color=color1, linewidth=2, label='Cumulative Time')
    ax1.set_ylabel('Cumulative Time (seconds)', fontsize=9, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Plot cumulative energy on second axis
    ax2 = ax1.twinx()
    color2 = '#E74C3C'
    if sum(energies) > 1e-6:
        ax2.fill_between(range(len(phase_names)), [0] + list(cumulative_energy[:-1]), cumulative_energy,
                         alpha=0.3, color=color2, step='post')
        ax2.step(range(len(phase_names) + 1), [0] + list(cumulative_energy),
                 where='post', color=color2, linewidth=2, label='Cumulative Energy')
    ax2.set_ylabel('Cumulative Energy (kWh)', fontsize=9, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Configure x-axis
    ax1.set_xticks(range(len(phase_names)))
    ax1.set_xticklabels([name.replace('_', ' ').title() for name in phase_names],
                        rotation=45, ha='right', fontsize=8)
    ax1.set_xlabel('Phases (in approximate order)', fontsize=9)

    # Add final values annotation
    final_time = cumulative_time[-1]
    final_energy = cumulative_energy[-1] if sum(energies) > 1e-6 else 0

    ax1.text(0.02, 0.98, f'Final Time: {_fmt_hms_val(final_time)}',
             transform=ax1.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle="round", facecolor=color1, alpha=0.2))

    if final_energy > 0:
        ax2.text(0.02, 0.90, f'Final Energy: {final_energy:.3f} kWh',
                 transform=ax1.transAxes, fontsize=8, va='top',
                 bbox=dict(boxstyle="round", facecolor=color2, alpha=0.2))

    ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

    if figures_dir:
        fig.savefig(figures_dir / "cost_cumulative_timeline.png",
                    dpi=_DEF_FIG_DPI, bbox_inches="tight", facecolor='white', edgecolor='none')

    return fig

def append_new_cost_pages(pdf: PdfPages, experiment_dir: Path, figures_dir: Path = None) -> int:
    """
    Generates and appends the set of "new" auxiliary cost metrics graphs to the PDF report.

    This function orchestrates the creation of:
    - Client/Server comparisons
    - Category breakdowns
    - Network traffic analysis
    - Energy/Time correlations
    - Cumulative timelines

    Parameters
    ----------
    pdf : PdfPages
        The open PDF file object to write to.
    experiment_dir : Path
        Root directory of the experiment containing the `costs` subdirectory.
    figures_dir : Path, optional
        Directory to save individual PNG files.

    Returns
    -------
    int
        The number of pages successfully added to the PDF.
    """
    """
    Add all new cost metrics pages to PDF.
    Returns the number of pages added.
    """
    pages_added = 0

    costs_json = experiment_dir / "costs" / "costs.json"
    costs = _load_costs_json(costs_json)
    if not costs:
        return pages_added

    try:
        phases = costs["phases"]
        energy = costs["energy"]
        flops = costs["flops"]
        bytes_io = costs["bytes"]
    except KeyError:
        return pages_added

    # Create all new graphs
    new_graphs = [
        _create_client_server_comparison(costs, phases, figures_dir),
        _create_phase_categories_breakdown(phases, figures_dir),
        _create_network_traffic_breakdown(costs, phases, figures_dir),
        _create_energy_time_correlation(phases, figures_dir),
        _create_cumulative_cost_timeline(phases, figures_dir),
    ]

    # Add valid graphs to PDF
    for graph in new_graphs:
        if graph is not None:
            pdf.savefig(graph, bbox_inches="tight")
            plt.close(graph)
            pages_added += 1

    return pages_added


# -------- MAIN APPEND FUNCTION --------

def append_cost_pages(pdf: PdfPages, experiment_dir: Path, figures_dir: Path = None) -> int:
    """
    Orchestrates the generation of the comprehensive Cost & Compute Metrics section of the report.

    This is the main entry point for this module. It loads the data and systematically
    generates a series of pages including:
    1. Training Phase Timeline (Gantt-style)
    2. Relative Time Distribution
    3. Energy & Emissions Distribution
    4. Power Consumption Profile
    5. Carbon Emissions Summary
    6. FLOPs Analysis
    7. I/O and Data Transfer Log
    8. Key Performance Indicators (KPI) cards
    9. Additional metrics (via `append_new_cost_pages`)

    Parameters
    ----------
    pdf : PdfPages
        The open PDF file object to write to.
    experiment_dir : Path
        Root directory of the experiment containing the `costs` subdirectory.
    figures_dir : Path, optional
        Directory to save individual PNG files.

    Returns
    -------
    int
        The total number of pages added to the PDF report.
    """
    """
    Add cost/compute metrics pages to PDF.
    Returns the number of pages added.
    """
    pages_added = 0

    costs_json = experiment_dir / "costs" / "costs.json"
    costs = _load_costs_json(costs_json)
    if not costs:
        logger.info(f"No costs.json found at {costs_json}, skipping cost pages.")
        return pages_added

    try:
        phases = costs["phases"]
        energy = costs["energy"]
        flops = costs["flops"]
        bytes_io = costs["bytes"]
    except KeyError as e:
        logger.warning(f"costs.json missing key {e}, skipping cost pages.")
        return pages_added

    co2_kg, kgco2_per_kwh = _try_emissions(costs, energy)

    # -------- PAGE: Training Phase Timeline --------
    # Visualizes the execution sequence of phases in a Gantt-chart style
    import re

    phase_names = list(phases.keys())
    phase_times = [phases[p]["wall_clock_sec"] for p in phase_names]
    starts = np.cumsum([0] + phase_times[:-1]).tolist()
    y = np.arange(len(phase_names))
    fig, ax = _create_consistent_figure("Training Phase Timeline", "Runtime stacked by phase")
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(phase_names))]
    ax.barh(y, phase_times, left=starts, height=0.6, color=colors, edgecolor="black", linewidth=0.5)

    # ---- A) Etichette tempo all'esterno, lato destro (HH:MM:SS) ----
    total_time = sum(phase_times)
    right_pad = max(total_time * 0.06, 1.0)  # margine per non tagliare il testo
    ax.set_xlim(0, total_time + right_pad)

    for i, (s, d) in enumerate(zip(starts, phase_times)):
        if d > 0:
            ax.text(
                s + d + total_time * 0.01,  # piccolo offset a destra della barra
                i,
                _fmt_hms_val(d),
                ha="left",
                va="center",
                fontsize=8,
                color="black",
                clip_on=False,
            )

    def _label_dash_style(key: str) -> str:
        """
        Normalizes internal phase keys into human-readable dashboard labels.
        Handles regex matching for dynamic keys (e.g., client indices).
        """
        # Casi espliciti (server)
        if key == "clf_cserver":
            return "Classification - Server"
        if key in {"server_eval", "srv_eval", "evaluation_server", "server_evaluation"}:
            return "Evaluation - Server"
        if key in {"server_classifier", "server_train", "srv_train", "server_training"}:
            return "Training - Server"

        # ⚠️ NUOVO: Server generation per client
        m = re.match(r"server_generation_client_(\d+)$", key)  # es. server_generation_client_000 -> Generation - Client 0 (Server)
        if m:
            return f"Generation - Client {int(m.group(1))} (Server)"

        # Client: pattern comuni
        m = re.match(r"train_c(\d+)$", key)  # es. train_c0 -> Training - Client 0
        if m:
            return f"Training - Client {int(m.group(1))}"

        m = re.match(r"client_(\d+)_gen$", key)  # es. client_000_gen -> Generation - Client 0
        if m:
            return f"Generation - Client {int(m.group(1))}"

        m = re.match(r"gen_c(\d+)$", key)  # es. gen_c0 -> Generation - Client 0
        if m:
            return f"Generation - Client {int(m.group(1))}"

        # Upload/Download phases
        m = re.match(r"client_(\d+)_upload$", key)
        if m:
            return f"Upload - Client {int(m.group(1))}"

        m = re.match(r"client_(\d+)_download$", key)
        if m:
            return f"Download - Client {int(m.group(1))}"

        if key == "server_ingest":
            return "Ingest - Server"

        # Fallback generici
        label = key.replace("_", " ").title().strip()

        if label.startswith("Server "):
            stage = label[len("Server "):].strip()
            return f"{stage} - Server"

        m = re.match(r"(.*)\s+Server$", label)
        if m:
            return f"{m.group(1).strip()} - Server"

        m = re.match(r"Client\s+0*([0-9]+)\s+(.*)", label)
        if m:
            return f"{m.group(2).strip()} - Client {int(m.group(1))}"

        return label

    ax.set_yticks(y)
    ax.set_yticklabels([_label_dash_style(p) for p in phase_names])
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative Time (s)", fontsize=9)

    # Export individual figure if requested
    if figures_dir:
        fig.savefig(figures_dir / "cost_training_timeline.png", dpi=_DEF_FIG_DPI, bbox_inches="tight",
                    facecolor='white', edgecolor='none')

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    pages_added += 1

    # -------- PAGE: Relative Time Distribution --------
    # Bar chart showing percentage of total runtime for each phase

    # Get significant phases for time distribution
    significant_phases = {k: v for k, v in phases.items() if v.get("wall_clock_sec", 0) > 0.1}
    if significant_phases:
        # ⬇️ NUOVA NOMENCLATURA
        names = [_label_dash_style(k) for k in significant_phases.keys()]
        times = [v["wall_clock_sec"] for v in significant_phases.values()]  # in secondi

        total_time = sum(times)
        perc = np.array([t / total_time * 100 for t in times]) if total_time > 0 else np.zeros(len(times))
        names = np.array(names)
        times = np.array(times)

        idx = np.argsort(perc)[::-1]
        perc, names, times = perc[idx], names[idx], times[idx]

        fig, ax = _create_consistent_figure("Relative Time Distribution Across Training Phases")
        bars = ax.barh(range(len(names)), perc,
                       color=[plt.get_cmap("tab10")(i % 10) for i in range(len(names))],
                       edgecolor="black", linewidth=0.5)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Percentage of Total Runtime (%)", fontsize=9)

        # spazio per le etichette a destra
        max_perc = float(np.max(perc)) if perc.size else 100.0
        ax.set_xlim(right=max_perc * 1.30 if max_perc > 0 else 1.0)
        x_off = max_perc * 0.01

        # Etichette: "XX.X% (HH:MM:SS)" a destra della percentuale
        for i, (b, p, t) in enumerate(zip(bars, perc, times)):
            ax.text(p + x_off, i, f" {p:.1f}% ({_fmt_hms_val(t)})", va="center", ha="left", fontsize=8)

        if figures_dir:
            fig.savefig(figures_dir / "cost_relative_time_distribution.png", dpi=_DEF_FIG_DPI,
                        bbox_inches="tight", facecolor='white', edgecolor='none')

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages_added += 1

    # -------- PAGE: Energy & Emissions Distribution by Phase --------
    # Combined bar chart for energy and CO2 contribution per phase

    ph_labels, ph_kwh, ph_co2 = [], [], []
    for k, v in phases.items():
        if v.get("wall_clock_sec", 0) > 0.1:  # Only significant phases
            # ⬇️ NUOVA NOMENCLATURA
            label = _label_dash_style(k)
            kwh, emission = _get_phase_energy(v)  # kWh, kg CO2e (tipicamente)
            if kwh > 1e-6:  # Only phases with measurable energy
                ph_labels.append(label)
                ph_kwh.append(kwh)
                ph_co2.append(emission)

    if ph_kwh:  # Only create if we have energy data
        total_kwh = sum(ph_kwh)
        per = np.array([(k / total_kwh * 100) if total_kwh > 0 else 0 for k in ph_kwh])
        ph_labels = np.array(ph_labels)
        ph_co2 = np.array(ph_co2)
        ph_kwh = np.array(ph_kwh)

        order = np.argsort(per)[::-1]
        per, ph_labels, ph_co2, ph_kwh = per[order], ph_labels[order], ph_co2[order], ph_kwh[order]

        subtitle = f"Total Consumption: {energy['kwh']:.3f} kWh | Total Emissions: {(f'{co2_kg:.3f} kg CO$_2$e' if co2_kg is not None else 'N/A')}"
        fig, ax = _create_consistent_figure("Energy & Emissions Distribution by Phase", subtitle)
        bars = ax.barh(range(len(ph_labels)), per,
                       color=[plt.get_cmap("tab10")(i % 10) for i in range(len(ph_labels))],
                       edgecolor="black", linewidth=0.5, height=0.6)

        ax.set_yticks(range(len(ph_labels)))
        ax.set_yticklabels(ph_labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Percentage of Total Energy/Emissions (%)", fontsize=9)

        # spazio per etichette estese (Wh e g CO2e)
        max_perc = float(np.max(per)) if per.size else 1.0
        ax.set_xlim(right=max_perc * 1.35 if max_perc > 0 else 1.0)
        x_off = max_perc * 0.01

        # Etichette: "XX.X% (YY.YY Wh / ZZ.ZZ g CO2e)" a destra della percentuale
        for i, (b, p, kwh, c_kg) in enumerate(zip(bars, per, ph_kwh, ph_co2)):
            wh = kwh * 1000.0  # converti kWh -> Wh
            co2_g = (c_kg or 0.0) * 1000  # kg -> g (gestione None)
            ax.text(p + x_off, i, f" {p:.1f}% ({wh:.2f} Wh / {co2_g:.2f} g CO$_2$e)",
                    va="center", ha="left", fontsize=7)

        # Footnote sorgente energia
        energy_source = "direct NVML measurements" if any(
            'kwh' in phases[p].get('energy', {}) for p in phases) else "time-proportional estimates"
        ax.text(0.99, 0.01, f"Phase energy from {energy_source}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="gray")

        if figures_dir:
            fig.savefig(figures_dir / "cost_energy_emissions_distribution.png", dpi=_DEF_FIG_DPI,
                        bbox_inches="tight", facecolor='white', edgecolor='none')

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pages_added += 1

    # -------- PAGE: Energy & Power --------
    # Summary of Total Energy (kWh) and Average/Peak Power (W)
    fig = plt.figure(figsize=(8, 8), dpi=_DEF_FIG_DPI)
    fig.text(0.5, 0.98, "Energy Consumption & Power", ha="center", va="top", fontsize=12,
             weight='normal')
    fig.text(0.5, 0.955, "Total consumption (kWh) and average/peak power draw (W)", ha="center", va="top", fontsize=9,
             style="italic")

    ax_kwh = fig.add_axes([0.15, 0.55, 0.7, 0.3])
    _ax_minimal(ax_kwh)
    tot_kwh = energy["kwh"]
    b = ax_kwh.bar(["Total Consumption"], [tot_kwh], color="#3498DB", width=0.4, edgecolor="black", linewidth=0.5)
    ax_kwh.text(b[0].get_x() + b[0].get_width() / 2., tot_kwh, f"{tot_kwh:.3f} kWh",
                ha="center", va="bottom", fontsize=9, weight="bold")
    ax_kwh.set_ylabel("Energy (kWh)", fontsize=9)
    ax_kwh.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    ax_p = fig.add_axes([0.15, 0.15, 0.7, 0.3])
    _ax_minimal(ax_p)
    try:
        avg_p = np.mean([energy["extra"]["nvml_power_start_W"], energy["extra"]["nvml_power_end_W"]])
        peak_p = max(energy["extra"]["nvml_power_start_W"], energy["extra"]["nvml_power_end_W"])
        cats = ["Average Draw", "Peak Draw"]
        vals = [avg_p, peak_p]
        bp = ax_p.bar(cats, vals, color=[plt.get_cmap("tab10")(3), plt.get_cmap("tab10")(4)],
                      width=0.4, edgecolor="black", linewidth=0.5)
        for bb, v in zip(bp, vals):
            ax_p.text(bb.get_x() + bb.get_width() / 2., v, f"{v:.1f} W",
                      ha="center", va="bottom", fontsize=9, weight="bold")
        ax_p.set_ylabel("Power (Watts)", fontsize=9)
    except KeyError:
        ax_p.text(0.5, 0.5, "Power data not available", ha="center",
                  va="center", fontsize=10, color="gray")

    ax_p.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    if figures_dir:
        fig.savefig(figures_dir / "cost_energy_power.png", dpi=_DEF_FIG_DPI, bbox_inches="tight",
                    facecolor='white', edgecolor='none')

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    pages_added += 1

    # -------- PAGE: Carbon Emissions --------
    # Visualizes Total Emissions and the Grid Carbon Intensity
    fig = plt.figure(figsize=(8, 8), dpi=_DEF_FIG_DPI)
    fig.text(0.5, 0.98, "Carbon Emissions (CO$_2$e)", ha="center", va="top", fontsize=12,
             weight='normal')
    fig.text(0.5, 0.955, "Total estimated emissions and grid intensity", ha="center", va="top", fontsize=9,
             style="italic")

    ax_a = fig.add_axes([0.15, 0.55, 0.7, 0.3])
    _ax_minimal(ax_a)
    val_co2 = co2_kg if co2_kg is not None else 0.0
    ba = ax_a.bar(["Total Emissions"], [val_co2], color="#E74C3C", width=0.4, edgecolor="black", linewidth=0.5)
    if co2_kg is not None:
        ax_a.text(ba[0].get_x() + ba[0].get_width() / 2., val_co2, f"{val_co2:.3f} kg CO$_2$e",
                  ha="center", va="bottom", fontsize=9, weight="bold")
    ax_a.set_ylabel("Kilograms CO$_2$e", fontsize=9)
    ax_a.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    ax_b = fig.add_axes([0.15, 0.15, 0.7, 0.3])
    _ax_minimal(ax_b)
    val_int = kgco2_per_kwh if kgco2_per_kwh is not None else 0.0
    bb = ax_b.bar(["Grid Intensity"], [val_int], color="#2ECC71", width=0.4, edgecolor="black", linewidth=0.5)
    if kgco2_per_kwh is not None:
        ax_b.text(bb[0].get_x() + bb[0].get_width() / 2., val_int, f"{val_int:.3f} kg CO$_2$e / kWh",
                  ha="center", va="bottom", fontsize=9, weight="bold")
    ax_b.set_ylabel("Emission Factor", fontsize=9)
    ax_b.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    if figures_dir:
        fig.savefig(figures_dir / "cost_carbon_emissions.png", dpi=_DEF_FIG_DPI, bbox_inches="tight",
                    facecolor='white', edgecolor='none')

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    pages_added += 1

    # -------- PAGE: FLOPs per step --------
    # Breakdown of Floating Point Operations (Forward vs Backward pass)

    tflops = flops["total_flops"] / 1e12 if flops.get("per_step_total_flops") else 0
    steps = flops.get("steps_counted", 0)

    # Titolo + sottotitolo aggiornati
    fig, ax = _create_consistent_figure(
        "Floating Point Operations per Training Step",
        subtitle=f"Total: {tflops:.2f} TFLOPS ({steps:,} steps)"
    )

    if flops.get("per_step_total_flops"):
        cats = ["Forward\nPass", "Backward\nPass", "Total\nper Step"]
        vals = [
            flops["per_step_forward_flops"] / 1e6,
            flops["per_step_backward_flops"] / 1e6,
            flops["per_step_total_flops"] / 1e6
        ]
        bars = ax.bar(
            cats, vals,
            color=[plt.get_cmap("tab10")(2), plt.get_cmap("tab10")(0), plt.get_cmap("tab10")(1)],
            edgecolor="black", linewidth=0.5, alpha=0.85
        )

        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2., b.get_height(),
                f"{v:.1f}",
                ha="center", va="bottom", fontsize=8, weight="bold"
            )

        ax.set_ylabel("MegaFLOPS", fontsize=9)

    else:
        ax.text(
            0.5, 0.5,
            "FLOPs data not available",
            ha="center", va="center", fontsize=10, color="gray"
        )

    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    if figures_dir:
        fig.savefig(
            figures_dir / "cost_flops_per_step.png",
            dpi=_DEF_FIG_DPI, bbox_inches="tight",
            facecolor='white', edgecolor='none'
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    pages_added += 1

    # -------- PAGE: Data Transfer & I/O (log) --------
    # Logarithmic scale plot of Network and Disk usage
    fig, ax = _create_consistent_figure("Data Transfer & I/O Operations")
    cats = ["Network\nRX", "Network\nTX", "Disk\nRead", "Disk\nWrite", "Artifacts"]
    vals_lin = [
        bytes_io["network_rx_bytes"] / 1e9,
        bytes_io["network_tx_bytes"] / 1e9,
        bytes_io["disk_read_bytes"] / 1e9,
        bytes_io["disk_write_bytes"] / 1e9,
        bytes_io["artifacts_written_bytes"] / 1e9,
    ]
    eps = 1e-6
    vals = [max(v, eps) for v in vals_lin]
    bars = ax.bar(range(len(cats)), vals, color=[plt.get_cmap("tab10")(i % 10) for i in range(len(cats))],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_yscale("log")
    min_pos = min(v for v in vals if v > 0)
    ax.set_ylim(min_pos / 5, max(vals) * 1.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f} GB"))
    ax.set_ylabel("Gigabytes (GB) - log scale", fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.25, linestyle="--", linewidth=0.5)
    for b, vr in zip(bars, vals_lin):
        ax.text(b.get_x() + b.get_width() / 2, max(vr, min_pos), f"{vr:.2f} GB", ha="center", va="bottom", fontsize=8)

    if figures_dir:
        fig.savefig(figures_dir / "cost_data_transfer_io.png", dpi=_DEF_FIG_DPI, bbox_inches="tight",
                    facecolor='white', edgecolor='none')

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    pages_added += 1

    # -------- PAGE: KPI (cards) --------
    # Grid of Summary Cards (KPIs) for quick reference
    fig, ax = _create_consistent_figure("Key Performance Indicators")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def _fmt_bytes_simple(b):
        """Helper to format bytes into human-readable strings (B to PB)."""
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        v = float(b)
        i = 0
        while i < len(units) - 1 and v >= 1024.0:
            v /= 1024.0
            i += 1
        return f"{v:.2f} {units[i]}"

    emissions_str = (f"{co2_kg:.1f} kg CO$_2$e" if co2_kg is not None else "—")
    intensity_str = (f"{kgco2_per_kwh:.3f} kg CO$_2$e/kWh" if kgco2_per_kwh is not None else "—")

    # Avg power
    try:
        avg_power = np.mean([energy["extra"]["nvml_power_start_W"], energy["extra"]["nvml_power_end_W"]])
        avg_power_str = f"{avg_power:.1f} W"
    except (KeyError, TypeError):
        avg_power_str = "—"

    kpis = [
        ("Total Runtime", _fmt_hms_val(costs["wall_clock"]["total_sec"])),
        ("Energy Consumed", f"{energy['kwh']:.3f} kWh"),
        ("Avg Power Draw", avg_power_str),
        ("Total FLOPS", f"{flops['total_flops'] / 1e12:.2f} TFLOPS"),
        ("Training Steps", f"{flops['steps_counted']:,}"),
        ("Data Transferred", _fmt_bytes_simple(bytes_io["network_tx_bytes"] + bytes_io["network_rx_bytes"])),
        ("Emissions (CO$_2$e)", emissions_str),
        ("Emission Intensity", intensity_str),
    ]

    from matplotlib.patches import FancyBboxPatch
    from matplotlib import cm

    # Palette & layout configuration for card grid
    cmap = cm.get_cmap("tab10")
    ncols = 2
    nrows = int(np.ceil(len(kpis) / ncols))
    x_pad, y_pad = 0.06, 0.06  # outer margins
    gap_x, gap_y = 0.03, 0.03  # gaps between cards
    card_w = (1 - 2 * x_pad - (ncols - 1) * gap_x) / ncols
    card_h = (1 - 2 * y_pad - (nrows - 1) * gap_y) / nrows

    def draw_card(ix, title, value, face, edge):
        """Draws a single KPI card with title and value."""
        r = ix // ncols
        c = ix % ncols
        y_top = 1 - y_pad - r * (card_h + gap_y)
        x_left = x_pad + c * (card_w + gap_x)

        # Card background
        card = FancyBboxPatch(
            (x_left, y_top - card_h),
            card_w,
            card_h,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            linewidth=0.8,
            edgecolor=edge,
            facecolor=face,
            alpha=0.25
        )
        ax.add_patch(card)

        # Center positions
        cx = x_left + card_w / 2
        cy = y_top - card_h / 2

        # Title centered above value
        ax.text(
            cx, cy + card_h * 0.12,
            title,
            fontsize=10, weight="bold",
            ha="center", va="center"
        )

        # Value centered
        ax.text(
            cx, cy - card_h * 0.12,
            value,
            fontsize=11, weight="bold",
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=edge,
                alpha=0.20,
                linewidth=0
            )
        )

    # Draw cards with rotating colors
    for i, (k, v) in enumerate(kpis):
        col = cmap(i % 10)
        face = col
        edge = col
        draw_card(i, k, v, face, edge)

    if figures_dir:
        fig.savefig(figures_dir / "cost_kpi.png", dpi=_DEF_FIG_DPI, bbox_inches="tight",
                    facecolor='white', edgecolor='none')

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    pages_added += 1

    # Add all new cost metric pages
    new_pages = append_new_cost_pages(pdf, experiment_dir, figures_dir)
    pages_added += new_pages

    logger.info(f"Added {new_pages} new cost metric pages to PDF")

    return pages_added