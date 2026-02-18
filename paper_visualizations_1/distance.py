import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ==========================================
# 1. NATURE / SCIENCE STYLE CONFIGURATION
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,        
    'axes.labelsize': 11,        
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,      
    'figure.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.constrained_layout.use': True
})

DATASETS = ["CIFAR", "BloodMNIST", "RetinaMNIST", "PathMNIST", "ISIC2019"]

NATURE_PALETTE = [
    "#004D40", # Deep Teal
    "#009E73", # Green
    "#CC79A7", # Pink
    "#555555", # Dark Gray
    "#888888", # Medium Gray
    "#BBBBBB", # Light Gray
    "#E69F00", # Orange
    "#56B4E9", # Sky Blue
    "#444444", # Very Dark Gray
    "#8172B3"  # Muted Purple
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 2. DATA LOADING HELPER FUNCTIONS
# ==========================================

def compute_min_distances(file_path, is_gen_to_gen=False):
    try:
        matrix = np.load(file_path)
        if is_gen_to_gen:
            np.fill_diagonal(matrix, np.inf)
        dists = np.min(matrix, axis=1)
        dists = dists[np.isfinite(dists)]
        return dists
    except Exception:
        return np.array([])

def load_distance_data_map(dataset_name, mode_suffix):
    folder = Path(BASE_DIR) / dataset_name
    data_map = {}
    if not folder.exists():
        return data_map

    pattern = re.compile(rf"class-(\d+)_{re.escape(mode_suffix)}\.npy")
    for f in folder.glob("*.npy"):
        match = pattern.search(f.name)
        if match:
            cid = int(match.group(1))
            is_g2g = ("gen-to-gen" in mode_suffix)
            dists = compute_min_distances(f, is_gen_to_gen=is_g2g)
            if len(dists) > 0:
                data_map[cid] = dists
    return data_map

def get_class_color(class_id, n_classes):
    if n_classes <= len(NATURE_PALETTE):
        return NATURE_PALETTE[class_id % len(NATURE_PALETTE)]
    else:
        cmap = plt.get_cmap("tab20")
        return mcolors.to_hex(cmap(class_id % 20))

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_ecdf_panel(ax, data_dict, n_classes, xlabel):
    if not data_dict:
        ax.text(0.5, 0.5, "N/A", ha='center', va='center', color='gray')
        ax.axis('off')
        return

    classes = sorted(data_dict.keys())
    for c in classes:
        d = np.sort(data_dict[c])
        if len(d) == 0: continue
        y = np.linspace(0, 1, len(d))
        
        color = get_class_color(c, n_classes)
        ax.plot(d, y, color=color, linewidth=1.5, alpha=0.9)

    ax.set_xlabel(xlabel) 
    ax.grid(False) 
    
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(left=True, bottom=True)

def plot_hist_panel(ax, data_dict, n_classes, xlabel):
    if not data_dict:
        ax.text(0.5, 0.5, "N/A", ha='center', va='center', color='gray')
        ax.axis('off')
        return

    classes = sorted(data_dict.keys())
    for c in classes:
        d = data_dict[c]
        if len(d) == 0: continue
        
        color = get_class_color(c, n_classes)
        # 1. Plot the filled area (Increased alpha slightly for visibility since outline is gone)
        ax.hist(d, bins=40, density=True, color=color, alpha=0.4, histtype='stepfilled')
        
        # 2. [REMOVED] The outline (border) drawing code below is commented out:
        # ax.hist(d, bins=40, density=True, color=color, alpha=0.9, histtype='step', linewidth=1.0)

    ax.set_xlabel(xlabel)
    ax.set_yticks([]) 
    ax.spines['left'].set_visible(False)
    
    # Ensure grid is off
    ax.grid(False)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def create_full_figure():
    nrows = 4 
    ncols = len(DATASETS)
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(16, 10), 
        constrained_layout=True,
        gridspec_kw={'wspace': 0.1, 'hspace': 0.1}
    )
    
    row_titles = [
        "Fidelity (ECDF)",  
        "Fidelity (Dist)",  
        "Diversity (ECDF)", 
        "Diversity (Dist)"  
    ]
    
    x_labels = [
        "Dist to Real", 
        "Dist to Gen"   
    ]

    for col_idx, ds_name in enumerate(DATASETS):
        print(f"Processing Dataset: {ds_name}")
        
        fid_data = load_distance_data_map(ds_name, "gen-to-real")
        div_data = load_distance_data_map(ds_name, "gen-to-gen")
        
        all_keys = set(fid_data.keys()) | set(div_data.keys())
        n_classes = max(all_keys) + 1 if all_keys else 10
            
        # --- Plot Fidelity Pair (Rows 0 & 1) ---
        ax_fid_ecdf = axes[0, col_idx]
        ax_fid_hist = axes[1, col_idx]
        
        plot_ecdf_panel(ax_fid_ecdf, fid_data, n_classes, xlabel=x_labels[0])
        plot_hist_panel(ax_fid_hist, fid_data, n_classes, xlabel=x_labels[0])
        
        if fid_data: 
             ax_fid_hist.sharex(ax_fid_ecdf)
             ax_fid_ecdf.tick_params(labelbottom=True) 

        # --- Plot Diversity Pair (Rows 2 & 3) ---
        ax_div_ecdf = axes[2, col_idx]
        ax_div_hist = axes[3, col_idx]
        
        plot_ecdf_panel(ax_div_ecdf, div_data, n_classes, xlabel=x_labels[1])
        plot_hist_panel(ax_div_hist, div_data, n_classes, xlabel=x_labels[1])
        
        if div_data:
            ax_div_hist.sharex(ax_div_ecdf)
            ax_div_ecdf.tick_params(labelbottom=True)

        ax_fid_ecdf.set_title(ds_name, fontweight='bold', pad=12)

    # --- Global Styling ---
    for r in range(nrows):
        axes[r, 0].set_ylabel(row_titles[r], fontweight='bold', labelpad=15)

    fig.align_ylabels(axes[:, 0])
    
    return fig

if __name__ == "__main__":
    fig = create_full_figure()
    output_file = "generative_metrics_no_border.pdf"
    print(f"Generating {output_file}...")
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    print("Done.")
