import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==========================================
# 1. NATURE / SCIENCE STYLE CONFIGURATION
# ==========================================
# Matched to your second graph's style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 3.5,
    'figure.dpi': 300,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

DATASETS = ["CIFAR", "BloodMNIST", "RetinaMNIST", "PathMNIST", "ISIC2019"]

MARKERS = {
    "Real": 'o',
    "Synthetic": '^'
}

# The specific palette extracted from your previous requests
# UPDATED: Added a 10th color (Muted Purple) so CIFAR-10 fits this palette
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
    "#8172B3"  # <--- NEW: Muted Purple
]

ALPHA = 0.7
S_2D = 8  # Slightly smaller to match the finer style

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_data(dataset_name):
    """Loads JSON data for 2D t-SNE."""
    path = os.path.join(BASE_DIR, dataset_name, "tsne2.json")
    
    if not os.path.exists(path):
        # Create dummy data for testing if file doesn't exist
        print(f"Warning: File not found {path}. Generating dummy data.")
        return np.random.rand(100, 2), np.random.randint(0, 5, 100), np.random.choice(["Real", "Synthetic"], 100)
        
    with open(path, 'r') as f:
        data = json.load(f)
    
    X = np.array(data['x'])
    labels = np.array(data['labels'])
    domains = np.array(data['domain'])
    
    return X, labels, domains

def get_custom_cmap(n_classes):
    """
    Returns a colormap matching the specific hex codes provided.
    Falls back to tab20 if classes exceed the palette size.
    """
    # CIFAR (10 classes) will now pass this check because len(NATURE_PALETTE) is 10
    if n_classes <= len(NATURE_PALETTE):
        return mcolors.ListedColormap(NATURE_PALETTE[:n_classes])
    else:
        return plt.cm.tab20

# ==========================================
# 3. PLOTTING LOGIC
# ==========================================

def create_figure():
    # Adjusted figsize to be consistent with the second graph's density
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(8.5, 2.0), constrained_layout=True)
    
    for col_idx, ds_name in enumerate(DATASETS):
        ax = axes[col_idx]
        
        # Force square aspect ratio for t-SNE
        ax.set_box_aspect(1) 
        
        data = load_data(ds_name)
        if data is not None:
            X, labels, domains = data
            unique_labels = np.unique(labels)
            
            # Apply the custom palette
            cmap = get_custom_cmap(len(unique_labels))
            
            for dom_type in ["Real", "Synthetic"]:
                mask = (domains == dom_type)
                if np.sum(mask) > 0:
                    c_vals = labels[mask]
                    
                    ax.scatter(
                        X[mask, 0], X[mask, 1],
                        c=c_vals,
                        cmap=cmap,
                        marker=MARKERS[dom_type],
                        s=S_2D,
                        alpha=ALPHA,
                        linewidth=0.1,
                        edgecolors='white',
                        vmin=0, vmax=len(unique_labels)-1
                    )

        # Style Axes
        ax.set_title(ds_name, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Ensure spines are hidden to match typical t-SNE clean look
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        if col_idx == 0:
            ax.set_ylabel("2D t-SNE", fontweight='bold')

    return fig

# ==========================================
# 4. SAVE
# ==========================================

if __name__ == "__main__":
    fig = create_figure()
    
    output_file = "tsne_embeddings_nature_style.pdf"
    print(f"Generating {output_file}...")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print("Done.")
