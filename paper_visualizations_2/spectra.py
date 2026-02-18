import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

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
    'lines.linewidth': 2.0,
    'figure.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.constrained_layout.use': True
})

# Full Nature Palette for reference
NATURE_PALETTE = [
    "#004D40", # Deep Teal (Previous Real)
    "#009E73", # Green
    "#CC79A7", # Pink
    "#555555", # Dark Gray
    "#888888", # Medium Gray
    "#BBBBBB", # Light Gray
    "#E69F00", # Orange (Previous Synthetic)
    "#56B4E9", # Sky Blue
    "#444444", # Very Dark Gray
    "#8172B3"  # Muted Purple
]

# --- NEW COLOR SELECTION ---
# distinct from the previous Teal/Orange pair
COLOR_REAL = "#56B4E9"   # Sky Blue
COLOR_SYNTH = "#8172B3"  # Muted Purple

DATASETS = ["CIFAR", "BloodMNIST", "RetinaMNIST", "PathMNIST", "ISIC2019"]
N_COMPONENTS = 20  # Number of Principal Components to plot (X-axis)
IMG_SIZE = (64, 64) # Resize images to this to ensure consistency

# ==========================================
# 2. DATA PROCESSING
# ==========================================

def load_and_flatten_images(folder_path, max_images=100):
    """
    Loads images from a folder, converts to Grayscale, resizes, and flattens.
    Returns: numpy array of shape (n_images, n_pixels)
    """
    image_list = []
    # Support png (and jpg just in case)
    files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    
    if len(files) == 0:
        return np.array([])

    # Limit sample size for speed and PCA consistency
    files = files[:max_images]

    for filename in files:
        try:
            # Convert to Grayscale (L) for structural variance analysis
            with Image.open(filename) as img:
                img = img.convert('L').resize(IMG_SIZE)
                image_list.append(np.array(img).flatten())
        except Exception:
            continue

    if not image_list:
        return np.array([])
        
    # Normalize 0-1
    return np.array(image_list) / 255.0

def get_pca_curve(data, n_components):
    """
    Computes Cumulative Explained Variance for a single batch of data.
    """
    # We need at least n_components images to compute n_components
    n_samples = data.shape[0]
    curr_components = min(n_components, n_samples)
    
    if curr_components < 2:
        return np.zeros(n_components)

    pca = PCA(n_components=curr_components)
    pca.fit(data)
    
    # Calculate cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    # Pad with last value if we didn't have enough components (rare edge case)
    if len(cumsum) < n_components:
        cumsum = np.pad(cumsum, (0, n_components - len(cumsum)), 'edge')
        
    return cumsum

def compute_dataset_stats(dataset_path, n_components):
    """
    Iterates through all classes in the dataset's 'real' and 'synthetic' folders.
    Calculates the PCA curve for every class, then computes Mean/Std across classes.
    """
    real_path_base = os.path.join(dataset_path, "real")
    synth_path_base = os.path.join(dataset_path, "synthetic")
    
    # Get list of class directories (handling both "class-000" and "class-000_name")
    if not os.path.exists(real_path_base):
        return None, None
        
    classes = [d for d in os.listdir(real_path_base) if os.path.isdir(os.path.join(real_path_base, d))]
    
    real_curves = []
    synth_curves = []
    
    for cls in classes:
        # Paths
        p_real = os.path.join(real_path_base, cls)
        p_synth = os.path.join(synth_path_base, cls)
        
        # Load Data
        data_real = load_and_flatten_images(p_real)
        data_synth = load_and_flatten_images(p_synth)
        
        # Compute Curves if data exists
        if len(data_real) > 0 and len(data_synth) > 0:
            real_curves.append(get_pca_curve(data_real, n_components))
            synth_curves.append(get_pca_curve(data_synth, n_components))
            
    if not real_curves:
        return None, None
        
    # Convert to matrix (n_classes, n_components)
    real_mat = np.array(real_curves)
    synth_mat = np.array(synth_curves)
    
    # Compute Statistics (Mean and Std across classes)
    stats_real = {
        "mean": np.mean(real_mat, axis=0),
        "std": np.std(real_mat, axis=0)
    }
    stats_synth = {
        "mean": np.mean(synth_mat, axis=0),
        "std": np.std(synth_mat, axis=0)
    }
    
    return stats_real, stats_synth

# ==========================================
# 3. PLOTTING
# ==========================================

def create_pca_figure():
    # 1 Row, 5 Columns
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), constrained_layout=True)
    
    x_axis = np.arange(1, N_COMPONENTS + 1)
    
    for i, ds_name in enumerate(DATASETS):
        ax = axes[i]
        print(f"Processing {ds_name}...")
        
        # Path assumption: Script is running in the root directory containing the dataset folders
        stats_real, stats_synth = compute_dataset_stats(ds_name, N_COMPONENTS)
        
        if stats_real is not None:
            # --- REAL DATA ---
            ax.plot(x_axis, stats_real['mean'], color=COLOR_REAL, label='Real', zorder=2)
            ax.fill_between(x_axis, 
                            stats_real['mean'] - stats_real['std'], 
                            stats_real['mean'] + stats_real['std'], 
                            color=COLOR_REAL, alpha=0.15, zorder=1)
            
            # --- SYNTHETIC DATA ---
            # Using dashed line for synthetic to help differentiation in B&W print
            ax.plot(x_axis, stats_synth['mean'], color=COLOR_SYNTH, linestyle='--', label='Synthetic', zorder=3)
            ax.fill_between(x_axis, 
                            stats_synth['mean'] - stats_synth['std'], 
                            stats_synth['mean'] + stats_synth['std'], 
                            color=COLOR_SYNTH, alpha=0.15, zorder=1)
        else:
            ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center', color='gray')

        # Formatting
        ax.set_title(ds_name, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        
        if i == 0:
            ax.set_ylabel("Cumul. Explained Variance", fontweight='bold')
            ax.set_xlabel("Top-k PCs")
            # Legend only in the first plot to avoid clutter
            ax.legend(loc='lower right', frameon=False)
        else:
            ax.set_yticklabels([])
            ax.set_xlabel("Top-k PCs")

    return fig

if __name__ == "__main__":
    fig = create_pca_figure()
    output_file = "pca_spectra_comparison_v2.pdf"
    print(f"Saving to {output_file}...")
    fig.savefig(output_file, bbox_inches='tight')
    print("Done.")
