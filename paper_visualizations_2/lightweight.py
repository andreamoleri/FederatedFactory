import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
# Make sure this matches the filename you generated
CSV_PATH = "metrics_per_class_official.csv"
DATASETS_ORDER = ["CIFAR", "BloodMNIST", "RetinaMNIST", "PathMNIST", "ISIC2019"]

# Nature-Style Configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.constrained_layout.use': True
})

PALETTE = {
    "CIFAR": "#004D40",      # Deep Teal
    "BloodMNIST": "#CC79A7", # Pink
    "RetinaMNIST": "#E69F00",# Orange
    "PathMNIST": "#56B4E9",  # Sky Blue
    "ISIC2019": "#8172B3"    # Muted Purple
}

# ==========================================
# 2. PLOTTING FUNCTION
# ==========================================
def plot_raincloud(df, metric, ax, palette):
    datasets = df["Dataset"].unique()
    
    # 1. Violin Plot (Density)
    sns.violinplot(data=df, x="Dataset", y=metric, ax=ax, palette=palette, 
                   inner=None, cut=0, linewidth=0, alpha=0.7)
    
    # Clip the violin to the left half
    for item in ax.collections:
        x0, y0, width, height = item.get_paths()[0].get_extents().bounds
        item.set_clip_path(patches.Rectangle((x0, y0), width/2, height, transform=ax.transData))
    
    # 2. Strip Plot (Raw Data Points)
    sns.stripplot(data=df, x="Dataset", y=metric, ax=ax, palette=palette, 
                  alpha=0.6, size=4, jitter=0.15, zorder=1)
    
    # Offset the strip plot to the right
    for dots in ax.collections[-len(datasets):]:
        dots.set_offsets(dots.get_offsets() + np.array([0.15, 0]))
    
    # 3. Box Plot (Summary Statistics)
    sns.boxplot(data=df, x="Dataset", y=metric, ax=ax, width=0.15,
                boxprops={'facecolor': 'none', "zorder": 5}, showfliers=False,
                whiskerprops={'linewidth': 1.5, "zorder": 5},
                capprops={'linewidth': 1.5, "zorder": 5},
                medianprops={'color': 'black', 'linewidth': 2, "zorder": 5})

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find '{CSV_PATH}'. Run the calculation script first.")
        return

    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # Filter: Keep only datasets that exist in DATASETS_ORDER
    valid_datasets = [d for d in DATASETS_ORDER if d in df["Dataset"].unique()]
    
    if not valid_datasets:
        print("Error: The CSV contains data, but none of the datasets match DATASETS_ORDER.")
        return

    # Categorical sorting to ensure correct order on X-axis
    df["Dataset"] = pd.Categorical(df['Dataset'], categories=valid_datasets, ordered=True)
    df = df.sort_values("Dataset")

    # Clean up any potential NaNs (failed FID calculations)
    initial_count = len(df)
    df = df.dropna(subset=["FID", "KID"])
    if len(df) < initial_count:
        print(f"Note: Dropped {initial_count - len(df)} rows containing NaN values.")

    print("Generating plot...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot FID
    plot_raincloud(df, "FID", axes[0], PALETTE)
    axes[0].set_ylabel("FID Score", fontweight='bold')
    axes[0].set_title("Per-Class Distribution of Image Fidelity", fontweight='bold', pad=15)
    axes[0].grid(axis='y', linestyle=':', alpha=0.4)

    # Plot KID
    plot_raincloud(df, "KID", axes[1], PALETTE)
    axes[1].set_ylabel("KID (×100)", fontweight='bold')
    axes[1].set_title("Per-Class Distribution of Kernel Inception Distance", fontweight='bold', pad=15)
    axes[1].grid(axis='y', linestyle=':', alpha=0.4)
    axes[1].set_xlabel("")

    # Custom Legend
    legend_elements = [
        patches.Patch(facecolor='gray', alpha=0.5, label='Density (Violin)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Individual Class (Strip)'),
        plt.Line2D([0], [0], color='black', lw=2, label='Median (Box)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

    output_path = "nature_metrics_raincloud_official.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Success! Graph saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
