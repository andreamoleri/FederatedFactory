import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import warnings

# ==========================================
# IMPORTANT: PYTORCH-FID INTEGRATION
# ==========================================
try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
except ImportError:
    raise ImportError("Strict reproduction requires the official library. Please run: pip install pytorch-fid")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
DATA_ROOT = "."  # Current directory
DATASETS_ORDER = ["CIFAR", "BloodMNIST", "RetinaMNIST", "PathMNIST", "ISIC2019"]
CSV_PATH = "metrics_per_class_official.csv"
GLOBAL_TXT_PATH = "metrics_global_official.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 50 
DIMS = 2048

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
# 2. METRIC CALCULATION UTILS
# ==========================================

class ImageFolderDataset(Dataset):
    def __init__(self, file_paths):
        self.files = file_paths
        
        self.transform = transforms.Compose([
            # Resize shortest edge to 299, keep aspect ratio
            transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
            # Crop the center 299x299
            transforms.CenterCrop(299),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img)

def get_activations(files, model, batch_size=50):
    """Extracts features using the official pytorch-fid Inception model."""
    model.eval()
    if len(files) == 0:
        return np.empty((0, DIMS), dtype=np.float64)
        
    dataset = ImageFolderDataset(files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    pred_arr = np.empty((len(files), DIMS), dtype=np.float64)
    start_idx = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            
            # PYTORCH-FID SPECIFIC:
            # The model returns a list of features. We requested the last block (2048).
            # Output shape is [N, 2048, 1, 1], so we must squeeze or view.
            pred = model(batch)[0] 
            pred = pred.view(pred.size(0), -1)
            
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred.cpu().numpy().astype(np.float64)
            start_idx = start_idx + pred.shape[0]
            
    return pred_arr

def calculate_kid(feat1, feat2):
    """Unbiased Kernel Inception Distance (KID)."""
    n = feat1.shape[0]
    m = feat2.shape[0]
    
    if n < 2 or m < 2:
        return float('nan')

    def kernel(x, y):
        # Polynomial kernel (degree=3, gamma=1/dims, coef0=1)
        return (np.dot(x, y.T) / x.shape[1] + 1) ** 3

    k_xx = kernel(feat1, feat1)
    k_yy = kernel(feat2, feat2)
    k_xy = kernel(feat1, feat2)
    
    kid = (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1)) + \
          (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1)) - \
          2 * np.sum(k_xy) / (n * m)
    return kid

# ==========================================
# 3. ROBUST COMPUTE LOOP
# ==========================================

def get_class_map():
    structure = {}
    print("Scanning directories...")
    
    for dataset in DATASETS_ORDER:
        ds_path = os.path.join(DATA_ROOT, dataset)
        real_path = os.path.join(ds_path, "real")
        syn_path = os.path.join(ds_path, "synthetic")
        
        if not os.path.exists(real_path): continue

        structure[dataset] = {}
        classes = sorted([d for d in os.listdir(real_path) if os.path.isdir(os.path.join(real_path, d))])
        
        for cls_name in classes:
            r_files = glob.glob(os.path.join(real_path, cls_name, "*.png"))
            s_files = glob.glob(os.path.join(syn_path, cls_name, "*.png"))
            
            if len(r_files) >= 2 and len(s_files) >= 2:
                structure[dataset][cls_name] = {
                    "real": r_files,
                    "syn": s_files
                }
    return structure

def compute_metrics():
    # 1. Initialize Caches
    if os.path.exists(CSV_PATH):
        print(f"Found existing {CSV_PATH}. Resuming...")
        df_existing = pd.read_csv(CSV_PATH)
        existing_pairs = set(zip(df_existing["Dataset"], df_existing["Class"]))
    else:
        print("Starting fresh...")
        pd.DataFrame(columns=["Dataset", "Class", "FID", "KID"]).to_csv(CSV_PATH, index=False)
        existing_pairs = set()

    with open(GLOBAL_TXT_PATH, "w") as f:
        f.write("Dataset,Global_FID,Global_KID\n")

    data_map = get_class_map()
    if not data_map:
        print("No data found!")
        return pd.read_csv(CSV_PATH)

    # 2. Load OFFICIAL Model
    print("Loading Official pytorch-fid InceptionV3...")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(DEVICE)
    model.eval()

    # 3. Processing Loop
    for dataset in DATASETS_ORDER:
        if dataset not in data_map: continue
        
        print(f"\nProcessing Dataset: {dataset}")
        classes = data_map[dataset]
        
        all_real_feats = []
        all_syn_feats = []
        
        for cls_name, files in tqdm(classes.items(), desc=f"Classes in {dataset}"):
            
            # --- FEATURE EXTRACTION (Must happen every time for Global Metrics) ---
            act_real = get_activations(files['real'], model)
            act_syn = get_activations(files['syn'], model)
            
            all_real_feats.append(act_real)
            all_syn_feats.append(act_syn)

            # --- CHECK RESUME ---
            if (dataset, cls_name) in existing_pairs:
                continue

            # --- PER-CLASS METRICS ---
            try:
                mu_r, sigma_r = np.mean(act_real, axis=0), np.cov(act_real, rowvar=False)
                mu_s, sigma_s = np.mean(act_syn, axis=0), np.cov(act_syn, rowvar=False)
                
                # Use OFFICIAL FID Calculation
                fid = calculate_frechet_distance(mu_r, sigma_r, mu_s, sigma_s)
            except Exception as e:
                # FID calculation can fail for small N (< Dims) due to singular matrix
                fid = float('nan')

            kid = calculate_kid(act_real, act_syn)
            
            new_row = pd.DataFrame([{
                "Dataset": dataset,
                "Class": cls_name,
                "FID": fid,
                "KID": kid * 100
            }])
            new_row.to_csv(CSV_PATH, mode='a', header=False, index=False)
        
        # --- GLOBAL CALCULATION ---
        print(f"Calculating Global Metrics for {dataset}...")
        
        if len(all_real_feats) > 0:
            global_real = np.concatenate(all_real_feats, axis=0)
            global_syn = np.concatenate(all_syn_feats, axis=0)
            
            # Free memory of list
            del all_real_feats, all_syn_feats
            torch.cuda.empty_cache()

            mu_gr, sigma_gr = np.mean(global_real, axis=0), np.cov(global_real, rowvar=False)
            mu_gs, sigma_gs = np.mean(global_syn, axis=0), np.cov(global_syn, rowvar=False)
            
            g_fid = calculate_frechet_distance(mu_gr, sigma_gr, mu_gs, sigma_gs)
            
            MAX_KID_SAMPLES = 5000
            if global_real.shape[0] > MAX_KID_SAMPLES:
                idx_r = np.random.choice(global_real.shape[0], MAX_KID_SAMPLES, replace=False)
                idx_s = np.random.choice(global_syn.shape[0], MAX_KID_SAMPLES, replace=False)
                g_kid = calculate_kid(global_real[idx_r], global_syn[idx_s])
            else:
                g_kid = calculate_kid(global_real, global_syn)
                
            print(f"  >> {dataset} GLOBAL FID: {g_fid:.4f} | GLOBAL KID: {g_kid*100:.4f}")
            
            with open(GLOBAL_TXT_PATH, "a") as f:
                f.write(f"{dataset},{g_fid},{g_kid*100}\n")
        else:
            print(f"  >> Skipping Global metrics for {dataset} (No data collected)")
            
    return pd.read_csv(CSV_PATH)

# ==========================================
# 4. PLOTTING
# ==========================================

def plot_raincloud(df, metric, ax, palette):
    datasets = df["Dataset"].unique()
    sns.violinplot(data=df, x="Dataset", y=metric, ax=ax, palette=palette, 
                   inner=None, cut=0, linewidth=0, alpha=0.7)
    for item in ax.collections:
        x0, y0, width, height = item.get_paths()[0].get_extents().bounds
        item.set_clip_path(patches.Rectangle((x0, y0), width/2, height, transform=ax.transData))
    sns.stripplot(data=df, x="Dataset", y=metric, ax=ax, palette=palette, 
                  alpha=0.6, size=4, jitter=0.15, zorder=1)
    for dots in ax.collections[-len(datasets):]:
        dots.set_offsets(dots.get_offsets() + np.array([0.15, 0]))
    sns.boxplot(data=df, x="Dataset", y=metric, ax=ax, width=0.15,
                boxprops={'facecolor': 'none', "zorder": 5}, showfliers=False,
                whiskerprops={'linewidth': 1.5, "zorder": 5},
                capprops={'linewidth': 1.5, "zorder": 5},
                medianprops={'color': 'black', 'linewidth': 2, "zorder": 5})

def main():
    np.random.seed(42)

    df = compute_metrics()
    
    # Filter only datasets that exist in the result
    valid_datasets = [d for d in DATASETS_ORDER if d in df["Dataset"].unique()]
    df["Dataset"] = pd.Categorical(df['Dataset'], categories=valid_datasets, ordered=True)
    df = df.sort_values("Dataset")

    if df.empty:
        print("No metrics computed. Exiting.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    plot_raincloud(df, "FID", axes[0], PALETTE)
    axes[0].set_ylabel("FID Score (Lower is Better)", fontweight='bold')
    axes[0].set_title("Per-Class Distribution of Image Fidelity (Official Benchmark)", fontweight='bold', pad=15)
    axes[0].grid(axis='y', linestyle=':', alpha=0.4)

    plot_raincloud(df, "KID", axes[1], PALETTE)
    axes[1].set_ylabel("KID (×100) (Lower is Better)", fontweight='bold')
    axes[1].set_title("Per-Class Distribution of Kernel Inception Distance", fontweight='bold', pad=15)
    axes[1].grid(axis='y', linestyle=':', alpha=0.4)
    axes[1].set_xlabel("")

    legend_elements = [
        patches.Patch(facecolor='gray', alpha=0.5, label='Density (Violin)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Individual Class (Strip)'),
        plt.Line2D([0], [0], color='black', lw=2, label='Median (Box)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

    output_path = "nature_metrics_raincloud_official.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\nGraph saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
