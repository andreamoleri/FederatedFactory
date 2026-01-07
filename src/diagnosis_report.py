# ==============================================================================
# FILE: src/diagnosis_report.py
# ==============================================================================
import os
import sys
import pickle
import re
import math
from pathlib import Path
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# ============================================================================
# HOTFIX FOR PYTORCH 2.6+
# ============================================================================
_orig_torch_load = torch.load


def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _safe_torch_load
# ============================================================================

# ----------------------------------------------------------------------------
# 0. PATH INJECTION & IMPORTS
# ----------------------------------------------------------------------------
current_file = Path(__file__).resolve()
src_root = current_file.parent  # src/
project_root = src_root.parent  # FederatedFactory/
edm2_root = src_root / "modules" / "EDM2"

if str(edm2_root) not in sys.path:
    sys.path.insert(0, str(edm2_root))
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

# EDM2 Imports
import dnnlib
from modules.EDM2.generate_images import edm_sampler

# Project Imports
from imports.data_management import get_dataset, prime_dataset_meta_for_transform
from imports.data_augmentation import build_transform

# ----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLING
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NEURIPS STYLE CONSTANTS
NUM_SAMPLES = 10  # Columns per row (High density)
CLASSES_PER_PAGE = 12  # Classes per page (Results in 12 rows of images)
SEEDS = range(0, NUM_SAMPLES)

# CONFIGURE MATPLOTLIB FOR PAPER QUALITY (SANS-SERIF)
plt.rcParams.update({
    "font.family": "sans-serif",  # Changed to Sans-Serif
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "figure.dpi": 300  # High DPI for print quality
})


# ----------------------------------------------------------------------------
# 2. UTILITIES
# ----------------------------------------------------------------------------
def parse_checkpoint_path(path: Path):
    parts = path.parts
    try:
        idx = parts.index("checkpoints")
        dataset = parts[idx + 1]
        if dataset.startswith("medmnist_") and ":" not in dataset:
            dataset = dataset.replace("medmnist_", "medmnist:", 1)
        partition = parts[idx + 2]
        client = parts[idx + 3]
        class_str = parts[idx + 4]
        class_id = int(re.search(r'\d+', class_str).group())
        filename = path.name
        step_match = re.search(r'network-snapshot-(\d+)', filename)
        step = int(step_match.group(1)) if step_match else 0
        return {
            "dataset": dataset,
            "partition": partition,
            "client": client,
            "class_id": class_id,
            "step": step,
            "path": path
        }
    except Exception as e:
        return None


def get_real_images_for_class(dataset_name, data_dir, class_id):
    """ Loads real images on CPU to save VRAM. """
    try:
        prime_dataset_meta_for_transform(dataset_name, data_dir)
    except:
        pass

    transform = build_transform(dataset_name, train=False, robustness=False)
    ds = get_dataset(dataset_name, data_dir, train=True, transform=transform)

    indices = []
    if hasattr(ds, 'targets'):
        targets = np.array(ds.targets)
        indices = np.where(targets == class_id)[0]
    elif hasattr(ds, 'labels'):
        targets = np.array(ds.labels)
        indices = np.where(targets == class_id)[0]
    else:
        for i in range(len(ds)):
            _, y = ds[i]
            if int(y) == class_id:
                indices.append(i)

    np.random.seed(42)
    np.random.shuffle(indices)

    if len(indices) == 0: return None

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, indices),
        batch_size=256,
        num_workers=0,
        shuffle=False
    )

    all_tensors = []
    for x, _ in loader:
        all_tensors.append(x)

    if not all_tensors: return None

    real_imgs = torch.cat(all_tensors)
    if real_imgs.min() < 0:
        real_imgs = (real_imgs + 1) / 2
        real_imgs = real_imgs.clamp(0, 1)

    return real_imgs


def generate_synthetic(net, num, device):
    net.eval().to(device)
    latents = torch.randn([num, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    labels = torch.zeros([num, net.label_dim], device=device) if net.label_dim > 0 else None

    images = edm_sampler(
        net=net, noise=latents, labels=labels,
        num_steps=32, sigma_min=0.002, sigma_max=80, rho=7,
        randn_like=torch.randn_like
    )

    images = (images + 1) / 2
    images = images.clamp(0, 1)
    return images


# ----------------------------------------------------------------------------
# 3. STREAMING NEAREST NEIGHBOR (RAW PIXEL L2 + AUTO-RESIZE)
# ----------------------------------------------------------------------------
def find_nearest_neighbors_streaming(synth_imgs, real_pool):
    n_synth = synth_imgs.shape[0]
    n_real = real_pool.shape[0]
    target_h, target_w = synth_imgs.shape[2], synth_imgs.shape[3]

    with torch.no_grad():
        synth_flat = synth_imgs.flatten(1)

    min_dists = torch.full((n_synth,), float('inf'), device=DEVICE)
    nearest_indices = torch.zeros((n_synth,), dtype=torch.long, device=DEVICE)
    batch_size = 128

    with torch.no_grad():
        for i in range(0, n_real, batch_size):
            batch_real = real_pool[i: i + batch_size].to(DEVICE)

            # Auto-Resize to match synthetic resolution
            if batch_real.shape[2] != target_h or batch_real.shape[3] != target_w:
                batch_real = F.interpolate(
                    batch_real,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )

            batch_flat = batch_real.flatten(1)
            dists = torch.cdist(synth_flat, batch_flat, p=2)
            batch_mins, batch_idxs = dists.min(dim=1)

            mask = batch_mins < min_dists
            min_dists[mask] = batch_mins[mask]
            nearest_indices[mask] = batch_idxs[mask] + i
            del batch_real, batch_flat, dists

    indices_cpu = nearest_indices.cpu()
    nearest_imgs = real_pool[indices_cpu]
    return nearest_imgs, min_dists.cpu()


# ----------------------------------------------------------------------------
# 4. REPORT GENERATION (NEURIPS STYLE)
# ----------------------------------------------------------------------------
def process_dataset_page(pdf, dataset_name, batch_metas, data_dir):
    """
    Creates a tightly packed, publication-quality figure.
    """
    num_classes = len(batch_metas)

    # Dimensions: Standard Academic Width (approx)
    fig_w = 12
    # Height scales with content, tight packing
    fig_h = 2.0 * num_classes + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Outer Grid: 1 row per class
    outer_gs = gridspec.GridSpec(num_classes, 1, figure=fig,
                                 hspace=0.25,  # Gap between different classes
                                 top=0.98, bottom=0.02, left=0.08, right=0.98)

    for idx, info in enumerate(batch_metas):
        class_id = info['class_id']

        # Inner Grid: 2 rows (Generated, Real), NUM_SAMPLES columns
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, NUM_SAMPLES,
                                                    subplot_spec=outer_gs[idx],
                                                    wspace=0.02, hspace=0.02)  # Zero-gap between pairs

        # Load & Generate
        try:
            with open(info['path'], 'rb') as f:
                data = pickle.load(f)
            net = data['ema'].to(DEVICE)
            synth_imgs = generate_synthetic(net, NUM_SAMPLES, DEVICE)

            real_pool = get_real_images_for_class(dataset_name, data_dir, class_id)
            if real_pool is None: continue

            nearest_reals, dists = find_nearest_neighbors_streaming(synth_imgs, real_pool)

            for s in range(NUM_SAMPLES):
                img_gen = synth_imgs[s].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                img_real = nearest_reals[s].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                dist_val = dists[s].item()

                # --- TOP ROW: Generated ---
                ax_gen = fig.add_subplot(inner_gs[0, s])
                ax_gen.imshow(img_gen)
                ax_gen.axis('off')

                # LABELS: Apply only on the first column (s=0)
                if s == 0:
                    # 1. Main Class Label (Far Left, Bold, Spans both rows visually)
                    # We attach it to the top axis but push it far left and down to center it
                    ax_gen.text(-0.5, 0.0, f"Class {class_id}", transform=ax_gen.transAxes,
                                ha='center', va='center', rotation=90, fontsize=9, fontweight='bold')

                    # 2. Sub-Label: Synthetic (Left of Top Row, Not Bold)
                    ax_gen.text(-0.25, 0.5, "Synthetic", transform=ax_gen.transAxes,
                                ha='center', va='center', rotation=90, fontsize=9, fontweight='normal', color='#444444')

                # --- BOTTOM ROW: Real ---
                ax_real = fig.add_subplot(inner_gs[1, s])
                ax_real.imshow(img_real)
                ax_real.axis('off')

                # LABELS: Apply only on the first column (s=0)
                if s == 0:
                    # 3. Sub-Label: Real (Left of Bottom Row, Not Bold)
                    ax_real.text(-0.25, 0.5, "Real", transform=ax_real.transAxes,
                                 ha='center', va='center', rotation=90, fontsize=9, fontweight='normal',
                                 color='#444444')

                # --- OVERLAY BADGE FOR DISTANCE (NeurIPS Style) ---
                dist_str = f"{dist_val:.1f}"

                # Constant color (Black) regardless of value
                bg_col = 'black'
                text_col = 'white'

                # Add text with bounding box
                ax_real.text(0.95, 0.05, f"$d={dist_str}$",
                             transform=ax_real.transAxes,
                             ha='right', va='bottom',
                             fontsize=6, color=text_col, fontweight='bold',
                             bbox=dict(facecolor=bg_col, alpha=0.6, edgecolor='none', pad=1.5))

            del net, data, real_pool, synth_imgs, nearest_reals
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error class {class_id}: {e}")
            continue

    pdf.savefig(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
def main():
    import argparse
    default_ckpt_dir = project_root / "commands" / "checkpoints"
    default_data_dir = project_root / "data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir))
    parser.add_argument("--checkpoints-dir", type=str, default=str(default_ckpt_dir))
    args = parser.parse_args()

    root = Path(args.checkpoints_dir)
    if not root.exists():
        print(f"❌ Checkpoints directory not found: {root}")
        return

    print(f">>> 🔍 Scanning: {root}")
    all_files = list(root.rglob("network-snapshot-*.pkl"))
    valid_files = [f for f in all_files if f.stat().st_size > 0]
    if not valid_files: return

    latest_snapshots = {}
    for f in valid_files:
        meta = parse_checkpoint_path(f)
        if not meta: continue
        key = (meta['dataset'], meta['partition'], meta['client'], meta['class_id'])
        if key not in latest_snapshots or meta['step'] > latest_snapshots[key]['step']:
            latest_snapshots[key] = meta

    final_list = list(latest_snapshots.values())
    dataset_tasks = defaultdict(list)
    for meta in final_list:
        dataset_tasks[meta['dataset']].append(meta)

    report_path = root / "neurips_comparison_report.pdf"
    print(f">>> 🎨 Generating Publication Report: {report_path}")

    with PdfPages(report_path) as pdf:
        for dataset_name, metas in tqdm(dataset_tasks.items(), desc="Datasets"):
            metas.sort(key=lambda x: x['class_id'])
            for i in range(0, len(metas), CLASSES_PER_PAGE):
                batch = metas[i: i + CLASSES_PER_PAGE]
                process_dataset_page(pdf, dataset_name, batch, args.data_dir)

    print(f"\n>>> ✅ Done! Report saved to: {report_path}")


if __name__ == "__main__":
    main()