# ==============================================================================
# FILE: src/checkpoint_trainer.py
# ==============================================================================
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# 0. CRITICAL PATH INJECTION (Must be BEFORE imports from EDM2)
# ----------------------------------------------------------------------------
# Resolve the project root relative to this file
current_file = Path(__file__).resolve()
src_root = current_file.parent  # src/
project_root = src_root.parent  # FederatedFactory/
edm2_root = src_root / "modules" / "EDM2"

# Inject EDM2 into sys.path so Python can find 'dnnlib'
if str(edm2_root) not in sys.path:
    sys.path.insert(0, str(edm2_root))

# ----------------------------------------------------------------------------
# 1. Standard Imports
# ----------------------------------------------------------------------------
import torch
import json
import shutil
import click
import argparse
import logging
import re
import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------------------
# 2. EDM2 Imports (Now safe to import)
# ----------------------------------------------------------------------------
import dnnlib  # <--- This failed before because path wasn't set
from torch_utils import distributed as dist
from training import training_loop

# ----------------------------------------------------------------------------
# 3. Hotfix for PyTorch 2.6+
# ----------------------------------------------------------------------------
_orig_torch_load = torch.load

def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _safe_torch_load
# ============================================================================

# --- EDM2 IMPORTS ---
# Ensure the EDM2 module is in the python path
current_file = Path(__file__).resolve()
src_root = current_file.parents[0]  # src/
edm2_root = src_root / "modules" / "EDM2"
if str(edm2_root) not in sys.path:
    sys.path.insert(0, str(edm2_root))

from torch_utils import distributed as dist
from training import training_loop

# Internal Project Imports
from logs.logger import get_logger
from jobs.experiment_setup import prepare_data
from imports.data_management import prime_dataset_meta_for_transform

logger = get_logger(__name__)


# ----------------------------------------------------------------------------
# 1. Dataset Materialization (EDM2 requires files on disk)
# ----------------------------------------------------------------------------
def materialize_subset_to_disk(subset, dest_dir, class_idx, target_shape=None):
    """
    Materializes images to disk, forcing a resize if target_shape is provided.

    Args:
        subset: The PyTorch subset/dataset to dump.
        dest_dir: Destination folder.
        class_idx: Class ID (for logging/debugging).
        target_shape: Optional tuple (C, H, W). If provided, all images will be
                      resized to (W, H). e.g., (3, 224, 224).
    """

    # --- ROBUST CACHE CHECK ---
    # If the folder exists, we must ensure the images inside match the expected
    # dimensions (target_shape). If not, we must purge the cache and re-dump.
    if os.path.exists(dest_dir):
        existing_files = [f for f in os.listdir(dest_dir) if f.endswith('.png')]

        # Only consider skipping if we have enough files
        if len(existing_files) >= len(subset):
            should_skip = True

            # If a target shape is enforced, verify the cache matches it
            if target_shape is not None and len(existing_files) > 0:
                from PIL import Image
                # Check the first image as a sample
                sample_path = os.path.join(dest_dir, existing_files[0])
                try:
                    with Image.open(sample_path) as test_img:
                        # target_shape is (C, H, W) -> we need (W, H)
                        expected_w, expected_h = target_shape[-1], target_shape[-2]
                        if test_img.size != (expected_w, expected_h):
                            logger.info(
                                f"   >>> ⚠️ Cache dimensions mismatch! Found {test_img.size}, "
                                f"expected {(expected_w, expected_h)}. Re-materializing..."
                            )
                            should_skip = False
                            # Purge bad cache
                            import shutil
                            shutil.rmtree(dest_dir)
                            # Re-create directory immediately
                            os.makedirs(dest_dir, exist_ok=True)
                except Exception as e:
                    logger.warning(f"   >>> Error verifying cache integrity: {e}. Re-materializing.")
                    should_skip = False
                    import shutil
                    shutil.rmtree(dest_dir)
                    os.makedirs(dest_dir, exist_ok=True)

            if should_skip:
                logger.info(f"   >>> Dataset for Class {class_idx} already exists at {dest_dir}. Skipping dump.")
                return

    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f"   >>> Materializing {len(subset)} images to {dest_dir}...")

    count = 0
    labels = []

    # Access underlying dataset if possible to get PIL images directly
    if hasattr(subset, 'dataset'):
        base_ds = subset.dataset
        indices = subset.indices
    else:
        # Fallback if it's a raw dataset
        base_ds = subset
        indices = range(len(subset))

    from PIL import Image
    import torchvision.transforms.functional as TF

    for idx in tqdm(indices, desc=f"Saving Class {class_idx}", leave=False):
        try:
            # 1. Extraction (same as before)
            if hasattr(base_ds, 'data') and isinstance(base_ds.data, np.ndarray):
                img_array = base_ds.data[idx]
                img = Image.fromarray(img_array)
            elif hasattr(base_ds, 'loader') and hasattr(base_ds, 'samples'):
                path, _ = base_ds.samples[idx]
                img = base_ds.loader(path)
            else:
                item = base_ds[idx]
                img_t = item[0] if isinstance(item, tuple) else item
                if isinstance(img_t, torch.Tensor):
                    if img_t.min() < 0: img_t = (img_t + 1) / 2
                    img = TF.to_pil_image(img_t)
                else:
                    img = img_t

            # 2. UPDATED RESIZING LOGIC
            # If we have a target shape (ISIC), use it.
            if target_shape is not None:
                tgt_h, tgt_w = target_shape[-2], target_shape[-1]
                # Force 32x32 minimum even if target_shape is smaller
                final_h = max(tgt_h, 32)
                final_w = max(tgt_w, 32)
                if img.size != (final_w, final_h):
                    img = img.resize((final_w, final_h), Image.LANCZOS)

            # If no target shape (MedMNIST/CIFAR), ensure 32x32 minimum
            else:
                curr_w, curr_h = img.size
                if curr_w < 32 or curr_h < 32:
                    # RetinaMNIST (28x28) goes here and becomes 32x32
                    img = img.resize((32, 32), Image.LANCZOS)

            # 3. Save (same as before)
            fname = f"img_{count:06d}.png"
            save_path = os.path.join(dest_dir, fname)
            img.save(save_path)
            labels.append([fname, 0])
            count += 1

        except Exception as e:
            logger.warning(f"Failed to save image {idx}: {e}")

    with open(os.path.join(dest_dir, 'dataset.json'), 'w') as f:
        json.dump({'labels': labels}, f)


# ----------------------------------------------------------------------------
# 2. Native EDM2 Training Wrapper
# ----------------------------------------------------------------------------
def run_edm2_native(run_dir, data_path, total_kimg, batch_size, dry_run=False):
    """
    Calls the official EDM2 training loop logic.
    """
    c = dnnlib.EasyDict()
    c.run_dir = str(run_dir)

    # Dataset Config
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=str(data_path),
                                       use_labels=False, xflip=True, cache=True)
    c.data_loader_kwargs = dnnlib.EasyDict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2,
                                           prefetch_factor=2)
    c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')

    # --- BATCH SIZE PHYSICS ---
    world_size = dist.get_world_size()

    # 1. Ensure global batch fits logic
    if batch_size < world_size:
        batch_size = world_size
    if batch_size % world_size != 0:
        batch_size = (batch_size // world_size) * world_size

    # 2. Determine Per-GPU Batch (Physical Limit)
    SAFE_GPU_LIMIT = 768
    batch_per_gpu_global = batch_size // world_size
    c.batch_gpu = min(batch_per_gpu_global, SAFE_GPU_LIMIT)

    while (batch_size // world_size) % c.batch_gpu != 0:
        c.batch_gpu -= 1

    c.batch_size = batch_size

    # --- ALIGNMENT HELPER ---
    def get_valid_interval(val, batch_size, alignment_constraint=0):
        if val is None: return None
        # Align to the larger of batch_size or the constraint (usually 1024 for EDM2 snapshots)
        # This assumes batch_size and constraint are compatible (powers of 2), which they are here.
        step = max(batch_size, alignment_constraint)
        return ((val + step - 1) // step) * step

    # Duration Logic (Constraint: Must be % batch_size)
    raw_nimg = total_kimg * 1000
    c.total_nimg = get_valid_interval(raw_nimg, batch_size)

    if c.total_nimg != raw_nimg and dist.get_rank() == 0:
        logger.info(f"   >>> Adjusted duration from {raw_nimg} to {c.total_nimg} to match batch alignment.")

    # Saving Intervals
    # Constraint 1: Status must be % batch_size
    c.status_nimg = get_valid_interval(50 * 1000, batch_size)

    # Constraint 2: Snapshots/Checkpoints must be % batch_size AND % 1024
    c.snapshot_nimg = get_valid_interval(500 * 1000, batch_size, alignment_constraint=1024)
    c.checkpoint_nimg = get_valid_interval(500 * 1000, batch_size, alignment_constraint=1024)

    # Optimizer & Loss
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.training_loop.EDM2Loss', P_mean=-0.8, P_std=1.6,
                                    sigma_data=0.5)
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=(0.9, 0.99), eps=1e-8)
    c.lr_kwargs = dnnlib.EasyDict(func_name='training.training_loop.learning_rate_schedule', ref_lr=0.01,
                                  ref_batches=35000)

    # Network Architecture
    c.network_kwargs = dnnlib.EasyDict(
        class_name='training.networks_edm2.Precond',
        model_channels=128,
        channel_mult=[1, 2, 2, 2],
        dropout=0.30,
        use_fp16=True
    )
    c.ema_kwargs = dnnlib.EasyDict(class_name='training.phema.PowerFunctionEMA')
    c.cudnn_benchmark = True
    c.seed = 0

    if not dry_run:
        if dist.get_rank() == 0:
            os.makedirs(c.run_dir, exist_ok=True)
            with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
                json.dump(c, f, indent=2)

            dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

        training_loop.training_loop(**c)


# ----------------------------------------------------------------------------
# 3. Main Logic
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Frozen Core Checkpoints (EDM2 Native)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--partition", type=str, required=True, choices=["dirichlet", "silos"])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default="./data")
    return parser.parse_args()


def run_training_campaign():
    args_cli = parse_args()

    # Initialize Distributed environment via EDM2 utils if not already done
    if not torch.distributed.is_initialized():
        dist.init()

    SEED = 0
    # 5000 kimg = 5 million images shown.
    # For small datasets (e.g. 500 imgs), this is 10,000 epochs.
    TOTAL_KIMG = 25000

    if "fed_isic2019" in args_cli.dataset or "heart" in args_cli.dataset:
        TARGET_BATCH_SIZE = 64  # Safe size for high-res images
    else:
        TARGET_BATCH_SIZE = 4096  # Efficient size for low-res (32x32)

    # 1. Setup Arguments for prepare_data
    args = argparse.Namespace()
    args.dataset = args_cli.dataset
    args.data_dir = args_cli.data_dir
    args.partition = args_cli.partition
    args.alpha = args_cli.alpha
    args.num_clients = 10
    args.batch_size = 128
    args.client_config = ""
    args.seed = SEED
    args.grayscale = False
    args.input_size = 0
    args.model = "diffusion"
    args.eval_samples_per_class = 0
    args.robustness = False

    if dist.get_rank() == 0:
        logger.info(f"STARTING JOB: Dataset={args.dataset} | Partition={args.partition}")
        try:
            prime_dataset_meta_for_transform(args_cli.dataset, args_cli.data_dir)
        except Exception as e:
            logger.warning(f"Metadata priming skipped: {e}")

    # 2. Data Prep (Only Rank 0 usually needs to split, but for consistency we all run it)
    try:
        # We assume prepare_data is deterministic with seed
        (base_train_set, _, _, num_classes, img_shape, chans,
         train_subsets_dict, train_subsets, present_classes, _, _, _) = prepare_data(args, torch.device("cpu"),
                                                                                     partition_seed=SEED)
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        sys.exit(1)

    # Handle Silos
    if not train_subsets_dict:
        for cls_id, subset in zip(present_classes, train_subsets):
            client_name = f"client_{cls_id}"
            train_subsets_dict[client_name] = {cls_id: subset}

    # 3. Training Loop
    # Define Checkpoint Output Root
    part_folder = f"{args.partition}_{args.alpha}" if args.alpha is not None else args.partition
    safe_dataset_name = args.dataset.replace(":", "_")

    # Instead of 'checkpoints', we use a temp dir for materialized images
    materialized_root = Path("temp_materialized_datasets") / safe_dataset_name / part_folder
    model_save_root = Path("checkpoints") / safe_dataset_name / part_folder

    for client_name, class_subsets in train_subsets_dict.items():
        if dist.get_rank() == 0:
            logger.info(f"[{args.dataset}] Processing Client: {client_name}")

        for class_id, class_dataset in class_subsets.items():
            if len(class_dataset) == 0: continue

            # A. Materialize Data (Rank 0 only)
            dataset_dir = materialized_root / client_name / f"class_{class_id}"

            if dist.get_rank() == 0:
                # PASS img_shape HERE
                materialize_subset_to_disk(class_dataset, dataset_dir, class_id, target_shape=img_shape)

            # Wait for Rank 0 to finish writing files
            torch.distributed.barrier()

            # B. Setup Run Directory
            run_dir = model_save_root / client_name / f"class_{class_id}"

            # C. Adjust Batch Size for small datasets
            # If dataset is smaller than target batch, we must reduce target batch.
            # But EDM2 likes powers of 2.
            dataset_len = len(class_dataset)
            eff_batch_size = TARGET_BATCH_SIZE

            if dataset_len < eff_batch_size:
                # Find nearest power of 2 smaller than dataset size
                # e.g. 500 samples -> max batch 256 or 128
                eff_batch_size = 2 ** int(np.log2(dataset_len))
                # Minimum viable batch for BN statistics
                eff_batch_size = max(eff_batch_size, 32)

            if dist.get_rank() == 0:
                logger.info(f"   >>> Training Class {class_id} (N={dataset_len}). Batch set to {eff_batch_size}")

            # D. Run Training
            run_edm2_native(run_dir, dataset_dir, TOTAL_KIMG, eff_batch_size)

            # Wait before next client
            torch.distributed.barrier()

    if dist.get_rank() == 0:
        logger.info("✅ Training Campaign Finished.")
        # Cleanup temp images
        if os.path.exists(materialized_root):
            import shutil
            shutil.rmtree(materialized_root)
            logger.info("🧹 Cleaned up temporary datasets.")


if __name__ == "__main__":
    # Ensure spawn method for DDP
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    run_training_campaign()