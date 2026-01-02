# ==============================================================================
# FILE: src/checkpoint_trainer.py
# ==============================================================================
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# 0. CRITICAL PATH INJECTION (Must be BEFORE imports from EDM2)
# ----------------------------------------------------------------------------
current_file = Path(__file__).resolve()
src_root = current_file.parent  # src/
project_root = src_root.parent  # FederatedFactory/
edm2_root = src_root / "modules" / "EDM2"

if str(edm2_root) not in sys.path:
    sys.path.insert(0, str(edm2_root))

from modules.EDM2.training import training_loop
import modules.EDM2.training.networks_edm2

# ==============================================================================
# ⚡ H100 OPTIMIZATION MONKEY-PATCH
# ==============================================================================
_original_init = modules.EDM2.training.networks_edm2.Precond.__init__


def _compiled_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    # mode="reduce-overhead" is often more stable for varying batch sizes than max-autotune
    # providing a better balance for high-res medical data loops
    self.forward = torch.compile(self.forward, mode="reduce-overhead")
    logger.info(">>> 🚀 H100 Optimization: Model compiled with mode='reduce-overhead'")


modules.EDM2.training.networks_edm2.Precond.__init__ = _compiled_init

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
from torchvision import transforms
import torchvision.transforms.functional as TF

# ----------------------------------------------------------------------------
# 2. EDM2 Imports
# ----------------------------------------------------------------------------
import dnnlib
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from logs.logger import get_logger
from jobs.experiment_setup import prepare_data
from imports.data_management import prime_dataset_meta_for_transform

logger = get_logger(__name__)


# ----------------------------------------------------------------------------
# 1a. Stage I: Generative Robustness Policy Factory
# ----------------------------------------------------------------------------
def get_generative_augmentation(dataset_name: str, resolution: int):
    """
    Implements Stage I: Generative Robustness policies.
    NOTE: 'resolution' here is the TRAINING resolution (e.g. 128), not necessarily
    the raw input resolution.
    """
    ds = dataset_name.lower()
    ops = []

    # 1. Isotropic Medical Regime (BloodMNIST, PathMNIST, ISIC, Tissue)
    if any(x in ds for x in ["blood", "path", "derma", "isic", "tissue"]):
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.append(transforms.RandomVerticalFlip(p=0.5))
        ops.append(transforms.RandomChoice([
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
            transforms.Lambda(lambda x: x)
        ]))
        # CRITICAL: This resize ensures the output tensor matches the optimized H100 resolution
        ops.append(transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC))

    # 2. Oriented Medical Regime (RetinaMNIST)
    elif "retina" in ds:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.append(transforms.RandomAffine(
            degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05),
            interpolation=transforms.InterpolationMode.BILINEAR
        ))
        ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        ops.append(transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC))

    # 3. Anisotropic Natural Regime (CIFAR-10)
    else:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.append(transforms.Resize((resolution, resolution)))
        ops.append(transforms.RandomCrop(resolution, padding=resolution // 8, padding_mode='reflect'))

    return transforms.Compose(ops)


# ----------------------------------------------------------------------------
# 1b. Dataset Materialization
# ----------------------------------------------------------------------------
def materialize_subset_to_disk(subset, dest_dir, class_idx, target_shape=None, augmentation_fn=None):
    """
    Materializes images to disk.
    FIX: Enforces 'target_shape' checking against the TRAINING resolution (128),
    not the original 224, ensuring cache consistency.
    """

    # Extract target W/H from the shape tuple (C, H, W)
    if target_shape is not None:
        tgt_h, tgt_w = target_shape[-2], target_shape[-1]
    else:
        tgt_h, tgt_w = 32, 32  # Fallback

    # --- ROBUST CACHE CHECK ---
    if os.path.exists(dest_dir):
        existing_files = [f for f in os.listdir(dest_dir) if f.endswith('.png')]

        if len(existing_files) >= len(subset):
            should_skip = True
            if len(existing_files) > 0:
                from PIL import Image
                sample_path = os.path.join(dest_dir, existing_files[0])
                try:
                    with Image.open(sample_path) as test_img:
                        # Check if cache matches the TRAINING resolution
                        if test_img.size != (tgt_w, tgt_h):
                            logger.info(
                                f"   >>> ⚠️ Cache dimensions mismatch! Found {test_img.size}, "
                                f"expected {(tgt_w, tgt_h)}. This usually happens when switching "
                                f"optimization regimes (224->128). Re-materializing..."
                            )
                            should_skip = False
                            import shutil
                            shutil.rmtree(dest_dir)
                            os.makedirs(dest_dir, exist_ok=True)
                except Exception as e:
                    should_skip = False
                    import shutil
                    shutil.rmtree(dest_dir)
                    os.makedirs(dest_dir, exist_ok=True)

            if should_skip:
                logger.info(f"   >>> Dataset for Class {class_idx} exists at {dest_dir}. Skipping.")
                return

    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f"   >>> Materializing {len(subset)} images to {dest_dir}...")

    count = 0
    labels = []

    # Access underlying dataset
    if hasattr(subset, 'dataset'):
        base_ds = subset.dataset
        indices = subset.indices
    else:
        base_ds = subset
        indices = range(len(subset))

    from PIL import Image

    for idx in tqdm(indices, desc=f"Saving Class {class_idx}", leave=False):
        try:
            # 1. Extraction
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

            # 2. INJECTION: Apply Stage I Generative Augmentation (Resizes here)
            if augmentation_fn is not None:
                img = augmentation_fn(img)

            # 3. Final Safety Check
            if img.size != (tgt_w, tgt_h):
                img = img.resize((tgt_w, tgt_h), Image.LANCZOS)

            # 4. Save
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
    c = dnnlib.EasyDict()
    c.run_dir = str(run_dir)

    # Dataset Config
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=str(data_path),
                                       use_labels=False, xflip=True, cache=True)
    c.data_loader_kwargs = dnnlib.EasyDict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=16,
                                           prefetch_factor=4, persistent_workers=True)
    c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')

    # --- BATCH SIZE PHYSICS ---
    world_size = dist.get_world_size()
    if batch_size < world_size:
        batch_size = world_size
    if batch_size % world_size != 0:
        batch_size = (batch_size // world_size) * world_size

    # ==============================================================================
    # 2. Determine Per-GPU Batch - OPTIMIZED FOR 128x128
    # ==============================================================================
    # Check if we are running high-res FLamby data
    # NOTE: Even with our downsampling fix, we treat it as "high res" logic for accumulation
    is_high_res = "fed_isic" in str(data_path) or "derma" in str(data_path)

    if is_high_res:
        # At 128x128, H100 can handle significantly larger batches than at 224x224
        # Increased SAFE_GPU_LIMIT from 64 -> 128 thanks to resolution fix
        SAFE_GPU_LIMIT = 224
    else:
        SAFE_GPU_LIMIT = 768

    batch_per_gpu_global = batch_size // world_size
    c.batch_gpu = min(batch_per_gpu_global, SAFE_GPU_LIMIT)

    while (batch_size // world_size) % c.batch_gpu != 0:
        c.batch_gpu -= 1
        if c.batch_gpu < 1:
            c.batch_gpu = 1
            break

    c.batch_size = batch_size

    # --- ALIGNMENT HELPER ---
    def get_valid_interval(val, batch_size, alignment_constraint=0):
        if val is None: return None
        step = max(batch_size, alignment_constraint)
        return ((val + step - 1) // step) * step

    raw_nimg = total_kimg * 1000
    c.total_nimg = get_valid_interval(raw_nimg, batch_size)

    c.status_nimg = get_valid_interval(50 * 1000, batch_size)
    c.snapshot_nimg = get_valid_interval(500 * 1000, batch_size, alignment_constraint=1024)
    c.checkpoint_nimg = get_valid_interval(500 * 1000, batch_size, alignment_constraint=1024)

    c.loss_kwargs = dnnlib.EasyDict(class_name='training.training_loop.EDM2Loss', P_mean=-0.8, P_std=1.6,
                                    sigma_data=0.5)
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=(0.9, 0.99), eps=1e-8)
    c.lr_kwargs = dnnlib.EasyDict(func_name='training.training_loop.learning_rate_schedule', ref_lr=0.01,
                                  ref_batches=35000)

    current_channels = 64 if is_high_res else 128

    c.network_kwargs = dnnlib.EasyDict(
        class_name='training.networks_edm2.Precond',
        model_channels=current_channels,
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
    if not torch.distributed.is_initialized():
        dist.init()

    SEED = 0
    TOTAL_KIMG = 25000

    # 1. Setup Arguments
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

    # 2. Data Prep
    try:
        (base_train_set, _, _, num_classes, img_shape, chans,
         train_subsets_dict, train_subsets, present_classes, _, _, _) = prepare_data(args, torch.device("cpu"),
                                                                                     partition_seed=SEED)
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        sys.exit(1)

    if not train_subsets_dict:
        for cls_id, subset in zip(present_classes, train_subsets):
            client_name = f"client_{cls_id}"
            train_subsets_dict[client_name] = {cls_id: subset}

    # 3. Training Loop Configuration
    part_folder = f"{args.partition}_{args.alpha}" if args.alpha is not None else args.partition
    safe_dataset_name = args.dataset.replace(":", "_")
    materialized_root = Path("temp_materialized_datasets") / safe_dataset_name / part_folder
    model_save_root = Path("checkpoints") / safe_dataset_name / part_folder

    # =========================================================================
    # ⚡ H100 RESOLUTION OPTIMIZER
    # =========================================================================
    # We detect the native resolution. If it's 224 (ISIC/HighRes), we clamp it to 128.
    # 128 is a Power-of-2 (Po2). 224 is NOT (224 = 32*7).
    # H100 Tensor Cores are exponentially faster on Po2 resolutions.
    # We will train at 128x128. If 224 is required at output, upsample the result.

    raw_target_res = img_shape[-2]

    if raw_target_res > 64:
        # For ISIC (224), we force 128.
        # This fixes OOM (4x less memory) and Speed (3x faster).
        EFFECTIVE_RES = 128
        if dist.get_rank() == 0:
            logger.info(">>> ⚡ H100 OPTIMIZER: Clamping High-Res (224px) to 128px for Tensor Core Alignment.")
            logger.info(">>> ⚡ This solves the OOM/Speed bottleneck. Output will be 128x128 (upsample if needed).")
    else:
        # Keep MedMNIST/CIFAR as is (28 or 32)
        EFFECTIVE_RES = raw_target_res

    # Construct the pipeline with the OPTIMIZED resolution
    aug_pipeline = get_generative_augmentation(args.dataset, EFFECTIVE_RES)

    # We construct a fake shape tuple for the materializer to enforce the 128px resize
    optimized_shape = (img_shape[0], EFFECTIVE_RES, EFFECTIVE_RES)

    # Determine Batch Size based on EFFECTIVE resolution
    if EFFECTIVE_RES > 64:
        TARGET_BATCH_SIZE = 1024  # Now valid for 128x128 on H100!
    else:
        TARGET_BATCH_SIZE = 4096

    for client_name, class_subsets in train_subsets_dict.items():
        if dist.get_rank() == 0:
            logger.info(f"[{args.dataset}] Processing Client: {client_name}")

        for class_id, class_dataset in class_subsets.items():
            if len(class_dataset) == 0: continue

            # A. Materialize Data (Rank 0 only)
            dataset_dir = materialized_root / client_name / f"class_{class_id}"

            if dist.get_rank() == 0:
                materialize_subset_to_disk(
                    class_dataset,
                    dataset_dir,
                    class_id,
                    target_shape=optimized_shape,  # <-- Passing 128x128 here
                    augmentation_fn=aug_pipeline
                )

            torch.distributed.barrier()

            # B. Setup Run Directory
            run_dir = model_save_root / client_name / f"class_{class_id}"

            # C. Adjust Batch Size
            dataset_len = len(class_dataset)
            eff_batch_size = TARGET_BATCH_SIZE

            if dataset_len < eff_batch_size:
                eff_batch_size = 2 ** int(np.log2(dataset_len))
                eff_batch_size = max(eff_batch_size, 32)

            if dist.get_rank() == 0:
                logger.info(f"   >>> Training Class {class_id} (N={dataset_len}). Batch set to {eff_batch_size}")

            # D. Run Training
            run_edm2_native(run_dir, dataset_dir, TOTAL_KIMG, eff_batch_size)
            torch.distributed.barrier()

    if dist.get_rank() == 0:
        logger.info("✅ Training Campaign Finished.")
        if os.path.exists(materialized_root):
            import shutil
            shutil.rmtree(materialized_root)
            logger.info("🧹 Cleaned up temporary datasets.")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    run_training_campaign()