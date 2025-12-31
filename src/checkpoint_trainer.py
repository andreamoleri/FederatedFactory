# ==============================================================================
# FILE: src/checkpoint_trainer.py
# ==============================================================================
from __future__ import annotations
import argparse
import torch
import sys
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# Internal imports
from utils import set_seed
from logs.logger import get_logger
from jobs.experiment_setup import prepare_data
from models.diffusion import DiT, DiffusionConfig
from models.trainers import train_diffusion

from imports.data_management import prime_dataset_meta_for_transform
from imports.data_augmentation import build_transform

logger = get_logger(__name__)


class TransformSubset(Dataset):
    """
    Wraps a subset and applies the domain-aware augmentation policy
    (transform) when an item is retrieved.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Frozen Core Checkpoints (EDM2)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to train on (e.g. cifar10, medmnist:bloodmnist).")
    parser.add_argument("--partition", type=str, required=True,
                        choices=["dirichlet", "silos"],
                        help="Data partition strategy.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Concentration parameter for Dirichlet (ignored for Silos).")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Root directory for datasets.")
    return parser.parse_args()


def run_training_campaign():
    args_cli = parse_args()

    # ==============================================================================
    # CONFIGURATION
    # ==============================================================================
    SEED = 0
    EPOCHS = 10000
    CHECKPOINT_EVERY = 500

    hyperparams = {
        "dit_embed": 128,
        "dit_channel_mult": [1, 2, 2, 2],
        "dit_dropout": 0.30,
        # [FIX] BATCH SIZE PHYSICS
        "batch_size": 128,         # Physical Limit (VRAM constrained)
        "target_batch_size": 4096, # Mathematical Requirement (EDM2 Convergence)
        "dp": False,
        "lr": 0.01,                # [FIX] Increased LR for larger effective batch size
        "num_clients": 10
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    # 1. Setup Arguments for prepare_data
    args = argparse.Namespace()
    args.dataset = args_cli.dataset
    args.data_dir = args_cli.data_dir
    args.partition = args_cli.partition
    args.alpha = args_cli.alpha
    args.num_clients = hyperparams["num_clients"]
    args.batch_size = hyperparams["batch_size"]
    args.client_config = ""
    args.seed = SEED
    args.grayscale = False
    args.input_size = 0
    args.model = "diffusion"
    args.eval_samples_per_class = 0
    args.robustness = False

    logger.info(f"===========================================================")
    logger.info(f"STARTING JOB: Dataset={args.dataset} | Partition={args.partition}")
    logger.info(f"===========================================================")

    try:
        prime_dataset_meta_for_transform(args_cli.dataset, args_cli.data_dir)
    except Exception as e:
        logger.warning(f"Metadata priming skipped/failed: {e}")

    # 2. Data Prep
    try:
        (base_train_set, _, _, num_classes, img_shape, chans,
         train_subsets_dict, train_subsets, present_classes, _, _, _) = prepare_data(args, torch.device("cpu"), partition_seed=SEED)
    except Exception as e:
        logger.error(f"Failed to prepare data for {args.dataset}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # --- HANDLE SILOS PARTITIONING ---
    if not train_subsets_dict:
        if train_subsets and present_classes:
            logger.info("[PARTITION] Silos mode detected (List-based). Converting to Client Dict...")
            for cls_id, subset in zip(present_classes, train_subsets):
                client_name = f"client_{cls_id}"
                train_subsets_dict[client_name] = {cls_id: subset}
        else:
            logger.error(f"❌ CRITICAL ERROR: Both 'train_subsets_dict' and 'train_subsets' list are empty.")
            sys.exit(1)

    total_samples_check = sum(len(ds) for sub in train_subsets_dict.values() for ds in sub.values())
    logger.info(f"✅ Data Loaded. Found {len(train_subsets_dict)} clients with {total_samples_check} total samples.")

    if img_shape[1] == 28:
        logger.info("[CONFIG] Detected 28x28 resolution. Adjusting channel_mult to [1, 2, 2].")
        hyperparams["dit_channel_mult"] = [1, 2, 2]

    # [FIX] GRADIENT ACCUMULATION CALCULATION
    # Calculate how many physical batches sum up to one target batch
    grad_accum_steps = max(1, hyperparams["target_batch_size"] // hyperparams["batch_size"])
    effective_batch = hyperparams["batch_size"] * grad_accum_steps
    logger.info(f"⚡ [PHYSICS] Target Batch: {hyperparams['target_batch_size']} | Physical: {hyperparams['batch_size']}")
    logger.info(f"⚡ [PHYSICS] Accumulation Steps: {grad_accum_steps} (Effective: {effective_batch})")

    logger.info(f"[AUGMENTATION] Building Stage I policy for {args.dataset}")
    train_transform = build_transform(args.dataset, train=True, robustness=True)

    # 3. Define Checkpoint Output Root
    part_folder = f"{args.partition}_{args.alpha}" if args.alpha is not None else args.partition
    safe_dataset_name = args.dataset.replace(":", "_")
    save_root = Path("checkpoints") / safe_dataset_name / part_folder
    save_root.mkdir(parents=True, exist_ok=True)

    # 4. Initialize and Train
    dit_cfg = DiffusionConfig(
        in_ch=chans,
        embed_dim=hyperparams["dit_embed"],
        channel_mult=hyperparams["dit_channel_mult"],
        dropout=hyperparams["dit_dropout"],
        depth=hyperparams.get("dit_depth", 4),
        num_heads=hyperparams.get("dit_heads", 4),
        patch_size=hyperparams.get("dit_patch", 2),
        num_classes=0,
        img_resolution=img_shape[1]
    )

    training_performed = False

    for client_name, class_subsets in train_subsets_dict.items():
        logger.info(f"[{args.dataset}] Entering Client: {client_name}")

        for class_id, class_dataset in class_subsets.items():
            if len(class_dataset) == 0:
                continue

            training_performed = True
            logger.info(f"   >>> Training Class {class_id} for {client_name} (Size: {len(class_dataset)})")

            class_dir = save_root / client_name / f"class_{class_id}"
            class_dir.mkdir(parents=True, exist_ok=True)

            augmented_dataset = TransformSubset(class_dataset, transform=train_transform)

            loader = DataLoader(
                augmented_dataset,
                batch_size=hyperparams["batch_size"],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True # [FIX] Important for consistent batch stats in EDM2
            )

            model = DiT(dit_cfg).to(device)
            hist = {}

            train_diffusion(
                model=model,
                loader=loader,
                device=device,
                epochs=EPOCHS,
                hist=hist,
                cid=f"{client_name}_c{class_id}",
                dp=hyperparams["dp"],
                tracker=None,
                checkpoint_every=CHECKPOINT_EVERY,
                checkpoint_dir=class_dir,
                lr=hyperparams["lr"],
                grad_accum_steps=grad_accum_steps  # [FIX] Passing accumulation steps
            )

            del model
            del loader
            torch.cuda.empty_cache()

    if not training_performed:
        logger.error("❌ ERROR: Script finished but no training loops were entered.")
        sys.exit(1)

    logger.info("✅ Training Campaign Finished Successfully.")


if __name__ == "__main__":
    run_training_campaign()