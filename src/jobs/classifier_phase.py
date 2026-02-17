"""
🤖 Classifier Training and Orchestration Module
-----------------------------------------------

This module serves as the central orchestration engine for training, evaluating,
and ensembling classifiers within a federated or distributed learning context.
It supports hybrid training regimes utilizing both real client data and
synthetic data generated via Variational Autoencoders (VAEs) or Diffusion Models.

🧠 Purpose:
    To facilitate robust classifier training across heterogeneous data partitions
    (e.g., skewed distributions, data silos) by leveraging generative augmentation
    and ensemble inference strategies (Product of Experts).

🔧 Core Functionalities:
    • orchestrated training of classifiers (CNNs) in local or server-based modes
    • synthetic data generation from VAE decoders and Diffusion Transformers (DiT)
    • weighted sampling strategies to balance class distributions across clients
    • ensemble prediction aggregation using log-probability summation
    • dynamic label remapping for partial-class training scenarios
    • **Dynamic Data Augmentation** for both real and synthetic streams

🎯 Intended Use:
    • Federated Learning research (specifically data heterogeneity)
    • Synthetic Data Augmentation experiments
    • Comparative analysis of "Local" vs. "Global" model performance

📁 Dependencies:
    • torch
    • numpy
    • sklearn
    • internal models (cnn, vae, diffusion, trainers)
    • imports.data_augmentation

📝 Notes:
    The module assumes the existence of a specific experiment tracking interface
    (`tracker`) and a file-system path configuration object (`P`).

Author: Andrea Moleri
File Location: src/jobs/classifier_phase.py
Last Modified: 12/12/2025
"""

from __future__ import annotations
import logging
import time
from typing import List, Dict, Optional, Tuple, Any, Set, Union

import torch
import torch.nn.functional as F  # Added for interpolation
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Dataset, random_split, Subset
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm  # Added for progress bars

# Internal Imports
from models.cnn import SimpleCNN
from models.trainers import train_classifier
from models.vae import Decoder
from models.diffusion import DiT, rectified_flow_sampler
from imports.data_augmentation import build_transform

logger = logging.getLogger(__name__)

# =============================================================================
# Dynamic Data Wrappers
# =============================================================================

class TransformSubset(Dataset):
    """
    Wraps a standard Subset or Dataset and forces a specific transform
    to be applied on __getitem__.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]

        # --- SYNTHETIC DATA ADAPTER ---
        # If input data is a Tensor (Synthetic), transform to PIL first
        if isinstance(x, torch.Tensor):
            # Synthetic data is typically [-1, 1]. Shift to [0, 1]
            x = (x + 1.0) / 2.0
            x = x.clamp(0, 1)
            # Convert to PIL Image to be compatible with transforms.ToTensor()
            x = TF.to_pil_image(x)

        if self.transform:
            x = self.transform(x)

        # --- [FIX] HARMONIZE LABEL TYPES ---
        # Real data loaders often return 'int', but synthetic TensorDataset returns 'Tensor'.
        # We enforce Tensor type here so the DataLoader collate_fn doesn't crash.
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        return x, y

    def __len__(self):
        return len(self.subset)


class TensorTransformDataset(Dataset):
    """
    Wraps static Tensors (e.g., synthetic data) and applies a transform
    on __getitem__. Ensures synthetic data gets the same noise/crops as real data.
    """
    def __init__(self, data_tensor: torch.Tensor, targets_tensor: torch.Tensor, transform=None):
        self.data = data_tensor
        self.targets = targets_tensor
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        # --- SYNTHETIC DATA ADAPTER ---
        if isinstance(x, torch.Tensor):
            x = (x + 1.0) / 2.0
            x = x.clamp(0, 1)
            x = TF.to_pil_image(x)

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

class GPUTransformDataset(Dataset):
    """
    Optimized Dataset for data already on GPU.
    Performs transforms directly on GPU tensors to avoid CPU<->GPU transfers.
    """
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        # Data is already on GPU in range [-1, 1] or [0, 1]
        # Assume standard synthetic range [-1, 1] for now, normalize to [0, 1] for transforms
        x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)

        # Apply transforms if they support GPU tensors (kornia/torchvision)
        # Note: Standard torchvision transforms might expect PIL or CPU tensors.
        # We assume here we are using simple transforms or custom ones.
        # For simplicity in this optimization step, if transform expects PIL, we skip or adapt.
        # Ideally, use Kornia for GPU transforms. Here we rely on torchvision's ability to handle tensors.
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

# =============================================================================
# Synthetic Generation Utilities
# =============================================================================

@torch.no_grad()
def synth_from_decoder(dec: Decoder, latent: int, n: int, device: torch.device) -> torch.Tensor:
    """
    Generates synthetic images using a pre-trained VAE decoder.
    Returns tensors on CPU.
    """
    dec.to(device).eval()
    # Batch generation if n is large to avoid VRAM OOM
    batch_size = 200 # Safe batch size
    imgs_list = []

    for i in range(0, n, batch_size):
        curr_batch = min(batch_size, n - i)
        z = torch.randn(curr_batch, latent, device=device)
        batch_imgs = dec(z).cpu()
        imgs_list.append(batch_imgs)

    dec.cpu()
    return torch.cat(imgs_list, dim=0)

@torch.no_grad()
def synth_from_diffusion(model: DiT, n: int, device: torch.device, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Generates synthetic images using a Diffusion Transformer (DiT).

    [RESOLUTION FIX]
    1. Detects Native Resolution of the checkpoint (e.g., 128).
    2. Generates at Native Resolution.
    3. Upsamples to Target Resolution (e.g., 224) if they differ.
    """
    C, H_target, W_target = target_shape

    # 1. Determine Native Resolution
    # EDM2 models store resolution in edm_net.img_resolution
    native_res = getattr(model.edm_net, 'img_resolution', H_target)

    model.to(device).eval()

    # 2. Generate at Native Resolution
    # rectified_flow_sampler already handles batching internally (max_batch=64)
    x_native = rectified_flow_sampler(
        model,
        n=n,
        shape=(C, native_res, native_res), # Use NATIVE shape
        steps=50,
        device=device
    )

    model.cpu()

    # 3. Resize if Native != Target
    if native_res != H_target or native_res != W_target:
        # Bicubic is standard for upsampling images for evaluation (e.g. Inception)
        x_out = F.interpolate(
            x_native,
            size=(H_target, W_target),
            mode='bicubic',
            antialias=True
        )
        return x_out.cpu()

    return x_native.cpu()

def generate_weighted_samples(
        client_gen_models: Dict[str, Dict[int, Any]],
        client_sample_counts: Dict[str, Dict[int, int]],
        samples_per_class: int,
        model_kind: str,
        latent_dim: int,
        device: torch.device,
        img_shape: Tuple[int, int, int],
        pre_generated_data: Optional[Dict[int, torch.Tensor]] = None
) -> Dict[int, torch.Tensor]:
    """
    Orchestrates weighted synthetic data generation across distributed clients.
    Includes caching support to avoid redundant generation.
    """
    weighted_samples = {}
    total_samples_per_class = {}
    all_classes = set()

    for _, counts in client_sample_counts.items():
        for cid, c in counts.items():
            all_classes.add(cid)
            total_samples_per_class[cid] = total_samples_per_class.get(cid, 0) + c

    # Add progress bar for weighted generation
    pbar = tqdm(sorted(all_classes), desc="Weighted Synth Gen", unit="class")

    for cid in pbar:
        target = samples_per_class if samples_per_class > 0 else total_samples_per_class.get(cid, 0)
        if target == 0:
            weighted_samples[cid] = torch.tensor([])
            continue

        # [CACHE CHECK]
        cached_images = None
        if pre_generated_data is not None and cid in pre_generated_data:
            cached_images = pre_generated_data[cid]

        current_count = len(cached_images) if cached_images is not None else 0
        needed = max(0, target - current_count)

        # If cache covers everything, skip generation logic
        if needed == 0 and cached_images is not None:
             weighted_samples[cid] = cached_images[:target]
             continue

        # Determine contributors for whatever is NEEDED
        contributors = []
        for cname, models in client_gen_models.items():
            if cid in models:
                contributors.append((cname, models[cid], client_sample_counts[cname].get(cid, 0)))

        contributors.sort(key=lambda x: x[2], reverse=True)
        total_real = total_samples_per_class[cid]

        cls_samples = []
        # Add cached images first
        if cached_images is not None:
            cls_samples.append(cached_images)

        remaining = needed # We only generate what is missing

        for i, (cname, model, count) in enumerate(contributors):
            # Proportional allocation of the *remaining/needed* amount?
            # Or proportional allocation of total target?
            # Standard approach: We need 'needed' more images. We distribute 'needed' proportionally.
            if i == len(contributors) - 1: n = remaining
            else: n = int(needed * (count / total_real))

            n = min(n, remaining)
            if n <= 0: continue

            if model_kind == "vae": xs = synth_from_decoder(model, latent_dim, n, device)
            else: xs = synth_from_diffusion(model, n, device, img_shape)
            cls_samples.append(xs)
            remaining -= n

        weighted_samples[cid] = torch.cat(cls_samples) if cls_samples else torch.tensor([])

    return weighted_samples

# =============================================================================
# Evaluation Utilities
# =============================================================================

def ensemble_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())

@torch.no_grad()
def ensemble_preds_poexp(classifiers: List[SimpleCNN], test_ld: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Product of Experts Ensemble. Returns (y_true, y_pred, y_probs).
    """
    for c in classifiers: c.to(device).eval()
    y_true, y_pred, y_probs = [], [], []

    for x, y in test_ld:
        x, y = x.to(device), y.to(device)
        log_probs = None
        for c in classifiers:
            # We work in log space for stability
            lp = torch.log_softmax(c(x), dim=1).clamp(min=np.log(1e-12))
            log_probs = lp if log_probs is None else log_probs + lp

        # Aggregate log_probs -> actual probabilities
        final_probs = torch.softmax(log_probs, dim=1)

        y_pred.append(log_probs.argmax(1).cpu())
        y_true.append(y.cpu())
        y_probs.append(final_probs.cpu())

    for c in classifiers: c.cpu()
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), torch.cat(y_probs).numpy()

@torch.no_grad()
def evaluate_single_classifier(model: SimpleCNN, ld: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (y_true, y_pred, y_probs).
    """
    model.to(device).eval()
    y_true, y_pred, y_probs = [], [], []
    for x, y in ld:
        # Move inputs to device if they aren't already there
        if x.device != device:
            x = x.to(device)

        logits = model(x)

        # Move targets to cpu for accumulation if they aren't already
        y_cpu = y.cpu() if y.device != torch.device('cpu') else y
        y_true.append(y_cpu)

        y_pred.append(logits.argmax(1).cpu())
        y_probs.append(torch.softmax(logits, dim=1).cpu())

    model.cpu()
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), torch.cat(y_probs).numpy()


# =============================================================================
# Main Training Orchestrator
# =============================================================================

def run_classifier_training(
        args: Any,
        device: torch.device,
        P: Any,
        train_subsets_dict: Dict[str, Dict[int, Any]],
        train_subsets: List[Any],
        present_classes: List[int],
        num_classes: int,
        chans: int,
        img_shape: Tuple[int, int, int],
        tracker: Optional[Any],
        hist: Any,
        gen_models: List[Optional[Any]],
        client_gen_models: Dict[str, Dict[int, Any]],
        client_sample_counts: Dict[str, Dict[int, int]],
        reserved_test_ld: DataLoader,
        pre_generated_data: Optional[Dict[int, torch.Tensor]] = None
) -> Tuple[int, int, float, np.ndarray, np.ndarray, np.ndarray, List[Any], Optional[Any], Dict[int, torch.Tensor]]:
    # ^ CHANGED Return Type hint to include Dict[int, torch.Tensor]

    classifier_steps = 0
    synth_images_total = 0
    clf_start = time.perf_counter()

    partition_mode = getattr(args, "partition", "silos")
    is_skew = partition_mode in ["skew", "dirichlet"]
    is_local = args.infer_mode == "local"

    trained_clfs = []
    single_clf = None
    y_true, y_pred, y_probs = np.array([]), np.array([]), np.array([])

    # NEW: Cache for synthetic data to prevent re-generation in evaluation phase
    # Initialize with pre_generated_data if available
    synthetic_cache: Dict[int, torch.Tensor] = pre_generated_data.copy() if pre_generated_data else {}

    # [RESOLUTION FIX] Ensure Cached Data Matches Target Resolution
    # If eval phase generated 128x128 but we need 224x224 for clf, resize cache now.
    target_res = img_shape[-1]
    for cid, cached_imgs in synthetic_cache.items():
        if len(cached_imgs) > 0 and cached_imgs.shape[-1] != target_res:
            synthetic_cache[cid] = F.interpolate(
                cached_imgs,
                size=(target_res, target_res),
                mode='bicubic',
                antialias=True
            )

    logger.info(f"[CLF-PHASE] Partition: {partition_mode}, Local Inference: {is_local}")

    # =========================================================================
    # [CRITICAL OOM FIX] VRAM CLEANUP BEFORE CLASSIFIER TRAINING
    # =========================================================================
    logger.info("[MEMORY] Offloading Generative Models to CPU to free VRAM for Classifier...")

    # 1. Move Client Generators to CPU
    for cname, models in client_gen_models.items():
        for cid, model in models.items():
            if model is not None:
                model.to("cpu")

    # 2. Move Global/List Generators to CPU
    for model in gen_models:
        if model is not None:
            model.to("cpu")

    # 3. Ensure Synthetic Data Cache is on CPU (Crucial)
    # If the cache is on GPU, it will explode memory when concatenated
    if synthetic_cache:
        for k, v in synthetic_cache.items():
            if isinstance(v, torch.Tensor):
                synthetic_cache[k] = v.to("cpu")

    # 4. Hard Flush
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"[MEMORY] VRAM Flushed. Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # =========================================================================

    # =========================================================================
    # PARTITION MODE: SKEW / DIRICHLET
    # =========================================================================
    if is_skew:
        if is_local:
            # -----------------------------------------------------------------
            # Case 1: Local Inference with Skewed Data
            # -----------------------------------------------------------------
            for cname, subsets in train_subsets_dict.items():
                logger.info(f"[SKEW-LOCAL] Preparing Client {cname}...")

                train_tf = build_transform(args.dataset, train=True, robustness=True)
                eval_tf = build_transform(args.dataset, train=False, robustness=False)

                real_subsets = list(subsets.values())
                if not real_subsets: continue
                real_ds_raw = ConcatDataset(real_subsets)

                val_len_real = int(len(real_ds_raw) * 0.2)
                tr_len_real = len(real_ds_raw) - val_len_real

                r_train_raw, r_val_raw = random_split(
                    real_ds_raw, [tr_len_real, val_len_real],
                    generator=torch.Generator().manual_seed(42)
                )

                r_train_ds = TransformSubset(r_train_raw, train_tf)
                r_val_ds = TransformSubset(r_val_raw, eval_tf)

                target = args.samples_per_class if args.samples_per_class > 0 else 2000
                synth_xs, synth_ys = [], []

                gen_desc = f"Client {cname} Generation ({args.model})"
                missing_classes = [cid for cid in range(num_classes) if cid not in subsets]

                if missing_classes:
                    logger.info(f"  Generating {target} synthetic images for {len(missing_classes)} missing classes...")

                    for cid in tqdm(missing_classes, desc=gen_desc):
                        s_imgs = torch.tensor([])

                        # [CACHE LOGIC START]
                        # 1. Check Cache
                        cached_images = synthetic_cache.get(cid)
                        current_count = len(cached_images) if cached_images is not None else 0
                        needed = max(0, target - current_count)

                        imgs_list = []
                        if cached_images is not None:
                             imgs_list.append(cached_images[:target]) # Take up to target
                             if needed == 0:
                                 #logger.info(f"  [Cache] Reusing {target} images for Class {cid}")
                                 pass

                        # 2. Generate Delta if needed
                        if needed > 0:
                            if getattr(args, "aggregation", "simple") == "weighted":
                                w_samps = generate_weighted_samples(client_gen_models, client_sample_counts, needed,
                                                                    args.model, args.latent_dim, device, img_shape,
                                                                    # Important: pass None here to avoid recursive cache check inside
                                                                    pre_generated_data=None)
                                new_imgs = w_samps.get(cid, torch.tensor([]))
                                imgs_list.append(new_imgs.cpu())
                            else:
                                if cid < len(gen_models) and gen_models[cid] is not None:
                                    m = gen_models[cid]
                                    if args.model == "vae":
                                        new_imgs = synth_from_decoder(m, args.latent_dim, needed, device)
                                    else:
                                        new_imgs = synth_from_diffusion(m, needed, device, img_shape)
                                    imgs_list.append(new_imgs.cpu())

                        if imgs_list:
                            s_imgs = torch.cat(imgs_list)

                            # [RESOLUTION FIX] Resize VAE/Cached to Target if mismatched
                            if s_imgs.shape[-1] != target_res:
                                s_imgs = F.interpolate(s_imgs, size=(target_res, target_res), mode='bicubic', antialias=True)

                            # Update Cache with full set
                            synthetic_cache[cid] = s_imgs
                        # [CACHE LOGIC END]

                        if len(s_imgs) > 0:
                            synth_xs.append(s_imgs)
                            synth_ys.append(torch.full((len(s_imgs),), cid, dtype=torch.long))

                if synth_xs:
                    s_X = torch.cat(synth_xs)
                    s_y = torch.cat(synth_ys)
                    synth_images_total += len(s_X)

                    s_raw_ds = TensorDataset(s_X, s_y)
                    val_len_s = int(len(s_raw_ds) * 0.2)
                    tr_len_s = len(s_raw_ds) - val_len_s

                    s_train_raw, s_val_raw = random_split(
                        s_raw_ds, [tr_len_s, val_len_s],
                        generator=torch.Generator().manual_seed(42)
                    )

                    s_train_ds = TransformSubset(s_train_raw, train_tf)
                    s_val_ds = TransformSubset(s_val_raw, eval_tf)

                    final_train_ds = ConcatDataset([r_train_ds, s_train_ds])
                    final_val_ds = ConcatDataset([r_val_ds, s_val_ds])
                else:
                    final_train_ds = r_train_ds
                    final_val_ds = r_val_ds

                tr_ld = DataLoader(
                    final_train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=True,
                    persistent_workers=True  # Keeps workers alive between epochs
                )
                val_ld = DataLoader(
                    final_val_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=True,
                    persistent_workers=True
                )

                if tracker: tracker.start_phase(f"client_{cname}_clf")
                clf, steps = train_classifier(SimpleCNN(chans, num_classes, input_resolution=img_shape[-1]), tr_ld, val_ld, device, args.clf_epochs,
                                              hist, cname, tracker=tracker)
                if tracker: tracker.end_phase(f"client_{cname}_clf")

                trained_clfs.append(clf)
                classifier_steps += steps
                torch.save(clf.state_dict(), P.root / "models" / "classifiers" / f"client-{cname}.pt")

            if tracker: tracker.start_phase("ensemble_evaluation")
            y_true, y_pred, y_probs = ensemble_preds_poexp(trained_clfs, reserved_test_ld, device)
            if tracker: tracker.end_phase("ensemble_evaluation")

        else:
            # -----------------------------------------------------------------
            # Case 2: Server-Side Inference with Skewed Data (Pure Synthetic)
            # -----------------------------------------------------------------
            if tracker: tracker.start_phase("server_generation")
            target = args.samples_per_class if args.samples_per_class > 0 else 2000
            logger.info(f"[SERVER] Generating {target} synthetic images per class...")

            if getattr(args, "aggregation", "simple") == "weighted":
                # The generate_weighted_samples function now handles caching internally if passed
                synth_samples = generate_weighted_samples(client_gen_models, client_sample_counts, target, args.model,
                                                          args.latent_dim, device, img_shape,
                                                          pre_generated_data=synthetic_cache)
            else:
                synth_samples = {}
                for cid in tqdm(range(num_classes), desc="Server Gen"):
                    # [CACHE LOGIC START]
                    cached_images = synthetic_cache.get(cid)
                    current_count = len(cached_images) if cached_images is not None else 0
                    needed = max(0, target - current_count)

                    imgs_list = []
                    if cached_images is not None:
                         imgs_list.append(cached_images[:target])

                    if needed > 0:
                        if cid < len(gen_models) and gen_models[cid]:
                            if args.model == "vae":
                                new_imgs = synth_from_decoder(gen_models[cid], args.latent_dim, needed, device)
                            else:
                                new_imgs = synth_from_diffusion(gen_models[cid], needed, device, img_shape)
                            imgs_list.append(new_imgs.cpu())

                    if imgs_list:
                        full_set = torch.cat(imgs_list)

                        # [RESOLUTION FIX]
                        if full_set.shape[-1] != target_res:
                            full_set = F.interpolate(full_set, size=(target_res, target_res), mode='bicubic', antialias=True)

                        synth_samples[cid] = full_set
                        synthetic_cache[cid] = full_set # Update cache
                    # [CACHE LOGIC END]

            if tracker: tracker.end_phase("server_generation")

            xs, ys = [], []
            for cid, imgs in synth_samples.items():
                if len(imgs) > 0:
                    xs.append(imgs)
                    ys.append(torch.full((len(imgs),), cid, dtype=torch.long))
                    synth_images_total += len(imgs)

            if xs:
                # Concatenate everything on CPU
                X, y = torch.cat(xs), torch.cat(ys)

                # Move entire dataset to GPU for speed
                logger.info(f"🚀 [SPEED] Moving entire synthetic dataset to GPU ({X.element_size() * X.numel() / 1024**3:.2f} GB)...")
                X = X.to(device)
                y = y.to(device)

                present_in_synth = sorted(list(set(y.cpu().tolist()))) # calculate classes on CPU list for speed
                label_map = {old: new for new, old in enumerate(present_in_synth)}
                y_mapped = torch.tensor([label_map[v.item()] for v in y], dtype=torch.long, device=device) # Map labels on GPU

                val_len = int(len(X) * 0.1)
                tr_len = len(X) - val_len

                # Manual split on GPU tensors
                indices = torch.randperm(len(X), device=device)
                train_idx = indices[:tr_len]
                val_idx = indices[tr_len:]

                # Create datasets pointing to GPU tensors
                # We use a simple TensorDataset but need to apply transforms manually or use a custom one.
                # Standard transforms like RandomCrop don't work natively on batches of GPU tensors in standard DataLoader.
                # We will use the custom GPUTransformDataset

                train_tf = build_transform(args.dataset, train=True, robustness=True)
                eval_tf = build_transform(args.dataset, train=False, robustness=False)

                # For GPU resident data, we set num_workers=0 to avoid multiprocessing overhead and copying
                tr_ld = DataLoader(
                    TensorDataset(X[train_idx], y_mapped[train_idx]),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0
                )

                val_ld = DataLoader(
                    TensorDataset(X[val_idx], y_mapped[val_idx]),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0
                )

                if tracker: tracker.start_phase("server_classifier")
                clf, steps = train_classifier(SimpleCNN(chans, len(present_in_synth), input_resolution=img_shape[-1]), tr_ld, val_ld, device,
                                              args.clf_epochs, hist, "server", tracker=tracker)
                if tracker: tracker.end_phase("server_classifier")

                classifier_steps += steps
                torch.save(clf.state_dict(), P.root / "models" / "classifiers" / "central.pt")
                single_clf = clf

                if tracker: tracker.start_phase("server_evaluation")
                # Prepare test set on GPU if possible
                all_imgs, all_lbls = [], []
                for x, lbl in reserved_test_ld:
                    all_imgs.append(x)
                    all_lbls.append(lbl)

                if all_imgs:
                    all_imgs = torch.cat(all_imgs).to(device)
                    all_lbls = torch.cat(all_lbls).to(device)

                    mask = torch.tensor([l.item() in label_map for l in all_lbls.cpu()], dtype=torch.bool, device=device)
                    if mask.sum() > 0:
                        filt_imgs = all_imgs[mask]
                        filt_lbls = torch.tensor([label_map[l.item()] for l in all_lbls[mask].cpu()], dtype=torch.long, device=device)

                        test_ld_mapped = DataLoader(TensorDataset(filt_imgs, filt_lbls), batch_size=args.batch_size, num_workers=0)

                        yt_map, yp_map, yprobs_map = evaluate_single_classifier(clf, test_ld_mapped, device)

                        rev_map = {v: k for k, v in label_map.items()}
                        y_true = np.array([rev_map[v] for v in yt_map])
                        y_pred = np.array([rev_map[v] for v in yp_map])

                        y_probs = np.zeros((len(yprobs_map), num_classes), dtype=np.float32)
                        for i, mapped_idx in enumerate(present_in_synth):
                            y_probs[:, mapped_idx] = yprobs_map[:, i]

                if tracker: tracker.end_phase("server_evaluation")

    # =========================================================================
    # PARTITION MODE: SILOS
    # =========================================================================
    else:
        if is_local:
            # -----------------------------------------------------------------
            # Case 3: Local Inference within Data Silos
            # -----------------------------------------------------------------
            for i, d in enumerate(present_classes):
                logger.info(f"[SILOS-LOCAL] Preparing Client {d} (Class {d})...")

                train_tf = build_transform(args.dataset, train=True, robustness=True)
                eval_tf = build_transform(args.dataset, train=False, robustness=False)

                real_subset = train_subsets[i]
                val_len_real = int(len(real_subset) * 0.2)
                tr_len_real = len(real_subset) - val_len_real

                r_train_raw, r_val_raw = random_split(
                    real_subset, [tr_len_real, val_len_real],
                    generator=torch.Generator().manual_seed(42)
                )

                r_train_ds = TransformSubset(r_train_raw, train_tf)
                r_val_ds = TransformSubset(r_val_raw, eval_tf)

                n_synth = args.samples_per_class if args.samples_per_class > 0 else 2000
                s_xs, s_ys = [], []

                if n_synth > 0:
                    valid_gen_models = [(od, gm) for od, gm in enumerate(gen_models) if od != d and gm is not None]

                    if valid_gen_models:
                        logger.info(
                            f"  Generating {n_synth} synthetic images from {len(valid_gen_models)} external models...")
                        for od, gm in tqdm(valid_gen_models, desc=f"Client {d} Synth Gen"):
                            # [CACHE LOGIC START]
                            cached_images = synthetic_cache.get(od)
                            current_count = len(cached_images) if cached_images is not None else 0
                            needed = max(0, n_synth - current_count)

                            imgs_list = []
                            if cached_images is not None:
                                imgs_list.append(cached_images[:n_synth])

                            if needed > 0:
                                if args.model == "vae":
                                    im = synth_from_decoder(gm, args.latent_dim, needed, device)
                                else:
                                    im = synth_from_diffusion(gm, needed, device, img_shape)
                                imgs_list.append(im.cpu())

                            if imgs_list:
                                im = torch.cat(imgs_list)
                                # [RESOLUTION FIX]
                                if im.shape[-1] != target_res:
                                    im = F.interpolate(im, size=(target_res, target_res), mode='bicubic', antialias=True)

                                synthetic_cache[od] = im # Update cache
                                s_xs.append(im)
                                s_ys.append(torch.full((len(im),), od, dtype=torch.long))
                            # [CACHE LOGIC END]

                if s_xs:
                    s_X = torch.cat(s_xs)
                    s_y = torch.cat(s_ys)
                    synth_images_total += len(s_X)

                    s_raw_ds = TensorDataset(s_X, s_y)
                    val_len_s = int(len(s_raw_ds) * 0.2)
                    tr_len_s = len(s_raw_ds) - val_len_s

                    s_train_raw, s_val_raw = random_split(
                        s_raw_ds, [tr_len_s, val_len_s],
                        generator=torch.Generator().manual_seed(42)
                    )

                    s_train_ds = TransformSubset(s_train_raw, train_tf)
                    s_val_ds = TransformSubset(s_val_raw, eval_tf)

                    train_ds = ConcatDataset([r_train_ds, s_train_ds])
                    val_ds = ConcatDataset([r_val_ds, s_val_ds])
                else:
                    train_ds = r_train_ds
                    val_ds = r_val_ds

                # NEW CODE:
                tr_ld = DataLoader(
                    train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=True,
                    persistent_workers=True
                )
                val_ld = DataLoader(
                    val_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=True,
                    persistent_workers=True
                )

                if tracker: tracker.start_phase(f"client_{d:03d}_clf")
                clf, steps = train_classifier(SimpleCNN(chans, num_classes, input_resolution=img_shape[-1]), tr_ld, val_ld, device, args.clf_epochs,
                                              hist, d, tracker=tracker)
                if tracker: tracker.end_phase(f"client_{d:03d}_clf")

                trained_clfs.append(clf)
                classifier_steps += steps
                torch.save(clf.state_dict(), P.root / "models" / "classifiers" / f"client-{d:03d}.pt")

            if tracker: tracker.start_phase("ensemble_evaluation")
            y_true, y_pred, y_probs = ensemble_preds_poexp(trained_clfs, reserved_test_ld, device)
            if tracker: tracker.end_phase("ensemble_evaluation")

        else:
            # -----------------------------------------------------------------
            # Case 4: Server Inference Aggregating Silos
            # -----------------------------------------------------------------
            if tracker: tracker.start_phase("server_generation")
            if args.samples_per_class > 0:
                n_synth = args.samples_per_class
            else:
                n_synth = max(len(s) for s in train_subsets) if train_subsets else 1000

            logger.info(f"[SERVER] Generating {n_synth} synthetic images per silo model...")

            xs, ys = [], []
            valid_models_zip = [(d, gm) for d, gm in zip(present_classes, gen_models) if gm is not None]

            for d, gm in tqdm(valid_models_zip, desc="Server Gen"):
                # [CACHE LOGIC START]
                cached_images = synthetic_cache.get(d)
                current_count = len(cached_images) if cached_images is not None else 0
                needed = max(0, n_synth - current_count)

                imgs_list = []
                if cached_images is not None:
                     imgs_list.append(cached_images[:n_synth])

                if needed > 0:
                    if args.model == "vae":
                        im = synth_from_decoder(gm, args.latent_dim, needed, device)
                    else:
                        im = synth_from_diffusion(gm, needed, device, img_shape)
                    imgs_list.append(im.cpu())

                if imgs_list:
                    im = torch.cat(imgs_list)
                    # [RESOLUTION FIX]
                    if im.shape[-1] != target_res:
                        im = F.interpolate(im, size=(target_res, target_res), mode='bicubic', antialias=True)

                    synthetic_cache[d] = im # Update cache
                    xs.append(im)
                    ys.append(torch.full((len(im),), d, dtype=torch.long))
                # [CACHE LOGIC END]

            X, y = torch.cat(xs), torch.cat(ys)
            synth_images_total = len(X)
            if tracker: tracker.end_phase("server_generation")

            # ----------------------------------------------------------------------
            # OPTIMIZATION: LOAD ENTIRE DATASET TO GPU
            # ----------------------------------------------------------------------
            logger.info(f"🚀 [SPEED] Moving entire dataset to GPU ({X.element_size() * X.numel() / 1024**3:.2f} GB)...")
            X = X.to(device)
            y = y.to(device)

            val_len = int(len(X) * 0.1)
            tr_len = len(X) - val_len

            # Manual split on GPU tensors using indices
            indices = torch.randperm(len(X), device=device)
            train_idx = indices[:tr_len]
            val_idx = indices[tr_len:]

            # IMPORTANT: TensorDataset creates dataset from tensors.
            # If we pass GPU tensors, it respects that.
            # We set num_workers=0 to avoid serialization overhead which is catastrophic for GPU tensors.

            tr_ld = DataLoader(
                TensorDataset(X[train_idx], y[train_idx]),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0
            )

            val_ld = DataLoader(
                TensorDataset(X[val_idx], y[val_idx]),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0
            )

            if tracker: tracker.start_phase("server_classifier")
            clf, steps = train_classifier(SimpleCNN(chans, num_classes, input_resolution=img_shape[-1]), tr_ld, val_ld, device, args.clf_epochs, hist,
                                          "server", tracker=tracker)
            if tracker: tracker.end_phase("server_classifier")

            classifier_steps += steps
            torch.save(clf.state_dict(), P.root / "models" / "classifiers" / "central.pt")
            single_clf = clf

            if tracker: tracker.start_phase("server_evaluation")

            # Prepare test data on GPU
            all_imgs, all_lbls = [], []
            for x, lbl in reserved_test_ld:
                all_imgs.append(x)
                all_lbls.append(lbl)

            if all_imgs:
                test_X = torch.cat(all_imgs).to(device)
                test_y = torch.cat(all_lbls).to(device)

                # We need a DataLoader for evaluation function
                # Even if on GPU, we batch it to avoid OOM during inference if dataset is huge
                test_ld_gpu = DataLoader(TensorDataset(test_X, test_y), batch_size=args.batch_size, num_workers=0)

                y_true, y_pred, y_probs = evaluate_single_classifier(clf, test_ld_gpu, device)

            if tracker: tracker.end_phase("server_evaluation")

    # Pass the populated synthetic_cache back to the experiment runner
    return classifier_steps, synth_images_total, clf_start, y_true, y_pred, y_probs, trained_clfs, single_clf, synthetic_cache
