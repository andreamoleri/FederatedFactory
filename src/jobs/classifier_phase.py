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

    CRITICAL FIX: This class now automatically handles Synthetic Tensors.
    If the input data is a Tensor (e.g., from Diffusion), it converts it back
    to a PIL image so that standard torchvision transforms (which often include
    ToTensor) work correctly without crashing.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]

        # --- SYNTHETIC DATA ADAPTER ---
        # If input is a Tensor (Synthetic), transform to PIL first
        if isinstance(x, torch.Tensor):
            # Synthetic data is typically [-1, 1]. Shift to [0, 1]
            x = (x + 1.0) / 2.0
            x = x.clamp(0, 1)
            # Convert to PIL Image to be compatible with transforms.ToTensor()
            x = TF.to_pil_image(x)

        if self.transform:
            x = self.transform(x)
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
def synth_from_diffusion(model: DiT, n: int, device: torch.device, img_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Generates synthetic images using a Diffusion Transformer (DiT) via Rectified Flow.
    Returns tensors on CPU.
    """
    C, H, W = img_shape
    model.to(device).eval()
    # rectified_flow_sampler already handles batching internally (max_batch=64)
    # steps=50 is standard, reduced for speed if necessary, but kept at 50 for quality
    x = rectified_flow_sampler(model, n=n, shape=(C, H, W), steps=50, device=device).cpu()
    model.cpu()
    return x.cpu()

def generate_weighted_samples(
        client_gen_models: Dict[str, Dict[int, Any]],
        client_sample_counts: Dict[str, Dict[int, int]],
        samples_per_class: int,
        model_kind: str,
        latent_dim: int,
        device: torch.device,
        img_shape: Tuple[int, int, int]
) -> Dict[int, torch.Tensor]:
    """
    Orchestrates weighted synthetic data generation across distributed clients.
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

        contributors = []
        for cname, models in client_gen_models.items():
            if cid in models:
                contributors.append((cname, models[cid], client_sample_counts[cname].get(cid, 0)))

        contributors.sort(key=lambda x: x[2], reverse=True)
        total_real = total_samples_per_class[cid]

        cls_samples = []
        remaining = target

        for i, (cname, model, count) in enumerate(contributors):
            if i == len(contributors) - 1: n = remaining
            else: n = int(target * (count / total_real))

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
        x = x.to(device)
        logits = model(x)

        y_true.append(y)
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
        reserved_test_ld: DataLoader
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
    synthetic_cache: Dict[int, torch.Tensor] = {}

    logger.info(f"[CLF-PHASE] Partition: {partition_mode}, Local Inference: {is_local}")

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
                        if getattr(args, "aggregation", "simple") == "weighted":
                            w_samps = generate_weighted_samples(client_gen_models, client_sample_counts, target,
                                                                args.model, args.latent_dim, device, img_shape)
                            s_imgs = w_samps.get(cid, torch.tensor([]))
                        else:
                            if cid < len(gen_models) and gen_models[cid] is not None:
                                m = gen_models[cid]
                                if args.model == "vae":
                                    s_imgs = synth_from_decoder(m, args.latent_dim, target, device)
                                else:
                                    s_imgs = synth_from_diffusion(m, target, device, img_shape)
                            else:
                                s_imgs = torch.tensor([])

                        if len(s_imgs) > 0:
                            # CACHE DATA: Only if we haven't cached this class yet or if we have a strategy to merge
                            # In local inference, different clients might generate the same class.
                            # For simplicity in evaluation, we cache the first generation or append.
                            # Here we simply overwrite or append for the global cache.
                            if cid not in synthetic_cache:
                                synthetic_cache[cid] = s_imgs.cpu()
                            else:
                                synthetic_cache[cid] = torch.cat([synthetic_cache[cid], s_imgs.cpu()])

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

                tr_ld = DataLoader(final_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
                val_ld = DataLoader(final_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

                if tracker: tracker.start_phase(f"client_{cname}_clf")
                clf, steps = train_classifier(SimpleCNN(chans, num_classes), tr_ld, val_ld, device, args.clf_epochs,
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
                synth_samples = generate_weighted_samples(client_gen_models, client_sample_counts, target, args.model,
                                                          args.latent_dim, device, img_shape)
            else:
                synth_samples = {}
                for cid in tqdm(range(num_classes), desc="Server Gen"):
                    if cid < len(gen_models) and gen_models[cid]:
                        if args.model == "vae":
                            synth_samples[cid] = synth_from_decoder(gen_models[cid], args.latent_dim, target, device)
                        else:
                            synth_samples[cid] = synth_from_diffusion(gen_models[cid], target, device, img_shape)
            if tracker: tracker.end_phase("server_generation")

            xs, ys = [], []
            for cid, imgs in synth_samples.items():
                if len(imgs) > 0:
                    # CACHE DATA
                    synthetic_cache[cid] = imgs.cpu()
                    xs.append(imgs)
                    ys.append(torch.full((len(imgs),), cid, dtype=torch.long))
                    synth_images_total += len(imgs)

            if xs:
                X, y = torch.cat(xs), torch.cat(ys)

                present_in_synth = sorted(list(set(y.tolist())))
                label_map = {old: new for new, old in enumerate(present_in_synth)}
                y_mapped = torch.tensor([label_map[v.item()] for v in y], dtype=torch.long)

                val_len = int(len(X) * 0.1)
                tr_len = len(X) - val_len

                raw_ds = TensorDataset(X, y_mapped)
                train_raw, val_raw = random_split(
                    raw_ds, [tr_len, val_len], generator=torch.Generator().manual_seed(42)
                )

                train_tf = build_transform(args.dataset, train=True, robustness=True)
                eval_tf = build_transform(args.dataset, train=False, robustness=False)

                train_ds = TransformSubset(train_raw, train_tf)
                val_ds = TransformSubset(val_raw, eval_tf)

                tr_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
                val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

                if tracker: tracker.start_phase("server_classifier")
                clf, steps = train_classifier(SimpleCNN(chans, len(present_in_synth)), tr_ld, val_ld, device,
                                              args.clf_epochs, hist, "server", tracker=tracker)
                if tracker: tracker.end_phase("server_classifier")

                classifier_steps += steps
                torch.save(clf.state_dict(), P.root / "models" / "classifiers" / "central.pt")
                single_clf = clf

                if tracker: tracker.start_phase("server_evaluation")
                all_imgs, all_lbls = [], []
                for x, lbl in reserved_test_ld:
                    all_imgs.append(x)
                    all_lbls.append(lbl)
                all_imgs = torch.cat(all_imgs)
                all_lbls = torch.cat(all_lbls)

                mask = torch.tensor([l.item() in label_map for l in all_lbls], dtype=torch.bool)
                if mask.sum() > 0:
                    filt_imgs = all_imgs[mask]
                    filt_lbls = torch.tensor([label_map[l.item()] for l in all_lbls[mask]], dtype=torch.long)
                    test_ld_mapped = DataLoader(TensorDataset(filt_imgs, filt_lbls), batch_size=args.batch_size)

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
                            if args.model == "vae":
                                im = synth_from_decoder(gm, args.latent_dim, n_synth, device)
                            else:
                                im = synth_from_diffusion(gm, n_synth, device, img_shape)

                            # CACHE DATA: Map external class ID to generated images
                            if od not in synthetic_cache:
                                synthetic_cache[od] = im.cpu()
                            # else: we rely on the first generation for evaluation to save time/space

                            s_xs.append(im);
                            s_ys.append(torch.full((len(im),), od, dtype=torch.long))

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

                tr_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
                val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

                if tracker: tracker.start_phase(f"client_{d:03d}_clf")
                clf, steps = train_classifier(SimpleCNN(chans, num_classes), tr_ld, val_ld, device, args.clf_epochs,
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
                if args.model == "vae":
                    im = synth_from_decoder(gm, args.latent_dim, n_synth, device)
                else:
                    im = synth_from_diffusion(gm, n_synth, device, img_shape)

                # CACHE DATA
                synthetic_cache[d] = im.cpu()

                xs.append(im);
                ys.append(torch.full((len(im),), d, dtype=torch.long))

            X, y = torch.cat(xs), torch.cat(ys)
            synth_images_total = len(X)
            if tracker: tracker.end_phase("server_generation")

            val_len = int(len(X) * 0.1)
            tr_len = len(X) - val_len

            raw_ds = TensorDataset(X, y)
            train_raw, val_raw = random_split(
                raw_ds, [tr_len, val_len], generator=torch.Generator().manual_seed(42)
            )

            train_tf = build_transform(args.dataset, train=True, robustness=True)
            eval_tf = build_transform(args.dataset, train=False, robustness=False)

            train_ds = TransformSubset(train_raw, train_tf)
            val_ds = TransformSubset(val_raw, eval_tf)

            tr_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
            val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

            if tracker: tracker.start_phase("server_classifier")
            clf, steps = train_classifier(SimpleCNN(chans, num_classes), tr_ld, val_ld, device, args.clf_epochs, hist,
                                          "server", tracker=tracker)
            if tracker: tracker.end_phase("server_classifier")

            classifier_steps += steps
            torch.save(clf.state_dict(), P.root / "models" / "classifiers" / "central.pt")
            single_clf = clf

            if tracker: tracker.start_phase("server_evaluation")
            y_true, y_pred, y_probs = evaluate_single_classifier(clf, reserved_test_ld, device)
            if tracker: tracker.end_phase("server_evaluation")

    # Pass the populated synthetic_cache back to the experiment runner
    return classifier_steps, synth_images_total, clf_start, y_true, y_pred, y_probs, trained_clfs, single_clf, synthetic_cache