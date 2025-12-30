"""
📊 Generative Model Evaluation Module
-------------------------------------

This module orchestrates the comprehensive evaluation of generative machine learning 
models. It synthesizes data from trained generators (VAEs or Diffusion models) and 
computes industry-standard quantitative metrics to assess image quality and diversity.

It includes advanced statistical analysis (PCA spectra, Histogram matching), 
VAE reconstruction metrics, and dataset export capabilities.

🧠 Purpose:
    To provide a robust, academic-grade evaluation pipeline that bridges the gap 
    between model training and analytical benchmarking.

🔧 Core Functionalities:
    • Synthesize evaluation samples (standard & weighted).
    • Compute metrics: FID, KID, Precision, Recall, IS.
    • Calculate VAE Reconstruction metrics (MSE, PSNR, SSIM, LPIPS).
    • Perform dimensionality reduction (t-SNE, PCA Spectra).
    • Generate distribution statistics (Gaussian Overlays/Histograms).
    • Export full datasets (Real/Synthetic) to disk.

📁 Dependencies:
    • torch, numpy, sklearn, torchvision
    • Internal modules: jobs.classifier_phase, metrics.evaluation, models.vae

Author: Andrea Moleri
File Location: src/jobs/evaluation_phase.py
Last Modified: 21/11/2025
"""

from __future__ import annotations
import logging
import json
import torch
import torch.nn.functional as F
import re
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.utils import save_image
import torchvision.transforms as T
from tqdm.auto import tqdm

from jobs.classifier_phase import synth_from_decoder, synth_from_diffusion, generate_weighted_samples
from jobs.baseline_runner import subset_to_tensor
from imports.data_augmentation import build_transform

from metrics.evaluation import (
    FeatureExtractor, fid_from_feats, kid_unbiased, precision_recall_knn,
    inception_score, save_pairwise_outputs, psnr_from_mse, ssim_simple
)
from utils import VGGPerceptualLoss  # Assumes VGGPerceptualLoss is in utils.py
from sklearn.manifold import TSNE
from models.vae import VAE, Decoder # Need VAE class for reconstruction

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions (Reintroduced from Monolithic)
# =============================================================================

class TransformSubset(torch.utils.data.Dataset):
    """
    Wraps a subset and applies a transform on the fly.
    Used here to convert raw PIL images from base_train_set to Tensors.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def _class_dirname(i: int, base_train_set) -> str:
    """Generates a directory name for a class, sanitizing the label if available."""
    label = None
    if hasattr(base_train_set, "classes") and base_train_set.classes:
        try:
            raw = str(base_train_set.classes[i])
            label = re.sub(r"[^a-zA-Z0-9_\-]+", "-", raw).strip("-_").lower()
        except Exception:
            label = None
    return f"class-{i:03d}" + (f"_{label}" if label else "")

def _save_tensor_dataset_by_class(
        root: Path,
        tensors_by_class: List[torch.Tensor],
        present_classes: List[int],
        base_train_set,
        prefix: str,
        batch_size: int = 256,
) -> None:
    """
    Save all images (range [-1,1]) to:
      root / datasets / <prefix> / class-000_<name> / img-000001.png
    """
    out_root = root / "datasets" / prefix
    out_root.mkdir(parents=True, exist_ok=True)

    total_imgs = sum(int(xs.size(0)) for xs in tensors_by_class if xs is not None and xs.numel() > 0)
    pbar = tqdm(total=total_imgs, desc=f"Saving {prefix} dataset", unit="img")

    for class_index, _class_id in enumerate(present_classes):
        if class_index >= len(tensors_by_class):
            continue
        xs = tensors_by_class[class_index]
        if xs is None or xs.numel() == 0:
            continue

        class_dir = out_root / _class_dirname(class_index, base_train_set)
        class_dir.mkdir(parents=True, exist_ok=True)

        n = xs.size(0)
        img_id = 1
        for idx in range(0, n, batch_size):
            xb = xs[idx: idx + batch_size]
            for k in range(xb.size(0)):
                fp = class_dir / f"img-{img_id:06d}.png"
                # Value range (-1, 1) automatically normalizes to [0, 1]
                save_image(xb[k], fp.as_posix(), normalize=True, value_range=(-1, 1))
                img_id += 1
            pbar.update(xb.size(0))

    pbar.close()

def _convert_dict_keys_to_int(d):
    """Convert all string keys in a nested dict to int keys where possible."""
    if not isinstance(d, dict):
        return d
    result = {}
    for k, v in d.items():
        try:
            new_key = int(k) if isinstance(k, str) and k.isdigit() else k
        except (ValueError, TypeError):
            new_key = k
        if isinstance(v, dict):
            result[new_key] = _convert_dict_keys_to_int(v)
        elif isinstance(v, list):
            result[new_key] = [_convert_dict_keys_to_int(item) for item in v]
        else:
            result[new_key] = v
    return result


# =============================================================================
# Main Evaluation Function
# =============================================================================

def run_evaluation(
        args, device, P, present_classes, reserved_test_imgs_list,
        gen_models, client_gen_models, client_sample_counts, img_shape,
        trained_clfs, single_clf, y_true, y_pred,
        train_subsets_dict, train_subsets, base_train_set, chans,
        pre_generated_data: Optional[Dict[int, torch.Tensor]] = None  # <--- NEW ARGUMENT
):
    """
    Executes the full evaluation suite for generative models.
    """
    logger.info("[Metrics] Computing FID, KID, Precision/Recall, IS, pairwise distances and t-SNE...")
    feat_extractor = FeatureExtractor(device)
    partition_mode = getattr(args, "partition", "silos")

    # -------------------------------------------------------------------------
    # Phase 1: Synthetic Data Generation (For Metrics)
    # -------------------------------------------------------------------------
    eval_synth_by_class = []

    for i, d in enumerate(present_classes):
        n_eval = args.eval_samples_per_class
        if args.samples_per_class > 0: n_eval = min(n_eval, args.samples_per_class)
        n_eval = max(1, n_eval)

        # NEW LOGIC: Check cache first
        synth_eval = None
        if pre_generated_data is not None and d in pre_generated_data:
            cached_data = pre_generated_data[d]
            if len(cached_data) >= n_eval:
                # We have enough data, slice it
                synth_eval = cached_data[:n_eval]
            else:
                # We have data, but not enough for the full metric evaluation count.
                # Use what we have, and generate the delta.
                # NOTE: For consistency with the user request ("same image used to train"),
                # we prioritize the cached data.
                needed = n_eval - len(cached_data)
                logger.info(f"[Metrics] Class {d}: Cached {len(cached_data)}, generating {needed} more for eval.")

                # ... Generation Logic for Delta ...
                if partition_mode in ["skew", "dirichlet"] and getattr(args, "aggregation", "simple") == "weighted":
                    w = generate_weighted_samples(client_gen_models, client_sample_counts, needed, args.model,
                                                  args.latent_dim, device, img_shape)
                    delta = w.get(d, torch.tensor([]))
                else:
                    if d < len(gen_models) and gen_models[d] is not None:
                        m = gen_models[d]
                        if hasattr(m, "decoder"): m = m.decoder
                        if args.model == "vae":
                            delta = synth_from_decoder(m, args.latent_dim, needed, device)
                        else:
                            delta = synth_from_diffusion(m, needed, device, img_shape)
                    else:
                        delta = torch.tensor([])

                synth_eval = torch.cat([cached_data, delta.cpu()])

        # If still None (not in cache), generate fully
        if synth_eval is None:
            if partition_mode in ["skew", "dirichlet"] and getattr(args, "aggregation", "simple") == "weighted":
                can_gen = d in [cid for models in client_gen_models.values() for cid in models.keys()]
                if can_gen:
                    w = generate_weighted_samples(client_gen_models, client_sample_counts, n_eval, args.model,
                                                  args.latent_dim, device, img_shape)
                    synth_eval = w.get(d, torch.tensor([]))
                else:
                    synth_eval = torch.tensor([])
            else:
                if d < len(gen_models) and gen_models[d] is not None:
                    m = gen_models[d]
                    if hasattr(m, "decoder"): m = m.decoder
                    if args.model == "vae":
                        synth_eval = synth_from_decoder(m, args.latent_dim, n_eval, device)
                    else:
                        synth_eval = synth_from_diffusion(m, n_eval, device, img_shape)
                else:
                    synth_eval = torch.tensor([])

        eval_synth_by_class.append(synth_eval)

    # -------------------------------------------------------------------------
    # Phase 2: Feature Extraction and Metric Calculation
    # -------------------------------------------------------------------------
    fid, kid, prec, rec, recog = {}, {}, {}, {}, {}
    min_nn, min_rr, div = {}, {}, {}

    def _predict(imgs, exp_lbl):
        ld = DataLoader(TensorDataset(imgs, exp_lbl), batch_size=args.batch_size)
        from jobs.classifier_phase import evaluate_single_classifier, ensemble_preds_poexp

        # FIX: Unpack 3 values (y_true, y_pred, y_probs) but ignore y_probs
        if trained_clfs:
            _, yp, _ = ensemble_preds_poexp(trained_clfs, ld, device)
        elif single_clf:
            _, yp, _ = evaluate_single_classifier(single_clf, ld, device)
        else:
            yp = np.full(len(exp_lbl), -1)
        return yp

    f_real_map, f_fake_map = {}, {}
    all_probs = []

    for i, d in enumerate(present_classes):
        real_c = reserved_test_imgs_list[i]
        fake_c = eval_synth_by_class[i]

        if len(real_c) == 0 or len(fake_c) == 0: continue

        fr, _ = feat_extractor.features_and_logits(real_c)
        ff, pf = feat_extractor.features_and_logits(fake_c)
        f_real_map[d] = fr
        f_fake_map[d] = ff
        if pf is not None: all_probs.append(pf)

        fid[d] = fid_from_feats(fr, ff)
        kid[d] = kid_unbiased(fr, ff)
        p, r = precision_recall_knn(fr, ff, k=args.pr_knn_k)
        prec[d] = p; rec[d] = r

        expected = torch.full((fake_c.size(0),), d, dtype=torch.long)
        yp = _predict(fake_c, expected)
        recog[d] = float((yp == expected.numpy()).mean())

        save_pairwise_outputs(P.root, d, ff, fr, topk=5)

        # Calculate simple diversity metrics based on feature distance
        D_gr = np.sqrt(np.maximum(0.0, np.sum(ff ** 2, axis=1, keepdims=True) + np.sum(fr ** 2, axis=1, keepdims=True).T - 2.0 * (ff @ fr.T)))
        min_nn[d] = np.min(D_gr, axis=1).astype(np.float32).tolist()

        D_rr = np.sqrt(np.maximum(0.0, np.sum(fr ** 2, axis=1, keepdims=True) + np.sum(fr ** 2, axis=1, keepdims=True).T - 2.0 * (fr @ fr.T)))
        np.fill_diagonal(D_rr, np.inf)
        min_rr[d] = np.min(D_rr, axis=1).astype(np.float32).tolist()

        D_ff = np.sqrt(np.maximum(0.0, np.sum(ff ** 2, axis=1, keepdims=True) + np.sum(ff ** 2, axis=1, keepdims=True).T - 2.0 * (ff @ ff.T)))
        np.fill_diagonal(D_ff, np.inf)
        div[d] = float(np.min(D_ff, axis=1).mean())

    IS = inception_score(np.concatenate(all_probs, axis=0)) if all_probs else None

    # -------------------------------------------------------------------------
    # Phase 3: Copy Detection & SOTA
    # -------------------------------------------------------------------------
    copy_rate_per_class = {}
    copy_thresholds = {}
    pct = float(args.copy_threshold_percentile)
    for d in present_classes:
        if d not in min_rr:
            copy_rate_per_class[d] = float("nan")
            copy_thresholds[d] = float("nan")
            continue
        rr = np.array(min_rr[d], dtype=np.float32)
        if rr.size == 0:
            copy_rate_per_class[d] = float("nan")
            copy_thresholds[d] = float("nan")
            continue
        tau = float(np.percentile(rr, pct))
        copy_thresholds[d] = tau
        if d in min_nn:
            gr = np.array(min_nn[d], dtype=np.float32)
            copy_rate_per_class[d] = float(np.mean(gr <= tau)) if gr.size > 0 else float("nan")

    sota = None
    if args.sota_json and Path(args.sota_json).exists():
        try:
            with open(args.sota_json, "r") as f:
                full = json.load(f)
            sota = full.get(args.dataset.lower(), None)
        except Exception as e:
            logger.warning(f"[Metrics] Could not read --sota-json: {e}")

    # -------------------------------------------------------------------------
    # Phase 4: T-SNE
    # -------------------------------------------------------------------------
    tsne2_payload = None
    tsne3_payload = None
    try:
        X_tsne, y_tsne, dom_tsne = [], [], []
        for d in list(present_classes)[:10]:
            if d not in f_real_map:
                continue
            fr, ff = f_real_map[d], f_fake_map[d]
            nr, nf = min(200, len(fr)), min(200, len(ff))
            if nr > 0:
                X_tsne.append(fr[:nr])
                y_tsne.extend([d] * nr)
                dom_tsne.extend(["Real"] * nr)
            if nf > 0:
                X_tsne.append(ff[:nf])
                y_tsne.extend([d] * nf)
                dom_tsne.extend(["Synthetic"] * nf)

        if X_tsne:
            X_ts = np.vstack(X_tsne).astype(np.float32)

            # t-SNE 2D (come prima)
            emb2 = TSNE(
                n_components=2,
                init="pca",
                learning_rate="auto",
            ).fit_transform(X_ts)
            tsne2_payload = {
                "x": emb2.tolist(),
                "labels": y_tsne,
                "domain": dom_tsne,
            }

            # NUOVO: t-SNE 3D
            emb3 = TSNE(
                n_components=3,
                init="pca",
                learning_rate="auto",
            ).fit_transform(X_ts)
            tsne3_payload = {
                "x": emb3.tolist(),   # lista di [x, y, z]
                "labels": y_tsne,
                "domain": dom_tsne,
            }

            tsne_dir = P.root / "artifacts" / "tsne"
            tsne_dir.mkdir(parents=True, exist_ok=True)
            (tsne_dir / "tsne2.json").write_text(json.dumps(tsne2_payload))
            (tsne_dir / "tsne3.json").write_text(json.dumps(tsne3_payload))
    except Exception as e:
        logger.warning(f"t-SNE failed: {e}")


    # -------------------------------------------------------------------------
    # Phase 5: PCA Spectra (SVD) [REINTRODUCED]
    # -------------------------------------------------------------------------
    pca_spectra = {"real": {}, "gen": {}}
    for c in present_classes:
        for kind, bank in [("real", f_real_map), ("gen", f_fake_map)]:
            if c not in bank:
                pca_spectra[kind][c] = []
                continue
            X = bank[c]
            if X.shape[0] < 2:
                pca_spectra[kind][c] = []
                continue
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            var = (S ** 2) / max(1e-9, (X.shape[0] - 1))
            evr = var / max(1e-12, var.sum())
            pca_spectra[kind][c] = evr[: min(10, evr.size)].tolist()

    # -------------------------------------------------------------------------
    # Phase 6: Full Dataset Reconstruction & Histograms [REINTRODUCED]
    # -------------------------------------------------------------------------
    gaussian_overlay_per_class = {}
    real_full_by_class = []

    # --- Reconstruct Transform for Real Data ---
    # We need to convert PIL images from base_train_set to Tensors [-1, 1]
    # matching the model's expected input.
    base_tfm = build_transform(args.dataset, train=False, robustness=False)

    ops = []
    # Input Size Override
    if getattr(args, "input_size", 0) > 0:
        sz = int(args.input_size)
        ops.append(T.Resize((sz, sz), antialias=True))

    # Grayscale Override
    if bool(getattr(args, "grayscale", False)) and chans == 3:
        ops.append(T.Grayscale(num_output_channels=3))

    ops.append(base_tfm)
    eval_tf = T.Compose(ops)

    for i, d in enumerate(present_classes):
        t0 = time.perf_counter()
        if partition_mode in ["skew", "dirichlet"]:
            real_train_imgs_list = []
            for client_subsets in train_subsets_dict.values():
                if d in client_subsets:
                    # FIX: Wrap PIL subset with TransformSubset
                    subset_clean = TransformSubset(client_subsets[d], eval_tf)
                    imgs = subset_to_tensor(subset_clean)
                    real_train_imgs_list.append(imgs)
            real_train_imgs = torch.cat(real_train_imgs_list) if real_train_imgs_list else torch.tensor([])
        else:
            # FIX: Wrap PIL subset with TransformSubset
            subset_clean = TransformSubset(train_subsets[i], eval_tf)
            real_train_imgs = subset_to_tensor(subset_clean)

        real_all = torch.cat([real_train_imgs, reserved_test_imgs_list[i]], dim=0) if len(real_train_imgs) > 0 else reserved_test_imgs_list[i]
        real_full_by_class.append(real_all)
        logger.info(f"[TIMING] real_full_by_class class={d}: {time.perf_counter() - t0:.2f}s  (n={len(real_all)})")

    synth_full_by_class = []
    for i, d in enumerate(present_classes):
        if args.samples_per_class > 0:
            target_count = args.samples_per_class
        else:
            target_count = real_full_by_class[i].size(0)

        if target_count == 0:
            synth_full_by_class.append(torch.tensor([]))
            continue

        # NEW LOGIC: Use Cache for Export
        # NEW LOGIC: Use Cache for Export
        xs = None
        used_cache = False  # <--- Flag to track cache usage

        if pre_generated_data is not None and d in pre_generated_data:
            cached_data = pre_generated_data[d]
            if len(cached_data) >= target_count:
                xs = cached_data[:target_count]
                used_cache = True  # <--- Set flag
                # logger.info(f"[DATASET EXPORT] Class {d}: Using {len(xs)} cached training samples.") <--- Removed redundant log
            else:
                # Not enough in cache
                needed = target_count - len(cached_data)

                # ... (Generation logic for delta) ...
                if partition_mode in ["skew", "dirichlet"] and getattr(args, "aggregation", "simple") == "weighted":
                    w = generate_weighted_samples(client_gen_models, client_sample_counts, needed, args.model,
                                                  args.latent_dim, device, img_shape)
                    delta = w.get(d, torch.tensor([]))
                else:
                    if d < len(gen_models) and gen_models[d] is not None:
                        m = gen_models[d]
                        if hasattr(m, "decoder"): m = m.decoder
                        if args.model == "vae":
                            delta = synth_from_decoder(m, args.latent_dim, needed, device)
                        else:
                            delta = synth_from_diffusion(m, needed, device, img_shape)
                    else:
                        delta = torch.tensor([])
                xs = torch.cat([cached_data, delta.cpu()])
                # Note: Mixed source, so we can consider it partially generated.
                # We won't set used_cache=True to keep the "generated" log for transparency on the extra compute.

        if xs is None:
            # Full generation fallback (Logic remains unchanged)
            if partition_mode in ["skew", "dirichlet"] and getattr(args, "aggregation", "simple") == "weighted":
                if d in [cid for models in client_gen_models.values() for cid in models.keys()]:
                    w = generate_weighted_samples(client_gen_models, client_sample_counts, target_count, args.model,
                                                  args.latent_dim, device, img_shape)
                    xs = w.get(d, torch.tensor([]))
                else:
                    xs = torch.tensor([])
            else:
                if d < len(gen_models) and gen_models[d] is not None:
                    m = gen_models[d]
                    if hasattr(m, "decoder"): m = m.decoder
                    if args.model == "vae":
                        xs = synth_from_decoder(m, args.latent_dim, target_count, device)
                    else:
                        xs = synth_from_diffusion(m, target_count, device, img_shape)
                else:
                    xs = torch.tensor([])

        synth_full_by_class.append(xs)

        # --- FIXED LOGGING MESSAGE ---
        if used_cache:
            # Was "Locally saved", now "Retrieved" to avoid confusion with disk write
            logger.info(f"[DATASET EXPORT] Class {d}: Retrieved {len(xs)} cached samples (target: {target_count})")
        else:
            logger.info(f"[DATASET EXPORT] Class {d}: Generated {len(xs)} synthetic samples (target: {target_count})")

    num_bins = 20
    bins_edges = np.linspace(-1.0, 1.0, num_bins + 1)

    for i, d in enumerate(present_classes):
        real_imgs_cls = real_full_by_class[i]
        synth_imgs_cls = synth_full_by_class[i]

        if len(real_imgs_cls) == 0 or len(synth_imgs_cls) == 0:
            continue

        real_feat_vals = real_imgs_cls.mean(dim=(1, 2, 3)).cpu().numpy().astype(np.float64)
        synth_feat_vals = synth_imgs_cls.mean(dim=(1, 2, 3)).cpu().numpy().astype(np.float64)

        real_mean = float(np.mean(real_feat_vals)) if real_feat_vals.size else float("nan")
        real_var = float(np.var(real_feat_vals)) if real_feat_vals.size else float("nan")
        real_n = int(real_feat_vals.shape[0])

        synth_mean = float(np.mean(synth_feat_vals)) if synth_feat_vals.size else float("nan")
        synth_var = float(np.var(synth_feat_vals)) if synth_feat_vals.size else float("nan")
        synth_n = int(synth_feat_vals.shape[0])

        real_counts, _ = np.histogram(real_feat_vals, bins=bins_edges)
        synth_counts, _ = np.histogram(synth_feat_vals, bins=bins_edges)

        gaussian_overlay_per_class[d] = {
            "bins": bins_edges.tolist(),
            "real": {"counts": real_counts.tolist(), "mean": real_mean, "var": real_var, "count": real_n},
            "synth": {"counts": synth_counts.tolist(), "mean": synth_mean, "var": synth_var, "count": synth_n},
        }

    # -------------------------------------------------------------------------
    # Phase 7: Dataset Export [REINTRODUCED]
    # -------------------------------------------------------------------------
    if getattr(args, "save_datasets", False):
        try:
            _save_tensor_dataset_by_class(P.root, real_full_by_class, present_classes, base_train_set, prefix="real")
            _save_tensor_dataset_by_class(P.root, synth_full_by_class, present_classes, base_train_set, prefix="synthetic")
            logger.info(f"[DATASET EXPORT] Datasets saved under: {P.root / 'datasets'}")
            total_synthetic = sum(int(xs.size(0)) for xs in synth_full_by_class if xs is not None and xs.numel() > 0)
            logger.info(f"[DATASET EXPORT] Total synthetic images saved: {total_synthetic}")
        except Exception as e:
            logger.warning(f"[DATASET EXPORT] Full export failed: {e}")
    else:
        logger.info("[DATASET EXPORT] Skipped (use --save-datasets to enable).")

    # -------------------------------------------------------------------------
    # Phase 8: VAE Reconstruction Metrics [REINTRODUCED]
    # -------------------------------------------------------------------------
    recon_metrics = None
    recon_dists = None

    if args.model == "vae":
        logger.info("[Metrics] Evaluating reconstructions (MSE/PSNR/SSIM/Perceptual VGG) on reserved test...")
        perc_loss = VGGPerceptualLoss().to(device) if (chans == 3 and not bool(args.grayscale)) else None

        vae_recon_mse, vae_recon_psnr, vae_recon_ssim, vae_recon_vgg = [], [], [], []

        with torch.no_grad():
            for i, d in enumerate(present_classes):
                real_cls = reserved_test_imgs_list[i]
                if len(real_cls) == 0: continue

                # Attempt to load the full VAE model from disk (Encoder + Decoder)
                # Since gen_models might only hold Decoders, we must reload the checkpoint.
                vae_tmp = None
                try:
                    if partition_mode in ["skew", "dirichlet"]:
                        # For skew, try to find a client that trained on this class
                        # We pick the first one available.
                        ckpt_path = None
                        for client_name, models in client_gen_models.items():
                            if d in models:
                                ckpt_path = P.root / "models" / "generators" / f"client_{client_name}_class_{d:03d}.pt"
                                break
                    else:
                        ckpt_path = P.root / "models" / "generators" / f"class-{d:03d}.pt"

                    if ckpt_path and ckpt_path.exists():
                        vae_tmp = VAE(chans, args.latent_dim)
                        vae_tmp.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                        vae_tmp.eval().to(device)
                except Exception as e:
                    logger.warning(f"[Metrics] Failed to load VAE for reconstruction class {d}: {e}")
                    vae_tmp = None

                if vae_tmp is not None:
                    for j in range(0, real_cls.size(0), args.batch_size):
                        xb = real_cls[j: j + args.batch_size].to(device)
                        rec, mu, logvar = vae_tmp(xb)

                        mse_arr = F.mse_loss(rec, xb, reduction="none").flatten(1).mean(1).detach().cpu().numpy()
                        vae_recon_mse.append(mse_arr)

                        if perc_loss is not None:
                            vggd = perc_loss(rec, xb).detach().cpu().numpy()
                        else:
                            vggd = F.l1_loss(rec, xb, reduction="none").flatten(1).mean(1).detach().cpu().numpy()
                        vae_recon_vgg.append(vggd)

                        psnr_arr = psnr_from_mse(mse_arr)
                        vae_recon_psnr.append(psnr_arr)

                        ssim_val = ssim_simple(rec, xb).item()
                        vae_recon_ssim.append(np.full(xb.size(0), ssim_val))

                    vae_tmp.cpu() # unload

        if vae_recon_mse:
            mse_all = np.concatenate(vae_recon_mse)
            psnr_all = np.concatenate(vae_recon_psnr)
            ssim_all = np.concatenate(vae_recon_ssim)
            vgg_all = np.concatenate(vae_recon_vgg)

            recon_metrics = {
                "mse_mean": float(np.mean(mse_all)),
                "psnr_mean": float(np.mean(psnr_all)),
                "ssim_mean": float(np.mean(ssim_all)),
                "perceptual_vgg_l1_mean": float(np.mean(vgg_all)),
            }

            def _downsample(arr: np.ndarray, maxn: int = 5000) -> List[float]:
                if arr.shape[0] <= maxn: return arr.tolist()
                idx_ds = np.random.choice(arr.shape[0], maxn, replace=False)
                return arr[idx_ds].tolist()

            recon_dists = {
                "mse": _downsample(mse_all),
                "psnr": _downsample(psnr_all),
                "ssim": _downsample(ssim_all),
                "vgg_l1": _downsample(vgg_all),
            }

    # -------------------------------------------------------------------------
    # Phase 9: Final Assembly
    # -------------------------------------------------------------------------
    gen_metrics = {
        "fid_per_class": _convert_dict_keys_to_int(fid),
        "kid_per_class": _convert_dict_keys_to_int(kid),
        "precision_per_class": _convert_dict_keys_to_int(prec),
        "recall_per_class": _convert_dict_keys_to_int(rec),
        "recognizability_per_class": _convert_dict_keys_to_int(recog),
        "diversity_nn_per_class": _convert_dict_keys_to_int(div),
        "inception_score": IS,
        "min_nn_dist_gen2real_per_class": _convert_dict_keys_to_int(min_nn),
        "min_rr_dist_per_class": _convert_dict_keys_to_int(min_rr),
        "copy_rate_per_class": _convert_dict_keys_to_int(copy_rate_per_class),
        "copy_thresholds": _convert_dict_keys_to_int(copy_thresholds),
        "pca_spectra": _convert_dict_keys_to_int(pca_spectra),
        "gaussian_overlay_per_class": _convert_dict_keys_to_int(gaussian_overlay_per_class),
        "sota": sota
    }

    if recon_metrics is not None:
        gen_metrics["reconstruction"] = recon_metrics
    if recon_dists is not None:
        gen_metrics["reconstruction_dists"] = recon_dists
    if tsne2_payload:
        gen_metrics["tsne2"] = tsne2_payload
    if tsne3_payload:
        gen_metrics["tsne3"] = tsne3_payload

    # Persist metrics to disk.
    with open(P.root / "metrics" / "generative.json", "w") as f:
        json.dump(gen_metrics, f, indent=2)

    return gen_metrics