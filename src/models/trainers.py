"""
🤖 Model Training Pipelines
---------------------------

This module encapsulates the training and evaluation loops for various deep learning
architectures, including Diffusion Transformers (DiT), Variational Autoencoders (VAE),
and standard Convolutional Neural Networks (CNN). It provides robust implementations
for both standard Stochastic Gradient Descent (SGD) and Differentially Private (DP-SGD)
training paradigms.

🧠 Purpose:
    To provide a unified, modular, and trackable training interface for federated
    learning experiments. It abstracts the complexity of manual microbatching,
    gradient clipping, and noise injection required for differential privacy.

🔧 Core Functionalities:
    • Train Diffusion Transformers with Rectified Flow objectives
    • Train Variational Autoencoders with reconstruction and KL divergence losses
    • Train supervised classifiers with cross-entropy loss
    • Implement Differentially Private SGD (DP-SGD) with manual gradient accumulation
    • Track computational costs (FLOPs) and training metrics via external trackers

🎯 Intended Use:
    • Federated learning clients performing local updates
    • Centralized training baselines for academic benchmarking
    • Performance profiling of privacy-preserving algorithms

📁 Dependencies:
    • torch
    • logs (custom logging utility)
    • utils (privacy utilities)
    • models (architecture definitions)
    • metrics (cost tracking)

📝 Notes:
    The module automatically enables TF32 precision on supported NVIDIA architectures
    (Ampere/Hopper) to enhance training throughput without significant precision loss.

Author: Andrea Moleri
File Location: src/models/trainers.py
Last Modified: 06/12/2025
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
import torch.optim as optim

from logs.logger import get_logger
from logs import messages as logmsg
from utils import _dp_add_noise_

from models.vae import VAE
from models.diffusion import DiT, rectified_flow_loss

from metrics.costs import ExperimentCostTracker

from pathlib import Path

# Enable TF32 (TensorFloat-32) on Ampere/Hopper GPUs to improve matrix multiplication
# and convolution performance while maintaining sufficient precision for deep learning.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = get_logger(__name__)


def train_diffusion(
        model: DiT,
        loader: DataLoader,
        device: torch.device,
        epochs: int,
        hist: Dict,
        cid: int | str,  # Updated type hint to allow 'server'
        dp: bool = False,
        dp_clip: float = 1.0,
        dp_noise_mult: float = 1.1,
        dp_microbatch: int = 8,
        tracker: Optional[ExperimentCostTracker] = None,
        checkpoint_every: int = 0,
        checkpoint_dir: Optional[Path] = None,
        lr: float = 1e-3,  # [FIX] Added missing lr parameter
) -> Tuple[DiT, int]:
    """
    Train a Diffusion Transformer using the Rectified Flow objective.

    This function handles both standard training and Differentially Private (DP) training.
    In the DP regime, it performs manual microbatching to manage memory usage while
    computing per-sample gradients, applies global norm clipping, and injects Gaussian noise.

    Args:
        model (DiT): The Diffusion Transformer model to be trained.
        loader (DataLoader): The iterable data loader containing training samples.
        device (torch.device): The computational device (CPU or CUDA) to use.
        epochs (int): The number of complete passes through the dataset.
        hist (Dict): A dictionary to store training history and loss metrics.
        cid (int): The client identifier, used for logging and history keys.
        dp (bool, optional): If True, enables Differential Privacy mechanisms (DP-SGD).
            Defaults to False.
        dp_clip (float, optional): The maximum L2 norm threshold for gradient clipping
            in DP mode. Defaults to 1.0.
        dp_noise_mult (float, optional): The noise multiplier for the Gaussian mechanism
            relative to the clipping threshold. Defaults to 1.1.
        dp_microbatch (int, optional): The number of samples processed per microbatch
            step in DP mode to manage memory. Defaults to 8.
        tracker (Optional[ExperimentCostTracker], optional): An object to track
            computational costs (FLOPs) and training time. Defaults to None.
        checkpoint_every (int): Epoch interval for saving checkpoints.
        checkpoint_dir (Path): Directory to save checkpoints.
        lr (float): Learning rate for the optimizer.

    Returns:
        Tuple[DiT, int]: A tuple containing the trained model and the total number
        of optimizer steps performed.
    """
    model.to(device)

    # [FIX] Now 'lr' refers to the function argument
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.99))

    hist.setdefault("vae_loss", {})
    hist["vae_loss"][cid] = []  # Maintain key consistency with VAE logging structure

    clip_C = float(dp_clip)
    microbatch = max(1, int(dp_microbatch))
    noise_std = float(dp_noise_mult) * clip_C
    step_count = 0

    # Initialize GradScaler only for the non-DP path to enable Automatic Mixed Precision (AMP).
    # DP-SGD logic typically handles precision manually or requires specific scaler handling.
    scaler = amp.GradScaler(
        "cuda",
        enabled=(not dp) and torch.cuda.is_available()
    )

    # Extract a representative mini-batch for FLOPs profiling.
    # This ensures the tracker knows the input tensor shape for cost estimation.
    rep_batch = None
    try:
        for xb, _ in loader:
            rep_batch = xb.to(device)[: min(2, xb.size(0))]
            break
    except Exception:
        rep_batch = None

    if tracker is not None and rep_batch is not None:
        tracker.register_model(
            model,
            rep_batch,
            backward_factor=2.0,
            loss_extra_fwd=0.0,
        )
        tracker.start_phase(f"train_c{cid}")

    for ep in range(epochs):
        model.train()
        total_loss_sum = 0.0

        for x, _y in loader:
            x = x.to(device, non_blocking=True)

            if not dp:
                # ---------- Standard (Non-Private) Training Path ----------
                # Uses standard backpropagation with Automatic Mixed Precision (AMP)
                # for memory efficiency and speed on compatible hardware.
                opt.zero_grad(set_to_none=True)

                amp_dtype = (
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                )
                with amp.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu",
                        dtype=amp_dtype,
                ):
                    loss = rectified_flow_loss(model, x)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    opt.step()

                total_loss_sum += loss.item() * x.size(0)
                step_count += 1

                if tracker is not None:
                    tracker.count_train_step(1)

            else:
                # ---------- Differentially Private (DP-SGD) Path ----------
                # Implements manual microbatching to calculate per-sample gradients,
                # clip them to a fixed norm, and aggregate them before adding noise.
                bs = x.size(0)
                opt.zero_grad(set_to_none=True)

                # Initialize gradient accumulators for manual aggregation.
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_accum = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                microbatch_count = 0
                amp_dtype = (
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                )

                # Process the batch in smaller microbatches.
                for mb_start in range(0, bs, microbatch):
                    x_mb = x[mb_start: mb_start + microbatch]

                    with amp.autocast(
                            device_type="cuda" if torch.cuda.is_available() else "cpu",
                            dtype=amp_dtype,
                    ):
                        # Normalize by microbatch size so gradients represent the mean.
                        loss_mb = rectified_flow_loss(model, x_mb) / x_mb.size(0)

                    loss_mb.backward()

                    # Calculate the global L2 norm of gradients for this microbatch.
                    total_norm_sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm_sq += float(
                                p.grad.data.norm(2).pow(2)
                            )
                    total_norm = math.sqrt(total_norm_sq)

                    # Determine the clipping coefficient (at most 1.0).
                    clip_coef = min(1.0, clip_C / (total_norm + 1e-6))

                    # Apply clipping and accumulate the result into grad_accum.
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        p.grad.data.mul_(clip_coef)
                        p.grad_accum.add_(p.grad.data)
                        p.grad.detach_()
                        p.grad.zero_()

                    microbatch_count += 1
                    total_loss_sum += loss_mb.item() * x_mb.size(0)

                # Add Gaussian noise, average over microbatches, and perform the optimizer step.
                for p in model.parameters():
                    if not hasattr(p, "grad_accum"):
                        continue
                    grad = p.grad_accum
                    _dp_add_noise_(grad, noise_std)
                    grad.div_(microbatch_count)
                    p.grad = grad
                    del p.grad_accum

                # Additional safety clipping for numerical stability (post-noise).
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()

                step_count += 1
                if tracker is not None:
                    tracker.count_train_step(1)

        avg_loss = total_loss_sum / len(loader.dataset)
        hist["vae_loss"][cid].append(avg_loss)
        tag = "DP" if dp else "STD"
        logger.info(
            logmsg.VAE_EPOCH.format(
                cid=cid,
                epoch=ep + 1,
                epochs=epochs,
                tag=tag,
                loss=avg_loss,
            )
        )

        # === CHECKPOINT LOGIC ===
        if checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0 and checkpoint_dir is not None:
            # EDM2-style naming convention adapted for Epochs
            # Format: checkpoint-client-{cid}-epoch-{epoch}.pt
            ckpt_name = f"checkpoint-client-{cid}-epoch-{ep + 1:04d}.pt"
            ckpt_path = checkpoint_dir / ckpt_name

            try:
                # Save state dictionary (compatible with DiT wrapper)
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"[CHECKPOINT] Saved model to {ckpt_path}")
            except Exception as e:
                logger.warning(f"[CHECKPOINT] Failed to save checkpoint: {e}")

    model.cpu()
    if tracker is not None:
        tracker.end_phase(f"train_c{cid}")

    return model, step_count


def train_vae(
        model: VAE,
        loader: DataLoader,
        device: torch.device,
        epochs: int,
        hist: Dict,
        cid: int,
        perceptual: nn.Module | None,
        dp: bool = False,
        dp_clip: float = 1.0,
        dp_noise_mult: float = 1.1,
        dp_microbatch: int = 8,
        tracker: Optional[ExperimentCostTracker] = None,
) -> Tuple[VAE, int]:
    """
    Train a Variational Autoencoder (VAE) with optional Differential Privacy.

    This function minimizes the Evidence Lower Bound (ELBO), consisting of a
    reconstruction loss (L1 and optional perceptual loss) and the Kullback-Leibler
    divergence (KLD).

    Args:
        model (VAE): The VAE model instance.
        loader (DataLoader): The training data loader returning (noisy, clean) pairs.
        device (torch.device): The execution device.
        epochs (int): Number of training epochs.
        hist (Dict): History dictionary for logging loss metrics.
        cid (int): Client identifier.
        perceptual (nn.Module | None): Optional neural network module for computing
            perceptual similarity loss (e.g., LPIPS).
        dp (bool, optional): Enable Differential Privacy training. Defaults to False.
        dp_clip (float, optional): Gradient clipping threshold for DP. Defaults to 1.0.
        dp_noise_mult (float, optional): Noise multiplier for DP. Defaults to 1.1.
        dp_microbatch (int, optional): Microbatch size for DP. Defaults to 8.
        tracker (Optional[ExperimentCostTracker], optional): FLOPs and time tracker.

    Returns:
        Tuple[VAE, int]: The trained model and total optimizer steps.
    """
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    hist.setdefault("vae_loss", {})
    hist["vae_loss"][cid] = []

    clip_C = float(dp_clip)
    microbatch = max(1, int(dp_microbatch))
    noise_std = float(dp_noise_mult) * clip_C
    step_count = 0

    # Profile the model using a representative batch for FLOPs tracking.
    rep_batch = None
    try:
        for noisy, _ in loader:
            rep_batch = noisy.to(device)[: min(2, noisy.size(0))]
            break
    except Exception:
        rep_batch = None

    if tracker is not None and rep_batch is not None:
        tracker.register_model(
            model,
            rep_batch,
            backward_factor=2.0,
            loss_extra_fwd=0.0,
        )
        tracker.start_phase(f"train_c{cid}")

    for ep in range(epochs):
        total = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)

            if not dp:
                # ---------- Standard Training Path ----------
                opt.zero_grad()
                recon, mu, logvar = model(noisy)

                # Reconstruction term (L1)
                rec = F.l1_loss(recon, clean, reduction="sum")
                if perceptual is not None:
                    # Add perceptual loss if a metric is provided
                    rec = rec + perceptual(recon, clean) * noisy.size(0)

                # Kullback-Leibler Divergence term (Analytical)
                kld = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )
                loss = rec + kld

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()

                total += loss.item()
                step_count += 1
                if tracker is not None:
                    tracker.count_train_step(1)

            else:
                # ---------- DP-SGD Path with Microbatching ----------
                batch_size = noisy.size(0)
                opt.zero_grad()
                for p in model.parameters():
                    p.grad = torch.zeros_like(p)

                microbatch_count = 0
                for mb_start in range(0, batch_size, microbatch):
                    mb_end = mb_start + microbatch
                    n_mb = noisy[mb_start:mb_end]
                    c_mb = clean[mb_start:mb_end]

                    recon, mu, logvar = model(n_mb)

                    # Compute losses for the microbatch
                    rec = F.l1_loss(recon, c_mb, reduction="sum")
                    if perceptual is not None:
                        rec = rec + perceptual(recon, c_mb) * n_mb.size(0)

                    kld = -0.5 * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp()
                    )

                    # Average loss per sample within the microbatch
                    loss_mb = (rec + kld) / n_mb.size(0)
                    loss_mb.backward()

                    # Per-microbatch gradient clipping
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        total_norm += p.grad.data.norm(2).pow(2)
                    total_norm = math.sqrt(total_norm)
                    clip_coef = clip_C / (total_norm + 1e-6)
                    clip_coef = min(1.0, clip_coef)

                    # Accumulate clipped gradients
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        p.grad.data.mul_(clip_coef)
                        p.grad_accum = (
                            p.grad_accum + p.grad.data
                            if hasattr(p, "grad_accum")
                            else p.grad.data.clone()
                        )
                        p.grad.detach_()
                        p.grad.zero_()

                    microbatch_count += 1
                    total += (rec + kld).item()

                # Apply noise and finalize the optimization step
                for p in model.parameters():
                    if not hasattr(p, "grad_accum"):
                        continue
                    grad = p.grad_accum
                    _dp_add_noise_(grad, noise_std)
                    grad.div_(microbatch_count)
                    p.grad = grad
                    del p.grad_accum

                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()

                step_count += 1
                if tracker is not None:
                    tracker.count_train_step(1)

        avg = total / len(loader.dataset)
        hist["vae_loss"][cid].append(avg)
        tag = "DP" if dp else "STD"
        logger.info(
            logmsg.VAE_EPOCH.format(
                cid=cid,
                epoch=ep + 1,
                epochs=epochs,
                tag=tag,
                loss=avg,
            )
        )

    model.cpu()
    if tracker is not None:
        tracker.end_phase(f"train_c{cid}")

    return model, step_count


def train_classifier(
        model: nn.Module,
        train_ld: DataLoader,
        val_ld: DataLoader,
        device: torch.device,
        epochs: int,
        hist: Dict,
        cid: int | str,
        tracker: Optional[ExperimentCostTracker] = None,
) -> Tuple[nn.Module, int]:
    """
    Train a standard classifier model (e.g., SimpleCNN) according to strict specifications:
    - Optimizer: SGD
    - LR: 0.1 (decayed via Cosine Annealing to 0)
    - Momentum: 0.9
    - Weight Decay: 1e-4

    Args:
        model (nn.Module): The classifier model.
        train_ld (DataLoader): DataLoader for training data.
        val_ld (DataLoader): DataLoader for validation data.
        device (torch.device): Execution device.
        epochs (int): Number of training epochs (Spec requires 300).
        hist (Dict): Dictionary to store accuracy and loss history.
        cid (int | str): Client identifier.
        tracker (Optional[ExperimentCostTracker], optional): FLOPs and time tracker.

    Returns:
        Tuple[nn.Module, int]: The trained model and the number of training steps.
    """
    model.to(device)

    # Specification 5: SGD, Momentum 0.9, Weight Decay 1e-4, Initial LR 0.1
    opt = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4
    )

    # Specification 5: Cosine Annealing Scheduler (decays to 0 over 'epochs')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)

    # Standard CrossEntropy (removed label_smoothing to adhere to standard ResNet baseline specs)
    ce = nn.CrossEntropyLoss()

    # Initialize history structure for accuracy and loss tracking
    for k in ("clf_train_acc", "clf_val_acc", "clf_train_loss", "clf_val_loss"):
        hist.setdefault(k, {})[cid] = []

    steps = 0

    # Profile the classifier for FLOPs tracking
    rep_batch = None
    try:
        for xb, yb in train_ld:
            rep_batch = xb.to(device)[: min(2, xb.size(0))]
            break
    except Exception:
        rep_batch = None

    if tracker is not None and rep_batch is not None:
        tracker.register_model(
            model,
            rep_batch,
            backward_factor=2.0,
            loss_extra_fwd=0.0,
        )
        tracker.start_phase(f"clf_c{cid}")

    for ep in range(epochs):
        # ---------------- Training Phase -----------------------------
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            out = model(x)
            loss = ce(out, y)

            loss.backward()
            opt.step()

            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            steps += 1

            if tracker is not None:
                tracker.count_train_step(1)

        # Step the scheduler after every epoch
        scheduler.step()

        tr_acc = correct / total if total > 0 else 0.0
        tr_loss = loss_sum / len(train_ld) if len(train_ld) > 0 else 0.0

        # ---------------- Validation Phase ---------------------------
        model.eval()
        val_loss_sum = 0.0
        correct_v = 0
        total_v = 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss_sum += ce(out, y).item()
                correct_v += (out.argmax(1) == y).sum().item()
                total_v += y.size(0)

        val_acc = correct_v / total_v if total_v > 0 else 0.0
        val_loss = val_loss_sum / len(val_ld) if len(val_ld) > 0 else 0.0

        # Record metrics
        hist["clf_train_acc"][cid].append(tr_acc)
        hist["clf_val_acc"][cid].append(val_acc)
        hist["clf_train_loss"][cid].append(tr_loss)
        hist["clf_val_loss"][cid].append(val_loss)

        # Retrieve current LR for logging
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"[CLF] Client {cid} | Epoch {ep + 1}/{epochs} | "
            f"LR: {current_lr:.6f} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

    model.cpu()
    if tracker is not None:
        tracker.end_phase(f"clf_c{cid}")

    return model, steps