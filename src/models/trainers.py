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

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

logger = get_logger(__name__)

import copy
from typing import Dict, Tuple, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader

# Internal imports based on your structure
from models.diffusion import DiT, rectified_flow_loss
from metrics.costs import ExperimentCostTracker
from utils import _dp_add_noise_
from logs.logger import get_logger
from logs import messages as logmsg

# Import EDM2 EMA implementation
# Ensure src/modules/EDM2 is in your PYTHONPATH or adjusted in imports
from training.phema import PowerFunctionEMA

logger = get_logger(__name__)

# ==============================================================================
# UPDATED FUNCTION: train_diffusion
# Precision Fix: Removed native AMP/GradScaler (EDM2 handles FP16 internally)
# Optimizer Fix: Adam, lr=1e-2, no weight decay
# EMA Fix: Added PowerFunctionEMA
# ==============================================================================

# [CRITICAL FIX] Disable TF32 to match EDM2 Reference stability requirements
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def train_diffusion(
        model: DiT,
        loader: DataLoader,
        device: torch.device,
        epochs: int,
        hist: Dict,
        cid: int | str,
        dp: bool = False,
        dp_clip: float = 1.0,
        dp_noise_mult: float = 1.1,
        dp_microbatch: int = 8,
        tracker: Optional[ExperimentCostTracker] = None,
        checkpoint_every: int = 0,
        checkpoint_dir: Optional[Path] = None,
        lr: float = 1e-2,
        grad_accum_steps: int = 1,
) -> Tuple[DiT, int]:
    """
    Train a Diffusion Transformer using the Rectified Flow objective.
    Matches EDM2 Reference: No Native AMP, Internal FP16, No Weight Decay.
    Now includes PowerFunctionEMA for stable, high-quality generation.
    """
    model.to(device)

    # [FIX] Use Adam without weight decay (EDM2 uses weight norm internally)
    # Reference uses betas=(0.9, 0.99), eps=1e-8
    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8)

    # [FIX] Initialize EMA
    # std=0.05 is a standard robust setting for EDM2 post-hoc reconstruction compatibility
    ema = PowerFunctionEMA(model, stds=[0.05])

    hist.setdefault("vae_loss", {})
    hist["vae_loss"][cid] = []

    # DP Settings
    clip_C = float(dp_clip)
    microbatch = max(1, int(dp_microbatch))
    noise_std = float(dp_noise_mult) * clip_C

    step_count = 0
    cur_nimg = 0

    # Profiling logic
    rep_batch = None
    try:
        for xb, _ in loader:
            rep_batch = xb.to(device)[: min(2, xb.size(0))]
            break
    except Exception:
        pass

    if tracker is not None and rep_batch is not None:
        tracker.register_model(model, rep_batch, backward_factor=2.0, loss_extra_fwd=0.0)
        tracker.start_phase(f"train_c{cid}")

    for ep in range(epochs):
        model.train()
        total_loss_sum = 0.0

        opt.zero_grad(set_to_none=True)

        for i, (x, _y) in enumerate(loader):
            # EDM2 expects float32 inputs in range [-1, 1].
            # Internal casting handles the rest.
            x = x.to(device, non_blocking=True).to(torch.float32)
            physical_batch_size = x.size(0)

            if not dp:
                # ---------- Standard Path (EDM2 Style) ----------

                # 1. Forward Pass (No autocast context)
                loss = rectified_flow_loss(model, x)
                loss = loss / grad_accum_steps

                # 2. Backward
                loss.backward()

                total_loss_sum += loss.item() * grad_accum_steps * physical_batch_size

                # 3. Optimizer Step (Accumulated)
                if (i + 1) % grad_accum_steps == 0:
                    # Optional: Clip grad norm (usually not needed for EDM2 but safe to keep loose)
                    # nn.utils.clip_grad_norm_(model.parameters(), 10.0)

                    opt.step()
                    opt.zero_grad(set_to_none=True)

                    # [FIX] Update EMA
                    effective_batch = physical_batch_size * grad_accum_steps
                    cur_nimg += effective_batch
                    ema.update(cur_nimg=cur_nimg, batch_size=effective_batch)

                    step_count += 1
                    if tracker is not None: tracker.count_train_step(1)

            else:
                # ---------- DP Path ----------
                # Warning: DP-SGD with EDM2 internal mixed precision is complex.
                # We strictly follow the non-AMP logic here for consistency with the fix.

                if i == 0: opt.zero_grad(set_to_none=True)

                # Initialize accumulators
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_accum = torch.zeros_like(p, memory_format=torch.preserve_format)

                microbatch_count = 0

                for mb_start in range(0, physical_batch_size, microbatch):
                    x_mb = x[mb_start: mb_start + microbatch]

                    # Forward (No autocast)
                    loss_mb = rectified_flow_loss(model, x_mb) / x_mb.size(0)
                    loss_mb.backward()

                    # Clip
                    total_norm_sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm_sq += float(p.grad.data.norm(2).pow(2))
                    total_norm = math.sqrt(total_norm_sq)
                    clip_coef = min(1.0, clip_C / (total_norm + 1e-6))

                    for p in model.parameters():
                        if p.grad is None: continue
                        p.grad.data.mul_(clip_coef)
                        p.grad_accum.add_(p.grad.data)
                        p.grad.detach_()
                        p.grad.zero_()

                    microbatch_count += 1
                    total_loss_sum += loss_mb.item() * x_mb.size(0)

                # Add Noise
                for p in model.parameters():
                    if not hasattr(p, "grad_accum"): continue
                    grad = p.grad_accum
                    _dp_add_noise_(grad, noise_std)
                    grad.div_(microbatch_count)
                    p.grad = grad
                    del p.grad_accum

                # nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()

                # Update EMA
                cur_nimg += physical_batch_size
                ema.update(cur_nimg=cur_nimg, batch_size=physical_batch_size)

                opt.zero_grad(set_to_none=True)
                step_count += 1
                if tracker is not None: tracker.count_train_step(1)

        # Handle remaining gradients if drop_last=False
        if not dp and (len(loader) % grad_accum_steps != 0):
            opt.step()
            opt.zero_grad(set_to_none=True)
            remaining_batch = (len(loader) % grad_accum_steps) * loader.batch_size
            cur_nimg += remaining_batch
            ema.update(cur_nimg=cur_nimg, batch_size=remaining_batch)
            step_count += 1

        # Logging - [CHANGED] Now prints every epoch
        avg_loss = total_loss_sum / len(loader.dataset)
        hist["vae_loss"][cid].append(avg_loss)

        tag = "DP" if dp else "STD"
        logger.info(logmsg.VAE_EPOCH.format(cid=cid, epoch=ep + 1, epochs=epochs, tag=tag, loss=avg_loss))

        # Checkpointing (Saving EMA weights)
        if checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0 and checkpoint_dir is not None:
            ckpt_path = checkpoint_dir / f"checkpoint-client-{cid}-epoch-{ep + 1:04d}.pt"
            try:
                # Retrieve the EMA model (smoothed weights)
                best_ema_model = ema.get()[0][0]
                torch.save(best_ema_model.state_dict(), ckpt_path)
                logger.info(f"[CHECKPOINT] Saved EMA model to {ckpt_path}")
            except Exception as e:
                logger.warning(f"Checkpoint failed: {e}")

    model.cpu()

    # Return the EMA smoothed model for evaluation/inference
    ema_model_final = ema.get()[0][0].cpu()

    if tracker is not None:
        tracker.end_phase(f"train_c{cid}")

    return ema_model_final, step_count


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