"""
🤖 EDM2 Wrapper Module (API Compatible)
---------------------------------------

This module rewrites the original custom Diffusion Transformer (DiT) API to use
NVIDIA's EDM2 (Elucidating the Design Space of Diffusion Models 2) backend.

🧠 Purpose:
    To leverage the SOTA EDM2 U-Net architecture and loss weighting while
    maintaining strict compatibility with the existing project structure
    (trainers, evaluation, and experiment runner).

    Crucially, this adapts EDM2 to work with Epoch-based training loops rather
    than the original infinite 'kimg' loop.

🔧 Core Functionalities:
    • Wraps `modules.EDM2.training.networks_edm2.Precond` as `DiT`.
    • Adapts `rectified_flow_loss` to use `EDM2Loss`.
    • Adapts `rectified_flow_sampler` to use `edm_sampler`.

Author: Andrea Moleri
File Location: src/models/diffusion.py
Last Modified: 30/12/2025
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

# ==============================================================================
# 1. EDM2 Path Injection
# ==============================================================================
# We need to expose the inner EDM2 modules to Python imports.
# Assuming standard structure: src/models/diffusion.py -> src/modules/EDM2
current_file = Path(__file__).resolve()
src_root = current_file.parents[1]  # src/
edm2_root = src_root / "modules" / "EDM2"

if str(edm2_root) not in sys.path:
    sys.path.insert(0, str(edm2_root))

# Now we can import internal EDM2 modules
try:
    import dnnlib
    from training.networks_edm2 import Precond
    from training.training_loop import EDM2Loss
    from generate_images import edm_sampler
except ImportError as e:
    raise ImportError(f"Could not import EDM2 modules from {edm2_root}. Error: {e}")


# ==============================================================================
# 2. Configuration (API Compatible)
# ==============================================================================

@dataclass
class DiffusionConfig:
    """
    Configuration dataclass. Kept compatible with original DiT config,
    but mapped internally to EDM2 U-Net parameters.
    """
    in_ch: int = 3
    # 'embed_dim' maps to EDM2 'model_channels'
    embed_dim: int = 128
    # 'depth'/heads are mostly ignored by standard U-Net but kept for API compat
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    patch_size: int = 2

    num_classes: int = 0
    class_dropout: float = 0.1
    dropout: float = 0.1
    drop_path: float = 0.0
    use_sincos_pos: bool = True

    # NEW: EDM2 needs explicit resolution at init time.
    # If input_size is passed via CLI, it should populate this.
    img_resolution: int = 32


# ==============================================================================
# 3. Model Wrapper (DiT -> EDM2 Precond)
# ==============================================================================

class DiT(nn.Module):
    """
    Wrapper class that presents an EDM2 Preconditioned U-Net as 'DiT'.

    This ensures `model = DiT(cfg)` works in existing scripts.
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg

        # Map DiffusionConfig to EDM2 Precond arguments
        # EDM2 uses 'model_channels' as the base width

        # We construct the EDM2 Precond wrapper which contains the UNet
        self.edm_net = Precond(
            img_resolution=cfg.img_resolution,
            img_channels=cfg.in_ch,
            label_dim=cfg.num_classes,
            use_fp16=False, # Disable internal FP16, let the trainer handle AMP
            sigma_data=0.5, # Standard EDM2 default
            model_channels=cfg.embed_dim,
            # Pass through dropout if the EDM2 UNet supports it via kwargs
            dropout=cfg.dropout
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, class_labels: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass for the EDM2 model.

        Args:
            x: Input images (B, C, H, W)
            sigma: Noise levels (B, 1, 1, 1) or (B,)
            class_labels: conditioning labels (B,) or one-hot.

        Returns:
            Denoised image (if used in sampler) or specific output based on mode.
        """
        # Ensure sigma is correct shape for EDM2 (B, ) -> reshape inside if needed
        if sigma.ndim == 1:
            sigma = sigma.reshape(-1, 1, 1, 1)

        # Handle class labels: EDM2 expects (B, label_dim) or None
        # The trainer might pass (B,) int labels. We need to one-hot them if label_dim > 0
        if self.cfg.num_classes > 0 and class_labels is not None:
            if class_labels.ndim == 1 and class_labels.dtype == torch.long:
                # Convert to one-hot
                class_labels = torch.nn.functional.one_hot(
                    class_labels, num_classes=self.cfg.num_classes
                ).to(x.dtype)

        return self.edm_net(x, sigma, class_labels, **kwargs)


# ==============================================================================
# 4. Loss Function (Rectified Flow -> EDM2Loss)
# ==============================================================================

# Instantiate a global loss object to avoid recreating it every batch
# EDM2 defaults: P_mean=-0.4, P_std=1.0, sigma_data=0.5
_edm_loss_fn = EDM2Loss(P_mean=-0.4, P_std=1.0, sigma_data=0.5)

def rectified_flow_loss(
        model: DiT,
        x1: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        p_uncond: float = 0.1,
) -> torch.Tensor:
    """
    Computes the EDM2 Loss instead of raw Rectified Flow loss.

    The API name is kept as `rectified_flow_loss` to satisfy `trainers.py` imports,
    but logically this performs the EDM2 weighting and noise sampling.

    Args:
        model: The DiT wrapper containing .edm_net
        x1: Real clean images (B, C, H, W)
        y: Class labels
        p_uncond: Probability for classifier-free guidance training (dropout)
    """

    # Handle Classifier-Free Guidance (CFG) label dropout
    # EDM2 usually handles this via an embedding, but we do it manually here
    # to match the trainer's expectation.
    labels = None
    if y is not None and model.cfg.num_classes > 0:
        # Clone to avoid modifying original tensor
        y_in = y.clone()
        if p_uncond > 0.0:
            mask = torch.rand(y.shape[0], device=y.device) < p_uncond
            # We assume the user's trainer logic handles the "null" token generation
            # usually by creating a one-hot vector of zeros.
            # However, since we convert to one-hot in forward(), we can pass a dummy
            # index if needed, or zero out the one-hot embedding inside.

            # Simple approach: pass labels, but rely on model to handle one-hot conversion.
            # If we want to drop, we pass a zero-tensor for one-hot.
            pass # We leave raw labels here, handled below in one-hot conversion

        # Convert to one-hot float
        labels = torch.nn.functional.one_hot(y_in, num_classes=model.cfg.num_classes).float()

        if p_uncond > 0.0:
            # Zero out the label embedding for dropped samples (standard CFG implementation)
            mask = torch.rand(y.shape[0], device=y.device) < p_uncond
            labels[mask] = 0

    # Call the EDM2 Loss functor
    # internal logic:
    # 1. Samples sigma
    # 2. Adds noise to x1 -> x_noisy
    # 3. Calls model(x_noisy, sigma)
    # 4. Computes weighted MSE
    loss = _edm_loss_fn(model.edm_net, x1, labels)

    return loss.mean()


# ==============================================================================
# 5. Sampler (Euler -> EDM Sampler)
# ==============================================================================

@torch.no_grad()
def rectified_flow_sampler(
        model: DiT,
        n: int,
        shape: Tuple[int, int, int],
        steps: int = 32, # Mapped to EDM steps
        device: torch.device | None = None,
        y: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0, # Maps to EDM guidance
        max_batch: int = 64,
        autocast_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Generates images using the EDM deterministic sampler.

    Args:
        model: The DiT wrapper.
        n: Number of images.
        shape: (C, H, W)
        steps: Number of sampling steps (NFE).
        guidance_scale: CFG scale. (Note: EDM uses 'guidance', usually > 1 for strength).
                        If caller passes 0.0 (no guidance), we map to EDM 1.0.
                        If caller passes > 0 (e.g. 1.5), we map to 1.0 + scale.
    """
    if device is None:
        device = next(model.parameters()).device

    C, H, W = shape

    # Setup latents (S_churn=0 for deterministic sampling)
    # Note: edm_sampler expects noise as input
    latents = torch.randn([n, C, H, W], device=device)

    # Handle Labels
    class_labels = None
    if y is not None and model.cfg.num_classes > 0:
        # Convert to one-hot for EDM2
        class_labels = torch.nn.functional.one_hot(y, num_classes=model.cfg.num_classes).float().to(device)

    # Map Guidance
    # Codebase convention: scale=0.0 means no guidance.
    # EDM convention: guidance=1.0 means no guidance.
    # Codebase convention: scale=2.0 means w * cond + (1-w)*uncond
    edm_guidance = 1.0 + guidance_scale if guidance_scale > 0 else 1.0

    # We use the same network for conditional and unconditional (self-guidance logic or standard CFG)
    # If guidance > 1, edm_sampler handles the mixing.

    # Generate in batches to save VRAM
    outputs = []
    for i in range(0, n, max_batch):
        end = min(i + max_batch, n)
        batch_latents = latents[i:end]
        batch_labels = class_labels[i:end] if class_labels is not None else None

        # Call EDM2 Sampler
        # net and gnet are the same model here (standard CFG)
        images = edm_sampler(
            net=model.edm_net,
            noise=batch_latents,
            labels=batch_labels,
            gnet=model.edm_net,
            num_steps=steps,
            guidance=edm_guidance,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            S_churn=0, # Deterministic
            randn_like=torch.randn_like
        )
        outputs.append(images.cpu())

    return torch.cat(outputs, dim=0)