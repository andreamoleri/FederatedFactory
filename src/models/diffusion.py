# ==============================================================================
# FILE: src/models/diffusion.py
# ==============================================================================
from __future__ import annotations

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

# ==============================================================================
# 1. EDM2 Path Injection
# ==============================================================================
current_file = Path(__file__).resolve()
src_root = current_file.parents[1]  # src/
edm2_root = src_root / "modules" / "EDM2"

if str(edm2_root) not in sys.path:
    sys.path.insert(0, str(edm2_root))

try:
    import dnnlib
    from training.networks_edm2 import Precond
    from training.training_loop import EDM2Loss
    from generate_images import edm_sampler
except ImportError as e:
    raise ImportError(f"Could not import EDM2 modules from {edm2_root}. Error: {e}")


# ==============================================================================
# 2. Configuration (Updated to match Colleague)
# ==============================================================================

@dataclass
class DiffusionConfig:
    in_ch: int = 3
    embed_dim: int = 128

    # NEW: Added channel multipliers to match colleague's [1, 2, 2, 2]
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 2])

    # Legacy args kept for compatibility, but mostly unused by EDM2
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    patch_size: int = 2

    num_classes: int = 0
    class_dropout: float = 0.1

    # UPDATED: Default dropout to match colleague
    dropout: float = 0.3
    drop_path: float = 0.0
    use_sincos_pos: bool = True
    img_resolution: int = 32


# ==============================================================================
# 3. Model Wrapper
# ==============================================================================

class DiT(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg

        # Map DiffusionConfig to EDM2 Precond arguments
        self.edm_net = Precond(
            img_resolution=cfg.img_resolution,
            img_channels=cfg.in_ch,
            label_dim=cfg.num_classes,
            use_fp16=True,  # Explicitly ENABLE FP16
            sigma_data=0.5,
            model_channels=cfg.embed_dim,

            # --- CRITICAL UPDATES FOR COLLEAGUE MATCHING ---
            channel_mult=cfg.channel_mult,  # Controls network depth/width per resolution
            dropout=cfg.dropout  # High dropout (0.3)
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, class_labels: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.reshape(-1, 1, 1, 1)

        if self.cfg.num_classes > 0 and class_labels is not None:
            if class_labels.ndim == 1 and class_labels.dtype == torch.long:
                class_labels = torch.nn.functional.one_hot(
                    class_labels, num_classes=self.cfg.num_classes
                ).to(x.dtype)

        return self.edm_net(x, sigma, class_labels, **kwargs)


# ==============================================================================
# 4. Loss Function
# ==============================================================================

# MATCHED: Colleague uses P_mean=-0.8, P_std=1.6
_edm_loss_fn = EDM2Loss(P_mean=-0.8, P_std=1.6, sigma_data=0.5)


def rectified_flow_loss(
        model: DiT,
        x1: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        p_uncond: float = 0.1,
) -> torch.Tensor:
    labels = None
    if y is not None and model.cfg.num_classes > 0:
        y_in = y.clone()
        if p_uncond > 0.0:
            pass  # CFG logic handled by zeroing one-hot below

        labels = torch.nn.functional.one_hot(y_in, num_classes=model.cfg.num_classes).float()

        if p_uncond > 0.0:
            mask = torch.rand(y.shape[0], device=y.device) < p_uncond
            labels[mask] = 0

    loss = _edm_loss_fn(model.edm_net, x1, labels)
    return loss.mean()


# ==============================================================================
# 5. Sampler
# ==============================================================================

@torch.no_grad()
def rectified_flow_sampler(
        model: DiT,
        n: int,
        shape: Tuple[int, int, int],
        steps: int = 32,
        device: torch.device | None = None,
        y: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        max_batch: int = 64,
        autocast_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device

    C, H, W = shape
    latents = torch.randn([n, C, H, W], device=device)

    class_labels = None
    if y is not None and model.cfg.num_classes > 0:
        class_labels = torch.nn.functional.one_hot(y, num_classes=model.cfg.num_classes).float().to(device)

    edm_guidance = 1.0 + guidance_scale if guidance_scale > 0 else 1.0

    outputs = []
    for i in range(0, n, max_batch):
        end = min(i + max_batch, n)
        batch_latents = latents[i:end]
        batch_labels = class_labels[i:end] if class_labels is not None else None

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
            S_churn=0,
            randn_like=torch.randn_like
        )
        outputs.append(images.cpu())

    return torch.cat(outputs, dim=0)