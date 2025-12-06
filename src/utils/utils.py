"""
LEGACY, ONLY UNCOMMENT IF PROBLEMS ARISE
from __future__ import annotations

import itertools
import math
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, utils as tv_utils

# ---- project-level imports ---------------------------------------------------
from imports.data_augmentation import denormalize
from models.vae import VAE, Decoder

__all__ = [
    "set_seed",
    "VGGPerceptualLoss",
    "sample_grid",
    "_dp_add_noise_",
    "grid_from_tensors",
    "plot_curves",
    "decoder_size_mb",
    "parse_csv_or_single",
    "expand_grid",
]

def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VGGPerceptualLoss(nn.Module):

    def __init__(self, weight: float = 1.0):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        # three blocks: conv1-2, conv3-4, conv5-6
        self.layers = nn.ModuleList(
            [vgg[:10].eval(), vgg[10:17].eval(), vgg[17:24].eval()]
        )
        for p in self.parameters():
            p.requires_grad = False
        self.weight = weight
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device, dtype=x.dtype)
        std = self.std.to(x.device, dtype=x.dtype)
        return ((x + 1) / 2 - mean) / std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = x.device
        x = self._preprocess(x.to(device))
        y = self._preprocess(y.to(device))
        loss = 0.0
        for i, layer in enumerate(self.layers):
            if next(layer.parameters()).device != device:
                self.layers[i] = layer.to(device)
            x, y = layer(x), layer(y)
            loss += F.l1_loss(x, y)
        return self.weight * loss


@torch.no_grad()
def sample_grid(model: VAE, latent_dim: int, out: Path, n: int = 64) -> None:
    z = torch.randn(n, latent_dim, device=next(model.parameters()).device)
    imgs = denormalize(model.decoder(z).cpu()).clamp(0, 1)
    tv_utils.save_image(imgs, out, nrow=8)


def _dp_add_noise_(param: torch.Tensor, stddev: float) -> None:
    if stddev > 0.0:
        param.add_(torch.randn_like(param) * stddev)


def grid_from_tensors(t: torch.Tensor, n: int = 8) -> np.ndarray:
    grid = tv_utils.make_grid(denormalize(t)[: n * n].clamp(0, 1), nrow=n)
    return grid.permute(1, 2, 0).cpu().numpy()




def plot_curves(
    hist: Dict[str, Dict[int | str, List[float]]],
    key: str,
    title: str,
    ylabel: str,
    ax: plt.Axes,
) -> None:
    for cid, series in hist[key].items():
        ax.plot(series, label=str(cid))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=5, fontsize=6)


def decoder_size_mb(decoder: Decoder) -> float:
    return sum(p.numel() for p in decoder.parameters()) * 4 / (1024 ** 2)


def parse_csv_or_single(value: str, cast):
    if "," in value:
        return [cast(v.strip()) for v in value.split(",") if v.strip() != ""]
    return [cast(value)]


def expand_grid(param_grid: Dict[str, List]):
    keys = list(param_grid.keys())
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo))
"""