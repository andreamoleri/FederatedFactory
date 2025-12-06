"""
🤖 Rectified Flow Diffusion Transformer Module
----------------------------------------------

This module implements a Diffusion Transformer (DiT) architecture leveraging 
Rectified Flow for generative modeling. It provides a complete backbone for 
image generation tasks, including the neural network architecture, differential 
equation solvers for sampling, and loss computation for training.

🧠 Purpose:
    Constructs a scalable, transformer-based diffusion model that maps noise 
    to data via a straight-line probability flow (Rectified Flow), offering 
    deterministic sampling and stable training dynamics.

🔧 Core Functionalities:
    • Diffusion Transformer (DiT) architecture with adaptive layer initialisation
    • 2D Sinusoidal Positional Embeddings for spatial awareness
    • Rectified Flow loss formulation (velocity matching)
    • Euler method ODE integrator for deterministic image generation
    • Classifier-Free Guidance (CFG) support for conditional generation
    • Exponential Moving Average (EMA) for model weight stabilization

🎯 Intended Use:
    • Academic research in generative modelling and computer vision
    • High-fidelity image synthesis pipelines
    • Production-grade inference engines requiring deterministic sampling

📁 Dependencies:
    • torch
    • math
    • dataclasses

📝 Notes:
    The implementation assumes a standard image tensor format (B, C, H, W) 
    normalized to the range [-1, 1].

Author: Andrea Moleri
File Location: src/models/diffusion.py
Last Modified: 06/12/2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Generates sinusoidal timestep embeddings.

    Constructs a vector representation for scalar timesteps using a combination
    of sine and cosine functions at different frequencies, facilitating the
    network's temporal awareness.

    Parameters
    ----------
    t : torch.Tensor
        The tensor of timesteps with shape `(N,)`.
    dim : int
        The dimensionality of the output embedding.

    Returns
    -------
    torch.Tensor
        The embedding tensor of shape `(N, dim)`.
    """
    half = dim // 2
    # Compute frequencies in log space to cover a wide range of timescales
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0), math.log(10000.0), steps=half, device=t.device, dtype=t.dtype
        )
    )
    # Outer product to create arguments for sin/cos: (N, 1) * (1, half) -> (N, half)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # Pad the last dimension if the requested dimension is odd
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (N, dim)


def _get_2d_pos_embed(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    """
    Computes 2D sinusoidal positional embeddings for a grid.

    Creates a fixed positional encoding based on the spatial grid coordinates
    (height and width), independently encoding row and column positions.

    Parameters
    ----------
    h : int
        Height of the grid (number of patches vertically).
    w : int
        Width of the grid (number of patches horizontally).
    dim : int
        The embedding dimension. Must be divisible by 4.
    device : torch.device
        The device on which to create the tensors.
    dtype : torch.dtype
        The data type for the tensors.

    Returns
    -------
    torch.Tensor
        Positional embeddings of shape `(h * w, dim)`.

    Raises
    ------
    AssertionError
        If `dim` is not divisible by 4.
    """
    assert dim % 4 == 0, "positional dim must be divisible by 4"
    dim_each = dim // 4

    grid_y = torch.arange(h, device=device, dtype=dtype)
    grid_x = torch.arange(w, device=device, dtype=dtype)
    # Generate a grid of coordinates
    y, x = torch.meshgrid(grid_y, grid_x, indexing="ij")

    omega = torch.exp(
        torch.linspace(0, math.log(10000), dim_each, device=device, dtype=dtype)
    )

    # Normalize grid coordinates by frequencies
    y = y.reshape(-1)[:, None] / omega[None, :]
    x = x.reshape(-1)[:, None] / omega[None, :]

    # Concatenate sin/cos encodings for both X and Y axes
    emb = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
    return emb  # (h*w, dim)


class DropPath(nn.Module):
    """
    Implements Stochastic Depth (Drop Path).

    Randomly drops residual paths (per sample) during training to act as a 
    regularizer, preventing co-adaptation of parallel paths.

    Parameters
    ----------
    drop_prob : float, optional
        Probability of dropping the path. Default is 0.0.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies drop path regularization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of arbitrary shape.

        Returns
        -------
        torch.Tensor
            The processed tensor, with paths randomly zeroed out during training.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Work with arbitrary input dimensions; scale broadcast mask to (B, 1, 1, ...)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        # Apply mask and normalize to maintain expected value
        return x * (rand < keep_prob) / keep_prob


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block.

    A standard feed-forward network consisting of two linear layers separated 
    by a GELU activation and dropout.

    Parameters
    ----------
    dim : int
        Input and output dimensionality.
    hidden_dim : int
        Dimensionality of the hidden layer.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(B, N, dim)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(B, N, dim)`.
        """
        x = F.gelu(self.fc1(x), approximate="tanh")
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Computes attention weights between all pairs of tokens in the sequence
    and aggregates information accordingly. Uses PyTorch's memory-efficient 
    `scaled_dot_product_attention`.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    attn_dropout : float, optional
        Dropout probability for attention weights. Default is 0.0.
    proj_dropout : float, optional
        Dropout probability for the output projection. Default is 0.0.
    bias : bool, optional
        Whether to add bias to the query, key, and value projections. Default is False.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            attn_dropout: float = 0.0,
            proj_dropout: float = 0.0,
            bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert (
                head_dim * num_heads == dim
        ), "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape `(B, N, C)`.

        Returns
        -------
        torch.Tensor
            Attended tokens of shape `(B, N, C)`.
        """
        B, N, C = x.shape
        # Project and reshape to (B, N, 3, heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)  # each: (B, N, heads, head_dim)

        # Transpose to (B, heads, N, head_dim) for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # torch 2.x SDPA: memory efficient attention
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )  # (B, heads, N, head_dim)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN).

    Modulates the scale and shift parameters of layer normalization based on 
    an external conditioning vector (e.g., time or class embeddings).

    Parameters
    ----------
    dim : int
        The feature dimension to normalize.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim * 2)

        # Follows DiT style zero-initialization for stability
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Applies adaptive normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape `(B, N, D)`.
        cond : torch.Tensor
            Conditioning vector of shape `(B, D)`.

        Returns
        -------
        torch.Tensor
            Modulated features of shape `(B, N, D)`.
        """
        # cond: (B, D)
        # Split conditioning into scale (gamma) and shift (beta); each (B, D)
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        x = self.norm(x)
        # Modulate the normalized features
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        return x


class TransformerBlock(nn.Module):
    """
    DiT Transformer Block.

    A core building block consisting of Adaptive Layer Norm, Self-Attention, 
    and a Feed-Forward MLP, integrated with residual connections and scaling.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        Ratio of MLP hidden dimension to embedding dimension.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    drop_path : float, optional
        Stochastic depth probability. Default is 0.0.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float,
            dropout: float = 0.0,
            drop_path: float = 0.0,
    ):
        super().__init__()
        self.adaln1 = AdaLayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads,
            attn_dropout=dropout,
            proj_dropout=dropout,
            bias=False,
        )
        self.adaln2 = AdaLayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.drop_path = DropPath(drop_path)

        # Residual scaling factor for stability (common in smaller DiT architectures)
        self.res_scale = 1 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Process the input through the Transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens.
        cond : torch.Tensor
            Conditioning vector.

        Returns
        -------
        torch.Tensor
            Output tokens.
        """
        # Attention branch
        h = self.attn(self.adaln1(x, cond))
        x = x + self.drop_path(h) * self.res_scale

        # MLP branch
        h = self.mlp(self.adaln2(x, cond))
        x = x + self.drop_path(h) * self.res_scale
        return x


@dataclass
class DiffusionConfig:
    """
    Configuration dataclass for the Diffusion Transformer (DiT).

    Attributes
    ----------
    in_ch : int
        Number of input image channels (e.g., 3 for RGB).
    embed_dim : int
        Dimension of the token embeddings.
    depth : int
        Number of Transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        Expansion ratio for the MLP hidden layer.
    patch_size : int
        Size of the spatial patch (P x P).
    num_classes : int
        Number of classes for conditional generation (0 for unconditional).
    class_dropout : float
        Probability of dropping class labels during training (Classifier-Free Guidance).
    dropout : float
        General dropout probability.
    drop_path : float
        Stochastic depth drop probability.
    use_sincos_pos : bool
        Whether to use fixed sinusoidal positional embeddings.
    """
    in_ch: int = 3
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    patch_size: int = 2

    num_classes: int = 0
    class_dropout: float = 0.1
    dropout: float = 0.0
    drop_path: float = 0.0
    use_sincos_pos: bool = True


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) Model.

    The main architecture that processes latent patches conditioned on time 
    and optionally class labels. It treats image patches as tokens in a sequence,
    similar to Vision Transformers (ViT).

    Parameters
    ----------
    cfg : DiffusionConfig
        The configuration object defining architecture hyperparameters.
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        P = cfg.patch_size
        self.P = P
        self.patch_dim = cfg.in_ch * P * P

        # Patch projection (pixels -> tokens)
        self.proj = nn.Linear(self.patch_dim, cfg.embed_dim, bias=True)

        # Time embedding -> cond vector
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

        # Optional class embedding (last index = "null"/CFG token)
        if cfg.num_classes > 0:
            self.label_emb = nn.Embedding(cfg.num_classes + 1, cfg.embed_dim)
        else:
            self.label_emb = None

        # Transformer backbone
        dpr = torch.linspace(0, cfg.drop_path, cfg.depth).tolist()  # schedule drop_path
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.embed_dim,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    drop_path=dpr[i],
                )
                for i in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, self.patch_dim)

    # ---------------------------------------------------------------------
    # Patchify / unpatchify
    # ---------------------------------------------------------------------
    def _patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Converts images into a sequence of patch tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape `(B, C, H, W)`.

        Returns
        -------
        tokens : torch.Tensor
            Flattened patch tokens of shape `(B, L, patch_dim)`.
        h : int
            Grid height (number of patches).
        w : int
            Grid width (number of patches).

        Raises
        ------
        AssertionError
            If H or W are not divisible by the patch size.
        """
        B, C, H, W = x.shape
        assert H % self.P == 0 and W % self.P == 0, "H/W must be divisible by patch_size"
        h, w = H // self.P, W // self.P
        # BCHW -> B (h w) (P P C)
        # Reshape logic to extract patches and flatten them
        x = (
            x.reshape(B, C, h, self.P, w, self.P)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(B, h * w, self.P * self.P * C)
        )
        return x, h, w

    def _unpatchify(self, x: torch.Tensor, h: int, w: int, C: int) -> torch.Tensor:
        """
        Reconstructs images from a sequence of patch tokens.

        Parameters
        ----------
        x : torch.Tensor
            Patch tokens of shape `(B, L, patch_dim)`.
        h : int
            Grid height (number of patches).
        w : int
            Grid width (number of patches).
        C : int
            Number of channels.

        Returns
        -------
        torch.Tensor
            Reconstructed images of shape `(B, C, H, W)`.
        """
        B, N, PD = x.shape
        P = self.P
        img = (
            x.reshape(B, h, w, P, P, C)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(B, C, h * P, w * P)
        )
        return img

    # ---------------------------------------------------------------------
    # Conditioning vector from time (and optionally class label)
    # ---------------------------------------------------------------------
    def _condition_vector(
            self,
            t: torch.Tensor,
            y: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the combined conditioning vector from timestep and class labels.

        Combines time (mandatory) and, if present, class labels into a unified
        embedding vector used to modulate the transformer blocks.

        Parameters
        ----------
        t : torch.Tensor
            Timestep tensor.
        y : torch.Tensor, optional
            Class label tensor.

        Returns
        -------
        torch.Tensor
            Conditioning vector of shape `(B, D)`.
        """
        t_emb = timestep_embedding(t, self.cfg.embed_dim)  # (B, D)
        t_emb = self.time_mlp(t_emb)  # (B, D)

        if self.label_emb is not None and y is not None:
            # Safety clamp for indices
            y_emb = self.label_emb(y.clamp(min=0))
            # -1 indicates "null" embedding (index cfg.num_classes)
            y_is_null = (y == -1)
            if y_is_null.any():
                y_emb[y_is_null] = self.label_emb.weight[self.cfg.num_classes]
            cond = t_emb + y_emb
        else:
            cond = t_emb

        return cond  # (B, D)

    # ---------------------------------------------------------------------
    # Forward main (t can be None -> useful for FLOPs tracker)
    # ---------------------------------------------------------------------
    def forward(
            self,
            x_t: torch.Tensor,
            t: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The main forward pass of the DiT model.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy input image `(B, C, H, W)`, typically with values in [-1, 1].
        t : torch.Tensor, optional
            Timestep `(B,)` in range [0, 1]. If None, a constant value (0.5) is
            used as fallback. This facilitates compatibility with profiling tools
            expecting a `model(x)` signature.
        y : torch.Tensor, optional
            Class labels `(B,)` (int64) in range [0, num_classes-1].
            Use -1 for the "null" class (Classifier-Free Guidance).

        Returns
        -------
        torch.Tensor
            Predicted velocity field `(B, C, H, W)`.
        """
        B, C, H, W = x_t.shape

        # Fallback for profilers / FLOPs estimators that do not pass t
        if t is None:
            t = torch.full(
                (B,),
                0.5,
                device=x_t.device,
                dtype=x_t.dtype if x_t.is_floating_point() else torch.float32,
            )

        tokens, h, w = self._patchify(x_t)  # (B, L, patch_dim)
        tok = self.proj(tokens)  # (B, L, D)

        cond = self._condition_vector(t, y)  # (B, D)

        # Initial conditional injection
        tok = tok + cond[:, None, :]  # (B, L, D)

        # Fixed 2D positional embedding (preserves original behavior)
        if self.cfg.use_sincos_pos:
            pos = _get_2d_pos_embed(
                h,
                w,
                self.cfg.embed_dim,
                x_t.device,
                x_t.dtype,
            )  # (L, D)
            tok = tok + pos[None, :, :]

        # Transformer blocks
        for blk in self.blocks:
            tok = blk(tok, cond)

        tok = self.norm(tok)
        out_tokens = self.head(tok)  # (B, L, patch_dim)
        v = self._unpatchify(out_tokens, h, w, C)  # (B, C, H, W)
        return v

    # ---------------------------------------------------------------------
    # forward_dummy: helper only for profiling / FLOPs
    # ---------------------------------------------------------------------
    def forward_dummy(self, x_t: torch.Tensor, *_, **__) -> torch.Tensor:
        """
        Safe dummy forward method for profilers, FLOPs counters, or energy estimators.

        - Accepts extra arguments to avoid TypeError.
        - Uses fixed t = 0.5 and no labels.
        - Returns a SCALAR value to prevent potential explosion during a dummy .backward().

        Parameters
        ----------
        x_t : torch.Tensor
            Input tensor.
        *_, **__ : 
            Catch-all for unused arguments.

        Returns
        -------
        torch.Tensor
            Scalar mean of the output.
        """
        B = x_t.size(0)
        t = torch.full(
            (B,),
            0.5,
            device=x_t.device,
            dtype=x_t.dtype if x_t.is_floating_point() else torch.float32,
        )
        v = self.forward(x_t, t, y=None)  # (B,C,H,W)
        return v.mean()


# =============================================================================
# Rectified Flow objective
# =============================================================================


def rectified_flow_loss(
        model: DiT,
        x1: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        p_uncond: float = 0.1,
) -> torch.Tensor:
    """
    Computes the Rectified Flow loss.

    Implements the linear interpolation path:
        $$x_t = (1 - t) x_0 + t x_1$$
    where the target velocity is $$u = x_1 - x_0$$.
    The objective minimizes the Mean Squared Error (MSE) between the model
    prediction $$v_{\\theta}(x_t, t)$$ and the target $$u$$.

    Parameters
    ----------
    model : DiT
        The diffusion model instance.
    x1 : torch.Tensor
        The target data sample (e.g., real image).
    y : torch.Tensor, optional
        Class labels for conditional generation.
    p_uncond : float, optional
        Probability of dropping labels for Classifier-Free Guidance training.
        Default is 0.1.

    Returns
    -------
    torch.Tensor
        The computed scalar Mean Squared Error loss.

    Notes
    -----
    Compatible with the previous signature: `rectified_flow_loss(model, x1)`.
    If `y` is provided and the model is class-conditional (`cfg.num_classes > 0`),
    Classifier-Free Guidance masking is applied during training.
    """

    B = x1.size(0)
    device = x1.device

    # x0 ~ N(0, I) (Sample Gaussian noise)
    x0 = torch.randn_like(x1)

    # t ~ Uniform(0,1)
    t = torch.rand(B, device=device)

    # Construct the linear path interpolation
    x_t = (1.0 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
    target = x1 - x0  # velocity / direction

    # Apply classifier-free guidance during training (label dropout)
    y_in = None
    if y is not None and getattr(model.cfg, "num_classes", 0) > 0:
        if p_uncond > 0.0:
            mask = torch.rand(B, device=device) < p_uncond
            y_mod = y.clone()
            y_mod[mask] = -1  # -1 => "null" token
            y_in = y_mod
        else:
            y_in = y

    pred = model(x_t, t, y_in)
    loss = F.mse_loss(pred, target, reduction="mean")
    return loss


# =============================================================================
# Sampler (deterministic ODE integration for Rectified Flow)
# =============================================================================

@torch.no_grad()
def rectified_flow_sampler(
        model: DiT,
        n: int,
        shape: Tuple[int, int, int],
        steps: int = 50,
        device: torch.device | None = None,
        y: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        max_batch: int = 64,
        autocast_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Generates samples by integrating the Rectified Flow ODE.

    Performs forward integration:
        $$dx/dt = v_{\\theta}(x_t, t)$$
    from $$t=0$$ to $$t=1$$, starting from $$x_0 \\sim N(0,I)$$.

    Parameters
    ----------
    model : DiT
        The trained diffusion model.
    n : int
        Number of samples to generate.
    shape : Tuple[int, int, int]
        Shape of a single sample (C, H, W).
    steps : int, optional
        Number of integration steps (Euler method). Default is 50.
    device : torch.device, optional
        Computation device. Infers from model parameters if None.
    y : torch.Tensor, optional
        Class labels for conditional generation.
    guidance_scale : float, optional
        Scale for Classifier-Free Guidance. If > 0, enables guidance logic
        (provided the model supports classes). Default is 0.0.
    max_batch : int, optional
        Maximum batch size for generation to manage GPU memory. Default is 64.
    autocast_dtype : torch.dtype, optional
        Data type for automatic mixed precision (AMP) inference.

    Returns
    -------
    torch.Tensor
        Batch of generated images with values clamped to [-1, 1].
    """
    if device is None:
        device = next(model.parameters()).device

    C, H, W = shape
    model.eval()

    xs = []
    for start in range(0, n, max_batch):
        mb = min(max_batch, n - start)

        x = torch.randn(mb, C, H, W, device=device)
        dt = 1.0 / steps

        use_cfg = (
                y is not None
                and getattr(model.cfg, "num_classes", 0) > 0
                and (guidance_scale > 0.0)
        )
        if y is not None:
            y_mb = y[start:start + mb].to(device)
        else:
            y_mb = None

        for i in range(steps):
            t_val = i * dt
            t = torch.full((mb,), t_val, device=device)

            # Use autocast to reduce memory/time on GPU
            if autocast_dtype is not None and device.type == "cuda":
                with torch.autocast("cuda", dtype=autocast_dtype):
                    if use_cfg:
                        # Conditional pass
                        v_cond = model(x, t, y_mb)
                        # Unconditional pass: all labels set to -1 ("null")
                        v_uncond = model(x, t, torch.full_like(y_mb, -1))
                        v = (1.0 + guidance_scale) * v_cond - guidance_scale * v_uncond
                    else:
                        v = model(x, t, y_mb if y_mb is not None else None)
            else:
                if use_cfg:
                    v_cond = model(x, t, y_mb)
                    v_uncond = model(x, t, torch.full_like(y_mb, -1))
                    v = (1.0 + guidance_scale) * v_cond - guidance_scale * v_uncond
                else:
                    v = model(x, t, y_mb if y_mb is not None else None)

            # Euler integration step
            x = x + dt * v

        xs.append(x.clamp_(-1.0, 1.0).cpu())

    return torch.cat(xs, dim=0)


# =============================================================================
# Exponential Moving Average helper (EMA)
# =============================================================================

class ModelEMA(nn.Module):
    """
    Exponential Moving Average (EMA) of model weights.

    Maintains a "stabilized" version of the model for final sampling.

    Usage Example:
        ema = ModelEMA(model, decay=0.999)
        ...
        for step in ...:
            loss.backward(); opt.step()
            ema.update(model)
        # Then, to generate:
        samples = rectified_flow_sampler(
            ema.ema_model, n, shape, steps, y, guidance_scale
        )

    Parameters
    ----------
    model : nn.Module
        The source model to track.
    decay : float, optional
        The decay factor for the moving average. Default is 0.9999.
    device : torch.device, optional
        Device to store the EMA model on.
    """

    def __init__(
            self,
            model: nn.Module,
            decay: float = 0.9999,
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.decay = decay
        self.device = device
        self.ema_model = self._clone_model(model)

        if device is not None:
            self.to(device)

        self._set_requires_grad(False)

    @torch.no_grad()
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """
        Creates a deep copy of the model structure.
        """
        # If the model has a cfg in the constructor (like DiT), we recreate it
        ema = type(model)(model.cfg) if hasattr(model, "cfg") else type(model)()
        ema.load_state_dict(model.state_dict(), strict=True)
        ema.eval()
        return ema

    def _set_requires_grad(self, requires_grad: bool):
        """
        Enables or disables gradient calculation for the EMA model.
        """
        for p in self.ema_model.parameters():
            p.requires_grad_(requires_grad)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Updates the EMA weights using the current model weights.

        Formula:
        $$ \theta_{ema} = \text{decay} \cdot \theta_{ema} + (1 - \text{decay}) \cdot \theta_{current} $$
        """
        d = self.decay
        for p_ema, p in zip(self.ema_model.parameters(), model.parameters()):
            if self.device is not None and p_ema.device != self.device:
                p_ema.data = p_ema.data.to(self.device)
            p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)

        # Copy buffers as well (e.g., LayerNorm stats, etc.)
        for b_ema, b in zip(self.ema_model.buffers(), model.buffers()):
            b_ema.copy_(b)


# =============================================================================
# Optimizer / scheduler helpers
# =============================================================================

def build_optimizer(
        model: nn.Module,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        betas=(0.9, 0.999),
):
    """
    Constructs an AdamW optimizer with selective weight decay.

    Applies weight decay to standard weights but excludes bias terms and 
    normalization parameters to improve training stability.

    Parameters
    ----------
    model : nn.Module
        The model to optimize.
    lr : float, optional
        Learning rate. Default is 2e-4.
    weight_decay : float, optional
        Weight decay coefficient. Default is 0.01.
    betas : Tuple[float, float], optional
        AdamW beta coefficients. Default is (0.9, 0.999).

    Returns
    -------
    torch.optim.AdamW
        Configured optimizer.
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)


def cosine_lr(
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
):
    """
    Creates a learning rate scheduler function with warmup and cosine decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    base_lr : float
        Peak learning rate after warmup.
    warmup_steps : int
        Number of steps for linear warmup.
    total_steps : int
        Total training steps.

    Returns
    -------
    Callable[[int], None]
        A function `step_fn(step)` that updates the optimizer's learning rate.
    """

    def step_fn(step: int):
        if step < warmup_steps:
            lr = base_lr * float(step + 1) / float(max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    return step_fn