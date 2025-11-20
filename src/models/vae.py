"""
🤖 Variational Autoencoder (VAE) Module
---------------------------------------

Neural components for a convolutional Variational Autoencoder comprising an
encoder, a decoder, and a lightweight residual block used in both pathways.

🧠 Purpose:
    Designed for image representation learning via stochastic latent variables,
    enabling generative modeling, compression, and downstream feature extraction.

🔧 Core Components:
    • ResidualBlock: two-layer Conv-BN-ReLU stack with an identity skip-connection
    • Encoder: strided convolutions to 4×4 spatial bottleneck; outputs μ and logσ²
    • Decoder: transposed convolutions from latent vector back to image space
    • VAE: end-to-end module tying encoder, reparameterization, and decoder

📊 Data Assumptions:
    • Input images are tensors of shape (N, C_in, H, W) with H = W = 32
    • Three strided downsamples (×2 each) map 32 → 16 → 8 → 4 spatially
    • Latent vectors are length `latent_dim`

🎯 Intended Use:
    • Academic demonstrations of VAEs
    • Research prototypes for generative modeling
    • Teaching materials for probabilistic deep learning

📝 Notes:
    • The output activation is tanh, implying inputs/targets are expected in [-1, 1].
    • Changing the input resolution or channel counts requires proportional
      adjustments to linear layer dimensions and convolutional shapes.

Author: Andrea Moleri
File Location: src/models/vae.py
Last Modified: 12/08/2025
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A two-layer residual convolutional block with identity skip connection.

    This block applies Conv-BatchNorm-ReLU twice and adds the input tensor
    back to the transformed output (pre-activation). The final nonlinearity
    is a ReLU applied to the summed tensor.

    Parameters
    ----------
    channels : int
        Number of input and output channels. The block preserves both channel
        count and spatial resolution.

    Returns
    -------
    torch.nn.Module
        An initialized residual block; call its ``forward`` method with a
        4D tensor of shape ``(N, channels, H, W)``.

    Notes
    -----
    - Because the skip path is an identity, input and output shapes must match.
    - Batch normalization statistics are learned per-channel across the batch.
    """

    def __init__(self, channels: int):
        super().__init__()
        # The convolution kernels use padding=1 to keep spatial dimensions
        # invariant (H, W unchanged). Channel count is fixed to `channels`.
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape ``(N, C, H, W)`` with ``C == channels``.

        Returns
        -------
        torch.Tensor
            Output feature map with the same shape as ``x``.

        Raises
        ------
        RuntimeError
            If tensor shapes are incompatible (e.g., channel mismatch) such
            that the elementwise addition cannot be performed.
        """
        # Residual addition before activation is a common pattern that helps
        # gradient flow; final activation provides nonlinearity after merging.
        return torch.relu(x + self.block(x))


class Encoder(nn.Module):
    """Convolutional encoder producing mean and log-variance of q(z|x).

    Architecture
    ------------
    - Three stages of (stride-2 convolution → ReLU → residual refinement)
      reducing spatial size by 2× per stage: 32 → 16 → 8 → 4.
    - Final 4×4×256 tensor is flattened and projected to latent parameters.

    Parameters
    ----------
    in_ch : int
        Number of input channels (e.g., 1 for grayscale, 3 for RGB).
    latent_dim : int
        Dimensionality of the latent space ``z``.

    Attributes
    ----------
    mu : torch.nn.Linear
        Linear head mapping the flattened representation to the latent mean.
    logvar : torch.nn.Linear
        Linear head mapping the flattened representation to the latent log-variance.

    Returns
    -------
    torch.nn.Module
        An initialized encoder; call its ``forward`` to obtain ``(mu, logvar)``.

    Notes
    -----
    - This encoder assumes input images are 32×32. Changing the input size
      requires updating the flattened dimension (currently ``256*4*4``).
    """

    def __init__(self, in_ch: int, latent_dim: int):
        super().__init__()
        # Each layer halves spatial dimensions via stride=2 and increases channels.
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.ReLU(True), ResidualBlock(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True), ResidualBlock(128)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True), ResidualBlock(256)
        )
        self.flatten = nn.Flatten()
        self.mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute latent mean and log-variance for the approximate posterior.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images with shape ``(N, in_ch, 32, 32)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - ``mu``: Tensor of shape ``(N, latent_dim)`` representing the mean.
            - ``logvar``: Tensor of shape ``(N, latent_dim)`` representing the
              natural-log variance (i.e., log σ²).

        Raises
        ------
        RuntimeError
            If input dimensions are incompatible with the network (e.g., wrong
            spatial size leading to mismatched flattened dimensions).
        """
        # Sequentially downsample and refine features; the final tensor is (N, 256, 4, 4).
        x = self.layer3(self.layer2(self.layer1(x)))
        h = self.flatten(x)  # Shape: (N, 256*4*4).
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """Convolutional decoder mapping latent vectors to image space.

    Architecture
    ------------
    - Linear projection from ``z`` to a 4×4×256 feature map.
    - Residual refinement at 256 channels.
    - Two upsampling stages via transposed convolutions to 8×8 and 16×16 with
      residual refinements.
    - Final transposed convolution upsamples to 32×32 and applies ``tanh``.

    Parameters
    ----------
    out_ch : int
        Number of output channels; typically equals the encoder's ``in_ch``.
    latent_dim : int
        Dimensionality of the latent vector provided to the decoder.

    Returns
    -------
    torch.nn.Module
        An initialized decoder; call its ``forward`` with a latent tensor.

    Notes
    -----
    - The final ``tanh`` suggests targets are scaled to [-1, 1]. If using
      [0, 1] targets, consider a sigmoid in downstream code (not modified here).
    """

    def __init__(self, out_ch: int, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.block = nn.Sequential(
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into images.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape ``(N, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed images with shape ``(N, out_ch, 32, 32)`` and values
            nominally in ``[-1, 1]`` due to the ``tanh`` activation.

        Raises
        ------
        RuntimeError
            If the latent dimensionality does not match the expected size.
        """
        # Project to spatial feature map and reshape to (N, 256, 4, 4) before upsampling.
        return self.block(self.fc(z).view(-1, 256, 4, 4))


class VAE(nn.Module):
    """Standard Variational Autoencoder with implicit β = 1.

    This module composes the encoder, a reparameterization step implementing
    the reparameterization trick, and the decoder. The forward pass returns the
    reconstruction along with the latent mean and log-variance for use in the
    evidence lower bound (ELBO).

    Parameters
    ----------
    in_ch : int
        Number of input and output channels for the image (e.g., 1 or 3).
    latent_dim : int
        Dimensionality of the latent variable ``z``.

    Attributes
    ----------
    encoder : Encoder
        Convolutional network producing ``mu`` and ``logvar`` from input images.
    decoder : Decoder
        Convolutional network mapping latent vectors back to images.

    Notes
    -----
    - The KL term weight is implicitly 1.0 (β-VAE with β=1). Any alternative
      weighting should be applied in the external loss function.
    """

    def __init__(self, in_ch: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(in_ch, latent_dim)
        self.decoder = Decoder(in_ch, latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vectors using the reparameterization trick.

        Computes ``z = μ + σ ⊙ ε`` where ``σ = exp(0.5 * logvar)`` and
        ``ε ~ N(0, I)`` is sampled with the same shape as ``σ``.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the approximate posterior with shape ``(N, latent_dim)``.
        logvar : torch.Tensor
            Log-variance of the approximate posterior with shape
            ``(N, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Sampled latent tensor ``z`` of shape ``(N, latent_dim)``.

        Notes
        -----
        - Sampling uses ``torch.randn_like`` to ensure ``ε`` matches device,
          dtype, and shape of ``std``.
        """
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, sample, and decode an input batch.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape ``(N, in_ch, 32, 32)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - ``recon``: Reconstructed images ``(N, in_ch, 32, 32)``.
            - ``mu``: Latent mean ``(N, latent_dim)``.
            - ``logvar``: Latent log-variance ``(N, latent_dim)``.

        Notes
        -----
        - Deterministic at test time only if ``logvar`` leads to near-zero
          ``std`` (not enforced here); otherwise outputs remain stochastic.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
