"""
🧪 Generative Model Evaluation Metrics
--------------------------------------

This module implements a comprehensive suite of quantitative metrics and utility
functions for evaluating the quality and diversity of generative models (e.g., GANs, VAEs).

🧠 Purpose:
    Designed for academic research to provide standardized comparisons between
    synthesized distributions and reference real-world data distributions. It
    encapsulates feature extraction via InceptionV3 and statistical distance calculations.

🔧 Core Functionalities:
    • Feature Extraction: Wraps InceptionV3 (ImageNet weights) to obtain embeddings and logits.
    • Distribution Metrics: Implements Fréchet Inception Distance (FID) and Kernel Inception Distance (KID).
    • Manifold Metrics: Computes Precision and Recall based on k-Nearest Neighbors (k-NN).
    • Image Quality: Provides Inception Score (IS), PSNR, and a simplified SSIM implementation.
    • Analysis Artifacts: Exports pairwise distance matrices and nearest-neighbor rankings.

🎯 Intended Use:
    • Benchmarking deep generative models.
    • Monitoring training progress in research pipelines.
    • Generating reproducible evaluation reports.

📁 Dependencies:
    • numpy
    • torch
    • torchvision (InceptionV3)

📝 Notes:
    The module gracefully degrades to pixel-level features if the InceptionV3
    weights cannot be loaded, though this is not recommended for SOTA comparisons.

Author: Andrea Moleri
File Location: src/metrics/evaluation.py
Last Modified: 21/11/2025
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional, List
from torchvision.models import inception_v3, Inception_V3_Weights

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    A wrapper class for the InceptionV3 architecture designed to extract
    high-dimensional feature representations and classification logits.

    This class facilitates the extraction of embeddings from the `avgpool` layer
    (2048-dim) for metrics like FID/KID, and softmax probabilities for Inception Score.
    It includes a fallback mechanism to raw pixel features if the deep model
    cannot be initialized.

    Attributes:
        device (torch.device): The computation device (CPU or CUDA).
        kind (str): The mode of extraction, either 'inception' or 'pixel'.
        model (Optional[nn.Module]): The underlying InceptionV3 model.
        has_logits (bool): Flag indicating if class probabilities can be computed.
        preproc (Optional[Callable]): The official ImageNet preprocessing pipeline.
    """

    def __init__(self, device: torch.device):
        """
        Initialize the FeatureExtractor and attempt to load InceptionV3.

        Args:
            device (torch.device): The device to load the model onto.
        """
        self.device = device
        self.kind = "pixel"
        self.model = None
        self.has_logits = False
        try:
            # Load official ImageNet weights (V1)
            weights = Inception_V3_Weights.IMAGENET1K_V1
            # Enable aux_logits as per default architecture, though not used for inference here
            m = inception_v3(weights=weights, aux_logits=True)
            m.eval().to(device)

            self.feat = None

            # Register a forward hook to intercept the output of the global average pooling layer.
            # This is necessary because the standard forward() method might return aux logits
            # or skip the pooling layer depending on the implementation version.
            def _hook(_module, _input, output):
                # Expected output shape at avgpool: (Batch, 2048, 1, 1)
                # Flatten to (Batch, 2048) for downstream matrix operations.
                self.feat = torch.flatten(output, 1).detach()

            m.avgpool.register_forward_hook(_hook)

            self.model = m
            self.kind = "inception"
            self.has_logits = True

            # Store the official preprocessing transforms (Resize, Normalize) associated with the weights
            self.preproc = weights.transforms()
            logger.info("[Metrics] InceptionV3 (ImageNet) loaded for features & logits.")
        except Exception as e:
            # Fail-safe: If model loading fails (e.g., network issues), revert to pixel-space metrics.
            self.model = None
            self.kind = "pixel"
            self.has_logits = False
            self.preproc = None
            logger.warning(f"[Metrics] InceptionV3 unavailable ({e}). Falling back to pixel features (not SOTA).")

    @torch.no_grad()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input tensors to match the requirements of the feature extractor.

        Handles channel alignment (1 to 3 channels) and normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) with values in [-1, 1].

        Returns:
            torch.Tensor: Preprocessed tensor ready for model consumption.
        """
        # Input assumption: x is in range [-1, 1], shape (N, C, H, W)
        N, C, H, W = x.shape
        
        # Ensure 3 channels for ImageNet compatibility
        if C == 1:
            x = x.expand(-1, 3, -1, -1)
        elif C > 3:
            x = x[:, :3]

        # Rescale from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        
        if self.kind == "inception":
            # The Inception weights transform expects input in [0, 1].
            # Processing is done in chunks to manage memory usage during resizing operations.
            xs = []
            # Batch size of 256 for preprocessing steps to avoid OOM on GPU during resize
            for i in range(0, x.size(0), 256):
                xb = x[i:i + 256].cpu()
                xb = self.preproc(xb)  # Applies Resize(299) and ImageNet Normalization
                xs.append(xb)
            x = torch.cat(xs, dim=0).to(self.device)
        else:
            # Fallback: Simple bilinear interpolation to 64x64 for raw pixel analysis
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        return x

    @torch.no_grad()
    def features_and_logits(self, x: torch.Tensor, batch: int = 128) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract high-level features and class probabilities from the input batch.

        Args:
            x (torch.Tensor): Input images, shape (N, C, H, W), range [-1, 1].
            batch (int): Batch size for inference loop.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                - Features: (N, 2048) embedding (or flattened pixels if fallback).
                - Logits/Probs: (N, 1000) softmax probabilities, or None if fallback.
        """
        if self.kind == "inception" and self.model is not None:
            feats = []
            probs = []
            # Process in batches to fit within GPU memory constraints
            for i in range(0, x.size(0), batch):
                xb = x[i:i + batch].to(self.device)
                xb = self.preprocess(xb)
                
                # Forward pass triggers the hook, populating self.feat
                _ = self.model(xb)
                
                feat = self.feat
                feats.append(feat.cpu())
                
                if self.has_logits:
                    # Compute logits using the fully connected layer
                    logits = self.model.fc(feat)  # Shape: (B, 1000)
                    p = torch.softmax(logits, dim=1).cpu()
                    probs.append(p)
            
            feats = torch.cat(feats, dim=0).numpy()
            probs_np = torch.cat(probs, dim=0).numpy() if probs else None
            return feats, probs_np
        else:
            # Fallback Logic: Downsample and flatten images to serve as raw feature vectors.
            # Note: Metrics calculated on this are not comparable to standard FID.
            x_small = F.interpolate((x + 1.0) / 2.0, size=(64, 64), mode="bilinear", align_corners=False)
            feats = x_small.flatten(1).cpu().numpy().astype(np.float32)
            return feats, None


def _cov_mean_feats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the empirical mean and covariance matrix of the feature set.

    Args:
        feats (np.ndarray): Feature matrix of shape (N, D).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Mean vector (D,).
            - Covariance matrix (D, D).
    """
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _trace_sqrt_product(sigma1: np.ndarray, sigma2: np.ndarray) -> float:
    r"""
    Compute the trace of the square root of the product of two covariance matrices.
    
    Mathematically computes: $Tr((\Sigma_1 \Sigma_2)^{1/2})$.
    
    Implementation Note:
        To ensure numerical stability and symmetry, this is computed as:
        $Tr((\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2})^{1/2})$.

    Args:
        sigma1 (np.ndarray): Covariance matrix 1.
        sigma2 (np.ndarray): Covariance matrix 2.

    Returns:
        float: The scalar trace value.
    """
    eps = 1e-6
    # Eigendecomposition of sigma1 + regularization for stability
    s1, U1 = np.linalg.eigh(sigma1 + np.eye(sigma1.shape[0]) * eps)
    
    # Construct sigma1^(1/2)
    sqrt_s1 = (U1 * np.sqrt(np.clip(s1, 0, None))) @ U1.T
    
    # Compute inner term A = sigma1^(1/2) * sigma2 * sigma1^(1/2)
    A = sqrt_s1 @ sigma2 @ sqrt_s1
    
    # Compute eigenvalues of A to find its square root trace
    # Note: 0.5 * (A + A.T) enforces symmetry in case of numerical drift
    sA, _ = np.linalg.eigh((A + A.T) * 0.5)
    
    return float(np.sum(np.sqrt(np.clip(sA, 0, None))))


def fid_from_feats(f_real: np.ndarray, f_fake: np.ndarray) -> float:
    r"""
    Calculate the Fréchet Inception Distance (FID) between real and fake features.

    FID measures the Wasserstein-2 distance between two multivariate Gaussian distributions.
    Formula: $||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$.

    Args:
        f_real (np.ndarray): Features from real data (N, D).
        f_fake (np.ndarray): Features from generated data (M, D).

    Returns:
        float: The computed FID score (lower is better).
    """
    mu_r, sig_r = _cov_mean_feats(f_real)
    mu_f, sig_f = _cov_mean_feats(f_fake)
    
    # Euclidean distance between means
    diff = mu_r - mu_f
    
    # Trace term
    tr = np.sum(np.diag(sig_r + sig_f)) - 2.0 * _trace_sqrt_product(sig_r, sig_f)
    
    fid = float(diff @ diff + tr)
    return float(max(fid, 0.0))


def kid_unbiased(f_real: np.ndarray, f_fake: np.ndarray) -> float:
    r"""
    Calculate the unbiased Kernel Inception Distance (KID).

    KID is the Maximum Mean Discrepancy (MMD) squared, utilizing a 
    polynomial kernel of degree 3: $k(x, y) = (x^T y / d + 1)^3$.
    It is generally less biased than FID for smaller sample sizes.

    Args:
        f_real (np.ndarray): Features from real data.
        f_fake (np.ndarray): Features from generated data.

    Returns:
        float: The unbiased KID estimate.
    """
    d = f_real.shape[1]

    def k(X, Y):
        """Polynomial kernel function."""
        return (X @ Y.T / d + 1.0) ** 3

    # Compute kernel matrices
    K_rr = k(f_real, f_real)
    K_ff = k(f_fake, f_fake)
    K_rf = k(f_real, f_fake)
    
    n = f_real.shape[0]
    m = f_fake.shape[0]
    
    # Unbiased estimator of MMD^2
    mmd = (
            (np.sum(K_rr) - np.trace(K_rr)) / (n * (n - 1))
            + (np.sum(K_ff) - np.trace(K_ff)) / (m * (m - 1))
            - 2.0 * np.sum(K_rf) / (m * n)
    )
    return float(mmd)


def precision_recall_knn(
        f_real: np.ndarray, f_fake: np.ndarray, k: int = 3
) -> Tuple[float, float]:
    """
    Compute Precision and Recall for Generative Models (Manifold method).
    
    Based on "Improved Precision and Recall Metric for Assessing Generative Models" 
    (Kynkäänniemi et al., 2019).

    Metrics:
        - **Precision**: Fidelity/Realism. Fraction of generated samples that reside 
          within the manifold of the real distribution.
        - **Recall**: Diversity/Coverage. Fraction of real samples that reside 
          within the manifold of the generated distribution.

    Args:
        f_real (np.ndarray): Features from real data.
        f_fake (np.ndarray): Features from generated data.
        k (int): The number of nearest neighbors used to estimate the manifold radius.

    Returns:
        Tuple[float, float]: (Precision, Recall).
    """

    def pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute efficient pairwise Euclidean distance using the identity:
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        """
        AA = np.sum(A * A, axis=1, keepdims=True)
        BB = np.sum(B * B, axis=1, keepdims=True).T
        # Clip at 0.0 to avoid numerical errors (negative variance)
        D2 = np.maximum(AA + BB - 2.0 * (A @ B.T), 0.0)
        return np.sqrt(D2)

    n_r = f_real.shape[0]
    n_f = f_fake.shape[0]
    
    # Ensure k is within valid bounds given sample size
    k_eff_r = int(max(1, min(k, n_r - 1))) if n_r > 1 else 1
    k_eff_f = int(max(1, min(k, n_f - 1))) if n_f > 1 else 1

    # 1. Determine Manifold Radii: Distance to the k-th nearest neighbor within the same set
    D_rr = pairwise_dist(f_real, f_real)
    np.fill_diagonal(D_rr, np.inf) # Ignore self-distance
    # Use partitioning to find k-th NN distance efficiently O(N)
    rad_r = np.partition(D_rr, kth=k_eff_r - 1, axis=1)[:, k_eff_r - 1]

    D_ff = pairwise_dist(f_fake, f_fake)
    np.fill_diagonal(D_ff, np.inf)
    rad_f = np.partition(D_ff, kth=k_eff_f - 1, axis=1)[:, k_eff_f - 1]

    # 2. Calculate Precision: Are fakes close enough to any real point?
    D_rf = pairwise_dist(f_fake, f_real)
    # Find closest real neighbor for each fake sample
    nn_r_idx = np.argmin(D_rf, axis=1)
    nn_r_dist = D_rf[np.arange(n_f), nn_r_idx]
    # Check if the distance is within the radius of that real neighbor
    precision = float(np.mean(nn_r_dist <= rad_r[nn_r_idx])) if n_f > 0 else 0.0

    # 3. Calculate Recall: Are reals close enough to any fake point?
    D_fr = pairwise_dist(f_real, f_fake)
    nn_f_idx = np.argmin(D_fr, axis=1)
    nn_f_dist = D_fr[np.arange(n_r), nn_f_idx]
    recall = float(np.mean(nn_f_dist <= rad_f[nn_f_idx])) if n_r > 0 else 0.0

    return precision, recall


def inception_score(probs: np.ndarray) -> float:
    r"""
    Calculate the Inception Score (IS) from class probabilities.

    IS measures the KL divergence between the conditional distribution $p(y|x)$ 
    and the marginal distribution $p(y)$. High IS indicates that images are 
    clear (low entropy $p(y|x)$) and diverse (high entropy $p(y)$).

    Args:
        probs (np.ndarray): Softmax probabilities (N, 1000).

    Returns:
        float: The calculated Inception Score.
    """
    # Clip probabilities to avoid log(0)
    p_yx = np.clip(probs, 1e-10, 1.0)
    
    # Marginal distribution p(y)
    p_y = p_yx.mean(axis=0, keepdims=True)
    
    # KL divergence for each image
    kl = np.sum(p_yx * (np.log(p_yx) - np.log(p_y)), axis=1)
    
    # IS = exp(E[KL])
    return float(np.exp(np.mean(kl)))


def psnr_from_mse(mse: np.ndarray, max_val: float = 2.0) -> np.ndarray:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) from Mean Squared Error (MSE).

    Assumes images are in the range [-1, 1], hence `max_val` defaults to 2.0.

    Args:
        mse (np.ndarray): Array of MSE values.
        max_val (float): Dynamic range of the pixel values.

    Returns:
        np.ndarray: Array of PSNR values in decibels (dB).
    """
    mse_safe = np.clip(mse, 1e-12, None) # Avoid division by zero
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse_safe)


def ssim_simple(x: torch.Tensor, y: torch.Tensor, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> torch.Tensor:
    """
    Calculate a simplified Structural Similarity Index (SSIM).
    
    This implementation computes global statistics rather than using a sliding
    Gaussian window, serving as a fast approximation for batch metrics.

    Args:
        x (torch.Tensor): Batch of images [-1, 1].
        y (torch.Tensor): Batch of reference images [-1, 1].
        C1 (float): Stability constant for luminance.
        C2 (float): Stability constant for contrast.

    Returns:
        torch.Tensor: Scalar tensor representing the mean SSIM over the batch.
    """
    # Normalize to [0, 1] range for standard SSIM formula interpretation
    x = (x + 1) / 2
    y = (y + 1) / 2
    
    # Calculate global means
    mu_x = x.mean(dim=(-2, -1), keepdim=True)
    mu_y = y.mean(dim=(-2, -1), keepdim=True)
    
    # Calculate global variances and covariance
    sigma_x = ((x - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(-2, -1), keepdim=True)
    
    # SSIM formula
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    
    return ssim_map.mean()


def save_pairwise_outputs(
        run_root: Path,
        class_id: int,
        f_fake: np.ndarray,
        f_real: np.ndarray,
        topk: int = 5,
) -> None:
    """
    Compute and persist pairwise artifacts for qualitative analysis.

    Generates:
      1. Full Distance Matrix (Gen vs Real).
      2. Internal Diversity Matrix (Gen vs Gen).
      3. Top-K Nearest Neighbor CSV (Gen -> Real) for visual inspection.

    Args:
        run_root (Path): Root directory for saving artifacts.
        class_id (int): Identifier for the specific class being analyzed.
        f_fake (np.ndarray): Generated feature vectors.
        f_real (np.ndarray): Real feature vectors.
        topk (int): Number of nearest neighbors to record in the CSV.
    """
    outp = run_root / "artifacts" / "pairwise"
    outp.mkdir(parents=True, exist_ok=True)

    cls = f"class-{class_id:03d}"

    # 1. Compute Gen-to-Real Distances (L2)
    # Uses vectorized expansion: (a-b)^2 = a^2 + b^2 - 2ab
    D_gr = np.sqrt(
        np.maximum(
            0.0,
            np.sum(f_fake ** 2, axis=1, keepdims=True)
            + np.sum(f_real ** 2, axis=1, keepdims=True).T
            - 2.0 * (f_fake @ f_real.T),
        )
    ).astype(np.float32)
    np.save(outp / f"{cls}_gen-to-real.npy", D_gr)

    # 2. Create Top-K CSV (Gen -> Real mapping)
    tk = int(min(topk, D_gr.shape[1])) if D_gr.size else 0
    if tk > 0:
        # Efficiently retrieve indices of the k smallest elements per row
        idx = np.argpartition(D_gr, kth=tk - 1, axis=1)[:, :tk]
        rows = []
        for i in range(D_gr.shape[0]):
            ii = idx[i]
            # Argpartition does not guarantee order, so sort the top k locally
            order = np.argsort(D_gr[i, ii])
            nn_idx = ii[order]
            nn_dist = D_gr[i, nn_idx]
            for j, (jj, dd) in enumerate(zip(nn_idx, nn_dist)):
                rows.append((i, j, int(jj), float(dd)))

        with open(outp / f"{cls}_top{tk}_gen-to-real.csv", "w") as f:
            f.write("gen_index,rank,real_index,distance\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f}\n")

    # 3. Compute Gen-to-Gen Distances (Internal Diversity)
    D_gg = np.sqrt(
        np.maximum(
            0.0,
            np.sum(f_fake ** 2, axis=1, keepdims=True)
            + np.sum(f_fake ** 2, axis=1, keepdims=True).T
            - 2.0 * (f_fake @ f_fake.T),
        )
    ).astype(np.float32)
    
    if D_gg.size:
        # Mask diagonal (self-distance) to infinity to avoid trivial zero matches
        np.fill_diagonal(D_gg, np.inf)
    np.save(outp / f"{cls}_gen-to-gen.npy", D_gg)
