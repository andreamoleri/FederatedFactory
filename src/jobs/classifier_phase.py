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

🎯 Intended Use:
    • Federated Learning research (specifically data heterogeneity)
    • Synthetic Data Augmentation experiments
    • Comparative analysis of "Local" vs. "Global" model performance

📁 Dependencies:
    • torch
    • numpy
    • sklearn
    • internal models (cnn, vae, diffusion, trainers)

📝 Notes:
    The module assumes the existence of a specific experiment tracking interface 
    (`tracker`) and a file-system path configuration object (`P`).

Author: Andrea Moleri
File Location: src/jobs/classifier_phase.py
Last Modified: 21/11/2025
"""

from __future__ import annotations
import logging
import time
from typing import List, Dict, Optional, Tuple, Any, Set, Union

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from models.cnn import SimpleCNN
from models.trainers import train_classifier
from models.vae import Decoder
from models.diffusion import DiT, rectified_flow_sampler

logger = logging.getLogger(__name__)

@torch.no_grad()
def synth_from_decoder(dec: Decoder, latent: int, n: int, device: torch.device) -> torch.Tensor:
    """
    Generates synthetic images using a pre-trained Variational Autoencoder (VAE) decoder.

    This function samples latent vectors from a standard normal distribution,
    decodes them into image space, and transfers the result to the CPU to conserve 
    accelerator memory.

    Parameters
    ----------
    dec : Decoder
        The pre-trained VAE decoder module.
    latent : int
        The dimensionality of the latent space $z$.
    n : int
        The number of synthetic samples to generate.
    device : torch.device
        The computational device (CPU or GPU) used for inference.

    Returns
    -------
    torch.Tensor
        A tensor of generated images on the CPU.
    """
    dec.to(device).eval()
    # Sample from standard normal distribution N(0, I) and decode
    imgs = dec(torch.randn(n, latent, device=device)).cpu()
    dec.cpu()
    return imgs

@torch.no_grad()
def synth_from_diffusion(model: DiT, n: int, device: torch.device, img_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Generates synthetic images using a Diffusion Transformer (DiT) via Rectified Flow.

    Utilizes a rectified flow sampler to solve the ODE trajectory from noise to data.
    Ensures the model is placed on the correct device for inference and returned to 
    CPU afterwards.

    Parameters
    ----------
    model : DiT
        The pre-trained Diffusion Transformer model.
    n : int
        The number of synthetic samples to generate.
    device : torch.device
        The computational device (CPU or GPU) used for inference.
    img_shape : Tuple[int, int, int]
        The target dimensions of the generated images (Channels, Height, Width).

    Returns
    -------
    torch.Tensor
        A tensor of generated images on the CPU.
    """
    C, H, W = img_shape
    model.to(device).eval()
    # Execute the sampling process (50 steps standard for rectified flow in this context)
    x = rectified_flow_sampler(model, n=n, shape=(C, H, W), steps=50, device=device).cpu()
    model.cpu()
    return x.cpu()

# =============================================================================
# AGGIUNTA: Funzione mancante ensemble_accuracy
# =============================================================================
def ensemble_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the accuracy of predictions.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: The accuracy score (0.0 to 1.0).
    """
    return float((y_true == y_pred).mean())
# =============================================================================

@torch.no_grad()
def ensemble_preds_poexp(classifiers: List[SimpleCNN], test_ld: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Evaluates an ensemble of classifiers using a Product of Experts (PoE) aggregation strategy.

    The function aggregates predictions by summing the log-probabilities from all 
    classifiers. This is mathematically equivalent to multiplying the probabilities 
    assuming independence:
    $$ \hat{y} = \arg\max_c \sum_{i} \log P_i(y=c|x) $$

    Parameters
    ----------
    classifiers : List[SimpleCNN]
        A list of trained classifier models to ensemble.
    test_ld : DataLoader
        The data loader providing the test dataset.
    device : torch.device
        The computational device used for inference.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - y_true: The ground truth labels.
        - y_pred: The predicted labels derived from the ensemble.
    """
    for c in classifiers: c.to(device).eval()
    y_true, y_pred = [], []
    
    for x, y in test_ld:
        x, y = x.to(device), y.to(device)
        log_probs = None
        
        # Accumulate log probabilities across all ensemble members
        for c in classifiers:
            # Clamp log probabilities to prevent numerical instability (log(0))
            lp = torch.log_softmax(c(x), dim=1).clamp(min=np.log(1e-12))
            log_probs = lp if log_probs is None else log_probs + lp
        
        y_pred.append(log_probs.argmax(1).cpu())
        y_true.append(y.cpu())
        
    for c in classifiers: c.cpu()
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()

@torch.no_grad()
def evaluate_single_classifier(model: SimpleCNN, ld: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates a single classifier on a provided dataset.

    Parameters
    ----------
    model : SimpleCNN
        The classifier model to evaluate.
    ld : DataLoader
        The data loader containing evaluation data.
    device : torch.device
        The computational device used for inference.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - y_true: Concatenated ground truth labels.
        - y_pred: Concatenated predicted labels.
    """
    model.to(device).eval()
    y_true, y_pred = [], []
    for x, y in ld:
        y_true.append(y)
        y_pred.append(model(x.to(device)).argmax(1).cpu())
    model.cpu()
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()

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

    This function allocates generation tasks to different client models based on the 
    availability and quantity of real samples they possess. It ensures that the 
    aggregated synthetic dataset respects the relative contributions (weights) of 
    each client for a given class.

    Parameters
    ----------
    client_gen_models : Dict[str, Dict[int, Any]]
        A nested dictionary mapping client names to their class-specific generator models.
    client_sample_counts : Dict[str, Dict[int, int]]
        A nested dictionary mapping client names to the count of real samples they hold per class.
    samples_per_class : int
        The target number of synthetic samples to generate per class.
    model_kind : str
        The architecture type of the generator ('vae' or 'diffusion').
    latent_dim : int
        Dimensionality of the latent space (only relevant for VAE).
    device : torch.device
        The computational device.
    img_shape : Tuple[int, int, int]
        Dimensions of the images to generate.

    Returns
    -------
    Dict[int, torch.Tensor]
        A dictionary mapping class IDs to the generated synthetic image tensors.
    """
    weighted_samples = {}
    total_samples_per_class = {}
    all_classes = set()
    
    # Aggregate global statistics: which classes exist and their total counts across the federation
    for _, counts in client_sample_counts.items():
        for cid, c in counts.items():
            all_classes.add(cid)
            total_samples_per_class[cid] = total_samples_per_class.get(cid, 0) + c

    for cid in sorted(all_classes):
        # Determine the generation target; default to the natural distribution count if target is 0
        target = samples_per_class if samples_per_class > 0 else total_samples_per_class.get(cid, 0)
        if target == 0: 
            weighted_samples[cid] = torch.tensor([])
            continue

        # Identify valid contributors for the current class ID (cid)
        contributors = []
        for cname, models in client_gen_models.items():
            if cid in models:
                contributors.append((cname, models[cid], client_sample_counts[cname].get(cid, 0)))
        
        # Prioritize contributors with more real data (higher weight/fidelity assumption)
        contributors.sort(key=lambda x: x[2], reverse=True)
        total_real = total_samples_per_class[cid]
        
        cls_samples = []
        remaining = target
        
        # Distribute generation quotas proportionally to real data holdings
        for i, (cname, model, count) in enumerate(contributors):
            # Assign remaining quota to the last contributor to avoid rounding errors
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
) -> Tuple[int, int, float, np.ndarray, np.ndarray, List[Any], Optional[Any]]:
    """
    Executes the primary classifier training and evaluation workflow.

    This function handles multiple experimental configurations, primarily defined by 
    the data partition scheme ('skew'/'dirichlet' vs. 'silos') and the inference 
    location ('local' vs. 'server'). It manages data loading, synthetic augmentation,
    model training, state serialization, and performance evaluation.

    Parameters
    ----------
    args : Any
        Configuration namespace containing hyperparameters (batch size, epochs, model type, etc.).
    device : torch.device
        The primary computational device.
    P : Any
        Path configuration object containing the project root and output directories.
    train_subsets_dict : Dict[str, Dict[int, Any]]
        Mapping of client IDs to their specific class-subset indices (used in Skew mode).
    train_subsets : List[Any]
        List of subset indices corresponding to classes (used in Silos mode).
    present_classes : List[int]
        List of class labels present in the current training context.
    num_classes : int
        Total number of distinct classes in the dataset.
    chans : int
        Number of image channels.
    img_shape : Tuple[int, int, int]
        Dimensions of the input images.
    tracker : Optional[Any]
        Experiment tracking interface for logging phases and metrics.
    hist : Any
        History object for recording training loss/accuracy curves.
    gen_models : List[Optional[Any]]
        List of generative models indexed by class ID (primarily for server/silos usage).
    client_gen_models : Dict[str, Dict[int, Any]]
        Map of client-specific generative models (for weighted sampling).
    client_sample_counts : Dict[str, Dict[int, int]]
        Map of data counts per client/class.
    reserved_test_ld : DataLoader
        The hold-out test set for final evaluation.

    Returns
    -------
    Tuple[int, int, float, np.ndarray, np.ndarray, List[Any], Optional[Any]]
        Returns a tuple containing:
        - classifier_steps: Total training steps performed.
        - synth_images_total: Total synthetic images generated.
        - clf_start: Timestamp of start.
        - y_true: Ground truth labels from evaluation.
        - y_pred: Predicted labels from evaluation.
        - trained_clfs: List of trained classifier models (for ensembles).
        - single_clf: The central classifier instance (if applicable).
    """
    classifier_steps = 0
    synth_images_total = 0
    clf_start = time.perf_counter()
    
    partition_mode = getattr(args, "partition", "silos")
    is_skew = partition_mode in ["skew", "dirichlet"]
    is_local = args.infer_mode == "local"
    
    trained_clfs = []
    single_clf = None
    y_true, y_pred = np.array([]), np.array([])

    # --------------------------------------------------
    # PARTITION MODE: SKEW / DIRICHLET
    # --------------------------------------------------
    if is_skew:
        if is_local:
            # Case: Local Inference with Skewed Data
            # Each client trains its own classifier using local real data + global synthetic data
            for cname, subsets in train_subsets_dict.items():
                # --- Prepare Real Data ---
                xs, ys = [], []
                for cid, sub in subsets.items():
                    # Dynamic import required by original design pattern
                    from jobs.baseline_runner import subset_to_tensor
                    imgs = subset_to_tensor(sub)
                    xs.append(imgs)
                    ys.append(torch.full((len(imgs),), cid, dtype=torch.long))
                
                if not xs: continue
                X, y = torch.cat(xs), torch.cat(ys)
                
                # --- Generate Synthetic Data ---
                # Augment missing or scarce classes
                target = args.samples_per_class if args.samples_per_class > 0 else 2000
                synth_xs, synth_ys = [], []
                
                for cid in range(num_classes):
                    if cid in subsets: continue # Skip if client already possesses real data for this class
                    
                    # Select generation strategy: Weighted (multi-client) vs Simple (single-source)
                    if getattr(args, "aggregation", "simple") == "weighted":
                         # Logic for weighted generation based on client contributions
                         w_samps = generate_weighted_samples(client_gen_models, client_sample_counts, target, args.model, args.latent_dim, device, img_shape)
                         s_imgs = w_samps.get(cid, torch.tensor([]))
                    else:
                         # Simple generation from the primary model list
                         if cid < len(gen_models) and gen_models[cid] is not None:
                             m = gen_models[cid]
                             if args.model == "vae": s_imgs = synth_from_decoder(m, args.latent_dim, target, device)
                             else: s_imgs = synth_from_diffusion(m, target, device, img_shape)
                         else: s_imgs = torch.tensor([])

                    if len(s_imgs) > 0:
                        synth_xs.append(s_imgs)
                        synth_ys.append(torch.full((len(s_imgs),), cid, dtype=torch.long))

                if synth_xs:
                    X = torch.cat([X, *synth_xs])
                    y = torch.cat([y, *synth_ys])
                    synth_images_total += sum(len(x) for x in synth_xs)

                # --- Train Classifier ---
                # Create stratified validation split
                Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                tr_ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
                val_ld = DataLoader(TensorDataset(Xval, yval), batch_size=args.batch_size)
                
                if tracker: tracker.start_phase(f"client_{cname}_clf")
                clf, steps = train_classifier(SimpleCNN(chans, num_classes), tr_ld, val_ld, device, args.clf_epochs, hist, cname, tracker=tracker)
                if tracker: tracker.end_phase(f"client_{cname}_clf")
                
                trained_clfs.append(clf)
                classifier_steps += steps
                torch.save(clf.state_dict(), P.root / "models" / "classifiers" / f"client-{cname}.pt")

            if tracker: tracker.start_phase("ensemble_evaluation")
            y_true, y_pred = ensemble_preds_poexp(trained_clfs, reserved_test_ld, device)
            if tracker: tracker.end_phase("ensemble_evaluation")

        else:
            # Case: Server-Side Inference with Skewed Data
            # Server trains a central classifier purely on synthetic data aggregated from clients
            if tracker: tracker.start_phase("server_generation")
            target = args.samples_per_class if args.samples_per_class > 0 else 2000
            
            if getattr(args, "aggregation", "simple") == "weighted":
                 synth_samples = generate_weighted_samples(client_gen_models, client_sample_counts, target, args.model, args.latent_dim, device, img_shape)
            else:
                 synth_samples = {}
                 for cid in range(num_classes):
                     if cid < len(gen_models) and gen_models[cid]:
                         if args.model == "vae": synth_samples[cid] = synth_from_decoder(gen_models[cid], args.latent_dim, target, device)
                         else: synth_samples[cid] = synth_from_diffusion(gen_models[cid], target, device, img_shape)
            
            if tracker: tracker.end_phase("server_generation")

            xs, ys = [], []
            for cid, imgs in synth_samples.items():
                if len(imgs) > 0:
                    xs.append(imgs)
                    ys.append(torch.full((len(imgs),), cid, dtype=torch.long))
                    synth_images_total += len(imgs)
            
            if xs:
                X, y = torch.cat(xs), torch.cat(ys)
                
                # --- Label Remapping Logic ---
                # Handle cases where synthetic data does not cover all classes 0 to K-1.
                # We map present classes to a contiguous range 0..N for training.
                present_in_synth = sorted(list(set(y.tolist())))
                label_map = {old: new for new, old in enumerate(present_in_synth)}
                y_mapped = torch.tensor([label_map[v.item()] for v in y], dtype=torch.long)
                
                Xtr, Xval, ytr, yval = train_test_split(X, y_mapped, test_size=0.1, stratify=y_mapped, random_state=42)
                tr_ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
                val_ld = DataLoader(TensorDataset(Xval, yval), batch_size=args.batch_size)

                if tracker: tracker.start_phase("server_classifier")
                # Train the classifier on the reduced/mapped label set
                clf, steps = train_classifier(SimpleCNN(chans, len(present_in_synth)), tr_ld, val_ld, device, args.clf_epochs, hist, "server", tracker=tracker)
                if tracker: tracker.end_phase("server_classifier")
                
                classifier_steps += steps
                torch.save(clf.state_dict(), P.root / "models" / "classifiers" / "central.pt")
                single_clf = clf
                
                # --- Evaluation with Remapping ---
                if tracker: tracker.start_phase("server_evaluation")
                
                # Filter the test set to only include classes the model was trained on
                all_imgs, all_lbls = [], []
                for x, lbl in reserved_test_ld:
                    all_imgs.append(x)
                    all_lbls.append(lbl)
                all_imgs = torch.cat(all_imgs)
                all_lbls = torch.cat(all_lbls)
                
                mask = torch.tensor([l.item() in label_map for l in all_lbls], dtype=torch.bool)
                if mask.sum() > 0:
                    filt_imgs = all_imgs[mask]
                    # Map test labels to the model's temporary label space
                    filt_lbls = torch.tensor([label_map[l.item()] for l in all_lbls[mask]], dtype=torch.long)
                    test_ld_mapped = DataLoader(TensorDataset(filt_imgs, filt_lbls), batch_size=args.batch_size)
                    
                    # Evaluate
                    yt_map, yp_map = evaluate_single_classifier(clf, test_ld_mapped, device)
                    
                    # Reverse map the predictions back to original class IDs
                    rev_map = {v: k for k, v in label_map.items()}
                    y_true = np.array([rev_map[v] for v in yt_map])
                    y_pred = np.array([rev_map[v] for v in yp_map])
                
                if tracker: tracker.end_phase("server_evaluation")
    
    # --------------------------------------------------
    # PARTITION MODE: SILOS
    # --------------------------------------------------
    else:
        if is_local:
            # Case: Local Inference within Data Silos
            # Each silo trains for its specific class 'd', augmenting with synthetic data for others
            for i, d in enumerate(present_classes):
                from jobs.baseline_runner import subset_to_tensor
                real_imgs = subset_to_tensor(train_subsets[i])
                
                n_synth = 2000 # Legacy default count
                if args.samples_per_class > 0: n_synth = args.samples_per_class
                
                # --- Build Hybrid Dataset ---
                s_xs, s_ys = [], []
                for od, gm in enumerate(gen_models):
                    if od == d or gm is None: continue
                    if args.model == "vae": im = synth_from_decoder(gm, args.latent_dim, n_synth, device)
                    else: im = synth_from_diffusion(gm, n_synth, device, img_shape)
                    s_xs.append(im); s_ys.append(torch.full((len(im),), od, dtype=torch.long))
                
                X = torch.cat([real_imgs, *s_xs]) if s_xs else real_imgs
                y = torch.cat([torch.full((len(real_imgs),), d, dtype=torch.long), *s_ys]) if s_ys else torch.full((len(real_imgs),), d, dtype=torch.long)
                synth_images_total += sum(len(x) for x in s_xs)
                
                Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                tr_ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
                val_ld = DataLoader(TensorDataset(Xval, yval), batch_size=args.batch_size)

                if tracker: tracker.start_phase(f"client_{d:03d}_clf")
                clf, steps = train_classifier(SimpleCNN(chans, num_classes), tr_ld, val_ld, device, args.clf_epochs, hist, d, tracker=tracker)
                if tracker: tracker.end_phase(f"client_{d:03d}_clf")
                
                trained_clfs.append(clf)
                classifier_steps += steps
                torch.save(clf.state_dict(), P.root / "models" / "classifiers" / f"client-{d:03d}.pt")

            if tracker: tracker.start_phase("ensemble_evaluation")
            y_true, y_pred = ensemble_preds_poexp(trained_clfs, reserved_test_ld, device)
            if tracker: tracker.end_phase("ensemble_evaluation")

        else:
            # Case: Server Inference Aggregating Silos
            # Server generates a balanced synthetic dataset covering all classes found in silos
            if tracker: tracker.start_phase("server_generation")
            # --- INIZIO CORREZIONE ---
            if args.samples_per_class > 0:
                n_synth = args.samples_per_class
            else:
                # Logica dinamica dal monolitico: usa il massimo numero di campioni reali trovati nei silos
                # Se train_subsets è vuoto (caso limite), fallback a 1000
                n_synth = max(len(s) for s in train_subsets) if train_subsets else 1000
            # --- FINE CORREZIONE ---
            xs, ys = [], []
            for d, gm in zip(present_classes, gen_models):
                if args.model == "vae": im = synth_from_decoder(gm, args.latent_dim, n_synth, device)
                else: im = synth_from_diffusion(gm, n_synth, device, img_shape)
                xs.append(im); ys.append(torch.full((len(im),), d, dtype=torch.long))
            
            X, y = torch.cat(xs), torch.cat(ys)
            synth_images_total = len(X)
            if tracker: tracker.end_phase("server_generation")
            
            Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
            tr_ld = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
            val_ld = DataLoader(TensorDataset(Xval, yval), batch_size=args.batch_size)

            if tracker: tracker.start_phase("server_classifier")
            clf, steps = train_classifier(SimpleCNN(chans, num_classes), tr_ld, val_ld, device, args.clf_epochs, hist, "server", tracker=tracker)
            if tracker: tracker.end_phase("server_classifier")
            
            classifier_steps += steps
            torch.save(clf.state_dict(), P.root / "models" / "classifiers" / "central.pt")
            single_clf = clf

            if tracker: tracker.start_phase("server_evaluation")
            y_true, y_pred = evaluate_single_classifier(clf, reserved_test_ld, device)
            if tracker: tracker.end_phase("server_evaluation")

    return classifier_steps, synth_images_total, clf_start, y_true, y_pred, trained_clfs, single_clf
