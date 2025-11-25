"""
🤖 Generative Model Training Orchestration Module
-----------------------------------------------

This module serves as the central orchestration engine for training, evaluating, 
and aggregating generative models (VAE and DiT) within a federated or distributed 
learning simulation.

🧠 Purpose:
    It governs the lifecycle of generative model training across simulated clients, 
    handling data partitioning strategies (Silos vs. Non-IID partitions), local 
    training loops, model aggregation (Federated Averaging), and network cost 
    accounting.

🔧 Core Functionalities:
    • Orchestrate local training for Variational Autoencoders (VAE) and Diffusion Transformers (DiT)
    • Manage data loading with optional noise injection for robustness experiments
    • execute simple parameter aggregation (FedAvg) across client models
    • Persist model checkpoints and generate visual sampling artifacts
    • Simulate and log network bandwidth usage (upload/download) for performance analysis

🎯 Intended Use:
    • Academic research on Federated Learning efficiency and privacy
    • Benchmarking generative models in distributed environments
    • Simulation of resource-constrained edge computing scenarios

📁 Dependencies:
    • torch (PyTorch)
    • models.vae / models.diffusion
    • utils (visualization and metrics)
    • jobs.experiment_setup

📝 Notes:
    The module assumes a specific directory structure for artifacts and relies on 
    an external `tracker` object for experimental metric logging.

Author: Andrea Moleri
File Location: src/jobs/generative_phase.py
Last Modified: 21/11/2025
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from imports.data_augmentation import NoisyCleanDataset
from models.trainers import train_vae, train_diffusion
from models.vae import VAE, Decoder
from models.diffusion import DiT, DiffusionConfig, rectified_flow_sampler
from utils import sample_grid, grid_from_tensors, decoder_size_mb
import matplotlib.pyplot as plt
from jobs.experiment_setup import _export_client_class_distribution

logger = logging.getLogger(__name__)

def _module_size_mb(m: nn.Module) -> float:
    """
    Calculates the total size of the model parameters in Megabytes (MB).

    Args:
        m (nn.Module): The PyTorch module to evaluate.

    Returns:
        float: The size of the model in MB.
    """
    return float(sum(p.numel() * p.element_size() for p in m.parameters()) / 1_000_000.0)

def sample_grid_diffusion(model: DiT, out_path: Path, n: int, device: torch.device, img_shape: Tuple[int, int, int], steps: int = 50):
    """
    Generates a grid of synthetic images using a trained Diffusion Transformer (DiT) 
    and saves the result to disk.

    Args:
        model (DiT): The trained diffusion model.
        out_path (Path): The filesystem path where the generated image grid will be saved.
        n (int): The number of samples to generate.
        device (torch.device): The computation device (CPU or CUDA).
        img_shape (Tuple[int, int, int]): The dimensions of the target image (Channels, Height, Width).
        steps (int, optional): The number of integration steps for the flow matching sampler. Defaults to 50.

    Returns:
        None
    """
    C, H, W = img_shape
    model.to(device).eval()
    
    # Generate samples using the rectified flow ODE solver
    imgs = rectified_flow_sampler(model, n=n, shape=(C, H, W), steps=steps, device=device).cpu()
    model.cpu()

    # Denormalize pixel values from [-1, 1] to [0, 1] for visualization
    imgs = (imgs + 1.0) / 2.0
    imgs = imgs.clamp(0.0, 1.0)

    # Create a grid visualization from the first 64 samples
    grid_img = grid_from_tensors(imgs[:64])
    plt.imsave(out_path, grid_img)

def aggregate_models_simple(client_models: Dict[str, Dict[int, torch.nn.Module]]) -> Dict[int, torch.nn.Module]:
    """
    Performs simple Federated Averaging (FedAvg) on a collection of client models, 
    grouping them by class label.

    This function computes the arithmetic mean of the state dictionaries of all 
    models associated with a specific class ID across different clients.

    Args:
        client_models (Dict[str, Dict[int, torch.nn.Module]]): A nested dictionary 
            mapping client IDs to a dictionary of class-specific models.

    Returns:
        Dict[int, torch.nn.Module]: A dictionary mapping class IDs to the aggregated 
            global model for that class.
    """
    logger.info("[SIMPLE AGGREGATION] Starting simple aggregation")

    # Group models by their target class ID
    class_models: Dict[int, List[torch.nn.Module]] = {}
    for _, models_dict in client_models.items():
        for class_id, model in models_dict.items():
            class_models.setdefault(class_id, []).append(model)

    aggregated: Dict[int, torch.nn.Module] = {}

    for class_id, models in class_models.items():
        if not models:
            continue

        logger.info(f"[SIMPLE AGGREGATION] Aggregating class {class_id} from {len(models)} models")

        # Initialize the averaged state dictionary with the structure of the first model
        avg_sd = models[0].state_dict().copy()

        # Iterate through all parameters and compute the average across models
        for k in avg_sd:
            # Only average floating point parameters; preserve integers (e.g., version counters)
            if avg_sd[k].dtype in [torch.float32, torch.float64, torch.float16]:
                s = torch.zeros_like(avg_sd[k])
                for m in models:
                    s += m.state_dict()[k]
                avg_sd[k] = s / len(models)

        tmpl = models[0]

        # -------------------- caso VAE Decoder --------------------
        if isinstance(tmpl, Decoder):
            agg_model = Decoder(
                tmpl.latent_dim,
                tmpl.output_channels,
                tmpl.hidden_dims,
            )
            logger.info(f"[SIMPLE AGGREGATION] Created VAE Decoder for class {class_id}")

        # -------------------- caso DiT (diffusion) ----------------
        elif isinstance(tmpl, DiT) or hasattr(tmpl, "config") or hasattr(tmpl, "in_ch"):
            try:
                # Se il modello ha già una config, usala direttamente
                if hasattr(tmpl, "config"):
                    cfg = tmpl.config
                else:
                    # Altrimenti ricostruisci DiffusionConfig dagli attributi del modello
                    cfg = DiffusionConfig(
                        in_ch=getattr(tmpl, "in_ch", 1),
                        embed_dim=getattr(tmpl, "embed_dim", 256),
                        depth=getattr(tmpl, "depth", 8),
                        num_heads=getattr(tmpl, "num_heads", 8),
                        mlp_ratio=getattr(tmpl, "mlp_ratio", 4.0),
                        patch_size=getattr(tmpl, "patch_size", 2),
                    )
                agg_model = DiT(cfg)
                logger.info(f"[SIMPLE AGGREGATION] Created DiT model for class {class_id}")
            except Exception as e:
                logger.warning(f"[SIMPLE AGGREGATION] Could not create DiT model for class {class_id}: {e}")
                # Fallback: prova a istanziare il tipo “nudo”
                try:
                    agg_model = type(tmpl)()
                except Exception:
                    # Ultima spiaggia: usa direttamente il primo modello
                    agg_model = tmpl

        # -------------------- altri tipi generici -----------------
        else:
            try:
                agg_model = type(tmpl)()
                logger.info(f"[SIMPLE AGGREGATION] Created generic model for class {class_id}")
            except Exception as e:
                logger.warning(f"[SIMPLE AGGREGATION] Could not create model for class {class_id}: {e}")
                agg_model = tmpl  # fallback

        # Carica lo state_dict medio dentro il modello aggregato
        try:
            agg_model.load_state_dict(avg_sd)
            logger.info(f"[SIMPLE AGGREGATION] Successfully loaded averaged state dict for class {class_id}")
        except Exception as e:
            logger.warning(f"[SIMPLE AGGREGATION] Could not load state dict for class {class_id}: {e}")
            agg_model = tmpl  # fallback: usa il modello originale

        aggregated[class_id] = agg_model

    return aggregated


def run_generative_training(
    args: Any, 
    device: torch.device, 
    P: Any, 
    train_subsets_dict: Dict[str, Dict[int, Any]], 
    train_subsets: List[Any], 
    present_classes: List[int], 
    num_classes: int, 
    chans: int, 
    img_shape: Tuple[int, int, int], 
    tracker: Any, 
    hist: Any, 
    perc_loss: nn.Module
) -> Tuple[List[Optional[nn.Module]], Dict[str, Dict[int, nn.Module]], Dict[str, Dict[int, int]], float, float]:
    """
    Executes the generative model training pipeline across simulated clients.

    This function handles the partitioning logic (Skew/Dirichlet vs. Silos), instantiates
    local training loops, manages checkpoints, performs model aggregation, and logs
    performance metrics.

    Args:
        args (Any): Configuration namespace containing hyperparameters (batch size, epochs, model type, etc.).
        device (torch.device): The hardware accelerator device.
        P (Any): Path configuration object containing root directories.
        train_subsets_dict (Dict[str, Dict[int, Any]]): Data partitions mapped by client and class (for Skew/Dirichlet).
        train_subsets (List[Any]): Data partitions list (for Silos).
        present_classes (List[int]): List of class IDs available in the current context.
        num_classes (int): Total number of classes in the dataset.
        chans (int): Number of image channels (e.g., 1 for grayscale, 3 for RGB).
        img_shape (Tuple[int, int, int]): Dimensions of the input images.
        tracker (Any): Experiment tracker object for logging metrics and phases.
        hist (Any): History object for accumulating loss/metric trends.
        perc_loss (nn.Module): Perceptual loss module (e.g., LPIPS) used for training VAEs.

    Returns:
        Tuple containing:
        - gen_models (List[Optional[nn.Module]]): List of aggregated or trained models indexed by class ID.
        - client_gen_models (Dict[str, Dict[int, nn.Module]]): Dictionary of models trained by specific clients.
        - client_sample_counts (Dict[str, Dict[int, int]]): Sample counts per client per class.
        - gen_step_total (float): Total number of optimization steps performed.
        - gen_start (float): Timestamp of when the training started.
    """
    gen_models: List[torch.nn.Module] = []
    client_gen_models: Dict[str, Dict[int, torch.nn.Module]] = {}
    client_sample_counts: Dict[str, Dict[int, int]] = {}
    gen_step_total = 0
    gen_start = time.perf_counter()

    partition_mode = getattr(args, "partition", "silos")

    # -------------------------------------------------------------------------
    # Logic for Complex Partitions (Skew / Dirichlet)
    # -------------------------------------------------------------------------
    if partition_mode in ["skew", "dirichlet"]:
        for client_name, class_subsets in train_subsets_dict.items():
            client_gen_models[client_name] = {}
            client_sample_counts[client_name] = {}

            for class_id, train_subset in class_subsets.items():
                logger.info(f"[{partition_mode.upper()}] Training client {client_name}, class {class_id}")
                
                # Configure Data Loading: Inject noise if training VAEs for denoising tasks
                if args.model == "vae":
                    ds_client = NoisyCleanDataset(train_subset, args.noise_std)
                    ld = DataLoader(ds_client, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
                else:
                    ld = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

                if tracker: tracker.start_phase(f"client_{client_name}_class_{class_id}_gen")

                # Branch: VAE Training
                if args.model == "vae":
                    vae, steps = train_vae(
                        VAE(chans, args.latent_dim), ld, device, args.epochs, hist, 
                        f"{client_name}_class_{class_id}", perc_loss, 
                        dp=args.dp, dp_clip=args.dp_clip, dp_noise_mult=args.dp_noise, dp_microbatch=args.dp_microbatch, tracker=tracker
                    )
                    model_to_save = vae.decoder
                    full_model = vae
                # Branch: Diffusion Transformer (DiT) Training
                else:
                    cfg = DiffusionConfig(in_ch=chans, embed_dim=int(args.dit_embed), depth=int(args.dit_depth), num_heads=int(args.dit_heads), mlp_ratio=4.0, patch_size=int(args.dit_patch))
                    dit, steps = train_diffusion(
                        DiT(cfg), ld, device, args.epochs, hist, f"{client_name}_class_{class_id}", 
                        dp=args.dp, dp_clip=args.dp_clip, dp_noise_mult=args.dp_noise, dp_microbatch=args.dp_microbatch, tracker=tracker
                    )
                    model_to_save = dit
                    full_model = dit

                if tracker: tracker.end_phase(f"client_{client_name}_class_{class_id}_gen")

                client_gen_models[client_name][class_id] = model_to_save
                client_sample_counts[client_name][class_id] = len(train_subset)

                # Persist model checkpoint to disk
                gen_ckpt = P.root / "models" / "generators" / f"client_{client_name}_class_{class_id:03d}.pt"
                torch.save(full_model.state_dict(), gen_ckpt)
                if tracker: tracker.record_artifact(gen_ckpt)

                # Generate and save visual samples for quality inspection
                out_png = P.root / "artifacts" / "samples" / f"client_{client_name}_class_{class_id:03d}.png"
                if args.model == "vae":
                    sample_grid(full_model, args.latent_dim, out_png)
                else:
                    sample_grid_diffusion(full_model, out_png, 64, device, img_shape)
                
                gen_step_total += steps

        # Aggregation Logic: Combine client models into a global model if configured
        if getattr(args, "aggregation", "simple") == "simple":
            aggregated = aggregate_models_simple(client_gen_models)
            gen_models = [aggregated.get(i, None) for i in range(num_classes)]
        else:
            gen_models = [None] * num_classes
            
        _export_client_class_distribution(P.root, client_sample_counts, num_classes)

    # -------------------------------------------------------------------------
    # Logic for Silos Partition (Isolated Clients)
    # -------------------------------------------------------------------------
    else: 
        silos_client_counts = {}
        for i, d in enumerate(present_classes):
            logger.info(f"[SILOS] Training client {d}")
            
            # In 'silos' mode, each client corresponds to a distinct class index 'd'
            if args.model == "vae":
                ds_client = NoisyCleanDataset(train_subsets[i], args.noise_std)
                ld = DataLoader(ds_client, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
            else:
                ld = DataLoader(train_subsets[i], batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
            
            if tracker: tracker.start_phase(f"client_{d:03d}_gen")

            if args.model == "vae":
                vae, steps = train_vae(
                    VAE(chans, args.latent_dim), ld, device, args.epochs, hist, d, perc_loss,
                    dp=args.dp, dp_clip=args.dp_clip, dp_noise_mult=args.dp_noise, dp_microbatch=args.dp_microbatch, tracker=tracker
                )
                model_to_save = vae.decoder
                full_model = vae
            else:
                cfg = DiffusionConfig(in_ch=chans, embed_dim=int(args.dit_embed), depth=int(args.dit_depth), num_heads=int(args.dit_heads), mlp_ratio=4.0, patch_size=int(args.dit_patch))
                dit, steps = train_diffusion(
                    DiT(cfg), ld, device, args.epochs, hist, d,
                    dp=args.dp, dp_clip=args.dp_clip, dp_noise_mult=args.dp_noise, dp_microbatch=args.dp_microbatch, tracker=tracker
                )
                model_to_save = dit
                full_model = dit

            if tracker: tracker.end_phase(f"client_{d:03d}_gen")
            
            gen_models.append(model_to_save)
            gen_step_total += steps

            # Persist model checkpoint
            gen_ckpt = P.root / "models" / "generators" / f"class-{d:03d}.pt"
            torch.save(full_model.state_dict(), gen_ckpt)
            if tracker: tracker.record_artifact(gen_ckpt)
            
            # Generate samples
            out_png = P.root / "artifacts" / "samples" / f"class-{d:03d}.png"
            if args.model == "vae": sample_grid(full_model, args.latent_dim, out_png)
            else: sample_grid_diffusion(full_model, out_png, 64, device, img_shape)
            
            silos_client_counts[f"client{d}"] = {d: len(train_subsets[i])}

        # Calculate and store model size metrics for analysis
        sz = _module_size_mb(gen_models[0])
        with open(P.root / "models" / "sizes.json", "w") as f:
            json.dump({"generator_mb": float(sz)}, f, indent=2)
        _export_client_class_distribution(P.root, silos_client_counts, num_classes)

    # ---------------------------------------------------------------
    # Network Accounting Logic
    # ---------------------------------------------------------------
    _handle_network_simulation(args, P, tracker, partition_mode, present_classes, client_gen_models)

    return gen_models, client_gen_models, client_sample_counts, gen_step_total, gen_start

def _handle_network_simulation(args: Any, P: Any, tracker: Any, partition_mode: str, present_classes: List[int], client_gen_models: Dict[str, Any]):
    """
    Simulates network bandwidth usage (upload/download) by analyzing file presence on disk.

    This function does not perform actual network IO but records 'virtual' transfers
    to the experiment tracker. It assumes that if a checkpoint file exists, it was 
    uploaded by the creator and potentially downloaded by peers or the server.

    Args:
        args (Any): Configuration namespace.
        P (Any): Path configuration object.
        tracker (Any): Experiment tracker for recording network events.
        partition_mode (str): The data partitioning strategy ("silos", "skew", etc.).
        present_classes (List[int]): List of class IDs present in the simulation.
        client_gen_models (Dict[str, Any]): Dictionary of trained models by client.

    Returns:
        None
    """
    if tracker is None: return
    
    # Map existing checkpoint files to their logical identifiers (client/class)
    gen_ckpts_map = {}
    if partition_mode in ["skew", "dirichlet"]:
        for p in (P.root / "models" / "generators").glob("client_*_class_*.pt"):
            try:
                parts = p.stem.split('_')
                gen_ckpts_map[(parts[1], int(parts[3]))] = p
            except: pass
    else:
        for p in (P.root / "models" / "generators").glob("class-*.pt"):
            try: gen_ckpts_map[int(p.stem.split("-")[1])] = p
            except: pass

    is_server = (args.infer_mode == "server")
    
    # -------------------------------------------------------------------------
    # Simulate Uploads (Client -> Server/Network)
    # -------------------------------------------------------------------------
    if partition_mode in ["skew", "dirichlet"]:
        for (cname, cid), p in gen_ckpts_map.items():
            if p.exists():
                ph = f"client_{cname}_class_{cid}_upload"
                tracker.start_phase(ph)
                tracker.note_network_file_transfer(p, "tx", phase=ph)
                tracker.end_phase(ph)
                if is_server: # Server ingest simulation
                    tracker.note_network_file_transfer(p, "rx", phase="server_ingest")
    else:
        for d in present_classes:
            p = gen_ckpts_map.get(d)
            if p and p.exists():
                ph = f"client_{d:03d}_upload"
                tracker.start_phase(ph)
                tracker.note_network_file_transfer(p, "tx", phase=ph)
                tracker.end_phase(ph)
                if is_server:
                    tracker.note_network_file_transfer(p, "rx", phase="server_ingest")

    # -------------------------------------------------------------------------
    # Simulate Downloads (Server -> Client / Peer -> Peer)
    # -------------------------------------------------------------------------
    # Only perform download accounting if not running in server mode 
    # (i.e., clients downloading models from others in a P2P or Federated manner)
    if not is_server:
        if partition_mode in ["skew", "dirichlet"]:
            for cname in client_gen_models.keys():
                ph = f"client_{cname}_download"
                tracker.start_phase(ph)
                for (src, cid), p in gen_ckpts_map.items():
                    # Download logic: If the model does not belong to me, and I do not
                    # already possess a model for this class, I need to download it.
                    if src != cname and cid not in client_gen_models.get(cname, {}):
                        if p.exists(): tracker.note_network_file_transfer(p, "rx", phase=ph)
                tracker.end_phase(ph)
        else:
            for target in present_classes:
                ph = f"client_{target:03d}_download"
                tracker.start_phase(ph)
                for src, p in gen_ckpts_map.items():
                    if src != target and p.exists():
                        tracker.note_network_file_transfer(p, "rx", phase=ph)
                tracker.end_phase(ph)
