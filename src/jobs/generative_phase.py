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
    • Apply scientifically grounded Data Augmentation policies (NeurIPS Standard)
    • Execute simple parameter aggregation (FedAvg) across client models
    • Persist model checkpoints and generate visual sampling artifacts
    • Simulate and log network bandwidth usage (upload/download) for performance analysis

🎯 Intended Use:
    • Academic research on Federated Learning efficiency and privacy
    • Benchmarking generative models in distributed environments
    • Simulation of resource-constrained edge computing scenarios

📁 Dependencies:
    • torch (PyTorch)
    • models.vae / models.diffusion
    • imports.data_augmentation (Scientific Transforms)
    • jobs.experiment_setup

📝 Notes:
    The module forces strict augmentation policies defined in `data_augmentation.py`
    to ensure generative models learn robust features rather than memorizing raw data.

Author: Andrea Moleri
File Location: src/jobs/generative_phase.py
Last Modified: 08/12/2025
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import json
import re  # Added for resolution inference

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Updated Import: Use the centralized transformation factory
from imports.data_augmentation import build_transform
from models.trainers import train_vae, train_diffusion
from models.vae import VAE, Decoder
from models.diffusion import DiT, DiffusionConfig, rectified_flow_sampler
from utils import sample_grid, grid_from_tensors
import matplotlib.pyplot as plt
from jobs.experiment_setup import _export_client_class_distribution

logger = logging.getLogger(__name__)

# ============================ Utilities ==================================

class TransformedSubset(Dataset):
    """
    A Dataset wrapper that applies a transformation pipeline dynamically upon access.
    """
    def __init__(self, subset: Any, transform: Callable):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def _module_size_mb(m: nn.Module) -> float:
    """Calculate the memory footprint of a PyTorch module's parameters."""
    return float(sum(p.numel() * p.element_size() for p in m.parameters()) / 1_000_000.0)

def _get_diffusion_config(args, chans, img_shape):
    """
    Helper to construct the Diffusion Config ensuring args.latent_dim is respected.
    This fixes the ISIC dimension mismatch (64 vs 128).
    """
    return DiffusionConfig(
        in_ch=chans,
        # [CRITICAL FIX] Use args.latent_dim (e.g. 64) explicitly.
        # Do not allow default fallback to 128.
        embed_dim=int(args.latent_dim),
        img_resolution=img_shape[-1],
        num_classes=0,  # Unconditional per client/class silo
        dropout=0.3,    # Match training dropout
        channel_mult=[1, 2, 2, 2] # Match EDM2 default architecture
    )

def sample_grid_diffusion(model: DiT, out_path: Path, n: int, device: torch.device, img_shape: Tuple[int, int, int], steps: int = 50):
    """
    Generate and save a grid of synthetic images using a Diffusion Transformer (DiT).
    """
    C, H, W = img_shape
    model.to(device).eval()

    # Execute the ordinary differential equation (ODE) solver for sampling
    imgs = rectified_flow_sampler(model, n=n, shape=(C, H, W), steps=steps, device=device).cpu()
    model.cpu()

    # Denormalize the pixel values from the range [-1, 1] to [0, 1] for visualization
    imgs = (imgs + 1.0) / 2.0
    imgs = imgs.clamp(0.0, 1.0)

    # Organize the tensor batch into a visual grid and save to disk
    grid_img = grid_from_tensors(imgs[:64])
    plt.imsave(out_path, grid_img)

def aggregate_models_simple(client_models: Dict[str, Dict[int, torch.nn.Module]]) -> Dict[int, torch.nn.Module]:
    """
    Perform Federated Averaging (FedAvg) on a collection of client models.
    """
    logger.info("[SIMPLE AGGREGATION] Starting simple aggregation")

    # Group models by target class ID across all clients
    class_models: Dict[int, List[torch.nn.Module]] = {}
    for _, models_dict in client_models.items():
        for class_id, model in models_dict.items():
            class_models.setdefault(class_id, []).append(model)

    aggregated: Dict[int, torch.nn.Module] = {}

    for class_id, models in class_models.items():
        if not models: continue

        logger.info(f"[SIMPLE AGGREGATION] Aggregating class {class_id} from {len(models)} models")

        # Initialize the averaged state dictionary with the structure of the first model
        avg_sd = models[0].state_dict().copy()

        # Iterate through all parameters and compute the element-wise mean
        for k in avg_sd:
            if avg_sd[k].dtype in [torch.float32, torch.float64, torch.float16]:
                s = torch.zeros_like(avg_sd[k])
                for m in models: s += m.state_dict()[k]
                avg_sd[k] = s / len(models)

        tmpl = models[0]

        # Reconstruct the specific model architecture to host the aggregated weights
        if isinstance(tmpl, Decoder):
            agg_model = Decoder(tmpl.latent_dim, tmpl.output_channels, tmpl.hidden_dims)
        elif isinstance(tmpl, DiT) or hasattr(tmpl, "cfg"):
            # Use configuration from template
            cfg = getattr(tmpl, "cfg", None)
            if cfg is None:
                # Fallback if cfg is missing (unlikely with new helper)
                cfg = DiffusionConfig(
                    in_ch=getattr(tmpl, "in_ch", 3),
                    embed_dim=getattr(tmpl, "embed_dim", 64)
                )
            agg_model = DiT(cfg)
        else:
            agg_model = type(tmpl)()

        agg_model.load_state_dict(avg_sd)
        aggregated[class_id] = agg_model

    return aggregated

# ============================ Main Training Orchestrator ========================

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
    Execute the generative model training phase across all simulated clients.
    """
    gen_models: List[torch.nn.Module] = []
    client_gen_models: Dict[str, Dict[int, torch.nn.Module]] = {}
    client_sample_counts: Dict[str, Dict[int, int]] = {}
    gen_step_total = 0
    gen_start = time.perf_counter()

    partition_mode = getattr(args, "partition", "silos")

    # -------------------------------------------------------------------------
    # PART A: Complex Partitions (Skew / Dirichlet)
    # -------------------------------------------------------------------------
    if partition_mode in ["skew", "dirichlet"]:
        for client_name, class_subsets in train_subsets_dict.items():
            client_gen_models[client_name] = {}
            client_sample_counts[client_name] = {}

            for class_id, train_subset in class_subsets.items():
                logger.info(f"[{partition_mode.upper()}] Training client {client_name}, class {class_id}")

                robust_transform = build_transform(args.dataset, train=True, robustness=True)
                ds_client = TransformedSubset(train_subset, robust_transform)

                ld = DataLoader(ds_client, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, pin_memory=True)

                if tracker: tracker.start_phase(f"client_{client_name}_class_{class_id}_gen")

                if args.model == "vae":
                    vae, steps = train_vae(
                        VAE(chans, args.latent_dim), ld, device, args.epochs, hist,
                        f"{client_name}_class_{class_id}", perc_loss,
                        dp=args.dp, dp_clip=args.dp_clip, dp_noise_mult=args.dp_noise, dp_microbatch=args.dp_microbatch, tracker=tracker
                    )
                    model_to_save = vae.decoder
                    full_model = vae
                else:
                    # [FIX] Use centralized helper
                    cfg = _get_diffusion_config(args, chans, img_shape)

                    dit, steps = train_diffusion(
                        DiT(cfg), ld, device, args.epochs, hist, f"{client_name}_class_{class_id}",
                        dp=args.dp, dp_clip=args.dp_clip, dp_noise_mult=args.dp_noise, dp_microbatch=args.dp_microbatch,
                        tracker=tracker,
                        checkpoint_every=args.checkpoint_every,
                        checkpoint_dir=P.root / "checkpoints"
                    )
                    model_to_save = dit
                    full_model = dit

                if tracker: tracker.end_phase(f"client_{client_name}_class_{class_id}_gen")

                client_gen_models[client_name][class_id] = model_to_save
                client_sample_counts[client_name][class_id] = len(train_subset)

                # Persist checkpoints
                gen_ckpt = P.root / "models" / "generators" / f"client_{client_name}_class_{class_id:03d}.pt"
                torch.save(full_model.state_dict(), gen_ckpt)
                if tracker: tracker.record_artifact(gen_ckpt)

                out_png = P.root / "artifacts" / "samples" / f"client_{client_name}_class_{class_id:03d}.png"
                if args.model == "vae": sample_grid(full_model, args.latent_dim, out_png)
                else: sample_grid_diffusion(full_model, out_png, 64, device, img_shape)

                gen_step_total += steps

        if getattr(args, "aggregation", "simple") == "simple":
            aggregated = aggregate_models_simple(client_gen_models)
            gen_models = [aggregated.get(i, None) for i in range(num_classes)]
        else:
            gen_models = [None] * num_classes

        _export_client_class_distribution(P.root, client_sample_counts, num_classes)

    # -------------------------------------------------------------------------
    # PART B: Silos Partition (Isolated Clients)
    # -------------------------------------------------------------------------
    else:
        silos_client_counts = {}
        for i, d in enumerate(present_classes):
            logger.info(f"[SILOS] Training client {d}")

            robust_transform = build_transform(args.dataset, train=True, robustness=True)
            ds_client = TransformedSubset(train_subsets[i], robust_transform)

            ld = DataLoader(ds_client, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)

            if tracker: tracker.start_phase(f"client_{d:03d}_gen")

            if args.model == "vae":
                vae, steps = train_vae(
                    VAE(chans, args.latent_dim), ld, device, args.epochs, hist, d, perc_loss,
                    dp=args.dp, dp_clip=args.dp_clip, dp_noise_mult=args.dp_noise, dp_microbatch=args.dp_microbatch, tracker=tracker
                )
                model_to_save = vae.decoder
                full_model = vae
            else:
                # [FIX] Use centralized helper
                cfg = _get_diffusion_config(args, chans, img_shape)

                dit, steps = train_diffusion(
                    DiT(cfg), ld, device, args.epochs, hist, d,
                    dp=args.dp, dp_clip=args.dp_clip, dp_noise_mult=args.dp_noise, dp_microbatch=args.dp_microbatch,
                    tracker=tracker,
                    checkpoint_every=args.checkpoint_every,
                    checkpoint_dir=P.root / "checkpoints"
                )
                model_to_save = dit
                full_model = dit

            if tracker: tracker.end_phase(f"client_{d:03d}_gen")

            gen_models.append(model_to_save)
            gen_step_total += steps

            gen_ckpt = P.root / "models" / "generators" / f"class-{d:03d}.pt"
            torch.save(full_model.state_dict(), gen_ckpt)
            if tracker: tracker.record_artifact(gen_ckpt)

            out_png = P.root / "artifacts" / "samples" / f"class-{d:03d}.png"
            if args.model == "vae": sample_grid(full_model, args.latent_dim, out_png)
            else: sample_grid_diffusion(full_model, out_png, 64, device, img_shape)

            silos_client_counts[f"client{d}"] = {d: len(train_subsets[i])}

        sz = _module_size_mb(gen_models[0])
        with open(P.root / "models" / "sizes.json", "w") as f:
            json.dump({"generator_mb": float(sz)}, f, indent=2)
        _export_client_class_distribution(P.root, silos_client_counts, num_classes)

    _handle_network_simulation(args, P, tracker, partition_mode, present_classes, client_gen_models)

    return gen_models, client_gen_models, client_sample_counts, gen_step_total, gen_start


def _infer_resolution_from_weights(state_dict: Dict[str, Any]) -> int:
    """
    Scans state_dict keys to find the highest resolution layer (e.g., '128x128').
    Used to adapt the model architecture to the specific checkpoint being loaded.
    """
    max_res = 32
    # Pattern looks for "128x128" inside keys like "unet.enc.128x128_block..."
    pattern = re.compile(r"(\d+)x(\d+)")
    
    # EDM2 models might be wrapped in 'ema', 'net' or direct.
    # The keys passed here are expected to be the direct state_dict of the Precond/DiT.
    for k in state_dict.keys():
        match = pattern.search(k)
        if match:
            # e.g., match groups ('128', '128')
            res = int(match.group(1))
            if res > max_res:
                max_res = res
    return max_res


def load_generative_checkpoints(
        args,
        device: torch.device,
        P: Any,
        train_subsets_dict: Dict[str, Dict[int, Any]],
        present_classes: List[int],
        chans: int,
        img_shape: Tuple[int, int, int],
        epoch_family: int
) -> Tuple[List[Optional[Any]], Dict[str, Dict[int, Any]], Dict[str, Dict[int, int]]]:
    """
    Loads pre-trained generative models from disk instead of training them.

    [CRITICAL FIXES]:
    1. Uses args.latent_dim explicitly (fixes 64 vs 128 channel mismatch).
    2. Handles EDM2 checkpoint structure (nested dictionaries).
    3. Handles client naming mismatches (client_0 vs client0).
    4. [NEW] Auto-detects checkpoint resolution to resolve 128 vs 224 mismatches.
    """
    logger.info(f"[GEN-LOAD] Loading checkpoints for Family: {epoch_family}")

    # Initialize containers
    max_class_id = max(present_classes) if present_classes else 0
    gen_models = [None] * (max_class_id + 1)

    client_gen_models: Dict[str, Dict[int, Any]] = {}
    client_sample_counts: Dict[str, Dict[int, int]] = {}

    # Define base path for checkpoints
    # Note: Phase 2 usually uses 'silos' or specific partitions generated in Phase 1
    if getattr(args, "partition", "silos") == "dirichlet" and getattr(args, "alpha", None) is not None:
        part_name = f"dirichlet_{args.alpha}"
    else:
        part_name = getattr(args, "partition", "silos")

    # Safe dataset name normalization (e.g. medmnist:bloodmnist -> medmnist_bloodmnist)
    safe_ds = args.dataset.replace(":", "_")

    project_root = Path(".").resolve()
    ckpt_root = project_root / "checkpoints" / safe_ds / part_name

    if not ckpt_root.exists():
        # Fallback for different CWD
        ckpt_root = Path("checkpoints") / safe_ds / part_name
        if not ckpt_root.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_root}")

    loaded_count = 0

    for client_name, class_subsets in train_subsets_dict.items():
        client_gen_models[client_name] = {}
        client_sample_counts[client_name] = {}

        # Handle naming divergence: "client_0" (code) vs "client_0" (folder)
        client_folder = ckpt_root / client_name
        if not client_folder.exists():
            # Fallback: try adding/removing underscore
            alt_name = client_name.replace("client_", "client") if "client_" in client_name else client_name.replace("client", "client_")
            client_folder = ckpt_root / alt_name

        for class_id, subset in class_subsets.items():
            client_sample_counts[client_name][class_id] = len(subset)

            # Format: training-state-0025001.pt
            epoch_str = f"{int(epoch_family):07d}"
            fname = f"training-state-{epoch_str}.pt"

            # Checkpoints are usually nested in "class_X" subfolders
            ckpt_path = client_folder / f"class_{class_id}" / fname

            if not ckpt_path.exists():
                logger.warning(f"⚠️ Checkpoint missing for {client_name} Class {class_id} at {ckpt_path}")
                continue

            try:
                # 1. Load Data to CPU first to inspect structure
                data = torch.load(ckpt_path, map_location="cpu")

                weights = None
                final_model = None

                if args.model == "diffusion":
                    # Handle EDM2/Standard structure
                    if 'ema' in data and isinstance(data['ema'], dict) and 'emas' in data['ema']:
                         weights = data['ema']['emas'][0]
                    elif 'net' in data:
                         weights = data['net']
                    else:
                         weights = data

                    # 2. AUTO-DETECT RESOLUTION
                    # We scan the keys in the weights to find the max resolution layer
                    detected_res = _infer_resolution_from_weights(weights)
                    
                    if detected_res != img_shape[-1]:
                        logger.info(f"   [Auto-Detect] Checkpoint Class {class_id} is {detected_res}x{detected_res} (Config was {img_shape[-1]}). Adapting model...")

                    # 3. Configure Model with DETECTED resolution
                    cfg = _get_diffusion_config(args, chans, img_shape)
                    cfg.img_resolution = detected_res # Override config
                    
                    model = DiT(cfg).to(device)
                    model.edm_net.load_state_dict(weights)
                    model.eval()
                    final_model = model

                elif args.model == "vae":
                    model = VAE(chans, args.latent_dim).to(device)
                    model.load_state_dict(data)
                    model.eval()
                    final_model = model.decoder
                else:
                    raise ValueError(f"Unknown generative model type: {args.model}")

                # Store references
                gen_models[class_id] = final_model
                client_gen_models[client_name][class_id] = final_model
                loaded_count += 1

            except Exception as e:
                logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
                continue

    logger.info(f"[GEN-LOAD] Successfully loaded {loaded_count} client models.")
    return gen_models, client_gen_models, client_sample_counts

def _handle_network_simulation(args: Any, P: Any, tracker: Any, partition_mode: str, present_classes: List[int], client_gen_models: Dict[str, Any]):
    """
    Simulate and log network bandwidth usage based on generated file artifacts.
    """
    if tracker is None: return

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

    # Simulate Uploads
    if partition_mode in ["skew", "dirichlet"]:
        for (cname, cid), p in gen_ckpts_map.items():
            if p.exists():
                ph = f"client_{cname}_class_{cid}_upload"
                tracker.start_phase(ph)
                tracker.note_network_file_transfer(p, "tx", phase=ph)
                tracker.end_phase(ph)
                if is_server: tracker.note_network_file_transfer(p, "rx", phase="server_ingest")
    else:
        for d in present_classes:
            p = gen_ckpts_map.get(d)
            if p and p.exists():
                ph = f"client_{d:03d}_upload"
                tracker.start_phase(ph)
                tracker.note_network_file_transfer(p, "tx", phase=ph)
                tracker.end_phase(ph)
                if is_server: tracker.note_network_file_transfer(p, "rx", phase="server_ingest")

    # Simulate Downloads
    if not is_server:
        if partition_mode in ["skew", "dirichlet"]:
            for cname in client_gen_models.keys():
                ph = f"client_{cname}_download"
                tracker.start_phase(ph)
                for (src, cid), p in gen_ckpts_map.items():
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
