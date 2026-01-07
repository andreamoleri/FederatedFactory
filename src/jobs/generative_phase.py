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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Updated Import: Use the centralized transformation factory
from imports.data_augmentation import build_transform
from models.trainers import train_vae, train_diffusion
from models.vae import VAE, Decoder
from models.diffusion import DiT, DiffusionConfig, rectified_flow_sampler
from utils import sample_grid, grid_from_tensors, decoder_size_mb
import matplotlib.pyplot as plt
from jobs.experiment_setup import _export_client_class_distribution

logger = logging.getLogger(__name__)

# ============================ Utilities ==================================

class TransformedSubset(Dataset):
    """
    A Dataset wrapper that applies a transformation pipeline dynamically upon access.

    This class decouples data storage from data augmentation, allowing
    computational transformations (e.g., normalization, robust augmentation)
    to be applied lazily during the training loop.
    """
    def __init__(self, subset: Any, transform: Callable):
        """
        Initialize the TransformedSubset.

        Parameters
        ----------
        subset : Any
            The underlying dataset or subset (must support indexing).
        transform : Callable
            A function or callable object (e.g., torchvision.transforms.Compose)
            that processes a single data sample.
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        """
        Retrieve a sample by index and apply the transformation.

        Parameters
        ----------
        index : int
            The index of the item to retrieve.

        Returns
        -------
        Tuple[Any, Any]
            A tuple containing the transformed input data and its associated label.
        """
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        """
        Return the size of the subset.

        Returns
        -------
        int
            The total number of samples in the subset.
        """
        return len(self.subset)

def _module_size_mb(m: nn.Module) -> float:
    """
    Calculate the memory footprint of a PyTorch module's parameters.

    Parameters
    ----------
    m : nn.Module
        The neural network module to evaluate.

    Returns
    -------
    float
        The total size of the model parameters in Megabytes (MB).
    """
    return float(sum(p.numel() * p.element_size() for p in m.parameters()) / 1_000_000.0)

def sample_grid_diffusion(model: DiT, out_path: Path, n: int, device: torch.device, img_shape: Tuple[int, int, int], steps: int = 50):
    """
    Generate and save a grid of synthetic images using a Diffusion Transformer (DiT).

    This function utilizes a Rectified Flow sampler to generate samples from
    noise, denormalizes the output, and saves the resulting grid to disk.

    Parameters
    ----------
    model : DiT
        The trained Diffusion Transformer model.
    out_path : Path
        The filesystem path where the generated image grid will be saved.
    n : int
        The total number of samples to generate.
    device : torch.device
        The computational device (CPU or GPU) to use for inference.
    img_shape : Tuple[int, int, int]
        The dimensions of the target image (Channels, Height, Width).
    steps : int, optional
        The number of integration steps for the flow matching sampler (default is 50).
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

    This function groups models by their associated class ID and computes the
    arithmetic mean of their parameters (state dictionaries). It handles dynamic
    reconstruction of model architectures (VAE Decoder or DiT) to load the
    aggregated weights.

    Parameters
    ----------
    client_models : Dict[str, Dict[int, torch.nn.Module]]
        A nested dictionary mapping client identifiers to a dictionary of
        class IDs and their corresponding local models.

    Returns
    -------
    Dict[int, torch.nn.Module]
        A dictionary mapping class IDs to the aggregated global models.
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
        # This factory logic handles both VAE Decoders and DiT configurations
        if isinstance(tmpl, Decoder):
            agg_model = Decoder(tmpl.latent_dim, tmpl.output_channels, tmpl.hidden_dims)
        elif isinstance(tmpl, DiT) or hasattr(tmpl, "config"):
            try:
                if hasattr(tmpl, "config"): cfg = tmpl.config
                else:
                    # Fallback configuration extraction if the config object is missing
                    cfg = DiffusionConfig(
                        in_ch=getattr(tmpl, "in_ch", 1),
                        embed_dim=getattr(tmpl, "embed_dim", 256),
                        depth=getattr(tmpl, "depth", 8),
                        num_heads=getattr(tmpl, "num_heads", 8),
                        mlp_ratio=getattr(tmpl, "mlp_ratio", 4.0),
                        patch_size=getattr(tmpl, "patch_size", 2),
                    )
                agg_model = DiT(cfg)
            except Exception as e:
                logger.warning(f"Aggregation Fallback for DiT: {e}")
                agg_model = type(tmpl)()
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

    This function manages the data loading, training execution, artifact generation,
    and model aggregation for the experiment. It supports varying partition modes,
    specifically 'silos' (isolated learning) and 'skew/dirichlet' (federated learning).

    Parameters
    ----------
    args : Any
        Configuration namespace containing hyperparameters (epochs, batch size, model type, etc.).
    device : torch.device
        The hardware device (CPU/GPU) for training.
    P : Any
        Path configuration object managing project directories.
    train_subsets_dict : Dict[str, Dict[int, Any]]
        Data partitions mapped by client name and class ID (used for complex partitions).
    train_subsets : List[Any]
        List of data subsets (used for simple silo partitioning).
    present_classes : List[int]
        List of class IDs present in the current experiment scope.
    num_classes : int
        Total number of classes in the dataset.
    chans : int
        Number of image channels.
    img_shape : Tuple[int, int, int]
        Dimensions of the input images.
    tracker : Any
        Experiment tracker for logging metrics and phases.
    hist : Any
        History object for recording loss curves.
    perc_loss : nn.Module
        Perceptual loss module (LPIPS or similar) used during VAE training.

    Returns
    -------
    Tuple[...]
        - gen_models: List of aggregated/global models per class.
        - client_gen_models: Dictionary of local models trained by clients.
        - client_sample_counts: Dictionary of sample counts per client per class.
        - gen_step_total: Total number of training steps executed.
        - gen_start: Timestamp of the phase start.
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
    # In this mode, clients may hold data for multiple classes. We iterate over
    # clients and then over the classes they possess.
    if partition_mode in ["skew", "dirichlet"]:
        for client_name, class_subsets in train_subsets_dict.items():
            client_gen_models[client_name] = {}
            client_sample_counts[client_name] = {}

            for class_id, train_subset in class_subsets.items():
                logger.info(f"[{partition_mode.upper()}] Training client {client_name}, class {class_id}")

                # Apply robust augmentation policies to prevent memorization
                # This ensures the generative model captures generalized features suitable for
                # downstream classification tasks rather than overfitting to specific samples.
                robust_transform = build_transform(args.dataset, train=True, robustness=True)
                ds_client = TransformedSubset(train_subset, robust_transform)

                ld = DataLoader(ds_client, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, pin_memory=True)

                if tracker: tracker.start_phase(f"client_{client_name}_class_{class_id}_gen")

                # Dispatch training to the appropriate model trainer (VAE vs. DiT)
                if args.model == "vae":
                    vae, steps = train_vae(
                        VAE(chans, args.latent_dim), ld, device, args.epochs, hist,
                        f"{client_name}_class_{class_id}", perc_loss,
                        dp=args.dp, dp_clip=args.dp_clip, dp_noise_mult=args.dp_noise, dp_microbatch=args.dp_microbatch, tracker=tracker
                    )
                    model_to_save = vae.decoder
                    full_model = vae
                else:
                    cfg = DiffusionConfig(in_ch=chans, embed_dim=int(args.dit_embed), depth=int(args.dit_depth),
                                          num_heads=int(args.dit_heads), mlp_ratio=4.0, patch_size=int(args.dit_patch))

                    # Fix: Pass checkpoint args to train_diffusion
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

                # Persist checkpoints and generate visual artifacts (samples)
                gen_ckpt = P.root / "models" / "generators" / f"client_{client_name}_class_{class_id:03d}.pt"
                torch.save(full_model.state_dict(), gen_ckpt)
                if tracker: tracker.record_artifact(gen_ckpt)

                out_png = P.root / "artifacts" / "samples" / f"client_{client_name}_class_{class_id:03d}.png"
                if args.model == "vae": sample_grid(full_model, args.latent_dim, out_png)
                else: sample_grid_diffusion(full_model, out_png, 64, device, img_shape)

                gen_step_total += steps

        # Perform Federation: Aggregate local models into a global model per class
        if getattr(args, "aggregation", "simple") == "simple":
            aggregated = aggregate_models_simple(client_gen_models)
            gen_models = [aggregated.get(i, None) for i in range(num_classes)]
        else:
            gen_models = [None] * num_classes

        _export_client_class_distribution(P.root, client_sample_counts, num_classes)

    # -------------------------------------------------------------------------
    # PART B: Silos Partition (Isolated Clients)
    # -------------------------------------------------------------------------
    # In this mode, we assume a one-to-one mapping where each client represents
    # a distinct class or isolated silo. No aggregation is typically performed here.
    else:
        silos_client_counts = {}
        for i, d in enumerate(present_classes):
            logger.info(f"[SILOS] Training client {d}")

            # Apply robust augmentation policies
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
                cfg = DiffusionConfig(in_ch=chans, embed_dim=int(args.dit_embed), depth=int(args.dit_depth),
                                      num_heads=int(args.dit_heads), mlp_ratio=4.0, patch_size=int(args.dit_patch))

                # Fix: Pass checkpoint args to train_diffusion
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

            # Persist checkpoints and generate visual artifacts
            gen_ckpt = P.root / "models" / "generators" / f"class-{d:03d}.pt"
            torch.save(full_model.state_dict(), gen_ckpt)
            if tracker: tracker.record_artifact(gen_ckpt)

            out_png = P.root / "artifacts" / "samples" / f"class-{d:03d}.png"
            if args.model == "vae": sample_grid(full_model, args.latent_dim, out_png)
            else: sample_grid_diffusion(full_model, out_png, 64, device, img_shape)

            silos_client_counts[f"client{d}"] = {d: len(train_subsets[i])}

        # Calculate and log the size of the generator model for resource tracking
        sz = _module_size_mb(gen_models[0])
        with open(P.root / "models" / "sizes.json", "w") as f:
            json.dump({"generator_mb": float(sz)}, f, indent=2)
        _export_client_class_distribution(P.root, silos_client_counts, num_classes)

    # ---------------------------------------------------------------
    # Network Accounting
    # ---------------------------------------------------------------
    # Delegate to the network simulation handler to estimate bandwidth usage
    _handle_network_simulation(args, P, tracker, partition_mode, present_classes, client_gen_models)

    return gen_models, client_gen_models, client_sample_counts, gen_step_total, gen_start


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
    Loads pre-trained generative checkpoints from disk.
    Supports EDM2 directory structure: checkpoints/{dataset}/{partition}/{client}/class_{cid}/training-state-{epoch}.pt
    """
    logger.info(f"[GEN-PHASE] Loading checkpoints for Epoch Family: {epoch_family}")

    # Initialize containers
    gen_models = [None] * (max(present_classes) + 1)
    client_gen_models: Dict[str, Dict[int, Any]] = {}
    client_sample_counts: Dict[str, Dict[int, int]] = {}

    # Define the partition path (e.g., silos vs dirichlet_0.1)
    if getattr(args, "partition", "silos") == "dirichlet" and getattr(args, "alpha", None) is not None:
        part_name = f"dirichlet_{args.alpha}"
    else:
        part_name = getattr(args, "partition", "silos")

    # Safe dataset name (e.g., medmnist:bloodmnist -> medmnist_bloodmnist)
    safe_ds = args.dataset.replace(":", "_")

    # Root path for checkpoints
    project_root = Path(".").resolve()  # Start from current working dir (Project Root)
    ckpt_root = project_root / "checkpoints" / safe_ds / part_name

    if not ckpt_root.exists():
        # Fallback: Try looking relative to where script is running if project_root fails
        ckpt_root = Path("checkpoints") / safe_ds / part_name
        if not ckpt_root.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_root}")

    loaded_count = 0

    # Iterate over clients expected in this experiment
    for client_name, class_subsets in train_subsets_dict.items():
        client_gen_models[client_name] = {}
        client_sample_counts[client_name] = {}

        for class_id, subset in class_subsets.items():
            client_sample_counts[client_name][class_id] = len(subset)

            # --- PATH CONSTRUCTION (EDM2 Style) ---
            # 1. Format epoch string (e.g., 25001 -> 0025001)
            epoch_str = f"{int(epoch_family):07d}"

            # 2. Build filename
            fname = f"training-state-{epoch_str}.pt"

            # 3. Build full path (checkpoints are inside "class_{class_id}" subfolders)
            # Correcting for potential client name mismatch
            # If logic expects 'client_0' but dict has 'client0', we try both.
            ckpt_path = ckpt_root / client_name / f"class_{class_id}" / fname

            if not ckpt_path.exists():
                 # Try adding/removing underscore if standard path fails
                 alt_client = client_name.replace("client_", "client") if "client_" in client_name else client_name.replace("client", "client_")
                 ckpt_path = ckpt_root / alt_client / f"class_{class_id}" / fname

            if not ckpt_path.exists():
                logger.warning(f"Checkpoint missing for {client_name} at {ckpt_path}")
                continue

            try:
                # Load the model architecture
                if args.model == "diffusion":
                    from models.diffusion import DiT, DiffusionConfig
                    # Reconstruct config
                    cfg = DiffusionConfig(
                        in_ch=chans,
                        embed_dim=128,  # Must match trainer
                        num_classes=0,  # Conditional via directory structure, usually 0 for EDM2 single-class
                        img_resolution=img_shape[-1],
                        dropout=0.3
                    )
                    model = DiT(cfg)
                    model.to(device)

                    # Load weights
                    data = torch.load(ckpt_path, map_location=device)

                    # --- FIX START ---
                    # EDM2 CheckpointIO saves a dictionary where 'ema' is a dict, not an object.
                    # The actual weights are inside data['ema']['emas'][0] (for the first std profile).
                    if 'ema' in data and isinstance(data['ema'], dict) and 'emas' in data['ema']:
                         # Extract the primary EMA weights
                         weights = data['ema']['emas'][0]
                         model.edm_net.load_state_dict(weights)
                    elif 'net' in data:
                         # Fallback to non-EMA weights
                         model.edm_net.load_state_dict(data['net'])
                    else:
                         # Fallback for raw state dict (unlikely for EDM2 but safe)
                         model.load_state_dict(data)
                    # --- FIX END ---

                    model.eval()

                elif args.model == "vae":
                    from models.vae import VAE
                    model = VAE(chans, args.latent_dim)
                    model.load_state_dict(torch.load(ckpt_path, map_location=device))
                    model.to(device).eval()

                else:
                    raise ValueError(f"Unknown generative model type: {args.model}")

                # Store model references
                # For silos, we map the single class model to the global class ID
                gen_models[class_id] = model
                client_gen_models[client_name][class_id] = model
                loaded_count += 1

            except Exception as e:
                logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")

    logger.info(f"[GEN-PHASE] Successfully loaded {loaded_count} client models.")
    return gen_models, client_gen_models, client_sample_counts

def _handle_network_simulation(args: Any, P: Any, tracker: Any, partition_mode: str, present_classes: List[int], client_gen_models: Dict[str, Any]):
    """
    Simulate and log network bandwidth usage based on generated file artifacts.

    This function iterates through generated model checkpoints and logs
    'tx' (transmission/upload) and 'rx' (reception/download) events
    to the experiment tracker, mimicking the costs of a real distributed system.

    Parameters
    ----------
    args : Any
        Configuration arguments containing inference mode settings.
    P : Any
        Path configuration object.
    tracker : Any
        Experiment tracker instance.
    partition_mode : str
        The data partitioning strategy used (e.g., 'skew', 'dirichlet', 'silos').
    present_classes : List[int]
        List of class IDs involved in the experiment.
    client_gen_models : Dict[str, Any]
        Dictionary of trained client models.
    """
    if tracker is None: return

    # Map logical model identifiers to their physical file paths
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

    # Simulate Uploads (Client -> Server)
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

    # Simulate Downloads (Server -> Client or Peer-to-Peer)
    if not is_server:
        if partition_mode in ["skew", "dirichlet"]:
            for cname in client_gen_models.keys():
                ph = f"client_{cname}_download"
                tracker.start_phase(ph)
                for (src, cid), p in gen_ckpts_map.items():
                    # Download models from other clients/classes not present locally
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