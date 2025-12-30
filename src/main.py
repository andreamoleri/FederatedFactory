from __future__ import annotations

"""
▶️ Experiment Entry Point & Orchestrator
---------------------------------------

This module serves as the primary entry point for executing machine learning 
experiments. It orchestrates the configuration, execution, and logging of both 
single-run experiments and extensive grid searches.

🧠 Purpose:
    To provide a robust, reproducible wrapper around the core experimental logic,
    ensuring consistent environment capture, cost tracking (computational and 
    carbon), and standardized output formatting for downstream analysis.

🔧 Core Functionalities:
    • CLI Argument Parsing: Handles complex configurations for Federated Learning, 
      Differential Privacy, and Generative Models (VAE/Diffusion).
    • Environment Capture: Snapshots Git commits, Conda/Pip packages, and 
      hardware specs to ensure reproducibility.
    • Grid Search: Automates parameter sweeping and sequential execution.
    • Cost & Metrics: Aggregates computational costs, energy usage, and 
      network traffic estimates into a final manifest.

🎯 Intended Use:
    • High-performance computing clusters for batch job execution.
    • Academic research requiring strict provenance and carbon accounting.

📁 Dependencies:
    • torch
    • argparse
    • local modules (jobs, utils, metrics)

📝 Notes:
    This script enforces single-threaded execution for underlying linear algebra 
    libraries (OMP, MKL) to prevent CPU oversubscription in cluster environments.

Author: Andrea Moleri
File Location: src/main.py
Last Modified: 21/11/2025
"""

import os as _os

# Thread environment optimizations
# Setting these variables to "1" prevents CPU oversubscription when running
# multiple jobs on the same node, ensuring predictable performance.
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("KMP_WARNINGS", "0")

import warnings
import logging
import argparse
import json
import shutil
import torch  # Retained only for _write_environment (version check)
from pathlib import Path
from typing import Dict, List, Any

# Disable specific warnings
warnings.filterwarnings("ignore", message="Multiple instances of codecarbon")
warnings.filterwarnings("ignore", category=UserWarning, module="fvcore")

fvcore_logger = logging.getLogger("fvcore.nn.jit_analysis")
fvcore_logger.setLevel(logging.ERROR)
codecarbon_logger = logging.getLogger("codecarbon")
codecarbon_logger.setLevel(logging.ERROR)

# -------------------------------------- logging ------------------------------
from logs.logger import get_logger
from logs import messages as logmsg

logger = get_logger(__name__)

# ------------------------------ utility helpers ------------------------------
# Imports required only for argument parsing or grid search expansion
from utils import (
    set_seed,
    parse_csv_or_single,
    expand_grid,
)
from imports.data_management import DATASET_META # Necessary for validating args.dataset
from metrics.costs import ExperimentCostTracker

# --------------------------- CORE IMPORTS -----------------------------
# The fundamental import connecting main to the new experiment runner
from jobs.experiment_runner import run_experiment


def _write_environment(root: Path):
    """
    Captures and writes details about the execution environment to text files.

    This function ensures reproducibility by logging the current Git commit,
    Conda environment, Pip packages, and PyTorch/CUDA configuration. It is
    designed to be fail-safe, catching exceptions to avoid halting the
    experiment if environmental data cannot be retrieved.

    Parameters
    ----------
    root : Path
        The root directory of the experiment output where the 'environment'
        sub-folder will be created.
    """
    env_dir = root / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess
        # git
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
            (env_dir / "git.txt").write_text(f"commit: {commit}\ndirty: {dirty}\n")
        except Exception:
            pass

        # 1. Conda export
        wrote_env = False
        try:
            conda_env = subprocess.check_output(["conda", "env", "export", "--no-builds"], text=True)
            (env_dir / "conda-environment.yaml").write_text(conda_env)
            wrote_env = True
        except Exception:
            pass

        # 2. Pip freeze fallback
        if not wrote_env:
            try:
                freeze = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
                (env_dir / "pip-freeze.txt").write_text(freeze)
            except Exception:
                pass

        # Torch env info
        try:
            torch_lines = [
                f"torch: {torch.__version__}",
                f"cuda_is_available: {torch.cuda.is_available()}",
                f"cuda_version: {torch.version.cuda}",
                f"cudnn: {getattr(torch.backends.cudnn, 'version', lambda: None)()}",
                f"devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}",
            ]
            (env_dir / "torch-env.txt").write_text("\n".join(map(str, torch_lines)) + "\n")
        except Exception:
            pass
    except Exception:
        pass

    # System info
    try:
        import platform, os
        (env_dir / "system.txt").write_text(
            f"platform: {platform.platform()}\n"
            f"python: {platform.python_version()}\n"
            f"cpu_count: {os.cpu_count()}\n"
        )
    except Exception:
        pass


def _sha256(p: Path) -> str:
    """
    Computes the SHA-256 checksum of a given file.

    Parameters
    ----------
    p : Path
        The path to the file to be hashed.

    Returns
    -------
    str
        The hexadecimal digest of the file's SHA-256 hash.
    """
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(root: Path, args, acc, num_classes, run_utc_iso8601: str | None = None):
    """
    Generates a JSON manifest file summarizing the experiment.

    The manifest acts as a central index for the experiment, containing
    configuration parameters, high-level metrics, directory structure,
    and integrity checksums for key artifacts.

    Parameters
    ----------
    root : Path
        The root directory of the experiment output.
    args : Namespace
        The parsed command-line arguments used for the experiment.
    acc : float
        The final accuracy (or primary metric) achieved.
    num_classes : int
        The number of classes in the dataset.
    run_utc_iso8601 : str | None, optional
        The ISO 8601 timestamp of the run execution, by default None.
    """
    manifest = {
        "run": {
            "folder": str(root),
        },
        "identity": {
            "model": args.model,
            "dataset": args.dataset,
            "seed": int(args.seed),
            "mode": args.infer_mode,
            "latent_dim": int(args.latent_dim),
            "dp": bool(args.dp),
            "input_size": int(getattr(args, "input_size", 0) or 0),
            "partition": getattr(args, "partition", "silos"),
            "aggregation": getattr(args, "aggregation", "simple"),
            "dirichlet_alpha": float(getattr(args, "alpha", 0.0)),
            "num_clients": int(getattr(args, "num_clients", 0)),
        },
        "quality": {
            "accuracy": float(acc)
        },
        "paths": {
            "metrics": "metrics/",
            "models": "models/",
            "artifacts": "artifacts/",
            "datasets": "datasets/",
            "costs": "costs/"
        },
        "checksums": {}
    }

    if run_utc_iso8601 is not None:
        manifest["run"]["utc_iso8601"] = run_utc_iso8601

    for rel in ("metrics/classifier.json",
                "metrics/generative.json",
                "metrics/confusion-matrix.csv",
                "models/classifiers/central.pt",
                "report.pdf"):
        fp = root / rel
        if fp.exists():
            manifest["checksums"][rel] = _sha256(fp)


def _sum_dir_bytes(path: Path) -> int:
    """
    Recursively calculates the total size in bytes of all files in a directory.

    Parameters
    ----------
    path : Path
        The directory to traverse.

    Returns
    -------
    int
        Total size in bytes. Returns 0 if the path does not exist.
    """
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def _finalize_costs(
        tmp_cost_dir: Path,
        final_folder: Path,
        args_ns,
        carbon_intensity_kg_per_kwh: float | None = None,
) -> None:
    """
    Finalizes cost/emission metrics and formalizes network statistics.

    This function aggregates temporary logging data, backfills carbon emission
    calculations based on energy usage and intensity factors, and estimates
    network traffic for Federated Learning scenarios based on model checkpoint sizes.

    Parameters
    ----------
    tmp_cost_dir : Path
        The temporary directory containing raw cost logs.
    final_folder : Path
        The destination directory for the finalized 'costs' folder.
    args_ns : Namespace
        The experiment configuration arguments.
    carbon_intensity_kg_per_kwh : float | None, optional
        The carbon intensity factor. If None, attempts to read from env vars
        or defaults to a fallback value (0.40).
    """
    try:
        if not tmp_cost_dir.exists():
            return

        costs_dir = final_folder / "costs"
        costs_dir.mkdir(parents=True, exist_ok=True)

        # 1) Copy ALL files from temp to final location
        try:
            for p in tmp_cost_dir.iterdir():
                if p.is_file():
                    shutil.copy2(p, costs_dir / p.name)
        except Exception:
            pass

        # 2) Load (or create) costs.json
        cj = costs_dir / "costs.json"
        data = {}
        if cj.exists():
            try:
                data = json.loads(cj.read_text())
            except Exception:
                data = {}

        # 3) Update useful metadata and disk usage
        try:
            data.setdefault("environment", {})["args"] = vars(args_ns)
            data.setdefault("bytes", {})
            total_written = 0
            try:
                for pp in final_folder.rglob("*"):
                    if pp.is_file():
                        try:
                            total_written += pp.stat().st_size
                        except Exception:
                            pass
            except Exception:
                pass
            data["bytes"]["artifacts_written_bytes"] = int(total_written)
        except Exception:
            pass

        # 4) BACKFILL emissions from energy.kwh
        try:
            if carbon_intensity_kg_per_kwh is None:
                try:
                    env_ci = float(_os.environ.get("CARBON_INTENSITY_KG_PER_KWH", ""))
                    carbon_intensity_kg_per_kwh = env_ci
                except Exception:
                    pass

            if carbon_intensity_kg_per_kwh is None:
                carbon_intensity_kg_per_kwh = 0.40

            kwh = data.get("energy", {}).get("kwh", None)
            if isinstance(kwh, (int, float)):
                data.setdefault("energy", {})
                if data["energy"].get("emissions_kg_co2e", None) is None:
                    emis = float(kwh) * float(carbon_intensity_kg_per_kwh)
                    data["energy"]["emissions_kg_co2e"] = float(emis)
                    data["energy"]["emissions_factor_kg_per_kwh"] = float(carbon_intensity_kg_per_kwh)

            phases = data.get("phases", {})
            total_sec = data.get("wall_clock", {}).get("total_sec", None)
            if isinstance(total_sec, (int, float)) and total_sec > 0 and phases:
                for ph_name, ph in phases.items():
                    sec = ph.get("wall_clock_sec", None)
                    if not isinstance(sec, (int, float)) or sec < 0:
                        continue
                    ph.setdefault("energy", {})
                    ph_kwh_measured = ph["energy"].get("kwh", None)
                    if isinstance(ph_kwh_measured, (int, float)) and ph_kwh_measured > 0:
                        ph["energy"]["emissions_kg_co2e"] = float(ph_kwh_measured) * float(carbon_intensity_kg_per_kwh)
                        ph["energy"]["emissions_factor_kg_per_kwh"] = float(carbon_intensity_kg_per_kwh)
                    else:
                        # Estimate based on time share if direct measurement is missing
                        share = sec / total_sec
                        ph_kwh = share * float(kwh)
                        ph_emis = ph_kwh * float(carbon_intensity_kg_per_kwh)
                        ph["energy"]["kwh_estimated_from_time_share"] = float(ph_kwh)
                        ph["energy"]["emissions_kg_co2e_estimated"] = float(ph_emis)
                        ph["energy"]["emissions_factor_kg_per_kwh"] = float(carbon_intensity_kg_per_kwh)
                data["phases"] = phases
        except Exception:
            pass

        # 5) NETWORK: Measured bytes from checkpoints
        try:
            gen_dir = final_folder / "models" / "generators"
            per_model_bytes = {}
            if gen_dir.exists():
                # Silos/FedAvg standard naming convention
                for p in gen_dir.glob("class-*.pt"):
                    try:
                        per_model_bytes[p.name] = int(p.stat().st_size)
                    except Exception:
                        pass
                # Skew/Dirichlet naming convention
                for p in gen_dir.glob("client_*_class_*.pt"):
                    try:
                        per_model_bytes[p.name] = int(p.stat().st_size)
                    except Exception:
                        pass

            n_models = len(per_model_bytes)
            if n_models > 0:
                server_total_bytes = sum(per_model_bytes.values())
                # Estimate local traffic: total size * (number of clients - 1)
                local_total_bytes = sum(sz * (n_models - 1) for sz in per_model_bytes.values())

                def _bytes_to_mb(x: int | float) -> float:
                    return float(x) / 1_000_000.0

                def _bytes_to_mib(x: int | float) -> float:
                    return float(x) / float(1024 ** 2)

                per_model_mib = {k: _bytes_to_mib(v) for k, v in per_model_bytes.items()}

                data["network"] = {
                    "method": "sum_checkpoint_file_sizes",
                    "units": "bytes",
                    "n_models": n_models,
                    "per_model_bytes": per_model_bytes,
                    "per_model_mib": per_model_mib,
                    "server_total_bytes": int(server_total_bytes),
                    "local_total_bytes": int(local_total_bytes),
                    "server_total_mb_1e6": _bytes_to_mb(server_total_bytes),
                    "server_total_mib_2pow20": _bytes_to_mib(server_total_bytes),
                    "local_total_mb_1e6": _bytes_to_mb(local_total_bytes),
                    "local_total_mib_2pow20": _bytes_to_mib(local_total_bytes),
                }
        except Exception:
            pass

        # 6) Write costs.json
        try:
            cj.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

        # 7) Clean up temporary directory
        try:
            for p in tmp_cost_dir.iterdir():
                if p.is_file():
                    p.unlink(missing_ok=True)
            tmp_cost_dir.rmdir()
        except Exception:
            pass

    except Exception:
        logger.warning("Finalizing cost metrics failed (non-fatal).")


def _is_experiment_done(out_dir_root: str, current_ns: argparse.Namespace) -> bool:
    """
    Scans the output directory recursively to see if a *successfully completed* run
    with identical parameters already exists.

    A run is considered "done" only if:
    1. It has a matching `args.json` configuration.
    2. It has a `manifest.json` file (indicating successful completion).
    """
    root_path = Path(out_dir_root)
    if not root_path.exists():
        return False

    # Convert current args to a dictionary for comparison
    current_vars = vars(current_ns)

    # keys to ignore during comparison (paths, runtime flags)
    ignored_keys = {'data_dir', 'out_dir', 'max_experiments', 'grid', 'save_datasets', 'network_accounting'}

    # Look for all args.json files in the output structure
    for args_file in root_path.rglob("args.json"):
        experiment_dir = args_file.parent

        # 1. CRITICAL CHECK: Does manifest.json exist?
        # If not, the previous run failed or didn't finish. We should NOT skip.
        if not (experiment_dir / "manifest.json").exists():
            continue

        # 2. Compare the configuration
        try:
            with open(args_file, 'r') as f:
                saved_args = json.load(f)

            is_match = True
            for k, v in current_vars.items():
                if k in ignored_keys:
                    continue

                # Compare string representations to handle slight type diffs (e.g. tuple vs list)
                # or missing keys in old runs.
                # We use str() to be robust against "10" vs 10 or "1e-4" vs 0.0001
                if str(saved_args.get(k)) != str(v):
                    is_match = False
                    break

            if is_match:
                print(f"Skipping experiment (Found completed run in: {experiment_dir})")
                return True

        except Exception:
            # If args.json is corrupted, treat it as not a match
            continue

    return False

def main():
    """
    Main execution routine.

    Parses CLI arguments to determine the experiment configuration. It supports
    two primary modes of operation:
    1. Grid Search: Iterates over combinatoric lists of parameters (comma-separated).
    2. Single Run: Executes a single experiment with the specified parameters.

    Workflow:
    - Argument parsing (including CSV expansion for grid search).
    - Iteration (Grid Mode) or direct execution (Single Mode).
    - Invocation of `ExperimentCostTracker` context manager.
    - Execution of `run_experiment` (core logic).
    - Finalization of costs, environment logging, and manifest creation.
    """
    logger = get_logger(__name__)

    # ------------------------------ CLI -------------------------------------
    cli = argparse.ArgumentParser()
    cli.add_argument("--grid", action="store_true", help="Enable grid search over comma-separated arguments")
    cli.add_argument("--max-experiments", type=int, default=None, help="Maximum number of experiments to run in grid mode")

    cli.add_argument("--dataset", default="mnist", help=f"Dataset name ({', '.join(sorted(DATASET_META.keys()))})")
    cli.add_argument("--seed", default="1234")
    cli.add_argument("--latent-dim", default="64")
    cli.add_argument("--noise-std", default="0.1")
    cli.add_argument("--infer-mode", default="local")
    cli.add_argument("--dp", default="false", help="true/false or comma-separated list")
    cli.add_argument("--grayscale", default="false", help=("true/false or comma-separated list."))
    cli.add_argument("--model", default="vae",
                      choices=["vae", "diffusion", "baseline:fedavg", "baseline:feddf", "baseline:fedprox", "baseline:feddyn", "baseline:scaffold"],
                      help=("Select the model or baseline method"))

    # Partitioning
    cli.add_argument("--partition", default="silos", choices=["silos", "skew", "dirichlet"],
                      help="Partition mode: 'silos', 'dirichlet', or 'skew'")
    cli.add_argument("--client-config", type=str, default="", help=("Client configuration for skew mode."))
    cli.add_argument("--aggregation", default="simple", choices=["simple", "weighted"], help=("Aggregation mode"))
    cli.add_argument("--samples-per-class", type=int, default=0, help=("Number of synthetic samples per class."))

    # Baseline args
    cli.add_argument("--baseline-epochs-per-round", type=int, default=10)
    cli.add_argument("--baseline-max-rounds", type=int, default=50)
    cli.add_argument("--baseline-patience", type=int, default=10)

    # Standard args
    cli.add_argument("--epochs", type=int, default=20)
    cli.add_argument("--clf-epochs", type=int, default=5)
    cli.add_argument("--batch-size", type=int, default=128)
    cli.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers")
    cli.add_argument("--data-dir", default="./data")
    cli.add_argument("--out-dir", default="../output")

    # Model Specifics
    cli.add_argument("--input-size", type=int, default=0)
    cli.add_argument("--dit-embed", type=int, default=256)
    cli.add_argument("--dit-depth", type=int, default=8)
    cli.add_argument("--dit-heads", type=int, default=8)
    cli.add_argument("--dit-patch", type=int, default=2)

    # === HYPERPARAMETERS FOR TUNING ===
    cli.add_argument("--learning-rate", type=float, default=0.1, help="Global/Local Learning Rate")
    cli.add_argument("--baseline-momentum", type=float, default=0.9, help="SGD Momentum")
    cli.add_argument("--baseline-weight-decay", type=float, default=1e-4, help="SGD Weight Decay")
    cli.add_argument("--baseline-clip-grad-norm", type=float, default=5.0, help="Max L2 norm for gradient clipping")


    # Method Specific
    cli.add_argument("--fedprox-mu", type=float, default=0.01, help="FedProx proximal term")
    cli.add_argument("--feddyn-alpha", type=float, default=0.01, help="FedDyn regularization alpha")

    # DP
    cli.add_argument("--dp-clip", type=float, default=1.0)
    cli.add_argument("--dp-noise", type=float, default=1.1)
    cli.add_argument("--dp-microbatch", type=int, default=8)

    # Dirichlet
    cli.add_argument(
    "--alpha",
    default="0.5",
    help="Concentration parameter(s) for Dirichlet, e.g. '0.1,0.5,1.0'"
)

    cli.add_argument("--num-clients", type=int, default=10, help="Number of clients")

    # Metrics
    cli.add_argument("--eval-samples-per-class", type=int, default=512)
    cli.add_argument("--pr-knn-k", type=int, default=3)
    cli.add_argument("--sota-json", type=str, default="")
    cli.add_argument("--copy-threshold-percentile", type=float, default=1.0)

    # Costs / Network
    cli.add_argument("--network-accounting", choices=["manual", "psutil", "mixed"], default="mixed")
    cli.add_argument("--carbon-intensity-kg-per-kwh", type=float, default=0.48)

    # Client config
    cli.add_argument("--client-fraction", type=float, default=1.0)
    cli.add_argument("--clients-per-round", type=int, default=None)

    # Misc
    cli.add_argument("--classes", type=str, default="", help="(NICO++) Comma-separated classes")
    cli.add_argument("--save-datasets", action="store_true")
    cli.add_argument("--robustness", type=str, default="false", help="Enable Gaussian noise (true/false)")

    args_raw, _unknown = cli.parse_known_args()

    # Parse list params
    list_params = {
        "dataset": parse_csv_or_single(args_raw.dataset, str),
        "seed": parse_csv_or_single(args_raw.seed, int),
        "latent_dim": parse_csv_or_single(args_raw.latent_dim, int),
        "noise_std": parse_csv_or_single(args_raw.noise_std, float),
        "infer_mode": parse_csv_or_single(args_raw.infer_mode, str),
        "dp": parse_csv_or_single(args_raw.dp, lambda x: x.lower() in ("1", "true", "yes")),
        "grayscale": parse_csv_or_single(args_raw.grayscale, lambda x: x.lower() in ("1", "true", "yes")),
        "model": parse_csv_or_single(args_raw.model, str),
        "partition": parse_csv_or_single(args_raw.partition, str),
        "alpha": parse_csv_or_single(args_raw.alpha, float),
        "learning_rate": parse_csv_or_single(args_raw.learning_rate, float),
        "fedprox_mu": parse_csv_or_single(args_raw.fedprox_mu, float),
        "feddyn_alpha": parse_csv_or_single(args_raw.feddyn_alpha, float),
        "robustness": parse_csv_or_single(args_raw.robustness, lambda x: x.lower() in ("1", "true", "yes")),
    }

    multiple_values = any(len(v) > 1 for v in list_params.values())

    # ======================================================================
    # GRID MODE
    # ======================================================================
    if args_raw.grid or multiple_values:
        grid = list(expand_grid(list_params))
        if args_raw.max_experiments is not None:
            grid = grid[: args_raw.max_experiments]

        logger.info(logmsg.GRID_LAUNCH.format(n=len(grid)))

        # --- LOOP 1: Planning / Logging ---
        planned_lines = []
        for i, g in enumerate(grid, 1):
            planned_lines.append(
                f"{i:3d}) dataset={g['dataset']} seed={g['seed']} "
                f"latent={g['latent_dim']} noise_std={g['noise_std']} "
                f"mode={g['infer_mode']} dp={g['dp']} "
                f"grayscale={g['grayscale']} model={g['model']} "
                f"partition={g['partition']} alpha={g['alpha']}"
            )
        msg = "Planned experiments:\n" + "\n".join(planned_lines)
        logger.info(msg)
        try:
            Path(args_raw.out_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(args_raw.out_dir) / "planned_experiments.txt", "w") as f:
                f.write(msg + "\n")
        except Exception:
            pass

        logger.info("")
        results = []

        # --- LOOP 2: Execution (THIS WAS MISSING) ---
        for i, g in enumerate(grid, 1):

            # Create the namespace for THIS specific experiment 'g'
            ns = argparse.Namespace(
                # --- Grid/List Params ---
                dataset=g["dataset"],
                seed=g["seed"],
                latent_dim=g["latent_dim"],
                noise_std=g["noise_std"],
                infer_mode=g["infer_mode"],
                dp=g["dp"],
                grayscale=g["grayscale"],
                model=g["model"],
                partition=g["partition"],
                alpha=g["alpha"],
                learning_rate=g["learning_rate"],
                fedprox_mu=g["fedprox_mu"],
                feddyn_alpha=g["feddyn_alpha"],

                # --- Static Args (passed from args_raw) ---
                baseline_momentum=args_raw.baseline_momentum,
                baseline_weight_decay=args_raw.baseline_weight_decay,
                baseline_clip_grad_norm=args_raw.baseline_clip_grad_norm,
                client_config=args_raw.client_config,
                aggregation=args_raw.aggregation,
                samples_per_class=args_raw.samples_per_class,
                baseline_epochs_per_round=args_raw.baseline_epochs_per_round,
                baseline_max_rounds=args_raw.baseline_max_rounds,
                baseline_patience=args_raw.baseline_patience,
                epochs=args_raw.epochs,
                clf_epochs=args_raw.clf_epochs,
                batch_size=args_raw.batch_size,
                workers=args_raw.workers,
                data_dir=args_raw.data_dir,
                out_dir=args_raw.out_dir,
                dp_clip=args_raw.dp_clip,
                dp_noise=args_raw.dp_noise,
                dp_microbatch=args_raw.dp_microbatch,
                eval_samples_per_class=args_raw.eval_samples_per_class,
                pr_knn_k=args_raw.pr_knn_k,
                sota_json=args_raw.sota_json,
                copy_threshold_percentile=args_raw.copy_threshold_percentile,
                classes=args_raw.classes,
                dit_embed=args_raw.dit_embed,
                dit_depth=args_raw.dit_depth,
                dit_heads=args_raw.dit_heads,
                dit_patch=args_raw.dit_patch,
                input_size=args_raw.input_size,
                save_datasets=args_raw.save_datasets,
                network_accounting=args_raw.network_accounting,
                client_fraction=args_raw.client_fraction,
                clients_per_round=args_raw.clients_per_round,
                num_clients=args_raw.num_clients,
                carbon_intensity_kg_per_kwh=args_raw.carbon_intensity_kg_per_kwh,
            )

            if _is_experiment_done(args_raw.out_dir, ns):
                logger.info(f"Skipping experiment {i}/{len(grid)}: Already completed.")
                continue

            logger.info("\n" + logmsg.GRID_SEPARATOR)
            logger.info(
                logmsg.GRID_EXPERIMENT_HDR.format(
                    current=i, total=len(grid), dataset=ns.dataset,
                    latent=ns.latent_dim, dp=ns.dp, mode=ns.infer_mode
                )
            )

            tmp_cost_dir = Path(ns.out_dir) / f".costs_tmp_run{i}"
            tmp_cost_dir.mkdir(parents=True, exist_ok=True)

            with ExperimentCostTracker(tmp_cost_dir, ns, network_accounting=ns.network_accounting) as _tracker:
                # CALL TO EXPERIMENT RUNNER
                acc, folder, hist, metrics, y_true, y_pred, num_classes = run_experiment(ns, run_id=i, tracker=_tracker)

            try:
                final_folder = Path(folder)
            except Exception:
                final_folder = Path(ns.out_dir)

            _finalize_costs(tmp_cost_dir, final_folder, ns,
                            carbon_intensity_kg_per_kwh=ns.carbon_intensity_kg_per_kwh)

            try:
                args_path = final_folder / "args.json"
                with args_path.open("w") as f:
                    json.dump(vars(ns), f, indent=2)
            except Exception as e:
                logger.warning(f"Could not write args.json: {e}")

            _write_environment(final_folder)
            _write_manifest(
                final_folder,
                ns,
                acc,
                num_classes,
                run_utc_iso8601=metrics.get("run_utc_iso8601")
            )

            results.append({"acc": acc, "folder": str(final_folder)})

        logger.info("\n" + logmsg.GRID_COMPLETED)
        for r in results:
            logger.info(logmsg.GRID_RESULT_LINE.format(folder=r["folder"], acc=r["acc"]))

    # ======================================================================
    # SINGLE RUN MODE
    # ======================================================================
    else:
        # ------------------------------------------------------------------
        # FIX: Dynamically copy ALL args so we don't lose learning_rate, etc.
        # ------------------------------------------------------------------

        # 1. Create a dictionary copy of all raw arguments
        ns_dict = vars(args_raw).copy()

        # 2. Overwrite the list-based arguments with their single values
        #    (We use [0] because in single mode, these lists only have 1 item)
        for key, val_list in list_params.items():
            if key in ns_dict:
                ns_dict[key] = val_list[0]

        # 3. Reconstruct Namespace from the updated dictionary
        ns = argparse.Namespace(**ns_dict)

        # ------------------------------------------------------------------
        # Execution (Same as before)
        # ------------------------------------------------------------------
        tmp_cost_dir = Path(ns.out_dir) / ".costs_tmp_single"
        tmp_cost_dir.mkdir(parents=True, exist_ok=True)

        with ExperimentCostTracker(tmp_cost_dir, ns, network_accounting=ns.network_accounting) as _tracker:
            # CALL TO EXPERIMENT RUNNER
            acc, folder, hist, metrics, y_true, y_pred, num_classes = run_experiment(ns, tracker=_tracker)

        try:
            final_folder = Path(folder)
        except Exception:
            final_folder = Path(ns.out_dir)

        _finalize_costs(tmp_cost_dir, final_folder, ns,
                        carbon_intensity_kg_per_kwh=ns.carbon_intensity_kg_per_kwh)

        # Save args.json
        try:
            args_path = final_folder / "args.json"
            with args_path.open("w") as f:
                json.dump(vars(ns), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write args.json: {e}")

        _write_environment(final_folder)
        _write_manifest(
            final_folder,
            ns,
            acc,
            num_classes,
            run_utc_iso8601=metrics.get("run_utc_iso8601")
        )


if __name__ == "__main__":
    main()