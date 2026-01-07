"""
🧪 Experiment Orchestration Module
----------------------------------

This module serves as the central execution harness for conducting federated
learning and generative modeling experiments. It orchestrates the complete
lifecycle of an experimental run, from environment initialization and data
partitioning to model training, evaluation, and metric persistence.

🧠 Purpose:
    Designed to facilitate rigorous academic research by providing a unified
    interface for running both baseline federated algorithms (e.g., FedAvg,
    FedProx) and novel generative-classifier hybrid architectures.

🔧 Core Functionalities:
    • Initialize deterministic experimental environments (seeding, logging)
    • Prepare and partition datasets for simulated federated settings
    • Execute baseline federated learning algorithms with early-exit logic
    • Orchestrate two-phase training: Generative (VAE/GAN) followed by Classifier
    • Compute and serialize comprehensive performance metrics and artifacts
    • Delegate evaluation and dataset export to the evaluation phase

🎯 Intended Use:
    • High-performance computing (HPC) cluster jobs
    • Reproducible research pipelines
    • Benchmarking of federated learning strategies

File Interactions:
    • Consumes configuration arguments via the `args` namespace
    • Delegates training tasks to `jobs.generative_phase` and `jobs.classifier_phase`
    • Writes structured logs, JSON metrics, and Numpy arrays to the file system

Author: Andrea Moleri
File Location: src/jobs/experiment_runner.py
Last Modified: 12/12/2025
"""
from __future__ import annotations
import logging
import time
import numpy as np
import json
import csv
import torch
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Importazioni da moduli locali
from utils import set_seed, VGGPerceptualLoss
from metrics.costs import ExperimentCostTracker
from models.baselines import FedAvgBaseline, FedProxBaseline, FedDFBaseline, FedDynBaseline, ScaffoldBaseline
from models.vae import Decoder
from jobs.baseline_runner import run_federated_baseline, subset_to_tensor
from jobs.experiment_setup import setup_experiment_env, prepare_data
from jobs.generative_phase import run_generative_training, load_generative_checkpoints
from jobs.classifier_phase import (
    run_classifier_training,
    ensemble_accuracy
)
from jobs.evaluation_phase import run_evaluation
from logs import messages as logmsg


logger = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_predictions_artifact(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, test_set: Any,
                              output_path: Path):
    """
    Saves predictions AND probabilities to a compressed .npz file.
    Output Path should end in .npz
    """
    # Create valid image names map if possible
    image_names = []
    if hasattr(test_set, 'imgs'):
        image_names = [Path(fp).name for fp, _ in test_set.imgs]
    elif hasattr(test_set, 'samples'):
        image_names = [Path(fp).name for fp, _ in test_set.samples]

    # Ensure lengths match (handling subsetting/lazy loading)
    limit = len(y_true)
    # Handle case where test set might be empty or lazy loaded differently
    if len(image_names) >= limit:
        image_names = np.array(image_names[:limit])
    else:
        image_names = np.array([f"img_{i}" for i in range(limit)])

    # Save exactly like the reference code so the notebook can parse it
    np.savez_compressed(
        output_path,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,  # Critical for AUROC
        image_names=image_names
    )
    return output_path


# ==============================================================================
#  MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(args: Any, run_id: int | None = None, tracker: ExperimentCostTracker | None = None):
    set_seed(args.seed)  # Sets seed {1..5} for Classifier initialization and Sampling

    # 1. Setup Environment
    P, time_iso = setup_experiment_env(args, run_id)

    # 2. Prepare Data
    # CRITICAL: If loading checkpoints, we MUST use the seed they were created with (42)
    # to ensure the data partitions match the trained models.
    # If standard training, we use the experiment seed.
    if args.checkpoint_epoch_family is not None:
        part_seed = 42
        logger.info(f"[EXPERIMENT] Using FIXED Partition Seed 42 to align with loaded checkpoints.")
    else:
        part_seed = args.seed

    prepared_data = prepare_data(args, torch.device("cpu"), partition_seed=part_seed)

    (base_train_set, test_set, tfm, num_classes, img_shape, chans,
     train_subsets_dict, train_subsets, present_classes,
     reserved_test_ld, reserved_test_imgs_list, test_imgs_tensor) = prepared_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Inizializzazione Metrics Scallari ---
    metrics: Dict[str, float | int | str | Dict[str, float | int]] = {
        "mode": args.infer_mode,
        "dp": bool(args.dp),
        "seed": args.seed,
        "latent_dim": args.latent_dim,
        "model": args.model,
        "partition": getattr(args, "partition", "silos"),
        "aggregation": getattr(args, "aggregation", "simple"),
        "run_utc_iso8601": time_iso,
    }

    # Inizializzazione variabili per evitare errori nel salvataggio finale
    gen_step_total = 0
    clf_steps = 0
    synth_total = 0
    clf_time = 0.0
    gen_time = 0.0

    # Calcolo totale immagini reali usate
    total_real_images = sum(
        len(subset) for client_subsets in train_subsets_dict.values() for subset in client_subsets.values()) \
        if getattr(args, "partition", "silos") in ["skew", "dirichlet"] else sum(len(s) for s in train_subsets)

    # Initialize empty arrays to prevent UnboundLocalError
    y_true, y_pred, y_probs = np.array([]), np.array([]), np.array([])

    # =========================================================================
    # PATH A: BASELINE FEDERATED LEARNING
    # =========================================================================
    if args.model.startswith("baseline:"):
        baseline_type = args.model.split(":")[1]
        logger.info(f"[BASELINE] Detected baseline type: {baseline_type}")

        if baseline_type == "fedavg":
            baseline = FedAvgBaseline(args, num_classes, chans, device)
        elif baseline_type == "fedprox":
            baseline = FedProxBaseline(args, num_classes, chans, device)
        elif baseline_type == "feddf":
            baseline = FedDFBaseline(args, num_classes, chans, device)
        elif baseline_type == "feddyn":
            baseline = FedDynBaseline(args, num_classes, chans, device)
        elif baseline_type == "scaffold":
            baseline = ScaffoldBaseline(args, num_classes, chans, device)
        else:
            raise ValueError(f"Unknown baseline: {baseline_type}")

        # Map partitions
        if getattr(args, "partition", "silos") == "silos":
            # FIX: Added underscore to match checkpoint_trainer convention (client_0 vs client0)
            ts_dict = {f"client_{d}": {d: train_subsets[i]} for i, d in enumerate(present_classes)}
            test_dict = {f"client_{d}": {} for d in present_classes}
        else:
            ts_dict = train_subsets_dict
            test_dict = {}

        ## --- UPDATED CALL: Expecting y_probs ---
        acc, hist, baseline_metrics, y_true, y_pred, y_probs = run_federated_baseline(
            baseline=baseline,
            train_subsets_dict=ts_dict,
            test_subsets_dict=test_dict,
            base_train_set=base_train_set,
            args=args,
            device=device,
            P=P,
            tracker=tracker,
            test_loader_override=reserved_test_ld,
            train_transform_override=tfm,
            eval_transform_override=None
        )

        # ✅ Update the main metrics dictionary with baseline results
        metrics["accuracy"] = acc
        metrics.update(baseline_metrics)

    # =========================================================================
    # PATH B: GENERATIVE + CLASSIFIER (Hybrid)
    # =========================================================================
    else:
        # 3. Generative Phase
        perc_loss = VGGPerceptualLoss().to(device) if (chans == 3 and not bool(args.grayscale)) else None
        hist = {"vae_loss": {}}
        gen_metrics = {}  # Initialize empty metrics to prevent NameError if evaluation is skipped

        # Update partitioning map for Silos to match checkpoint naming (client_0)
        # This ensures the loader finds the folders created by checkpoint_trainer.py
        if getattr(args, "partition", "silos") == "silos":
             # We reconstruct the dict with correct keys "client_X" instead of "clientX"
             # Only needed if train_subsets_dict was not already set correctly by prepare_data
             # (prepare_data returns a list for silos, so we map it here)
             train_subsets_dict = {f"client_{d}": {d: train_subsets[i]} for i, d in enumerate(present_classes)}

        # BRANCHING LOGIC
        if args.checkpoint_epoch_family is not None:
            # LOAD MODE
            gen_step_total = 0
            gen_start = time.perf_counter()  # Minimal time
            logger.info(f"[GEN-PHASE] Attempting to load checkpoints with Epoch Family: {args.checkpoint_epoch_family}")

            gen_models, client_gen_models, client_sample_counts = load_generative_checkpoints(
                args, device, P, train_subsets_dict, present_classes, chans, img_shape, args.checkpoint_epoch_family
            )

            # --- SAFETY CHECK ---
            loaded_count = sum(1 for m in gen_models if m is not None)
            if loaded_count == 0:
                error_msg = (
                    f"❌ CRITICAL ERROR: 0 Checkpoints loaded for family '{args.checkpoint_epoch_family}'.\n"
                    f"   The code looked in folders like 'checkpoints/{args.dataset}/silos/client_0/...'\n"
                    f"   Please verify:\n"
                    f"   1. The folder structure exists.\n"
                    f"   2. The filenames contain the string '{args.checkpoint_epoch_family}' (e.g. training-state-0025001.pt).\n"
                    f"   3. The client names match (client_0 vs client0)."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            # --------------------

            # Populate sample counts since we skipped training return
            for cname, subsets in train_subsets_dict.items():
                if cname not in client_sample_counts: client_sample_counts[cname] = {}
                for cid, sub in subsets.items():
                    client_sample_counts[cname][cid] = len(sub)
        else:
            # TRAIN MODE
            gen_models, client_gen_models, client_sample_counts, gen_step_total, gen_start = run_generative_training(
                args, device, P, train_subsets_dict, train_subsets, present_classes, num_classes,
                chans, img_shape, tracker, hist, perc_loss
            )

        gen_time = time.perf_counter() - gen_start
        metrics["vae_steps"] = gen_step_total

        # 4. Evaluation Phase (Optional - can be expensive)
        # Uncomment the following lines if you want to compute FID/KID metrics
        # gen_metrics = run_evaluation(
        #     args, device, P, present_classes, reserved_test_imgs_list,
        #     gen_models, client_gen_models, client_sample_counts, img_shape,
        #     [], None, [], [], train_subsets_dict, train_subsets, base_train_set, chans
        # )

        # 5. Classifier Phase (CRITICAL: Generates 'acc')
        (clf_steps, synth_total, clf_start, y_true, y_pred, y_probs,
         trained_clfs, single_clf, synthetic_cache) = run_classifier_training(
            args, device, P, train_subsets_dict, train_subsets, present_classes,
            num_classes, chans, img_shape, tracker, hist, gen_models,
            client_gen_models, client_sample_counts, reserved_test_ld
        )

        clf_time = time.perf_counter() - clf_start
        acc = ensemble_accuracy(y_true, y_pred)

        if args.model == "vae" and len(gen_models) > 0 and gen_models[0] is not None:
            from jobs.generative_phase import _module_size_mb
            metrics.setdefault("decoder_mb", _module_size_mb(gen_models[0]))

        metrics["real_real_fid"] = gen_metrics.get("real_real_fid", {})
        metrics["real_real_kid"] = gen_metrics.get("real_real_kid", {})
        metrics["accuracy"] = acc

    # ==========================================================================
    # FINAL ARTIFACTS SERIALIZATION
    # ==========================================================================

    metrics.update({
        "classifier_training_time_sec": clf_time,
        "classifier_steps": clf_steps,
        "synthetic_images_generated": synth_total,
        "real_images_used": total_real_images,
        "test_images": len(test_imgs_tensor),
        "total_training_time_sec": gen_time + clf_time,
    })

    # 1. Classifier.json
    clf_metrics_data = {
        "model": args.model,
        "dataset": args.dataset,
        "mode": args.infer_mode,
        "partition": metrics["partition"],
        "aggregation": metrics["aggregation"],
        "dp": bool(args.dp),
        "seed": int(args.seed),
        "latent_dim": int(args.latent_dim),
        "input_size": int(getattr(args, "input_size", 0) or 0),
        "accuracy": float(metrics["accuracy"]),
        "classifier_steps": int(metrics["classifier_steps"]),
        "classifier_training_time_sec": float(metrics["classifier_training_time_sec"]),
        "synthetic_images_generated": int(metrics["synthetic_images_generated"]),
        "real_images_used": int(metrics["real_images_used"]),
        "test_images": int(metrics["test_images"]),
        "total_training_time_sec": float(metrics["total_training_time_sec"]),
    }

    if args.model.startswith("baseline:"):
        clf_metrics_data["baseline_type"] = args.model.split(":")[1]
        if "best_round" in metrics:
            clf_metrics_data["best_round"] = metrics["best_round"]

    with open(P.root / "metrics" / "classifier.json", "w") as f:
        json.dump(clf_metrics_data, f, indent=2)

    # 2. History.csv
    hist_csv = P.root / "metrics" / "history.csv"
    with open(hist_csv, "w") as f:
        f.write("step,split_or_client,metric,value\n")
        for metric_name, series_by_key in hist.items():
            for key, values in series_by_key.items():
                try:
                    if hasattr(values, '__iter__') and not isinstance(values, (str, dict)):
                        for step, val in enumerate(values):
                            f.write(f"{step},{key},{metric_name},{float(val)}\n")
                    else:
                        f.write(f"0,{key},{metric_name},{float(values)}\n")
                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not write history entry {metric_name}/{key}: {e}")
                    continue

    # 3. Confusion Matrix
    if len(y_true) > 0 and len(y_pred) > 0:
        actual_classes = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=actual_classes)
        np.savetxt(P.root / "metrics" / "confusion-matrix.csv", cm, delimiter=",", fmt="%d")
    else:
        np.savetxt(P.root / "metrics" / "confusion-matrix.csv", np.array([[0]]), delimiter=",", fmt="%d")

    # 4. Test Predictions (.npz) - UPDATED
    if len(y_true) > 0:
        # Changed to .npz to reflect that it contains multiple arrays
        predictions_path = P.root / "artifacts" / "predictions.npz"
        # Passing y_probs here
        save_predictions_artifact(y_true, y_pred, y_probs, test_set, predictions_path)
        logger.info(f"[EXPORT] Test predictions saved to: {predictions_path}")

    # 5. Args.json
    with open(P.root / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 6. Client Class Distribution CSV
    if "client_class_distribution" in metrics:
        try:
            dist_dir = P.root / "distributions"
            dist_dir.mkdir(parents=True, exist_ok=True)
            csv_path = dist_dir / "client_class_distribution.csv"

            dist_data = metrics["client_class_distribution"]

            all_classes_int = set()
            for c_counts in dist_data.values():
                for k in c_counts.keys():
                    try:
                        all_classes_int.add(int(k))
                    except (ValueError, TypeError):
                        pass
            sorted_classes = sorted(list(all_classes_int))

            header = ["client"] + [f"class_{c}" for c in sorted_classes]

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for client_name, counts in dist_data.items():
                    row = [client_name]
                    for c in sorted_classes:
                        row.append(counts.get(str(c), 0))
                    writer.writerow(row)

            logger.info(f"[EXPORT] Class distribution CSV saved to: {csv_path}")
        except Exception as e:
            logger.warning(f"[EXPORT] Failed to export class distribution CSV: {e}")

    logger.info(logmsg.EXPERIMENT_SAVED.format(out_dir=str(P.root)))

    # Returning y_probs as well
    return acc, P.root, hist, metrics, y_true, y_pred, num_classes