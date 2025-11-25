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
Last Modified: 21/11/2025
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
from models.baselines import FedAvgBaseline, FedProxBaseline, FedDFBaseline, FedDynBaseline
from models.vae import Decoder
from jobs.baseline_runner import run_federated_baseline, subset_to_tensor
from jobs.experiment_setup import setup_experiment_env, prepare_data
from jobs.generative_phase import run_generative_training
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

def save_predictions_artifact(y_true: np.ndarray, y_pred: np.ndarray, test_set: Any, output_path: Path):
    """
    Salva le previsioni in formato .npy con struttura:
    array strutturato con campi: ['image_name', 'actual_label', 'predicted_label']
    """
    if len(y_true) == 0:
        np.save(output_path, np.zeros(0, dtype=[('image_name', 'U100'), ('actual_label', 'i4'), ('predicted_label', 'i4')]))
        return output_path

    image_names = []
    if hasattr(test_set, 'imgs'):
        image_names = [Path(fp).name for fp, _ in test_set.imgs]
    elif hasattr(test_set, 'samples'):
        image_names = [Path(fp).name for fp, _ in test_set.samples]
    else:
        # Fallback: usa indici numerici
        image_names = [f"image_{i:06d}" for i in range(len(y_true))]

    image_names = image_names[:len(y_true)]

    dtype = [
        ('image_name', 'U100'),
        ('actual_label', 'i4'),
        ('predicted_label', 'i4')
    ]

    predictions_array = np.zeros(len(y_true), dtype=dtype)
    predictions_array['image_name'] = image_names
    predictions_array['actual_label'] = y_true
    predictions_array['predicted_label'] = y_pred

    np.save(output_path, predictions_array)
    return output_path


# ==============================================================================
#  MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(
    args: Any,
    run_id: int | None = None,
    tracker: ExperimentCostTracker | None = None
):
    """
    Executes a single experimental run based on the provided configuration.
    """
    set_seed(args.seed)

    # 1. Setup Environment and Data Loading
    P, time_iso = setup_experiment_env(args, run_id)

    prepared_data = prepare_data(args, torch.device("cpu"))

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

    # =========================================================================
    # PATH A: BASELINE FEDERATED LEARNING
    # =========================================================================
    if args.model.startswith("baseline:"):
        baseline_type = args.model.split(":")[1]
        logger.info(f"[BASELINE] Detected baseline type: {baseline_type}")

        if baseline_type == "fedavg": baseline = FedAvgBaseline(args, num_classes, chans, device)
        elif baseline_type == "fedprox": baseline = FedProxBaseline(args, num_classes, chans, device)
        elif baseline_type == "feddf": baseline = FedDFBaseline(args, num_classes, chans, device)
        elif baseline_type == "feddyn": baseline = FedDynBaseline(args, num_classes, chans, device)
        else: raise ValueError(f"Unknown baseline: {baseline_type}")

        # Map partitions to the expected client/class structure for baseline runner
        if getattr(args, "partition", "silos") == "silos":
            ts_dict = {f"client{d}": {d: train_subsets[i]} for i, d in enumerate(present_classes)}
            test_dict = {f"client{d}": {} for d in present_classes}
        else:
            ts_dict = train_subsets_dict
            test_dict = {}

        # --- Esecuzione Baseline ---
        acc, hist, baseline_metrics, y_true, y_pred = run_federated_baseline(
            baseline, ts_dict, test_dict, base_train_set, args, device, P, tracker
        )

        # Merge metrics
        metrics.update(baseline_metrics)
        metrics["accuracy"] = acc

        # Salva metriche specifiche della baseline
        with open(P.root / "metrics" / "baseline.json", "w") as f:
            json.dump(baseline_metrics, f, indent=2)

    # =========================================================================
    # PATH B: GENERATIVE + CLASSIFIER (Hybrid)
    # =========================================================================
    else:
        # 3. Generative Phase
        perc_loss = VGGPerceptualLoss().to(device) if (chans == 3 and not bool(args.grayscale)) else None
        hist = {"vae_loss": {}}

        gen_models, client_gen_models, client_sample_counts, gen_step_total, gen_start = run_generative_training(
            args, device, P, train_subsets_dict, train_subsets, present_classes, num_classes,
            chans, img_shape, tracker, hist, perc_loss
        )
        gen_time = time.perf_counter() - gen_start
        metrics["vae_steps"] = gen_step_total

        # 4. Classifier Phase
        clf_steps, synth_total, clf_start, y_true, y_pred, trained_clfs, single_clf = run_classifier_training(
            args, device, P, train_subsets_dict, train_subsets, present_classes, num_classes,
            chans, img_shape, tracker, hist,
            gen_models, client_gen_models, client_sample_counts, reserved_test_ld
        )
        clf_time = time.perf_counter() - clf_start

        # 5. Metrics & Evaluation
        acc = ensemble_accuracy(y_true, y_pred)

        gen_metrics = run_evaluation(
            args, device, P, present_classes, reserved_test_imgs_list,
            gen_models, client_gen_models, client_sample_counts, img_shape,
            trained_clfs, single_clf, y_true, y_pred,
            train_subsets_dict, train_subsets, base_train_set, chans
        )
        metrics["gen_metrics"] = gen_metrics
        
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

    # 4. Test Predictions (.npy)
    if len(y_true) > 0:
        predictions_path = P.root / "artifacts" / "test_predictions.npy"
        save_predictions_artifact(y_true, y_pred, test_set, predictions_path)
        logger.info(f"[EXPORT] Test predictions saved to: {predictions_path}")

    # 5. Args.json
    with open(P.root / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # 6. Client Class Distribution CSV (NECESSARIO PER VISUALIZZAZIONE)
    # Questo blocco converte il dizionario 'client_class_distribution' creato in baseline_runner
    # nel formato CSV richiesto: colonne "client", "class_0", "class_1"...
    if "client_class_distribution" in metrics:
        try:
            dist_dir = P.root / "distributions"
            dist_dir.mkdir(parents=True, exist_ok=True)
            csv_path = dist_dir / "client_class_distribution.csv"

            dist_data = metrics["client_class_distribution"]  # dict[client, dict[class, count]]

            # Trova tutte le classi uniche per creare l'header corretto (ordinato)
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
                        # Le chiavi nel JSON sono stringhe
                        row.append(counts.get(str(c), 0))
                    writer.writerow(row)
            
            logger.info(f"[EXPORT] Class distribution CSV saved to: {csv_path}")
        except Exception as e:
            logger.warning(f"[EXPORT] Failed to export class distribution CSV: {e}")

    logger.info(logmsg.EXPERIMENT_SAVED.format(out_dir=str(P.root)))
    
    return acc, P.root, hist, metrics, y_true, y_pred, num_classes
