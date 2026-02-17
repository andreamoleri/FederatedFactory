"""
📊 PoE Ensemble Evaluation and Comparison Module (Multi-Seed Aggregation)
-----------------------------------------------------------------------

This module facilitates the formal evaluation of Product-of-Experts (PoE)
ensemble techniques against individual local classifiers. It analyzes multiple
experiment directories, groups runs by configuration (aggregating over seeds),
and generates statistical proofs with error bars.

🧠 Purpose:
    To provide a rigorous statistical comparison between isolated client
    models (local classifiers) and their aggregated inference (PoE),
    quantifying improvements in accuracy and calibration across multiple seeds.

🔧 Core Functionalities:
    • Recursive scanning of output directories (H100/L40)
    • Grouping of experiments by hyperparameters (ignoring seed)
    • Comparative inference (Local vs PoE)
    • Statistical aggregation (Mean ± Std Dev)
    • Generation of publication-ready plots with error bars
    • Detailed per-client accuracy tables (saved to CSV)

🎯 Intended Use:
    • Research paper rebuttals
    • Validating stability of FL aggregation methods

Author: Andrea Moleri (Updated)
File Location: src/poe_proof.py
"""

import argparse
import json
import sys
import os
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ----------------------------------------------------------------------------
# 0. PATH INJECTION & IMPORTS
# ----------------------------------------------------------------------------
# Ensure the original project modules can be imported
current_file = Path(__file__).resolve()
src_root = current_file.parent  # src/
project_root = src_root.parent  # FederatedFactory/

if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

try:
    from models.cnn import SimpleCNN
    from imports.data_management import get_dataset, prime_dataset_meta_for_transform
    from imports.data_augmentation import build_transform
except ImportError as e:
    print("Import error: ensure this file is located in the 'src/' folder of the project.")
    raise e

# ----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLING
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 300
})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# 2. UTILITIES
# ----------------------------------------------------------------------------

def load_run_args(run_dir: Path) -> argparse.Namespace:
    """Loads args.json from a run directory."""
    args_path = run_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found in {run_dir}")
    with open(args_path, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)


def get_config_signature(args: argparse.Namespace) -> str:
    """
    Creates a unique string signature for an experiment configuration,
    excluding random seeds and paths. Used to group runs.
    """
    # Key parameters that define the experiment logic
    key_params = [
        args.dataset,
        getattr(args, 'partition', 'silos'),
        getattr(args, 'alpha', 'N/A'),
        getattr(args, 'num_clients', 'N/A'),
        args.model,
        args.latent_dim,
        args.dp,
        args.infer_mode
    ]
    return " | ".join(str(p) for p in key_params)


def load_classifiers(run_dir: Path, num_classes: int, channels: int, resolution: int):
    """
    Loads all client classifier models from a run directory.
    
    Now accepts 'resolution' to ensure SimpleCNN is initialized
    with the correct architecture (kernel sizes) matching the checkpoint.
    """
    model_dir = run_dir / "models" / "classifiers"
    models = []
    model_names = []

    if not model_dir.exists():
        return [], []

    # Pattern matching for client-*.pt files
    for model_path in sorted(model_dir.glob("client-*.pt")):
        try:
            # Pass resolution here to ensure correct conv1 kernel size
            model = SimpleCNN(channels, num_classes, input_resolution=resolution).to(DEVICE)
            
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
            model_names.append(model_path.stem)
        except Exception as e:
            print(f"Warning: Failed to load {model_path}: {e}")

    return models, model_names


@torch.no_grad()
def get_poe_metrics(models, dataloader):
    """
    Runs inference using individual models and the PoE ensemble.
    Returns dictionaries of metrics, including per-client breakdowns.
    """
    if not models:
        return None

    # Accumulators
    single_logits_accum = [[] for _ in models]
    all_y_true = []

    # Inference Loop
    for x, y in dataloader:
        x = x.to(DEVICE)
        all_y_true.append(y.cpu())

        for i, model in enumerate(models):
            logits = model(x)
            single_logits_accum[i].append(logits.cpu())

    # Stack results
    y_true = torch.cat(all_y_true).numpy()
    
    # single_logits_all shape: (Num_Models, Num_Samples, Num_Classes)
    single_logits_all = torch.stack([torch.cat(acc, dim=0) for acc in single_logits_accum])

    # 1. Calculate Individual Metrics
    single_probs = torch.softmax(single_logits_all, dim=2).numpy()
    
    local_accuracies = []
    local_aurocs = []

    for i in range(len(models)):
        probs = single_probs[i]
        preds = np.argmax(probs, axis=1)
        
        acc = accuracy_score(y_true, preds)
        local_accuracies.append(acc)

        try:
            if probs.shape[1] > 2:
                auroc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
            else:
                # Binary case handling
                auroc = roc_auc_score(y_true, probs[:, 1])
        except ValueError:
            auroc = 0.5 # Fallback
        local_aurocs.append(auroc)

    # 2. Calculate PoE Metrics
    # PoE Logits = Sum(LogSoftmax(logits_i))
    log_probs_all = torch.log_softmax(single_logits_all, dim=2)
    poe_logits = torch.sum(log_probs_all, dim=0)
    poe_probs = torch.softmax(poe_logits, dim=1).numpy()
    poe_preds = np.argmax(poe_probs, axis=1)

    poe_acc = accuracy_score(y_true, poe_preds)
    try:
        if poe_probs.shape[1] > 2:
            poe_auroc = roc_auc_score(y_true, poe_probs, multi_class='ovr', average='macro')
        else:
            poe_auroc = roc_auc_score(y_true, poe_probs[:, 1])
    except ValueError:
        poe_auroc = 0.5

    # 3. Identify Best Local Model (by Accuracy)
    best_idx = np.argmax(local_accuracies)
    best_acc_val = local_accuracies[best_idx]
    best_auroc_val = local_aurocs[best_idx]

    return {
        "local_acc_avg": np.mean(local_accuracies),
        "local_acc_best": best_acc_val,
        "local_auroc_best": best_auroc_val,
        "poe_acc": poe_acc,
        "local_auroc_avg": np.mean(local_aurocs),
        "poe_auroc": poe_auroc,
        "local_accuracies": local_accuracies,  
        "local_aurocs": local_aurocs           
    }


# ----------------------------------------------------------------------------
# 3. ANALYSIS ORCHESTRATOR
# ----------------------------------------------------------------------------

def analyze_experiment_group(signature, runs, data_dir, output_root):
    """
    Analyzes a group of runs (same config, different seeds).
    Returns aggregated stats AND a list of detailed per-client results.
    """
    print(f"\n>>> 🔬 Analyzing Group: {signature}")
    print(f"    Found {len(runs)} seeds.")

    # Load config from first run to get data parameters
    ref_args = load_run_args(runs[0])
    
    # Setup Data (Loaded once per group to save time, assuming same dataset)
    try:
        prime_dataset_meta_for_transform(ref_args.dataset, data_dir)
        tfm = build_transform(ref_args.dataset, train=False)
        test_set = get_dataset(ref_args.dataset, data_dir, train=False, transform=tfm)
        
        # Determine classes, channels, AND RESOLUTION
        try:
            num_classes = len(test_set.classes)
        except:
            num_classes = len(set(getattr(test_set, 'targets', getattr(test_set, 'labels', []))))
        
        sample_img, _ = test_set[0]
        channels = sample_img.shape[0]
        resolution = sample_img.shape[1]  # Extract resolution from data
        
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"❌ Data loading failed for {ref_args.dataset}: {e}")
        return None, []

    # Storage for aggregation
    agg_metrics = defaultdict(list)
    detailed_results = []  # To store row-level data for CSV

    # Wrap in tqdm but allow printing inside
    pbar = tqdm(runs, desc="Processing Seeds")
    
    for run_dir in pbar:
        try:
            models, model_names = load_classifiers(run_dir, num_classes, channels, resolution)
            if not models:
                continue
            
            m = get_poe_metrics(models, test_loader)
            
            if m:
                # Store aggregated metrics
                keys_to_agg = ["local_acc_avg", "local_acc_best", "local_auroc_best", 
                               "poe_acc", "local_auroc_avg", "poe_auroc"]
                
                for k in keys_to_agg:
                    agg_metrics[k].append(m[k])
                
                # --- PRINT INDIVIDUAL CLIENT TABLE ---
                tqdm.write(f"\nRun: {run_dir.name} ({run_dir.parent.name})")
                
                rows = []
                # Add individual client results to detailed list
                for i, name in enumerate(model_names):
                    acc = m['local_accuracies'][i]
                    auroc = m['local_aurocs'][i]
                    rows.append([name, f"{acc:.4f}", f"{auroc:.4f}"])
                    
                    detailed_results.append({
                        "signature": signature,
                        "dataset": ref_args.dataset,
                        "partition": getattr(ref_args, 'partition', 'silos'),
                        "run_id": run_dir.name,
                        "model_name": name,
                        "type": "local_client",
                        "accuracy": acc,
                        "auroc": auroc
                    })
                
                # Add Summary rows for this run to detailed list
                detailed_results.append({
                    "signature": signature,
                    "dataset": ref_args.dataset,
                    "partition": getattr(ref_args, 'partition', 'silos'),
                    "run_id": run_dir.name,
                    "model_name": "AVG Local",
                    "type": "aggregate",
                    "accuracy": m['local_acc_avg'],
                    "auroc": m['local_auroc_avg']
                })
                detailed_results.append({
                    "signature": signature,
                    "dataset": ref_args.dataset,
                    "partition": getattr(ref_args, 'partition', 'silos'),
                    "run_id": run_dir.name,
                    "model_name": "PoE Ensemble",
                    "type": "ensemble",
                    "accuracy": m['poe_acc'],
                    "auroc": m['poe_auroc']
                })

                rows.append(["-"*20, "-"*8, "-"*8])
                rows.append(["AVG Local", f"{m['local_acc_avg']:.4f}", f"{m['local_auroc_avg']:.4f}"])
                rows.append(["BEST Local", f"{m['local_acc_best']:.4f}", f"{m['local_auroc_best']:.4f}"]) 
                rows.append(["PoE Ensemble", f"{m['poe_acc']:.4f}", f"{m['poe_auroc']:.4f}"])
                
                tqdm.write(tabulate(rows, headers=["Client Model", "Accuracy", "AUROC"], tablefmt="simple"))
                tqdm.write("-" * 50)
                    
            # Clean up VRAM
            for model in models:
                del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            tqdm.write(f"Error in run {run_dir.name}: {e}")
            continue

    if not agg_metrics:
        print("    ⚠️ No valid metrics computed for this group.")
        return None, []

    # Calculate Stats (Mean +/- Std)
    stats = {}
    for k, v in agg_metrics.items():
        stats[f"{k}_mean"] = np.mean(v)
        stats[f"{k}_std"] = np.std(v)
        stats[f"{k}_raw"] = v # Keep raw data for advanced plotting if needed

    # Generate Plot for this group
    generate_group_plot(stats, signature, output_root, ref_args)
    
    return stats, detailed_results


def generate_group_plot(stats, signature, output_root, args):
    """Generates a bar chart with error bars for a specific experiment group."""
    
    # Setup data
    metrics = ['Accuracy', 'AUROC']
    
    # Extract data
    local_avg_means = [stats['local_acc_avg_mean'], stats['local_auroc_avg_mean']]
    local_avg_stds  = [stats['local_acc_avg_std'], stats['local_auroc_avg_std']]
    
    # [FIX] Now use real stats for Best Local AUROC
    local_best_means = [stats['local_acc_best_mean'], stats['local_auroc_best_mean']]
    local_best_stds  = [stats['local_acc_best_std'], stats['local_auroc_best_std']]
    
    poe_means = [stats['poe_acc_mean'], stats['poe_auroc_mean']]
    poe_stds  = [stats['poe_acc_std'], stats['poe_auroc_std']]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Bars
    rects1 = ax.bar(x - width, local_avg_means, width, yerr=local_avg_stds, 
                    label='Avg Local', capsize=5, color='#A9A9A9', alpha=0.9)
    
    ax.bar(x, local_best_means, width, yerr=local_best_stds, 
           label='Best Local', capsize=5, color='#4682B4', alpha=0.9)
           
    rects3 = ax.bar(x + width, poe_means, width, yerr=poe_stds, 
                    label='PoE (Ours)', capsize=5, color='#DC143C', alpha=0.9)

    # Styling
    ax.set_ylabel('Score')
    ax.set_title(f'PoE Proof: {args.dataset} ({args.partition})')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Annotations
    def autolabel(rects, stds):
        for rect, std in zip(rects, stds):
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}\n(±{std:.3f})',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    autolabel(rects1, local_avg_stds)
    autolabel(rects3, poe_stds)

    # Safe filename
    safe_sig = signature.replace(" | ", "_").replace("/", "-").replace(":", "-")
    safe_sig = safe_sig[:100] # Trim length
    out_path = output_root / f"poe_proof_{safe_sig}.png"
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"    📈 Plot saved to: {out_path}")


# ----------------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze PoE performance across multiple seeds.")
    
    # Default paths based on user prompt
    default_dirs = [
        str(project_root.parent / "FederatedFactory" / "federatedfactory_output_H100"),
        str(project_root.parent / "FederatedFactory" / "federatedfactory_output_L40")
    ]
    
    parser.add_argument("--search-dirs", nargs='+', default=default_dirs,
                        help="List of root directories to recursively search for experiment runs.")
    parser.add_argument("--data-dir", type=str, default=str(project_root / "data"),
                        help="Path to the dataset directory.")
    parser.add_argument("--out-dir", type=str, default=str(project_root / "analysis_results"),
                        help="Where to save the consolidated report and plots.")
    
    args = parser.parse_args()
    
    # 1. Scanning
    print(">>> 🔍 Scanning for experiments...")
    experiments = defaultdict(list)
    
    for d in args.search_dirs:
        root_path = Path(d)
        if not root_path.exists():
            print(f"⚠️ Warning: Directory not found: {root_path}")
            continue
            
        # Find all args.json files
        for args_file in root_path.rglob("args.json"):
            run_dir = args_file.parent
            
            # Check if this run finished (has models)
            if not (run_dir / "models" / "classifiers").exists():
                continue
                
            try:
                run_args = load_run_args(run_dir)
                # Create a signature that excludes SEED
                sig = get_config_signature(run_args)
                experiments[sig].append(run_dir)
            except Exception as e:
                pass

    if not experiments:
        print("❌ No valid experiment runs found.")
        return

    print(f"✅ Found {len(experiments)} unique configurations across {sum(len(v) for v in experiments.values())} total runs.")

    # 2. Analysis Loop
    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    summary_table = []
    all_detailed_results = []
    
    headers = ["Dataset", "Partition", "Alpha", "Mode", "N_Seeds", 
               "Local Acc (Avg)", "PoE Acc", "Gain (%)", "PoE AUROC"]

    for sig, runs in experiments.items():
        stats, details = analyze_experiment_group(sig, runs, args.data_dir, output_root)
        
        if details:
            all_detailed_results.extend(details)
        
        if stats:
            # Parse signature back for table
            parts = sig.split(" | ")
            # [Dataset, Partition, Alpha, Clients, Model, Latent, DP, Mode]
            
            gain = (stats['poe_acc_mean'] - stats['local_acc_avg_mean']) * 100
            
            row = [
                parts[0], # Dataset
                parts[1], # Partition
                parts[2], # Alpha
                parts[7], # Mode (Local/Server)
                len(runs),
                f"{stats['local_acc_avg_mean']:.4f} ±{stats['local_acc_avg_std']:.3f}",
                f"{stats['poe_acc_mean']:.4f} ±{stats['poe_acc_std']:.3f}",
                f"{gain:+.2f}",
                f"{stats['poe_auroc_mean']:.4f} ±{stats['poe_auroc_std']:.3f}"
            ]
            summary_table.append(row)

    # 3. Final Report (Console)
    print("\n" + "="*80)
    print("📊 POE VALIDATION SUMMARY REPORT")
    print("="*80)
    print(tabulate(summary_table, headers=headers, tablefmt="github"))
    
    # 4. Save Summaries to CSV
    
    # A) Summary CSV
    summary_csv_path = output_root / "poe_summary_report.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(summary_table)
        
    # B) Detailed Client CSV
    detailed_csv_path = output_root / "poe_detailed_clients.csv"
    if all_detailed_results:
        keys = all_detailed_results[0].keys()
        with open(detailed_csv_path, "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_detailed_results)
    
    print(f"\n✅ Analysis Complete.")
    print(f"   Summary CSV saved to: {summary_csv_path}")
    print(f"   Detailed Client CSV saved to: {detailed_csv_path}")

if __name__ == "__main__":
    main()
