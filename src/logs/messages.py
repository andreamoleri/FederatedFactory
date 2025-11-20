"""
⚙️ Log Message Registry Module
------------------------------

This module acts as a centralized repository for all log message templates and
console output strings used throughout the application.

🧠 Purpose:
    By externalizing string literals into named constants, this module promotes:
    • Consistency in log formatting across different modules
    • Ease of maintenance when modifying output styles
    • Separation of content from application logic

🔧 Core Functionalities:
    • Define standard headers and separators for visual organization
    • Provide format-string templates with named placeholders for dynamic data
      injection (e.g., metrics, paths, progress indicators)

🎯 Intended Use:
    • To be imported by experiment runners, trainers, and utility modules
    • Used with the string `.format()` method to populate placeholders at runtime

📝 Notes:
    The templates utilize Python's standard format specification mini-language
    (e.g., `{loss:.4f}`), enabling precise control over numerical representation.

Author: Andrea Moleri
File Location: src/logs/messages.py
Last Modified: 20/11/2025
"""

# ——— single-experiment / training ———

# Template for the header displayed at the start of a client's processing block.
# Expects: {client_id}
VAE_CLIENT_HEADER = "=== VAE Client {client_id} ==="

# Template for logging Variational Autoencoder (VAE) training progress.
# Expects: {cid} (client ID), {epoch}, {epochs}, {tag}, {loss}
VAE_EPOCH = "  VAE {cid} ep {epoch}/{epochs} ({tag}) loss={loss:.4f}"

# Template for logging Classifier (Clf) training progress.
# Expects: {cid}, {epoch}, {epochs}, {tr_acc} (training accuracy), {val_acc} (validation accuracy)
CLF_EPOCH = "    Clf {cid} ep {epoch}/{epochs} trAcc={tr_acc:.3f} valAcc={val_acc:.3f}"

# Status message indicating the start of report generation.
GENERATING_PDF = "Generating vae_report.pdf …"

# Status message indicating the successful completion of report generation.
PDF_GENERATED = "PDF report generated."

# Template for confirming the persistence of experiment results.
# Expects: {out_dir} (output directory path)
EXPERIMENT_SAVED = "Experiment saved to: {out_dir}"

# ——— grid-search ———

# Notification displayed when initiating a batch of experiments.
# Expects: {n} (number of experiments)
GRID_LAUNCH = "Launching grid-search with {n} experiments …"

# A visual separator line composed of hash characters, used to delimit
# distinct sections in the grid search log output.
GRID_SEPARATOR = "#" * 70

# Header template for an individual experiment within a grid search sequence.
# Expects: {current}, {total}, {dataset}, {latent}, {dp}, {mode}
GRID_EXPERIMENT_HDR = "Experiment {current}/{total}  ({dataset}, L={latent}, DP={dp}, mode={mode})"

# Final completion message for the grid search process.
GRID_COMPLETED = "===== GRID SEARCH COMPLETED ====="

# Template for summarizing the final result of a specific experiment folder.
# Expects: {folder}, {acc} (accuracy)
GRID_RESULT_LINE = "{folder} -> {acc:.4f}"