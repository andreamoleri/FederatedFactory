#!/bin/bash

# ==============================================================================
# SETUP & PATH RESOLUTION
# ==============================================================================
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Get the directory where this script is located (commands/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 2. Navigate up one level to the Project Root (FederatedFactory/)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. Define the specific experiment target directory
#    (The path provided in your request)
TARGET_EXP_DIR="/home/amoleri/PycharmProjects/FederatedFactory/output/diffusion/cifar/seed-42/server-L128/silos/2026-01-07/T19-20-55Z/"

# ==============================================================================
# EXECUTION
# ==============================================================================

echo ">>> 📂 Project Root: $PROJECT_ROOT"
echo ">>> 🎯 Target Experiment: $TARGET_EXP_DIR"

# Check if target exists
if [ ! -d "$TARGET_EXP_DIR" ]; then
    echo "❌ Error: Target directory does not exist: $TARGET_EXP_DIR"
    exit 1
fi

# 4. Export PYTHONPATH
#    This ensures python can find 'src.imports', 'src.metrics', etc.
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# 5. Run the PDF Report Generator
#    --base-dir: Points to the specific run folder. The script logic detects
#                if this folder contains 'datasets'/'metrics' and generates
#                the report specifically for it.
#    --export-figures: (Optional) Saves individual PNGs alongside the PDF.

echo ">>> 🚀 Generating PDF Report..."

python "$PROJECT_ROOT/src/reports/pdf_report.py" \
    --base-dir "$TARGET_EXP_DIR" \
    --export-figures

echo ">>> ✅ Done. Check 'report.pdf' inside the target directory."