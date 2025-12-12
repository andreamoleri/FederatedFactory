#!/bin/bash

# ==============================================================================
# NEURIPS / ICML REPRODUCIBILITY SCRIPT
# ==============================================================================
# Description: Executes the hyperparameter sweep described in Section 5.1.
# - Seeds: {1..5}
# - Budget: 200 Rounds x 5 Local Epochs
# - Grid Search: FedProx (mu), FedDyn (alpha)
# ==============================================================================

set -o errexit
set -o pipefail
set -o nounset

# --- PATH SETUP ---
# Robustly find the project root regardless of where the script is called from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assuming script is in /commands, root is one level up
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MAIN_PY="$PROJECT_ROOT/src/main.py"
PYTHON_EXEC="$(which python)"

# Define Log Directories
LOG_WORK_DIR="$PROJECT_ROOT/logs/neurips_sweep"
LOG_STORAGE_DIR="$LOG_WORK_DIR/console_logs"
mkdir -p "$LOG_STORAGE_DIR"

# --- EXPERIMENTAL CONSTANTS (From Latex Section 5.1) ---
SEEDS=(1 2 3 4 5)
ROUNDS=200
LOCAL_EPOCHS=5
BATCH_SIZE=128
CLIENTS=10          # "Cross-Silo" usually implies fewer, larger clients.
FRACTION=1.0        # Full participation (Section 5.1.2)
PATIENCE=20         # Slightly relaxed patience for difficult seeds

# Data location (Override if needed via env var)
DATA_DIR="${MY_DATA_DIR:-$PROJECT_ROOT/data}"

# Grid Search Values (From Latex Section 5.1.2)
# "We performed a validation grid search for regularization coefficients {1e-3, 1e-2, 1e-1, 1.0}"
REGULARIZATION_GRID=("0.001" "0.01" "0.1" "1.0")

# Datasets to benchmark
# Syntax: "dataset_name|input_size"
DATASETS=(
  "cifar|32"
  "medmnist:pathmnist|28"
  "fed_camelyon16|224"
)

# ==============================================================================
# 1. GPU & CONCURRENCY CONFIGURATION
# ==============================================================================
# Optimization flags for PyTorch
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Detect GPUs automatically
TARGET_GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')
if [[ -z "$TARGET_GPU_LIST" ]]; then
    echo "⚠️  No GPUs detected. Defaulting to CPU (GPU 0 placeholder)."
    TARGET_GPU_LIST="0"
fi

# Count GPUs
NUM_GPUS=$(echo "$TARGET_GPU_LIST" | wc -w)

# Config: How many distinct experiments to run on a SINGLE GPU at once?
# For ResNet50 on 224x224 (Camelyon), keep this to 1. For CIFAR, maybe 2.
JOBS_PER_GPU=1
TOTAL_CONCURRENCY=$((NUM_GPUS * JOBS_PER_GPU))

# Export for GNU Parallel
export NUM_GPUS
export TARGET_GPU_LIST

echo ">>> 🖥️  Hardware Detected: $NUM_GPUS GPUs."
echo ">>> 🚀 Concurrency Level: $TOTAL_CONCURRENCY jobs in parallel."

# ==============================================================================
# 2. JOB QUEUE GENERATION
# ==============================================================================
CMD_FILE="$LOG_WORK_DIR/queue.txt"
JOBLOG="$LOG_WORK_DIR/joblog.txt"

# Clear previous queue generation (but keep joblog for resume capability)
> "$CMD_FILE"

echo ">>> 📝 Generating Experimental Queue..."

JOB_COUNT=1

for SEED in "${SEEDS[@]}"; do
  for DS_ENTRY in "${DATASETS[@]}"; do
    IFS="|" read -r DATASET INPUT_SIZE <<< "$DS_ENTRY"

    # Sanitize dataset name for file paths (replace colons with underscores)
    SAFE_NAME="${DATASET//:/_}"

    # Common arguments for ALL baselines (Ceteris Paribus)
    COMMON_ARGS="--dataset \"$DATASET\" --input-size $INPUT_SIZE \
                 --data-dir \"$DATA_DIR\" \
                 --num-clients $CLIENTS --client-fraction $FRACTION \
                 --baseline-max-rounds $ROUNDS --baseline-epochs-per-round $LOCAL_EPOCHS \
                 --baseline-patience $PATIENCE --batch-size $BATCH_SIZE \
                 --seed $SEED --partition dirichlet --alpha 0.5"

    # --- TRACK 1: FedAvg (Standard Baseline) ---
    OUT_DIR="$PROJECT_ROOT/output/neurips/fedavg/${SAFE_NAME}/seed_${SEED}"
    LOG_FILE="$LOG_STORAGE_DIR/job${JOB_COUNT}_fedavg_${SAFE_NAME}_s${SEED}.log"

    # Note: We use > log 2>&1 to capture output per job without cluttering main console
    echo "mkdir -p \"$OUT_DIR\" && $PYTHON_EXEC $MAIN_PY $COMMON_ARGS \
          --model baseline:fedavg \
          --learning-rate 0.1 --baseline-momentum 0.9 \
          --out-dir \"$OUT_DIR\" > \"$LOG_FILE\" 2>&1" >> "$CMD_FILE"
    ((JOB_COUNT++))

    # --- TRACK 2: FedProx (Grid Search) ---
    for MU in "${REGULARIZATION_GRID[@]}"; do
        OUT_DIR="$PROJECT_ROOT/output/neurips/fedprox/mu_${MU}/${SAFE_NAME}/seed_${SEED}"
        LOG_FILE="$LOG_STORAGE_DIR/job${JOB_COUNT}_fedprox_mu${MU}_${SAFE_NAME}_s${SEED}.log"

        echo "mkdir -p \"$OUT_DIR\" && $PYTHON_EXEC $MAIN_PY $COMMON_ARGS \
              --model baseline:fedprox \
              --learning-rate 0.1 --baseline-momentum 0.9 \
              --fedprox-mu $MU \
              --out-dir \"$OUT_DIR\" > \"$LOG_FILE\" 2>&1" >> "$CMD_FILE"
        ((JOB_COUNT++))
    done

    # --- TRACK 3: FedDyn (Grid Search) ---
    for ALPHA in "${REGULARIZATION_GRID[@]}"; do
        OUT_DIR="$PROJECT_ROOT/output/neurips/feddyn/alpha_${ALPHA}/${SAFE_NAME}/seed_${SEED}"
        LOG_FILE="$LOG_STORAGE_DIR/job${JOB_COUNT}_feddyn_alpha${ALPHA}_${SAFE_NAME}_s${SEED}.log"

        echo "mkdir -p \"$OUT_DIR\" && $PYTHON_EXEC $MAIN_PY $COMMON_ARGS \
              --model baseline:feddyn \
              --learning-rate 0.1 --baseline-momentum 0.9 \
              --feddyn-alpha $ALPHA \
              --out-dir \"$OUT_DIR\" > \"$LOG_FILE\" 2>&1" >> "$CMD_FILE"
        ((JOB_COUNT++))
    done

    # --- TRACK 4: SCAFFOLD (Special Constraints) ---
    # Constraint: Momentum=0.0 per Latex Section 5.1.2
    OUT_DIR="$PROJECT_ROOT/output/neurips/scaffold/${SAFE_NAME}/seed_${SEED}"
    LOG_FILE="$LOG_STORAGE_DIR/job${JOB_COUNT}_scaffold_${SAFE_NAME}_s${SEED}.log"

    echo "mkdir -p \"$OUT_DIR\" && $PYTHON_EXEC $MAIN_PY $COMMON_ARGS \
          --model baseline:scaffold \
          --learning-rate 0.1 \
          --baseline-momentum 0.0 \
          --out-dir \"$OUT_DIR\" > \"$LOG_FILE\" 2>&1" >> "$CMD_FILE"
    ((JOB_COUNT++))

  done
done

TOTAL_JOBS=$(wc -l < "$CMD_FILE")
echo ">>> ✅ Queue Ready: $TOTAL_JOBS experiments defined."

# ==============================================================================
# 3. PARALLEL EXECUTION
# ==============================================================================

# Check for GNU Parallel
if ! command -v parallel &> /dev/null; then
    echo "❌ GNU Parallel is not installed. Please install it (sudo apt-get install parallel)."
    exit 1
fi

echo ">>> 🏁 STARTING EXECUTION..."
echo ">>> Monitor progress via: tail -f $JOBLOG"

parallel --jobs "$TOTAL_CONCURRENCY" \
    --joblog "$JOBLOG" \
    --resume-failed \
    --env PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES \
    --env NUM_GPUS \
    --env TARGET_GPU_LIST \
    --bar \
    '
    # Capture the job slot number (1..N) provided by Parallel
    JOB_SLOT={%}

    # --- DYNAMIC GPU ASSIGNMENT LOGIC ---
    # 1. Read the space-separated list of GPUs into an array
    IFS=" " read -r -a GPU_ARRAY <<< "$TARGET_GPU_LIST"

    # 2. Map the Job Slot to an array index using modulo
    #    (Slot 1 -> Index 0, Slot 2 -> Index 1, Slot 3 -> Index 0...)
    ARRAY_INDEX=$(( (JOB_SLOT - 1) % NUM_GPUS ))

    # 3. Extract the actual Physical GPU ID from the array
    GPU_ID=${GPU_ARRAY[$ARRAY_INDEX]}

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    # Execute the command passed from the text file
    eval {}

    # Capture exit code for the joblog
    exit $?
    ' :::: "$CMD_FILE"

echo ">>> 🎉 All NeurIPS Sweeps Completed Successfully."