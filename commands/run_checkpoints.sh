#!/bin/bash

# ==============================================================================
# SETUP & ROBUSTNESS
# ==============================================================================
set -o errexit
set -o nounset
set -o pipefail

# 1. Robust Project Root Detection
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
current_dir="$SCRIPT_DIR"
PROJECT_ROOT=""
while [[ "$current_dir" != "/" ]]; do
    if [[ -d "$current_dir/src" ]]; then
        PROJECT_ROOT="$current_dir"
        break
    fi
    current_dir="$(dirname "$current_dir")"
done

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "❌ ERROR: Could not locate 'src' directory. Please run this from within the project."
    exit 1
fi

TRAINER_PY="$PROJECT_ROOT/src/checkpoint_trainer.py"
PYTHON_EXEC="$(which python)"

# ==============================================================================
# GPU CONFIGURATION
# ==============================================================================
# Leave empty to auto-detect, or set "0 1" for your 2x H100s
MANUAL_GPU_IDS=""

if [[ -n "$MANUAL_GPU_IDS" ]]; then
    TARGET_GPU_LIST="$MANUAL_GPU_IDS"
else
    # Auto-detect using nvidia-smi
    TARGET_GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')
fi

# Configuration for H100s
# These models (dim=64) are very small. An H100 can handle many in parallel.
# Conservative estimate: 3 jobs per GPU.
JOBS_PER_GPU=5

# Calculate Counts
NUM_GPUS=$(echo "$TARGET_GPU_LIST" | wc -w)
TOTAL_CONCURRENCY=$((NUM_GPUS * JOBS_PER_GPU))

echo ">>> 📊 Configuration: Using $NUM_GPUS GPUs ($TARGET_GPU_LIST)"
echo ">>> 🚀 Concurrency:   $JOBS_PER_GPU jobs/GPU = $TOTAL_CONCURRENCY total parallel jobs"

export NUM_GPUS
export TARGET_GPU_LIST

# ==============================================================================
# DIRECTORIES
# ==============================================================================
LOG_WORK_DIR="$SCRIPT_DIR/run_checkpoints_work"
LOG_STORAGE_DIR="$LOG_WORK_DIR/logs"
DATA_DIR="$PROJECT_ROOT/data"

mkdir -p "$LOG_STORAGE_DIR"
mkdir -p "$DATA_DIR"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# ==============================================================================
# BUILD EXPERIMENT QUEUE
# ==============================================================================
CMD_FILE="$LOG_WORK_DIR/checkpoint_queue.txt"
JOBLOG="$LOG_WORK_DIR/joblog.txt"
> "$CMD_FILE"

DATASETS=("cifar10" "bloodmnist" "pathmnist" "retinamnist" "fed-isic2019")

# 1. Generate Dirichlet Commands (Alpha = 0.1)
for DS in "${DATASETS[@]}"; do
    LOG_FILE="$LOG_STORAGE_DIR/${DS}_dirichlet_0.1.log"
    CMD="$PYTHON_EXEC $TRAINER_PY --dataset $DS --partition dirichlet --alpha 0.1 --data-dir $DATA_DIR > $LOG_FILE 2>&1"
    echo "$CMD" >> "$CMD_FILE"
done

# 2. Generate Silos Commands (Alpha is irrelevant/None)
for DS in "${DATASETS[@]}"; do
    LOG_FILE="$LOG_STORAGE_DIR/${DS}_silos.log"
    CMD="$PYTHON_EXEC $TRAINER_PY --dataset $DS --partition silos --data-dir $DATA_DIR > $LOG_FILE 2>&1"
    echo "$CMD" >> "$CMD_FILE"
done

# ==============================================================================
# EXECUTION
# ==============================================================================
COUNT=$(wc -l < "$CMD_FILE")

if ! command -v parallel &> /dev/null; then
    echo "❌ GNU Parallel is not installed. Please install it (apt install parallel)."
    exit 1
fi

echo ">>> STARTING GENERATION OF FROZEN CORES ($COUNT Configurations)"

parallel --jobs "$TOTAL_CONCURRENCY" \
    --retries 1 \
    --joblog "$JOBLOG" \
    --resume-failed \
    --env PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES \
    --env NUM_GPUS \
    --env TARGET_GPU_LIST \
    --line-buffer \
    '
    JOB_SLOT={%}

    # Round-robin GPU assignment
    IFS=" " read -r -a GPU_ARRAY <<< "$TARGET_GPU_LIST"
    ARRAY_INDEX=$(( (JOB_SLOT - 1) % NUM_GPUS ))
    GPU_ID=${GPU_ARRAY[$ARRAY_INDEX]}

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    echo "🚀 [GPU $GPU_ID] Starting Checkpoint Job {#}: {}"

    # Execute
    eval {}

    status=$?
    if [ $status -eq 0 ]; then
        echo "✅ [GPU $GPU_ID] Job {#} Finished"
    else
        echo "❌ [GPU $GPU_ID] Job {#} Failed"
        exit 1
    fi
    ' :::: "$CMD_FILE"

echo ">>> 🎉 All Frozen Checkpoints Generated."