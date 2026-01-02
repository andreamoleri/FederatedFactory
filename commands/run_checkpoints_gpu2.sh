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
# Find torchrun relative to python executable or in path
TORCHRUN_EXEC="$(dirname "$PYTHON_EXEC")/torchrun"
if [[ ! -f "$TORCHRUN_EXEC" ]]; then
    TORCHRUN_EXEC="torchrun"
fi

# ==============================================================================
# GPU & UTILIZATION CONFIGURATION
# ==============================================================================
# Optimization flags
export PYTHONUNBUFFERED=1
export MAX_WORKERS=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_MODULE_LOADING=LAZY

# Force IPv4 to stop the socket address errors
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo

# ------------------------------------------------------------------------------
# [CONFIG] SELECT YOUR GPUS HERE
# ------------------------------------------------------------------------------
# Option A: Set specific GPUs, e.g., "0,1" (COMMA SEPARATED)
# Option B: Leave empty "" to auto-detect.
MANUAL_GPU_IDS=""

# ------------------------------------------------------------------------------
# LOGIC: DETERMINE TARGET LIST
# ------------------------------------------------------------------------------
if [[ -n "$MANUAL_GPU_IDS" ]]; then
    echo ">>> ⚙️  User manually selected GPUs: $MANUAL_GPU_IDS"
    # Ensure comma separation for CUDA_VISIBLE_DEVICES
    COMMA_GPU_LIST="$MANUAL_GPU_IDS"
else
    echo ">>> 🤖 Auto-detecting all available GPUs..."
    # Get list as comma separated (0,1)
    COMMA_GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')

    if [[ -z "$COMMA_GPU_LIST" ]]; then
        echo ">>> ⚠️  Auto-detect failed, defaulting to GPU 0"
        COMMA_GPU_LIST="0"
    fi
fi

# Count GPUs by counting commas + 1
NUM_GPUS=$(echo "$COMMA_GPU_LIST" | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

# --- CRITICAL CHANGE FOR DDP ---
# We export visibility GLOBALLY so torchrun sees all devices.
export CUDA_VISIBLE_DEVICES="$COMMA_GPU_LIST"

echo ">>> 📊 Configuration: DDP Enabled on $NUM_GPUS GPUs ($COMMA_GPU_LIST)"
echo ">>> 🚀 Strategy:      Running 1 Job at a time, utilizing ALL GPUs."

# ==============================================================================
# ORGANIZATION
# ==============================================================================
LOG_WORK_DIR="$SCRIPT_DIR/run_checkpoints_work_ddp"
LOG_STORAGE_DIR="$LOG_WORK_DIR/logs"

mkdir -p "$LOG_STORAGE_DIR"
echo ">>> 📂 Internal queues stored in: $LOG_WORK_DIR"
echo ">>> 📝 Console logs stored in:    $LOG_STORAGE_DIR"

if [[ -n "${MY_DATA_DIR:-}" ]]; then
    DATA_DIR="$MY_DATA_DIR"
elif [[ -d "$PROJECT_ROOT/data" ]]; then
    DATA_DIR="$PROJECT_ROOT/data"
else
    mkdir -p "$PROJECT_ROOT/data"
    DATA_DIR="$PROJECT_ROOT/data"
fi
echo ">>> ✅  Using data directory: $DATA_DIR"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# ==============================================================================
# EXPERIMENT CONFIG & SMART RESUME FILES
# ==============================================================================
CMD_FILE="$LOG_WORK_DIR/checkpoint_queue.txt"
PREV_CMD_FILE="$LOG_WORK_DIR/.checkpoint_queue.last"
JOBLOG="$LOG_WORK_DIR/joblog.txt"

# Reset queue
> "$CMD_FILE"

# --- DATASET CONFIGURATION ---
DATASETS=(
    # "cifar"
    # "medmnist:bloodmnist"
    # "medmnist:pathmnist"
    # "medmnist:retinamnist"
    "fed_isic2019"
)

# ==============================================================================
# GENERATE COMMANDS (MODIFIED FOR TORCHRUN)
# ==============================================================================
echo ">>> GENERATING CHECKPOINT COMMANDS..."

JOB_COUNT=1

# Generate a random port to prevent collisions if script is restarted quickly
MASTER_PORT=$(shuf -i 29500-29999 -n 1)

for DS in "${DATASETS[@]}"; do

    SAFE_DS="${DS//:/_}"

    # --------------------------------------------------------------------------
    # Silos Commands (DDP ENABLED)
    # --------------------------------------------------------------------------
    LOG_FILENAME="job${JOB_COUNT}_${SAFE_DS}_silos.log"
    LOG_FILE="$LOG_STORAGE_DIR/$LOG_FILENAME"

    # CRITICAL CHANGE: Using torchrun instead of python
    CMD="$TORCHRUN_EXEC --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT $TRAINER_PY \
    --dataset \"$DS\" \
    --partition silos \
    --data-dir \"$DATA_DIR\" > \"$LOG_FILE\" 2>&1"

    echo "$CMD" >> "$CMD_FILE"
    ((JOB_COUNT++))

done

# ==============================================================================
# ROBUST RESUME LOGIC
# ==============================================================================
CHANGES_DETECTED=0

if [ -f "$PREV_CMD_FILE" ]; then
    NEW_SUM=$(sha256sum "$CMD_FILE" | awk '{print $1}')
    OLD_SUM=$(sha256sum "$PREV_CMD_FILE" | awk '{print $1}')

    if [ "$NEW_SUM" != "$OLD_SUM" ]; then
        CHANGES_DETECTED=1
    fi
else
    CHANGES_DETECTED=1
fi

if [ "$CHANGES_DETECTED" -eq 1 ]; then
    echo ">>> ⚠️  CONFIGURATION CHANGED."
    echo ">>> Resetting joblog to ensure new parameters are executed."
    if [ -f "$JOBLOG" ]; then
        mv "$JOBLOG" "${JOBLOG}.bak_$(date +%s)"
    fi
    cp "$CMD_FILE" "$PREV_CMD_FILE"
else
    echo ">>> ✅ CONFIGURATION UNCHANGED."
    echo ">>> Keeping joblog to RESUME from where we left off."

    if [[ -s "$JOBLOG" ]]; then
        COMPLETED_COUNT=$(awk -F'\t' 'NR>1 && $7=="0" {count++} END {print count+0}' "$JOBLOG")
        echo ">>> Skipping $COMPLETED_COUNT jobs that finished successfully."
    fi
fi

# ==============================================================================
# EXECUTION (SEQUENTIAL DDP)
# ==============================================================================
COUNT=$(wc -l < "$CMD_FILE")

if ! command -v parallel &> /dev/null; then
    echo "❌ GNU Parallel is not installed."
    exit 1
fi

echo ">>> STARTING EXECUTION OF FROZEN CORES ($COUNT Configurations)"
# We run JOBS=1 because one job now consumes the ENTIRE node (All GPUs)
echo ">>> CONFIG: Running 1 Distributed Job at a time (using all $NUM_GPUS GPUs)"

parallel --jobs 1 \
    --retries 1 \
    --joblog "$JOBLOG" \
    --resume-failed \
    --env PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES \
    --line-buffer \
    '
    echo "🚀 [DDP] Starting Distributed Job {#}"

    # Execute and capture exit status explicitly
    eval {}
    status=$?

    if [ $status -eq 0 ]; then
        echo "✅ [DDP] Job {#} Finished"
    else
        echo "❌ [DDP] Job {#} Failed (Exit Code: $status)"
        exit 1
    fi
    ' :::: "$CMD_FILE"

echo ">>> 🎉 All Frozen Checkpoints Generated."