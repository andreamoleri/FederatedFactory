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
# GPU & UTILIZATION CONFIGURATION
# ==============================================================================
# Optimization flags
export PYTHONUNBUFFERED=1
export MAX_WORKERS=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY

# ------------------------------------------------------------------------------
# [CONFIG] SELECT YOUR GPUS HERE
# ------------------------------------------------------------------------------
# Option A: Set specific GPUs, e.g., "0 1"
# Option B: Leave empty "" to auto-detect.
MANUAL_GPU_IDS=""

# ------------------------------------------------------------------------------
# LOGIC: DETERMINE TARGET LIST
# ------------------------------------------------------------------------------
if [[ -n "$MANUAL_GPU_IDS" ]]; then
    echo ">>> ⚙️  User manually selected GPUs: $MANUAL_GPU_IDS"
    TARGET_GPU_LIST="$MANUAL_GPU_IDS"
else
    echo ">>> 🤖 Auto-detecting all available GPUs..."
    TARGET_GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')
    if [[ -z "$TARGET_GPU_LIST" ]]; then
        echo ">>> ⚠️  Auto-detect failed, defaulting to GPU 0"
        TARGET_GPU_LIST="0"
    fi
fi

# Clean up whitespace
TARGET_GPU_LIST=$(echo "$TARGET_GPU_LIST" | xargs)

# Calculate Counts
NUM_GPUS=$(echo "$TARGET_GPU_LIST" | wc -w)
JOBS_PER_GPU=4
TOTAL_CONCURRENCY=$((NUM_GPUS * JOBS_PER_GPU))

echo ">>> 📊 Configuration: Using $NUM_GPUS GPUs ($TARGET_GPU_LIST)"
echo ">>> 🚀 Concurrency:   $JOBS_PER_GPU jobs/GPU = $TOTAL_CONCURRENCY total parallel jobs"

export NUM_GPUS
export TARGET_GPU_LIST

# ==============================================================================
# ORGANIZATION
# ==============================================================================
LOG_WORK_DIR="$SCRIPT_DIR/run_checkpoints_work"
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
    "cifar"
    "medmnist:bloodmnist"
    "medmnist:pathmnist"
    "medmnist:retinamnist"
    "fed_isic2019"
)

# ==============================================================================
# GENERATE COMMANDS
# ==============================================================================
echo ">>> GENERATING CHECKPOINT COMMANDS..."

JOB_COUNT=1

for DS in "${DATASETS[@]}"; do

    # Sanitization for log filenames
    SAFE_DS="${DS//:/_}"

    # --------------------------------------------------------------------------
    # 1. Dirichlet Commands (Alpha = 0.1)
    # --------------------------------------------------------------------------
    LOG_FILENAME="job${JOB_COUNT}_${SAFE_DS}_dirichlet_0.1.log"
    LOG_FILE="$LOG_STORAGE_DIR/$LOG_FILENAME"

    # NOTE: We use strict quoting here like the "Good" script
    CMD="$PYTHON_EXEC $TRAINER_PY \
    --dataset \"$DS\" \
    --partition dirichlet \
    --alpha 0.1 \
    --data-dir \"$DATA_DIR\" > \"$LOG_FILE\" 2>&1"

    echo "$CMD" >> "$CMD_FILE"
    ((JOB_COUNT++))

    # --------------------------------------------------------------------------
    # 2. Silos Commands
    # --------------------------------------------------------------------------
    LOG_FILENAME="job${JOB_COUNT}_${SAFE_DS}_silos.log"
    LOG_FILE="$LOG_STORAGE_DIR/$LOG_FILENAME"

    CMD="$PYTHON_EXEC $TRAINER_PY \
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

    # ==========================================================================
    # DISPLAY ALREADY COMPLETED JOBS
    # ==========================================================================
    if [[ -s "$JOBLOG" ]]; then
        echo ""
        echo ">>> 📋 ALREADY COMPLETED JOBS:"
        # Table Header
        printf "  %-6s %-25s %-15s %-10s\n" "ID" "DATASET" "PARTITION" "ALPHA"
        echo "  -------------------------------------------------------------"

        # Parse joblog
        awk -F'\t' 'NR>1 && $7=="0" {
            id=$1
            cmd=$0;

            d="?"; p="?"; a="-";

            if (match(cmd, /--dataset ([^ ]+)/, arr))   d=arr[1];
            if (match(cmd, /--partition ([^ ]+)/, arr)) p=arr[1];
            if (match(cmd, /--alpha ([^ ]+)/, arr))     a=arr[1];

            gsub(/"/, "", d);

            printf "  %-6s %-25s %-15s %-10s\n", id, d, p, a
        }' "$JOBLOG" | sort -n

        echo "  -------------------------------------------------------------"
        COMPLETED_COUNT=$(awk -F'\t' 'NR>1 && $7=="0" {count++} END {print count+0}' "$JOBLOG")
        echo ">>> Skipping $COMPLETED_COUNT jobs that finished successfully."
        echo ""
    fi
fi

# ==============================================================================
# EXECUTION
# ==============================================================================
COUNT=$(wc -l < "$CMD_FILE")

if ! command -v parallel &> /dev/null; then
    echo "❌ GNU Parallel is not installed."
    exit 1
fi

echo ">>> STARTING EXECUTION OF FROZEN CORES ($COUNT Configurations)"
echo ">>> CONFIG: $NUM_GPUS GPUs | $JOBS_PER_GPU Jobs/GPU | $TOTAL_CONCURRENCY Concurrent Jobs"

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

    # --- ROBUST ARRAY LOGIC (MATCHING COLLEAGUE SCRIPT) ---
    # Convert string list to array
    IFS=" " read -r -a GPU_ARRAY <<< "$TARGET_GPU_LIST"

    # Calculate index
    ARRAY_INDEX=$(( (JOB_SLOT - 1) % NUM_GPUS ))

    # Get Physical GPU ID
    GPU_ID=${GPU_ARRAY[$ARRAY_INDEX]}
    # ------------------------------------------------------

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    echo "🚀 [GPU $GPU_ID] Starting Checkpoint Job {#}"

    # Execute and capture exit status explicitly
    # Note: The command string already includes redirection
    eval {}
    status=$?

    if [ $status -eq 0 ]; then
        echo "✅ [GPU $GPU_ID] Job {#} Finished"
    else
        echo "❌ [GPU $GPU_ID] Job {#} Failed (Exit Code: $status)"
        exit 1
    fi
    ' :::: "$CMD_FILE"

echo ">>> 🎉 All Frozen Checkpoints Generated."