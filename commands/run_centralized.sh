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

CENTRAL_PY="$PROJECT_ROOT/src/central_main.py"
PYTHON_EXEC="$(which python)"

# ==============================================================================
# ORGANIZATION
# ==============================================================================
LOG_WORK_DIR="$SCRIPT_DIR/run_centralized_baselines"
LOG_STORAGE_DIR="$LOG_WORK_DIR/logs"

mkdir -p "$LOG_STORAGE_DIR"
echo ">>> 📂 Internal queues stored in: $LOG_WORK_DIR"
echo ">>> 📝 Console logs stored in:    $LOG_STORAGE_DIR"

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================
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
# GPU & UTILIZATION CONFIGURATION
# ==============================================================================
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Adjust based on your GPU VRAM.
JOBS_PER_GPU=2

# ==============================================================================
# EXPERIMENT CONFIG & SMART RESUME FILES
# ==============================================================================
CMD_FILE="$LOG_WORK_DIR/central_queue.txt"
PREV_CMD_FILE="$LOG_WORK_DIR/.central_queue.last"
JOBLOG="$LOG_WORK_DIR/central_joblog.txt"

# Pulisco il file corrente per rigenerarlo
> "$CMD_FILE"

# Standard NeurIPS Hyperparameters
EPOCHS=300
BATCH_SIZE=128
LR=0.1
SEEDS=(1 2 3 4 5)

DATASETS=(
  # "cifar"
  # "medmnist:retinamnist"
  # "medmnist:bloodmnist"
  "medmnist:pathmnist"
  # "fed_camelyon16"
  # "fed_isic2019"
)

# ==============================================================================
# GENERATE COMMANDS
# ==============================================================================
echo ">>> GENERATING CENTRALIZED COMMANDS..."

JOB_COUNT=1

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do

    # --- INPUT SIZE STRATEGY ---
    L_DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
    INPUT_SIZE=224

    case "$L_DATASET" in
        *"cifar"*) INPUT_SIZE=32 ;;
        *"medmnist"*) INPUT_SIZE=28 ;;
        *"camelyon"*|*"isic"*|*"nico"*) INPUT_SIZE=224 ;;
        *) echo "⚠️  Unknown dataset type for $DATASET, defaulting to 224." ;;
    esac

    # Output directory
    OUT_DIR="$PROJECT_ROOT/output/centralized_experiments/${DATASET}/resnet50/seed_${SEED}"

    # Log naming
    SAFE_DATASET_NAME="${DATASET//:/_}"
    LOG_FILENAME="job${JOB_COUNT}_${SAFE_DATASET_NAME}_seed_${SEED}.log"
    LOG_FILE="$LOG_STORAGE_DIR/$LOG_FILENAME"

    # COSTRUZIONE COMANDO
    # Nota: Manteniamo il redirect log interno al comando, così ogni job ha il suo file specifico
    CMD="mkdir -p \"$OUT_DIR\" && $PYTHON_EXEC $CENTRAL_PY \
      --dataset \"$DATASET\" \
      --data-dir \"$DATA_DIR\" \
      --out-dir \"$OUT_DIR\" \
      --epochs $EPOCHS \
      --batch-size $BATCH_SIZE \
      --input-size $INPUT_SIZE \
      --lr $LR \
      --seed $SEED > \"$LOG_FILE\" 2>&1"

    echo "$CMD" >> "$CMD_FILE"
    ((JOB_COUNT++))

  done
done

# ==============================================================================
# ROBUST RESUME LOGIC (PORTED FROM CODE B)
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
    # DISPLAY ALREADY COMPLETED JOBS (Adapted for Centralized Flags)
    # ==========================================================================
    if [[ -s "$JOBLOG" ]]; then
        echo ""
        echo ">>> 📋 ALREADY COMPLETED JOBS:"
        # Table Header
        printf "  %-6s %-25s %-10s %-10s\n" "ID" "DATASET" "SEED" "EPOCHS"
        echo "  -----------------------------------------------------------"

        # Parse joblog specifically for Centralized args (--dataset, --seed, --epochs)
        awk -F'\t' 'NR>1 && $7=="0" {
            id=$1
            cmd=$0;

            d="?"; s="?"; e="?";

            if (match(cmd, /--dataset ([^ ]+)/, arr))   d=arr[1];
            if (match(cmd, /--seed ([^ ]+)/, arr))      s=arr[1];
            if (match(cmd, /--epochs ([^ ]+)/, arr))    e=arr[1];

            gsub(/"/, "", d); # Remove quotes from dataset name if present

            printf "  %-6s %-25s %-10s %-10s\n", id, d, s, e
        }' "$JOBLOG" | sort -n

        echo "  -----------------------------------------------------------"
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

# --- MODIFIED: RESTRICT TO GPUS 1, 2, 3 ---
# Define the IDs of the GPUs you want to use (space separated)
export TARGET_GPU_LIST="1 2 3"

# Count them automatically
NUM_GPUS=$(echo "$TARGET_GPU_LIST" | wc -w)
export NUM_GPUS
export TARGET_GPU_LIST

TOTAL_CONCURRENCY=$((NUM_GPUS * JOBS_PER_GPU))

echo ">>> STARTING EXECUTION OF $COUNT JOBS"
echo ">>> CONFIG: $NUM_GPUS GPUs | $JOBS_PER_GPU Jobs/GPU | $TOTAL_CONCURRENCY Concurrent Jobs"

parallel --jobs "$TOTAL_CONCURRENCY" \
    --retries 3 \
    --joblog "$JOBLOG" \
    --resume-failed \
    --env PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES \
    --env NUM_GPUS \
    --env TARGET_GPU_LIST \
    --line-buffer \
    '
    JOB_SLOT={%}

    # --- LOGIC CHANGE START ---
    # Convert the string list "1 2 3" into an array
    IFS=" " read -r -a GPU_ARRAY <<< "$TARGET_GPU_LIST"

    # Calculate which index of the array to use (0, 1, or 2)
    ARRAY_INDEX=$(( (JOB_SLOT - 1) % NUM_GPUS ))

    # Get the actual Physical GPU ID (1, 2, or 3)
    GPU_ID=${GPU_ARRAY[$ARRAY_INDEX]}
    # --- LOGIC CHANGE END ---

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    echo "🚀 [GPU $GPU_ID] Starting Job {#}"

    # Execute command
    eval {}
    status=$?

    if [ $status -eq 0 ]; then
        echo "✅ [GPU $GPU_ID] Job {#} Finished"
    else
        echo "❌ [GPU $GPU_ID] Job {#} Failed"
        exit 1
    fi
    ' :::: "$CMD_FILE"

echo ">>> 🎉 All Centralized Experiments Completed."