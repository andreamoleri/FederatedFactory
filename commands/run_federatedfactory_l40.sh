#!/bin/bash

# ==============================================================================
# H100 OPTIMIZED RUNNER - PARALLEL MODE + ZOMBIE KILLER + ROBUSTNESS
# ==============================================================================
set -o errexit
set -o nounset
set -o pipefail

# ==============================================================================
# 1. ROBUST PROJECT ROOT DETECTION
# ==============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
current_dir="$SCRIPT_DIR"
PROJECT_ROOT=""

# Walk up the directory tree until we find the 'src' folder
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

# Jump to Project Root so relative paths work
cd "$PROJECT_ROOT"
echo ">>> 📂 Working Directory: $(pwd)"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
MAIN_PY="src/main.py"
PYTHON_EXEC="$(which python)"
USER_ID=$(id -u)

# Check for dependencies
if ! command -v parallel &> /dev/null; then
    echo "❌ GNU Parallel is not installed. Please install it (sudo apt install parallel)."
    exit 1
fi

# ==============================================================================
# 2. ZOMBIE KILLER (THE CLEANUP)
# ==============================================================================
echo ">>> ☢️  INITIATING CLEANUP PROTOCOL..."

# Kill any python script running 'src/main.py' belonging to THIS user
pkill -u "$USER_ID" -f "src/main.py" || true
echo "    - Killed old training processes."

# Kill any previous MPS control daemons
pkill -u "$USER_ID" -f "nvidia-cuda-mps" || true
echo "    - Killed old MPS servers."

sleep 2

# ==============================================================================
# 3. CONFIGURE MPS (USER MODE)
# ==============================================================================
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${USER_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${USER_ID}"

rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"
mkdir -p "$CUDA_MPS_LOG_DIRECTORY"

echo ">>> ⚡ Starting Fresh NVIDIA MPS Server..."
nvidia-cuda-mps-control -d

# ==============================================================================
# 4. GPU & RESOURCE CONFIGURATION
# ==============================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------------------------------------------------------------------------
# [CONFIG] SELECT YOUR GPUS HERE
# ------------------------------------------------------------------------------
# Option A: Set specific GPUs, e.g., "1" or "2 4" or "0 1 2 3"
# Option B: Leave empty "" to auto-detect ALL available GPUs.
MANUAL_GPU_IDS="3"

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

JOBS_PER_GPU=1
TOTAL_CONCURRENCY=$((NUM_GPUS * JOBS_PER_GPU))
WORKERS=2

echo ">>> 🚀 H100 Mode: Detected $NUM_GPUS GPUs ($TARGET_GPU_LIST)"
echo ">>> 🔄 Strategy: $JOBS_PER_GPU job(s) per GPU = $TOTAL_CONCURRENCY Total Concurrent Jobs"

# ==============================================================================
# 5. DIRECTORY & DATA
# ==============================================================================
LOG_WORK_DIR="$SCRIPT_DIR/run_federatedfactory_L40"
LOG_STORAGE_DIR="$LOG_WORK_DIR/logs"
mkdir -p "$LOG_STORAGE_DIR"
DATA_DIR="data"
mkdir -p "$DATA_DIR"

CMD_FILE="$LOG_WORK_DIR/queue.txt"
PREV_CMD_FILE="$LOG_WORK_DIR/.queue.last" # Checksum file
JOBLOG="$LOG_WORK_DIR/joblog.txt"

> "$CMD_FILE"

# ==============================================================================
# 6. EXPERIMENT PARAMETERS
# ==============================================================================
SEEDS=(1 2 3 4 5)
DATASETS=(
    # "cifar"
    # "medmnist:bloodmnist"
    # "medmnist:retinamnist"
    # "medmnist:pathmnist"
    "fed_isic2019"
)
PARTITIONS=("silos") # TODO: "dirichlet" in the future
INFER_MODES=("server") # TODO: "local" in the H100 

# Fixed Hyperparameters
CHECKPOINT_FAMILY="0025001"
AGGREGATION="weighted"
CLF_EPOCHS=300 # TODO: 300
SAMPLES_PER_CLASS=10000 # TODO: 10000
MODEL="diffusion"
ALPHA_VAL=0.1
BATCH_SIZE=64 # TODO: I changed it to 64128was 64

echo ">>> GENERATING COMMANDS..."

JOB_COUNT=1

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    L_DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

    # Robust Resolution Logic using case statement
    INPUT_SIZE=32
    LATENT_DIM=128

    case "$L_DATASET" in
        *"isic"*)
            INPUT_SIZE=128 
            LATENT_DIM=64
            ;;
        *"nico"*)
            INPUT_SIZE=224
            LATENT_DIM=128
            ;;
        *)
            # Default for medmnist/cifar
            INPUT_SIZE=32
            LATENT_DIM=128
            ;;
    esac

    for PARTITION in "${PARTITIONS[@]}"; do

        # 1. Initialize variables (Added from Block B)
        ALPHA_ARG=""
        ALPHA_SUFFIX=""

        # 2. Logic to set suffix if partition is dirichlet
        if [ "$PARTITION" == "dirichlet" ]; then
            ALPHA_ARG="--alpha $ALPHA_VAL"
            ALPHA_SUFFIX="_alpha_${ALPHA_VAL}"
        fi

        for MODE in "${INFER_MODES[@]}"; do
            SAFE_DS="${DATASET//:/_}"
            OUT_ROOT="federatedfactory_output_L40"

            # 3. Updated Log Filename (Now includes ${PARTITION} and ${ALPHA_SUFFIX})
            # Old Block A: job${JOB_COUNT}_${SAFE_DS}_${MODE}_seed${SEED}.log
            # New Block A:
            LOG_FILE="$LOG_STORAGE_DIR/job${JOB_COUNT}_${SAFE_DS}_${PARTITION}${ALPHA_SUFFIX}_${MODE}_seed${SEED}.log"

            # TODO: Add back --save-datasets, remove --synthetic-data-dir
            CMD="$PYTHON_EXEC $MAIN_PY \
                --dataset \"$DATASET\" \
                --partition \"$PARTITION\" $ALPHA_ARG \
                --infer-mode \"$MODE\" \
                --seed $SEED \
                --input-size $INPUT_SIZE \
                --model \"$MODEL\" \
                --latent-dim $LATENT_DIM \
                --aggregation \"$AGGREGATION\" \
                --checkpoint-epoch-family \"$CHECKPOINT_FAMILY\" \
                --clf-epochs $CLF_EPOCHS \
                --samples-per-class $SAMPLES_PER_CLASS \
                --batch-size $BATCH_SIZE \
                --workers $WORKERS \
                --save-datasets \
                --data-dir \"$DATA_DIR\" \
                --out-dir \"$OUT_ROOT\" > \"$LOG_FILE\" 2>&1"

            echo "$CMD" >> "$CMD_FILE"
            ((JOB_COUNT++))
        done
    done
  done
done

# ==============================================================================
# 7. ROBUST RESUME CHECK
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
    echo ">>> ⚠️  CONFIGURATION CHANGED (or first run)."
    echo ">>> Resetting joblog to ensure new parameters are executed."
    if [ -f "$JOBLOG" ]; then
        mv "$JOBLOG" "${JOBLOG}.bak_$(date +%s)"
    fi
    cp "$CMD_FILE" "$PREV_CMD_FILE"
else
    echo ">>> ✅ CONFIGURATION UNCHANGED."
    echo ">>> Keeping joblog to RESUME from where we left off."

    # Display completed jobs table
    if [[ -s "$JOBLOG" ]]; then
        echo ""
        echo ">>> 📋 ALREADY COMPLETED JOBS:"
        printf "  %-6s %-25s %-10s %-10s %-10s\n" "ID" "DATASET" "PARTITION" "MODE" "SEED"
        echo "  -----------------------------------------------------------------------"

        awk -F'\t' 'NR>1 && $7=="0" {
            id=$1
            cmd=$0;

            d="?"; p="?"; m="?"; s="?";

            if (match(cmd, /--dataset ([^ ]+)/, arr))   d=arr[1];
            if (match(cmd, /--partition ([^ ]+)/, arr)) p=arr[1];
            if (match(cmd, /--infer-mode ([^ ]+)/, arr)) m=arr[1];
            if (match(cmd, /--seed ([^ ]+)/, arr))      s=arr[1];

            gsub(/"/, "", d);
            gsub(/"/, "", p);
            gsub(/"/, "", m);

            printf "  %-6s %-25s %-10s %-10s %-10s\n", id, d, p, m, s
        }' "$JOBLOG" | sort -n

        echo "  -----------------------------------------------------------------------"
        COMPLETED_COUNT=$(awk -F'\t' 'NR>1 && $7=="0" {count++} END {print count+0}' "$JOBLOG")
        echo ">>> Skipping $COMPLETED_COUNT jobs that finished successfully."
        echo ""
    fi
fi

# ==============================================================================
# 8. EXECUTION
# ==============================================================================
COUNT=$(wc -l < "$CMD_FILE")
echo ">>> STARTING $COUNT JOBS..."

export TARGET_GPU_LIST
export NUM_GPUS
export CUDA_MPS_PIPE_DIRECTORY
export CUDA_MPS_LOG_DIRECTORY

# Using GNU Parallel with MPS logic maintained
parallel --jobs "$TOTAL_CONCURRENCY" \
    --retries 1 \
    --joblog "$JOBLOG" \
    --resume-failed \
    --env PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES \
    --env CUDA_MPS_PIPE_DIRECTORY \
    --env TARGET_GPU_LIST \
    --env NUM_GPUS \
    --line-buffer \
    '
    JOB_SLOT={%}
    IFS=" " read -r -a GPU_ARRAY <<< "$TARGET_GPU_LIST"
    IDX=$(( (JOB_SLOT - 1) % NUM_GPUS ))
    GPU_ID=${GPU_ARRAY[IDX]}
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    echo "🚀 [Slot {#} -> GPU $GPU_ID (MPS)] Starting..."
    eval {}
    exit_status=$?

    if [ $exit_status -eq 0 ]; then
        echo "✅ [Slot {#}] Finished Successfully"
    else
        echo "❌ [Slot {#}] Failed with exit code $exit_status"
        exit $exit_status
    fi
    ' :::: "$CMD_FILE"

# ==============================================================================
# 9. CLEANUP
# ==============================================================================
echo ">>> 🛑 Stopping MPS..."
echo "quit" | nvidia-cuda-mps-control
rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
echo ">>> 🎉 Done."
