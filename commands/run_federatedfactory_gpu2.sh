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

# 2. CRITICAL FIX: Jump to Project Root so relative paths (checkpoints/) work
cd "$PROJECT_ROOT"
echo ">>> 📂 Working Directory set to: $(pwd)"

MAIN_PY="src/main.py"
PYTHON_EXEC="$(which python)"

# ==============================================================================
# GPU & UTILIZATION CONFIGURATION
# ==============================================================================
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY

# ------------------------------------------------------------------------------
# [CONFIG] SELECT YOUR GPUS HERE
# ------------------------------------------------------------------------------
# Option A: Set specific GPUs, e.g., "1" or "2 4" or "0 1 2 3"
# Option B: Leave empty "" to auto-detect ALL available GPUs.
MANUAL_GPU_IDS="0 1"

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

# Diffusion + Classification is VRAM heavy. Keeping it to 1 job per GPU is safest.
JOBS_PER_GPU=10
TOTAL_CONCURRENCY=$((NUM_GPUS * JOBS_PER_GPU))

echo ">>> 📊 Configuration: Using $NUM_GPUS GPUs ($TARGET_GPU_LIST)"
echo ">>> 🚀 Concurrency:   $JOBS_PER_GPU jobs/GPU = $TOTAL_CONCURRENCY total parallel jobs"

export NUM_GPUS
export TARGET_GPU_LIST

# ==============================================================================
# ORGANIZATION
# ==============================================================================
# We create the work dir relative to the script location (commands/) to keep root clean
LOG_WORK_DIR="$SCRIPT_DIR/run_federatedfactory_work_gpu2"
LOG_STORAGE_DIR="$LOG_WORK_DIR/logs"

mkdir -p "$LOG_STORAGE_DIR"
echo ">>> 📂 Internal queues stored in: $LOG_WORK_DIR"
echo ">>> 📝 Console logs stored in:    $LOG_STORAGE_DIR"

# Data Dir is now relative to PROJECT_ROOT because we did `cd $PROJECT_ROOT`
DATA_DIR="data"
mkdir -p "$DATA_DIR"
echo ">>> ✅  Using data directory: $DATA_DIR"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# ==============================================================================
# EXPERIMENT CONFIG & SMART RESUME FILES
# ==============================================================================
CMD_FILE="$LOG_WORK_DIR/factory_queue.txt"
PREV_CMD_FILE="$LOG_WORK_DIR/.factory_queue.last"
JOBLOG="$LOG_WORK_DIR/joblog.txt"

# Reset queue file
> "$CMD_FILE"

# --- EXPERIMENT PARAMETERS ---
SEEDS=(1 2 3 4 5)
DATASETS=(
    "cifar"
    # "medmnist:bloodmnist"
    # "medmnist:retinamnist"
    # "medmnist:pathmnist"
    "fed_isic2019"
)
PARTITIONS=("silos") # TODO: "dirichlet" in the future
INFER_MODES=("server" "local")

# Fixed Hyperparameters
CHECKPOINT_FAMILY="0025001"
AGGREGATION="weighted"
CLF_EPOCHS=3 # TODO: 300
SAMPLES_PER_CLASS=10 # TODO: 10000
MODEL="diffusion"
ALPHA_VAL=0.1

# ==============================================================================
# GENERATE COMMANDS
# ==============================================================================
echo ">>> GENERATING EXPERIMENT COMMANDS..."

JOB_COUNT=1

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do

    # --- 1. RESOLUTION & CHANNEL CONFIGURATION ---
    # Matches logic in src/checkpoint_trainer.py
    L_DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

    # Defaults
    INPUT_SIZE=32
    CURRENT_LATENT_DIM=128

    case "$L_DATASET" in
        # ---------------------------------------------------------
        # Case A: Low-Res Datasets (CIFAR, MedMNIST)
        # ---------------------------------------------------------
        *"medmnist"*|*"cifar"*)
            INPUT_SIZE=32
            CURRENT_LATENT_DIM=128
            ;;

        # ---------------------------------------------------------
        # Case B: High-Res Datasets (ISIC) - H100 Optimized
        # ---------------------------------------------------------
        # ISIC checkpoints were trained at 128px res with 64 channels
        *"isic"*)
            INPUT_SIZE=128
            CURRENT_LATENT_DIM=64
            ;;

        *"nico"*)
            INPUT_SIZE=224
            CURRENT_LATENT_DIM=128
            ;;
    esac

    for PARTITION in "${PARTITIONS[@]}"; do

        # Dirichlet specific arg
        ALPHA_ARG=""
        ALPHA_SUFFIX=""
        if [ "$PARTITION" == "dirichlet" ]; then
            ALPHA_ARG="--alpha $ALPHA_VAL"
            ALPHA_SUFFIX="_alpha_${ALPHA_VAL}"
        fi

        for MODE in "${INFER_MODES[@]}"; do

            # Construct Output Directory Structure
            SAFE_DS="${DATASET//:/_}"

            # Since we are in PROJECT_ROOT, "output" is just "output"
            OUT_ROOT="output"

            # Log Filename (keep explicit path for logs)
            LOG_FILENAME="job${JOB_COUNT}_${SAFE_DS}_${PARTITION}${ALPHA_SUFFIX}_${MODE}_seed${SEED}.log"
            LOG_FILE="$LOG_STORAGE_DIR/$LOG_FILENAME"

            # Build Command
            # Note: We use $CURRENT_LATENT_DIM instead of hardcoded 128
            CMD="$PYTHON_EXEC $MAIN_PY \
                --dataset \"$DATASET\" \
                --partition \"$PARTITION\" $ALPHA_ARG \
                --infer-mode \"$MODE\" \
                --seed $SEED \
                --input-size $INPUT_SIZE \
                --model \"$MODEL\" \
                --latent-dim $CURRENT_LATENT_DIM \
                --aggregation \"$AGGREGATION\" \
                --checkpoint-epoch-family \"$CHECKPOINT_FAMILY\" \
                --clf-epochs $CLF_EPOCHS \
                --samples-per-class $SAMPLES_PER_CLASS \
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
# EXECUTION
# ==============================================================================
COUNT=$(wc -l < "$CMD_FILE")

if ! command -v parallel &> /dev/null; then
    echo "❌ GNU Parallel is not installed. Please install it (sudo apt install parallel)."
    exit 1
fi

echo ">>> STARTING EXECUTION OF $COUNT EXPERIMENTS"
echo ">>> CONFIG: $NUM_GPUS GPUs | $JOBS_PER_GPU Jobs/GPU | $TOTAL_CONCURRENCY Concurrent Jobs"

# We use --joblog to track success/fail
# We use --resume-failed so re-running the script picks up dropped jobs
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
    # 1. Calculate GPU ID based on Job Slot
    JOB_SLOT={%}
    IFS=" " read -r -a GPU_ARRAY <<< "$TARGET_GPU_LIST"
    ARRAY_INDEX=$(( (JOB_SLOT - 1) % NUM_GPUS ))
    GPU_ID=${GPU_ARRAY[$ARRAY_INDEX]}

    # 2. Set Visibility
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    echo "🚀 [GPU $GPU_ID] Starting Job {#}"

    # 3. Execute the command passed from the text file
    # We use eval because the line contains redirections (> log.txt)
    eval {}
    exit_status=$?

    # 4. Report Status
    if [ $exit_status -eq 0 ]; then
        echo "✅ [GPU $GPU_ID] Job {#} Finished"
    else
        echo "❌ [GPU $GPU_ID] Job {#} Failed (Exit Code: $exit_status)"
        # Parallel needs a non-zero exit to record failure in joblog
        exit $exit_status
    fi
    ' :::: "$CMD_FILE"

echo ">>> 🎉 All FederatedFactory Experiments Completed."