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

MAIN_PY="$PROJECT_ROOT/src/main.py"
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
# Option A: Set specific GPUs (space-separated string), e.g., "0" or "0 1" or "1 3"
# Option B: Leave empty "" to automatically detect ALL available GPUs using nvidia-smi.
MANUAL_GPU_IDS="0 1 2 3"

# ------------------------------------------------------------------------------
# LOGIC: DETERMINE TARGET LIST
# ------------------------------------------------------------------------------
if [[ -n "$MANUAL_GPU_IDS" ]]; then
    echo ">>> ⚙️  User manually selected GPUs: $MANUAL_GPU_IDS"
    TARGET_GPU_LIST="$MANUAL_GPU_IDS"
else
    echo ">>> 🤖 Auto-detecting all available GPUs..."
    TARGET_GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')

    # Fallback if nvidia-smi fails
    if [[ -z "$TARGET_GPU_LIST" ]]; then
        echo ">>> ⚠️  Auto-detect failed, defaulting to GPU 0"
        TARGET_GPU_LIST="0"
    fi
fi

# Clean up whitespace (ensure single spaces)
TARGET_GPU_LIST=$(echo "$TARGET_GPU_LIST" | xargs)

# Calculate Counts
NUM_GPUS=$(echo "$TARGET_GPU_LIST" | wc -w)
JOBS_PER_GPU=5
TOTAL_CONCURRENCY=$((NUM_GPUS * JOBS_PER_GPU))

echo ">>> 📊 Configuration: Using $NUM_GPUS GPUs ($TARGET_GPU_LIST)"
echo ">>> 🚀 Concurrency:   $JOBS_PER_GPU jobs/GPU = $TOTAL_CONCURRENCY total parallel jobs"

export NUM_GPUS
export TARGET_GPU_LIST

# ==============================================================================
# ORGANIZATION
# ==============================================================================
LOG_WORK_DIR="$SCRIPT_DIR/run_federated_baselines_gpu3"
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
# EXPERIMENT CONFIG & SMART RESUME FILES
# ==============================================================================
CMD_FILE="$LOG_WORK_DIR/experiment_queue.txt"
PREV_CMD_FILE="$LOG_WORK_DIR/.experiment_queue.last"
JOBLOG="$LOG_WORK_DIR/joblog.txt"

# Reset queue
> "$CMD_FILE"

# --- GLOBAL HYPERPARAMETERS ---
SEEDS=(1 2 3 4 5)
BS=128
LATENT=128
ROUNDS=200
PATIENCE=15
CLIENTS=10
FRACTION=1.0
EPOCHS_LIST=(5)
LRS=("0.1")

DATASETS=(
  # "cifar"
  # "medmnist:retinamnist"
  # "medmnist:bloodmnist"
  "medmnist:pathmnist"
  # "fed_camelyon16"
  # "fed_isic2019"
)

# ==============================================================================
# HYPERPARAMETER GRID SEARCH CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. DEFINE PARAMETER GRIDS FOR DIRICHLET (Alpha = 0.1)
# ------------------------------------------------------------------------------
ALPHAS=("0.1") # Dirichlet Partition Alpha

PROX_MUS_DIR=("0.001")
DYN_ALPHAS_DIR=("0.0001")
MOMS_DIR=("0.0")

# ------------------------------------------------------------------------------
# 2. DEFINE PARAMETER GRIDS FOR SILOS (Partition = Silos)
# ------------------------------------------------------------------------------
# You can now change these independently of the Dirichlet parameters
PROX_MUS_SILOS=("0.001")    # e.g., change to "0.01" for Silos
DYN_ALPHAS_SILOS=("0.0001")  # e.g., change to "0.1" for Silos
MOMS_SILOS=("0.0")          # e.g., change to "0.9" for Silos

# ==============================================================================
# BUILD MODEL LISTS
# ==============================================================================
MODELS_DIRICHLET=()
MODELS_SILOS=()

# --- BUILD DIRICHLET LIST ---
for lr in "${LRS[@]}"; do
    # FedAvg (Same for both usually, but generated per list)
    MODELS_DIRICHLET+=("baseline:fedavg|--learning-rate $lr")

    # FedProx
    for mu in "${PROX_MUS_DIR[@]}"; do
        MODELS_DIRICHLET+=("baseline:fedprox|--learning-rate $lr --fedprox-mu $mu")
    done
    # FedDyn
    for alpha in "${DYN_ALPHAS_DIR[@]}"; do
        MODELS_DIRICHLET+=("baseline:feddyn|--learning-rate $lr --feddyn-alpha $alpha")
    done
    # Scaffold
    for mom in "${MOMS_DIR[@]}"; do
        MODELS_DIRICHLET+=("baseline:scaffold|--learning-rate $lr --baseline-momentum $mom")
    done
done

# --- BUILD SILOS LIST ---
for lr in "${LRS[@]}"; do
    # FedAvg
    MODELS_SILOS+=("baseline:fedavg|--learning-rate $lr")

    # FedProx
    for mu in "${PROX_MUS_SILOS[@]}"; do
        MODELS_SILOS+=("baseline:fedprox|--learning-rate $lr --fedprox-mu $mu")
    done
    # FedDyn
    for alpha in "${DYN_ALPHAS_SILOS[@]}"; do
        MODELS_SILOS+=("baseline:feddyn|--learning-rate $lr --feddyn-alpha $alpha")
    done
    # Scaffold
    for mom in "${MOMS_SILOS[@]}"; do
        MODELS_SILOS+=("baseline:scaffold|--learning-rate $lr --baseline-momentum $mom")
    done
done

# ==============================================================================
# GENERATE COMMANDS
# ==============================================================================
echo ">>> GENERATING FEDERATED COMMANDS..."

JOB_COUNT=1

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do

    # --- 1. SCIENTIFIC RESOLUTION ENFORCEMENT ---
    L_DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
    INPUT_SIZE=""
    HAS_NATURAL_SPLIT=true

    case "$L_DATASET" in
        *"cifar"*)
            INPUT_SIZE=32
            HAS_NATURAL_SPLIT=true
            ;;
        *"medmnist"*)
            INPUT_SIZE=28
            HAS_NATURAL_SPLIT=true
            ;;
        *"camelyon"*|*"isic"*|*"nico"*)
            INPUT_SIZE=224
            HAS_NATURAL_SPLIT=true
            ;;
        *)
            echo "❌ ERROR: Unknown dataset '$DATASET'."
            exit 1
            ;;
    esac

    SAFE_DATASET_NAME="${DATASET//:/_}"

    for EPS in "${EPOCHS_LIST[@]}"; do

        # ----------------------------------------------------------------------
        # EXPERIMENT A: DIRICHLET LOOPS (Uses MODELS_DIRICHLET)
        # ----------------------------------------------------------------------
        for MODEL_ENTRY in "${MODELS_DIRICHLET[@]}"; do
            MODEL="${MODEL_ENTRY%%|*}"
            SPECIFIC_ARGS="${MODEL_ENTRY#*|}"
            ARG_SUFFIX=$(echo "$SPECIFIC_ARGS" | sed 's/--//g' | tr ' ' '_')

            BASE_ARGS="--model \"$MODEL\" \
            --dataset \"$DATASET\" \
            --input-size $INPUT_SIZE \
            --num-clients $CLIENTS \
            --client-fraction $FRACTION \
            --latent-dim $LATENT \
            --dp false \
            --seed $SEED \
            --baseline-epochs-per-round $EPS \
            --baseline-max-rounds $ROUNDS \
            --baseline-patience $PATIENCE \
            --batch-size $BS \
            --data-dir \"$DATA_DIR\" \
            --robustness false \
            $SPECIFIC_ARGS"

            for ALPHA in "${ALPHAS[@]}"; do
                OUT_DIR="$PROJECT_ROOT/output/main_experiments_gpu3/baselines_dirichlet/epochs_${EPS}/${SAFE_DATASET_NAME}/${MODEL}${ARG_SUFFIX}/alpha_${ALPHA}/seed_${SEED}"

                LOG_FILENAME="job${JOB_COUNT}_${SAFE_DATASET_NAME}_${MODEL}${ARG_SUFFIX}_dirichlet_${ALPHA}.log"
                LOG_FILE="$LOG_STORAGE_DIR/$LOG_FILENAME"

                CMD="mkdir -p \"$OUT_DIR\" && $PYTHON_EXEC $MAIN_PY $BASE_ARGS \
                --partition dirichlet --alpha $ALPHA --out-dir \"$OUT_DIR\" > \"$LOG_FILE\" 2>&1"

                echo "$CMD" >> "$CMD_FILE"
                ((JOB_COUNT++))
            done
        done

        # ----------------------------------------------------------------------
        # EXPERIMENT B: SILOS LOOPS (Uses MODELS_SILOS)
        # ----------------------------------------------------------------------
        if [ "$HAS_NATURAL_SPLIT" = true ]; then
            for MODEL_ENTRY in "${MODELS_SILOS[@]}"; do
                MODEL="${MODEL_ENTRY%%|*}"
                SPECIFIC_ARGS="${MODEL_ENTRY#*|}"
                ARG_SUFFIX=$(echo "$SPECIFIC_ARGS" | sed 's/--//g' | tr ' ' '_')

                BASE_ARGS="--model \"$MODEL\" \
                --dataset \"$DATASET\" \
                --input-size $INPUT_SIZE \
                --num-clients $CLIENTS \
                --client-fraction $FRACTION \
                --latent-dim $LATENT \
                --dp false \
                --seed $SEED \
                --baseline-epochs-per-round $EPS \
                --baseline-max-rounds $ROUNDS \
                --baseline-patience $PATIENCE \
                --batch-size $BS \
                --data-dir \"$DATA_DIR\" \
                --robustness false \
                $SPECIFIC_ARGS"

                OUT_DIR="$PROJECT_ROOT/output/main_experiments_gpu3/baselines_silos/epochs_${EPS}/${SAFE_DATASET_NAME}/${MODEL}${ARG_SUFFIX}/seed_${SEED}"

                LOG_FILENAME="job${JOB_COUNT}_${SAFE_DATASET_NAME}_${MODEL}${ARG_SUFFIX}_silos.log"
                LOG_FILE="$LOG_STORAGE_DIR/$LOG_FILENAME"

                CMD="mkdir -p \"$OUT_DIR\" && $PYTHON_EXEC $MAIN_PY $BASE_ARGS \
                --partition silos --out-dir \"$OUT_DIR\" > \"$LOG_FILE\" 2>&1"

                echo "$CMD" >> "$CMD_FILE"
                ((JOB_COUNT++))
            done
        fi

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
    # DISPLAY ALREADY COMPLETED JOBS (Adapted for Federated Flags)
    # ==========================================================================
    if [[ -s "$JOBLOG" ]]; then
        echo ""
        echo ">>> 📋 ALREADY COMPLETED JOBS:"
        # Table Header
        printf "  %-6s %-25s %-20s %-15s\n" "ID" "DATASET" "MODEL" "PARTITION"
        echo "  --------------------------------------------------------------------------"

        # Parse joblog for Federated args (--dataset, --model, --partition)
        awk -F'\t' 'NR>1 && $7=="0" {
            id=$1
            cmd=$0;

            d="?"; m="?"; p="?";

            if (match(cmd, /--dataset ([^ ]+)/, arr))   d=arr[1];
            if (match(cmd, /--model ([^ ]+)/, arr))     m=arr[1];
            if (match(cmd, /--partition ([^ ]+)/, arr)) p=arr[1];

            gsub(/"/, "", d);
            gsub(/"/, "", m);

            printf "  %-6s %-25s %-20s %-15s\n", id, d, m, p
        }' "$JOBLOG" | sort -n

        echo "  --------------------------------------------------------------------------"
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

echo ">>> STARTING EXECUTION OF $COUNT JOBS"
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

    # --- ROBUST ARRAY LOGIC START ---
    # Convert the string list "0 1 2 3" into an array
    IFS=" " read -r -a GPU_ARRAY <<< "$TARGET_GPU_LIST"

    # Calculate which index of the array to use
    ARRAY_INDEX=$(( (JOB_SLOT - 1) % NUM_GPUS ))

    # Get the actual Physical GPU ID
    GPU_ID=${GPU_ARRAY[$ARRAY_INDEX]}
    # --- ROBUST ARRAY LOGIC END ---

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

echo ">>> 🎉 All Federated Experiments Completed."