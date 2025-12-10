#!/bin/bash

# ==============================================================================
# SETUP & ROBUSTNESS
# ==============================================================================
set -o errexit
set -o nounset
set -o pipefail

# 1. Robust Project Root Detection
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT=""
current_dir="$SCRIPT_DIR"
while [[ "$current_dir" != "/" ]]; do
    if [[ -d "$current_dir/src" ]]; then
        PROJECT_ROOT="$current_dir"
        break
    fi
    current_dir="$(dirname "$current_dir")"
done

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "❌ ERROR: Could not locate 'src' directory. Run this from within the project."
    exit 1
fi

MAIN_PY="$PROJECT_ROOT/src/main.py"
PYTHON_EXEC="$(which python)"

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
# H100/CLUSTER CONFIGURATION
# ==============================================================================
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ "$NUM_GPUS" -eq 0 ]]; then NUM_GPUS=1; fi

JOBS_PER_GPU=2
TOTAL_JOBS=$((NUM_GPUS * JOBS_PER_GPU))

# Optimization flags for Cluster/H100
export PYTHONUNBUFFERED=1
export MAX_WORKERS=0
export OMP_NUM_THREADS=4 # Slightly increased for data loading
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY

# ==============================================================================
# EXPERIMENT CONFIG
# ==============================================================================
LOG_WORK_DIR="$SCRIPT_DIR/run_federated_baselines"
mkdir -p "$LOG_WORK_DIR"
CMD_FILE="$LOG_WORK_DIR/experiment_queue.txt"
PREV_CMD_FILE="$LOG_WORK_DIR/.experiment_queue.last"
JOBLOG="$LOG_WORK_DIR/joblog.txt"

# Reset queue
> "$CMD_FILE"

# --- HYPERPARAMETERS ---
SEEDS=(1) # TODO: Add 2 3 4 5 for final run
BS=128
LATENT=128
ROUNDS=200
PATIENCE=15
CLIENTS=10
FRACTION=1.0
EPOCHS_LIST=(5 10)

DATASETS=(
  "cifar"
  "medmnist:retinamnist"
  "medmnist:bloodmnist"
  "fed_camelyon16"
  "fed_isic2019"
)

# Format: "model_name|args"
MODELS=(
    "baseline:fedavg|--learning-rate 0.1"
    "baseline:fedprox|--learning-rate 0.05 --fedprox-mu 0.01"
    "baseline:feddyn|--learning-rate 0.05 --feddyn-alpha 0.01"
    "baseline:scaffold|--learning-rate 0.1 --baseline-momentum 0.9"
)

ALPHAS=("0.1" "0.5")

# ==============================================================================
# GENERATE COMMANDS
# ==============================================================================
echo ">>> GENERATING COMMANDS..."

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do

    # --- 1. SCIENTIFIC RESOLUTION ENFORCEMENT ---
    # Ensures architectural consistency with the centralized baseline
    L_DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
    INPUT_SIZE=""
    HAS_NATURAL_SPLIT=false

    case "$L_DATASET" in
        *"cifar"*)
            INPUT_SIZE=32
            HAS_NATURAL_SPLIT=false # CIFAR is artificial, no "Silos"
            ;;
        *"medmnist"*)
            INPUT_SIZE=28
            HAS_NATURAL_SPLIT=false # MedMNIST is usually centralized pre-split
            ;;
        *"camelyon"*|*"isic"*|*"nico"*)
            INPUT_SIZE=224
            HAS_NATURAL_SPLIT=true  # These have real hospitals/centers
            ;;
        *)
            echo "❌ ERROR: Unknown dataset '$DATASET'. Cannot assign valid resolution."
            exit 1
            ;;
    esac

    # Correct naming for paths (Fixes the typo: no space after colon)
    SAFE_DATASET_NAME="${DATASET//:/_}"

    for MODEL_ENTRY in "${MODELS[@]}"; do
      MODEL="${MODEL_ENTRY%%|*}"
      SPECIFIC_ARGS="${MODEL_ENTRY#*|}"

      for EPS in "${EPOCHS_LIST[@]}"; do

        # Base arguments
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
        $SPECIFIC_ARGS"

        # --- EXPERIMENT A: DIRICHLET (Synthentic Non-IID) ---
        # Valid for ALL datasets (we can always synthesize partitions)
        for ALPHA in "${ALPHAS[@]}"; do
          OUT_DIR="$PROJECT_ROOT/output/main_experiments/baselines_dirichlet/epochs_${EPS}/${SAFE_DATASET_NAME}/${MODEL}/alpha_${ALPHA}/seed_${SEED}"

          # Skip if completed (optional check, handled by parallel resume mostly)
          CMD="$PYTHON_EXEC $MAIN_PY $BASE_ARGS --partition dirichlet --alpha $ALPHA --out-dir \"$OUT_DIR\""
          echo "$CMD" >> "$CMD_FILE"
        done

        # --- EXPERIMENT B: SILOS (Natural/Real-World Non-IID) ---
        # ONLY valid for datasets that actually represent different centers
        if [ "$HAS_NATURAL_SPLIT" = true ]; then
            OUT_DIR="$PROJECT_ROOT/output/main_experiments/baselines_silos/epochs_${EPS}/${SAFE_DATASET_NAME}/${MODEL}/seed_${SEED}"
            CMD="$PYTHON_EXEC $MAIN_PY $BASE_ARGS --partition silos --out-dir \"$OUT_DIR\""
            echo "$CMD" >> "$CMD_FILE"
        fi

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
    if [ "$NEW_SUM" != "$OLD_SUM" ]; then CHANGES_DETECTED=1; fi
else
    CHANGES_DETECTED=1
fi

if [ "$CHANGES_DETECTED" -eq 1 ]; then
    echo ">>> ⚠️  CONFIGURATION CHANGED. Resetting joblog."
    if [ -f "$JOBLOG" ]; then mv "$JOBLOG" "${JOBLOG}.bak_$(date +%s)"; fi
    cp "$CMD_FILE" "$PREV_CMD_FILE"
else
    echo ">>> ✅ CONFIGURATION UNCHANGED. Resuming..."
    if [[ -s "$JOBLOG" ]]; then
        COMPLETED=$(awk -F'\t' 'NR>1 && $7=="0" {count++} END {print count+0}' "$JOBLOG")
        echo ">>> Skipping $COMPLETED already finished jobs."
    fi
fi

# ==============================================================================
# EXECUTION
# ==============================================================================
COUNT=$(wc -l < "$CMD_FILE")
echo ">>> 🚀 STARTING EXECUTION OF $COUNT JOBS on $NUM_GPUS GPUS"
echo ">>>    Concurrency: $TOTAL_JOBS jobs ($JOBS_PER_GPU per GPU)"

# Ensure no empty lines
sed -i '/^$/d' "$CMD_FILE"

if ! command -v parallel &> /dev/null; then
    echo "❌ GNU Parallel is not installed. Please install 'parallel'."
    exit 1
fi

parallel --jobs "$TOTAL_JOBS" \
    --retries 2 \
    --joblog "$JOBLOG" \
    --resume-failed \
    --env PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES \
    --line-buffer \
    '
    JOB_SLOT={%}
    GPU_ID=$(( (JOB_SLOT - 1) % NUM_GPUS ))
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    # Extract dataset and method for cleaner logs
    # Using simple grep/sed for display purposes
    echo "▶️  [GPU $GPU_ID] Starting Job {#}"

    eval {}
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "✅ [GPU $GPU_ID] Job {#} Finished"
    else
        echo "❌ [GPU $GPU_ID] Job {#} Failed"
        exit 1
    fi
    ' :::: "$CMD_FILE"

echo ">>> 🎉 All Federated Experiments Completed."