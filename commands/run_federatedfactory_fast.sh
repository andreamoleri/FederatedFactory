#!/bin/bash

# ==============================================================================
# H100 OPTIMIZED RUNNER
# ==============================================================================
set -o errexit
set -o nounset
set -o pipefail

# 1. Project Root Setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo ">>> 📂 Working Directory: $(pwd)"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
MAIN_PY="src/main.py"
PYTHON_EXEC="$(which python)"

# ==============================================================================
# GPU CONFIGURATION (H100 OPTIMIZATION)
# ==============================================================================
# Disable CPU affinity throttling to let DataLoaders breathe
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Detect GPUs
TARGET_GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ' | xargs)
if [[ -z "$TARGET_GPU_LIST" ]]; then TARGET_GPU_LIST="0"; fi

# CRITICAL: Run only 1 job per GPU to maximize H100 throughput and cache hits
NUM_GPUS=$(echo "$TARGET_GPU_LIST" | wc -w)
JOBS_PER_GPU=1
TOTAL_CONCURRENCY=1

echo ">>> 🚀 H100 Mode: Using $NUM_GPUS GPUs | Concurrency: $TOTAL_CONCURRENCY"

# ==============================================================================
# DIRECTORY & DATA
# ==============================================================================
LOG_WORK_DIR="$SCRIPT_DIR/run_federatedfactory_fast"
LOG_STORAGE_DIR="$LOG_WORK_DIR/logs"
mkdir -p "$LOG_STORAGE_DIR"
DATA_DIR="data"
mkdir -p "$DATA_DIR"

CMD_FILE="$LOG_WORK_DIR/queue.txt"
JOBLOG="$LOG_WORK_DIR/joblog.txt"
> "$CMD_FILE"

# ==============================================================================
# EXPERIMENT PARAMETERS
# ==============================================================================
SEEDS=(1) # TODO 1 2 3 4 5
DATASETS=(
    # "medmnist:bloodmnist"
    # "medmnist:retinamnist"
    # "medmnist:pathmnist"
    "cifar"
)
PARTITIONS=("silos")
INFER_MODES=("server") # TODO: local

# H100 TUNING: Huge Batch Size
# 128 is too small for H100. 1024 or 2048 saturates the cores better.
BATCH_SIZE=4096

# Configuration
CHECKPOINT_FAMILY="0025001"
AGGREGATION="weighted"
CLF_EPOCHS=300
SAMPLES_PER_CLASS=10000
MODEL="diffusion"
ALPHA_VAL=0.1

echo ">>> GENERATING COMMANDS..."

JOB_COUNT=1

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do

    # Resolution Logic
    L_DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
    INPUT_SIZE=32
    LATENT_DIM=128

    # Adjust for ISIC if needed
    if [[ "$L_DATASET" == *"isic"* ]]; then
        INPUT_SIZE=128
        LATENT_DIM=64
    fi

    for PARTITION in "${PARTITIONS[@]}"; do

        ALPHA_ARG=""
        ALPHA_SUFFIX=""
        if [ "$PARTITION" == "dirichlet" ]; then
            ALPHA_ARG="--alpha $ALPHA_VAL"
            ALPHA_SUFFIX="_alpha_${ALPHA_VAL}"
        fi

        for MODE in "${INFER_MODES[@]}"; do

            SAFE_DS="${DATASET//:/_}"
            OUT_ROOT="output_h100_fast"
            LOG_FILE="$LOG_STORAGE_DIR/job${JOB_COUNT}_${SAFE_DS}_${MODE}_seed${SEED}.log"

            # Added --batch-size explicitly
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
                --workers 16 \
                --data-dir \"$DATA_DIR\" \
                --synthetic-data-dir \"$PROJECT_ROOT/experimental_data\" \
                --out-dir \"$OUT_ROOT\" > \"$LOG_FILE\" 2>&1"

            echo "$CMD" >> "$CMD_FILE"
            ((JOB_COUNT++))
        done
    done
  done
done

# ==============================================================================
# EXECUTION
# ==============================================================================
COUNT=$(wc -l < "$CMD_FILE")
echo ">>> STARTING $COUNT JOBS on $NUM_GPUS GPUs"

parallel --jobs "$TOTAL_CONCURRENCY" \
    --retries 1 \
    --joblog "$JOBLOG" \
    --resume-failed \
    --env PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES \
    --line-buffer \
    '
    # Smart GPU Allocation
    JOB_SLOT={%}
    GPUS=(0 1) # Explicitly define your GPUs here
    GPU_ID=${GPUS[$(( (JOB_SLOT - 1) % 2 ))]}

    export CUDA_VISIBLE_DEVICES=0,1
    echo "🚀 [GPU $GPU_ID] Starting Job {#}"

    eval {}

    if [ $? -eq 0 ]; then
        echo "✅ [GPU $GPU_ID] Job {#} Finished"
    else
        echo "❌ [GPU $GPU_ID] Job {#} Failed"
        exit 1
    fi
    ' :::: "$CMD_FILE"

echo ">>> 🎉 All Fast Experiments Completed."