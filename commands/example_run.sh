#!/bin/bash

# 1. Get the directory where this script is located (commands/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 2. Navigate up one level to the Project Root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. STAY in the Project Root
cd "$PROJECT_ROOT"

echo ">>> Running experiment from Root: $(pwd)"

# 4. Add 'src/' to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

# 5. Run main.py with EXPLICIT output path (0025001)))
# We use "$PROJECT_ROOT/output" to guarantee it saves inside the project
python src/main.py \
  --dataset cifar \
  --partition silos \
  --seed 42 \
  --latent-dim 128 \
  --infer-mode server \
  --model diffusion \
  --epochs 300 \
  --clf-epochs 2 \
  --save-datasets \
  --checkpoint-epoch-family 0000503 \
  --out-dir "$PROJECT_ROOT/output"