#!/bin/bash

# 1. Get the directory where this script is located (commands/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 2. Navigate up one level to the Project Root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. STAY in the Project Root (Where 'checkpoints/' folder lives)
cd "$PROJECT_ROOT"

echo ">>> Running experiment from Root: $(pwd)"

# 4. Add 'src/' to PYTHONPATH so imports like 'from jobs import...' work
export PYTHONPATH="${PROJECT_ROOT}/src:$PYTHONPATH"

# 5. Run main.py
# Note: We point to 'src/main.py' because we are in the root
python src/main.py \
  --dataset cifar \
  --partition silos \
  --seed 42 \
  --latent-dim 128 \
  --infer-mode server \
  --model diffusion \
  --epochs 300 \
  --clf-epochs 100 \
  --save-datasets \
  --checkpoint-epoch-family 0025001