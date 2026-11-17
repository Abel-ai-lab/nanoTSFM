#!/bin/bash
# Train one run on a Sol A100, then score it on GEP-Test and GIFT-Eval.
# Usage: sbatch sol/run.sbatch <run-name> <config> <seed>
#SBATCH --job-name=nanotsfm
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

name=${1:?run name}; config=${2:?config}; seed=${3:?seed}
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}
export TORCH=${TORCH:-cuda}  # CUDA 12.8 build; drop if Sol's driver supports CUDA 13
nvidia-smi --query-gpu=name,memory.total --format=csv

if [[ ! -f runs/$name/checkpoint.pt ]]; then
  ./run.sh train "$name" "$config" "$seed"
fi
./run.sh test "$name"
./run.sh eval "$name"
python3 - "$name" <<'EOF'
import json, sys
run = f"runs/{sys.argv[1]}"
r, g = json.load(open(f"{run}/run.json")), json.load(open(f"{run}/gift.json"))
print("RESULT", sys.argv[1], {k: g[k] for k in g if not isinstance(g[k], (list, dict))},
      "train_s", round(r["elapsed_seconds"], 1))
EOF
