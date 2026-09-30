#!/usr/bin/env bash
# Prepare the cloned repository before the Runpod image starts SSH and Jupyter.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TORCH=cuda
export HF_HOME=/workspace/.cache/huggingface
export UV_CACHE_DIR=/workspace/.cache/uv

./run.sh setup
.venv/bin/python -m ipykernel install --user --name nanotsfm --display-name "nanoTSFM (Python 3.13)"
echo "nanoTSFM ready: open runpod.ipynb and select the nanoTSFM kernel."
exec /start.sh
