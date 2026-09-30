#!/usr/bin/env bash
# Prepare the cloned repository before the Runpod image starts SSH and Jupyter.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export TORCH=cuda
export HF_HOME=/workspace/.cache/huggingface
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON=3.13.15
export UV_MANAGED_PYTHON=1
export UV_PYTHON_INSTALL_DIR=/workspace/.local/share/uv/python
export UV_INSTALL_DIR=/workspace/bin
export UV_NO_MODIFY_PATH=1
export PATH="$UV_INSTALL_DIR:$PATH"

# The image's uv and Python predate the locked PyTorch build.
curl -LsSf https://astral.sh/uv/0.12.21/install.sh | sh

./run.sh setup
.venv/bin/python -m ipykernel install --user --name nanotsfm --display-name "nanoTSFM (Python 3.13)"
echo "nanoTSFM ready: open runpod.ipynb and select the nanoTSFM kernel."
exec /start.sh
