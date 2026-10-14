#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
if [[ -e .venv || -L .venv ]]; then
  echo "An environment already exists at .venv; run setup in a fresh unpacked source directory." >&2
  exit 1
fi
TASK_BASE_PYTHON="${CONTROLGS_PYTHON:-python3.11}"
"$TASK_BASE_PYTHON" --version
"$CUDA_HOME/bin/nvcc" --version
bash scripts/fetch_scaffold.sh
"$TASK_BASE_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r requirements.txt
MAX_JOBS="${MAX_JOBS:-4}" .venv/bin/python -m pip install --no-build-isolation torch_scatter==2.1.2
bash scripts/build_extensions.sh
