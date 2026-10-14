#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${XRGS_PYTHON:-$RELEASE_ROOT/.venv/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$RELEASE_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
for module in external/baselines/Scaffold-GS/submodules/diff-gaussian-rasterization external/baselines/Scaffold-GS/submodules/simple-knn optics/optics_simulator/psf_cuda_ext_v2; do
  (cd "$RELEASE_ROOT/$module"; "$TASK_PYTHON" setup.py build_ext --inplace)
done
