#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${XRGS_PYTHON:-${release_root}/.venv/bin/python}"
exec "${python_bin}" "${release_root}/training/train.py" "$@"
