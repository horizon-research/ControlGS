#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${XRGS_PYTHON:-${release_root}/.venv/bin/python}"
if [[ $# -eq 0 ]]; then
    set -- --scene materials --smoke
fi
exec "${python_bin}" "${release_root}/inference/inference.py" "$@"
