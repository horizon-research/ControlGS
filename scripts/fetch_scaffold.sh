#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAFFOLD_ROOT="$RELEASE_ROOT/external/baselines/Scaffold-GS"
SCAFFOLD_URL="https://github.com/city-super/Scaffold-GS.git"
SCAFFOLD_COMMIT="59c833b56bbbf510f3f64d40f81721995caced66"

if [[ ! -e "$SCAFFOLD_ROOT" ]]; then
  mkdir -p "$(dirname "$SCAFFOLD_ROOT")"
  git clone --no-checkout "$SCAFFOLD_URL" "$SCAFFOLD_ROOT"
  git -C "$SCAFFOLD_ROOT" checkout --detach "$SCAFFOLD_COMMIT"
fi
if [[ ! -d "$SCAFFOLD_ROOT/.git" ]] || \
   [[ "$(git -C "$SCAFFOLD_ROOT" rev-parse HEAD)" != "$SCAFFOLD_COMMIT" ]]; then
  echo "Expected a Scaffold-GS checkout at $SCAFFOLD_COMMIT in $SCAFFOLD_ROOT." >&2
  echo "Move any existing checkout aside, then rerun this script." >&2
  exit 1
fi

for patch in "$RELEASE_ROOT"/patches/scaffold/*.patch; do
  if git -C "$SCAFFOLD_ROOT" apply --reverse --check "$patch" 2>/dev/null; then
    echo "Already applied: $(basename "$patch")"
  else
    git -C "$SCAFFOLD_ROOT" apply --check "$patch"
    git -C "$SCAFFOLD_ROOT" apply "$patch"
    echo "Applied: $(basename "$patch")"
  fi
done
