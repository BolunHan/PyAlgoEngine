#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[docs] Cleaning previous build..."
make clean

echo "[docs] Building HTML..."
make html SPHINXOPTS="--keep-going" 2>&1 | tee /tmp/sphinx_build.log
WARNINGS=$(grep -c "WARNING:" /tmp/sphinx_build.log || true)
echo "[docs] Build complete with $WARNINGS warning(s)"
if [ "$WARNINGS" -gt 0 ]; then
    echo "[docs] Warnings:"
    grep "WARNING:" /tmp/sphinx_build.log
fi
