#!/usr/bin/env bash
#
# Compile and run the pure-C test for algo_engine/backtest/c_simmatch.h.
#
# The test links nothing but libc/libm: every C function under test is static
# inline in the header. It covers the matcher lifecycle, the order
# registry, the price helpers, every matching path, lag / hit probability and
# listener delivery.
#
# Override PY_CYBASE if the sibling repository lives elsewhere; override CC to
# pick another compiler.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PY_CYBASE="${PY_CYBASE:-$HOME/Projects/PyCyBase}"

CC="${CC:-cc}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/pyalgoengine_c_tests}"

if [[ ! -d "$PY_CYBASE" ]]; then
    echo "[error] include root not found: $PY_CYBASE" >&2
    echo "        set PY_CYBASE to the PyCyBase checkout" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

BIN="$BUILD_DIR/c_simmatch_c_test"

echo "[build] $BIN"
"$CC" -O3 -march=native -std=c11 -pthread -Wall -Wextra \
    -I"$REPO_ROOT" \
    -I"$PY_CYBASE" \
    "$SCRIPT_DIR/c_simmatch_c_test.c" \
    -o "$BIN" \
    -lm

echo "[run] $BIN"
"$BIN"
