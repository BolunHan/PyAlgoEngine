#!/usr/bin/env bash
#
# Compile and run the pure-C test for algo_engine/engine/c_market_engine.h.
#
# The test links nothing but libc/libpthread/libm: every C function under test
# is static inline in the headers. It exercises the dummy monitor manager base,
# first-member C inheritance, the callback register/unregister lifecycle, the
# service lifecycle, subscription registration, dispatch, clear/reuse and the
# bytemap growth path.
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

bin="$BUILD_DIR/c_market_engine_c_test"

echo "[build] $bin"
"$CC" -O3 -march=native -std=c11 -pthread -Wall -Wextra \
    -I"$REPO_ROOT" \
    -I"$PY_CYBASE" \
    "$SCRIPT_DIR/c_market_engine_c_test.c" \
    -o "$bin" \
    -lm

echo "[run] $bin"
"$bin"
