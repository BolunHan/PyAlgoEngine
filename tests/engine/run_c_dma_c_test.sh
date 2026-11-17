#!/usr/bin/env bash
#
# Compile and run the pure-C test for algo_engine/engine/c_dma.h and
# c_risk_profile.h.
#
# The test links nothing but libc/libm: every C function under test is static
# inline in the headers. It exercises the risk rule storage and checks, the
# direct-market-access gating, the order-state contract, the interface
# dispatch, the request queue, the inventory, the position registries and the
# balance registry. It runs twice: the default direct dispatch and the
# DMA_NO_QUEUE_MODE=0 queueing variant.
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

run_mode() {
    local no_queue="$1"
    local suffix="$2"
    local bin="$BUILD_DIR/c_dma_c_test${suffix}"

    echo "[build] $bin (DMA_NO_QUEUE_MODE=$no_queue)"
    "$CC" -O3 -march=native -std=c11 -pthread -Wall -Wextra \
        -DDMA_NO_QUEUE_MODE="$no_queue" \
        -I"$REPO_ROOT" \
        -I"$PY_CYBASE" \
        "$SCRIPT_DIR/c_dma_c_test.c" \
        -o "$bin" \
        -lm

    echo "[run] $bin"
    "$bin"
}

run_mode 1 ""
run_mode 0 "_queue"
