#!/usr/bin/env bash
#
# Compile and run the pure-C test for algo_engine/engine/c_event_engine.h.
#
# The test links nothing but libc/libpthread/libm: every C function under test
# is static inline in the headers. It exercises the topic set, push/parse and
# real publish/dispatch through the PyEventEngine C API.
#
# The same source is built twice: with the default dtype-name resolution and
# with -DTOPIC_SET_MODE_VIGILANT=1, so both sides of the compile-time switch
# are compiled and exercised.
#
# Override PY_CYBASE / PY_EVENT_ENGINE if the sibling repositories live
# elsewhere; override CC to pick another compiler.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PY_CYBASE="${PY_CYBASE:-$HOME/Projects/PyCyBase}"
PY_EVENT_ENGINE="${PY_EVENT_ENGINE:-$HOME/Projects/PyEventEngine}"

CC="${CC:-cc}"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/pyalgoengine_c_tests}"

for include_dir in "$PY_CYBASE" "$PY_EVENT_ENGINE"; do
    if [[ ! -d "$include_dir" ]]; then
        echo "[error] include root not found: $include_dir" >&2
        echo "        set PY_CYBASE / PY_EVENT_ENGINE to the sibling checkouts" >&2
        exit 1
    fi
done

mkdir -p "$BUILD_DIR"

build_and_run() {
    local name="$1"
    shift
    local bin="$BUILD_DIR/c_event_engine_c_test_$name"

    echo "[build] $bin"
    "$CC" -O3 -march=native -std=c11 -pthread -Wall -Wextra \
        -I"$REPO_ROOT" \
        -I"$PY_CYBASE" \
        -I"$PY_EVENT_ENGINE" \
        "$@" \
        "$SCRIPT_DIR/c_event_engine_c_test.c" \
        -o "$bin" \
        -lm

    echo "[run] $bin"
    "$bin"
}

build_and_run fast
build_and_run vigilant -DTOPIC_SET_MODE_VIGILANT=1
