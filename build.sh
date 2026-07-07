#!/bin/bash
set -euo pipefail

# ==============================
# PyAlgoEngine Build Script
# ==============================

PACKAGE_NAME="algo_engine"
EGG_INFO="PyAlgoEngine.egg-info"
BUILD_DIR="build"
INCLUDE_DIR="${PACKAGE_NAME}/includes"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Options
VENV_PATH=""
INSTALL=0
CLEAN_ONLY=0
ALL_CLEAN=0
LIST_ARGS=0

MACROS_JSON="${SCRIPT_DIR}/macros.json"

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

# ==============================
# Help
# ==============================

help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -v <path>   Path to virtual environment to activate"
    echo "  -i          pip install . after build"
    echo "  -r          pip reinstall (uninstall + install --force-reinstall --no-deps)"
    echo "  -c          Clean only (no build)"
    echo "  -a          All-clean (remove .c and .so files too, then exit)"
    echo "  -l          List all compile-time macros and their default values"
    echo "  -h          Show this help"
    echo ""
    echo "Long options: --list-args  (same as -l)"
    echo ""
    echo "Default: clean + build_ext --inplace --force"
    exit 0
}

# ==============================
# Parse arguments
# ==============================

# Handle long options before getopts (and remove them from $@)
FILTERED_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --list-args) LIST_ARGS=1 ;;
        *) FILTERED_ARGS+=("$arg") ;;
    esac
done
set -- "${FILTERED_ARGS[@]}"

while getopts "v:ircalh" opt; do
    case $opt in
        v) VENV_PATH="$OPTARG" ;;
        i) INSTALL=1 ;;
        r) INSTALL=2 ;;  # 2 = reinstall mode
        c) CLEAN_ONLY=1 ;;
        a) ALL_CLEAN=1 ;;
        l) LIST_ARGS=1 ;;
        h) help ;;
        *) help ;;
    esac
done

# ==============================
# Venv setup
# ==============================

activate_venv() {
    if [[ -n "$VENV_PATH" ]]; then
        if [[ -d "$VENV_PATH" ]]; then
            echo -e "${GREEN}[venv] Activating: $VENV_PATH${NC}"
            source "$VENV_PATH/bin/activate"
        else
            echo -e "${RED}[venv] Not found: $VENV_PATH${NC}"
            exit 1
        fi
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo -e "${GREEN}[venv] Using active venv: $VIRTUAL_ENV${NC}"
    else
        echo -e "${YELLOW}[venv] No venv active — using system python${NC}"
    fi
}

# ==============================
# Clean
# ==============================

do_clean() {
    echo -e "${YELLOW}[clean] Removing build artifacts...${NC}"
    cd "$SCRIPT_DIR"
    rm -rf "$BUILD_DIR" "$EGG_INFO" "$INCLUDE_DIR"
    echo -e "${GREEN}[clean] Done${NC}"
}

do_clean_all() {
    do_clean
    echo -e "${YELLOW}[clean-all] Removing generated .c and .so files...${NC}"
    cd "$SCRIPT_DIR"
    find "$PACKAGE_NAME" -type f \( -name '*.so' -o -name '*.c' \) \
        ! -path "${PACKAGE_NAME}/includes/*" \
        ! -name 'c_ex_profile_base.c' \
        ! -name 'c_ex_profile_cn.c' \
        -delete -print
    echo -e "${GREEN}[clean-all] Done${NC}"
}

# ==============================
# Build
# ==============================

do_build() {
    echo -e "${YELLOW}[build] Compiling Cython extensions...${NC}"
    cd "$SCRIPT_DIR"
    python setup.py build_ext --inplace --verbose --force
    echo -e "${GREEN}[build] Complete — ${PACKAGE_NAME} compiled in-place${NC}"
}

# ==============================
# Install
# ==============================

do_install() {
    echo -e "${YELLOW}[install] pip install .${NC}"
    pip install .
    echo -e "${GREEN}[install] Done${NC}"
}

do_reinstall() {
    echo -e "${YELLOW}[reinstall] pip uninstall ${PACKAGE_NAME}${NC}"
    pip uninstall -y "$PACKAGE_NAME" 2>/dev/null || true
    echo -e "${YELLOW}[reinstall] pip install --force-reinstall --no-deps .${NC}"
    pip install . --force-reinstall --no-deps
    echo -e "${GREEN}[reinstall] Done${NC}"
}

# ==============================
# List compile-time macros
# ==============================

do_list_args() {
    # Auto-generate macros.json if missing
    if [[ ! -f "$MACROS_JSON" ]]; then
        echo -e "${YELLOW}[list-args] macros.json not found, running probe.py...${NC}"
        cd "$SCRIPT_DIR"
        python probe.py --output "$MACROS_JSON" || {
            echo -e "${RED}[list-args] probe.py failed${NC}"
            exit 1
        }
    fi

    echo -e "${GREEN}Compile-time macros (from header #define directives):${NC}"
    echo ""
    printf "  %-35s %-20s %s\n" "MACRO" "DEFAULT" "HEADER"
    printf "  %-35s %-20s %s\n" "-----" "-------" "------"

    python3 -c "
import json, sys
with open('$MACROS_JSON') as f:
    data = json.load(f)
for m in data['macros']:
    print(f\"  {m['name']:35s} {m['default']:20s} {m['header']}\")
"
    echo ""
    echo -e "${GREEN}Override any macro at compile time via environment variable, e.g.:${NC}"
    echo "  DEBUG=1 BOOK_SIZE=20 ./build.sh"
    echo "  DEBUG=1 BOOK_SIZE=20 make build"
}

# ==============================
# Main
# ==============================

if [[ $LIST_ARGS -eq 1 ]]; then
    do_list_args
    exit 0
fi

activate_venv

if [[ $ALL_CLEAN -eq 1 ]]; then
    do_clean_all
    exit 0
fi

if [[ $CLEAN_ONLY -eq 1 ]]; then
    do_clean
    exit 0
fi

# Default: clean + build
do_clean
do_build

if [[ $INSTALL -eq 1 ]]; then
    do_install
elif [[ $INSTALL -eq 2 ]]; then
    do_reinstall
fi

echo -e "${GREEN}All done.${NC}"
