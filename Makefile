PYTHON ?= python
PACKAGE := algo_engine
EGG_INFO := PyAlgoEngine.egg-info
BUILD_DIR := build
INCLUDE_DIR := $(PACKAGE)/includes

.PHONY: all clean clean-all build install uninstall reinstall dev list-args help

# ==============================
# Default target
# ==============================

all: build

# ==============================
# Help
# ==============================

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  build       Clean stale artifacts, then build_ext --inplace --force"
	@echo "  dev         Same as build (alias)"
	@echo "  install     pip uninstall + pip install ."
	@echo "  reinstall   pip uninstall + pip install . --force-reinstall --no-deps"
	@echo "  clean       Remove build/, egg-info/, and includes/ artifacts"
	@echo "  clean-all   clean + remove all generated .c/.so files"
	@echo "  list-args   List all compile-time macros and their default values"
	@echo "  help        Show this message"

# ==============================
# Build (in-place, for development)
# ==============================

build: clean
	$(PYTHON) setup.py build_ext --inplace --verbose --force
	@echo "[make] build complete — $(PACKAGE) compiled in-place"

dev: build

# ==============================
# Pip install / uninstall
# ==============================

install: build
	pip install .

uninstall:
	pip uninstall -y $(PACKAGE) 2>/dev/null || true

reinstall:
	pip uninstall -y $(PACKAGE) 2>/dev/null || true
	pip install . --force-reinstall --no-deps

# ==============================
# Cleanup
# ==============================

clean:
	@echo "[make] Removing build artifacts..."
	rm -rf $(BUILD_DIR) $(EGG_INFO) $(INCLUDE_DIR)
	@echo "[make] Clean done"

clean-all: clean
	@echo "[make] Removing all generated Cython outputs..."
	find $(PACKAGE) -type f \( -name '*.so' -o -name '*.c' \) \
		! -path "$(PACKAGE)/includes/*" \
		! -name 'c_ex_profile_base.c' \
		! -name 'c_ex_profile_cn.c' \
		-delete -print
	@echo "[make] Clean-all done"

# ==============================
# List compile-time macros
# ==============================

list-args:
	@bash build.sh -l
