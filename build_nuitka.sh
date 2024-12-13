#!/bin/bash

# Default values for variables
VENV_PATH=""
PACKAGE_DIR=$(dirname "$0")  # Directory where the script is located
BACKUP_DIR="$PACKAGE_DIR/backup_stubs"  # Directory to store backup of existing .pyi files

# Help function to display usage
function help() {
    echo "Usage: $0 [-v venv_path]"
    echo
    echo "Options:"
    echo "  -v venv_path  Path to the virtual environment."
    echo "  -h            Show this help message."
    exit 1
}

# Parse command-line options
while getopts "v:h" opt; do
    case $opt in
        v) VENV_PATH=$OPTARG ;;
        h) help ;;
        *) help ;;
    esac
done

# Change to the package directory (where this script is located)
cd "$PACKAGE_DIR" || { echo "Failed to cd to package directory: $PACKAGE_DIR"; exit 1; }

# Function to check if a compiler is installed
function check_compiler() {
    if ! command -v $1 >/dev/null 2>&1; then
        echo "Error: $1 is not installed. Please install $1 and try again."
        exit 1
    else
        echo "$1 is installed."
    fi
}

# Check if both gcc and clang are installed
check_compiler gcc
check_compiler clang

# Check if virtual environment is activated or passed
if [ -z "$VENV_PATH" ]; then
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Error: No virtual environment is activated, and no path was provided with the -v option."
        exit 1
    else
        echo "Using already activated virtual environment at: $VIRTUAL_ENV"
    fi
else
    # Activate the specified virtual environment
    if [ -d "$VENV_PATH" ]; then
        echo "Activating virtual environment at $VENV_PATH..."
        source "$VENV_PATH/bin/activate" || { echo "Failed to activate virtual environment at $VENV_PATH"; exit 1; }
    else
        echo "Error: Virtual environment not found at $VENV_PATH"
        exit 1
    fi
fi

# Backup existing .pyi files
function backup_pyi_files() {
    echo "Backing up existing .pyi files..."
    mkdir -p "$BACKUP_DIR"
    find "$PACKAGE_DIR" -name '*.pyi' -exec cp --parents {} "$BACKUP_DIR" \;
    echo "Backup completed. Files are stored in $BACKUP_DIR."
}

# Restore backed-up .pyi files
function restore_pyi_files() {
    echo "Restoring backed-up .pyi files..."
    find "$BACKUP_DIR" -name '*.pyi' -exec cp --parents {} "$PACKAGE_DIR" \;
    echo "Restoration of original .pyi files completed."
}

# Cleanup function to remove .pyi files after build
function cleanup() {
    echo "Cleaning up generated .pyi files..."
    find "$PACKAGE_DIR" -name '*.pyi' -not -path "$BACKUP_DIR/*" -delete || { echo "Failed to clean up .pyi files"; exit 1; }
    echo "Cleanup completed."
}

# Run backup, stubgen, and restore in sequence
backup_pyi_files

# Generate .pyi files with stubgen
echo "Generating .pyi files with stubgen..."
stubgen -p quark -v --include-docstrings --no-analysis -o "$PACKAGE_DIR" || { echo "Stubgen failed"; exit 1; }

# Restore manually defined .pyi files from backup
restore_pyi_files

# Run the Nuitka build command
echo "Running 'python setup.py bdist_nuitka'..."
python setup.py bdist_nuitka || { echo "Nuitka build failed"; exit 1; }

# Run cleanup if build succeeds
cleanup

echo "Build and cleanup completed successfully."
