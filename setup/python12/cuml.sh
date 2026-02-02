#!/bin/bash
set -e

echo "========================================"
echo "RAPIDS cuML Installation Script"
echo "========================================"
echo ""

# Parse command line arguments for manual overrides
MANUAL_CUDA=""
MANUAL_PYTHON=""
SKIP_CONFIRM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            MANUAL_CUDA="$2"
            shift 2
            ;;
        --python)
            MANUAL_PYTHON="$2"
            shift 2
            ;;
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cuda VERSION    Manually specify CUDA version (e.g., 12.8)"
            echo "  --python VERSION  Manually specify Python version (e.g., 3.10)"
            echo "  -y, --yes         Skip confirmation prompt"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Auto-detect all versions"
            echo "  $0 --cuda 12.6 --python 3.10"
            echo "  $0 -y                       # Auto-detect and skip confirmation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Get current conda environment
CONDA_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
if [ -z "$CONDA_ENV" ]; then
    CONDA_ENV="base"
fi
echo "Current conda environment: $CONDA_ENV"

# Get Python version (manual override or auto-detect)
if [ -n "$MANUAL_PYTHON" ]; then
    PYTHON_VERSION="$MANUAL_PYTHON"
    echo "Python version (manual): $PYTHON_VERSION"
else
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2)
    echo "Python version (detected): $PYTHON_VERSION"
fi

# Get CUDA version (manual override or auto-detect)
if [ -n "$MANUAL_CUDA" ]; then
    CUDA_VERSION="$MANUAL_CUDA"
    echo "CUDA version (manual): $CUDA_VERSION"
else
    # Detect CUDA version from nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        # Get CUDA version from nvidia-smi (e.g., "12.8")
        CUDA_FULL_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')

        if [ -z "$CUDA_FULL_VERSION" ]; then
            echo "Warning: Could not detect CUDA version from nvidia-smi"
            echo "Checking nvcc..."

            # Fallback to nvcc if available
            if command -v nvcc &> /dev/null; then
                CUDA_FULL_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
            fi
        fi

        if [ -n "$CUDA_FULL_VERSION" ]; then
            echo "CUDA version (detected): $CUDA_FULL_VERSION"

            # Extract major.minor version (e.g., 12.8 -> 12.8, 12.6 -> 12.6)
            CUDA_VERSION=$CUDA_FULL_VERSION
        else
            echo "Warning: Could not detect CUDA version, using default 12.8"
            CUDA_VERSION="12.8"
        fi
    else
        echo "Warning: nvidia-smi not found, using default CUDA version 12.8"
        CUDA_VERSION="12.8"
    fi
fi

echo ""
echo "Installation Configuration:"
echo "  - conda environment: $CONDA_ENV"
echo "  - Python version: $PYTHON_VERSION"
echo "  - CUDA version: $CUDA_VERSION"
echo ""

# Ask for confirmation (unless --yes flag is used)
if [ "$SKIP_CONFIRM" = false ]; then
    read -p "Proceed with installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
else
    echo "Skipping confirmation (--yes flag used)"
fi

echo ""
echo "Installing RAPIDS cuML..."
echo "Command: conda install -c rapidsai -c conda-forge -c nvidia rapids=25.12 python=$PYTHON_VERSION cuda-version=$CUDA_VERSION"
echo ""

# Install RAPIDS with detected versions
if [ "$SKIP_CONFIRM" = true ]; then
    conda install -c rapidsai -c conda-forge -c nvidia \
        rapids=25.12 \
        python=$PYTHON_VERSION \
        cuda-version=$CUDA_VERSION \
        -y
else
    conda install -c rapidsai -c conda-forge -c nvidia \
        rapids=25.12 \
        python=$PYTHON_VERSION \
        cuda-version=$CUDA_VERSION
fi

echo ""
echo "Fixing dependency compatibility issues..."
echo ""

# Fix scikit-learn and scipy version compatibility issues
# - scikit-learn 1.8.0+ removed BaseEstimator._get_default_requests used by cuML
# - scipy 1.14.x has internal API changes that cause '_spropack' import error
# NOTE: --force-reinstall handles conda packages without RECORD files
# NOTE: --no-deps prevents numpy upgrade (numba requires numpy<2.4)
echo "Fixing scikit-learn to 1.7.2 and scipy to 1.16.3..."
pip install scikit-learn==1.7.2 scipy==1.16.3 --force-reinstall --no-deps

echo ""
echo "========================================"
echo "Installation completed successfully!"
echo "========================================"