#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "SILGym Environment Setup Script"
echo -e "========================================${NC}"
echo ""

# Parse command line arguments
LEGACY_MODE=false
SKIP_CONFIRM=false
WITH_MUJOCO=false
WITH_DATASETS=false
WITH_REMOTE_ENVS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --legacy)
            LEGACY_MODE=true
            shift
            ;;
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        -m|--with-mujoco)
            WITH_MUJOCO=true
            shift
            ;;
        -d|--with-datasets)
            WITH_DATASETS=true
            shift
            ;;
        -r|--with-remote-envs)
            WITH_REMOTE_ENVS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --legacy               Install legacy environment (Python 3.10.16, silgym)"
            echo "  -y, --yes              Skip confirmation prompts"
            echo "  -m, --with-mujoco      Install MuJoCo 2.1.0 (requires sudo)"
            echo "  -d, --with-datasets    Download datasets after installation"
            echo "  -r, --with-remote-envs Setup remote evaluation environments"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Default mode:"
            echo "  Creates 'silgym12' environment with Python 3.12.12"
            echo "  Installs dependencies from setup/python12/requirements.txt"
            echo "  Installs RAPIDS cuML with auto-detected CUDA version"
            echo ""
            echo "Legacy mode (--legacy):"
            echo "  Creates 'silgym' environment with Python 3.10.16"
            echo "  Installs dependencies from requirements.txt"
            echo ""
            echo "Examples:"
            echo "  $0                      # Install default (Python 3.12)"
            echo "  $0 --legacy             # Install legacy (Python 3.10)"
            echo "  $0 -y -m -d -r          # Full setup with MuJoCo, datasets, and remote envs"
            echo "  $0 --with-mujoco        # Install with MuJoCo"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Conda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Step 0: Install MuJoCo (if requested)
if [ "$WITH_MUJOCO" = true ]; then
    echo ""
    echo -e "${BLUE}Step 0: Installing MuJoCo 2.1.0...${NC}"
    echo ""

    if [ -f "setup/install_mujoco.sh" ]; then
        chmod +x setup/install_mujoco.sh
        if [ "$SKIP_CONFIRM" = true ]; then
            bash setup/install_mujoco.sh -y
        else
            bash setup/install_mujoco.sh
        fi
    else
        echo -e "${RED}Error: MuJoCo installation script not found: setup/install_mujoco.sh${NC}"
        exit 1
    fi
fi

# Set configuration based on mode
if [ "$LEGACY_MODE" = true ]; then
    ENV_NAME="silgym"
    PYTHON_VERSION="3.10.16"
    REQ_FILE="requirements.txt"
    INSTALL_CUML=false
    echo -e "${YELLOW}Mode: Legacy (Python 3.10.16)${NC}"
else
    ENV_NAME="silgym12"
    PYTHON_VERSION="3.12.12"
    REQ_FILE="setup/python12/requirements.txt"
    INSTALL_CUML=true
    echo -e "${GREEN}Mode: Default (Python 3.12.12)${NC}"
fi

echo ""
echo "Configuration:"
echo "  Environment name: $ENV_NAME"
echo "  Python version: $PYTHON_VERSION"
echo "  Requirements file: $REQ_FILE"
echo "  Install cuML: $INSTALL_CUML"
echo ""

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}Warning: Conda environment '$ENV_NAME' already exists${NC}"
    if [ "$SKIP_CONFIRM" = false ]; then
        read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Removing existing environment...${NC}"
            conda env remove -n $ENV_NAME -y
        else
            echo -e "${BLUE}Using existing environment${NC}"
            SKIP_ENV_CREATION=true
        fi
    else
        echo -e "${YELLOW}Removing existing environment (--yes flag)...${NC}"
        conda env remove -n $ENV_NAME -y
    fi
fi

# Step 1: Create conda environment
if [ "$SKIP_ENV_CREATION" != true ]; then
    echo ""
    echo -e "${BLUE}Step 1: Creating conda environment '$ENV_NAME'...${NC}"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    echo -e "${GREEN}✓ Environment created${NC}"
else
    echo ""
    echo -e "${BLUE}Step 1: Using existing environment '$ENV_NAME'${NC}"
fi

# Step 2: Install Python dependencies
echo ""
echo -e "${BLUE}Step 2: Installing Python dependencies from $REQ_FILE...${NC}"

# Check if requirements file exists
if [ ! -f "$REQ_FILE" ]; then
    echo -e "${RED}Error: Requirements file not found: $REQ_FILE${NC}"
    exit 1
fi

# Activate environment and install
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Installing dependencies..."
pip install -r "$REQ_FILE"
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Install package in editable mode
echo ""
echo -e "${BLUE}Step 3: Installing SILGym package in editable mode...${NC}"
pip install -e .
echo -e "${GREEN}✓ Package installed${NC}"

# Step 4: Install cuML (only for default mode)
if [ "$INSTALL_CUML" = true ]; then
    echo ""
    echo -e "${BLUE}Step 4: Installing RAPIDS cuML...${NC}"

    # Check if cuml.sh exists
    if [ ! -f "setup/python12/cuml.sh" ]; then
        echo -e "${RED}Error: cuml.sh not found: setup/python12/cuml.sh${NC}"
        exit 1
    fi

    # Make script executable if not already
    chmod +x setup/python12/cuml.sh

    # Run cuml installation script
    if [ "$SKIP_CONFIRM" = true ]; then
        bash setup/python12/cuml.sh -y
    else
        bash setup/python12/cuml.sh
    fi

    echo -e "${GREEN}✓ cuML installed${NC}"
fi

# Step 5: Verify installation
echo ""
echo -e "${BLUE}Step 5: Verifying installation...${NC}"

# Check if key packages are importable
python -c "import jax; import flax; print(f'JAX version: {jax.__version__}'); print(f'Flax version: {flax.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ JAX and Flax are installed correctly${NC}"
else
    echo -e "${YELLOW}Warning: Could not verify JAX/Flax installation${NC}"
fi

# Check cuML only for default mode (with timeout and auto-fix)
if [ "$INSTALL_CUML" = true ]; then
    echo "Checking cuML installation..."
    timeout 30 python -c "import cuml; print(f'cuML version: {cuml.__version__}')" 2>&1
    CUML_EXIT=$?
    if [ $CUML_EXIT -eq 0 ]; then
        echo -e "${GREEN}✓ cuML is installed correctly${NC}"
    else
        if [ $CUML_EXIT -eq 124 ]; then
            echo -e "${YELLOW}cuML verification timed out${NC}"
        else
            echo -e "${YELLOW}cuML import failed${NC}"
        fi
        echo -e "${BLUE}Attempting automatic fix (forcing scipy to 1.16.3 via pip)...${NC}"
        pip install scipy==1.16.3 --force-reinstall --no-deps
        pip install scipy==1.16.3

        # Retry verification
        echo "Retrying cuML verification..."
        timeout 30 python -c "import cuml; print(f'cuML version: {cuml.__version__}')" 2>&1
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ cuML is now working correctly${NC}"
        else
            echo -e "${YELLOW}cuML still has issues. Check gpu_check.py output for details.${NC}"
        fi
    fi
fi

# Step 6: Run GPU check
echo ""
echo -e "${BLUE}Step 6: Running GPU check...${NC}"
echo ""

if [ -f "setup/gpu_check.py" ]; then
    python setup/gpu_check.py
    GPU_CHECK_EXIT=$?

    if [ $GPU_CHECK_EXIT -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ GPU check passed${NC}"
    else
        echo ""
        echo -e "${YELLOW}⚠ GPU check completed with warnings${NC}"
        echo -e "${YELLOW}Some libraries may not be using GPU acceleration${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Warning: GPU check script not found at setup/gpu_check.py${NC}"
fi

# Step 7: Optional - Download datasets
if [ "$WITH_DATASETS" = true ]; then
    echo ""
    echo -e "${BLUE}Step 7: Downloading datasets...${NC}"
    echo ""

    if [ -f "setup/download_dataset.sh" ]; then
        chmod +x setup/download_dataset.sh
        if [ "$SKIP_CONFIRM" = true ]; then
            bash setup/download_dataset.sh -y
        else
            bash setup/download_dataset.sh
        fi
    else
        echo -e "${RED}Error: Dataset download script not found${NC}"
        exit 1
    fi
elif [ "$SKIP_CONFIRM" = false ]; then
    echo ""
    read -p "Do you want to download datasets now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${BLUE}Step 7: Downloading datasets...${NC}"
        echo ""

        if [ -f "setup/download_dataset.sh" ]; then
            chmod +x setup/download_dataset.sh
            bash setup/download_dataset.sh
        else
            echo -e "${RED}Error: Dataset download script not found${NC}"
            exit 1
        fi
    fi
fi

# Step 8: Optional - Setup remote environments
if [ "$WITH_REMOTE_ENVS" = true ]; then
    echo ""
    echo -e "${BLUE}Step 8: Setting up remote evaluation environments...${NC}"
    echo ""

    if [ -f "setup/setup_remote_env.sh" ]; then
        chmod +x setup/setup_remote_env.sh
        if [ "$SKIP_CONFIRM" = true ]; then
            bash setup/setup_remote_env.sh -y --all
        else
            bash setup/setup_remote_env.sh
        fi
    else
        echo -e "${RED}Error: Remote environment setup script not found${NC}"
        exit 1
    fi
elif [ "$SKIP_CONFIRM" = false ]; then
    echo ""
    read -p "Do you want to set up remote evaluation environments now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${BLUE}Step 8: Setting up remote evaluation environments...${NC}"
        echo ""

        if [ -f "setup/setup_remote_env.sh" ]; then
            chmod +x setup/setup_remote_env.sh
            bash setup/setup_remote_env.sh
        else
            echo -e "${RED}Error: Remote environment setup script not found${NC}"
            exit 1
        fi
    fi
fi

# Final instructions
echo ""
echo -e "${GREEN}========================================"
echo "Installation completed successfully!"
echo -e "========================================${NC}"
echo ""

if [ "$LEGACY_MODE" = false ]; then
    echo "Additional setup for experiments:"
    echo "  1. Ensure MuJoCo 2.1.0 is installed at ~/.mujoco/mujoco210"
    echo "  2. Add to your ~/.bashrc:"
    echo "     export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/.mujoco/mujoco210/bin"
    echo "     export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia"
    echo "     export MUJOCO_GL=egl"
    echo "     export XLA_PYTHON_CLIENT_PREALLOCATE=false"
    echo ""
fi

# Show next steps
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Next Steps${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "$WITH_DATASETS" = false ]; then
    echo "1. Download datasets (if not done):"
    echo -e "   ${YELLOW}bash setup/download_dataset.sh${NC}"
    echo ""
fi

if [ "$WITH_REMOTE_ENVS" = false ]; then
    echo "2. Setup remote evaluation environments:"
    echo -e "   ${YELLOW}bash setup/setup_remote_env.sh${NC}"
    echo ""
fi

echo -e "${BLUE}To start an evaluation server (in a separate terminal):${NC}"
echo ""
echo "  # Kitchen"
echo -e "  ${YELLOW}conda activate kitchen_eval${NC}"
echo -e "  ${YELLOW}python remoteEnv/kitchen/kitchen_server.py${NC}  # Port 9999"
echo ""
echo "  # MMWorld"
echo -e "  ${YELLOW}conda activate mmworld_eval${NC}"
echo -e "  ${YELLOW}python remoteEnv/multiStageMetaworld/mmworld_server.py${NC}  # Port 8888"
echo ""
echo "  # LIBERO"
echo -e "  ${YELLOW}conda activate libero${NC}"
echo -e "  ${YELLOW}python remoteEnv/libero/libero_server.py${NC}  # Port 7777"
echo ""

echo -e "${BLUE}To run training:${NC}"
echo ""
echo "  # Kitchen with PTGM"
echo -e "  ${YELLOW}conda activate $ENV_NAME${NC}"
echo -e "  ${YELLOW}python exp/trainer.py --env kitchen --scenario_type kitchenem \\${NC}"
echo -e "  ${YELLOW}    --algorithm ptgm --lifelong s20b4/append4 --seed 0${NC}"
echo ""
echo "  # MMWorld with LazySI"
echo -e "  ${YELLOW}python exp/trainer.py --env mmworld --scenario_type mmworldem \\${NC}"
echo -e "  ${YELLOW}    --algorithm lazysi --lifelong ptgm/s20b4/ptgm/s20b4 --seed 0${NC}"
echo ""

echo "For more examples and detailed documentation:"
echo "  - exp/README.md (comprehensive trainer documentation)"
echo "  - CLAUDE.md (project overview and usage guide)"
echo ""

# Offer to activate the environment
if [ "$SKIP_CONFIRM" = false ]; then
    read -p "Do you want to activate the '$ENV_NAME' environment now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${GREEN}Activating $ENV_NAME environment...${NC}"
        echo -e "${YELLOW}Note: Type 'exit' to return to your previous environment${NC}"
        echo ""

        # Start a new bash shell with the environment activated
        exec bash --init-file <(echo "
            # Source the user's bashrc first
            if [ -f ~/.bashrc ]; then
                source ~/.bashrc
            fi

            # Activate conda environment
            conda activate $ENV_NAME

            # Show environment info
            echo ''
            echo -e '\033[0;32m========================================'
            echo 'Environment: $ENV_NAME activated'
            echo -e '========================================\033[0m'
            echo ''
            echo 'Python version: \$(python --version)'
            echo 'Conda environment: \$(conda info --envs | grep '*' | awk '{print \$1}')'
            echo ''
        ")
    else
        echo ""
        echo "Environment not activated. To activate later, run:"
        echo -e "  ${BLUE}conda activate $ENV_NAME${NC}"
        echo ""
    fi
else
    echo "To activate the environment, run:"
    echo -e "  ${BLUE}conda activate $ENV_NAME${NC}"
    echo ""
fi
