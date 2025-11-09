#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "SILGym Remote Environment Setup"
echo -e "========================================${NC}"
echo ""

# Parse arguments
SKIP_CONFIRM=false
SETUP_ALL=false
SETUP_KITCHEN=false
SETUP_MMWORLD=false
SETUP_LIBERO=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        --all)
            SETUP_ALL=true
            shift
            ;;
        --kitchen)
            SETUP_KITCHEN=true
            shift
            ;;
        --mmworld)
            SETUP_MMWORLD=true
            shift
            ;;
        --libero)
            SETUP_LIBERO=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -y, --yes        Skip confirmation prompts"
            echo "  --all            Set up all remote environments"
            echo "  --kitchen        Set up Kitchen evaluation environment"
            echo "  --mmworld        Set up MMWorld evaluation environment"
            echo "  --libero         Set up LIBERO evaluation environment"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "If no environment flag is specified, an interactive menu will be shown."
            echo ""
            echo "Examples:"
            echo "  $0                    # Interactive menu"
            echo "  $0 --kitchen          # Setup Kitchen only"
            echo "  $0 --all -y           # Setup all without prompts"
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

# Interactive menu if no flags specified
if [ "$SETUP_ALL" = false ] && [ "$SETUP_KITCHEN" = false ] && [ "$SETUP_MMWORLD" = false ] && [ "$SETUP_LIBERO" = false ]; then
    echo "Select which remote environment(s) to set up:"
    echo ""
    echo "1) Kitchen (D4RL) - Python 3.8.18"
    echo "2) MMWorld (Multi-Stage Metaworld) - Python 3.10.16"
    echo "3) LIBERO - Python 3.10.16"
    echo "4) All environments"
    echo "5) Cancel"
    echo ""
    read -p "Enter your choice (1-5): " choice

    case $choice in
        1)
            SETUP_KITCHEN=true
            ;;
        2)
            SETUP_MMWORLD=true
            ;;
        3)
            SETUP_LIBERO=true
            ;;
        4)
            SETUP_ALL=true
            ;;
        5)
            echo "Setup cancelled."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
fi

# If --all is specified, enable all
if [ "$SETUP_ALL" = true ]; then
    SETUP_KITCHEN=true
    SETUP_MMWORLD=true
    SETUP_LIBERO=true
fi

# Track what was set up
SETUP_COUNT=0
SETUP_SUMMARY=()

# ============================================================================
# Kitchen Environment Setup
# ============================================================================

if [ "$SETUP_KITCHEN" = true ]; then
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  Setting up Kitchen Environment${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    ENV_NAME="kitchen_eval"
    PYTHON_VERSION="3.8.18"

    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}Warning: Environment '$ENV_NAME' already exists${NC}"

        if [ "$SKIP_CONFIRM" = false ]; then
            read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Removing existing environment...${NC}"
                conda env remove -n $ENV_NAME -y
            else
                echo -e "${BLUE}Skipping Kitchen setup (using existing environment)${NC}"
                SETUP_SUMMARY+=("Kitchen: Skipped (using existing environment)")
                SETUP_KITCHEN=false
            fi
        else
            echo -e "${YELLOW}Removing existing environment (--yes flag)...${NC}"
            conda env remove -n $ENV_NAME -y
        fi
    fi

    if [ "$SETUP_KITCHEN" = true ]; then
        # Step 1: Create environment
        echo -e "${BLUE}[1/3] Creating conda environment '$ENV_NAME' (Python $PYTHON_VERSION)...${NC}"
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
        echo -e "${GREEN}✓ Environment created${NC}"

        # Step 2: Install requirements
        echo ""
        echo -e "${BLUE}[2/3] Installing Kitchen requirements...${NC}"

        REQ_FILE="remoteEnv/kitchen/requirements.txt"
        if [ ! -f "$REQ_FILE" ]; then
            echo -e "${RED}Error: Requirements file not found: $REQ_FILE${NC}"
            exit 1
        fi

        eval "$(conda shell.bash hook)"
        conda activate $ENV_NAME
        pip install -r "$REQ_FILE"
        echo -e "${GREEN}✓ Requirements installed${NC}"

        # Step 3: Verify installation
        echo ""
        echo -e "${BLUE}[3/3] Verifying installation...${NC}"
        python -c "import gym; import d4rl" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Kitchen environment ready${NC}"
        else
            echo -e "${YELLOW}⚠ Warning: Could not verify installation${NC}"
        fi

        conda deactivate

        SETUP_COUNT=$((SETUP_COUNT + 1))
        SETUP_SUMMARY+=("Kitchen: ✓ Successfully set up (kitchen_eval, Python 3.8.18, Port 9999)")
    fi
fi

# ============================================================================
# MMWorld Environment Setup
# ============================================================================

if [ "$SETUP_MMWORLD" = true ]; then
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  Setting up MMWorld Environment${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    ENV_NAME="mmworld_eval"
    PYTHON_VERSION="3.10.16"

    # Check if mmworld package exists
    MMWORLD_DIR="remoteEnv/multiStageMetaworld/mmworld"
    if [ ! -d "$MMWORLD_DIR" ]; then
        echo -e "${YELLOW}MMWorld package not found. Downloading...${NC}"
        echo ""

        # Check if gdown is installed
        if ! command -v gdown &> /dev/null; then
            echo "Installing gdown..."
            pip install gdown
        fi

        # Download mmworld.zip
        echo -e "${BLUE}[1/4] Downloading mmworld.zip from Google Drive...${NC}"
        MMWORLD_URL="https://drive.google.com/uc?id=1BzWk9vbJIaEkklfeA0F2C8ncRJCoPHcz"
        gdown "$MMWORLD_URL" -O mmworld.zip

        if [ $? -ne 0 ]; then
            echo -e "${RED}Error: Failed to download mmworld.zip${NC}"
            exit 1
        fi

        echo -e "${GREEN}✓ Download completed${NC}"

        # Extract
        echo ""
        echo -e "${BLUE}[2/4] Extracting mmworld package...${NC}"
        unzip -q mmworld.zip -d remoteEnv/multiStageMetaworld/

        if [ $? -ne 0 ]; then
            echo -e "${RED}Error: Failed to extract mmworld.zip${NC}"
            exit 1
        fi

        echo -e "${GREEN}✓ Extraction completed${NC}"

        # Clean up
        rm mmworld.zip
        STEP_OFFSET=2
    else
        echo -e "${GREEN}✓ MMWorld package found${NC}"
        STEP_OFFSET=0
    fi

    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}Warning: Environment '$ENV_NAME' already exists${NC}"

        if [ "$SKIP_CONFIRM" = false ]; then
            read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Removing existing environment...${NC}"
                conda env remove -n $ENV_NAME -y
            else
                echo -e "${BLUE}Skipping MMWorld setup (using existing environment)${NC}"
                SETUP_SUMMARY+=("MMWorld: Skipped (using existing environment)")
                SETUP_MMWORLD=false
            fi
        else
            echo -e "${YELLOW}Removing existing environment (--yes flag)...${NC}"
            conda env remove -n $ENV_NAME -y
        fi
    fi

    if [ "$SETUP_MMWORLD" = true ]; then
        # Create environment
        echo ""
        echo -e "${BLUE}[$((STEP_OFFSET + 1))/4] Creating conda environment '$ENV_NAME' (Python $PYTHON_VERSION)...${NC}"
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
        echo -e "${GREEN}✓ Environment created${NC}"

        # Install mmworld dependencies
        echo ""
        echo -e "${BLUE}[$((STEP_OFFSET + 2))/4] Setting up MMWorld package...${NC}"

        eval "$(conda shell.bash hook)"
        conda activate $ENV_NAME

        cd "$MMWORLD_DIR"
        if [ -f "env.sh" ]; then
            bash env.sh
        fi
        pip install -e .
        cd ../../..

        echo -e "${GREEN}✓ MMWorld package installed${NC}"

        # Verify installation
        echo ""
        echo -e "${BLUE}[$((STEP_OFFSET + 3))/4] Verifying installation...${NC}"
        python -c "import mmworld" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ MMWorld environment ready${NC}"
        else
            echo -e "${YELLOW}⚠ Warning: Could not verify installation${NC}"
        fi

        conda deactivate

        SETUP_COUNT=$((SETUP_COUNT + 1))
        SETUP_SUMMARY+=("MMWorld: ✓ Successfully set up (mmworld_eval, Python 3.10.16, Port 8888)")
    fi
fi

# ============================================================================
# LIBERO Environment Setup
# ============================================================================

if [ "$SETUP_LIBERO" = true ]; then
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  Setting up LIBERO Environment${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    ENV_NAME="libero"
    PYTHON_VERSION="3.10.16"

    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        echo -e "${YELLOW}Warning: Environment '$ENV_NAME' already exists${NC}"

        if [ "$SKIP_CONFIRM" = false ]; then
            read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Removing existing environment...${NC}"
                conda env remove -n $ENV_NAME -y
            else
                echo -e "${BLUE}Skipping LIBERO setup (using existing environment)${NC}"
                SETUP_SUMMARY+=("LIBERO: Skipped (using existing environment)")
                SETUP_LIBERO=false
            fi
        else
            echo -e "${YELLOW}Removing existing environment (--yes flag)...${NC}"
            conda env remove -n $ENV_NAME -y
        fi
    fi

    if [ "$SETUP_LIBERO" = true ]; then
        # Step 1: Create environment
        echo -e "${BLUE}[1/3] Creating conda environment '$ENV_NAME' (Python $PYTHON_VERSION)...${NC}"
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
        echo -e "${GREEN}✓ Environment created${NC}"

        # Step 2: Install requirements
        echo ""
        echo -e "${BLUE}[2/3] Installing LIBERO requirements...${NC}"

        REQ_FILE="remoteEnv/libero/requirements.txt"
        if [ ! -f "$REQ_FILE" ]; then
            echo -e "${YELLOW}Warning: Requirements file not found: $REQ_FILE${NC}"
            echo "You may need to install LIBERO dependencies manually."
            echo "See docs/trainer_execution_guide.md for details."
        else
            eval "$(conda shell.bash hook)"
            conda activate $ENV_NAME
            pip install -r "$REQ_FILE"
            echo -e "${GREEN}✓ Requirements installed${NC}"

            # Step 3: Verify installation
            echo ""
            echo -e "${BLUE}[3/3] Verifying installation...${NC}"
            python -c "import libero" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ LIBERO environment ready${NC}"
            else
                echo -e "${YELLOW}⚠ Warning: Could not verify installation${NC}"
                echo "You may need to install additional dependencies."
            fi

            conda deactivate
        fi

        SETUP_COUNT=$((SETUP_COUNT + 1))
        SETUP_SUMMARY+=("LIBERO: ✓ Successfully set up (libero, Python 3.10.16, Port 7777)")
    fi
fi

# ============================================================================
# Final Summary
# ============================================================================

echo ""
echo -e "${GREEN}========================================"
echo "Remote Environment Setup Complete!"
echo -e "========================================${NC}"
echo ""

if [ $SETUP_COUNT -eq 0 ]; then
    echo -e "${YELLOW}No new environments were set up.${NC}"
    echo ""
else
    echo "Summary:"
    for item in "${SETUP_SUMMARY[@]}"; do
        echo "  $item"
    done
    echo ""
fi

# Show server launch commands
echo -e "${BOLD}To start evaluation servers:${NC}"
echo ""

if [ "$SETUP_KITCHEN" = true ] || conda env list | grep -q "^kitchen_eval "; then
    echo -e "${CYAN}Kitchen Server:${NC}"
    echo "  conda activate kitchen_eval"
    echo "  python remoteEnv/kitchen/kitchen_server.py"
    echo "  # Listens on port 9999"
    echo ""
fi

if [ "$SETUP_MMWORLD" = true ] || conda env list | grep -q "^mmworld_eval "; then
    echo -e "${CYAN}MMWorld Server:${NC}"
    echo "  conda activate mmworld_eval"
    echo "  python remoteEnv/multiStageMetaworld/mmworld_server.py"
    echo "  # Listens on port 8888"
    echo ""
fi

if [ "$SETUP_LIBERO" = true ] || conda env list | grep -q "^libero "; then
    echo -e "${CYAN}LIBERO Server:${NC}"
    echo "  conda activate libero"
    echo "  python remoteEnv/libero/libero_server.py"
    echo "  # Listens on port 7777"
    echo ""
fi

# Show training examples
echo -e "${BOLD}Example training commands:${NC}"
echo ""

echo -e "${CYAN}Kitchen with PTGM:${NC}"
echo "  conda activate silgym12  # or silgym for legacy"
echo "  python exp/trainer.py \\"
echo "    --env kitchen \\"
echo "    --scenario_type kitchenem \\"
echo "    --sync_type sync \\"
echo "    --algorithm ptgm \\"
echo "    --lifelong s20b4/append4 \\"
echo "    --dec ddpm \\"
echo "    --seed 0"
echo ""

echo -e "${CYAN}MMWorld with LazySI:${NC}"
echo "  conda activate silgym12  # or silgym for legacy"
echo "  python exp/trainer.py \\"
echo "    --env mmworld \\"
echo "    --scenario_type mmworldem \\"
echo "    --sync_type sync \\"
echo "    --algorithm lazysi \\"
echo "    --lifelong ptgm/s20b4/ptgm/s20b4 \\"
echo "    --dec ddpm \\"
echo "    --dist_type maha \\"
echo "    --seed 0"
echo ""

echo -e "${CYAN}LIBERO with SILC:${NC}"
echo "  conda activate silgym12  # or silgym for legacy"
echo "  python exp/trainer.py \\"
echo "    --env libero-l \\"
echo "    --scenario_type goal \\"
echo "    --sync_type sync \\"
echo "    --algorithm silc \\"
echo "    --lifelong ptgm/s20b4/ptgm/s20b4 \\"
echo "    --dec ddpm \\"
echo "    --action_chunk 4 \\"
echo "    --seed 0"
echo ""

echo -e "${YELLOW}Note: Always start the evaluation server BEFORE running training!${NC}"
echo ""
echo "For more information, see:"
echo "  - exp/README.md (detailed trainer documentation)"
echo "  - CLAUDE.md (project overview)"
echo "  - docs/trainer_execution_guide.md (advanced configuration)"
echo ""
