#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "MuJoCo 2.1.0 Installation Script"
echo -e "========================================${NC}"
echo ""

# Parse command line arguments
SKIP_CONFIRM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -y, --yes    Skip confirmation prompts"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if MuJoCo is already installed
MUJOCO_PATH="$HOME/.mujoco/mujoco210"
if [ -d "$MUJOCO_PATH" ]; then
    echo -e "${YELLOW}Warning: MuJoCo already installed at $MUJOCO_PATH${NC}"
    echo -e "${GREEN}Skipping installation.${NC}"
    exit 0
fi

# Warn about sudo requirements
echo -e "${YELLOW}This script requires sudo privileges to install system dependencies.${NC}"
echo ""
echo "The following operations will be performed:"
echo "  1. Install system dependencies via apt-get (requires sudo)"
echo "  2. Download MuJoCo 2.1.0 to ~/.mujoco/mujoco210"
echo "  3. Add environment variables to ~/.bashrc"
echo ""

if [ "$SKIP_CONFIRM" = false ]; then
    read -p "Do you want to proceed? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Installation cancelled.${NC}"
        exit 0
    fi
fi

# Step 1: Install system dependencies
echo ""
echo -e "${BLUE}Step 1: Installing system dependencies...${NC}"

sudo apt-get update
sudo apt-get install -y \
    wget tar \
    libgl1-mesa-dev \
    libglew-dev \
    libglfw3 libglfw3-dev \
    patchelf

sudo apt-get install -y xvfb libosmesa6-dev

echo -e "${GREEN}System dependencies installed${NC}"

# Step 2: Download and install MuJoCo
echo ""
echo -e "${BLUE}Step 2: Downloading MuJoCo 2.1.0...${NC}"

mkdir -p ~/.mujoco
cd ~/.mujoco

wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz

echo -e "${GREEN}MuJoCo 2.1.0 installed at ~/.mujoco/mujoco210${NC}"

# Step 3: Add environment variables to ~/.bashrc
echo ""
echo -e "${BLUE}Step 3: Configuring environment variables...${NC}"

# Check if variables already exist in .bashrc
if ! grep -q "MUJOCO_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# MuJoCo environment variables (added by install_mujoco.sh)" >> ~/.bashrc
    echo 'export MUJOCO_HOME=$HOME/.mujoco/mujoco210' >> ~/.bashrc
    echo 'export PATH=$PATH:$MUJOCO_HOME/bin' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_HOME/bin' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
    echo 'export MUJOCO_GL=egl' >> ~/.bashrc
    echo -e "${GREEN}Environment variables added to ~/.bashrc${NC}"
else
    echo -e "${YELLOW}Environment variables already exist in ~/.bashrc${NC}"
fi

# Final message
echo ""
echo -e "${GREEN}========================================"
echo "MuJoCo 2.1.0 installation completed!"
echo -e "========================================${NC}"
echo ""
echo "Please run the following command to apply changes:"
echo -e "  ${YELLOW}source ~/.bashrc${NC}"
echo ""
