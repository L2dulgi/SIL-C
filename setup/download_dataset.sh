#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo "SILGym Dataset Download Script"
echo -e "========================================${NC}"
echo ""

# Parse arguments
SKIP_CONFIRM=false
DATA_DIR="data"

while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -y, --yes           Skip confirmation prompts"
            echo "  --data-dir DIR      Specify data directory (default: data)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "This script downloads the evolving_datasets.zip from Google Drive"
            echo "and extracts it to the specified data directory."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo -e "${YELLOW}Warning: gdown is not installed${NC}"
    echo "Installing gdown..."
    pip install gdown
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install gdown${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ gdown installed${NC}"
fi

# Check if data directory exists
if [ -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}Warning: Data directory '$DATA_DIR' already exists${NC}"

    # Check if evolving_world or evolving_kitchen already exist
    if [ -d "$DATA_DIR/evolving_world" ] || [ -d "$DATA_DIR/evolving_kitchen" ]; then
        echo -e "${YELLOW}Datasets appear to be already downloaded${NC}"

        if [ "$SKIP_CONFIRM" = false ]; then
            read -p "Do you want to re-download and overwrite? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${BLUE}Skipping download${NC}"
                echo ""
                echo "Existing datasets found in: $DATA_DIR"
                echo "  - evolving_world/"
                echo "  - evolving_kitchen/"
                exit 0
            fi
        else
            echo -e "${YELLOW}Re-downloading datasets (--yes flag)${NC}"
        fi
    fi
else
    echo -e "${BLUE}Creating data directory: $DATA_DIR${NC}"
    mkdir -p "$DATA_DIR"
fi

# Download dataset
echo ""
echo -e "${BLUE}Step 1/3: Downloading evolving_datasets.zip from Google Drive...${NC}"
echo "This may take several minutes depending on your internet connection."
echo ""

DATASET_URL="https://drive.google.com/uc?id=1DbSFIUgt_Ys0l4988VXshE50z7IWL_Kq"
DOWNLOAD_FILE="$DATA_DIR/evolving_datasets.zip"

# Download with progress
gdown "$DATASET_URL" -O "$DOWNLOAD_FILE"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to download dataset${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Download completed${NC}"

# Check file size
FILE_SIZE=$(du -h "$DOWNLOAD_FILE" | cut -f1)
echo "Downloaded file size: $FILE_SIZE"

# Extract dataset
echo ""
echo -e "${BLUE}Step 2/3: Extracting dataset...${NC}"
echo "This may take several minutes."
echo ""

cd "$DATA_DIR"
unzip -q evolving_datasets.zip

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to extract dataset${NC}"
    cd ..
    exit 1
fi

cd ..
echo -e "${GREEN}✓ Extraction completed${NC}"

# Verify extraction
echo ""
echo -e "${BLUE}Step 3/3: Verifying extracted files...${NC}"

EXPECTED_DIRS=("evolving_world" "evolving_kitchen")
MISSING_DIRS=()

for dir in "${EXPECTED_DIRS[@]}"; do
    if [ -d "$DATA_DIR/$dir" ]; then
        echo -e "${GREEN}✓ Found: $DATA_DIR/$dir${NC}"
    else
        echo -e "${YELLOW}⚠ Missing: $DATA_DIR/$dir${NC}"
        MISSING_DIRS+=("$dir")
    fi
done

# Clean up zip file
if [ "$SKIP_CONFIRM" = false ]; then
    echo ""
    read -p "Do you want to remove the downloaded zip file? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$DOWNLOAD_FILE"
        echo -e "${GREEN}✓ Zip file removed${NC}"
    else
        echo -e "${BLUE}Keeping zip file: $DOWNLOAD_FILE${NC}"
    fi
else
    echo ""
    echo -e "${BLUE}Removing zip file...${NC}"
    rm "$DOWNLOAD_FILE"
    echo -e "${GREEN}✓ Zip file removed${NC}"
fi

# Final summary
echo ""
echo -e "${GREEN}========================================"
echo "Dataset Download Complete!"
echo -e "========================================${NC}"
echo ""
echo "Datasets extracted to: $DATA_DIR/"

if [ ${#MISSING_DIRS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All expected directories found${NC}"
    echo ""
    echo "Available datasets:"
    for dir in "${EXPECTED_DIRS[@]}"; do
        echo "  - $dir/"
    done
else
    echo -e "${YELLOW}⚠ Some directories were not found:${NC}"
    for dir in "${MISSING_DIRS[@]}"; do
        echo "  - $dir/"
    done
    echo ""
    echo "This may be expected if the dataset structure has changed."
fi

echo ""
echo "You can now proceed with remote environment setup or training."
echo ""
