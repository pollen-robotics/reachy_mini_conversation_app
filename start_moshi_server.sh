#!/bin/bash

# Start Moshi Server for PersonaPlex
# This script helps launch the Moshi server with appropriate settings

set -e

# Default values
PORT=8998
DEVICE="auto"
PERSONAPLEX_DIR="$HOME/personaplex"
CPU_OFFLOAD=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --personaplex-dir)
            PERSONAPLEX_DIR="$2"
            shift 2
            ;;
        --cpu-offload)
            CPU_OFFLOAD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT              WebSocket port (default: 8998)"
            echo "  --device DEVICE          Device: mps, cuda, cpu, or auto (default: auto)"
            echo "  --personaplex-dir DIR    PersonaPlex installation directory (default: ~/personaplex)"
            echo "  --cpu-offload            Enable CPU offload for models (requires accelerate package)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Auto-detect device if set to auto
if [ "$DEVICE" = "auto" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use MPS
        DEVICE="mps"
        echo -e "${YELLOW}Detected macOS - using MPS device${NC}"
    elif command -v nvidia-smi &> /dev/null; then
        # Linux with NVIDIA GPU
        DEVICE="cuda"
        echo -e "${GREEN}Detected NVIDIA GPU - using CUDA device${NC}"
    else
        # Fallback to CPU
        DEVICE="cpu"
        echo -e "${YELLOW}No GPU detected - using CPU device (slow)${NC}"
    fi
fi

# Check if PersonaPlex directory exists
if [ ! -d "$PERSONAPLEX_DIR" ]; then
    echo -e "${RED}Error: PersonaPlex directory not found at: $PERSONAPLEX_DIR${NC}"
    echo -e "${YELLOW}Please clone PersonaPlex first:${NC}"
    echo "  git clone https://github.com/NVIDIA/personaplex.git $PERSONAPLEX_DIR"
    echo "  cd $PERSONAPLEX_DIR"
    echo "  uv venv"
    echo "  uv pip install moshi/."
    exit 1
fi

# Check if moshi subdirectory exists
if [ ! -d "$PERSONAPLEX_DIR/moshi" ]; then
    echo -e "${RED}Error: moshi directory not found in PersonaPlex${NC}"
    echo -e "${YELLOW}Make sure you cloned the full PersonaPlex repository${NC}"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo -e "${YELLOW}Please install uv first:${NC}"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$PERSONAPLEX_DIR/.venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    cd "$PERSONAPLEX_DIR"
    uv venv
    echo -e "${GREEN}Virtual environment created${NC}"
    echo -e "${YELLOW}Installing PersonaPlex/Moshi...${NC}"
    uv pip install "$PERSONAPLEX_DIR/moshi/."

    # Apply macOS compatibility fix
    echo -e "${YELLOW}Applying macOS compatibility fix...${NC}"
    if [ -f "$PERSONAPLEX_DIR/moshi/moshi/models/lm.py" ]; then
        sed -i.bak 's/state = torch\.load(path)$/state = torch.load(path, map_location=self.lm_model.device)/' "$PERSONAPLEX_DIR/moshi/moshi/models/lm.py"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Fix applied successfully${NC}"
        else
            echo -e "${YELLOW}Warning: Could not apply fix automatically. You may need to edit moshi/moshi/models/lm.py manually.${NC}"
        fi
    fi

    # Offer to install accelerate for CPU offload support
    if [ "$CPU_OFFLOAD" = true ]; then
        echo -e "${YELLOW}Installing accelerate package for CPU offload...${NC}"
        uv pip install accelerate
    fi
else
    source "$PERSONAPLEX_DIR/.venv/bin/activate"

    # Check if accelerate is needed but not installed
    if [ "$CPU_OFFLOAD" = true ]; then
        if ! python -c "import accelerate" 2>/dev/null; then
            echo -e "${YELLOW}accelerate package not found. Installing for CPU offload...${NC}"
            uv pip install accelerate
        fi
    fi
fi

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set${NC}"
    echo -e "${YELLOW}You need a HuggingFace token to download the PersonaPlex model${NC}"
    echo -e "${YELLOW}1. Accept the license at: https://huggingface.co/nvidia/personaplex-7b-v1${NC}"
    echo -e "${YELLOW}2. Get your token at: https://huggingface.co/settings/tokens${NC}"
    echo -e "${YELLOW}3. Set it: export HF_TOKEN=your_token${NC}"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set environment variables
if [ "$DEVICE" = "mps" ]; then
    echo -e "${YELLOW}Setting PYTORCH_ENABLE_MPS_FALLBACK=1 for macOS${NC}"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

# Create temporary SSL directory
SSL_DIR=$(mktemp -d)
echo -e "${GREEN}Created SSL directory: $SSL_DIR${NC}"

# Print configuration
echo ""
echo -e "${GREEN}=== Moshi Server Configuration ===${NC}"
echo "  Device: $DEVICE"
echo "  Port: $PORT"
echo "  PersonaPlex Dir: $PERSONAPLEX_DIR"
echo "  SSL Dir: $SSL_DIR"
echo "  CPU Offload: $CPU_OFFLOAD"
echo ""
echo -e "${YELLOW}Starting Moshi server...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping Moshi server...${NC}"
    rm -rf "$SSL_DIR"
    echo -e "${GREEN}Cleaned up SSL directory${NC}"
    exit 0
}

trap cleanup INT TERM

# Start the Moshi server
cd "$PERSONAPLEX_DIR"

# Build command with optional flags
CMD="python -m moshi.server --ssl \"$SSL_DIR\" --device \"$DEVICE\" --port \"$PORT\""
if [ "$CPU_OFFLOAD" = true ]; then
    CMD="$CMD --cpu-offload"
fi

# Execute the command
eval $CMD
