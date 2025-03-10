#!/bin/bash
# setup.sh
# Bash setup script for Maggie AI Assistant on Linux

set -e

# Print with colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Create directories
dirs=(
    "logs"
    "models"
    "models/tts"
    "models/tts/en_US-kathleen-medium"
    "recipes"
    "templates"
)

for dir in "${dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}Created directory: $dir${NC}"
    else
        echo -e "${YELLOW}Directory already exists: $dir${NC}"
    fi
done

# Check for Python 3.10
if command -v python3.10 &> /dev/null; then
    echo -e "${GREEN}Found $(python3.10 --version)${NC}"
    PYTHON_CMD=python3.10
elif command -v python3 &> /dev/null && python3 --version | grep -q "Python 3\.10"; then
    echo -e "${GREEN}Found $(python3 --version)${NC}"
    PYTHON_CMD=python3
else
    echo -e "${YELLOW}Python 3.10 not found. Checking for alternatives...${NC}"
    if command -v python3 &> /dev/null; then
        echo -e "${YELLOW}Found $(python3 --version), will use this instead${NC}"
        echo -e "${YELLOW}Note: Python 3.10 is recommended${NC}"
        PYTHON_CMD=python3
    else
        echo -e "${RED}Python 3.x not found. Please install Python 3.10:${NC}"
        echo -e "${RED}sudo apt update && sudo apt install python3.10 python3.10-venv python3.10-dev${NC}"
        exit 1
    fi
fi

# Check for CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}Found CUDA $CUDA_VERSION${NC}"
    if [[ "$CUDA_VERSION" != 11.8* ]]; then
        echo -e "${YELLOW}Warning: CUDA 11.8 is recommended, but found $CUDA_VERSION${NC}"
    fi
else
    echo -e "${YELLOW}CUDA not found. Please install CUDA 11.8 for GPU acceleration:${NC}"
    echo -e "${YELLOW}https://developer.nvidia.com/cuda-11-8-0-download-archive${NC}"
fi

# Check for audio dependencies
echo -e "${CYAN}Checking for audio dependencies...${NC}"
MISSING_DEPS=()

for pkg in portaudio19-dev libsndfile1 ffmpeg; do
    if ! dpkg -s $pkg &> /dev/null; then
        MISSING_DEPS+=($pkg)
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo -e "${YELLOW}Missing audio dependencies: ${MISSING_DEPS[*]}${NC}"
    echo -e "${YELLOW}Installing...${NC}"
    sudo apt update && sudo apt install -y "${MISSING_DEPS[@]}"
else
    echo -e "${GREEN}All audio dependencies are installed${NC}"
fi

# Create virtual environment
echo -e "${CYAN}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
else
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${CYAN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${CYAN}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -e ".[gpu]"

# Create example config
if [ ! -f "config.yaml" ]; then
    if [ -f "config.yaml.example" ]; then
        cp config.yaml.example config.yaml
        echo -e "${GREEN}Created config.yaml from example${NC}"
    else
        echo -e "${YELLOW}Warning: config.yaml.example not found, cannot create config.yaml${NC}"
    fi
else
    echo -e "${YELLOW}config.yaml already exists${NC}"
fi

# Remind about model downloads
echo -e "\n${GREEN}Setup completed!${NC}"
echo -e "\n${CYAN}Reminder: You need to download the following models:${NC}"
echo -e "${CYAN}1. Mistral 7B Instruct GPTQ model: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ${NC}"
echo -e "${CYAN}2. Piper TTS voice model: https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US${NC}"
echo -e "\n${CYAN}And don't forget to add your Picovoice access key in config.yaml!${NC}"
echo -e "${CYAN}Get a free key at: https://console.picovoice.ai/${NC}"

# Ask if user wants to run model downloads
read -p "Would you like to attempt automatic model downloads? (y/n) " download_models
if [ "$download_models" = "y" ]; then
    # Check for git-lfs
    if command -v git-lfs &> /dev/null; then
        echo -e "${GREEN}Found $(git lfs --version)${NC}"
    else
        echo -e "${CYAN}Installing Git LFS...${NC}"
        sudo apt install -y git-lfs
        git lfs install
    fi
    
    # Download Mistral model
    mistral_dir="models/mistral-7b-instruct-v0.3-GPTQ-4bit"
    if [ ! -d "$mistral_dir" ]; then
        echo -e "${CYAN}Downloading Mistral 7B model... (this may take a while)${NC}"
        git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ "$mistral_dir"
    else
        echo -e "${YELLOW}Mistral model directory already exists${NC}"
    fi
    
    # Download Piper TTS voice
    voice_dir="models/tts/en_US-kathleen-medium"
    onnx_file="$voice_dir/en_US-kathleen-medium.onnx"
    json_file="$voice_dir/en_US-kathleen-medium.json"
    
    if [ ! -f "$onnx_file" ]; then
        echo -e "${CYAN}Downloading Piper TTS ONNX model...${NC}"
        onnx_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx"
        wget -O "$onnx_file" "$onnx_url"
    else
        echo -e "${YELLOW}Piper TTS ONNX file already exists${NC}"
    fi
    
    if [ ! -f "$json_file" ]; then
        echo -e "${CYAN}Downloading Piper TTS JSON config...${NC}"
        json_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json"
        wget -O "$json_file" "$json_url"
    else
        echo -e "${YELLOW}Piper TTS JSON file already exists${NC}"
    fi
fi

# Remind about Picovoice key
echo -e "\n${MAGENTA}Don't forget to edit config.yaml with your Picovoice access key!${NC}"
echo -e "${MAGENTA}Get a free key at: https://console.picovoice.ai/${NC}"

# Ask if user wants to run the application
read -p "Would you like to start Maggie now? (y/n) " run_app
if [ "$run_app" = "y" ]; then
    echo -e "${CYAN}Starting Maggie...${NC}"
    python main.py
else
    echo -e "\n${CYAN}To run Maggie later, use:${NC}"
    echo -e "${CYAN}source venv/bin/activate${NC}"
    echo -e "${CYAN}python main.py${NC}"
fi
