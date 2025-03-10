#!/bin/bash
# =====================================================
# Maggie AI Assistant - Linux Setup Script
# =====================================================
# This script automates the installation of Maggie AI Assistant 
# on Linux systems, optimized for AMD Ryzen 9 5900X and RTX 3080
# =====================================================

set -e

# Print with colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${CYAN}Maggie AI Assistant - Linux Setup${NC}"
echo -e "${CYAN}===================================${NC}"

# Check if running as root (for system optimizations)
IS_ROOT=0
if [ "$(id -u)" -eq 0 ]; then
    IS_ROOT=1
    echo -e "${GREEN}Running with root privileges - will apply system optimizations${NC}"
else
    echo -e "${YELLOW}Running without root privileges - some optimizations will be skipped${NC}"
    echo -e "${YELLOW}Run with sudo for full system optimization${NC}"
fi

# Create directories
echo -e "\n${CYAN}Creating required directories...${NC}"
dirs=(
    "logs"
    "models"
    "models/tts"
    "models/tts/en_US-kathleen-medium"
    "recipes"
    "templates"
    "cache"
    "cache/tts"
)

for dir in "${dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}Created directory: $dir${NC}"
    else
        echo -e "${YELLOW}Directory already exists: $dir${NC}"
    fi
done

# Check for Python 3.10 specifically
echo -e "\n${CYAN}Checking Python version...${NC}"
if command -v python3.10 &> /dev/null; then
    PYTHON_VERSION=$(python3.10 --version)
    if [[ $PYTHON_VERSION == *"3.10."* ]]; then
        echo -e "${GREEN}Found $PYTHON_VERSION - Compatible version${NC}"
        PYTHON_CMD=python3.10
    else
        echo -e "${RED}Unexpected version output: $PYTHON_VERSION${NC}"
        exit 1
    fi
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    if [[ $PYTHON_VERSION == *"3.10."* ]]; then
        echo -e "${GREEN}Found $PYTHON_VERSION - Compatible version${NC}"
        PYTHON_CMD=python3
    else
        echo -e "${RED}Incompatible Python version: $PYTHON_VERSION${NC}"
        echo -e "${RED}Maggie requires Python 3.10.x specifically. Other versions will not work.${NC}"
        echo -e "${RED}Please install Python 3.10 and try again:${NC}"
        echo -e "${YELLOW}sudo apt update && sudo apt install python3.10 python3.10-venv python3.10-dev${NC}"
        exit 1
    fi
else
    echo -e "${RED}Python 3.10 not found. Please install Python 3.10:${NC}"
    echo -e "${YELLOW}sudo apt update && sudo apt install python3.10 python3.10-venv python3.10-dev${NC}"
    exit 1
fi

# Check for CUDA
echo -e "\n${CYAN}Checking NVIDIA GPU and CUDA support...${NC}"
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}Found CUDA $CUDA_VERSION${NC}"
    
    # Check for RTX 3080
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader)
        echo -e "${GREEN}Detected GPU: $GPU_INFO${NC}"
        
        if [[ "$GPU_INFO" == *"3080"* ]]; then
            echo -e "${GREEN}RTX 3080 detected - Using optimal configuration${NC}"
        fi
    fi
    
    if [[ "$CUDA_VERSION" != 11.8* ]]; then
        echo -e "${YELLOW}Warning: CUDA 11.8 is recommended, but found $CUDA_VERSION${NC}"
    fi
else
    echo -e "${YELLOW}CUDA not found. GPU acceleration will be disabled.${NC}"
    echo -e "${YELLOW}For optimal performance with RTX 3080, install CUDA 11.8:${NC}"
    echo -e "${YELLOW}https://developer.nvidia.com/cuda-11-8-0-download-archive${NC}"
fi

# Check for audio dependencies
echo -e "\n${CYAN}Checking for audio dependencies...${NC}"
MISSING_DEPS=()

for pkg in portaudio19-dev libsndfile1 ffmpeg; do
    if ! dpkg -s $pkg &> /dev/null; then
        MISSING_DEPS+=($pkg)
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo -e "${YELLOW}Missing audio dependencies: ${MISSING_DEPS[*]}${NC}"
    
    if [ "$IS_ROOT" -eq 1 ]; then
        echo -e "${CYAN}Installing missing dependencies...${NC}"
        apt update && apt install -y "${MISSING_DEPS[@]}"
    else
        echo -e "${YELLOW}Please install missing dependencies:${NC}"
        echo -e "${YELLOW}sudo apt update && sudo apt install -y ${MISSING_DEPS[*]}${NC}"
        
        read -p "Attempt to install dependencies now? (y/n) " install_deps
        if [ "$install_deps" = "y" ]; then
            echo "Running with sudo:"
            sudo apt update && sudo apt install -y "${MISSING_DEPS[@]}"
        fi
    fi
else
    echo -e "${GREEN}All audio dependencies are installed${NC}"
fi

# Optimize system for Ryzen 9 5900X if running as root
if [ "$IS_ROOT" -eq 1 ]; then
    echo -e "\n${CYAN}Applying system optimizations for Ryzen 9 5900X...${NC}"
    
    # Check if CPU is Ryzen 9
    if grep -q "Ryzen 9" /proc/cpuinfo; then
        echo -e "${GREEN}Detected Ryzen 9 processor${NC}"
        
        # Set CPU governor to performance
        if [ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]; then
            echo -e "${CYAN}Setting CPU governor to performance mode...${NC}"
            echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
            echo -e "${GREEN}CPU governor set to performance${NC}"
        fi
        
        # Set process scheduling priority
        echo -e "${CYAN}Setting process scheduling priority...${NC}"
        if command -v chrt &> /dev/null; then
            echo -e "${GREEN}Setting high scheduling priority for audio${NC}"
            # We'll set this later when running the actual app
        fi
    else
        echo -e "${YELLOW}Ryzen 9 processor not detected - skipping specific optimizations${NC}"
    fi
else
    echo -e "\n${YELLOW}Skipping system optimizations (requires root)${NC}"
fi

# Create virtual environment
echo -e "\n${CYAN}Creating Python virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
else
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${CYAN}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}Virtual environment activated${NC}"

# Upgrade pip and setuptools
echo -e "\n${CYAN}Upgrading pip and setuptools...${NC}"
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo -e "\n${CYAN}Installing PyTorch with CUDA support (optimized for RTX 3080)...${NC}"
pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install dependencies
echo -e "\n${CYAN}Installing Maggie dependencies...${NC}"
pip install -e .

# Install GPU dependencies
echo -e "\n${CYAN}Installing GPU-specific dependencies...${NC}"
pip install -e ".[gpu]"

# Create config file from example
echo -e "\n${CYAN}Setting up configuration...${NC}"
if [ ! -f "config.yaml" ]; then
    if [ -f "config.yaml.example" ]; then
        cp config.yaml.example config.yaml
        echo -e "${GREEN}Created config.yaml from example${NC}"
        echo -e "${YELLOW}[IMPORTANT] You need to edit config.yaml to add your Picovoice access key${NC}"
    else
        echo -e "${YELLOW}Warning: config.yaml.example not found, cannot create config.yaml${NC}"
    fi
else
    echo -e "${YELLOW}config.yaml already exists${NC}"
fi

# Model download section
echo -e "\n${MAGENTA}==================================================${NC}"
echo -e "${MAGENTA}Would you like to download required models?${NC}"
echo -e "${MAGENTA}This includes Mistral 7B and TTS voice models${NC}"
echo -e "${MAGENTA}(May take significant time and bandwidth)${NC}"
echo -e "${MAGENTA}==================================================${NC}"
read -p "Download models? (y/n) " download_models

if [ "$download_models" = "y" ]; then
    # Check for git-lfs
    echo -e "\n${CYAN}Checking for Git LFS...${NC}"
    if command -v git-lfs &> /dev/null; then
        echo -e "${GREEN}Found $(git-lfs --version)${NC}"
    else
        echo -e "${CYAN}Installing Git LFS...${NC}"
        if [ "$IS_ROOT" -eq 1 ]; then
            apt install -y git-lfs
        else
            sudo apt install -y git-lfs
        fi
        git lfs install
    fi
    
    # Download Mistral model
    echo -e "\n${CYAN}Checking Mistral LLM model...${NC}"
    mistral_dir="models/mistral-7b-instruct-v0.3-GPTQ-4bit"
    if [ ! -d "$mistral_dir" ]; then
        echo -e "${CYAN}Downloading Mistral 7B model... (this may take a while)${NC}"
        git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ "$mistral_dir"
    else
        echo -e "${YELLOW}Mistral model directory already exists${NC}"
    fi
    
    # Download Piper TTS voice
    echo -e "\n${CYAN}Checking TTS voice model...${NC}"
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

# Create recipe template
echo -e "\n${CYAN}Creating recipe template...${NC}"
python main.py --create-template

# Verify system
echo -e "\n${CYAN}Verifying system configuration...${NC}"
python main.py --verify

# Optimize configuration for Ryzen 9 5900X and RTX 3080
echo -e "\n${CYAN}Optimizing system configuration for Ryzen 9 5900X and RTX 3080...${NC}"
python main.py --optimize

# Remind about Picovoice key
echo -e "\n${MAGENTA}==================================================${NC}"
echo -e "${MAGENTA}IMPORTANT: You need a Picovoice access key for wake word detection.${NC}"
echo -e "${MAGENTA}${NC}"
echo -e "${MAGENTA}1. Visit https://console.picovoice.ai/ to create a free account${NC}"
echo -e "${MAGENTA}2. Get your access key from the console${NC}"
echo -e "${MAGENTA}3. Edit config.yaml and add your key in the wake_word.porcupine_access_key field${NC}"
echo -e "${MAGENTA}==================================================${NC}"

# Create startup script
echo -e "\n${CYAN}Creating startup script...${NC}"
cat > start_maggie.sh << 'EOF'
#!/bin/bash
# Maggie AI Assistant startup script

# Activate virtual environment
source venv/bin/activate

# Set high priority for better audio performance
if command -v chrt &> /dev/null; then
    exec chrt --rr 10 python main.py "$@"
else
    exec python main.py "$@"
fi
EOF

chmod +x start_maggie.sh
echo -e "${GREEN}Created startup script: start_maggie.sh${NC}"

# Ask if user wants to run the application
echo -e "\n${CYAN}Setup completed successfully!${NC}"
read -p "Would you like to start Maggie now? (y/n) " run_app
if [ "$run_app" = "y" ]; then
    echo -e "\n${CYAN}Starting Maggie...${NC}"
    ./start_maggie.sh
else
    echo -e "\n${CYAN}To run Maggie later, use:${NC}"
    echo -e "${GREEN}./start_maggie.sh${NC}"
    echo -e "\n${CYAN}Or manually:${NC}"
    echo -e "${GREEN}source venv/bin/activate${NC}"
    echo -e "${GREEN}python main.py${NC}"
fi

echo -e "\n${GREEN}Thank you for installing Maggie AI Assistant!${NC}"