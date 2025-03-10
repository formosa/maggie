# Maggie - Installation Guide

## Project Overview

Maggie is a voice-activated AI assistant implementing a Finite State Machine (FSM) architecture with event-driven state transitions and modular utility objects. The application leverages local language models for processing and understanding spoken commands, with specific optimizations for AMD Ryzen 9 5900X CPU and NVIDIA GeForce RTX 3080 GPU hardware.

**Key Features:**
- Wake word detection ("Maggie") with minimal CPU usage
- Local speech recognition and text-to-speech capabilities
- GPU-accelerated language model inference
- Recipe creation utility with speech-to-document processing
- Multi-threaded architecture optimized for modern multicore CPUs

---

## Table of Contents

1. [Windows Installation Guide](#windows-installation-guide)
   - [System Requirements (Windows)](#system-requirements-windows)
   - [Quick Installation (Windows)](#quick-installation-windows)
   - [Detailed Installation Steps (Windows)](#detailed-installation-steps-windows)
   - [Optimizations for Ryzen 9 5900X and RTX 3080 (Windows)](#optimizations-for-ryzen-9-5900x-and-rtx-3080-windows)
   - [Startup and Configuration (Windows)](#startup-and-configuration-windows)
   - [Troubleshooting (Windows)](#troubleshooting-windows)
   - [Performance Verification (Windows)](#performance-verification-windows)

2. [Linux Installation Guide](#linux-installation-guide)
   - [System Requirements (Linux)](#system-requirements-linux)
   - [Quick Installation (Linux)](#quick-installation-linux)
   - [Detailed Installation Steps (Linux)](#detailed-installation-steps-linux)
   - [Optimizations for Ryzen 9 5900X and RTX 3080 (Linux)](#optimizations-for-ryzen-9-5900x-and-rtx-3080-linux)
   - [Startup and Configuration (Linux)](#startup-and-configuration-linux)
   - [Troubleshooting (Linux)](#troubleshooting-linux)
   - [Performance Verification (Linux)](#performance-verification-linux)

---

# Windows Installation Guide

## System Requirements (Windows)

### Hardware Requirements
- **CPU:** AMD Ryzen 9 5900X or equivalent 12-core processor
- **GPU:** NVIDIA GeForce RTX 3080 with 10GB VRAM or equivalent
- **RAM:** 32GB DDR4-3200 or faster
- **Storage:** At least 15GB of free disk space

### Software Requirements
- **OS:** Windows 11 Pro (64-bit)
- **Python:** Exactly version 3.10.x (Python 3.11+ is NOT compatible)
- **CUDA:** CUDA 11.8 and cuDNN
- **Git:** Git with LFS support for downloading models

## Quick Installation (Windows)

1. **Clone the repository**
   ```powershell
   git clone https://github.com/formosa/maggie.git
   cd maggie
   ```

2. **Run the setup script with administrator privileges**
   - Right-click on `setup_windows.bat`
   - Select "Run as administrator"
   - Follow the on-screen instructions

3. **Obtain Picovoice access key**
   - Visit [https://console.picovoice.ai/](https://console.picovoice.ai/) to register
   - Get your free access key
   - Edit `config.yaml` to add your key in the `wake_word.porcupine_access_key` field

## Detailed Installation Steps (Windows)

### 1. Install Python 3.10.x
1. Download Python 3.10.11 installer from [python.org](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)
2. Run the installer with administrator privileges
3. **Important:** Check "Add Python to PATH" during installation
4. Verify installation by running `python --version` in Command Prompt

### 2. Install CUDA Toolkit 11.8
1. Download [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) for Windows
2. Run the installer and follow the instructions
3. Verify installation by running `nvcc --version` in Command Prompt

### 3. Manual Installation Steps

If you prefer not to use the automated setup script, follow these steps:

#### 3.1 Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### 3.2 Install dependencies

```powershell
# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support
pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install Maggie and its dependencies
pip install -e .
pip install -e ".[gpu]"
```

#### 3.3 Download required models

**Install Git LFS:**
```powershell
git lfs install
```

**Download Mistral 7B Instruct GPTQ model:**
```powershell
git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ models/mistral-7b-instruct-v0.3-GPTQ-4bit
```

**Download Piper TTS voice model:**
```powershell
# Create directory for voice model
mkdir -p models/tts/en_US-kathleen-medium

# Download ONNX model and JSON config
Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx" -OutFile "models/tts/en_US-kathleen-medium/en_US-kathleen-medium.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json" -OutFile "models/tts/en_US-kathleen-medium/en_US-kathleen-medium.json"
```

#### 3.4 Configure Maggie

1. Copy the example configuration file:
   ```powershell
   Copy-Item config.yaml.example config.yaml
   ```

2. Edit `config.yaml` to add your Picovoice access key:
   ```yaml
   wake_word:
     sensitivity: 0.5
     keyword_path: null
     porcupine_access_key: "YOUR_KEY_HERE"  # Replace with your key
   ```

#### 3.5 Create recipe template and verify system

```powershell
# Create recipe template
python main.py --create-template

# Verify system configuration
python main.py --verify

# Optimize for your hardware
python main.py --optimize
```

## Optimizations for Ryzen 9 5900X and RTX 3080 (Windows)

### CPU Optimizations
- Thread allocation optimized for 12-core Ryzen 9 processor
  - Uses 8 worker threads by default (optimal balance of performance and responsiveness)
  - Configurable in `config.yaml` under `threading.max_workers`
- Process priority adjustments for improved responsiveness
  - Wake word detection runs at below-normal priority to minimize impact
  - Audio processing runs at above-normal priority for real-time performance

### GPU Optimizations
- CUDA 11.8 configuration tuned for RTX 3080's 10GB VRAM
  - Uses 32 GPU layers for LLM inference (optimal for 10GB VRAM)
  - Automatic adjustment if memory pressure is detected
- Tensor Core utilization with optimized precision settings
  - Uses float16 precision for maximum Tensor Core performance
  - Optimized kernel selection for Ampere architecture
- Model quantization optimized for RTX 3080
  - 4-bit quantization for LLM to maximize VRAM efficiency
  - ONNX runtime acceleration for TTS model

### Memory Optimizations
- Configured for optimal use of 32GB RAM
  - Uses up to 75% of system memory (24GB)
  - Reserves memory for OS and other applications
- Dynamic model loading/unloading based on memory pressure
  - Models are unloaded when memory usage exceeds 85%
  - Automatic garbage collection tuning
- Cache settings for TTS and other components
  - Audio cache to avoid redundant processing
  - Inference result caching for common queries

## Startup and Configuration (Windows)

### Starting Maggie

```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Start Maggie
python main.py
```

### Advanced Configuration Options

Edit `config.yaml` to customize Maggie's behavior:

```yaml
# Wake word sensitivity (0.0-1.0)
wake_word:
  sensitivity: 0.5

# Speech recognition options
speech:
  whisper:
    model_size: "base"  # Options: tiny, base, small, medium
    compute_type: "float16"  # Optimized for RTX 3080

# LLM configuration
llm:
  gpu_layers: 32  # Optimized for RTX 3080 (10GB VRAM)
  precision: "float16"  # Best for RTX 3080 Tensor Cores

# Threading configuration for Ryzen 9 5900X
threading:
  max_workers: 8  # Uses 8 of 12 cores for optimal balance
```

## Troubleshooting (Windows)

### Python Version Issues
- **Problem:** Incorrect Python version
- **Check:** Run `python --version` in Command Prompt
- **Solution:** Install Python 3.10.x from [python.org](https://www.python.org/downloads/release/python-31011/)
- **Verification:** After installation, restart Command Prompt and check version again

### CUDA Installation Issues
- **Problem:** CUDA not detected or incorrect version
- **Check:** Run `nvcc --version` and `python -c "import torch; print(torch.cuda.is_available())"`
- **Solution:** 
  1. Ensure NVIDIA drivers are up to date
  2. Install CUDA 11.8 from [NVIDIA website](https://developer.nvidia.com/cuda-11-8-0-download-archive)
  3. Add CUDA bin directory to PATH

### PyAudio Installation Errors
- **Problem:** PyAudio fails to install with pip
- **Solution:**
  ```powershell
  pip install pipwin
  pipwin install pyaudio
  ```

### Model Loading Failures
- **Problem:** "Model not found" or similar errors
- **Check:** Verify model directories exist and contain the required files
- **Solution:**
  1. Ensure Git LFS is installed (`git lfs install`)
  2. Re-download models using the commands in the installation steps
  3. Check disk space (need at least 15GB free)

## Performance Verification (Windows)

### Check GPU Acceleration

Run the following in Python:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### Verify System Optimization

```powershell
python main.py --verify
```

### Monitor Resource Usage
- **CPU Utilization:** 
  - Open Task Manager (Ctrl+Shift+Esc)
  - Go to "Performance" tab
  - Check CPU usage while Maggie is running
  - Should see multiple cores active but not all at 100%

- **GPU Utilization:**
  - Open Task Manager
  - Go to "Performance" tab > "GPU" 
  - Monitor memory usage during model loading and inference
  - Should stay under 8GB for normal operation

- **Memory Usage:**
  - Should remain under 24GB (75% of 32GB)
  - Check in Task Manager > "Performance" > "Memory"

---

# Linux Installation Guide

## System Requirements (Linux)

### Hardware Requirements
- **CPU:** AMD Ryzen 9 5900X or equivalent 12-core processor
- **GPU:** NVIDIA GeForce RTX 3080 with 10GB VRAM or equivalent
- **RAM:** 32GB DDR4-3200 or faster
- **Storage:** At least 15GB of free disk space

### Software Requirements
- **OS:** Ubuntu 22.04+ or other Linux distribution (64-bit)
- **Python:** Exactly version 3.10.x (Python 3.11+ is NOT compatible)
- **CUDA:** CUDA 11.8 and cuDNN
- **Git:** Git with LFS support for downloading models
- **Audio libraries:** PortAudio, libsndfile, and FFmpeg

## Quick Installation (Linux)

1. **Clone the repository**
   ```bash
   git clone https://github.com/formosa/maggie.git
   cd maggie
   ```

2. **Make the setup script executable and run it**
   ```bash
   chmod +x setup_linux.sh
   sudo ./setup_linux.sh  # Run with sudo for full system optimization
   ```

3. **Obtain Picovoice access key**
   - Visit [https://console.picovoice.ai/](https://console.picovoice.ai/) to register
   - Get your free access key
   - Edit `config.yaml` to add your key in the `wake_word.porcupine_access_key` field

## Detailed Installation Steps (Linux)

### 1. Install Python 3.10.x
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
python3.10 --version  # Verify installation
```

### 2. Install CUDA Toolkit 11.8
```bash
# Download CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
# Install CUDA Toolkit
sudo sh cuda_11.8.0_520.61.05_linux.run
# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
# Verify installation
nvcc --version
```

### 3. Install Audio Dependencies
```bash
sudo apt install portaudio19-dev libsndfile1 ffmpeg
```

### 4. Manual Installation Steps

If you prefer not to use the automated setup script, follow these steps:

#### 4.1 Create and activate a virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

#### 4.2 Install dependencies

```bash
# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support
pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install Maggie and its dependencies
pip install -e .
pip install -e ".[gpu]"
```

#### 4.3 Download required models

**Install Git LFS:**
```bash
sudo apt install git-lfs
git lfs install
```

**Download Mistral 7B Instruct GPTQ model:**
```bash
git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ models/mistral-7b-instruct-v0.3-GPTQ-4bit
```

**Download Piper TTS voice model:**
```bash
# Create directory for voice model
mkdir -p models/tts/en_US-kathleen-medium

# Download ONNX model and JSON config
wget -P models/tts/en_US-kathleen-medium https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx
wget -P models/tts/en_US-kathleen-medium https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json
```

#### 4.4 Configure Maggie

1. Copy the example configuration file:
   ```bash
   cp config.yaml.example config.yaml
   ```

2. Edit `config.yaml` to add your Picovoice access key:
   ```yaml
   wake_word:
     sensitivity: 0.5
     keyword_path: null
     porcupine_access_key: "YOUR_KEY_HERE"  # Replace with your key
   ```

#### 4.5 Create recipe template and verify system

```bash
# Create recipe template
python main.py --create-template

# Verify system configuration
python main.py --verify

# Optimize for your hardware
python main.py --optimize
```

## Optimizations for Ryzen 9 5900X and RTX 3080 (Linux)

### CPU Optimizations
- CPU governor configuration for maximum performance
  - Sets governor to "performance" mode when running with root privileges
  - Prevents frequency scaling for consistent response times
- Process priority and scheduling optimizations
  - Real-time scheduling priority for audio processing
  - Uses `chrt` command to set priorities when available
- Thread allocation optimized for 12-core Ryzen 9 processor
  - Uses 8 worker threads by default (optimal balance for 12-core CPU)
  - Configurable in `config.yaml` under `threading.max_workers`

### GPU Optimizations
- CUDA 11.8 configuration tuned for RTX 3080's 10GB VRAM
  - Uses 32 GPU layers for LLM inference (optimal for 10GB VRAM)
  - Automatic adjustment if memory pressure is detected
- Tensor Core utilization with optimized precision settings
  - Uses float16 precision for maximum Tensor Core performance
  - Optimized kernel selection for Ampere architecture
- NVIDIA X server settings optimization (when running with GUI)
  - Power management mode set to "Prefer Maximum Performance"
  - Optimized compositor pipeline for reduced latency

### Memory Optimizations
- Configured for optimal use of 32GB RAM
  - Uses up to 75% of system memory (24GB)
  - Reserves memory for OS and other applications
- Dynamic model loading/unloading based on memory pressure
  - Models are unloaded when memory usage exceeds 85%
  - Automatic garbage collection tuning
- Cache settings for TTS and other components
  - Audio cache to avoid redundant processing
  - Inference result caching for common queries

## Startup and Configuration (Linux)

### Starting Maggie

```bash
# Using the startup script (recommended)
./start_maggie.sh

# Or manually
source venv/bin/activate
python main.py
```

### Advanced Configuration Options

Edit `config.yaml` to customize Maggie's behavior:

```yaml
# Wake word sensitivity (0.0-1.0)
wake_word:
  sensitivity: 0.5

# Speech recognition options
speech:
  whisper:
    model_size: "base"  # Options: tiny, base, small, medium
    compute_type: "float16"  # Optimized for RTX 3080

# LLM configuration
llm:
  gpu_layers: 32  # Optimized for RTX 3080 (10GB VRAM)
  precision: "float16"  # Best for RTX 3080 Tensor Cores

# Threading configuration for Ryzen 9 5900X
threading:
  max_workers: 8  # Uses 8 of 12 cores for optimal balance
```

### System Service Setup (Optional)

To run Maggie as a system service:

1. Create a systemd service file:
   ```bash
   sudo nano /etc/systemd/system/maggie.service
   ```

2. Add the following content:
   ```
   [Unit]
   Description=Maggie AI Assistant
   After=network.target

   [Service]
   User=YOUR_USERNAME
   WorkingDirectory=/path/to/maggie
   ExecStart=/path/to/maggie/start_maggie.sh
   Restart=on-failure
   RestartSec=5
   Nice=-10

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl enable maggie
   sudo systemctl start maggie
   ```

## Troubleshooting (Linux)

### Python Version Issues
- **Problem:** Incorrect Python version
- **Check:** Run `python3 --version` and `python3.10 --version`
- **Solution:** Install Python 3.10 with `sudo apt install python3.10 python3.10-venv python3.10-dev`
- **Verification:** Run `python3.10 --version` to confirm

### CUDA Installation Issues
- **Problem:** CUDA not detected or incorrect version
- **Check:** Run `nvcc --version` and `python3 -c "import torch; print(torch.cuda.is_available())"`
- **Solution:** 
  1. Ensure NVIDIA drivers are installed: `nvidia-smi`
  2. Check CUDA PATH: `echo $PATH | grep cuda`
  3. Check LD_LIBRARY_PATH: `echo $LD_LIBRARY_PATH`
  4. Reinstall CUDA Toolkit 11.8 if needed

### Audio Issues
- **Problem:** PyAudio or audio playback errors
- **Solution:**
  ```bash
  sudo apt install libasound2-dev libportaudio2
  pip uninstall pyaudio
  pip install --no-binary :all: pyaudio
  ```

### Permission Issues
- **Problem:** "Permission denied" errors with audio or GPU
- **Solution:**
  1. Add user to audio group: `sudo usermod -a -G audio $USER`
  2. Check GPU permissions: `ls -la /dev/nvidia*`
  3. Log out and log back in for group changes to take effect

### CUDA Out of Memory Errors
- **Problem:** CUDA out of memory during model loading
- **Solution:**
  1. Reduce GPU layers in config.yaml: `gpu_layers: 24` (instead of 32)
  2. Enable auto adjustment: `gpu_layer_auto_adjust: true`
  3. Close other GPU-intensive applications

## Performance Verification (Linux)

### Check GPU Acceleration

Run the following in Python:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### Verify System Optimization

```bash
python main.py --verify
```

### Monitor Resource Usage
- **CPU Utilization:** 
  - Use `htop` or `mpstat -P ALL` to monitor CPU usage
  - Should see multiple cores active but not all at 100%

- **GPU Utilization:**
  - Use `nvidia-smi` or `watch -n 1 nvidia-smi` to monitor GPU
  - Check memory usage during model loading and inference
  - Should stay under 8GB for normal operation

- **Memory Usage:**
  - Monitor with `free -h` or `htop`
  - Should remain under 24GB (75% of 32GB)

For additional help or to report issues, please visit:
- GitHub repository: https://github.com/formosa/maggie
- Documentation: https://formosa.github.io/maggie/docs