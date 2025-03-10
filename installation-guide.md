# Maggie AI Assistant - Installation Guide

This guide provides detailed instructions for installing Maggie AI Assistant on both Windows and Linux systems.

## Prerequisites

Before installing Maggie, ensure your system meets the following requirements:

- **CPU**: AMD Ryzen 9 5900X or equivalent 12-core processor
- **GPU**: NVIDIA GeForce RTX 3080 with 10GB VRAM or equivalent
- **RAM**: 32GB DDR4-3200 or faster
- **OS**: Windows 11 Pro (64-bit) or Ubuntu 22.04+ (64-bit)
- **Python**: 3.10.x
- **CUDA**: CUDA 11.8 and cuDNN

## Required Downloads

1. **Python 3.10**
   - Windows: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
   - Linux: Use your distribution's package manager or [https://www.python.org/downloads/source/](https://www.python.org/downloads/source/)

2. **CUDA Toolkit 11.8**
   - [https://developer.nvidia.com/cuda-11-8-0-download-archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)

3. **cuDNN for CUDA 11.x**
   - [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

4. **Visual C++ Redistributable** (Windows only)
   - [https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

5. **Porcupine Wake Word Engine License**
   - [https://console.picovoice.ai/](https://console.picovoice.ai/)

6. **Mistral 7B Instruct Model (GPTQ 4-bit)**
   - [https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ)

7. **Piper TTS Voice Models**
   - [https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US)

## Installation Instructions

### Windows (PowerShell)

1. **Install Python 3.10**
   ```powershell
   # Download the installer from python.org and run with admin privileges
   # Ensure you check "Add Python to PATH" during installation
   
   # Verify installation
   python --version
   ```

2. **Install CUDA and cuDNN**
   ```powershell
   # Download and install CUDA Toolkit 11.8 from the NVIDIA website
   # Download cuDNN and extract to the CUDA installation directory
   
   # Add to Path if not already done by the installer
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin", "Machine")
   ```

3. **Clone the repository**
   ```powershell
   git clone https://github.com/formosa/maggie.git
   cd maggie
   ```

4. **Create and activate a virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

5. **Install dependencies**
   ```powershell
   # Install core dependencies
   pip install -e .
   
   # Install GPU dependencies
   pip install -e ".[gpu]"
   ```

6. **Download and install models**
   ```powershell
   # Create model directories
   mkdir -p models/tts
   
   # Download Mistral model
   # Use git-lfs to download from Hugging Face
   git lfs install
   git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ models/mistral-7b-instruct-v0.3-GPTQ-4bit
   
   # Download Piper TTS voice (example)
   Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx" -OutFile "models/tts/en_US-kathleen-medium/en_US-kathleen-medium.onnx"
   Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json" -OutFile "models/tts/en_US-kathleen-medium/en_US-kathleen-medium.json"
   ```

7. **Configure Maggie**
   ```powershell
   # Copy example config file
   Copy-Item config.yaml.example config.yaml
   
   # Edit config file with your Picovoice access key
   notepad config.yaml
   ```

### Linux (Bash)

1. **Install Python 3.10**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev python3-pip
   
   # Verify installation
   python3.10 --version
   ```

2. **Install CUDA and cuDNN**
   ```bash
   # CUDA Toolkit
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   
   # cuDNN (download from NVIDIA website first)
   tar -xzvf cudnn-linux-x86_64-8.x.x.x_cuda11-archive.tar.xz
   sudo cp cudnn-linux-x86_64-8.x.x.x_cuda11-archive/include/cudnn*.h /usr/local/cuda-11.8/include
   sudo cp cudnn-linux-x86_64-8.x.x.x_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64
   sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
   
   # Add to PATH
   echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Install audio dependencies**
   ```bash
   sudo apt install portaudio19-dev libsndfile1 ffmpeg
   ```

4. **Clone the repository**
   ```bash
   git clone https://github.com/formosa/maggie.git
   cd maggie
   ```

5. **Create and activate a virtual environment**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

6. **Install dependencies**
   ```bash
   # Install core dependencies
   pip install -e .
   
   # Install GPU dependencies
   pip install -e ".[gpu]"
   ```

7. **Download and install models**
   ```bash
   # Create model directories
   mkdir -p models/tts/en_US-kathleen-medium
   
   # Download Mistral model
   # Use git-lfs to download from Hugging Face
   git lfs install
   git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ models/mistral-7b-instruct-v0.3-GPTQ-4bit
   
   # Download Piper TTS voice
   wget -P models/tts/en_US-kathleen-medium https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx
   wget -P models/tts/en_US-kathleen-medium https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json
   ```

8. **Configure Maggie**
   ```bash
   # Copy example config file
   cp config.yaml.example config.yaml
   
   # Edit config file with your Picovoice access key
   nano config.yaml
   ```

## Verifying Installation

After completing the installation, verify that everything works correctly:

1. **Test CUDA availability**
   ```python
   # Run Python
   python
   
   # In Python interpreter
   import torch
   print(torch.cuda.is_available())  # Should print True
   print(torch.cuda.get_device_name(0))  # Should print your GPU name
   exit()
   ```

2. **Run Maggie in debug mode**
   ```bash
   # PowerShell (Windows)
   python main.py --debug
   
   # Bash (Linux)
   python main.py --debug
   ```

3. **Check log output**
   - The application should start and log information about hardware detection
   - The GUI should appear with the status "IDLE"
   - Say the wake word "Maggie" to activate the assistant

## Troubleshooting

### Common Issues on Windows

1. **PyAudio installation errors**
   ```powershell
   # Install pipwin
   pip install pipwin
   
   # Install PyAudio using pipwin
   pipwin install pyaudio
   ```

2. **CUDA not detected**
   ```powershell
   # Check environment variables
   $env:Path -split ";"
   $env:CUDA_PATH
   
   # Ensure CUDA_PATH is set
   [Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8", "Machine")
   ```

### Common Issues on Linux

1. **PortAudio errors**
   ```bash
   sudo apt install libasound2-dev
   pip uninstall pyaudio
   pip install --no-binary :all: pyaudio
   ```

2. **Library not found errors**
   ```bash
   # Check library paths
   echo $LD_LIBRARY_PATH
   
   # Install missing libraries
   sudo apt install libblas-dev liblapack-dev libatlas-base-dev
   ```
