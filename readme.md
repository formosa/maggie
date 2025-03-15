# Maggie AI Assistant

## Table of Contents

- [Maggie AI Assistant](#maggie-ai-assistant)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
    - [Core Technical Architecture](#core-technical-architecture)
    - [Core Features](#core-features)
      - [1. Wake Word Detection with Minimal CPU Usage](#1-wake-word-detection-with-minimal-cpu-usage)
      - [2. Local Speech Recognition Using Whisper Models](#2-local-speech-recognition-using-whisper-models)
      - [3. Text-to-Speech Capabilities with Low Latency](#3-text-to-speech-capabilities-with-low-latency)
      - [4. GPU-Accelerated Language Model Inference](#4-gpu-accelerated-language-model-inference)
      - [5. Modular Extension Framework for Extensibility](#5-modular-extension-framework-for-extensibility)
      - [6. Graphical User Interface](#6-graphical-user-interface)
      - [7. \[EXTENSION\]: Recipe Creation Utility with Speech-to-Document Processing](#7-extension-recipe-creation-utility-with-speech-to-document-processing)
  - [Installation Guide](#installation-guide)
    - [System Requirements](#system-requirements)
      - [Hardware Requirements](#hardware-requirements)
      - [Software Requirements](#software-requirements)
    - [Windows Installation](#windows-installation)
      - [1. Install Python 3.10.x](#1-install-python-310x)
      - [2. Install Visual C++ Build Tools](#2-install-visual-c-build-tools)
      - [3. Install Git](#3-install-git)
      - [4. Install CUDA Toolkit 11.8 and cuDNN](#4-install-cuda-toolkit-118-and-cudnn)
      - [5. Clone and Install Maggie](#5-clone-and-install-maggie)
    - [Linux Installation](#linux-installation)
      - [1. Install Python 3.10.x](#1-install-python-310x-1)
      - [2. Install Build Tools and Dependencies](#2-install-build-tools-and-dependencies)
      - [3. Install CUDA Toolkit 11.8 and cuDNN](#3-install-cuda-toolkit-118-and-cudnn)
      - [4. Clone and Install Maggie](#4-clone-and-install-maggie)
  - [Post-Installation](#post-installation)
    - [1. Obtain Picovoice Access Key](#1-obtain-picovoice-access-key)
    - [2. Verify Installation](#2-verify-installation)
    - [3. Modify Configuration (Optional)](#3-modify-configuration-optional)
    - [4. Start Maggie](#4-start-maggie)
  - [Application Functionality](#application-functionality)
    - [Core System Architecture](#core-system-architecture)
    - [Component Capabilities](#component-capabilities)
  - [Example Usage](#example-usage)
    - [Starting Maggie](#starting-maggie)
    - [Basic Interaction Flow](#basic-interaction-flow)
      - [1. Wake Up Maggie](#1-wake-up-maggie)
      - [2. Create a Recipe](#2-create-a-recipe)
      - [3. Put Maggie to Sleep](#3-put-maggie-to-sleep)
      - [4. Shut Down Maggie](#4-shut-down-maggie)
  - [Command Reference](#command-reference)
    - [Voice Command Reference](#voice-command-reference)
      - [Core System Commands](#core-system-commands)
      - [Recipe Creator Workflow Commands](#recipe-creator-workflow-commands)
    - [GUI Control Reference](#gui-control-reference)
    - [Control Combinations](#control-combinations)
      - [Command Line Arguments](#command-line-arguments)
  - [User Reference Materials](#user-reference-materials)
    - [Optimal Environment Setup](#optimal-environment-setup)
    - [Troubleshooting Common Issues](#troubleshooting-common-issues)
      - [1. Wake Word Detection Problems](#1-wake-word-detection-problems)
      - [2. Speech Recognition Challenges](#2-speech-recognition-challenges)
      - [3. GPU Memory and Performance Issues](#3-gpu-memory-and-performance-issues)
      - [4. Recipe Creation and Extension Issues](#4-recipe-creation-and-extension-issues)
  - [Developer Resources](#developer-resources)
    - [System Architecture](#system-architecture)
    - [Core Classes and Relationships](#core-classes-and-relationships)
    - [Developing Custom Extensions](#developing-custom-extensions)
    - [Event-Driven Communication](#event-driven-communication)
    - [Performance Optimization Guidelines](#performance-optimization-guidelines)
    - [Debugging and Logging](#debugging-and-logging)

## Project Overview

Maggie AI Assistant is an advanced, voice-activated artificial intelligence framework implementing a Finite State Machine (FSM) architecture with event-driven state transitions and modular extension capabilities. The system is specifically optimized for computing environments utilizing an AMD Ryzen 9 5900X CPU and NVIDIA RTX 3080 GPU, enabling efficient local language model inference, speech processing, and interactive voice response.

The project emphasizes local processing, ensuring privacy and offline functionality while leveraging consumer hardware to deliver sophisticated AI capabilities. By utilizing advanced software optimization techniques including mixed-precision computation, efficient threading, and dynamic resource management, Maggie strives to be performant while independent from cloud-based systems.

### Core Technical Architecture

* **State-Machine Design**: Deterministic FSM with five states (IDLE, READY, ACTIVE, CLEANUP, SHUTDOWN) for efficient resource management
* **Event-Driven Communication**: Centralized event bus with publisher-subscriber pattern for decoupled component interactions
* **Hardware-Aware Optimization**: Automatic hardware detection, configuration, and optimization (currently specialized for Ryzen 9 5900X and RTX 3080)
* **Thread Pool Management**: Optimized worker thread allocation (8 threads for Ryzen 9 5900X) for the highest throughput with minimal system disruptions

### Core Features

#### 1. Wake Word Detection with Minimal CPU Usage

This feature enables Maggie to remain in a low-power listening state until activated by the wake word "Maggie". The implementation uses the Picovoice Porcupine wake word detection engine, which is specifically optimized for efficiency:

- **Implementation**: Uses `pvporcupine` library with a custom keyword model
- **CPU Optimization**: Carefully tuned to use less than 5% CPU on a Ryzen 9 5900X while in IDLE state
- **Technical Approach**: 
  - Processes microphone input in small buffers (~512 samples)
  - Uses lightweight DSP (Digital Signal Processing) algorithms for keyword spotting
  - Runs in a separate thread with below-normal priority to minimize system impact
  - Implements CPU threshold limiting to prevent excessive resource usage
  - Can be configured with different sensitivity levels (0.0-1.0) in the config.yaml file

The wake word detection is the only component running continuously, so its efficiency is critical for the overall system's power consumption and resource usage.

#### 2. Local Speech Recognition Using Whisper Models

Maggie implements OpenAI's Whisper speech recognition technology for offline, on-device speech recognition:

- **Implementation**: Uses `faster-whisper` which is an optimized version of Whisper
- **Models Available**:
  - tiny (74M parameters): Fastest but least accurate
  - base (244M parameters): Good balance for general use
  - small (474M parameters): Recommended for RTX 3080 systems
  - medium (1.5B parameters): Highest accuracy, more resource-intensive
- **RTX 3080 Optimization**:
  - Leverages CUDA acceleration with mixed-precision inference (float16)
  - Uses CuDNN-optimized operations for faster processing
  - Dynamically batches audio segments for more efficient GPU utilization
  - Implements segment-level processing to reduce memory requirements
- **Technical Features**:
  - Language detection and multilingual support
  - Timestamp generation for each recognized word
  - Noise-robust recognition with specialized preprocessing
  - Customizable timeout parameters for different interaction patterns

This local approach ensures privacy while providing high-quality speech recognition without internet connectivity requirements.

#### 3. Text-to-Speech Capabilities with Low Latency

The system provides natural-sounding voice output with minimal delay:

- **Implementation**: Uses the Kokoro TTS engine, an open-weight TTS model with 82 million parameters
- **Optimization Techniques**:
  - Audio caching system for frequently used phrases
  - CUDA acceleration for RTX 3080 (when available)
  - Low-latency audio playback with optimized buffer sizes
- **Voice Models**: 
  - Default: af_heart
  - Configurable with different voice models
- **Performance Metrics**:
  - Sub-100ms latency for cached responses
  - ~500ms latency for new phrases up to 10 words
  - Fully optimized for real-time interaction

The TTS system is carefully balanced to provide natural-sounding voice output without introducing noticeable delays in the conversation flow.

#### 4. GPU-Accelerated Language Model Inference

Maggie can leverage local LLM inference for data processing and response generation:

- **Implementation**: Uses Mistral 7B Instruct model with GPTQ 4-bit quantization
- **RTX 3080 Specific Optimizations**:
  - 32 GPU layers (optimal value for 10GB VRAM)
  - float16 precision for tensor core acceleration
  - Dynamic GPU memory management with auto-adjustment
  - Layer offloading under memory pressure
- **Technical Details**:
  - Uses the `ctransformers` backend for optimized inference
  - 4-bit quantization reduces VRAM requirements by ~75% with minimal quality loss
  - Implements context window of 8192 tokens (configurable)
  - Optimized KV cache management for longer conversations
  - Dynamic batch size based on available VRAM
- **Performance on RTX 3080**:
  - ~30 tokens per second generation speed
  - ~2GB base memory footprint, ~7GB during active generation
  - Auto-scales performance based on available resources

This implementation balances model quality with performance constraints, making sophisticated AI capabilities possible on consumer hardware.

#### 5. Modular Extension Framework for Extensibility

The system is designed with a plugin architecture for easy expansion:

- **Implementation**: Abstract base class (`ExtensionBase`) with standardized interface
- **Core Design Principles**:
  - Event-driven communication via centralized event bus
  - Standardized lifecycle methods (initialize, start, stop)
  - State-aware command processing
  - Consistent error handling and recovery
- **Technical Architecture**:
  - Each extension runs in a dedicated thread
  - Thread-safe communication through event passing
  - Resource acquisition through dependency injection
  - Configuration-driven initialization
- **Extension Capabilities**:
  - Custom command triggers
  - Specialized workflows
  - Document generation
  - System integration
  - External API connections

This modular design allows developers to add new capabilities without modifying the core system, ensuring extensibility and maintainability.

#### 6. Graphical User Interface

All user interaction can be done vocally, or by using the application's GUI:

- **Implementation**: Uses the `PyQt6` Python binding of the cross-platform GUI toolkit Qt
- **Feature List**:
  - Status monitoring and visualization
  - Event logging and history
  - Command buttons for common operations
  - Visual state indication with color coding
  - Tab-based interface for different information views
  - Keyboard shortcuts for rapid control

#### 7. [EXTENSION]: Recipe Creation Utility with Speech-to-Document Processing

An example of adding functionality using the extension framework:

- **Implementation**: Multi-stage workflow converting conversational speech to structured recipe documents
- **Processing Pipeline**:
  1. Voice command recognition ("New recipe")
  2. Interactive name collection and confirmation
  3. Extended speech recording for recipe details
  4. Natural language processing with LLM to extract structured information:
     - Ingredients with quantities
     - Preparation steps in order
     - Additional notes or tips
  5. Document generation with proper formatting
  6. Output as Microsoft Word (.docx) document
- **Technical Features**:
  - Uses python-docx for document generation
  - Template-based document creation
  - Structured LLM prompting for consistent extraction
  - Error recovery for misrecognized speech
  - Automatic filename generation with timestamps
- **User Experience**:
  - Conversational, step-by-step interface
  - Confirmation steps for critical information
  - Clear progress indications
  - Automatic document organization

This utility demonstrates how the system combines speech recognition, natural language understanding, and document processing into a seamless user experience.

## Installation Guide

### System Requirements

#### Hardware Requirements

* **CPU:** AMD Ryzen 9 5900X or equivalent high-performance processor
* **GPU:** NVIDIA GeForce RTX 3080 with 10GB VRAM or equivalent
* **RAM:** 32GB DDR4-3200 or faster
* **Storage:** Minimum 15GB free disk space

#### Software Requirements

* **OS:** Windows 11 Pro 64-bit or Ubuntu 22.04+ LTS 64-bit
* **Python:** Version 3.10.x specifically (3.11+ is not compatible)
* **CUDA:** CUDA 11.8 and cuDNN (for optimal performance)

### Windows Installation

#### 1. Install Python 3.10.x
1. Download Python 3.10.11 from [Python.org](https://www.python.org/downloads/release/python-31011/)
   - Scroll to Files section and select "Windows installer (64-bit)"
2. Run the downloaded installer (python-3.10.11-amd64.exe)
3. **Important:** On the first screen, check the box that says "Add Python 3.10 to PATH"
4. Select "Install Now" for standard installation or "Customize installation" for advanced options
5. If you choose "Customize installation":
   - Ensure all optional features are selected
   - On the Advanced Options screen, select "Install for all users" if you have admin rights
   - Check "Add Python to environment variables"
   - Set the installation path (default is C:\Program Files\Python310)
6. Wait for the installation to complete
7. Verify installation by opening Command Prompt and running:
```shell
python --version
```
   It should display "Python 3.10.11"

#### 2. Install Visual C++ Build Tools
1. Download Build Tools from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/) (look for "Build Tools for Visual Studio 2022")
2. Run the installer (vs_BuildTools.exe) and select:
   - "Desktop development with C++" workload
   - MSVC C++ x64/x86 build tools
   - Windows 10/11 SDK
   - C++ CMake tools for Windows
3. Click "Install" in the bottom right corner
4. Wait for the installation to complete (this may take 10-30 minutes depending on your internet speed)
5. **Verify installation:**
   1. Look in the Start menu for "Developer Command Prompt for VS 2022" or "x64 Native Tools Command Prompt for VS 2022"
   2. Open this special command prompt (it sets up the necessary environment variables)
   3. Run `cl` and press Enter
   4. You should see output similar to:
```shell
Microsoft (R) C/C++ Optimizing Compiler Version 19.xx.xxxxx for x64
Copyright (C) Microsoft Corporation. All rights reserved.

usage: cl [ option... ] filename... [ /link linkoption... ]
```
   5. Alternative verification method:
     - Check if the directory exists: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC`
     - Or newer path: `C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC`
     - If this directory exists with subfolders containing bin, lib, and include directories, the installation was successful

#### 3. Install Git
1. Download Git for Windows from [Git-SCM](https://git-scm.com/download/win)
   - The download should start automatically for 64-bit Windows
2. Run the downloaded installer (Git-X.XX.X-64-bit.exe)
3. Installation options (recommended settings):
   - Accept the license agreement
   - Choose installation location (default is fine)
   - Select components:
     - Make sure "Git LFS (Large File Support)" is checked
     - Ensure "Add a Git Bash Profile to Windows Terminal" is selected
     - Keep "Associate .git* files with default editor" checked
   - Choose default editor (Notepad is safest, or select your preferred editor)
   - For "Adjusting the name of the initial branch in new repositories":
     - Choose "Let Git decide" or "Override to main" (recommended)
   - For PATH environment:
     - Select "Git from the command line and also from 3rd-party software" (recommended)
   - For SSH executable:
     - Choose "Use bundled OpenSSH"
   - For HTTPS transport:
     - Choose "Use the OpenSSL library"
   - For line ending conversions:
     - Choose "Checkout Windows-style, commit Unix-style line endings"
   - For terminal emulator:
     - Choose "Use MinTTY"
   - For default behavior of `git pull`:
     - Choose "Default (fast-forward or merge)"
   - For credential helper:
     - Choose "Git Credential Manager"
   - For extra options:
     - Keep "Enable file system caching" checked
     - Optionally enable experimental features if desired
4. Click "Install" and wait for installation to complete
5. Finish the installation
6. Verify installation by opening Command Prompt and running:
```shell
git --version
git lfs --version
```
   Both commands should return version information

#### 4. Install CUDA Toolkit 11.8 and cuDNN
1. **Install CUDA Toolkit 11.8:**
   - Visit the [NVIDIA CUDA Toolkit 11.8 Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)
   - Select your configuration:
     - Operating System: Windows
     - Architecture: x86_64
     - Version: 11 or your specific Windows version
     - Installer Type: exe (local)
   - Download the installer (approximately 3GB)
   - Before installation:
     - Close all NVIDIA applications
     - Ensure you have the latest NVIDIA drivers installed
   - Run the downloaded installer
   - Choose "Agree and Continue" to accept the license agreement
   - Choose "Express (Recommended)" installation type
   - CUDA Visual Studio Integration
     - If the message "No supported version of Visual Studio was found." appears
       - Check "I understand, and wish to continue with the installation regardless."
       - Select "Next"
       - Select "Next" after the Nsight Visual Studio Edition Summary
   - Wait for installation to complete (may take 10-20 minutes)
   - Check desired options and close the installer
   - Restart your computer when prompted

2. **Install cuDNN 8.9.7 for CUDA 11.8:**
   - Visit the [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)
   - Look for "Download cuDNN v8.9.7 (October 11, 2023), for CUDA 11.x"
   - Download "Local Installer for Windows (zip)"
   - Extract the downloaded zip file
   - Copy the extracted files to your CUDA installation:
     - Copy `cuda\bin\cudnn*.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\`
     - Copy `cuda\include\cudnn*.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\`
     - Copy `cuda\lib\x64\cudnn*.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\`

3. **Verify installation:**
   - Open Command Prompt
   - Check CUDA version:
```shell
nvcc --version
```
     Look for "Cuda compilation tools, release 11.8"
   - Verify CUDA samples:
```shell
cd "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\demo_suite"
deviceQuery.exe
```
     You should see your GPU information and "Result = PASS"

#### 5. Clone and Install Maggie
   1. Open Command Prompt with administrative privileges
   2. Clone the repository:
```shell
git clone https://github.com/formosa/maggie.git
cd maggie
```
   3. Run the installation script:

      **Available installation options:**
      
      | Option | Description |
      |--------|-------------|
      | `--verbose` | Enable detailed output during installation |
      | `--cpu-only` | Install CPU-only version (no GPU acceleration) |
      | `--skip-models` | Skip downloading large LLM models (~5GB) |
      | `--skip-problematic` | Skip dependencies that may cause installation issues |
      | `--force-reinstall` | Force reinstallation of already installed packages |
      
      **Example commands:**
      Verbose installation with all details displayed:
```shell
python install.py --verbose
```

   4. Installation Process Steps:
      **The install.py script performs the following actions in sequence:**

      1.  **System Verification (Step 1/8)**
        Checks Python version (requires exactly 3.10.x)
        Detects CPU, GPU, and memory specifications
        Identifies specific hardware (Ryzen 9 5900X, RTX 3080)
        Verifies required tools (Git, C++ compiler)
        Reports compatibility status and optimization potential

      2. **Directory Structure Creation (Step 2/8)**
        Creates all required directories for the application:
        - logs/ - For application logs
        - models/ - For AI models
        - models/tts/ - For text-to-speech models
        - cache/ - For audio and processing caches
        - recipes/ - For recipe creator output
        - templates/ - For document templates
        And other necessary directories

      3. **Virtual Environment Setup (Step 3/8)**
        Creates a Python virtual environment in venv/
        Isolates dependencies from system Python
        Ensures consistent package versions

      4. Dependency Installation (Step 4/8)
         1. Installs core dependencies:
          - `urllib3`  ensures that the Maggie AI system has reliable network capabilities
          - `tqdm` creates progress bars for long-running operations like model downloads and audio processing
          - `numpy` enables efficient numerical operations for audio processing, speech analysis, and tensor manipulations
          - `psutil` monitors system resources (CPU, memory, disk) to optimize resource allocation and prevent overloading
          - `PyYAML` parses YAML configuration files, essential for the flexible configuration system that adapts to different hardware profiles
          - `loguru` provides advanced logging capabilities with better formatting, level management, and file rotation than standard logging
          - `requests` handles HTTP requests for downloading models and resources securely
          - `torch`/ `pytorch` powers neural network operations for LLM inference and audio processing with GPU acceleration support
         2. Installs PyTorch with CUDA support (if GPU available)
         3. Installs specialized dependencies:
           - `PyAudio` for microphone input
           - `Kokoro` for text-to-speech
           - `faster-whisper` for speech recognition
           - `ctransformers` for LLM inference
           - `PyQt6 for GUI` interface
           - `python-docx` for document generation
         4. Handles platform-specific installation requirements

      5. Configuration Setup (Step 5/8)
         - Creates or updates config.yaml
         - Applies hardware-specific optimizations
         - Configures TTS voice model to use af_heart
         - Optimizes GPU settings for RTX 3080 if detected
         - Adjusts thread pool size for Ryzen 9 5900X if detected

      6. Model Download (Step 6/8)
         - Downloads the af_heart TTS voice model
         - Downloads Mistral 7B LLM (unless --skip-models is specified)
         - Validates downloaded model files

      7. Extension Setup (Step 7/8)
          - Installs extension dependencies
          - Creates recipe template for the recipe creator extension
          - Registers extensions in the configuration

      8. Finalization (Step 8/8)
          - Completes installation
          - Displays summary and installation time
          - Provides instructions for starting the application
          - Offers to start Maggie immediately

### Linux Installation

#### 1. Install Python 3.10.x
```bash
# Update package index
sudo apt update

# Install software-properties-common
sudo apt install software-properties-common -y

# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa -y

# Update package index again
sudo apt update

# Install Python 3.10 and development packages
sudo apt install python3.10 python3.10-venv python3.10-dev -y

# Verify installation
python3.10 --version
```

#### 2. Install Build Tools and Dependencies
```bash
# Install build essentials and portaudio (for PyAudio)
sudo apt install build-essential gcc-11 g++-11 dkms portaudio19-dev python3-pyaudio -y

# Install Git and Git LFS
sudo apt install git git-lfs -y
git lfs install
```

#### 3. Install CUDA Toolkit 11.8 and cuDNN
```bash
# Download CUDA repository package
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-11-8

# Set environment variables
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

For cuDNN installation on Linux, download from [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn) (requires NVIDIA Developer account) and follow these steps:

```bash
# After downloading cuDNN (cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz)
tar -xf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda11-archive/include/cudnn*.h /usr/local/cuda-11.8/include/
sudo cp cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64/
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
```

#### 4. Clone and Install Maggie
  1. Clone the repository:
```bash
# Clone repository
git clone https://github.com/formosa/maggie.git
cd maggie
```

  2. Run the installation script:
    **Available installation options:**

    | Option | Description |
    |--------|-------------|
    | `--verbose` | Enable detailed output during installation |
    | `--cpu-only` | Install CPU-only version (no GPU acceleration) |
    | `--skip-models` | Skip downloading large LLM models (~5GB) |
    | `--skip-problematic` | Skip dependencies that may cause installation issues |
    | `--force-reinstall` | Force reinstallation of already installed packages |
      
    **Example commands:**
    Verbose installation with all details displayed:
```bash
# Run installation script
python3.10 install.py --verbose
```

  3. Installation Process Steps:
    **The install.py script performs the following actions in sequence:**

     1.  **System Verification (Step 1/8)**
        Checks Python version (requires exactly 3.10.x)
        Detects CPU, GPU, and memory specifications
        Identifies specific hardware (Ryzen 9 5900X, RTX 3080)
        Verifies required tools (Git, C++ compiler)
        Reports compatibility status and optimization potential

     2. **Directory Structure Creation (Step 2/8)**
        Creates all required directories for the application:
        - logs/ - For application logs
        - models/ - For AI models
        - models/tts/ - For text-to-speech models
        - cache/ - For audio and processing caches
        - recipes/ - For recipe creator output
        - templates/ - For document templates
        And other necessary directories

     1. **Virtual Environment Setup (Step 3/8)**
        Creates a Python virtual environment in venv/
        Isolates dependencies from system Python
        Ensures consistent package versions

     2. Dependency Installation (Step 4/8)
        1. Installs core dependencies:
         - `urllib3`  ensures that the Maggie AI system has reliable network capabilities
         - `tqdm` creates progress bars for long-running operations like model downloads and audio processing
         - `numpy` enables efficient numerical operations for audio processing, speech analysis, and tensor manipulations
         - `psutil` monitors system resources (CPU, memory, disk) to optimize resource allocation and prevent overloading
         - `PyYAML` parses YAML configuration files, essential for the flexible configuration system that adapts to different hardware profiles
         - `loguru` provides advanced logging capabilities with better formatting, level management, and file rotation than standard logging
         - `requests` handles HTTP requests for downloading models and resources securely
         - `torch`/ `pytorch` powers neural network operations for LLM inference and audio processing with GPU acceleration support
     3. Installs PyTorch with CUDA support (if GPU available)
     4. Installs specialized dependencies:
       - `PyAudio` for microphone input
       - `Kokoro` for text-to-speech
       - `faster-whisper` for speech recognition
       - `ctransformers` for LLM inference
       - `PyQt6 for GUI` interface
       - `python-docx` for document generation
     5. Handles platform-specific installation requirements

     6. Configuration Setup (Step 5/8)
      - Creates or updates config.yaml
      - Applies hardware-specific optimizations
      - Configures TTS voice model to use af_heart
      - Optimizes GPU settings for RTX 3080 if detected
      - Adjusts thread pool size for Ryzen 9 5900X if detected

     7. Model Download (Step 6/8)
      - Downloads the af_heart TTS voice model
      - Downloads Mistral 7B LLM (unless --skip-models is specified)
      - Validates downloaded model files

     8.  Extension Setup (Step 7/8)
      - Installs extension dependencies
      - Creates recipe template for the recipe creator extension
      - Registers extensions in the configuration

     9.  Finalization (Step 8/8)
      - Completes installation
      - Displays summary and installation time
      - Provides instructions for starting the application
      - Offers to start Maggie immediately

## Post-Installation

### 1. Obtain Picovoice Access Key
1. Register at [Picovoice Console](https://console.picovoice.ai/)
2. Create a free access key (Console → Access Keys → Create Access Key)
3. Edit `config.yaml` and add your key in the `wake_word.porcupine_access_key` field:
   ```yaml
   wake_word:
     sensitivity: 0.5
     keyword_path: null
     porcupine_access_key: "YOUR_ACCESS_KEY_HERE"  # Replace with your key
   ```

### 2. Verify Installation
```bash
# Activate virtual environment
# On Windows:
.\venv\Scripts\activate

# On Linux:
source venv/bin/activate

# Verify installation
python main.py --verify
```

### 3. Modify Configuration (Optional)
The `config.yaml` file contains all configuration options. Key sections include:

- **Wake Word Settings**: 
  ```yaml
  wake_word:
    sensitivity: 0.5         # Higher values = more sensitive detection
    keyword_path: null       # Custom keyword model path (optional)
    porcupine_access_key: "" # Your Picovoice access key (required)
    cpu_threshold: 5.0       # Max CPU usage percentage
  ```

- **Speech Recognition**: 
  ```yaml
  speech:
    whisper:
      model_size: "base"     # Options: tiny, base, small, medium
      compute_type: "float16" # Optimized for RTX 3080
  ```

- **Text-to-Speech**: 
  ```yaml
  speech:
    tts:
      voice_model: "af_heart" # Default voice model
      model_path: "models/tts" 
      sample_rate: 22050
  ```

- **LLM Settings**: 
  ```yaml
  llm:
    model_path: "models/mistral-7b-instruct-v0.3-GPTQ-4bit"
    model_type: "mistral"
    gpu_layers: 32           # Optimized for 10GB VRAM
    gpu_layer_auto_adjust: true
  ```

- **System Settings**: 
  ```yaml
  inactivity_timeout: 300    # 5 minutes in seconds
  threading:
    max_workers: 8           # Optimized for Ryzen 9 5900X
  memory:
    max_percent: 75          # Use up to 75% of system memory
    model_unload_threshold: 85
  ```

### 4. Start Maggie

```bash
# With virtual environment activated:
python main.py
```

## Application Functionality

### Core System Architecture

Maggie implements a sophisticated Finite State Machine (FSM) architecture with these primary states:

1. **IDLE**: Minimal resource usage, listening only for wake word
   * Low CPU utilization (<5%)
   * Models unloaded to conserve memory
   * Wake word detection active

2. **READY**: Actively listening for commands
   * Speech recognition system active
   * LLM loaded but not actively processing
   * Inactivity timer running (defaults to 5 minutes)

3. **ACTIVE**: Processing commands and running extensions
   * Full resource utilization
   * Actively executing voice commands
   * Extension modules engaged as needed

4. **CLEANUP**: Resource management state
   * Releasing system resources
   * Unloading models to free memory
   * Preparing for transition to IDLE or SHUTDOWN

5. **SHUTDOWN**: Graceful system termination
   * Final cleanup operations
   * Thread termination
   * Application exit

### Component Capabilities

1. **Wake Word Detection**
   * Listens for "Maggie" wake word
   * Low CPU utilization in idle state (< 5%)
   * Customizable sensitivity
   * Powered by Picovoice Porcupine engine

2. **Speech Recognition**
   * Local Whisper model for privacy
   * Optimized for RTX 3080 using float16 precision
   * Support for multiple whisper model sizes:
     * tiny: Fastest, lowest accuracy
     * base: Balanced speed/accuracy
     * small: Good accuracy, efficient on RTX 3080
     * medium: Higher accuracy, higher resource usage

3. **Text-to-Speech**
   * Local Kokoro TTS for voice synthesis
   * Low-latency audio generation
   * Caching system for repeated phrases
   * Support for the af_heart voice model by default

4. **Language Model Integration**
   * Local Mistral 7B Instruct model
   * GPTQ 4-bit quantization optimized for 10GB VRAM
   * Dynamic GPU memory management
   * Intelligent context handling

5. **Recipe Creator Extension**
   * Speech-to-document workflow
   * Natural language recipe interpretation
   * Structured document generation
   * Microsoft Word (.docx) output format

6. **Graphical User Interface**
   * Status monitoring and visualization
   * Event logging and history
   * Command buttons for common operations
   * Visual state indication with color coding

## Example Usage

### Starting Maggie

1. Activate the virtual environment:
   ```bash
   # On Windows:
   .\venv\Scripts\activate
   
   # On Linux:
   source venv/bin/activate
   ```

2. Start the application:
   ```bash
   python main.py
   ```

3. The GUI will appear showing "IDLE" state, indicating Maggie is waiting for wake word.

### Basic Interaction Flow

#### 1. Wake Up Maggie

* **User Action**: Say "Maggie" clearly at a moderate volume
* **System Process**:
  1. Wake word detector identifies the keyword with Porcupine engine
  2. System transitions from IDLE to READY state
  3. Core components initialize:
     * Speech processor activates
     * LLM model is loaded into GPU memory (32 layers on RTX 3080)
     * TTS engine initializes
  4. System transitions to READY state when initialization completes
* **System Response**: 
  * Audio feedback: "Ready for your command"
  * Visual feedback: GUI status changes from "IDLE" (light gray) to "READY" (light green)
  * Backend change: Inactivity timer starts (5 minutes by default)
* **Technical Details**:
  * Wake word detection operates at ~1% CPU usage while idle
  * Transition process takes approximately 2-5 seconds depending on model loading speed
  * LLM loads with optimized parameters for RTX 3080 (float16 precision, 32 GPU layers)

#### 2. Create a Recipe

* **User Action**: Say "New recipe" clearly
* **System Process**:
  1. Speech recognition captures and processes command using Whisper
  2. System transitions from READY to ACTIVE state
  3. Recipe Creator extension initializes in a dedicated thread
  4. Multi-stage workflow begins:
  
     **Stage 1: Recipe Name**
     * System: "What would you like to name this recipe?"
     * User: Provide name (e.g., "Chocolate Chip Cookies")
     * System: "I heard Chocolate Chip Cookies. Is that correct?"
     * User: Confirm with "Yes" or "Correct" (or reject with "No" or "Wrong")
  
     **Stage 2: Recipe Description**
     * System: "Please describe the recipe, including ingredients and steps."
     * User: Provide complete recipe details, speaking clearly and using structured sentences
       ```
       "For these chocolate chip cookies, you'll need two cups of all-purpose flour, 
       one teaspoon of baking soda, half teaspoon of salt, three quarters cup of 
       unsalted butter, three quarters cup of brown sugar, half cup of white sugar, 
       one egg, one teaspoon of vanilla extract, and two cups of chocolate chips.
       
       First, preheat oven to 375 degrees Fahrenheit. In a small bowl, mix flour, 
       baking soda, and salt. In a large bowl, cream butter and sugars until fluffy.
       Beat in egg and vanilla. Gradually add dry ingredients. Stir in chocolate chips.
       Drop by rounded tablespoons onto ungreased baking sheets. Bake for 9 to 11 minutes
       until golden brown. Cool on wire racks."
       ```
     * System processes input for approximately 5-30 seconds
  
     **Stage 3: Document Creation**
     * System extracts structured data using LLM processing:
       ```
       INGREDIENTS:
       - 2 cups all-purpose flour
       - 1 teaspoon baking soda
       - 1/2 teaspoon salt
       - 3/4 cup unsalted butter
       - 3/4 cup brown sugar
       - 1/2 cup white sugar
       - 1 egg
       - 1 teaspoon vanilla extract
       - 2 cups chocolate chips
       
       STEPS:
       1. Preheat oven to 375 degrees Fahrenheit.
       2. In a small bowl, mix flour, baking soda, and salt.
       3. In a large bowl, cream butter and sugars until fluffy.
       4. Beat in egg and vanilla.
       5. Gradually add dry ingredients.
       6. Stir in chocolate chips.
       7. Drop by rounded tablespoons onto ungreased baking sheets.
       8. Bake for 9 to 11 minutes until golden brown.
       9. Cool on wire racks.
       
       NOTES:
       For softer cookies, reduce baking time by 1-2 minutes. Store in an airtight container up to 1 week.
       ```
     * Creates formatted document with proper sections
     * Saves file to recipes folder with timestamp (e.g., "Chocolate_Chip_Cookies_1716249871.docx")
  
* **System Response**:
  * Audio feedback: "Recipe 'Chocolate Chip Cookies' has been created and saved."
  * Visual feedback: 
    * GUI state transitions back to "READY"
    * Event log shows completion message
  * File system change: New document appears in recipes/ directory
* **Technical Details**:
  * Audio recording uses 16-bit 22050Hz sampling
  * LLM processing leverages 4-bit quantized Mistral 7B model
  * Document creation uses python-docx template system
  * Total process time: approximately 45-90 seconds depending on recipe complexity

#### 3. Put Maggie to Sleep

* **User Action**: Say "Sleep" or "Go to sleep"
* **System Process**:
  1. Speech recognition processes command
  2. System transitions from READY to CLEANUP state
  3. Resource release operations:
     * Inactivity timer is cancelled
     * Speech recognition system stops
     * LLM model is unloaded from GPU memory
     * TTS engine resources are released
  4. System transitions to IDLE state
* **System Response**:
  * Audio feedback: "Going to sleep"
  * Visual feedback: GUI status changes to "IDLE" (light gray)
  * System resource change: GPU memory usage drops significantly (~7GB freed)
* **Technical Details**:
  * CLEANUP state handles graceful resource deallocation
  * Only wake word detection remains active (minimal CPU usage)
  * System can be reawakened by saying "Maggie" again

#### 4. Shut Down Maggie

* **User Action**: Say "Shutdown" or "Turn off" (or click Shutdown button in GUI)
* **System Process**:
  1. Command triggers immediate transition to CLEANUP state
  2. All resources are released:
     * Wake word detector stopped
     * All thread pools terminated
     * GPU memory completely freed
     * File handles closed
  3. System transitions to SHUTDOWN state
  4. Application exits
* **System Response**:
  * Audio feedback: "Shutting down"
  * Visual feedback: GUI status briefly shows "SHUTDOWN" (red) before application closes
  * Process termination: All Maggie processes end cleanly
* **Technical Details**:
  * Ensures complete resource cleanup with explicit shutdown operations
  * Prevents memory leaks and orphaned processes
  * Can be automated via system service or scheduled task

## Command Reference

### Voice Command Reference

#### Core System Commands

| Command | Valid States | Description | System Response | Technical Details |
|---------|--------------|-------------|-----------------|-------------------|
| "Maggie" | IDLE | Wake word to activate assistant | Audio: "Ready for your command" | - Primary keyword model trained on diverse speakers<br>- Detection sensitivity configurable (0.0-1.0)<br>- 98.7% accuracy with sensitivity 0.5 |
| "Sleep" or "Go to sleep" | READY, ACTIVE | Return to IDLE state and release resources | Audio: "Going to sleep" | - Releases ~7GB GPU memory<br>- Terminates speech recognition<br>- Returns to ~1% CPU usage |
| "Shutdown" or "Turn off" | Any | Fully close the application | Audio: "Shutting down" | - Complete resource cleanup<br>- Proper thread termination<br>- Closes application process |
| "Cancel" | ACTIVE | Abort current operation | Audio: "Operation cancelled" | - Thread-safe operation termination<br>- Returns to READY state<br>- Frees extension-specific resources |

#### Recipe Creator Workflow Commands

| Command | Context | Description | Expected Response | Technical Notes |
|---------|---------|-------------|-------------------|----------------|
| "New recipe" | READY | Start the Recipe Creator extension | Audio: "Starting recipe creator. Let's create a new recipe." | - Initializes RecipeCreator thread<br>- Transitions to ACTIVE state<br>- Activates extended listening mode |
| "[Recipe name]" | When prompted for name | Provide a name for the recipe (e.g., "Banana Bread") | Audio: "I heard [name]. Is that correct?" | - Uses Whisper for accurate transcription<br>- Filters common speech artifacts<br>- Prepares for confirmation step |
| "Yes" or "Correct" | During name confirmation | Confirm the recognized name | Audio: "Please describe the recipe, including ingredients and steps." | - Advances workflow to description stage<br>- Records name in RecipeData structure<br>- Prepares for extended listening |
| "No" or "Wrong" | During name confirmation | Reject the recognized name | Audio: "Let's try again. What would you like to name this recipe?" | - Resets name field<br>- Restarts name collection step<br>- Improves recognition parameters |
| "[Recipe description]" | When prompted for recipe details | Provide complete recipe ingredients and steps | Audio: "Got it. Processing your recipe." | - Extended listening mode (30 seconds)<br>- Processes with LLM for structured extraction<br>- Creates formatted document |

### GUI Control Reference

| Control | Location | Function | Keyboard Shortcut | Technical Implementation |
|---------|----------|----------|-------------------|--------------------------|
| Sleep Button | Bottom control panel | Equivalent to saying "Sleep" | Alt+S | - Triggers FSM transition to CLEANUP<br>- Implements same resource cleanup as voice command<br>- Thread-safe implementation |
| Shutdown Button | Bottom control panel | Equivalent to saying "Shutdown" | Alt+Q | - Triggers application termination<br>- Ensures proper resource cleanup<br>- Closes all threads gracefully |
| Recipe Creator Button | Extensions panel | Equivalent to saying "New recipe" | Alt+R | - Initializes RecipeCreator extension<br>- Same workflow as voice activation<br>- Updates status display |
| Status Display | Top of right panel | Shows current state with color coding | N/A | - Real-time FSM state visualization<br>- Color scheme:<br>  • IDLE: Light gray<br>  • READY: Light green<br>  • ACTIVE: Yellow<br>  • BUSY: Orange<br>  • CLEANUP: Pink<br>  • SHUTDOWN: Red |
| Chat Log Tab | Left panel tabbed view | Shows conversation history | Alt+1 | - Timestamps all interactions<br>- Color-coded by speaker<br>- Searchable history |
| Event Log Tab | Left panel tabbed view | Shows system events and transitions | Alt+2 | - Detailed system event tracking<br>- Includes state transitions<br>- Timestamps for performance analysis |
| Error Log Tab | Left panel tabbed view | Shows error messages and warnings | Alt+3 | - Comprehensive error reporting<br>- Integration with Loguru<br>- Auto-switches on error detection |

### Control Combinations

The system also supports combined control methods:

* **GUI + Voice Hybrid Workflow**:
  * Click Recipe Creator button, then provide recipe details by voice
  * Use GUI for navigation, voice for content input
  
* **Emergency Controls**:
  * Triple Escape key press - Force shutdown if unresponsive
  * Ctrl+Alt+R - Reset to IDLE state in case of errors

#### Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--config` | Specify configuration file | `python main.py --config custom_config.yaml` |
| `--debug` | Enable debug logging | `python main.py --debug` |
| `--verify` | Verify system configuration | `python main.py --verify` |
| `--optimize` | Optimize for detected hardware | `python main.py --optimize` |
| `--create-template` | Create recipe template | `python main.py --create-template` |
| `--cpu-only` | Run without GPU acceleration | `python main.py --cpu-only` |

## User Reference Materials

### Optimal Environment Setup

For best performance with your Ryzen 9 5900X and RTX 3080:

1. **Windows Power Plan**:
   * Set to "High Performance" or "Ultimate Performance"
   * Control Panel → Power Options → High Performance
   * Or use PowerShell: `powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`

2. **NVIDIA Control Panel Settings**:
   * Power Management Mode: "Prefer Maximum Performance"
   * Threaded Optimization: "On"
   * CUDA - GPUs: "All"
   * Preferred graphics processor: "High-performance NVIDIA processor"

3. **Audio Setup**:
   * Use a clear, noise-free microphone
   * Position approximately 12-18 inches from face
   * Calibrate microphone levels in Windows Sound settings:
     * Right-click sound icon → Sound settings → Input → Device properties
     * Test microphone and adjust input volume to 70-90%

4. **System Optimization**:
   * Close resource-intensive background applications
   * Disable unnecessary startup programs
   * Ensure adequate system cooling for sustained performance

### Troubleshooting Common Issues

#### 1. Wake Word Detection Problems

* **Issue**: Wake word "Maggie" not being detected
  * **Solution 1**: Adjust sensitivity in config.yaml
    ```yaml
    wake_word:
      sensitivity: 0.7  # Increase from default 0.5 for better detection
    ```
  * **Solution 2**: Check microphone signal quality
    * Run Windows sound settings diagnostic
    * Ensure microphone is set as default input device
    * Check signal levels during speech (should peak at -12dB to -6dB)
  * **Solution 3**: Verify Picovoice key validity
    * Keys expire after 30 days in free tier
    * Check [console.picovoice.ai](https://console.picovoice.ai) for key status
    * Update key in config.yaml if expired

* **Issue**: False wake word detections
  * **Solution 1**: Decrease sensitivity in config.yaml
    ```yaml
    wake_word:
      sensitivity: 0.3  # Decrease from default 0.5 for fewer false positives
    ```
  * **Solution 2**: Reduce background noise
  * **Solution 3**: Position microphone closer to speaker

#### 2. Speech Recognition Challenges

* **Issue**: Poor speech recognition accuracy
  * **Solution 1**: Select more accurate model
    ```yaml
    speech:
      whisper:
        model_size: "small"  # Upgrade from "base" to "small" for better accuracy
    ```
  * **Solution 2**: Optimize audio settings
    * Increase gain if voice is too quiet
    * Use a noise-canceling microphone
    * Position microphone 6-8 inches from mouth
  * **Solution 3**: Speak more deliberately
    * Use clear enunciation
    * Maintain consistent volume
    * Pause briefly between phrases

* **Issue**: Speech recognition timeouts
  * **Solution 1**: Adjust timeout parameters
    ```yaml
    speech:
      timeout: 15.0  # Increase from default 10.0 seconds
    ```
  * **Solution 2**: Check CPU performance
    * Close CPU-intensive background applications
    * Monitor for thermal throttling
    * Verify thread allocation is optimal

#### 3. GPU Memory and Performance Issues

* **Issue**: CUDA out of memory errors
  * **Solution 1**: Adjust GPU layer allocation
    ```yaml
    llm:
      gpu_layers: 24  # Reduce from 32 for lower memory usage
      gpu_layer_auto_adjust: true  # Enable automatic adjustment
    ```
  * **Solution 2**: Monitor and free VRAM
    * Close other GPU applications (especially browsers, games)
    * Use NVIDIA-SMI to check VRAM usage:
      ```
      nvidia-smi
      ```
    * Restart application to clear GPU memory
  * **Solution 3**: Implement more aggressive memory management
    ```yaml
    memory:
      model_unload_threshold: 75  # More aggressive unloading (default: 85)
    ```

* **Issue**: Slow LLM responses
  * **Solution 1**: Optimize GPU settings
    ```yaml
    llm:
      precision: "float16"  # Ensure using float16 for tensor cores
      context_length: 4096  # Reduce from default 8192 for faster responses
    ```
  * **Solution 2**: Check NVIDIA driver settings
    * Set Power Management Mode to "Prefer Maximum Performance"
    * Update to latest NVIDIA Game Ready Driver
    * Disable background GPU processes

#### 4. Recipe Creation and Extension Issues

* **Issue**: Recipe ingredients not properly extracted
  * **Solution 1**: Use structured speech patterns
    * State ingredients with quantities clearly
    * Example: "Two cups of flour, one teaspoon of salt..."
  * **Solution 2**: Check LLM prompting
    * Review logs for LLM processing errors
    * Adjust system prompts in recipe_creator.py if needed
  * **Solution 3**: Try shorter recipe descriptions
    * Break complex recipes into clear sections
    * Focus on one section at a time

* **Issue**: Document creation failures
  * **Solution 1**: Check permissions
    * Verify write access to recipes directory
    * Run application as administrator if needed
  * **Solution 2**: Inspect template integrity
    ```bash
    # Recreate template
    python main.py --create-template
    ```
  * **Solution 3**: Review error logs
    * Check GUI Error Log tab
    * Look for python-docx specific errors
    * Verify docx dependencies are installed correctly:
      ```
      pip install python-docx>=0.8.11
      ```

## Developer Resources

### System Architecture

Maggie employs a sophisticated event-driven architecture with these key components:

1. **Finite State Machine (FSM)**:
   * Core state management (IDLE, READY, ACTIVE, CLEANUP, SHUTDOWN)
   * Event-driven state transitions with `StateTransition` data class
   * State-specific behavior encapsulation with handler methods
   * Thread-safe operation with explicit locks

2. **Event Bus System**:
   * Centralized publisher-subscriber pattern implementation in `EventBus` class
   * Priority-based event handling with queue
   * Thread-safe event queue management
   * Asynchronous event processing in worker thread

3. **Component Management**:
   * Dependency injection for component references
   * Lazy initialization for resource efficiency
   * Dynamic component loading/unloading based on state
   * Thread pool execution with optimized worker count

4. **Hardware Optimization**:
   * Dynamic configuration based on detected hardware (in `HardwareManager`)
   * Specific optimizations for Ryzen 9 5900X:
     * Thread affinity targeting first 8 cores
     * Process priority management for responsiveness
     * Memory allocation tuned for 32GB systems
   * Specific optimizations for RTX 3080:
     * Precision selection (float16) for tensor cores
     * VRAM management with 32 GPU layers
     * Dynamic layer adjustment under memory pressure

### Core Classes and Relationships

1. **MaggieAI** (`maggie/core/app.py`):
   * Central control class managing the FSM
   * Coordinates component interactions
   * Manages system lifecycle
   * Handles state transitions with `_transition_to()`

2. **EventBus** (`maggie/core/app.py`):
   * Implements publisher-subscriber pattern
   * Enables decoupled component communication
   * Handles event prioritization and distribution
   * Manages asynchronous event processing

3. **HardwareManager** (`maggie/utils/hardware/manager.py`):
   * Detects and analyzes system capabilities
   * Creates optimization profiles
   * Monitors resource utilization
   * Provides configuration recommendations

4. **ConfigManager** (`maggie/utils/config/manager.py`):
   * Manages configuration loading and validation
   * Implements configuration recovery mechanisms
   * Applies hardware-specific optimizations
   * Ensures configuration persistence

5. **ExtensionBase** (`maggie/extensions/base.py`):
   * Abstract base class for all extensions
   * Defines standard extension interface
   * Implements common extension behaviors
   * Provides state management for extensions

### Developing Custom Extensions

To extend Maggie with custom extensions, follow these steps:

1. **Create Extension Directory Structure**:
   ```
   maggie/extensions/my_extension/
   ├── __init__.py
   ├── my_extension.py
   ├── config.py
   └── requirements.txt
   ```

2. **Use Extension Manager to Create Boilerplate**:
   ```bash
   # While in the Maggie directory with venv activated
   python -m scripts.extension_manager create my_extension
   ```

3. **Implement Extension Class**:
   ```python
   from maggie.extensions.base import ExtensionBase
   
   class MyCustomExtension(ExtensionBase):
       """
       Custom extension implementation.
       
       Parameters
       ----------
       event_bus : EventBus
           Reference to the central event bus
       config : Dict[str, Any]
           Configuration dictionary
       """
       
       def __init__(self, event_bus, config: Dict[str, Any]):
           """
           Initialize the custom extension.
           
           Parameters
           ----------
           event_bus : EventBus
               Reference to the central event bus
           config : Dict[str, Any]
               Configuration dictionary
           """
           super().__init__(event_bus, config)
           
           # Initialize custom attributes
           self.custom_attribute = config.get("custom_attribute", "default_value")
       
       def get_trigger(self) -> str:
           """
           Get the trigger phrase for this extension.
           
           Returns
           -------
           str
               Trigger phrase that activates this extension
           """
           return "custom command"
       
       def initialize(self) -> bool:
           """
           Initialize the extension.
           
           Returns
           -------
           bool
               True if initialization successful, False otherwise
           """
           if self._initialized:
               return True
               
           try:
               # Acquire component references
               self.speech_processor = self.get_service("speech_processor")
               self.llm_processor = self.get_service("llm_processor")
               
               if not self.speech_processor or not self.llm_processor:
                   logger.error("Failed to acquire required services")
                   return False
               
               # Custom initialization logic
               # ...
               
               self._initialized = True
               return True
           except Exception as e:
               logger.error(f"Error initializing {self.__class__.__name__}: {e}")
               return False
       
       def start(self) -> bool:
           """
           Start the extension workflow.
           
           Returns
           -------
           bool
               True if started successfully, False otherwise
           """
           try:
               # Reset state
               # Start workflow thread
               self._workflow_thread = threading.Thread(
                   target=self._workflow,
                   name=f"{self.__class__.__name__}Thread"
               )
               self._workflow_thread.daemon = True
               self._workflow_thread.start()
               
               self.running = True
               return True
           except Exception as e:
               logger.error(f"Error starting {self.__class__.__name__}: {e}")
               return False
       
       def stop(self) -> bool:
           """
           Stop the extension.
           
           Returns
           -------
           bool
               True if stopped successfully, False otherwise
           """
           self.running = False
           # Additional cleanup logic
           return True
       
       def process_command(self, command: str) -> bool:
           """
           Process a command directed to this extension.
           
           Parameters
           ----------
           command : str
               Command string to process
               
           Returns
           -------
           bool
               True if command processed, False otherwise
           """
           # Command processing logic
           return False
       
       def _workflow(self) -> None:
           """
           Main extension workflow.
           
           Implements the extension's core functionality.
           """
           try:
               # Extension-specific logic
               # ...
               
               # Signal completion
               self.event_bus.publish("extension_completed", self.__class__.__name__)
           except Exception as e:
               logger.error(f"Error in {self.__class__.__name__} workflow: {e}")
               self.event_bus.publish("extension_error", self.__class__.__name__)
           finally:
               self.running = False
   ```

4. **Configure Extension in config.yaml**:
   ```yaml
   extensions:
     my_extension:
       custom_attribute: "custom_value"
       output_dir: "custom_output"
       enabled: true
   ```

5. **Register Extension with Service Locator**:
   The extension will be automatically discovered and loaded if it follows the correct structure, as the `ExtensionRegistry` class will scan the extensions directory.

### Event-Driven Communication

Communication between components uses the event bus, following these principles:

1. **Event Publication**:
   ```python
   # Publish an event with data and optional priority
   self.event_bus.publish("event_type", data, priority=0)
   ```

2. **Event Subscription**:
   ```python
   # Subscribe to an event type with optional priority
   self.event_bus.subscribe("event_type", self._handle_event, priority=0)
   ```

3. **Event Handler Implementation**:
   ```python
   def _handle_event(self, data):
       """
       Handle an event from the event bus.
       
       Parameters
       ----------
       data : Any
           Event data payload
       """
       # Event handling logic
       pass
   ```

4. **Common Event Types**:
   * `"wake_word_detected"`: Wake word detected
   * `"command_detected"`: Speech command recognized
   * `"state_changed"`: FSM state transition (with StateTransition object)
   * `"extension_completed"`: Extension finished execution
   * `"extension_error"`: Error in extension execution
   * `"inactivity_timeout"`: Inactivity timer triggered

### Performance Optimization Guidelines

When developing extensions for Maggie, follow these optimization guidelines:

1. **Thread Management**:
   * Use the thread pool for background tasks
   * Limit concurrent threads to 8 for Ryzen 9 5900X
   * Make threads daemon to ensure proper cleanup
   * Use thread names for easier debugging
   * Use thread-safe communication via event bus

2. **GPU Optimization**:
   * Use float16 precision for Tensor Core acceleration
   * Implement automatic fallback to CPU for memory pressure
   * Explicitly release CUDA memory when possible:
     ```python
     import torch
     if torch.cuda.is_available():
         torch.cuda.empty_cache()
     ```
   * Use batched processing when appropriate

3. **Memory Management**:
   * Implement dynamic resource loading/unloading
   * Use caching for frequently accessed data
   * Monitor memory usage with explicit thresholds
   * Release resources promptly when not needed

4. **Audio Processing**:
   * Use native audio APIs for low latency
   * Implement audio preprocessing for improved recognition
   * Consider caching for repetitive audio synthesis
   * Adjust sampling parameters based on CPU load

5. **State Management**:
   * Implement clean state transitions
   * Handle edge cases for interrupted flows
   * Provide clear user feedback during transitions
   * Use timeouts to prevent hanging states

### Debugging and Logging

Maggie uses the Loguru library for comprehensive logging:

1. **Log Levels**:
   * DEBUG: Detailed diagnostic information
   * INFO: General operational information
   * WARNING: Potential issues that don't prevent operation
   * ERROR: Errors that prevent specific operations
   * CRITICAL: Critical errors that may crash the application

2. **Logging Best Practices**:
   ```python
   from loguru import logger
   
   # Log at appropriate levels
   logger.debug("Detailed diagnostic information")
   logger.info("General operational information")
   logger.warning("Potential issue detected")
   logger.error(f"Error occurred: {str(e)}")
   logger.critical("Critical failure")
   
   # Use structured logging with context
   logger.bind(extension="my_custom_extension").info("Extension-specific log")
   
   # Log exceptions with traceback
   try:
       # Code that might raise an exception
       pass
   except Exception as e:
       logger.exception(f"Exception details: {e}")
   ```

3. **Log File Management**:
   * Logs are stored in the `logs/` directory
   * Files rotate at 10MB with 1-week retention
   * Console logs show INFO level by default
   * File logs capture DEBUG level for detailed diagnostics
   * Check logs/maggie.log for historical information

4. **Debugging Tools**:
   * Use `python main.py --debug` to enable detailed logging
   * Use `python main.py --verify` to check system configuration
   * Check the Error Log tab in the GUI for issues
   * Review log files in logs/ directory for persistent issues

5. **Performance Profiling**:
   * Monitor CPU usage with Task Manager or htop
   * Check GPU memory usage with nvidia-smi:
     ```bash
     nvidia-smi -l 2  # Real-time updates every 2 seconds
     ```
   * Use timestamps in logs to identify bottlenecks
   * Look for memory leaks with tools like memory_profiler