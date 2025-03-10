# Maggie AI Assistant

## Overview
Maggie is a Python-based AI assistant implementing a Finite State Machine (FSM) architecture with event-driven state transitions and modular utility objects. The application is optimized for systems with AMD Ryzen 9 5900X CPU, NVIDIA GeForce RTX 3080 GPU, and 32GB of RAM running on Windows 11 Pro.

## System Architecture
- **Object-Oriented Design**: Modular components with well-defined interfaces
- **Event-Driven**: State transitions triggered by events
- **Finite State Machine**: Seven distinct states (Idle, Startup, Ready, Active, Busy, Cleanup, Shutdown)
- **Multi-Threaded**: Concurrent processing for responsive operation

## Core Features
- **Wake Word Detection**: Efficient detection of the wake word "Maggie" with minimal CPU usage
- **State Management**: Smart transitions between states with proper resource handling
- **Speech Processing**: Real-time audio transcription and text-to-speech capabilities
- **LLM Integration**: Local language model inference using Mistral 7B with GPU acceleration
- **GUI Interface**: Status indicators, logs, and control buttons
- **Recipe Creator Utility**: Create Microsoft Word documents from spoken recipes

## System Requirements
- **CPU**: AMD Ryzen 9 5900X (or similar 12-core processor)
- **GPU**: NVIDIA GeForce RTX 3080 with 10GB VRAM (or equivalent)
- **RAM**: 32GB DDR4-3200 (or faster)
- **OS**: Windows 11 Pro (64-bit)
- **Python**: Version 3.10.x ONLY (3.10.0 to 3.10.12)

## Python Version Compatibility

Maggie requires Python 3.10.x specifically. Python 3.11, 3.12, and 3.13 are NOT compatible due to:
1. CUDA and PyTorch integration dependencies
2. Version-specific dependencies in ML libraries
3. Breaking changes in newer Python versions that affect the codebase

If you have multiple Python versions installed, ensure you use Python 3.10 when creating your virtual environment.

## Documentation

- [Installation Guide](INSTALLATION.md) - Detailed setup instructions for Windows and Linux
- [User Tutorial](USER_TUTORIAL.md) - Step-by-step guide for using Maggie
- [Command Reference](COMMAND_REFERENCE.md) - Complete list of voice commands and functions

## Installation

For detailed installation instructions including prerequisites, model downloads, and troubleshooting:

1. See the [Installation Guide](INSTALLATION.md) for step-by-step setup on both Windows and Linux.
2. Follow the [User Tutorial](USER_TUTORIAL.md) to learn how to use Maggie's features.
3. Refer to the [Command Reference](COMMAND_REFERENCE.md) for a complete list of available commands.

### Required Downloads

Before installing, you'll need to download:

1. **Python version 3.10.x**
   - Windows:
    - Select the desired Python 3.10.x Installer from [python.org](https://www.python.org/downloads/windows/)
    - Or, use the direct link for the Python 3.10.11 Installer: [python-3.10.11-amd64.exe](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)
   - Download from [https://www.python.org/downloads/windows/](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)

2. **CUDA Toolkit 11.8**
   - [https://developer.nvidia.com/cuda-11-8-0-download-archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)

3. **cuDNN for CUDA 11.x**
   - [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

4. **Visual C++ Redistributable** (Windows only)
   - [https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

5. **Porcupine Wake Word Engine License**
   - Get a free access key from [Picovoice Console](https://console.picovoice.ai/)

6. **Mistral 7B Instruct Model (GPTQ 4-bit)**
   - Download from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ)

7. **Piper TTS Voice Models**
   - Download from [Hugging Face](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US)


### Quick Installation (Windows PowerShell)

```powershell
# Clone the repository
git clone https://github.com/yourusername/maggie.git
cd maggie

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install with GPU support
pip install -e ".[gpu]"

# Download models (example - see Installation Guide for details)
mkdir -p models/tts/en_US-kathleen-medium
# Download models manually from links above

# Configure your Picovoice access key
notepad config.yaml
```

### Quick Installation (Linux Bash)

```bash
# Clone the repository
git clone https://github.com/yourusername/maggie.git
cd maggie

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install with GPU support
pip install -e ".[gpu]"

# Download models (example - see Installation Guide for details)
mkdir -p models/tts/en_US-kathleen-medium
# Download models manually from links above

# Configure your Picovoice access key
nano config.yaml
```

## Core Commands
- **"Maggie"** - Wake word to activate the assistant
- **"New recipe"** - Start the recipe creator utility
- **"Sleep"** or **"Go to sleep"** - Put Maggie into idle state
- **"Shutdown"** or **"Turn off"** - Shut down the application

See the [Command Reference](COMMAND_REFERENCE.md) for a complete list of commands.

## Project Structure
```
maggie/
├── main.py                 # Entry point
├── maggie.py               # Main application class
├── config.yaml             # Configuration file
├── pyproject.toml          # Project metadata and dependencies
├── utils/                  # Utility modules
│   ├── config.py           # Configuration handler
│   ├── gui.py              # GUI implementation
│   ├── tts.py              # Text-to-speech module
│   ├── utility_base.py     # Base class for utilities
│   └── recipe_creator.py   # Recipe creator utility
├── models/                 # Model files
│   ├── tts/                # TTS model files
│   └── ...                 # LLM model files
├── logs/                   # Log files
└── recipes/                # Output directory for recipes
```

## State Transition Flow
1. **Idle** → **Startup**: Triggered by wake word detection
2. **Startup** → **Ready**: After initialization and welcome message
3. **Ready** → **Active**: Upon recognition of task command
4. **Active** → **Ready**: After task completion
5. **Ready** → **Idle**: After inactivity timeout
6. **Any State** → **Cleanup** → **Idle/Shutdown**: Upon sleep/shutdown commands

## Performance Optimizations
- Wake word detection optimized for <5% CPU usage
- LLM inference using GPU acceleration (GPTQ 4-bit quantization)
- Whisper model using float16 compute for RTX 3080
- Thread management optimized for Ryzen 9 5900X
- Memory management for 32GB system

## License
[MIT License](LICENSE)

## Acknowledgements
This project uses several open-source libraries and models:
- [Mistral AI](https://mistral.ai/) for the Mistral-7B model
- [Whisper](https://github.com/openai/whisper) from OpenAI
- [Picovoice](https://picovoice.ai/) for Porcupine wake word detection
- [Piper TTS](https://github.com/rhasspy/piper) for text-to-speech
m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download models
You'll need to download the following models:
- Mistral-7B-Instruct-v0.3-GPTQ-4bit for the LLM
- Piper TTS voice models
- Whisper model (will be downloaded automatically on first use)

Place the models in the appropriate directories as specified in the configuration file:
- LLM models → `models/` directory
- TTS models → `models/tts/` directory

### 5. Configure Porcupine wake word
You'll need to obtain an access key from Picovoice for the Porcupine wake word detection system. Visit [https://console.picovoice.ai/](https://console.picovoice.ai/) to create a free account and generate your access key. Once obtained, add your key to the configuration file under the `wake_word.porcupine_access_key` field.

## Usage

### Starting the application
```bash
python main.py
```

### Command line options
```bash
python main.py --config path/to/config.yaml --debug
```

- `--config`: Specify a custom configuration file path
- `--debug`: Enable debug logging

## Configuration
The configuration file (`config.yaml`) contains settings for all components:

```yaml
# Example configuration snippet
wake_word:
  sensitivity: 0.5
  
speech:
  whisper:
    model_size: "base"
    
llm:
  model_path: "models/mistral-7b-instruct-v0.3-GPTQ-4bit"
  
utilities:
  recipe_creator:
    output_dir: "recipes"
```

## Core Commands
- "Maggie" - Wake word to activate the assistant
- "New recipe" - Start the recipe creator utility
- "Sleep" or "Go to sleep" - Put Maggie into idle state
- "Shutdown" or "Turn off" - Shut down the application

## Project Structure
```
maggie/
├── main.py                 # Entry point
├── maggie.py               # Main application class
├── config.yaml             # Configuration file
├── utils/                  # Utility modules
│   ├── config.py           # Configuration handler
│   ├── gui.py              # GUI implementation
│   ├── tts.py              # Text-to-speech module
│   ├── utility_base.py     # Base class for utilities
│   └── recipe_creator.py   # Recipe creator utility
├── models/                 # Model files
│   ├── tts/                # TTS model files
│   └── ...                 # LLM model files
├── logs/                   # Log files
└── recipes/                # Output directory for recipes
```

## State Transition Flow
1. **Idle** → **Startup**: Triggered by wake word detection
2. **Startup** → **Ready**: After initialization and welcome message
3. **Ready** → **Active**: Upon recognition of task command
4. **Active** → **Ready**: After task completion
5. **Ready** → **Idle**: After inactivity timeout
6. **Any State** → **Cleanup** → **Idle/Shutdown**: Upon sleep/shutdown commands

## Performance Optimizations
- Wake word detection optimized for <5% CPU usage
- LLM inference using GPU acceleration (GPTQ 4-bit quantization)
- Whisper model using float16 compute for RTX 3080
- Thread management optimized for Ryzen 9 5900X
- Memory management for 32GB system

## Dependencies
For a complete list of dependencies, see `requirements.txt`. Key components include:
- pvporcupine: Wake word detection
- SpeechRecognition & faster-whisper: Speech recognition
- piper-tts: Text-to-speech
- ctransformers: LLM inference with GPU acceleration
- PyQt6: GUI framework
- python-docx: Word document processing
- transitions: State machine implementation

## License
[MIT License](LICENSE)

## Acknowledgements
This project uses several open-source libraries and models:
- [Mistral AI](https://mistral.ai/) for the Mistral-7B model
- [Whisper](https://github.com/openai/whisper) from OpenAI
- [Picovoice](https://picovoice.ai/) for Porcupine wake word detection
- [Piper TTS](https://github.com/rhasspy/piper) for text-to-speech
