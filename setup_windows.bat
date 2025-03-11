@echo off
REM =====================================================
REM Maggie - Windows Setup Script
REM =====================================================
REM This script automates the installation of Maggie 
REM on Windows systems, optimized for AMD Ryzen 9 5900X and RTX 3080
REM =====================================================

REM Get the directory where the script is located and change to it
SET SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%
echo Working directory set to: %CD%

echo Maggie - Windows Setup
echo ===================================

REM Check for admin rights
echo Checking administrator privileges...
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Not running with administrator privileges
    echo Some features may not work correctly without admin rights
    echo Particularly: system-wide dependencies and GPU optimizations
    
    set /p continue_anyway=Continue anyway? (y/n): 
    if /i "%continue_anyway%" neq "y" (
        echo Setup canceled. Please restart with administrator privileges.
        pause
        exit /b 1
    )
    echo Continuing with limited privileges...
) else (
    echo Running with administrator privileges - full functionality available
)

REM Create directories
echo.
echo Creating required directories...
for %%d in (logs models models\tts models\tts\en_US-kathleen-medium recipes templates cache cache\tts) do (
    if not exist "%%d" (
        mkdir "%%d"
        echo Created directory: %%d
    ) else (
        echo Directory already exists: %%d
    )
)

REM Check Python version
echo.
echo Checking Python version...
python --version > pythonversion.txt 2>&1
type pythonversion.txt | find "Python 3.10" > nul
if errorlevel 1 (
    echo [ERROR] Python 3.10.x is required for Maggie
    echo Current version:
    type pythonversion.txt
    echo.
    echo Please install Python 3.10 from:
    echo https://www.python.org/downloads/release/python-31011/
    echo.
    echo During installation, make sure to check:
    echo - "Add Python to PATH"
    echo - "Install for all users"
    del pythonversion.txt
    pause
    exit /b 1
)
del pythonversion.txt
echo Python 3.10 verified successfully

REM Check CUDA support for RTX 3080
echo.
echo Checking NVIDIA GPU and CUDA support...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No CUDA-capable GPU detected')" > gpucheck.txt 2>&1
set CUDA_CHECK_ERROR=0
type gpucheck.txt | find "No module named" > nul
if not errorlevel 1 set CUDA_CHECK_ERROR=1
if %CUDA_CHECK_ERROR% == 1 (
    echo PyTorch not installed yet, will install with CUDA support
) else (
    type gpucheck.txt
    type gpucheck.txt | find "RTX 3080" > nul
    if not errorlevel 1 (
        echo RTX 3080 detected - Using optimal configuration
    )
)
del gpucheck.txt

REM Create and activate virtual environment
echo.
echo Setting up Python virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists
)

echo Activating virtual environment...
call .\venv\Scripts\activate.bat
echo Virtual environment activated

REM Upgrade pip and setuptools
echo.
echo Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel

REM Install PyTorch with CUDA support for RTX 3080
echo.
echo Installing PyTorch with CUDA 11.8 support (optimized for RTX 3080)...
pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

REM Install dependencies
echo.
echo Installing Maggie dependencies...
pip install -e .

REM Install GPU dependencies
echo.
echo Installing GPU-specific dependencies...
pip install -e ".[gpu]"

REM Create configuration file
echo.
echo Setting up configuration...
if not exist "config.yaml" (
    if exist "config.yaml.example" (
        copy config.yaml.example config.yaml
        echo Created config.yaml from example
        echo [IMPORTANT] You need to edit config.yaml to add your Picovoice access key
    ) else (
        echo Warning: config.yaml.example not found, cannot create config.yaml
    )
) else (
    echo config.yaml already exists
)

REM Model download section
echo.
echo =========================================
echo Would you like to download required models?
echo This includes Mistral 7B and TTS voice models
echo (May take significant time and bandwidth)
set /p download_models=Download models? (y/n): 

if /i "%download_models%"=="y" (
    REM Check git-lfs
    echo.
    echo Checking for Git LFS...
    git lfs --version >nul 2>&1
    if errorlevel 1 (
        echo Installing Git LFS...
        git lfs install
    ) else (
        echo Git LFS is available
    )
    
    REM Download Mistral model
    echo.
    echo Checking Mistral LLM model...
    set mistral_dir=models\mistral-7b-instruct-v0.3-GPTQ-4bit
    if not exist %mistral_dir% (
        echo Downloading Mistral 7B model... (this may take a while)
        git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ %mistral_dir%
    ) else (
        echo Mistral model directory already exists
    )
    
    REM Download TTS model
    echo.
    echo Checking TTS voice model...
    set voice_dir=models\tts\en_US-kathleen-medium
    set onnx_file=%voice_dir%\en_US-kathleen-medium.onnx
    set json_file=%voice_dir%\en_US-kathleen-medium.json
    
    if not exist %onnx_file% (
        echo Downloading Piper TTS voice model...
        powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx', '%onnx_file%')"
    ) else (
        echo Piper TTS ONNX file already exists
    )
    
    if not exist %json_file% (
        echo Downloading Piper TTS JSON config...
        powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json', '%json_file%')"
    ) else (
        echo Piper TTS JSON file already exists
    )
)

REM Create recipe template
echo.
echo Creating recipe template...
python main.py --create-template

REM System verification
echo.
echo Verifying system configuration...
python main.py --verify

REM Optimize configuration for Ryzen 9 5900X and RTX 3080
echo.
echo Optimizing system configuration for Ryzen 9 5900X and RTX 3080...
python main.py --optimize

REM Remind about Picovoice key
echo.
echo =========================================
echo IMPORTANT: You need a Picovoice access key for wake word detection.
echo.
echo 1. Visit https://console.picovoice.ai/ to create a free account
echo 2. Get your access key from the console
echo 3. Edit config.yaml and add your key in the wake_word.porcupine_access_key field
echo =========================================

REM Ask to start Maggie
echo.
set /p run_app=Would you like to start Maggie now? (y/n): 
if /i "%run_app%"=="y" (
    echo Starting Maggie...
    python main.py
) else (
    echo.
    echo To run Maggie later, use:
    echo .\venv\Scripts\activate.bat
    echo python main.py
)

REM Deactivate virtual environment
echo.
echo Deactivating virtual environment...
deactivate
echo Setup completed successfully
echo.
echo Thank you for installing Maggie AI Assistant!
pause

echo.
echo Script completed - press any key to close this window
pause > nul