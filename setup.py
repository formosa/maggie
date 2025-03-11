#!/usr/bin/env python3
"""
Maggie AI Assistant - Unified Setup Script
=========================================
Automated setup script for Maggie AI Assistant installation.

This script provides a unified entry point for installing and configuring 
the Maggie AI Assistant on Windows and Linux systems. It automatically 
detects the operating system, generates the appropriate platform-specific
installation script, and guides the user through the installation process.

Optimized for:
- Windows 11 Pro with PowerShell or Command Prompt
- Linux (Ubuntu/Debian-based distributions)
- AMD Ryzen 9 5900X CPU
- NVIDIA RTX 3080 GPU
"""

import os
import sys
import platform
import subprocess
import shutil
import argparse
from typing import Dict, Any, List, Tuple, Optional


class SetupManager:
    """
    Manages the installation and setup process for Maggie AI Assistant.
    
    This class provides functionality for detecting the system platform,
    generating appropriate installation scripts, and executing the
    installation process with user guidance.
    
    Attributes
    ----------
    args : argparse.Namespace
        Command-line arguments parsed from user input
    platform : str
        Detected platform ('windows' or 'linux')
    use_powershell : bool
        Whether to use PowerShell on Windows
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the setup manager.
        
        Parameters
        ----------
        args : argparse.Namespace
            Command-line arguments
        """
        self.args = args
        self.platform = self._detect_platform()
        self.use_powershell = self._check_powershell() if self.platform == 'windows' else False
        
    def _detect_platform(self) -> str:
        """
        Detect the current operating system platform.
        
        Returns
        -------
        str
            'windows' or 'linux'
            
        Raises
        ------
        RuntimeError
            If the platform is not supported
        """
        system = platform.system().lower()
        if system == 'windows':
            print("[INFO] Windows platform detected")
            return 'windows'
        elif system == 'linux':
            print("[INFO] Linux platform detected")
            return 'linux'
        else:
            print("[ERROR] Unsupported platform:", system)
            print("[ERROR] This setup script supports Windows and Linux only")
            raise RuntimeError(f"Unsupported platform: {system}")
    
    def _check_powershell(self) -> bool:
        """
        Check if PowerShell is available on Windows.
        
        Returns
        -------
        bool
            True if PowerShell is available, False otherwise
        """
        try:
            subprocess.run(
                ['powershell', '-Command', 'echo "PowerShell available"'],
                capture_output=True, text=True, check=True
            )
            print("[INFO] PowerShell is available")
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("[INFO] PowerShell is not available, will use Command Prompt")
            return False
    
    def _check_python_version(self) -> bool:
        """
        Check if the current Python version is 3.10.x.
        
        Returns
        -------
        bool
            True if Python version is 3.10.x, False otherwise
        """
        version = platform.python_version_tuple()
        if int(version[0]) != 3 or int(version[1]) != 10:
            print(f"[WARNING] Unsupported Python version: {platform.python_version()}")
            print("[WARNING] Maggie requires Python 3.10.x specifically")
            
            if self.args.force:
                print("[WARNING] Continuing anyway due to --force flag")
                return True
            
            return False
        
        print(f"[INFO] Compatible Python version: {platform.python_version()}")
        return True
    
    def _check_gpu(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for NVIDIA GPU and CUDA support.
        
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            Boolean indicating GPU availability and info dictionary
        """
        gpu_info = {"available": False, "name": None, "cuda": None, "rtx_3080": False}
        
        try:
            # Try to import torch to check CUDA
            import torch
            
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["name"] = torch.cuda.get_device_name(0)
                gpu_info["cuda"] = torch.version.cuda
                gpu_info["rtx_3080"] = "3080" in gpu_info["name"]
                
                print(f"[INFO] GPU detected: {gpu_info['name']}")
                print(f"[INFO] CUDA version: {gpu_info['cuda']}")
                
                if gpu_info["rtx_3080"]:
                    print("[INFO] NVIDIA RTX 3080 detected - optimal hardware for Maggie")
                
                return True, gpu_info
            else:
                print("[WARNING] CUDA not available, GPU acceleration disabled")
                print("[WARNING] Performance may be significantly reduced without GPU acceleration")
                
                if self.args.force:
                    print("[WARNING] Continuing anyway due to --force flag")
                    return True, gpu_info
                
                return False, gpu_info
                
        except ImportError:
            print("[WARNING] PyTorch not installed, cannot detect GPU")
            print("[WARNING] Will try to install PyTorch with CUDA support")
            
            # Check for NVIDIA GPU without PyTorch
            try:
                if self.platform == 'windows':
                    output = subprocess.check_output('wmic path win32_VideoController get name', shell=True).decode()
                    if 'NVIDIA' in output:
                        gpu_info["available"] = True
                        if '3080' in output:
                            gpu_info["rtx_3080"] = True
                            gpu_info["name"] = "NVIDIA RTX 3080"
                            print("[INFO] NVIDIA RTX 3080 detected from system query")
                        else:
                            print("[INFO] NVIDIA GPU detected from system query")
                        return True, gpu_info
                elif self.platform == 'linux':
                    output = subprocess.check_output('lspci | grep -i nvidia', shell=True).decode()
                    if output:
                        gpu_info["available"] = True
                        if '3080' in output.lower():
                            gpu_info["rtx_3080"] = True
                            gpu_info["name"] = "NVIDIA RTX 3080"
                            print("[INFO] NVIDIA RTX 3080 detected from system query")
                        else:
                            print("[INFO] NVIDIA GPU detected from system query")
                        return True, gpu_info
            except subprocess.SubprocessError:
                pass
            
            print("[WARNING] No NVIDIA GPU detected, continuing without GPU acceleration")
            return True, gpu_info
    
    def generate_windows_batch_script(self) -> str:
        """
        Generate a Windows batch script for installation.
        
        Returns
        -------
        str
            Path to the generated script
        """
        script_path = "setup.bat"
        with open(script_path, 'w') as f:
            f.write("@echo off\n")
            f.write("echo Maggie AI Assistant - Windows Setup\n")
            f.write("echo ===================================\n\n")
            
            # Administrator check
            f.write(":: Check for admin rights\n")
            f.write("net session >nul 2>&1\n")
            f.write("if %errorLevel% neq 0 (\n")
            f.write("    echo [ERROR] This script requires administrator privileges\n")
            f.write("    echo Please right-click and select \"Run as administrator\"\n")
            f.write("    pause\n")
            f.write("    exit /b 1\n")
            f.write(")\n\n")
            
            # Create directories
            f.write(":: Create directories\n")
            for directory in ["logs", "models", "models\\tts", "models\\tts\\en_US-kathleen-medium", "recipes", "templates"]:
                f.write(f"if not exist \"{directory}\" (\n")
                f.write(f"    mkdir \"{directory}\"\n")
                f.write(f"    echo Created directory: {directory}\n")
                f.write(") else (\n")
                f.write(f"    echo Directory already exists: {directory}\n")
                f.write(")\n")
            f.write("\n")
            
            # Check Python version
            f.write(":: Check Python version\n")
            f.write("python --version > pythonversion.txt 2>&1\n")
            f.write("type pythonversion.txt | find \"Python 3.10\" > nul\n")
            f.write("if errorlevel 1 (\n")
            f.write("    echo [ERROR] Python 3.10.x is required\n")
            f.write("    echo Current version:\n")
            f.write("    type pythonversion.txt\n")
            f.write("    echo Please install Python 3.10 from https://www.python.org/downloads/release/python-31011/\n")
            f.write("    del pythonversion.txt\n")
            if not self.args.force:
                f.write("    pause\n")
                f.write("    exit /b 1\n")
            f.write(")\n")
            f.write("del pythonversion.txt\n\n")
            
            # Create and activate virtual environment
            f.write(":: Create and activate virtual environment\n")
            f.write("if not exist \"venv\" (\n")
            f.write("    echo Creating virtual environment...\n")
            f.write("    python -m venv venv\n")
            f.write(") else (\n")
            f.write("    echo Virtual environment already exists\n")
            f.write(")\n\n")
            
            f.write(":: Activate virtual environment\n")
            f.write("call .\\venv\\Scripts\\activate.bat\n")
            f.write("echo Virtual environment activated\n\n")
            
            # Install dependencies
            f.write(":: Install dependencies\n")
            f.write("echo Installing dependencies...\n")
            f.write("python -m pip install --upgrade pip\n")
            
            # Install PyTorch with CUDA
            f.write(":: Install PyTorch with CUDA support\n")
            f.write("pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118\n")
            
            # Install main dependencies
            f.write(":: Install main dependencies\n")
            f.write("pip install -e .\n")
            
            # Install GPU dependencies if available
            f.write(":: Install GPU dependencies\n")
            f.write("pip install -e \".[gpu]\"\n\n")
            
            # Create config from example if needed
            f.write(":: Create config from example if needed\n")
            f.write("if not exist \"config.yaml\" (\n")
            f.write("    if exist \"config.yaml.example\" (\n")
            f.write("        copy config.yaml.example config.yaml\n")
            f.write("        echo Created config.yaml from example\n")
            f.write("    ) else (\n")
            f.write("        echo Warning: config.yaml.example not found, cannot create config.yaml\n")
            f.write("    )\n")
            f.write(") else (\n")
            f.write("    echo config.yaml already exists\n")
            f.write(")\n\n")
            
            # Model download section
            f.write(":: Ask about model downloads\n")
            f.write("echo.\n")
            f.write("echo =========================================\n")
            f.write("echo Would you like to download required models?\n")
            f.write("echo This includes Mistral 7B and TTS voice models\n")
            f.write("echo (May take significant time and bandwidth)\n")
            f.write("set /p download_models=Download models? (y/n): \n")
            f.write("if /i \"%download_models%\"==\"y\" (\n")
            
            # Git LFS check
            f.write("    :: Check git-lfs\n")
            f.write("    git lfs --version >nul 2>&1\n")
            f.write("    if errorlevel 1 (\n")
            f.write("        echo Installing Git LFS...\n")
            f.write("        git lfs install\n")
            f.write("    ) else (\n")
            f.write("        echo Git LFS is available\n")
            f.write("    )\n")
            
            # Mistral model download
            f.write("    :: Download Mistral model\n")
            f.write("    set mistral_dir=models\\mistral-7b-instruct-v0.3-GPTQ-4bit\n")
            f.write("    if not exist %mistral_dir% (\n")
            f.write("        echo Downloading Mistral 7B model... (this may take a while)\n")
            f.write("        git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ %mistral_dir%\n")
            f.write("    ) else (\n")
            f.write("        echo Mistral model directory already exists\n")
            f.write("    )\n")
            
            # TTS model download
            f.write("    :: Download TTS model\n")
            f.write("    set voice_dir=models\\tts\\en_US-kathleen-medium\n")
            f.write("    set onnx_file=%voice_dir%\\en_US-kathleen-medium.onnx\n")
            f.write("    set json_file=%voice_dir%\\en_US-kathleen-medium.json\n")
            
            f.write("    if not exist %onnx_file% (\n")
            f.write("        echo Downloading Piper TTS voice model...\n")
            f.write("        powershell -Command \"(New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx', '%onnx_file%')\"\n")
            f.write("    ) else (\n")
            f.write("        echo Piper TTS ONNX file already exists\n")
            f.write("    )\n")
            
            f.write("    if not exist %json_file% (\n")
            f.write("        echo Downloading Piper TTS JSON config...\n")
            f.write("        powershell -Command \"(New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json', '%json_file%')\"\n")
            f.write("    ) else (\n")
            f.write("        echo Piper TTS JSON file already exists\n")
            f.write("    )\n")
            
            f.write(")\n\n")
            
            # Template creation
            f.write(":: Create recipe template\n")
            f.write("python main.py --create-template\n\n")
            
            # System verification
            f.write(":: Verify system\n")
            f.write("python main.py --verify\n\n")
            
            # Reminder about Picovoice access key
            f.write(":: Remind about Picovoice key\n")
            f.write("echo.\n")
            f.write("echo Don't forget to edit config.yaml with your Picovoice access key!\n")
            f.write("echo Get a free key at: https://console.picovoice.ai/\n")
            
            # Ask to start Maggie
            f.write(":: Ask to start Maggie\n")
            f.write("echo.\n")
            f.write("set /p run_app=Would you like to start Maggie now? (y/n): \n")
            f.write("if /i \"%run_app%\"==\"y\" (\n")
            f.write("    echo Starting Maggie...\n")
            f.write("    python main.py\n")
            f.write(") else (\n")
            f.write("    echo.\n")
            f.write("    echo To run Maggie later, use:\n")
            f.write("    echo .\\venv\\Scripts\\activate.bat\n")
            f.write("    echo python main.py\n")
            f.write(")\n\n")
            
            # Deactivate virtual environment
            f.write(":: Deactivate virtual environment\n")
            f.write("deactivate\n")
            f.write("echo Setup completed\n")
            f.write("pause\n")
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"[INFO] Generated Windows batch script: {script_path}")
        return script_path
    
    def generate_windows_powershell_script(self) -> str:
        """
        Generate a Windows PowerShell script for installation.
        
        Returns
        -------
        str
            Path to the generated script
        """
        script_path = "setup.ps1"
        with open(script_path, 'w') as f:
            f.write("# setup.ps1\n")
            f.write("# PowerShell setup script for Maggie AI Assistant on Windows\n\n")
            
            # Check administrator privileges
            f.write("# Check if running as administrator\n")
            f.write("$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)\n")
            f.write("if (-not $isAdmin) {\n")
            f.write("    Write-Host \"This script should be run as Administrator. Please restart with admin privileges.\" -ForegroundColor Red\n")
            if not self.args.force:
                f.write("    exit 1\n")
            f.write("}\n\n")
            
            # Create directories
            f.write("# Create directories\n")
            f.write("$dirs = @(\n")
            f.write("    \"logs\",\n")
            f.write("    \"models\",\n")
            f.write("    \"models/tts\",\n")
            f.write("    \"models/tts/en_US-kathleen-medium\",\n")
            f.write("    \"recipes\",\n")
            f.write("    \"templates\"\n")
            f.write(")\n\n")
            
            f.write("foreach ($dir in $dirs) {\n")
            f.write("    if (-not (Test-Path $dir)) {\n")
            f.write("        New-Item -Path $dir -ItemType Directory -Force | Out-Null\n")
            f.write("        Write-Host \"Created directory: $dir\" -ForegroundColor Green\n")
            f.write("    } else {\n")
            f.write("        Write-Host \"Directory already exists: $dir\" -ForegroundColor Yellow\n")
            f.write("    }\n")
            f.write("}\n\n")
            
            # Check specifically for Python 3.10
            f.write("# Check specifically for Python 3.10\n")
            f.write("try {\n")
            f.write("    $pythonVersion = python --version 2>&1\n")
            f.write("    if ($pythonVersion -match \"Python 3\\.10\\.\\d+\") {\n")
            f.write("        Write-Host \"Found $pythonVersion - Compatible version\" -ForegroundColor Green\n")
            f.write("    } else {\n")
            f.write("        Write-Host \"Incompatible Python version: $pythonVersion\" -ForegroundColor Red\n")
            f.write("        Write-Host \"Maggie requires Python 3.10.x specifically. Other versions will not work.\" -ForegroundColor Red\n")
            f.write("        Write-Host \"Please install Python 3.10 from https://www.python.org/downloads/release/python-31011/\" -ForegroundColor Red\n")
            if not self.args.force:
                f.write("        exit 1\n")
            f.write("    }\n")
            f.write("} catch {\n")
            f.write("    Write-Host \"Python not found. Please install Python 3.10.x from https://www.python.org/downloads/release/python-31011/\" -ForegroundColor Red\n")
            if not self.args.force:
                f.write("    exit 1\n")
            f.write("}\n\n")
            
            # Check for CUDA
            f.write("# Check for CUDA\n")
            f.write("try {\n")
            f.write("    $nvccVersion = nvcc --version 2>&1\n")
            f.write("    if ($nvccVersion -match \"release 11\\.8\") {\n")
            f.write("        Write-Host \"Found CUDA 11.8\" -ForegroundColor Green\n")
            f.write("    } else {\n")
            f.write("        Write-Host \"Warning: CUDA 11.8 is recommended, but found different version\" -ForegroundColor Yellow\n")
            f.write("    }\n")
            f.write("} catch {\n")
            f.write("    Write-Host \"CUDA not found. Please install CUDA 11.8 from https://developer.nvidia.com/cuda-11-8-0-download-archive\" -ForegroundColor Yellow\n")
            f.write("}\n\n")
            
            # Create and activate virtual environment
            f.write("# Create virtual environment\n")
            f.write("Write-Host \"Creating virtual environment...\" -ForegroundColor Cyan\n")
            f.write("if (Test-Path \"venv\") {\n")
            f.write("    Write-Host \"Virtual environment already exists\" -ForegroundColor Yellow\n")
            f.write("} else {\n")
            f.write("    python -m venv venv\n")
            f.write("    Write-Host \"Virtual environment created\" -ForegroundColor Green\n")
            f.write("}\n\n")
            
            f.write("# Activate virtual environment\n")
            f.write("Write-Host \"Activating virtual environment...\" -ForegroundColor Cyan\n")
            f.write("& \"./venv/Scripts/Activate.ps1\"\n\n")
            
            # Install dependencies
            f.write("# Install dependencies\n")
            f.write("Write-Host \"Installing dependencies...\" -ForegroundColor Cyan\n")
            f.write("pip install --upgrade pip\n")
            f.write("pip install -e \".[gpu]\"\n\n")
            
            # Create configuration file
            f.write("# Create example config\n")
            f.write("if (-not (Test-Path \"config.yaml\")) {\n")
            f.write("    if (Test-Path \"config.yaml.example\") {\n")
            f.write("        Copy-Item -Path \"config.yaml.example\" -Destination \"config.yaml\"\n")
            f.write("        Write-Host \"Created config.yaml from example\" -ForegroundColor Green\n")
            f.write("    } else {\n")
            f.write("        Write-Host \"Warning: config.yaml.example not found, cannot create config.yaml\" -ForegroundColor Yellow\n")
            f.write("    }\n")
            f.write("} else {\n")
            f.write("    Write-Host \"config.yaml already exists\" -ForegroundColor Yellow\n")
            f.write("}\n\n")
            
            # Model download section
            f.write("# Remind about model downloads\n")
            f.write("Write-Host \"`nReminder: You need to download the following models:\" -ForegroundColor Cyan\n")
            f.write("Write-Host \"1. Mistral 7B Instruct GPTQ model: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ\" -ForegroundColor Cyan\n")
            f.write("Write-Host \"2. Piper TTS voice model: https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US\" -ForegroundColor Cyan\n")
            f.write("Write-Host \"`nAnd don't forget to add your Picovoice access key in config.yaml!\" -ForegroundColor Cyan\n")
            f.write("Write-Host \"Get a free key at: https://console.picovoice.ai/\" -ForegroundColor Cyan\n\n")
            
            # Automated model downloads
            f.write("# Ask if user wants to run model downloads\n")
            f.write("$downloadModels = Read-Host \"Would you like to attempt automatic model downloads? (y/n)\"\n")
            f.write("if ($downloadModels -eq \"y\") {\n")
            
            # Git LFS check
            f.write("    # Check for git-lfs\n")
            f.write("    try {\n")
            f.write("        $gitLfsVersion = git lfs --version 2>&1\n")
            f.write("        Write-Host \"Found $gitLfsVersion\" -ForegroundColor Green\n")
            f.write("    } catch {\n")
            f.write("        Write-Host \"Installing Git LFS...\" -ForegroundColor Cyan\n")
            f.write("        git lfs install\n")
            f.write("    }\n")
            
            # Mistral model download
            f.write("    # Download Mistral model\n")
            f.write("    $mistralDir = \"models/mistral-7b-instruct-v0.3-GPTQ-4bit\"\n")
            f.write("    if (-not (Test-Path $mistralDir)) {\n")
            f.write("        Write-Host \"Downloading Mistral 7B model... (this may take a while)\" -ForegroundColor Cyan\n")
            f.write("        git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ $mistralDir\n")
            f.write("    } else {\n")
            f.write("        Write-Host \"Mistral model directory already exists\" -ForegroundColor Yellow\n")
            f.write("    }\n")
            
            # TTS model download
            f.write("    # Download Piper TTS voice\n")
            f.write("    $voiceDir = \"models/tts/en_US-kathleen-medium\"\n")
            f.write("    $onnxFile = \"$voiceDir/en_US-kathleen-medium.onnx\"\n")
            f.write("    $jsonFile = \"$voiceDir/en_US-kathleen-medium.json\"\n")
            
            f.write("    if (-not (Test-Path $onnxFile)) {\n")
            f.write("        Write-Host \"Downloading Piper TTS voice model...\" -ForegroundColor Cyan\n")
            f.write("        $onnxUrl = \"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx\"\n")
            f.write("        Invoke-WebRequest -Uri $onnxUrl -OutFile $onnxFile\n")
            f.write("    } else {\n")
            f.write("        Write-Host \"Piper TTS ONNX file already exists\" -ForegroundColor Yellow\n")
            f.write("    }\n")
            
            f.write("    if (-not (Test-Path $jsonFile)) {\n")
            f.write("        $jsonUrl = \"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json\"\n")
            f.write("        Invoke-WebRequest -Uri $jsonUrl -OutFile $jsonFile\n")
            f.write("    } else {\n")
            f.write("        Write-Host \"Piper TTS JSON file already exists\" -ForegroundColor Yellow\n")
            f.write("    }\n")
            f.write("}\n\n")
            
            # Create recipe template
            f.write("# Create recipe template\n")
            f.write("python main.py --create-template\n\n")
            
            # Verify system
            f.write("# Verify system configuration\n")
            f.write("python main.py --verify\n\n")
            
            # Remind about Picovoice key
            f.write("# Remind about Picovoice key\n")
            f.write("Write-Host \"`nDon't forget to edit config.yaml with your Picovoice access key!\" -ForegroundColor Magenta\n")
            f.write("Write-Host \"Get a free key at: https://console.picovoice.ai/\" -ForegroundColor Magenta\n\n")
            
            # Ask if user wants to run the application
            f.write("# Ask if user wants to run the application\n")
            f.write("$runApp = Read-Host \"Would you like to start Maggie now? (y/n)\"\n")
            f.write("if ($runApp -eq \"y\") {\n")
            f.write("    Write-Host \"Starting Maggie...\" -ForegroundColor Cyan\n")
            f.write("    python main.py\n")
            f.write("} else {\n")
            f.write("    Write-Host \"`nTo run Maggie later, use:\" -ForegroundColor Cyan\n")
            f.write("    Write-Host \"./venv/Scripts/Activate.ps1\" -ForegroundColor Cyan\n")
            f.write("    Write-Host \"python main.py\" -ForegroundColor Cyan\n")
            f.write("}\n")
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"[INFO] Generated Windows PowerShell script: {script_path}")
        return script_path
    
    def generate_linux_script(self) -> str:
        """
        Generate a Linux bash script for installation.
        
        Returns
        -------
        str
            Path to the generated script
        """
        script_path = "setup.sh"
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# setup.sh\n")
            f.write("# Bash setup script for Maggie AI Assistant on Linux\n\n")
            
            f.write("set -e\n\n")
            
            # Print with colors
            f.write("# Print with colors\n")
            f.write("RED='\\033[0;31m'\n")
            f.write("GREEN='\\033[0;32m'\n")
            f.write("YELLOW='\\033[0;33m'\n")
            f.write("CYAN='\\033[0;36m'\n")
            f.write("MAGENTA='\\033[0;35m'\n")
            f.write("NC='\\033[0m' # No Color\n\n")
            
            # Create directories
            f.write("# Create directories\n")
            f.write("dirs=(\n")
            f.write("    \"logs\"\n")
            f.write("    \"models\"\n")
            f.write("    \"models/tts\"\n")
            f.write("    \"models/tts/en_US-kathleen-medium\"\n")
            f.write("    \"recipes\"\n")
            f.write("    \"templates\"\n")
            f.write(")\n\n")
            
            f.write("for dir in \"${dirs[@]}\"; do\n")
            f.write("    if [ ! -d \"$dir\" ]; then\n")
            f.write("        mkdir -p \"$dir\"\n")
            f.write("        echo -e \"${GREEN}Created directory: $dir${NC}\"\n")
            f.write("    else\n")
            f.write("        echo -e \"${YELLOW}Directory already exists: $dir${NC}\"\n")
            f.write("    fi\n")
            f.write("done\n\n")
            
            # Check for Python 3.10 specifically
            f.write("# Check for Python 3.10 specifically\n")
            f.write("if command -v python3.10 &> /dev/null; then\n")
            f.write("    PYTHON_VERSION=$(python3.10 --version)\n")
            f.write("    if [[ $PYTHON_VERSION == *\"3.10.\"* ]]; then\n")
            f.write("        echo -e \"${GREEN}Found $PYTHON_VERSION - Compatible version${NC}\"\n")
            f.write("        PYTHON_CMD=python3.10\n")
            f.write("    else\n")
            f.write("        echo -e \"${RED}Unexpected version output: $PYTHON_VERSION${NC}\"\n")
            if not self.args.force:
                f.write("        exit 1\n")
            f.write("    fi\n")
            f.write("elif command -v python3 &> /dev/null; then\n")
            f.write("    PYTHON_VERSION=$(python3 --version)\n")
            f.write("    if [[ $PYTHON_VERSION == *\"3.10.\"* ]]; then\n")
            f.write("        echo -e \"${GREEN}Found $PYTHON_VERSION - Compatible version${NC}\"\n")
            f.write("        PYTHON_CMD=python3\n")
            f.write("    else\n")
            f.write("        echo -e \"${RED}Incompatible Python version: $PYTHON_VERSION${NC}\"\n")
            f.write("        echo -e \"${RED}Maggie requires Python 3.10.x specifically. Other versions will not work.${NC}\"\n")
            f.write("        echo -e \"${RED}Please install Python 3.10 and try again.${NC}\"\n")
            if not self.args.force:
                f.write("        exit 1\n")
            f.write("    fi\n")
            f.write("else\n")
            f.write("    echo -e \"${RED}Python 3.10 not found. Please install Python 3.10:${NC}\"\n")
            f.write("    echo -e \"${RED}sudo apt update && sudo apt install python3.10 python3.10-venv python3.10-dev${NC}\"\n")
            if not self.args.force:
                f.write("    exit 1\n")
            f.write("fi\n\n")
            
            # Check for CUDA
            f.write("# Check for CUDA\n")
            f.write("if command -v nvcc &> /dev/null; then\n")
            f.write("    CUDA_VERSION=$(nvcc --version | grep \"release\" | awk '{print $6}' | cut -c2-)\n")
            f.write("    echo -e \"${GREEN}Found CUDA $CUDA_VERSION${NC}\"\n")
            f.write("    if [[ \"$CUDA_VERSION\" != 11.8* ]]; then\n")
            f.write("        echo -e \"${YELLOW}Warning: CUDA 11.8 is recommended, but found $CUDA_VERSION${NC}\"\n")
            f.write("    fi\n")
            f.write("else\n")
            f.write("    echo -e \"${YELLOW}CUDA not found. Please install CUDA 11.8 for GPU acceleration:${NC}\"\n")
            f.write("    echo -e \"${YELLOW}https://developer.nvidia.com/cuda-11-8-0-download-archive${NC}\"\n")
            f.write("fi\n\n")
            
            # Check for audio dependencies
            f.write("# Check for audio dependencies\n")
            f.write("echo -e \"${CYAN}Checking for audio dependencies...${NC}\"\n")
            f.write("MISSING_DEPS=()\n\n")
            
            f.write("for pkg in portaudio19-dev libsndfile1 ffmpeg; do\n")
            f.write("    if ! dpkg -s $pkg &> /dev/null; then\n")
            f.write("        MISSING_DEPS+=($pkg)\n")
            f.write("    fi\n")
            f.write("done\n\n")
            
            f.write("if [ ${#MISSING_DEPS[@]} -ne 0 ]; then\n")
            f.write("    echo -e \"${YELLOW}Missing audio dependencies: ${MISSING_DEPS[*]}${NC}\"\n")
            f.write("    echo -e \"${YELLOW}Installing...${NC}\"\n")
            f.write("    sudo apt update && sudo apt install -y \"${MISSING_DEPS[@]}\"\n")
            f.write("else\n")
            f.write("    echo -e \"${GREEN}All audio dependencies are installed${NC}\"\n")
            f.write("fi\n\n")
            
            # Create virtual environment
            f.write("# Create virtual environment\n")
            f.write("echo -e \"${CYAN}Creating virtual environment...${NC}\"\n")
            f.write("if [ -d \"venv\" ]; then\n")
            f.write("    echo -e \"${YELLOW}Virtual environment already exists${NC}\"\n")
            f.write("else\n")
            f.write("    $PYTHON_CMD -m venv venv\n")
            f.write("    echo -e \"${GREEN}Virtual environment created${NC}\"\n")
            f.write("fi\n\n")
            
            # Activate virtual environment
            f.write("# Activate virtual environment\n")
            f.write("echo -e \"${CYAN}Activating virtual environment...${NC}\"\n")
            f.write("source venv/bin/activate\n\n")
            
            # Install dependencies
            f.write("# Install dependencies\n")
            f.write("echo -e \"${CYAN}Installing dependencies...${NC}\"\n")
            f.write("pip install --upgrade pip\n")
            f.write("pip install -e \".[gpu]\"\n\n")
            
            # Create config file from example
            f.write("# Create example config\n")
            f.write("if [ ! -f \"config.yaml\" ]; then\n")
            f.write("    if [ -f \"config.yaml.example\" ]; then\n")
            f.write("        cp config.yaml.example config.yaml\n")
            f.write("        echo -e \"${GREEN}Created config.yaml from example${NC}\"\n")
            f.write("    else\n")
            f.write("        echo -e \"${YELLOW}Warning: config.yaml.example not found, cannot create config.yaml${NC}\"\n")
            f.write("    fi\n")
            f.write("else\n")
            f.write("    echo -e \"${YELLOW}config.yaml already exists${NC}\"\n")
            f.write("fi\n\n")
            
            # Remind about model downloads
            f.write("# Remind about model downloads\n")
            f.write("echo -e \"\\n${GREEN}Setup completed!${NC}\"\n")
            f.write("echo -e \"\\n${CYAN}Reminder: You need to download the following models:${NC}\"\n")
            f.write("echo -e \"${CYAN}1. Mistral 7B Instruct GPTQ model: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ${NC}\"\n")
            f.write("echo -e \"${CYAN}2. Piper TTS voice model: https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US${NC}\"\n")
            f.write("echo -e \"\\n${CYAN}And don't forget to add your Picovoice access key in config.yaml!${NC}\"\n")
            f.write("echo -e \"${CYAN}Get a free key at: https://console.picovoice.ai/${NC}\"\n\n")
            
            # Ask if user wants to run model downloads
            f.write("# Ask if user wants to run model downloads\n")
            f.write("read -p \"Would you like to attempt automatic model downloads? (y/n) \" download_models\n")
            f.write("if [ \"$download_models\" = \"y\" ]; then\n")
            
            # Check for git-lfs
            f.write("    # Check for git-lfs\n")
            f.write("    if command -v git-lfs &> /dev/null; then\n")
            f.write("        echo -e \"${GREEN}Found $(git lfs --version)${NC}\"\n")
            f.write("    else\n")
            f.write("        echo -e \"${CYAN}Installing Git LFS...${NC}\"\n")
            f.write("        sudo apt install -y git-lfs\n")
            f.write("        git lfs install\n")
            f.write("    fi\n")
            
            # Download Mistral model
            f.write("    # Download Mistral model\n")
            f.write("    mistral_dir=\"models/mistral-7b-instruct-v0.3-GPTQ-4bit\"\n")
            f.write("    if [ ! -d \"$mistral_dir\" ]; then\n")
            f.write("        echo -e \"${CYAN}Downloading Mistral 7B model... (this may take a while)${NC}\"\n")
            f.write("        git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ \"$mistral_dir\"\n")
            f.write("    else\n")
            f.write("        echo -e \"${YELLOW}Mistral model directory already exists${NC}\"\n")
            f.write("    fi\n")
            
            # Download Piper TTS voice
            f.write("    # Download Piper TTS voice\n")
            f.write("    voice_dir=\"models/tts/en_US-kathleen-medium\"\n")
            f.write("    onnx_file=\"$voice_dir/en_US-kathleen-medium.onnx\"\n")
            f.write("    json_file=\"$voice_dir/en_US-kathleen-medium.json\"\n")
            
            f.write("    if [ ! -f \"$onnx_file\" ]; then\n")
            f.write("        echo -e \"${CYAN}Downloading Piper TTS ONNX model...${NC}\"\n")
            f.write("        onnx_url=\"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx\"\n")
            f.write("        wget -O \"$onnx_file\" \"$onnx_url\"\n")
            f.write("    else\n")
            f.write("        echo -e \"${YELLOW}Piper TTS ONNX file already exists${NC}\"\n")
            f.write("    fi\n")
            
            f.write("    if [ ! -f \"$json_file\" ]; then\n")
            f.write("        echo -e \"${CYAN}Downloading Piper TTS JSON config...${NC}\"\n")
            f.write("        json_url=\"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json\"\n")
            f.write("        wget -O \"$json_file\" \"$json_url\"\n")
            f.write("    else\n")
            f.write("        echo -e \"${YELLOW}Piper TTS JSON file already exists${NC}\"\n")
            f.write("    fi\n")
            f.write("fi\n\n")
            
            # Create recipe template
            f.write("# Create recipe template\n")
            f.write("python main.py --create-template\n\n")
            
            # Verify system
            f.write("# Verify system\n")
            f.write("python main.py --verify\n\n")
            
            # Remind about Picovoice key
            f.write("# Remind about Picovoice key\n")
            f.write("echo -e \"\\n${MAGENTA}Don't forget to edit config.yaml with your Picovoice access key!${NC}\"\n")
            f.write("echo -e \"${MAGENTA}Get a free key at: https://console.picovoice.ai/${NC}\"\n\n")
            
            # Ask if user wants to run the application
            f.write("# Ask if user wants to run the application\n")
            f.write("read -p \"Would you like to start Maggie now? (y/n) \" run_app\n")
            f.write("if [ \"$run_app\" = \"y\" ]; then\n")
            f.write("    echo -e \"${CYAN}Starting Maggie...${NC}\"\n")
            f.write("    python main.py\n")
            f.write("else\n")
            f.write("    echo -e \"\\n${CYAN}To run Maggie later, use:${NC}\"\n")
            f.write("    echo -e \"${CYAN}source venv/bin/activate${NC}\"\n")
            f.write("    echo -e \"${CYAN}python main.py${NC}\"\n")
            f.write("fi\n")
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"[INFO] Generated Linux script: {script_path}")
        return script_path
    
    def run(self) -> bool:
        """
        Run the setup process.
        
        Returns
        -------
        bool
            True if setup successful, False otherwise
        """
        # Check Python version
        if not self._check_python_version() and not self.args.force:
            print("[ERROR] Python version check failed")
            print("[INFO] Use --force to continue anyway")
            return False
        
        # Check GPU
        gpu_available, gpu_info = self._check_gpu()
        
        # Generate appropriate script
        if self.platform == 'windows':
            if self.use_powershell:
                script_path = self.generate_windows_powershell_script()
            else:
                script_path = self.generate_windows_batch_script()
        else:
            script_path = self.generate_linux_script()
        
        # Print instructions
        print("\n=== Setup Script Generated ===")
        print(f"Script location: {os.path.abspath(script_path)}")
        
        if self.platform == 'windows':
            if self.use_powershell:
                print("\nTo run the setup script:")
                print("1. Right-click on the script file")
                print("2. Select 'Run with PowerShell'")
                print("\nOr run from PowerShell:")
                print(f"  .\\{script_path}")
            else:
                print("\nTo run the setup script:")
                print(f"  {script_path}")
        else:
            print("\nTo run the setup script:")
            print("  chmod +x setup.sh")
            print("  ./setup.sh")
        
        # Ask if the user wants to run the script now
        if not self.args.no_execute:
            try:
                response = input("\nDo you want to run the setup script now? (y/n): ").strip().lower()
                if response == 'y':
                    print("\nRunning setup script...")
                    
                    if self.platform == 'windows':
                        if self.use_powershell:
                            subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', script_path])
                        else:
                            subprocess.run([script_path])
                    else:
                        subprocess.run(['bash', script_path])
                        
                    print("\nSetup script completed.")
                    return True
            except KeyboardInterrupt:
                print("\nSetup aborted.")
                return False
        
        return True


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Maggie AI Assistant Setup")
    parser.add_argument("--force", action="store_true", help="Continue setup even if requirements are not met")
    parser.add_argument("--no-execute", action="store_true", help="Generate scripts but don't run them")
    return parser.parse_args()


def main() -> int:
    """
    Main function for the setup script.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure)
    """
    print("=== Maggie AI Assistant Setup ===")
    print("This script will guide you through the installation process")
    
    args = parse_args()
    
    try:
        # Create setup manager and run setup
        setup_manager = SetupManager(args)
        if setup_manager.run():
            print("\nSetup complete. Run 'python main.py' to start Maggie.")
            return 0
        else:
            print("\nSetup failed. Please check the error messages above.")
            return 1
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())