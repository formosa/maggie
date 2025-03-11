"""
Maggie AI Assistant - Unified Installation Script
=============================================
Handles installation and setup of the Maggie AI Assistant.

This script provides a unified installation experience for
Windows and Linux, with specific optimizations for
AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.
"""

import os
import sys
import platform
import subprocess
import argparse
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

class MaggieInstaller:
    """
    Unified installer for Maggie AI Assistant.
    
    Provides platform-specific installation and configuration
    for Windows and Linux systems, with optimizations for
    AMD Ryzen 9 5900X and NVIDIA RTX 3080.
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to display verbose output, by default False
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the installer.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to display verbose output, by default False
        """
        self.verbose = verbose
        self.platform = platform.system()
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # Define colors for output
        if self.platform == "Windows":
            os.system("")  # Enable VT100 escape sequences on Windows
            
        # ANSI color codes
        self.colors = {
            "reset": "\033[0m",
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m"
        }
        
        # Required directories
        self.required_dirs = [
            "logs",
            "models",
            "models/tts",
            "models/tts/en_US-kathleen-medium",
            "recipes",
            "templates",
            "cache",
            "cache/tts"
        ]
        
        # Check if running as admin/root
        self.is_admin = self._check_admin()
        
    def _print(self, message: str, color: str = None):
        """
        Print a message with optional color.
        
        Parameters
        ----------
        message : str
            Message to print
        color : str, optional
            Color name, by default None
        """
        if color and color in self.colors:
            print(f"{self.colors[color]}{message}{self.colors['reset']}")
        else:
            print(message)
            
    def _check_admin(self) -> bool:
        """
        Check if running with administrative privileges.
        
        Returns
        -------
        bool
            True if running as admin/root, False otherwise
        """
        if self.platform == "Windows":
            try:
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            except:
                return False
        else:
            return os.geteuid() == 0
            
    def _run_command(self, cmd: List[str], check: bool = True, shell: bool = False) -> Tuple[int, str, str]:
        """
        Run a command and return the result.
        
        Parameters
        ----------
        cmd : List[str]
            Command to run
        check : bool, optional
            Whether to check for errors, by default True
        shell : bool, optional
            Whether to run as shell command, by default False
            
        Returns
        -------
        Tuple[int, str, str]
            Return code, stdout, stderr
        """
        if self.verbose:
            self._print(f"Running command: {' '.join(cmd)}", "cyan")
            
        try:
            process = subprocess.Popen(
                cmd if not shell else " ".join(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=shell,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if check and process.returncode != 0:
                self._print(f"Command failed: {' '.join(cmd)}", "red")
                self._print(f"Error: {stderr}", "red")
                
            return process.returncode, stdout, stderr
            
        except Exception as e:
            self._print(f"Error running command: {e}", "red")
            return -1, "", str(e)
            
    def _check_python_version(self) -> bool:
        """
        Check if Python version is 3.10.x.
        
        Returns
        -------
        bool
            True if Python version is compatible, False otherwise
        """
        version = platform.python_version_tuple()
        
        if int(version[0]) != 3 or int(version[1]) != 10:
            self._print(f"ERROR: Unsupported Python version: {platform.python_version()}", "red")
            self._print("Maggie requires Python 3.10.x specifically", "red")
            
            # Suggest installation instructions
            if self.platform == "Windows":
                self._print("Please install Python 3.10 from: https://www.python.org/downloads/release/python-31011/", "yellow")
            else:
                self._print("Please install Python 3.10 using:", "yellow")
                self._print("sudo apt update && sudo apt install python3.10 python3.10-venv python3.10-dev", "yellow")
                
            return False
            
        self._print(f"Found Python {platform.python_version()} - Compatible version", "green")
        return True
        
    def _check_gpu(self) -> Dict[str, Any]:
        """
        Check GPU availability and capabilities.
        
        Returns
        -------
        Dict[str, Any]
            GPU information
        """
        gpu_info = {
            "available": False,
            "name": None,
            "is_rtx_3080": False,
            "cuda_available": False,
            "cuda_version": None
        }
        
        try:
            # Check if PyTorch is installed
            import_returncode, _, _ = self._run_command(
                [sys.executable, "-c", "import torch; print('PyTorch available')"],
                check=False
            )
            
            if import_returncode != 0:
                self._print("PyTorch not installed yet, will install with CUDA support", "yellow")
                return gpu_info
                
            # Check CUDA availability
            returncode, stdout, _ = self._run_command([
                sys.executable, 
                "-c", 
                "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No CUDA-capable GPU detected')"
            ], check=False)
            
            if returncode != 0:
                self._print("Error checking GPU capabilities", "yellow")
                return gpu_info
                
            # Parse output
            for line in stdout.splitlines():
                if "CUDA available: True" in line:
                    gpu_info["available"] = True
                    gpu_info["cuda_available"] = True
                elif "GPU:" in line:
                    gpu_name = line.split("GPU:")[1].strip()
                    gpu_info["name"] = gpu_name
                    
                    if "3080" in gpu_name:
                        gpu_info["is_rtx_3080"] = True
                        self._print(f"RTX 3080 detected - Will use optimized settings", "green")
                    
            # Check CUDA version if available
            if gpu_info["cuda_available"]:
                returncode, stdout, _ = self._run_command([
                    sys.executable,
                    "-c",
                    "import torch; print(f'CUDA Version: {torch.version.cuda}')"
                ], check=False)
                
                if returncode == 0 and "CUDA Version:" in stdout:
                    gpu_info["cuda_version"] = stdout.split("CUDA Version:")[1].strip()
                    self._print(f"CUDA Version: {gpu_info['cuda_version']}", "green")
                    
                    # Check if CUDA 11.8 (optimal for RTX 3080)
                    if gpu_info["cuda_version"].startswith("11.8"):
                        self._print("Optimal CUDA version 11.8 detected", "green")
                    else:
                        self._print(f"Note: CUDA 11.8 recommended for best performance with RTX 3080", "yellow")
                        
        except Exception as e:
            self._print(f"Error checking GPU: {e}", "yellow")
            
        return gpu_info
        
    def _create_directories(self) -> bool:
        """
        Create required directories.
        
        Returns
        -------
        bool
            True if all directories created successfully, False otherwise
        """
        self._print("\nCreating required directories...", "cyan")
        
        success = True
        for directory in self.required_dirs:
            dir_path = os.path.join(self.base_dir, directory)
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    self._print(f"Created directory: {directory}", "green")
                except Exception as e:
                    self._print(f"Error creating directory {directory}: {e}", "red")
                    success = False
            else:
                self._print(f"Directory already exists: {directory}", "yellow")
                
        # Create maggie package directory structure for setup.py
        maggie_dir = os.path.join(self.base_dir, "maggie")
        if not os.path.exists(maggie_dir):
            try:
                os.makedirs(maggie_dir, exist_ok=True)
                # Create __init__.py in maggie directory
                with open(os.path.join(maggie_dir, "__init__.py"), "w") as f:
                    f.write("# Maggie AI Assistant package\n")
                self._print(f"Created package directory: maggie", "green")
            except Exception as e:
                self._print(f"Error creating package directory: {e}", "red")
                success = False
        else:
            self._print(f"Package directory already exists: maggie", "yellow")
            
        # Create maggie/utils directory for setup.py
        utils_dir = os.path.join(maggie_dir, "utils")
        if not os.path.exists(utils_dir):
            try:
                os.makedirs(utils_dir, exist_ok=True)
                # Create __init__.py in maggie/utils directory
                with open(os.path.join(utils_dir, "__init__.py"), "w") as f:
                    f.write("# Maggie AI Assistant utilities package\n")
                self._print(f"Created package directory: maggie/utils", "green")
            except Exception as e:
                self._print(f"Error creating package directory: {e}", "red")
                success = False
        else:
            self._print(f"Package directory already exists: maggie/utils", "yellow")
                
        return success
        
    def _setup_virtual_env(self) -> bool:
        """
        Set up virtual environment.
        
        Returns
        -------
        bool
            True if virtual environment created successfully, False otherwise
        """
        self._print("\nSetting up Python virtual environment...", "cyan")
        
        venv_dir = os.path.join(self.base_dir, "venv")
        
        # Check if venv already exists
        if os.path.exists(venv_dir):
            self._print("Virtual environment already exists", "yellow")
            return True
            
        # Create venv
        python_cmd = sys.executable
        returncode, stdout, stderr = self._run_command([python_cmd, "-m", "venv", venv_dir])
        
        if returncode != 0:
            self._print(f"Error creating virtual environment: {stderr}", "red")
            return False
            
        self._print("Virtual environment created successfully", "green")
        return True
        
    def _install_dependencies(self) -> bool:
        """
        Install dependencies in virtual environment.
        
        Returns
        -------
        bool
            True if dependencies installed successfully, False otherwise
        """
        self._print("\nInstalling dependencies...", "cyan")
        
        # Determine pip command based on platform
        if self.platform == "Windows":
            pip_cmd = os.path.join(self.base_dir, "venv", "Scripts", "pip")
            python_cmd = os.path.join(self.base_dir, "venv", "Scripts", "python")
        else:
            pip_cmd = os.path.join(self.base_dir, "venv", "bin", "pip")
            python_cmd = os.path.join(self.base_dir, "venv", "bin", "python")
            
        # Upgrade pip, setuptools, wheel
        self._print("Upgrading pip, setuptools, and wheel...", "cyan")
        returncode, _, _ = self._run_command([pip_cmd, "install", "--upgrade", "pip", "setuptools", "wheel"])
        
        if returncode != 0:
            self._print("Error upgrading pip, setuptools, and wheel", "red")
            return False
            
        # Install PyTorch with CUDA support
        self._print("Installing PyTorch with CUDA 11.8 support (optimized for RTX 3080)...", "cyan")
        returncode, _, _ = self._run_command([
            pip_cmd, "install", "torch==2.0.1+cu118", "--extra-index-url", "https://download.pytorch.org/whl/cu118"
        ])
        
        if returncode != 0:
            self._print("Error installing PyTorch with CUDA support", "red")
            return False
            
        # Install individual dependencies from requirements.txt directly
        self._print("Installing Maggie dependencies...", "cyan")
        
        # Create a temporary requirements file that excludes PyTorch (already installed)
        temp_req_path = os.path.join(self.base_dir, "temp_requirements.txt")
        try:
            with open(os.path.join(self.base_dir, "requirements.txt"), "r") as f:
                req_content = f.read()
            
            # Remove torch line and CUDA specific lines
            filtered_content = "\n".join([
                line for line in req_content.split("\n") 
                if not line.startswith("torch") and not "cu118" in line
            ])
            
            with open(temp_req_path, "w") as f:
                f.write(filtered_content)
                
            # Install from the temporary requirements file
            returncode, _, _ = self._run_command([pip_cmd, "install", "-r", temp_req_path])
            
            # Clean up
            os.remove(temp_req_path)
            
            if returncode != 0:
                self._print("Error installing Maggie dependencies", "red")
                return False
                
        except Exception as e:
            self._print(f"Error processing requirements file: {e}", "red")
            if os.path.exists(temp_req_path):
                os.remove(temp_req_path)
            return False
            
        # Install GPU-specific dependencies
        self._print("Installing GPU-specific dependencies...", "cyan")
        returncode, _, _ = self._run_command([pip_cmd, "install", "onnxruntime-gpu==1.15.1"])
        
        if returncode != 0:
            self._print("Error installing GPU-specific dependencies", "red")
            return False
            
        self._print("Dependencies installed successfully", "green")
        return True
        
    def _setup_config(self) -> bool:
        """
        Set up configuration file.
        
        Returns
        -------
        bool
            True if configuration set up successfully, False otherwise
        """
        self._print("\nSetting up configuration...", "cyan")
        
        config_path = os.path.join(self.base_dir, "config.yaml")
        example_path = os.path.join(self.base_dir, "config.yaml.example")
        
        # If example doesn't exist but we have config-yaml-example.txt
        alt_example_path = os.path.join(self.base_dir, "config-yaml-example.txt")
        if not os.path.exists(example_path) and os.path.exists(alt_example_path):
            try:
                shutil.copy(alt_example_path, example_path)
                self._print(f"Created config example from {alt_example_path}", "green")
            except Exception as e:
                self._print(f"Error creating config example: {e}", "red")
                return False
        
        # Check if config already exists
        if os.path.exists(config_path):
            self._print("Configuration file already exists", "yellow")
            return True
            
        # Check if example exists
        if not os.path.exists(example_path):
            self._print("Configuration example file not found", "red")
            return False
            
        # Copy example to config
        try:
            shutil.copy(example_path, config_path)
            self._print(f"Configuration file created from example", "green")
            self._print("NOTE: You need to edit config.yaml to add your Picovoice access key", "yellow")
            return True
        except Exception as e:
            self._print(f"Error creating configuration file: {e}", "red")
            return False
            
    def _download_models(self) -> bool:
        """
        Download required models.
        
        Returns
        -------
        bool
            True if models downloaded successfully, False otherwise
        """
        self._print("\nChecking model downloads...", "cyan")
        
        # Ask user if they want to download models
        response = input(f"{self.colors['magenta']}Download models? This may take significant time and bandwidth (y/n): {self.colors['reset']}")
        
        if response.lower() != "y":
            self._print("Skipping model downloads", "yellow")
            return True
            
        # Check for git-lfs
        self._print("Checking for Git LFS...", "cyan")
        returncode, _, _ = self._run_command(["git", "lfs", "version"], check=False)
        
        if returncode != 0:
            self._print("Installing Git LFS...", "cyan")
            if self.platform == "Windows":
                self._run_command(["git", "lfs", "install"], check=False)
            else:
                if self.is_admin:
                    self._run_command(["apt", "install", "-y", "git-lfs"], check=False)
                else:
                    self._run_command(["sudo", "apt", "install", "-y", "git-lfs"], check=False)
                self._run_command(["git", "lfs", "install"], check=False)
                
        # Download Mistral model
        mistral_dir = os.path.join(self.base_dir, "models", "mistral-7b-instruct-v0.3-GPTQ-4bit")
        if not os.path.exists(mistral_dir):
            self._print("Downloading Mistral 7B model... (this may take a while)", "cyan")
            returncode, _, _ = self._run_command([
                "git", "clone", "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ", mistral_dir
            ])
            
            if returncode != 0:
                self._print("Error downloading Mistral model", "red")
                return False
        else:
            self._print("Mistral model directory already exists", "yellow")
            
        # Download TTS voice model
        voice_dir = os.path.join(self.base_dir, "models", "tts", "en_US-kathleen-medium")
        onnx_file = os.path.join(voice_dir, "en_US-kathleen-medium.onnx")
        json_file = os.path.join(voice_dir, "en_US-kathleen-medium.json")
        
        if not os.path.exists(onnx_file):
            self._print("Downloading TTS ONNX model...", "cyan")
            if self.platform == "Windows":
                # Use PowerShell for Windows
                self._run_command([
                    "powershell", "-Command",
                    f"(New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx', '{onnx_file}')"
                ], shell=True)
            else:
                # Use wget for Linux
                self._run_command(["wget", "-O", onnx_file, "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx"])
        else:
            self._print("TTS ONNX file already exists", "yellow")
            
        if not os.path.exists(json_file):
            self._print("Downloading TTS JSON config...", "cyan")
            if self.platform == "Windows":
                # Use PowerShell for Windows
                self._run_command([
                    "powershell", "-Command",
                    f"(New-Object System.Net.WebClient).DownloadFile('https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json', '{json_file}')"
                ], shell=True)
            else:
                # Use wget for Linux
                self._run_command(["wget", "-O", json_file, "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json"])
        else:
            self._print("TTS JSON file already exists", "yellow")
            
        self._print("Model downloads completed", "green")
        return True
        
    def _create_recipe_template(self) -> bool:
        """
        Create recipe template.
        
        Returns
        -------
        bool
            True if template created successfully, False otherwise
        """
        self._print("\nCreating recipe template...", "cyan")
        
        # Determine python command based on platform
        if self.platform == "Windows":
            python_cmd = os.path.join(self.base_dir, "venv", "Scripts", "python")
        else:
            python_cmd = os.path.join(self.base_dir, "venv", "bin", "python")
            
        # Try to create the template
        try:
            # Create the template directory if it doesn't exist
            template_dir = os.path.join(self.base_dir, "templates")
            os.makedirs(template_dir, exist_ok=True)
            
            # Check if template already exists
            template_path = os.path.join(template_dir, "recipe_template.docx")
            if os.path.exists(template_path):
                self._print(f"Recipe template already exists: {template_path}", "yellow")
                return True
                
            # Try to create the template using docx
            try:
                from docx import Document
                
                doc = Document()
                doc.add_heading("Recipe Name", level=1)
                
                # Add metadata section
                doc.add_heading("Recipe Information", level=2)
                info_table = doc.add_table(rows=3, cols=2)
                info_table.style = 'Table Grid'
                info_table.cell(0, 0).text = "Preparation Time"
                info_table.cell(0, 1).text = "00 minutes"
                info_table.cell(1, 0).text = "Cooking Time"
                info_table.cell(1, 1).text = "00 minutes"
                info_table.cell(2, 0).text = "Servings"
                info_table.cell(2, 1).text = "0 servings"
                
                # Add ingredients section
                doc.add_heading("Ingredients", level=2)
                doc.add_paragraph("• Ingredient 1", style='ListBullet')
                doc.add_paragraph("• Ingredient 2", style='ListBullet')
                doc.add_paragraph("• Ingredient 3", style='ListBullet')
                
                # Add instructions section
                doc.add_heading("Instructions", level=2)
                doc.add_paragraph("1. Step 1", style='ListNumber')
                doc.add_paragraph("2. Step 2", style='ListNumber')
                doc.add_paragraph("3. Step 3", style='ListNumber')
                
                # Add notes section
                doc.add_heading("Notes", level=2)
                doc.add_paragraph("Add any additional notes, tips, or variations here.")
                
                # Save the template
                doc.save(template_path)
                self._print(f"Recipe template created at {template_path}", "green")
                return True
                
            except ImportError:
                # If python-docx is not available, try running main.py with --create-template
                returncode, _, _ = self._run_command([python_cmd, "main.py", "--create-template"])
                
                if returncode != 0:
                    self._print("Error creating recipe template using main.py", "red")
                    return False
                    
                self._print("Recipe template created successfully", "green")
                return True
                
        except Exception as e:
            self._print(f"Error creating recipe template: {e}", "red")
            return False
        
    def _optimize_config(self) -> bool:
        """
        Optimize configuration for detected hardware.
        
        Returns
        -------
        bool
            True if configuration optimized successfully, False otherwise
        """
        self._print("\nOptimizing configuration for detected hardware...", "cyan")
        
        # Determine python command based on platform
        if self.platform == "Windows":
            python_cmd = os.path.join(self.base_dir, "venv", "Scripts", "python")
        else:
            python_cmd = os.path.join(self.base_dir, "venv", "bin", "python")
            
        returncode, _, _ = self._run_command([python_cmd, "main.py", "--optimize"])
        
        if returncode != 0:
            self._print("Error optimizing configuration", "red")
            return False
            
        self._print("Configuration optimized for your hardware", "green")
        return True
        
    def _verify_system(self) -> bool:
        """
        Verify system configuration.
        
        Returns
        -------
        bool
            True if system verification passed, False otherwise
        """
        self._print("\nVerifying system configuration...", "cyan")
        
        # Determine python command based on platform
        if self.platform == "Windows":
            python_cmd = os.path.join(self.base_dir, "venv", "Scripts", "python")
        else:
            python_cmd = os.path.join(self.base_dir, "venv", "bin", "python")
            
        returncode, stdout, _ = self._run_command([python_cmd, "main.py", "--verify"])
        
        if returncode != 0:
            self._print("System verification failed", "red")
            return False
            
        self._print("System verification passed", "green")
        return True
        
    def install(self) -> bool:
        """
        Run the full installation process.
        
        Returns
        -------
        bool
            True if installation successful, False otherwise
        """
        self._print("=== Maggie AI Assistant Installation ===", "cyan")
        self._print(f"Platform: {self.platform}", "cyan")
        
        if not self.is_admin:
            if self.platform == "Windows":
                self._print("Note: Running without Administrator privileges", "yellow")
                self._print("Some optimizations may not be applied", "yellow")
            else:
                self._print("Note: Running without root privileges", "yellow")
                self._print("Some optimizations may not be applied", "yellow")
        
        # Check Python version
        if not self._check_python_version():
            return False
            
        # Check GPU
        gpu_info = self._check_gpu()
        
        # Create directories
        if not self._create_directories():
            return False
            
        # Setup virtual environment
        if not self._setup_virtual_env():
            return False
            
        # Install dependencies
        if not self._install_dependencies():
            return False
            
        # Setup configuration
        if not self._setup_config():
            return False
            
        # Download models
        if not self._download_models():
            return False
            
        # Create recipe template
        if not self._create_recipe_template():
            return False
            
        # Optimize configuration
        if not self._optimize_config():
            return False
            
        # Verify system
        if not self._verify_system():
            return False
            
        # Installation complete
        self._print("\n=== Installation Complete ===", "green")
        self._print("Reminders:", "cyan")
        self._print("1. Edit config.yaml to add your Picovoice access key from https://console.picovoice.ai/", "yellow")
        self._print("2. To run Maggie:", "yellow")
        
        if self.platform == "Windows":
            self._print("   .\venv\Scripts\activate", "cyan")
            self._print("   python main.py", "cyan")
        else:
            self._print("   source venv/bin/activate", "cyan")
            self._print("   python main.py", "cyan")
            
        # Ask if user wants to start Maggie
        response = input(f"\n{self.colors['magenta']}Would you like to start Maggie now? (y/n): {self.colors['reset']}")
        
        if response.lower() == "y":
            self._print("\nStarting Maggie...", "cyan")
            
            if self.platform == "Windows":
                python_cmd = os.path.join(self.base_dir, "venv", "Scripts", "python")
            else:
                python_cmd = os.path.join(self.base_dir, "venv", "bin", "python")
                
            self._run_command([python_cmd, "main.py"])
            
        return True


def main():
    """
    Main entry point for the installer.
    """
    parser = argparse.ArgumentParser(description="Maggie AI Assistant Installer")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    installer = MaggieInstaller(verbose=args.verbose)
    success = installer.install()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())