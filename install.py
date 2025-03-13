#!/usr/bin/env python3
"""
Maggie AI Assistant - Unified Installation Script
=============================================
Handles installation and setup of the Maggie AI Assistant.

This script provides a unified installation experience for
Windows and Linux, with specific optimizations for
AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.

The installation process includes:
1. System compatibility verification
2. Environment setup
3. Dependency installation
4. Model downloads
5. Configuration creation
6. System optimization

Example
-------
To install Maggie AI Assistant:
    $ python install.py

For verbose output:
    $ python install.py --verbose

To skip problematic packages:
    $ python install.py --skip-problematic-packages

Notes
-----
- Requires Python 3.10.x specifically
- Windows installation requires Visual C++ Build Tools and Windows SDK
- GPU acceleration requires NVIDIA GPU with CUDA support
- Optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080
"""

# Standard library imports
import os
import sys
import platform
import subprocess
import argparse
import shutil
import re
import urllib.request
import time
import zipfile
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
    skip_problematic : bool, optional
        Whether to skip problematic packages that may have compilation issues,
        by default False
    """
    
    def __init__(self, verbose: bool = False, skip_problematic: bool = False):
        """
        Initialize the installer.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to display verbose output, by default False
        skip_problematic : bool, optional
            Whether to skip problematic packages that may have compilation issues,
            by default False
        """
        self.verbose = verbose
        self.skip_problematic = skip_problematic
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
            "cache/tts",
            "downloads",  # Directory for downloaded files
            "downloads/wheels",  # Directory for wheel files
            "site-packages"  # Directory for manually installed packages
        ]
        
        # Check if running as admin/root
        self.is_admin = self._check_admin()
        
        # Flags for special handling
        self.has_cpp_compiler = False
        self.has_git = False
        self.has_windows_sdk = False
    
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
    
    def _download_file(self, url: str, destination: str) -> bool:
        """
        Download a file from a URL.
        
        Parameters
        ----------
        url : str
            URL to download from
        destination : str
            Path to save the file to
            
        Returns
        -------
        bool
            True if download was successful, False otherwise
        """
        try:
            self._print(f"Downloading {url}...", "cyan")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Try to download with progress reporting
            with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
                file_size = int(response.info().get('Content-Length', 0))
                downloaded = 0
                block_size = 1024 * 8  # 8KB blocks
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    # Show progress
                    if file_size > 0:
                        percent = int(downloaded * 100 / file_size)
                        sys.stdout.write(f"\rDownloaded: {downloaded} / {file_size} bytes ({percent}%)")
                        sys.stdout.flush()
            
            if file_size > 0:
                sys.stdout.write("\n")
                
            self._print(f"Download completed: {destination}", "green")
            return True
            
        except Exception as e:
            self._print(f"Error downloading file: {e}", "red")
            return False
    
    def _fix_windows_sdk_paths(self, sdk_path: str) -> bool:
        """
        Configure Windows SDK paths correctly for C++ compilation.
        
        Parameters
        ----------
        sdk_path : str
            Path to the Windows SDK Include directory
            
        Returns
        -------
        bool
            True if successful, False otherwise
            
        Notes
        -----
        Modifies environment variables to ensure Windows SDK headers 
        are properly detected during compilation. Critical for resolving
        the 'stddef.h' missing header issue on Windows 11.
        """
        try:
            self._print("Attempting to fix Windows SDK paths...", "cyan")
            
            # Find latest SDK version
            versions = []
            for item in os.listdir(sdk_path):
                if os.path.isdir(os.path.join(sdk_path, item)) and item.startswith("10."):
                    versions.append(item)
                    
            if not versions:
                self._print("No SDK versions found in Windows SDK directory", "red")
                return False
                
            # Sort versions to get the latest one
            versions.sort(key=lambda v: [int(x) for x in v.split('.')])
            latest_version = versions[-1]
            self._print(f"Using Windows SDK version {latest_version}", "green")
            
            # Set environment variables
            sdk_root = os.path.dirname(sdk_path)  # Parent of Include directory
            os.environ["WindowsSdkDir"] = sdk_root
            os.environ["WindowsSdkVersion"] = latest_version
            os.environ["UniversalCRTSdkDir"] = sdk_root
            
            # Add include paths to INCLUDE environment variable
            include_paths = []
            
            # Critical paths that need to be in INCLUDE
            for subdir in ["ucrt", "um", "shared", "cppwinrt"]:
                path = os.path.join(sdk_path, latest_version, subdir)
                if os.path.exists(path):
                    include_paths.append(path)
                    
            if not include_paths:
                self._print("No valid include paths found in Windows SDK", "red")
                return False
                
            # Get existing INCLUDE value and merge
            current_include = os.environ.get("INCLUDE", "")
            new_include = ";".join(include_paths)
            if current_include:
                new_include = new_include + ";" + current_include
                
            os.environ["INCLUDE"] = new_include
            
            # Check if stddef.h is now accessible
            stddef_found = False
            for path in include_paths:
                possible_stddef = os.path.join(path, "stddef.h")
                if os.path.exists(possible_stddef):
                    self._print(f"Successfully located stddef.h at {possible_stddef}", "green")
                    stddef_found = True
                    break
                    
            # Set additional environment variables that might help
            os.environ["UCRTVersion"] = latest_version
            os.environ["LIB"] = os.path.join(sdk_root, "Lib", latest_version, "ucrt", "x64") + ";" + os.environ.get("LIB", "")
            
            if stddef_found:
                return True
            else:
                self._print("stddef.h still not found after path configuration", "yellow")
                return False
                
        except Exception as e:
            self._print(f"Error fixing Windows SDK paths: {e}", "red")
            return False
            
    def _check_git(self) -> bool:
        """
        Check if Git is installed and in PATH.
        
        Returns
        -------
        bool
            True if Git is available, False otherwise
        """
        try:
            returncode, stdout, _ = self._run_command(["git", "--version"], check=False)
            
            if returncode == 0 and "git version" in stdout:
                self._print("Git found: " + stdout.strip(), "green")
                self.has_git = True
                return True
            
            # Git not found, provide instructions
            self._print("Git not found in PATH", "yellow")
            self._print("Some dependencies require Git for installation", "yellow")
            self._print("To install Git on Windows:", "yellow")
            self._print("1. Download from https://git-scm.com/download/win", "yellow")
            self._print("2. Install with default options", "yellow")
            self._print("3. Restart this installation after installing Git", "yellow")
            
            return False
            
        except Exception as e:
            self._print(f"Error checking for Git: {e}", "red")
            return False
    
    def _check_windows_sdk(self) -> bool:
        """
        Check if Windows SDK is properly installed and configured.
        
        Verifies that the Windows SDK is installed and includes the necessary
        header files for C++ compilation. Specifically checks for the stddef.h
        header file that is required for building C++ extensions.
        
        Returns
        -------
        bool
            True if Windows SDK is properly installed and configured, False otherwise
        """
        if self.platform != "Windows":
            return True  # Not relevant for non-Windows platforms
        
        try:
            # Check for Windows SDK installation
            sdk_paths = [
                "C:\\Program Files (x86)\\Windows Kits\\10\\Include",
                "C:\\Program Files\\Windows Kits\\10\\Include"
            ]
            
            sdk_installed = False
            sdk_path = None
            for path in sdk_paths:
                if os.path.exists(path):
                    sdk_installed = True
                    sdk_path = path
                    self._print(f"Found Windows SDK at {path}", "green")
                    break
            
            if not sdk_installed:
                self._print("Windows SDK not found", "yellow")
                self._print("Some packages may fail to build", "yellow")
                self._print("To install Windows SDK:", "yellow")
                self._print("1. Download from https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/", "yellow")
                self._print("2. Run the installer and select 'Windows SDK for Desktop C++ x64 Apps'", "yellow")
                self._print("3. Restart this installation after installing Windows SDK", "yellow")
                return False
            
            # Check for Universal CRT include directories
            ucrt_paths = [
                os.path.join(sdk_path, version, "ucrt") 
                for version in os.listdir(sdk_path)
                if os.path.isdir(os.path.join(sdk_path, version))
            ]
            
            stddef_found = False
            for ucrt_path in ucrt_paths:
                if os.path.exists(ucrt_path):
                    stddef_path = os.path.join(ucrt_path, "stddef.h")
                    if os.path.exists(stddef_path):
                        stddef_found = True
                        self._print(f"Found stddef.h in Windows SDK at {stddef_path}", "green")
                        self.has_windows_sdk = True
                        break
            
            if not stddef_found:
                # Attempt to fix Windows SDK paths automatically
                self._print("Required header file 'stddef.h' not found in expected location", "yellow")
                self._print("Attempting to fix Windows SDK configuration...", "cyan")
                
                if self._fix_windows_sdk_paths(sdk_path):
                    self._print("Successfully fixed Windows SDK paths", "green")
                    self.has_windows_sdk = True
                    return True
                else:
                    self._print("Failed to fix Windows SDK paths", "red")
                    self._print("This will cause problems when building C++ extensions", "yellow")
                    self._print("Possible solutions:", "yellow")
                    self._print("1. Reinstall Visual C++ Build Tools with the 'Desktop development with C++' workload", "yellow")
                    self._print("2. Ensure 'Windows 10 SDK' component is selected during installation", "yellow")
                    self._print("3. Use the --skip-problematic-packages flag to skip packages requiring compilation", "yellow")
                    return False
            
            return True
        
        except Exception as e:
            self._print(f"Error checking Windows SDK: {e}", "red")
            return False
    
    def _check_cpp_compiler(self) -> bool:
        """
        Check if C++ compiler is available for building wheels.
        
        Verifies that the necessary C++ compiler and build tools are available
        for building Python extensions from source. On Windows, checks for
        Visual C++ Build Tools and Windows SDK. On Linux, checks for GCC.
        
        Returns
        -------
        bool
            True if compiler is available, False otherwise
        """
        if self.platform == "Windows":
            # On Windows, check for Visual C++ Build Tools
            try:
                # First, check for cl.exe (MSVC compiler)
                returncode, _, _ = self._run_command(["where", "cl.exe"], check=False)
                if returncode == 0:
                    self._print("Visual C++ compiler (cl.exe) found", "green")
                    # Now check for Windows SDK
                    sdk_ok = self._check_windows_sdk()
                    self.has_cpp_compiler = returncode == 0 and sdk_ok
                    return self.has_cpp_compiler
                
                # Check if Visual Studio Build Tools are installed by checking registry
                returncode, stdout, _ = self._run_command(
                    ["reg", "query", "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\14.0\\VC\\Runtimes\\x64", "/v", "Version"],
                    check=False, shell=True
                )
                
                if returncode == 0:
                    self._print("Visual C++ Build Tools found in registry", "green")
                    # Now check for Windows SDK
                    sdk_ok = self._check_windows_sdk()
                    self.has_cpp_compiler = returncode == 0 and sdk_ok
                    return self.has_cpp_compiler
                
                # If we get here, compiler not found
                self._print("Visual C++ Build Tools not found", "yellow")
                self._print("Some packages may fail to build", "yellow")
                self._print("To install Visual C++ Build Tools:", "yellow")
                self._print("1. Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/", "yellow")
                self._print("2. Select 'Desktop development with C++' workload", "yellow")
                self._print("3. Ensure 'Windows 10 SDK' and 'MSVC C++ x64/x86 build tools' are selected", "yellow")
                self._print("4. Restart this installation after installing Build Tools", "yellow")
                
                return False
            except Exception as e:
                self._print(f"Error checking for Visual C++ compiler: {e}", "red")
                return False
        else:
            # On Linux, check for gcc/g++
            try:
                returncode, _, _ = self._run_command(["which", "gcc"], check=False)
                if returncode == 0:
                    self._print("GCC compiler found", "green")
                    self.has_cpp_compiler = True
                    return True
                
                self._print("GCC compiler not found. Some packages may fail to build.", "yellow")
                self._print("To install GCC on Ubuntu/Debian: sudo apt install build-essential", "yellow")
                return False
            except Exception as e:
                self._print(f"Error checking for GCC compiler: {e}", "red")
                return False
            
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
        
    def _install_basic_dependencies(self, python_cmd: str) -> bool:
        """
        Install basic dependencies needed for further installation.
        
        Parameters
        ----------
        python_cmd : str
            Path to Python executable in virtual environment
            
        Returns
        -------
        bool
            True if installation successful, False otherwise
        """
        self._print("Installing basic dependencies...", "cyan")
        
        # Upgrade pip, setuptools, and wheel using the proper method
        self._print("Upgrading pip, setuptools, and wheel...", "cyan")
        returncode, stdout, stderr = self._run_command([python_cmd, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        
        if returncode != 0:
            self._print(f"Error upgrading pip, setuptools, and wheel: {stderr}", "red")
            return False
        
        # Install requests for URL operations
        self._print("Installing requests package...", "cyan")
        returncode, _, stderr = self._run_command([python_cmd, "-m", "pip", "install", "requests"])
        
        if returncode != 0:
            self._print(f"Error installing requests package: {stderr}", "red")
            return False
            
        # Install other basic dependencies
        basic_deps = ["urllib3", "tqdm"]
        self._print(f"Installing basic packages: {', '.join(basic_deps)}...", "cyan")
        returncode, _, stderr = self._run_command([python_cmd, "-m", "pip", "install", "--upgrade"] + basic_deps)
        
        if returncode != 0:
            self._print(f"Error installing basic packages: {stderr}", "red")
            return False
            
        return True
        
    def _download_file_with_requests(self, url: str, destination: str) -> bool:
        """
        Download a file using the requests library with progress reporting.
        
        Parameters
        ----------
        url : str
            URL to download from
        destination : str
            Path to save the file to
            
        Returns
        -------
        bool
            True if download was successful, False otherwise
        """
        try:
            import requests
            from tqdm import tqdm
            
            self._print(f"Downloading {url}...", "cyan")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Create a streaming request to get content
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                # Use tqdm for progress bar if available
                with open(destination, 'wb') as f:
                    for chunk in tqdm(r.iter_content(chunk_size=8192), 
                                      total=total_size//8192, 
                                      unit='KB',
                                      desc=os.path.basename(destination)):
                        f.write(chunk)
                        
            self._print(f"Download completed: {destination}", "green")
            return True
            
        except ImportError:
            # Fall back to urllib if requests is not available
            return self._download_file(url, destination)
            
        except Exception as e:
            self._print(f"Error downloading file: {e}", "red")
            return False
        
    def _download_github_repo(self, repo_url: str, branch: str, dest_dir: str) -> bool:
        """
        Download a GitHub repository without using git.
        
        Parameters
        ----------
        repo_url : str
            GitHub repository URL (format: username/repository)
        branch : str
            Branch to download (usually "main" or "master")
        dest_dir : str
            Destination directory for the repository
            
        Returns
        -------
        bool
            True if download was successful, False otherwise
        """
        try:
            # Format the repository URL for the zip download
            if repo_url.startswith("https://github.com/"):
                repo_path = repo_url.replace("https://github.com/", "")
            else:
                repo_path = repo_url
                
            # Remove .git extension if present
            repo_path = repo_path.replace(".git", "")
            
            # Create the download URL
            download_url = f"https://github.com/{repo_path}/archive/refs/heads/{branch}.zip"
            
            # Download the zip file
            zip_path = os.path.join(self.base_dir, "downloads", f"{repo_path.replace('/', '_')}_{branch}.zip")
            
            if not self._download_file_with_requests(download_url, zip_path):
                return False
                
            # Extract the zip file
            self._print(f"Extracting {zip_path}...", "cyan")
            
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(dest_dir))
                
                # Get the name of the extracted directory (should be repo-branch)
                extracted_dir = os.path.join(os.path.dirname(dest_dir), 
                                            zip_ref.namelist()[0].split('/')[0])
                
            # Move the extracted directory to the destination
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
                
            shutil.move(extracted_dir, dest_dir)
            
            # Clean up the zip file
            os.remove(zip_path)
            
            self._print(f"Downloaded and extracted repository to {dest_dir}", "green")
            return True
            
        except Exception as e:
            self._print(f"Error downloading GitHub repository: {e}", "red")
            return False
    
    def _install_from_wheel_url(self, python_cmd: str, package_name: str, wheel_url: str) -> bool:
        """
        Install a package from a wheel URL.
        
        Parameters
        ----------
        python_cmd : str
            Path to Python executable in virtual environment
        package_name : str
            Name of the package
        wheel_url : str
            URL to the wheel file
            
        Returns
        -------
        bool
            True if installation successful, False otherwise
        """
        try:
            self._print(f"Installing {package_name} from wheel...", "cyan")
            
            # Download the wheel
            wheel_dir = os.path.join(self.base_dir, "downloads", "wheels")
            os.makedirs(wheel_dir, exist_ok=True)
            
            wheel_filename = os.path.basename(wheel_url)
            wheel_path = os.path.join(wheel_dir, wheel_filename)
            
            if not self._download_file_with_requests(wheel_url, wheel_path):
                return False
                
            # Install the wheel
            returncode, _, stderr = self._run_command([
                python_cmd, "-m", "pip", "install", wheel_path
            ])
            
            if returncode != 0:
                self._print(f"Error installing {package_name} from wheel: {stderr}", "red")
                return False
                
            self._print(f"Successfully installed {package_name} from wheel", "green")
            return True
            
        except Exception as e:
            self._print(f"Error installing {package_name} from wheel: {e}", "red")
            return False

    def _install_piper_phonemize_win11(self, python_cmd: str) -> bool:
        """
        Install piper_phonemize package specifically for Windows 11.
        
        Parameters
        ----------
        python_cmd : str
            Path to Python executable
        
        Returns
        -------
        bool
            True if installation successful, False otherwise
        
        Notes
        -----
        Uses pre-built wheels with additional compatibility fixes for Windows 11,
        avoiding compilation issues with stddef.h and other SDK dependencies.
        """
        try:
            self._print("Installing piper-phonemize using Windows 11 enhanced method...", "cyan")
            
            # Create wheel directory if it doesn't exist
            wheel_dir = os.path.join(self.base_dir, "downloads", "wheels")
            os.makedirs(wheel_dir, exist_ok=True)
            
            # Get Python version for wheel compatibility
            py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
            
            # Extended list of wheel URLs with additional fallbacks for Windows 11
            wheel_urls = [
                # Primary wheels
                f"https://github.com/rhasspy/piper-phonemize/releases/download/v1.2.0/piper_phonemize-1.2.0-{py_version}-{py_version}-win_amd64.whl",
                f"https://github.com/rhasspy/piper-phonemize/releases/download/v1.2.0/piper_phonemize-1.2.0-{py_version}-{py_version}m-win_amd64.whl",
                # Fixed fallbacks for Python 3.10 on Windows 11
                "https://github.com/rhasspy/piper-phonemize/releases/download/v1.2.0/piper_phonemize-1.2.0-cp310-cp310-win_amd64.whl",
                # Alternative mirror URLs
                "https://github.com/MycroftAI/piper-phonemize-wheels/raw/main/piper_phonemize-1.2.0-cp310-cp310-win_amd64.whl",
                # Try Python 3.9 wheel as last resort (might work with cp310 in some cases)
                "https://github.com/rhasspy/piper-phonemize/releases/download/v1.2.0/piper_phonemize-1.2.0-cp39-cp39-win_amd64.whl"
            ]
            
            for wheel_url in wheel_urls:
                wheel_filename = os.path.basename(wheel_url)
                wheel_path = os.path.join(wheel_dir, wheel_filename)
                
                self._print(f"Trying wheel from: {wheel_url}", "cyan")
                
                try:
                    # Download wheel
                    if not self._download_file_with_requests(wheel_url, wheel_path):
                        continue
                        
                    # Install wheel with --no-deps to avoid dependency resolution issues
                    returncode, _, stderr = self._run_command([
                        python_cmd, "-m", "pip", "install", wheel_path, "--no-deps"
                    ])
                    
                    if returncode == 0:
                        # Install dependencies separately
                        self._print("Installing piper-phonemize dependencies...", "cyan")
                        self._run_command([
                            python_cmd, "-m", "pip", "install", "numpy", "onnxruntime==1.15.1"
                        ])
                        
                        self._print("Successfully installed piper-phonemize from wheel", "green")
                        return True
                    else:
                        self._print(f"Error installing wheel: {stderr}", "yellow")
                except Exception as e:
                    self._print(f"Error with wheel {wheel_filename}: {e}", "yellow")
                    continue
            
            self._print("All Windows 11 compatible wheels failed to install", "yellow")
            return False
            
        except Exception as e:
            self._print(f"Error in Windows 11 piper-phonemize installation: {e}", "red")
            return False

    def _install_piper_phonemize_bin(self, python_cmd: str) -> bool:
        """
        Install pre-built piper-phonemize wheel.
        
        Attempts to download and install a pre-built wheel for piper-phonemize,
        trying multiple sources and Python version combinations for better compatibility.
        
        Parameters
        ----------
        python_cmd : str
            Path to Python executable in virtual environment
            
        Returns
        -------
        bool
            True if installation successful, False otherwise
        """
        try:
            self._print("Installing pre-built piper-phonemize wheel...", "cyan")
            
            # Get Python version info for wheel
            py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
            
            # Try multiple wheel sources and Python ABI tags for better reliability
            wheel_urls = [
                f"https://github.com/rhasspy/piper-phonemize/releases/download/v1.2.0/piper_phonemize-1.2.0-{py_version}-{py_version}-win_amd64.whl",
                f"https://github.com/rhasspy/piper-phonemize/releases/download/v1.2.0/piper_phonemize-1.2.0-{py_version}-{py_version}m-win_amd64.whl",
                # Try alternative mirrors if available
                "https://github.com/rhasspy/piper-phonemize/releases/download/v1.2.0/piper_phonemize-1.2.0-cp310-cp310-win_amd64.whl"
            ]
            
            for wheel_url in wheel_urls:
                # Download the wheel
                wheel_dir = os.path.join(self.base_dir, "downloads", "wheels")
                os.makedirs(wheel_dir, exist_ok=True)
                
                wheel_filename = os.path.basename(wheel_url)
                wheel_path = os.path.join(wheel_dir, wheel_filename)
                
                self._print(f"Trying wheel from: {wheel_url}", "cyan")
                
                try:
                    if self._download_file_with_requests(wheel_url, wheel_path):
                        # Install the wheel
                        returncode, _, stderr = self._run_command([
                            python_cmd, "-m", "pip", "install", wheel_path
                        ])
                        
                        if returncode == 0:
                            self._print("Successfully installed piper-phonemize from wheel", "green")
                            return True
                        else:
                            self._print(f"Error installing wheel: {stderr}", "red")
                except Exception as download_error:
                    self._print(f"Error with this wheel source: {download_error}", "yellow")
                    continue  # Try next URL
            
            self._print("All pre-built wheels failed to install", "yellow")
            return False
            
        except Exception as e:
            self._print(f"Error installing piper-phonemize from wheel: {e}", "red")
            return False

    def _install_piper_phonemize(self, python_cmd: str) -> bool:
        """
        Install piper-phonemize package using multiple fallback strategies.
        
        Attempts to install piper-phonemize using several methods:
        1. Windows 11 specific installation method (for Windows 11)
        2. Pre-built wheel (for supported platforms)
        3. Direct installation from GitHub (if Git is available)
        4. Installation from downloaded source
        5. Skipping if all methods fail
        
        Parameters
        ----------
        python_cmd : str
            Path to Python executable
            
        Returns
        -------
        bool
            True if installation successful or skipped, False otherwise
        """
        self._print("Installing piper-phonemize directly...", "cyan")
        
        # Check if we should skip due to --skip-problematic-packages flag
        if self.skip_problematic:
            self._print("Skipping piper-phonemize installation (--skip-problematic-packages flag enabled)", "yellow")
            return True
        
        # For Windows, try the enhanced Windows 11 installation method first
        if self.platform == "Windows":
            if self._install_piper_phonemize_win11(python_cmd):
                return True
        
        # Then try the regular pre-built wheel approach
        if self.platform == "Windows":
            if self._install_piper_phonemize_bin(python_cmd):
                return True
        
        # If wheel installation fails and we have Git, try direct installation
        if self.has_git:
            self._print("Installing piper-phonemize from GitHub (non-editable)...", "cyan")
            returncode, _, stderr = self._run_command([
                python_cmd, "-m", "pip", "install", "git+https://github.com/rhasspy/piper-phonemize.git"
            ])
            
            if returncode == 0:
                self._print("Successfully installed piper-phonemize from GitHub", "green")
                return True
            
            self._print(f"Error installing piper-phonemize from GitHub: {stderr}", "red")
        
        # If we get here, try downloading the repository and installing
        try:
            # Create temporary directory for the package
            package_dir = os.path.join(self.base_dir, "downloads", "piper-phonemize")
            
            # Download the repository
            if not self._download_github_repo("rhasspy/piper-phonemize", "master", package_dir):
                return False
            
            # Try to install in non-editable mode
            self._print("Installing piper-phonemize from sources (non-editable)...", "cyan")
            returncode, _, stderr = self._run_command([
                python_cmd, "-m", "pip", "install", package_dir
            ])
            
            if returncode == 0:
                self._print("Successfully installed piper-phonemize from sources", "green")
                return True
                
            self._print(f"Error installing piper-phonemize from sources: {stderr}", "red")
            
            # If all installation methods failed, ask user if they want to skip this dependency
            self._print("This package requires Visual C++ Build Tools and Windows SDK", "yellow")
            response = input(f"{self.colors['magenta']}All installation methods for piper-phonemize failed. Skip this dependency? (y/n): {self.colors['reset']}")
            
            if response.lower() == "y":
                self._print("Skipping piper-phonemize installation", "yellow")
                return True  # Return True to continue with installation
            
            return False
            
        except Exception as e:
            self._print(f"Error installing piper-phonemize: {e}", "red")
            
            # If installation fails due to exception, ask user if they want to skip
            response = input(f"{self.colors['magenta']}An error occurred while installing piper-phonemize. Skip this dependency? (y/n): {self.colors['reset']}")
            
            if response.lower() == "y":
                self._print("Skipping piper-phonemize installation", "yellow")
                return True  # Return True to continue with installation
            
            return False
            
    def _install_whisper_streaming(self, python_cmd: str) -> bool:
        """
        Install whisper-streaming package directly.
        
        Parameters
        ----------
        python_cmd : str
            Path to Python executable in virtual environment
            
        Returns
        -------
        bool
            True if installation successful, False otherwise
        """
        self._print("Installing whisper-streaming directly...", "cyan")
        
        try:
            # Create temporary directory for the package
            package_dir = os.path.join(self.base_dir, "downloads", "whisper-streaming")
            
            # Download the repository
            if not self._download_github_repo("ufal/whisper_streaming", "master", package_dir):
                return False
            
            # Check the repository structure
            if not os.path.exists(os.path.join(package_dir, "setup.py")) and \
               not os.path.exists(os.path.join(package_dir, "pyproject.toml")):
                # This isn't a standard Python package - inspect the structure
                self._print("Repository isn't a standard Python package", "yellow")
                
                # Look for Python files in the root
                python_files = [f for f in os.listdir(package_dir) if f.endswith('.py')]
                self._print(f"Found Python files: {', '.join(python_files)}", "cyan")
                
                # Create a site-packages directory
                site_packages_dir = os.path.join(self.base_dir, "site-packages")
                os.makedirs(site_packages_dir, exist_ok=True)
                
                # Copy all Python files to the site-packages directory
                for py_file in python_files:
                    shutil.copy(
                        os.path.join(package_dir, py_file),
                        os.path.join(site_packages_dir, py_file)
                    )
                    
                # Create __init__.py to make it a package
                with open(os.path.join(site_packages_dir, "__init__.py"), "w") as f:
                    f.write("# Whisper Streaming package\n")
                    
                # Add site-packages directory to Python path in venv by creating .pth file
                venv_site_packages = None
                
                if self.platform == "Windows":
                    venv_site_packages = os.path.join(self.base_dir, "venv", "Lib", "site-packages")
                else:
                    # Try to find site-packages directory for Linux
                    for lib_dir in ["lib", "lib64"]:
                        for py_dir in ["python3.10", "python3"]:
                            test_path = os.path.join(self.base_dir, "venv", lib_dir, py_dir, "site-packages")
                            if os.path.exists(test_path):
                                venv_site_packages = test_path
                                break
                        if venv_site_packages:
                            break
                            
                if venv_site_packages:
                    with open(os.path.join(venv_site_packages, "whisper_streaming.pth"), "w") as f:
                        f.write(site_packages_dir)
                        
                    self._print(f"Added {site_packages_dir} to Python path", "green")
                    
                    # Install any requirements from the repository
                    req_file = os.path.join(package_dir, "requirements.txt")
                    if os.path.exists(req_file):
                        self._print("Installing whisper-streaming requirements...", "cyan")
                        returncode, _, stderr = self._run_command([
                            python_cmd, "-m", "pip", "install", "-r", req_file
                        ])
                        
                        if returncode != 0:
                            self._print(f"Error installing whisper-streaming requirements: {stderr}", "red")
                            
                    self._print("Successfully set up whisper-streaming module", "green")
                    return True
                else:
                    self._print("Could not find site-packages directory in virtual environment", "red")
                    return False
            else:
                # Standard Python package - install it
                self._print("Installing whisper-streaming as a standard package...", "cyan")
                returncode, _, stderr = self._run_command([
                    python_cmd, "-m", "pip", "install", package_dir
                ])
                
                if returncode != 0:
                    self._print(f"Error installing whisper-streaming: {stderr}", "red")
                    return False
                    
                self._print("Successfully installed whisper-streaming", "green")
                return True
                
        except Exception as e:
            self._print(f"Error installing whisper-streaming: {e}", "red")
            return False
    
    def _find_prebuilt_wheel_urls(self) -> Dict[str, str]:
        """
        Find pre-built wheel URLs for required packages.
        
        Searches for pre-built wheels that match the current Python version
        and platform, favoring wheels that avoid compilation issues on Windows.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping package names to wheel URLs
        """
        wheel_urls = {}
        py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        platform_tag = "win_amd64" if self.platform == "Windows" else "linux_x86_64"
        
        try:
            import requests
            
            # 1. Find llama-cpp-python wheel
            self._print("Looking for pre-built llama-cpp-python wheel...", "cyan")
            
            # Try GitHub releases
            try:
                # For Windows with Python 3.10
                if self.platform == "Windows":
                    # Try multiple URLs based on different Python ABI tags
                    wheel_urls["llama-cpp-python"] = f"https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.11/llama_cpp_python-0.2.11-{py_version}-{py_version}-win_amd64.whl"
                    self._print(f"Found pre-built wheel for llama-cpp-python", "green")
                else:
                    # For Linux, search the releases
                    response = requests.get("https://api.github.com/repos/abetlen/llama-cpp-python/releases/tags/v0.2.11")
                    if response.status_code == 200:
                        release_data = response.json()
                        for asset in release_data.get("assets", []):
                            asset_name = asset["name"]
                            if (f"llama_cpp_python-0.2.11" in asset_name and 
                                f"{py_version}" in asset_name and 
                                f"{platform_tag}" in asset_name):
                                wheel_urls["llama-cpp-python"] = asset["browser_download_url"]
                                self._print(f"Found pre-built wheel for llama-cpp-python: {asset_name}", "green")
                                break
            except Exception as e:
                self._print(f"Error searching GitHub for llama-cpp-python wheels: {e}", "yellow")
            
            # 2. Add piper-phonemize wheel for Windows
            if self.platform == "Windows":
                # Try both ABI tags for better compatibility
                wheel_urls["piper-phonemize"] = f"https://github.com/rhasspy/piper-phonemize/releases/download/v1.2.0/piper_phonemize-1.2.0-{py_version}-{py_version}-win_amd64.whl"
                self._print("Found pre-built wheel for piper-phonemize", "green")
                
                # Add other difficult-to-compile packages for Windows
                wheel_urls["PyAudio"] = f"https://download.lfd.uci.edu/pythonlibs/archived/PyAudio-0.2.13-{py_version}-{py_version}-win_amd64.whl"
                self._print("Found pre-built wheel for PyAudio", "green")
            
        except ImportError:
            self._print("Could not import requests module for finding wheels", "yellow")
            
        return wheel_urls
    
    def _install_dependencies(self) -> bool:
        """
        Install dependencies in virtual environment.
        
        Installs all required dependencies, handling special cases for packages
        that require compilation or have pre-built wheels. Provides fallback
        mechanisms and options to skip problematic dependencies.
        
        Returns
        -------
        bool
            True if critical dependencies installed successfully, False otherwise
        """
        self._print("\nInstalling dependencies...", "cyan")
        
        # Check for Git first
        self._check_git()
        
        # Check for C++ compiler
        self._check_cpp_compiler()
        
        # Determine python command based on platform
        if self.platform == "Windows":
            python_cmd = os.path.join(self.base_dir, "venv", "Scripts", "python")
        else:
            python_cmd = os.path.join(self.base_dir, "venv", "bin", "python")
            
        # Upgrade pip, setuptools, wheel using proper method
        if not self._install_basic_dependencies(python_cmd):
            self._print("Warning: Failed to install basic dependencies, continuing anyway", "yellow")
        
        # Install PyTorch with CUDA support
        self._print("Installing PyTorch with CUDA 11.8 support (optimized for RTX 3080)...", "cyan")
        returncode, _, stderr = self._run_command([
            python_cmd, "-m", "pip", "install", "torch==2.0.1+cu118", "--extra-index-url", "https://download.pytorch.org/whl/cu118"
        ])
        
        if returncode != 0:
            self._print(f"Error installing PyTorch with CUDA support: {stderr}", "red")
            self._print("Continuing with installation, but GPU acceleration may not work", "yellow")
        
        # Get pre-built wheels if possible
        wheel_urls = self._find_prebuilt_wheel_urls()
        
        # Create a temporary requirements file that excludes special dependencies we'll install separately
        temp_req_path = os.path.join(self.base_dir, "temp_requirements.txt")
        try:
            with open(os.path.join(self.base_dir, "requirements.txt"), "r") as f:
                req_content = f.read()
            
            # Remove lines for dependencies that require special handling
            filtered_content = "\n".join([
                line for line in req_content.split("\n") 
                if line and not line.startswith("#") and  # Skip comments and empty lines
                not line.startswith("torch") and not "cu118" in line and
                not "whisper" in line.lower() and 
                not "piper" in line.lower() and
                not "llama-cpp-python" in line.lower() and
                not "PyAudio" in line.lower()  # PyAudio often needs special handling on Windows
            ])
            
            with open(temp_req_path, "w") as f:
                f.write(filtered_content)
                
            # Install from the temporary requirements file
            self._print("Installing standard dependencies...", "cyan")
            returncode, _, stderr = self._run_command([python_cmd, "-m", "pip", "install", "-r", temp_req_path])
            
            # Clean up
            os.remove(temp_req_path)
            
            if returncode != 0:
                self._print(f"Error installing standard dependencies: {stderr}", "red")
                self._print("Continuing with installation of critical components", "yellow")
            
            # Install specialized dependencies in the correct order
            
            # 1. Install piper-phonemize (dependency for piper-tts)
            piper_phonemize_installed = False
            
            # First try pre-built wheel if available for Windows
            if "piper-phonemize" in wheel_urls:
                piper_phonemize_installed = self._install_from_wheel_url(
                    python_cmd, "piper-phonemize", wheel_urls["piper-phonemize"]
                )
            
            # If wheel installation failed, try other methods
            if not piper_phonemize_installed:
                piper_phonemize_installed = self._install_piper_phonemize(python_cmd)
            
            # 2. Now install piper-tts after its dependency is installed or if user opted to skip
            if piper_phonemize_installed:
                self._print("Installing piper-tts...", "cyan")
                returncode, _, stderr = self._run_command([
                    python_cmd, "-m", "pip", "install", "piper-tts==1.2.0"
                ])
                
                if returncode != 0:
                    self._print(f"Error installing piper-tts: {stderr}", "red")
                    self._print("Continuing installation process", "yellow")
                    self._print("Note: Text-to-speech functionality will be limited", "yellow")
            else:
                self._print("Skipping piper-tts installation due to missing dependency", "yellow")
                self._print("Note: Text-to-speech functionality will be limited", "yellow")
                
            # 3. Install whisper-streaming using our custom method
            whisper_installed = self._install_whisper_streaming(python_cmd)
                
            if not whisper_installed:
                self._print("Warning: whisper-streaming installation failed", "yellow")
                self._print("Voice recognition may not work properly", "yellow")
                
            # 4. Install llama-cpp-python
            llama_installed = False
            
            # Try pre-built wheel first
            if "llama-cpp-python" in wheel_urls:
                llama_installed = self._install_from_wheel_url(
                    python_cmd, "llama-cpp-python", wheel_urls["llama-cpp-python"]
                )
            
            # If wheel installation failed or no wheel found, try source installation
            if not llama_installed:
                self._print("Installing llama-cpp-python...", "cyan")
                
                # If we have a compiler, we can try to build from source
                if self.has_cpp_compiler:
                    returncode, _, stderr = self._run_command([
                        python_cmd, "-m", "pip", "install", "llama-cpp-python==0.2.11", "--no-cache-dir"
                    ])
                    
                    if returncode == 0:
                        llama_installed = True
                        self._print("Successfully installed llama-cpp-python", "green")
                    else:
                        self._print(f"Error installing llama-cpp-python: {stderr}", "red")
                        self._print("You may need to install llama-cpp-python manually", "yellow")
                        self._print("Note: LLM functionality will be limited without this package", "yellow")
                else:
                    self._print("Cannot install llama-cpp-python without C++ compiler", "yellow")
                    self._print("Please install Visual C++ Build Tools", "yellow")
                    self._print("Note: LLM functionality will be limited without this package", "yellow")
            
            # 5. Install PyAudio if available as pre-built wheel
            if "PyAudio" in wheel_urls:
                self._print("Installing PyAudio from pre-built wheel...", "cyan")
                pyaudio_installed = self._install_from_wheel_url(
                    python_cmd, "PyAudio", wheel_urls["PyAudio"]
                )
                
                if not pyaudio_installed:
                    self._print("Error installing PyAudio from wheel", "red")
                    self._print("You may need to install PyAudio manually", "yellow")
                    self._print("Note: Audio input functionality will be limited", "yellow")
            
            # 6. Install GPU-specific dependencies
            self._print("Installing GPU-specific dependencies...", "cyan")
            returncode, _, stderr = self._run_command([python_cmd, "-m", "pip", "install", "onnxruntime-gpu==1.15.1"])
            
            if returncode != 0:
                self._print(f"Error installing GPU-specific dependencies: {stderr}", "red")
                self._print("Continuing with CPU-only operation", "yellow")
                
            self._print("Dependencies installed successfully", "green")
            return True
                
        except Exception as e:
            self._print(f"Error processing requirements file: {e}", "red")
            if os.path.exists(temp_req_path):
                os.remove(temp_req_path)
            return False
        
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
            
    def _download_models_direct(self) -> bool:
        """
        Download required models directly without using Git.
        
        Returns
        -------
        bool
            True if models downloaded successfully, False otherwise
        """
        # 1. Download TTS voice model
        self._print("Downloading TTS models...", "cyan")
        
        voice_dir = os.path.join(self.base_dir, "models", "tts", "en_US-kathleen-medium")
        onnx_file = os.path.join(voice_dir, "en_US-kathleen-medium.onnx")
        json_file = os.path.join(voice_dir, "en_US-kathleen-medium.json")
        
        os.makedirs(voice_dir, exist_ok=True)
        
        onnx_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.onnx"
        json_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/kathleen/medium/en_US-kathleen-medium.json"
        
        success = True
        
        if not os.path.exists(onnx_file):
            if not self._download_file_with_requests(onnx_url, onnx_file):
                success = False
        else:
            self._print("TTS ONNX file already exists", "yellow")
            
        if not os.path.exists(json_file):
            if not self._download_file_with_requests(json_url, json_file):
                success = False
        else:
            self._print("TTS JSON file already exists", "yellow")
            
        return success
            
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
        
        # Download TTS models directly (doesn't require Git)
        if not self._download_models_direct():
            self._print("Warning: Error downloading some models", "yellow")
            self._print("Some functionality may not work correctly", "yellow")
            
        # Download Mistral model - requires Git
        if self.has_git:
            # Check for git-lfs
            self._print("Checking for Git LFS...", "cyan")
            returncode, _, _ = self._run_command(["git", "lfs", "version"], check=False)
            
            if returncode != 0:
                self._print("Git LFS not found, installing...", "cyan")
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
        else:
            self._print("Git not found, skipping Mistral model download", "yellow")
            self._print("LLM functionality will not work without the model", "yellow")
            self._print("Please install Git and run the installation again", "yellow")
            
        self._print("Model downloads completed", "green")
        return True
        
    def _create_recipe_template(self) -> bool:
        """
        Create recipe template document.
        
        Creates a Microsoft Word document template for recipes if it doesn't exist.
        This template is used by the recipe creator utility to generate formatted recipe documents.
        
        Returns
        -------
        bool
            True if template created successfully or already exists, False otherwise
        
        Notes
        -----
        Uses python-docx if available, otherwise tries to run main.py with --create-template flag.
        """
        self._print("\nChecking recipe template...", "cyan")
        
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
                
                # Add metadata section
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
                
                # Add ingredients section with better formatting
                doc.add_heading("Ingredients", level=2)
                doc.add_paragraph("• Ingredient 1", style='ListBullet')
                doc.add_paragraph("• Ingredient 2", style='ListBullet')
                doc.add_paragraph("• Ingredient 3", style='ListBullet')
                
                # Add steps section with numbered steps
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
    
    def _windows_optimizations(self) -> None:
        """
        Apply Windows-specific optimizations.
        
        Sets process priority and CPU affinity for optimal performance
        on Windows systems, with special optimizations for Ryzen 9 5900X.
        This includes power plan settings and CPU/memory optimizations.
        """
        try:
            # Set power plan to high performance
            self._print("Setting power plan to High Performance...", "cyan")
            returncode, _, _ = self._run_command([
                "powercfg", "/s", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"  # High Performance power scheme GUID
            ], check=False, shell=True)
            
            # If the high performance plan isn't available, try the Ultimate Performance plan
            if returncode != 0:
                self._run_command([
                    "powercfg", "/s", "e9a42b02-d5df-448d-aa00-03f14749eb61"  # Ultimate Performance power scheme GUID
                ], check=False, shell=True)
                
            # Process priority and affinity optimizations
            import psutil
            # Get current process
            process = psutil.Process(os.getpid())
            
            # Set process priority to high
            process.nice(psutil.HIGH_PRIORITY_CLASS)
            self._print("Set process priority to high", "green")
            
            # CPU affinity optimization for Ryzen 9 5900X
            # Use the first 8 cores (16 threads for Ryzen)
            cpu_count = psutil.cpu_count(logical=True)
            if cpu_count >= 16:  # Likely Ryzen 9 with hyperthreading
                # Create affinity mask for first 8 physical cores (16 threads)
                # For Ryzen, logical processors are arranged as:
                # 0, 2, 4, ... are first 12 cores, 1, 3, 5, ... are their hyperthreaded pairs
                affinity = list(range(16))
                process.cpu_affinity(affinity)
                self._print(f"Set CPU affinity to first 8 physical cores", "green")
                
            # Configure Windows memory management for better performance
            self._run_command([
                "reg", "add", "HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management", 
                "/v", "LargeSystemCache", "/t", "REG_DWORD", "/d", "1", "/f"
            ], check=False, shell=True)
            
            # Disable system visual effects for better performance
            self._run_command([
                "reg", "add", "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\VisualEffects", 
                "/v", "VisualFXSetting", "/t", "REG_DWORD", "/d", "2", "/f"
            ], check=False, shell=True)
            
        except ImportError:
            self._print("psutil not available for Windows optimizations", "yellow")
        except Exception as e:
            self._print(f"Failed to apply Windows performance optimizations: {e}", "yellow")
    
    def _linux_optimizations(self) -> None:
        """
        Apply Linux-specific optimizations.
        
        Sets CPU governor and process nice level for optimal performance
        on Linux systems. Configures CPU frequency scaling and process priority,
        with specific optimizations for AMD Ryzen 9 5900X processors.
        """
        try:
            # Check if running as root (required for some optimizations)
            is_root = os.geteuid() == 0
            
            if is_root:
                # Set CPU governor to performance for all cores
                self._print("Setting CPU governor to performance mode...", "cyan")
                os.system("echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
                self._print("Set CPU governor to performance mode", "green")
                
                # Optimize CPU frequency scaling for Ryzen 9 5900X
                for cpu_dir in os.listdir("/sys/devices/system/cpu"):
                    if cpu_dir.startswith("cpu") and cpu_dir != "cpuidle" and cpu_dir != "cpufreq":
                        try:
                            # Set minimum frequency to 3.7 GHz (3700000 KHz) if available
                            min_freq_path = f"/sys/devices/system/cpu/{cpu_dir}/cpufreq/scaling_min_freq"
                            if os.path.exists(min_freq_path):
                                with open(min_freq_path, "w") as f:
                                    f.write("3700000")
                        except:
                            pass
                
                # Set I/O scheduler to deadline for better I/O performance
                for disk in os.listdir("/sys/block"):
                    if disk.startswith("sd") or disk.startswith("nvme"):
                        try:
                            scheduler_path = f"/sys/block/{disk}/queue/scheduler"
                            if os.path.exists(scheduler_path):
                                with open(scheduler_path, "w") as f:
                                    f.write("deadline")
                        except:
                            pass
            else:
                self._print("Not running as root, skipping some system optimizations", "yellow")
                
            # Set process nice level for better responsiveness
            os.nice(-10)  # Higher priority (lower nice value)
            self._print("Set process nice level to -10", "green")
            
            # Set GPU optimization if applicable
            if os.path.exists("/proc/driver/nvidia/params"):
                # Apply NVIDIA driver optimizations using nvidia-settings if available
                self._run_command([
                    "nvidia-settings", "-a", "[gpu:0]/GPUPowerMizerMode=1"
                ], check=False)
                
        except Exception as e:
            self._print(f"Failed to apply Linux performance optimizations: {e}", "yellow")
    
    def _common_optimizations(self) -> None:
        """
        Apply common optimizations for all platforms.
        
        Implements cross-platform optimizations including Python memory management,
        garbage collection tuning, and process naming for easier identification.
        Applies optimizations specifically beneficial for AI workloads.
        """
        try:
            # Set process name for better identification
            try:
                import setproctitle
                setproctitle.setproctitle("maggie-ai-assistant")
                self._print("Set process title to 'maggie-ai-assistant'", "green")
            except ImportError:
                pass
            
            # Python memory and GC optimizations
            try:
                import gc
                
                # Disable automatic garbage collection and run it manually when needed
                gc.disable()
                self._print("Disabled automatic garbage collection for better performance", "green")
                
                # Run initial collection to start with clean state
                gc.collect()
            except:
                pass
            
            # Optimize Python threading for better parallelism
            try:
                import threading
                threading.stack_size(256 * 1024)  # 256 KB stack for threads (default is often too large)
            except:
                pass
            
            # Set Python-specific optimizations
            if hasattr(sys, 'set_int_max_str_digits'):
                # Limit maximum string conversion size (Python 3.10+)
                sys.set_int_max_str_digits(4096)
                self._print("Set Python integer-to-string conversion limit to 4096 digits", "green")
                
            # Configure NumPy/TensorFlow thread optimizations if installed
            try:
                # Set number of threads for parallel operations
                os.environ["OMP_NUM_THREADS"] = "8"  # Optimal for Ryzen 9 5900X
                os.environ["MKL_NUM_THREADS"] = "8"
                os.environ["NUMEXPR_NUM_THREADS"] = "8"
                
                self._print("Set optimal thread count for numerical libraries", "green")
            except:
                pass
            
            # CUDA optimizations for RTX 3080
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Consistent device numbering
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Dynamic memory growth
            
            self._print("Applied common cross-platform optimizations", "green")
        except Exception as e:
            self._print(f"Failed to apply common optimizations: {e}", "yellow")
            
    def _optimize_system(self) -> bool:
        """
        Optimize system settings for best performance.
        
        Applies platform-specific and hardware-specific optimizations to maximize
        performance for the Maggie AI Assistant, with specific enhancements for
        AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.
        
        Returns
        -------
        bool
            True if optimization succeeded, False otherwise
            
        Notes
        -----
        This function optimizes system settings for running Maggie,
        including process priority and thread affinity on Windows,
        and governor settings on Linux.
        """
        try:
            # Apply platform-specific optimizations
            if self.platform == "Windows":
                self._windows_optimizations()
            elif self.platform == "Linux":
                self._linux_optimizations()
        
            # Apply common optimizations
            self._common_optimizations()
        
            return True
        except Exception as e:
            self._print(f"Failed to optimize system: {e}", "red")
            return False

    def _verify_system(self) -> bool:
        """
        Verify system configuration and installed components.
        
        Performs a comprehensive verification of the system configuration,
        installed dependencies, and hardware compatibility. This ensures
        that all components are properly installed and configured before
        starting the application.
        
        Returns
        -------
        bool
            True if system verification passed, False otherwise
        
        Notes
        -----
        This function runs the main.py script with the --verify flag to perform
        various checks including Python version, CUDA availability, required
        directories, and critical dependencies.
        """
        self._print("\nVerifying system configuration...", "cyan")
        
        # Determine python command based on platform
        if self.platform == "Windows":
            python_cmd = os.path.join(self.base_dir, "venv", "Scripts", "python")
        else:
            python_cmd = os.path.join(self.base_dir, "venv", "bin", "python")
        
        # First perform our own basic verification
        if not self._check_python_version():
            self._print("Python version verification failed", "red")
            return False
        
        # Check for PyTorch with CUDA
        try:
            returncode, stdout, _ = self._run_command([
                python_cmd, 
                "-c", 
                "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
            ], check=False)
            
            if "CUDA available: True" in stdout:
                self._print("CUDA is available for PyTorch", "green")
            else:
                self._print("CUDA is not available for PyTorch, GPU acceleration will be limited", "yellow")
        except:
            self._print("Could not verify PyTorch CUDA support", "yellow")
            
        # Run the built-in verification script
        returncode, stdout, stderr = self._run_command([python_cmd, "main.py", "--verify"])
        
        if returncode != 0:
            self._print("System verification failed", "red")
            
            # If we have error output, show it for debugging
            if stderr:
                self._print("Verification errors:", "red")
                for line in stderr.splitlines():
                    self._print(f"  {line}", "red")
                    
            self._print("Some functionality may not work correctly", "yellow")
            
            # Ask user if they want to continue anyway
            response = input(f"{self.colors['magenta']}System verification failed. Continue anyway? (y/n): {self.colors['reset']}")
            
            if response.lower() != "y":
                return False
                
            self._print("Continuing despite verification failure", "yellow")
            return True
            
        self._print("System verification passed", "green")
        return True
        
    def install(self) -> bool:
        """
        Run the full installation process.
        
        Executes the complete installation workflow for Maggie AI Assistant,
        handling system checks, dependency installation, configuration setup,
        model downloads, and system optimization. Provides a comprehensive
        summary of the installation status and next steps.
        
        Returns
        -------
        bool
            True if installation successful, False otherwise
        
        Notes
        -----
        This is the main method that orchestrates the entire installation process.
        It performs all necessary steps in the correct order, with appropriate
        error handling and user feedback at each stage.
        """
        self._print("=== Maggie AI Assistant Installation ===", "cyan")
        self._print(f"Platform: {self.platform}", "cyan")
        
        # Print current timestamp for installation timing
        start_time = time.time()
        self._print(f"Installation started at: {time.strftime('%Y-%m-%d %H:%M:%S')}", "cyan")
        
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
        self._print("\nInstalling dependencies (this may take some time)...", "cyan")
        dependency_result = self._install_dependencies()
        if not dependency_result:
            self._print("Some dependencies failed to install", "yellow")
            self._print("Continuing with installation, but some features may not work", "yellow")
            
            # Ask user if they want to continue despite dependency failures
            response = input(f"{self.colors['magenta']}Some dependencies failed to install. Continue anyway? (y/n): {self.colors['reset']}")
            
            if response.lower() != "y":
                self._print("Installation aborted by user due to dependency failures", "red")
                return False
            
        # Setup configuration
        if not self._setup_config():
            return False
            
        # Download models
        if not self._download_models():
            self._print("Some models failed to download", "yellow")
            self._print("Continuing with installation, but some features may not work", "yellow")
            
        # Create recipe template
        if not self._create_recipe_template():
            self._print("Recipe template creation failed", "yellow")
            self._print("Recipe functionality may not work properly", "yellow")
            
        # Optimize system
        if not self._optimize_system():
            self._print("System optimization failed", "yellow")
            self._print("Using default configuration", "yellow")
            
        # Verify system
        verify_result = self._verify_system()
            
        # Installation complete
        end_time = time.time()
        installation_time = end_time - start_time
        self._print(f"\n=== Installation Complete ({installation_time:.1f} seconds) ===", "green")
        
        # Show summary of installation status
        self._print("\nInstallation Summary:", "cyan")
        self._print(f"Platform: {self.platform} ({platform.release()})", "green")
        self._print(f"Python: {platform.python_version()}", "green")
        
        # Hardware detection
        if gpu_info["available"]:
            gpu_text = f"Detected: {gpu_info['name']}"
            if gpu_info["is_rtx_3080"]:
                gpu_text += " (Optimized)"
            self._print(f"GPU: {gpu_text}", "green")
        else:
            self._print("GPU: Not detected or not compatible", "yellow")
            
        # Tools detection
        self._print(f"Git found: {'Yes' if self.has_git else 'No'}", "green" if self.has_git else "yellow")
        self._print(f"C++ compiler found: {'Yes' if self.has_cpp_compiler else 'No'}", "green" if self.has_cpp_compiler else "yellow")
        self._print(f"Windows SDK found: {'Yes' if self.has_windows_sdk else 'No'}", "green" if self.has_windows_sdk else "yellow")
        self._print(f"System verification: {'Passed' if verify_result else 'Failed'}", "green" if verify_result else "yellow")
        
        # Reminders
        self._print("\nImportant Reminders:", "cyan")
        self._print("1. Edit config.yaml to add your Picovoice access key from https://console.picovoice.ai/", "yellow")
        self._print("2. To run Maggie AI Assistant:", "yellow")
        
        if self.platform == "Windows":
            self._print("   .\\venv\\Scripts\\activate", "cyan")
            self._print("   python main.py", "cyan")
        else:
            self._print("   source venv/bin/activate", "cyan")
            self._print("   python main.py", "cyan")
        
        # Required tools not found warnings
        missing_tools = []
        if not self.has_git:
            missing_tools.append(("Git", "https://git-scm.com/download/win"))
            
        if not self.has_cpp_compiler:
            missing_tools.append(("Visual C++ Build Tools", "https://visualstudio.microsoft.com/visual-cpp-build-tools/"))
            
        if not self.has_windows_sdk and self.platform == "Windows":
            missing_tools.append(("Windows SDK", "https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/"))
            
        if missing_tools:
            self._print("\nMissing Required Tools:", "yellow")
            for i, (tool, url) in enumerate(missing_tools, 1):
                self._print(f"{i}. {tool} not found", "yellow")
                self._print(f"   Install from: {url}", "yellow")
            self._print("\nInstalling these tools will enable full functionality.", "yellow")
            
        # Ask if user wants to start Maggie
        response = input(f"\n{self.colors['magenta']}Would you like to start Maggie now? (y/n): {self.colors['reset']}")
        
        if response.lower() == "y":
            self._print("\nStarting Maggie AI Assistant...", "cyan")
            
            if self.platform == "Windows":
                python_cmd = os.path.join(self.base_dir, "venv", "Scripts", "python")
            else:
                python_cmd = os.path.join(self.base_dir, "venv", "bin", "python")
                
            self._run_command([python_cmd, "main.py"])
            
        return True


def main():
    """
    Main entry point for the installer.
    
    Parses command-line arguments and runs the installation process.
    
    Returns
    -------
    int
        Exit code - 0 for success, 1 for error
    """
    parser = argparse.ArgumentParser(description="Maggie AI Assistant Installer")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--skip-problematic-packages", action="store_true",
                        help="Skip packages that require compilation and may have compatibility issues")
    args = parser.parse_args()
    
    installer = MaggieInstaller(verbose=args.verbose, skip_problematic=args.skip_problematic_packages)
    success = installer.install()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())