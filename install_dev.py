#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from collections.abc import MutableMapping # Use abc for compatibility

# --- Helper Classes (ColorOutput, ProgressTracker) ---
# (ColorOutput and ProgressTracker classes remain the same as previous version)
class ColorOutput:
    """Handles colored terminal output."""
    def __init__(self, force_enable: bool = False):
        self.enabled = force_enable or self._supports_color()
        if self.enabled:
            self.colors = {
                'reset': '\x1b[0m', 'bold': '\x1b[1m', 'red': '\x1b[91m',
                'green': '\x1b[92m', 'yellow': '\x1b[93m', 'blue': '\x1b[94m',
                'magenta': '\x1b[95m', 'cyan': '\x1b[96m', 'white': '\x1b[97m'
            }
        else:
            self.colors = {color: '' for color in ['reset', 'bold', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']}

    def _supports_color(self) -> bool:
        """Checks if the terminal supports color."""
        if platform.system() == 'Windows':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                stdout = kernel32.GetStdHandle(-11)
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(stdout, ctypes.byref(mode)):
                    if (mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING) == ENABLE_VIRTUAL_TERMINAL_PROCESSING:
                        return True
                    mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
                    if kernel32.SetConsoleMode(stdout, mode):
                        return True
            except:
                 if int(platform.release()) >= 10: return True
            return False
        if os.environ.get('NO_COLOR'): return False
        if os.environ.get('FORCE_COLOR'): return True
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def print(self, message: str, color: Optional[str] = None, bold: bool = False):
        """Prints a message with optional color and bold formatting."""
        formatted = message
        if self.enabled:
            if bold and 'bold' in self.colors:
                formatted = f"{self.colors['bold']}{formatted}"
            if color and color in self.colors:
                formatted = f"{self.colors[color]}{formatted}"
            if (bold or color) and 'reset' in self.colors:
                formatted = f"{formatted}{self.colors['reset']}"
        print(formatted)

    def input(self, prompt: str, color: Optional[str] = None, bold: bool = False) -> str:
        """Gets input with a formatted prompt."""
        formatted = prompt
        if self.enabled:
            if bold and 'bold' in self.colors:
                formatted = f"{self.colors['bold']}{formatted}"
            if color and color in self.colors:
                formatted = f"{self.colors[color]}{formatted}"
            if (bold or color) and 'reset' in self.colors:
                formatted = f"{formatted}{self.colors['reset']}"
        return input(formatted)

class ProgressTracker:
    """Tracks and displays installation progress."""
    def __init__(self, color: ColorOutput, total_steps: int = 10):
        self.color = color
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()

    def start_step(self, step_name: str):
        """Starts a new installation step."""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        self.color.print(f"\n[{self.current_step}/{self.total_steps}] {step_name} (Elapsed: {elapsed:.1f}s)", color='cyan', bold=True)

    def complete_step(self, success: bool = True, message: Optional[str] = None):
        """Marks a step as complete or failed."""
        if success: status = '✓ Complete'; color = 'green'
        else: status = '✗ Failed'; color = 'red'
        msg = f"  {status}"
        if message: msg += f": {message}"
        self.color.print(msg, color=color)

    def elapsed_time(self) -> float: return time.time() - self.start_time

    def display_summary(self, success: bool = True):
        """Displays the final installation summary."""
        elapsed = self.elapsed_time()
        if success: status = 'Installation Completed Successfully'; color = 'green'
        else: status = 'Installation Completed with Errors'; color = 'yellow'
        self.color.print(f"\n=== {status} ===", color=color, bold=True)
        self.color.print(f"Total time: {elapsed:.1f} seconds")

# --- MaggieInstaller Class - Modified for Poetry ---

class MaggieInstaller:
    """Handles the installation process for Maggie AI Assistant using Poetry."""
    def __init__(self, verbose: bool = False, cpu_only: bool = False, skip_models: bool = False):
        self.verbose = verbose
        self.cpu_only = cpu_only
        self.skip_models = skip_models
        self.base_dir = Path(__file__).parent.resolve()
        self.platform_system = platform.system()
        self.platform_machine = platform.machine()
        self.required_dirs = [
            'downloads', 'logs', 'maggie', 'maggie/cache', 'maggie/cache/tts',
            'maggie/core', 'maggie/extensions', 'maggie/models', 'maggie/models/llm',
            'maggie/models/stt', 'maggie/models/tts', 'maggie/templates',
            'maggie/templates/extension', 'maggie/utils', 'maggie/utils/hardware',
            'maggie/utils/config', 'maggie/utils/llm', 'maggie/utils/stt', 'maggie/utils/tts'
        ]
        self.color = ColorOutput()
        self.total_steps = 7 # 1:Verify+Tools, 2:Dirs, 3:Poetry Install+HWDetect, 4:Config, 5:Models, 6:Templates, 7:Complete
        self.progress = ProgressTracker(self.color, self.total_steps)
        self.is_admin = self._check_admin_privileges()
        self.has_git = False
        self.has_cpp_compiler = False
        self.has_poetry = False
        self.hardware_info = {
            'cpu': {'is_ryzen_9_5900x': False, 'model': '', 'cores': 0, 'threads': 0},
            'gpu': {'is_rtx_3080': False, 'model': '', 'vram_gb': 0, 'cuda_available': False, 'cuda_version': '', 'cudnn_available': False, 'cudnn_version': ''},
            'memory': {'total_gb': 0, 'available_gb': 0, 'is_32gb': False}
        }

    def _check_admin_privileges(self) -> bool:
        """Checks for administrator privileges."""
        try:
            if self.platform_system == 'Windows':
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else: return os.geteuid() == 0
        except Exception as e:
            if self.verbose: self.color.print(f"Admin check failed: {e}", "yellow")
            return False

    def _run_command(self, command: List[str], check: bool = True, shell: bool = False, capture_output: bool = True, cwd: Optional[Union[str, Path]] = None, env: Optional[Dict] = None) -> Tuple[int, str, str]:
        """Runs a shell command."""
        if self.verbose: self.color.print(f"Running command: {' '.join(command)} in {cwd or self.base_dir}", 'cyan')
        try:
            full_env = os.environ.copy()
            if env: full_env.update(env)
            process = subprocess.Popen(
                command if not shell else ' '.join(command),
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                shell=shell, text=True, encoding='utf-8', errors='replace',
                cwd=str(cwd or self.base_dir), env=full_env
            )
            stdout, stderr = process.communicate()
            return_code = process.returncode
            if check and return_code != 0:
                 error_msg = f"Command '{' '.join(command)}' failed with code {return_code}"
                 if stderr: error_msg += f": {stderr.strip()}"
                 if self.verbose: self.color.print(error_msg, 'red')
            return return_code, stdout.strip() if stdout else '', stderr.strip() if stderr else ''
        except FileNotFoundError:
            if self.verbose: self.color.print(f"Error: Command not found: {command[0]}", 'red')
            return -1, '', f"Command not found: {command[0]}"
        except Exception as e:
            if self.verbose: self.color.print(f"Error executing command '{' '.join(command)}': {e}", 'red')
            return -1, '', str(e)

    def _download_file(self, url: str, destination: str, show_progress: bool = True) -> bool:
        """Downloads a file from a URL with improved progress."""
        # (Code remains the same as previous version)
        dest_path = Path(destination)
        try:
            self.color.print(f"Downloading {url}", 'blue')
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
            with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, 'wb') as out_file: # Add timeout
                content_length_header = response.info().get('Content-Length')
                file_size = int(content_length_header) if content_length_header else 0
                downloaded = 0
                block_size = 8192 * 16 # Use larger 128KB chunks
                start_time = time.time()
                if show_progress and file_size > 0:
                    self.color.print(f"Total file size: {file_size / 1024 / 1024:.1f} MB")
                    progress_bar_width = 40
                    last_percent_reported = -1
                    while True:
                        buffer = response.read(block_size)
                        if not buffer: break
                        downloaded += len(buffer)
                        out_file.write(buffer)
                        percent = int(downloaded * 100 / file_size)
                        if percent > last_percent_reported:
                           last_percent_reported = percent
                           filled_width = int(progress_bar_width * downloaded / file_size)
                           bar = '█' * filled_width + '-' * (progress_bar_width - filled_width)
                           elapsed = time.time() - start_time
                           speed_mbps = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                           print(f"\r  Progress: |{bar}| {percent}% ({downloaded/1024/1024:.1f}/{file_size/1024/1024:.1f} MB) {speed_mbps:.1f} MB/s  ", end="")
                    print() # New line after progress bar finishes
                else: # No progress bar if size unknown or zero
                    self.color.print("Downloading (size unknown)...", "blue")
                    while True:
                        buffer = response.read(block_size)
                        if not buffer: break
                        downloaded += len(buffer)
                        out_file.write(buffer)
            self.color.print(f"Download completed: {dest_path}", 'green')
            return True
        except urllib.error.URLError as e:
             self.color.print(f"Error downloading file (URL Error: {e.reason}): {url}", 'red')
             if dest_path.exists():
                 try: dest_path.unlink()
                 except OSError: pass
             return False
        except Exception as e:
            self.color.print(f"Error downloading file {url}: {e}", 'red')
            if dest_path.exists():
                try: dest_path.unlink()
                except OSError: pass
            return False

    def _verify_python_version(self) -> bool:
        """Verifies the Python version is 3.10.x."""
        # (Code remains the same as previous version)
        version = platform.python_version_tuple()
        if int(version[0]) != 3 or int(version[1]) != 10:
            self.color.print(f"ERROR: Incompatible Python version: {platform.python_version()}", 'red', bold=True)
            self.color.print('Maggie requires Python 3.10.x specifically.', 'red')
            self.color.print('Poetry should be configured to use Python 3.10.', 'red')
            if self.platform_system == 'Windows':
                self.color.print('Install Python 3.10 from: https://www.python.org/downloads/release/python-31011/', 'yellow')
            else:
                self.color.print('Ensure Python 3.10 is available and selected by Poetry.', 'yellow')
                self.color.print('Consider using pyenv or similar tools to manage Python versions.', 'yellow')
            return False
        self.color.print(f"Python {platform.python_version()} - Compatible ✓", 'green')
        return True

    def _detect_hardware(self) -> None:
        """Detects CPU, Memory, and GPU hardware. Should be called AFTER poetry install."""
        # (Code remains the same as previous version)
        self.color.print('Detecting Hardware Configuration...', 'cyan', bold=True)
        self.hardware_info['cpu'] = self._detect_cpu()
        self.hardware_info['memory'] = self._detect_memory()
        self.hardware_info['gpu'] = self._detect_gpu() if not self.cpu_only else {'available': False, 'cuda_available': False}
        self._print_hardware_summary()

    def _detect_cpu(self)->Dict[str,Any]:
        """Detects CPU information."""
        # (Code remains the same as previous version)
        cpu_info = {'is_ryzen_9_5900x': False, 'model':'Unknown','cores':0,'threads':0}
        try: cpu_info['model'] = platform.processor() or 'Unknown'
        except Exception: pass
        try:
             import psutil
             cpu_info['cores']=psutil.cpu_count(logical=False) or 0
             cpu_info['threads']=psutil.cpu_count(logical=True) or 0
        except ImportError:
             cpu_info['threads'] = os.cpu_count() or 0
             cpu_info['cores'] = cpu_info['threads'] // 2 if cpu_info['threads'] > 1 else cpu_info['threads']
        except Exception as e:
             if self.verbose: self.color.print(f"CPU count error: {e}", "yellow")
        if self.platform_system == 'Windows':
            try:
                import wmi
                c = wmi.WMI()
                processor = c.Win32_Processor()[0]
                cpu_info['model'] = processor.Name.strip()
                if cpu_info['cores'] == 0 and hasattr(processor, 'NumberOfCores'): cpu_info['cores'] = processor.NumberOfCores
                if cpu_info['threads'] == 0 and hasattr(processor, 'NumberOfLogicalProcessors'): cpu_info['threads'] = processor.NumberOfLogicalProcessors
            except ImportError:
                 if self.verbose: self.color.print("WMI not found, skipping detailed Windows CPU detection.", "yellow")
            except Exception as e:
                 if self.verbose: self.color.print(f"WMI CPU detection error: {e}", "yellow")
        model_lower = cpu_info['model'].lower()
        if 'ryzen 9' in model_lower and '5900x' in model_lower: cpu_info['is_ryzen_9_5900x'] = True
        return cpu_info

    def _detect_memory(self)->Dict[str,Any]:
        """Detects memory information."""
        # (Code remains the same as previous version)
        memory_info={'total_gb':0,'available_gb':0,'is_32gb':False}
        try:
            import psutil
            mem=psutil.virtual_memory()
            memory_info['total_gb']=mem.total / (1024**3)
            memory_info['available_gb']=mem.available / (1024**3)
            memory_info['is_32gb'] = memory_info['total_gb'] >= 30.0
        except ImportError:
             if self.verbose: self.color.print("psutil not installed, cannot determine exact RAM.", "yellow")
        except Exception as e:
             if self.verbose: self.color.print(f"Memory detection error: {e}", "yellow")
        return memory_info

    def _detect_gpu(self)->Dict[str,Any]:
        """Detects GPU information using PyTorch (assumes installed via Poetry)."""
        # (Code remains the same as previous version)
        gpu_info = {'available': False, 'is_rtx_3080': False, 'model': 'Unknown','vram_gb': 0, 'cuda_available': False, 'cuda_version': '', 'cudnn_available': False, 'cudnn_version': ''}
        if self.cpu_only: return gpu_info
        check_script = """
import torch, sys, platform
try:
    cuda_available = torch.cuda.is_available()
    print(f"CUDA_Available: {cuda_available}")
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Device_Count: {device_count}")
        if device_count > 0:
            props = torch.cuda.get_device_properties(0)
            print(f"Device_Name: {props.name}")
            print(f"VRAM_GB: {props.total_memory / (1024**3):.2f}")
            print(f"CUDA_Version: {torch.version.cuda}")
            cudnn_available = torch.backends.cudnn.is_available()
            print(f"cuDNN_Available: {cudnn_available}")
            if cudnn_available: print(f"cuDNN_Version: {torch.backends.cudnn.version()}")
except Exception as e: print(f"GPU_Check_Error: {e!r}", file=sys.stderr)
"""
        returncode, stdout, stderr = self._run_command(['poetry', 'run', 'python', '-c', check_script], check=False, capture_output=True)
        if returncode != 0 or "GPU_Check_Error" in stderr:
            if self.verbose: self.color.print('PyTorch check script failed.', 'yellow'); self.color.print(f"Error: {stderr}", "yellow")
            smi_path = "nvidia-smi"
            if self.platform_system == "Windows":
                program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
                smi_path_win = Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
                if smi_path_win.exists(): smi_path = str(smi_path_win)
            rc_smi, _, _ = self._run_command([smi_path], check=False, capture_output=False)
            if rc_smi == 0: self.color.print('nvidia-smi found. GPU likely present but env needs config.', 'yellow'); gpu_info['available'] = True
            return gpu_info
        for line in stdout.splitlines():
            try:
                if ':' not in line: continue
                key, value = line.split(':', 1); key = key.strip(); value = value.strip()
                if key == 'CUDA_Available': gpu_info['cuda_available'] = (value == 'True')
                elif key == 'Device_Name': gpu_info['model'] = value; gpu_info['available'] = True; gpu_info['is_rtx_3080'] = '3080' in value
                elif key == 'VRAM_GB': gpu_info['vram_gb'] = float(value)
                elif key == 'CUDA_Version': gpu_info['cuda_version'] = value
                elif key == 'cuDNN_Available': gpu_info['cudnn_available'] = (value == 'True')
                elif key == 'cuDNN_Version': gpu_info['cudnn_version'] = str(value)
            except Exception as e: if self.verbose: self.color.print(f"Error parsing GPU info: {e}", "yellow")
        if not gpu_info['cuda_available']: self.color.print('No CUDA-capable GPU detected by PyTorch.', 'yellow')
        return gpu_info

    def _check_tools(self) -> Dict[str, bool]:
        """Checks for required tools: Git, C++ Compiler, Poetry. Installs Poetry if missing."""
        tools_status = {'git': False, 'cpp_compiler': False, 'poetry': False}
        self.color.print("Checking required tools (Poetry, Git, C++ Compiler)...", "cyan")

        # Check Poetry FIRST - attempt install if missing
        returncode_poetry, stdout_poetry, _ = self._run_command(['poetry', '--version'], check=False)
        if returncode_poetry == 0:
            tools_status['poetry'] = True
            self.has_poetry = True
            self.color.print(f"  Poetry found: {stdout_poetry.strip()} ✓", 'green')
        else:
            self.color.print('  Poetry not found.', 'yellow')
            self.color.print('  Attempting to install Poetry using "pip install --user poetry"...', 'blue')
            self.color.print('  NOTE: Using pip to install Poetry globally or user-wide is not the official recommended method.', 'yellow')
            self.color.print('        Consider using the official installer for better isolation: https://python-poetry.org/docs/#installation', 'yellow')

            # Use sys.executable to ensure pip matches the current python
            pip_cmd = [sys.executable, '-m', 'pip', 'install', '--user', 'poetry']
            rc_pip_install, out_pip, err_pip = self._run_command(pip_cmd, check=False, capture_output=True)

            if rc_pip_install == 0:
                self.color.print('  Poetry installed via pip successfully.', 'green')
                # Verify installation
                returncode_poetry_after, stdout_poetry_after, _ = self._run_command(['poetry', '--version'], check=False)
                if returncode_poetry_after == 0:
                    tools_status['poetry'] = True
                    self.has_poetry = True
                    self.color.print(f"  Poetry found: {stdout_poetry_after.strip()} ✓", 'green')
                    self.color.print('  IMPORTANT: You may need to restart your terminal or add the user script path to your PATH environment variable for the `poetry` command to be available immediately.', 'magenta')
                else:
                    self.color.print('  Poetry installed via pip, but `poetry --version` command still fails.', 'red')
                    self.color.print('  Please ensure the Python user script directory is in your PATH.', 'red')
                    self.color.print('  Installation cannot continue without a working Poetry command.', 'red')
                    return tools_status # Exit checks, poetry failed
            else:
                self.color.print('  Failed to install Poetry using pip.', 'red', bold=True)
                if err_pip: self.color.print(f"  Pip Error: {err_pip}", 'red')
                self.color.print('  Please install Poetry manually: https://python-poetry.org/docs/#installation', 'yellow')
                return tools_status # Exit checks, poetry failed

        # Check Git (Code remains the same as previous version)
        returncode_git, stdout_git, _ = self._run_command(['git', '--version'], check=False)
        if returncode_git == 0: tools_status['git'] = True; self.has_git = True; self.color.print(f"  Git found: {stdout_git.strip()} ✓", 'green')
        else:
            self.color.print('  Git not found - Required for downloading some models.', 'yellow')
            # (Instructions remain the same)

        # Check C++ Compiler (Code remains the same as previous version)
        compiler_found = False
        if self.platform_system == 'Windows':
            rc_where, _, _ = self._run_command(['where', 'cl.exe'], check=False, capture_output=False)
            if rc_where == 0: compiler_found = True; self.color.print('  Visual C++ compiler (cl.exe) found in PATH ✓', 'green')
            else: self.color.print('  Visual C++ compiler (cl.exe) not found in PATH.', 'yellow')
        else: # Linux/MacOS
            rc_gpp, _, _ = self._run_command(['which', 'g++'], check=False, capture_output=False)
            rc_gcc, _, _ = self._run_command(['which', 'gcc'], check=False, capture_output=False)
            if rc_gpp == 0 or rc_gcc == 0: compiler_found = True; compiler_name = "g++" if rc_gpp == 0 else "gcc"; self.color.print(f'  C++ compiler ({compiler_name}) found ✓', 'green')
            else: self.color.print('  C++ compiler (g++/gcc) not found.', 'yellow')
        if compiler_found: tools_status['cpp_compiler'] = True; self.has_cpp_compiler = True
        else:
            self.color.print('  Required for building some packages from source if wheels are unavailable.', 'yellow')
            # (Instructions remain the same)

        return tools_status

    def _create_directories(self) -> bool:
        """Creates required directories."""
        # (Code remains the same as previous version)
        self.color.print("Creating required directories...", "cyan")
        try:
            for directory in self.required_dirs:
                dir_path = self.base_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                if self.verbose: self.color.print(f"  Ensured directory exists: {dir_path}", 'green')
            package_dirs = ['maggie', 'maggie/core', 'maggie/extensions', 'maggie/utils','maggie/utils/config', 'maggie/utils/hardware', 'maggie/utils/llm','maggie/utils/stt', 'maggie/utils/tts']
            for pkg_dir_str in package_dirs:
                pkg_dir = self.base_dir / pkg_dir_str
                pkg_dir.mkdir(parents=True, exist_ok=True)
                init_path = pkg_dir / '__init__.py'
                if not init_path.exists():
                    try: init_path.touch(); self.color.print(f"  Created __init__.py in {pkg_dir_str}", 'green')
                    except Exception as e_init: self.color.print(f"Error creating __init__.py in {pkg_dir_str}: {e_init}", 'red')
            self.color.print('Directory structure verified/created.', 'green')
            return True
        except Exception as e:
            self.color.print(f"Error creating directories: {e}", 'red')
            return False

    def _generate_default_pyproject_toml(self) -> bool:
        """Generates a default pyproject.toml file if one does not exist."""
        pyproject_path = self.base_dir / 'pyproject.toml'
        self.color.print(f"File 'pyproject.toml' not found.", 'yellow')
        self.color.print(f"Generating default '{pyproject_path}'...", 'blue')

        # Define the default TOML content as a raw string
        # Based on the previously generated example
        default_toml_content = """\
# pyproject.toml (Generated by install_dev.py)
# Please review and customize dependencies as needed.

[tool.poetry]
name = "maggie"
version = "0.1.0"
description = "Maggie AI Assistant (Default Configuration)"
authors = ["Your Name <you@example.com>"] # Please update author info
license = "MIT"
readme = "README.md"
packages = [{include = "maggie"}] # Assumes package code is in 'maggie' dir

dependencies = { python = "~3.10" } # Requires Python 3.10.x

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

# --- PyTorch Source Definition for CUDA 11.8 ---
[[tool.poetry.source]]
name = "pytorch_cu118"
url = "https://download.pytorch.org/whl/cu118"
priority = "explicit"


[tool.poetry.dependencies]
# --- Pinned Dependencies (Using specific source) ---
torch = {version = "2.1.2", source = "pytorch_cu118"}
torchvision = {version = "0.16.2", source = "pytorch_cu118"}
torchaudio = {version = "0.16.2", source = "pytorch_cu118"}

# --- Core Dependencies (Let Poetry Resolve) ---
# Add all direct dependencies identified previously. Use flexible constraints.
PyYAML = "^6.0"
loguru = "^0.7"
numpy = "*" # Let torch/others constrain this
tqdm = "^4.66"
requests = "^2.31"
psutil = "^5.9"

# --- STT Dependencies ---
pvporcupine = "^3.0"
SpeechRecognition = "^3.10"
PyAudio = "*" # May require system libraries (portaudio) and build tools
faster-whisper = ">=0.10.0" # Avoids tokenizer<0.15 conflict
sounddevice = "^0.4.6"

# --- TTS Dependencies ---
# Install kokoro directly from GitHub (main branch)
kokoro = {git = "https://github.com/hexgrad/kokoro.git"}
# kokoro-onnx had conflicts, let Poetry try to resolve. May need adjustment.
kokoro-onnx = "*"
huggingface-hub = ">=0.15.1"
soundfile = ">=0.12.1"
librosa = ">=0.10.0"
phonemizer = ">=3.2.1"

# --- LLM Dependencies ---
transformers = "^4.38"
optimum = "^1.17"
accelerate = "^0.27"
safetensors = ">=0.4.0"
sentencepiece = ">=0.1.99"
protobuf = "*"
ninja = ">=1.11.1"

# --- UI and Document Processing ---
PySide6 = "^6.5"
python-docx = ">=0.8.11"
transitions = "^0.9.0"
mammoth = "^1.7.0"

# --- Optional GPU Dependencies (defined in extras) ---
onnxruntime-gpu = { version = "*", optional = true }
auto-gptq = { version = "^0.7.1", optional = true }


# --- Define Optional Dependency Groups ---
[tool.poetry.extras]
gpu = ["onnxruntime-gpu", "auto-gptq"]


# --- Development Dependencies (Optional) ---
[tool.poetry.group.dev.dependencies]
pytest = "*"
black = "*"
isort = "*"
flake8 = "*"
pipdeptree = "*" # Useful for debugging dependencies


# --- Build System Definition (Required for Poetry) ---
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
"""
        try:
            with open(pyproject_path, 'w', encoding='utf-8') as f:
                f.write(default_toml_content)
            self.color.print(f"Default 'pyproject.toml' created successfully.", 'green')
            self.color.print("Please review the generated file and adjust dependencies if needed.", 'yellow')
            return True
        except Exception as e:
            self.color.print(f"ERROR: Failed to write default pyproject.toml: {e}", 'red')
            return False


    def _install_with_poetry(self) -> bool:
        """Installs dependencies using Poetry, generating pyproject.toml if needed."""
        self.color.print("Installing dependencies using Poetry...", "cyan")
        self.color.print("This may take a while depending on the number of dependencies and network speed.", "blue")

        pyproject_path = self.base_dir / 'pyproject.toml'

        # Generate default pyproject.toml if it doesn't exist
        if not pyproject_path.exists():
            if not self._generate_default_pyproject_toml():
                return False # Stop if generation failed

        # Check again if the file exists (generation might have failed)
        if not pyproject_path.exists():
             self.color.print(f"ERROR: pyproject.toml still not found after attempting generation.", 'red')
             return False

        # Basic check for [tool.poetry.dependencies] section in existing/generated file
        try:
            with open(pyproject_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check for both sections to be safe
                if '[tool.poetry.dependencies]' not in content or '[tool.poetry]' not in content:
                    self.color.print("ERROR: pyproject.toml exists but is missing '[tool.poetry]' or '[tool.poetry.dependencies]' section.", 'red', bold=True)
                    self.color.print("Please ensure it is correctly configured for Poetry.", 'red')
                    return False
        except Exception as e:
            self.color.print(f"Error reading pyproject.toml: {e}", 'red')
            return False

        # Prepare poetry install command (same as before)
        poetry_cmd = ['poetry', 'install', '--no-interaction']
        if not self.cpu_only:
            has_gpu_extra = False
            try:
                with open(pyproject_path, 'r', encoding='utf-8') as f:
                    toml_content = f.read()
                    if '[tool.poetry.extras]' in toml_content and 'gpu = [' in toml_content: has_gpu_extra = True
                if has_gpu_extra:
                    self.color.print("Including [gpu] extras for Poetry installation.", "blue")
                    poetry_cmd.extend(['--extras', 'gpu'])
                else: self.color.print("No [gpu] extra found/defined in pyproject.toml, installing base dependencies.", "yellow")
            except Exception as e: self.color.print(f"Could not check for GPU extras: {e}", "yellow"); self.color.print("Attempting base install.", "yellow")

        if self.verbose: poetry_cmd.append('-vvv')

        # Run poetry install (same as before)
        self.color.print(f"Running: {' '.join(poetry_cmd)}", "blue")
        returncode, stdout, stderr = self._run_command(poetry_cmd, check=False, capture_output=True)
        if stdout: self.color.print(f"Poetry output:\n{stdout}", 'cyan')
        if stderr: self.color.print(f"Poetry errors/warnings:\n{stderr}", 'red' if returncode != 0 else 'yellow')

        if returncode != 0:
            self.color.print("ERROR: Poetry failed to resolve dependencies or install packages.", 'red', bold=True)
            self.color.print("Please check the output above for specific conflict details.", 'red')
            # (Error messages remain the same)
            return False
        else:
            self.color.print("Poetry successfully resolved and installed dependencies ✓", 'green')
            return True

    # --- Model Downloading Methods (_download_whisper_model, etc.) ---
    # (Code remains the same as previous version)
    def _download_whisper_model(self)->bool:
        """Downloads the Whisper base.en model using huggingface_hub via Poetry."""
        if self.skip_models: self.color.print('Skipping Whisper model download (--skip-models)', 'yellow'); return True
        model_dir = self.base_dir / 'maggie/models/stt/whisper-base.en'
        essential_files = ['model.bin', 'config.json', 'tokenizer.json', 'vocab.json']
        repo_id = "openai/whisper-base.en"
        if model_dir.exists():
            try:
                files_in_dir = {f.name for f in model_dir.iterdir() if f.is_file()}
                if all(essential_file in files_in_dir for essential_file in essential_files): self.color.print('Whisper base.en model is already available ✓', 'green'); return True
                else: self.color.print('Whisper model directory exists but appears incomplete. Re-downloading...', 'yellow')
            except Exception as e: self.color.print(f"Error checking existing Whisper model directory: {e}", 'yellow')
        model_dir.mkdir(parents=True, exist_ok=True)
        download_script = f'''
import os, sys
from huggingface_hub import snapshot_download, HfApi, hf_hub_download
model_dir_str = r"{str(model_dir).replace(os.sep, '/')}"
repo_id = "{repo_id}"
try:
    print(f"Attempting to download model '{{repo_id}}' to '{{model_dir_str}}'")
    snapshot_download(repo_id=repo_id, local_dir=model_dir_str, allow_patterns=["*.json", "*.bin", "*.txt", "preprocessor_config.json", "generation_config.json"], ignore_patterns=["*.safetensors", "*.h5", "*.ot", "flax*", "tf*"], local_dir_use_symlinks=False, resume_download=True)
    print("Model downloaded successfully via snapshot_download.")
    sys.exit(0)
except Exception as e_snap:
    print(f"Snapshot download failed: {{e_snap!r}}", file=sys.stderr)
    print("Attempting to download essential files individually (fallback)...", file=sys.stderr)
    essential_files_list = ['config.json', 'generation_config.json', 'model.bin', 'preprocessor_config.json', 'tokenizer.json', 'vocab.json']
    all_downloaded = True
    for filename in essential_files_list:
        try: print(f"Downloading {{filename}}..."); hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_dir_str, resume_download=True)
        except Exception as e_file: print(f"Failed to download {{filename}}: {{e_file!r}}", file=sys.stderr); all_downloaded = False
    if all_downloaded: print("Essential files downloaded individually (fallback)."); sys.exit(0)
    else: print("Failed to download some essential model files via fallback.", file=sys.stderr); sys.exit(1)
'''
        self.color.print(f'Downloading Whisper model ({repo_id}) from Hugging Face...', 'cyan')
        returncode, stdout, stderr = self._run_command(['poetry', 'run', 'python', '-c', download_script], check=False, capture_output=True)
        if self.verbose and stdout: self.color.print(f"Whisper download stdout:\n{stdout}", "cyan")
        if stderr: self.color.print(f"Whisper download stderr:\n{stderr}", "yellow" if returncode == 0 else "red")
        if returncode != 0: self.color.print("Error downloading Whisper model.", 'red'); return False
        try:
            files_in_dir = {f.name for f in model_dir.iterdir() if f.is_file()}
            missing_files = [f for f in essential_files if f not in files_in_dir]
            if not missing_files: self.color.print('Whisper base.en model downloaded and verified successfully ✓', 'green'); return True
            else: self.color.print('Whisper model download appears incomplete.', 'yellow'); self.color.print(f'Missing essential files: {", ".join(missing_files)}', 'yellow'); return False
        except Exception as e: self.color.print(f"Error verifying downloaded Whisper model: {e}", 'red'); return False

    def _download_af_heart_model(self)->bool:
        """Downloads the af_heart.pt TTS voice model."""
        if self.skip_models: self.color.print('Skipping af_heart model download (--skip-models)', 'yellow'); return True
        model_dir = self.base_dir / 'maggie/models/tts'; model_path = model_dir / 'af_heart.pt'; MIN_SIZE = 40 * 1024 * 1024
        if model_path.exists():
            try:
                file_size = model_path.stat().st_size
                if file_size >= MIN_SIZE: self.color.print(f"af_heart voice model verified ({file_size / (1024 * 1024):.2f} MB) ✓", 'green'); return True
                else: self.color.print(f"Existing af_heart model incorrect size: {file_size / (1024 * 1024):.2f} MB. Re-downloading...", 'yellow')
            except Exception as e: self.color.print(f"Error checking existing af_heart model: {e}. Re-downloading...", 'yellow')
        model_dir.mkdir(parents=True, exist_ok=True)
        model_urls = ['https://huggingface.co/hexgrad/kokoro-voices/resolve/main/af_heart.pt', 'https://github.com/hexgrad/kokoro/releases/download/v0.1/af_heart.pt']
        download_successful = False
        for url in model_urls:
            self.color.print(f"Attempting download from: {url}", 'cyan')
            if self._download_file(url, str(model_path)):
                try:
                    file_size = model_path.stat().st_size
                    if file_size >= MIN_SIZE: self.color.print(f"af_heart model download successful ({file_size / (1024 * 1024):.2f} MB) ✓", 'green'); download_successful = True; break
                    else: self.color.print(f"Downloaded file incorrect size: {file_size / (1024 * 1024):.2f} MB", 'yellow'); model_path.unlink(missing_ok=True)
                except Exception as e_verify: self.color.print(f"Error verifying downloaded af_heart model: {e_verify}", 'red'); model_path.unlink(missing_ok=True)
        if not download_successful: self.color.print('Failed to download af_heart voice model.', 'red'); return False
        return True

    def _download_kokoro_onnx_models(self)->bool:
        """Downloads the necessary ONNX model files for kokoro-onnx."""
        if self.skip_models: self.color.print('Skipping kokoro-onnx model download (--skip-models)', 'yellow'); return True
        self.color.print("Downloading Kokoro ONNX models...", "cyan")
        model_dir = self.base_dir / 'maggie/models/tts'; model_dir.mkdir(parents=True, exist_ok=True)
        kokoro_onnx_repo = "onnx-community/Kokoro-82M-v1.0-ONNX"
        required_assets = [
            {'type': 'file', 'name': 'model.onnx', 'repo': kokoro_onnx_repo, 'subpath': 'onnx/model.onnx', 'min_size': 10*1024*1024},
            {'type': 'file', 'name': 'model_quantized.onnx', 'repo': kokoro_onnx_repo, 'subpath': 'onnx/model_quantized.onnx', 'min_size': 5*1024*1024, 'optional': True},
            {'type': 'file', 'name': 'voices-v1.0.bin', 'repo': kokoro_onnx_repo, 'subpath': 'voices-v1.0.bin', 'min_size': 5*1024*1024},
            {'type': 'file', 'name': 'tokens.txt', 'repo': kokoro_onnx_repo, 'subpath': 'tokens.txt', 'min_size': 100},
            {'type': 'dir', 'name': 'espeak-ng-data', 'repo': "rhasspy/espeak-ng-data", 'repo_type': 'dataset'},
        ]
        all_successful = True
        for asset in required_assets:
            asset_name = asset['name']; asset_path = model_dir / asset_name; min_size = asset.get('min_size', 0); is_optional = asset.get('optional', False)
            if asset_path.exists():
                is_ok = False
                if asset['type'] == 'dir' and asset_path.is_dir() and any(asset_path.iterdir()): is_ok = True
                elif asset['type'] == 'file' and asset_path.is_file():
                    try:
                        file_size = asset_path.stat().st_size
                        if file_size >= min_size: is_ok = True
                        else: self.color.print(f"Existing {asset_name} incorrect size. Re-downloading...", 'yellow')
                    except Exception as e_check: self.color.print(f"Error checking {asset_name}: {e_check}. Re-downloading...", 'yellow')
                if is_ok: self.color.print(f"{asset_name} already exists ✓", 'green'); continue
            self.color.print(f"Downloading {asset_name} from {asset['repo']}...", 'cyan')
            repo_id = asset['repo']; repo_type = asset.get('repo_type', 'model'); filename = asset.get('subpath', asset_name)
            if asset['type'] == 'file': dl_script = f'import sys; from huggingface_hub import hf_hub_download; try: hf_hub_download(repo_id="{repo_id}", repo_type="{repo_type}", filename="{filename}", local_dir=r"{str(model_dir)}", local_dir_use_symlinks=False, resume_download=True); sys.exit(0); except Exception as e: print(f"Download failed: {{e!r}}", file=sys.stderr); sys.exit(1)'
            elif asset['type'] == 'dir': dl_script = f'import sys; from huggingface_hub import snapshot_download; try: snapshot_download(repo_id="{repo_id}", repo_type="{repo_type}", local_dir=r"{str(asset_path)}", local_dir_use_symlinks=False, resume_download=True); sys.exit(0); except Exception as e: print(f"Download failed: {{e!r}}", file=sys.stderr); sys.exit(1)'
            else: self.color.print(f"Unknown asset type '{asset['type']}'", 'red'); all_successful = False if not is_optional else all_successful; continue
            rc_dl, out_dl, err_dl = self._run_command(['poetry', 'run', 'python', '-c', dl_script], check=False, capture_output=True)
            if self.verbose and out_dl: self.color.print(f"{asset_name} download stdout:\n{out_dl}", "cyan")
            if err_dl: self.color.print(f"{asset_name} download stderr:\n{err_dl}", "yellow" if rc_dl == 0 else "red")
            if rc_dl == 0:
                if asset['type'] == 'file':
                    try:
                        file_size = asset_path.stat().st_size
                        if file_size >= min_size: self.color.print(f"{asset_name} download successful ✓", 'green')
                        else: self.color.print(f"Downloaded {asset_name} incorrect size.", 'yellow'); asset_path.unlink(missing_ok=True); all_successful = False if not is_optional else all_successful
                    except Exception as e_verify: self.color.print(f"Error verifying {asset_name}: {e_verify}", 'red'); all_successful = False if not is_optional else all_successful
                elif asset['type'] == 'dir':
                     if asset_path.is_dir() and any(asset_path.iterdir()): self.color.print(f"{asset_name} download successful ✓", 'green')
                     else: self.color.print(f"Downloaded {asset_name} dir empty.", 'yellow'); all_successful = False if not is_optional else all_successful
            else: self.color.print(f"Failed to download {asset_name}.", 'red'); all_successful = False if not is_optional else all_successful
        if all_successful: self.color.print('Kokoro ONNX models/data downloaded successfully ✓', 'green'); return True
        else: self.color.print('Some Kokoro ONNX models/data failed.', 'yellow'); return False

    def _download_mistral_model(self)->bool:
        """Downloads the Mistral 7B Instruct GPTQ model using Git LFS."""
        if self.skip_models: self.color.print('Skipping Mistral model download (--skip-models)', 'yellow'); return True
        mistral_dir = self.base_dir / 'maggie/models/llm/mistral-7b-instruct-v0.3-GPTQ-4bit'; repo_url = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GPTQ'
        if mistral_dir.exists() and any(mistral_dir.iterdir()):
             self.color.print(f"Mistral dir exists: {mistral_dir}", 'yellow'); response = self.color.input("  Download again (y) or keep existing (n)? [n]: ", color='magenta')
             if response.lower() != 'y': self.color.print("Keeping existing model files.", 'green'); return True
             else: self.color.print("Removing existing dir...", 'yellow'); shutil.rmtree(mistral_dir, ignore_errors=True)
        mistral_dir.parent.mkdir(parents=True, exist_ok=True)
        if not self.has_git: self.color.print('Git not found. Cannot download Mistral.', 'red'); return False
        lfs_check_code, _, _ = self._run_command(['git', 'lfs', '--version'], check=False, capture_output=False)
        if lfs_check_code != 0: self.color.print("Git LFS not found.", 'red'); return False
        self._run_command(['git', 'lfs', 'install', '--skip-repo'], check=False, capture_output=False)
        self.color.print(f"Downloading Mistral model from {repo_url} using Git LFS...", 'cyan'); self.color.print("This is ~5GB and may take time.", 'blue')
        clone_env = os.environ.copy(); clone_env['GIT_LFS_SKIP_SMUDGE'] = '1'
        returncode_clone, _, stderr_clone = self._run_command(['git', 'clone', repo_url, str(mistral_dir)], check=False, capture_output=True, env=clone_env)
        if returncode_clone != 0 and 'already exists' not in stderr_clone: self.color.print('Error initiating Git clone.', 'red'); self.color.print(f"Git error: {stderr_clone}", 'red'); shutil.rmtree(mistral_dir, ignore_errors=True); return False
        self.color.print("Git repo cloned. Pulling LFS files...", 'cyan')
        returncode_lfs, stdout_lfs, stderr_lfs = self._run_command(['git', 'lfs', 'pull'], check=False, capture_output=True, cwd=str(mistral_dir))
        if self.verbose and stdout_lfs: self.color.print(f"LFS pull output:\n{stdout_lfs}", "cyan")
        if stderr_lfs: self.color.print(f"LFS pull stderr:\n{stderr_lfs}", "yellow" if returncode_lfs == 0 else "red")
        if returncode_lfs != 0: self.color.print('Error pulling Git LFS files.', 'red'); return False
        essential_configs = ['config.json', 'tokenizer.json', 'quantize_config.json']; has_safetensors = False
        try:
             files_in_dir = {f.name for f in mistral_dir.iterdir()}; missing_configs = [f for f in essential_configs if f not in files_in_dir]; has_safetensors = any(f.endswith('.safetensors') for f in files_in_dir)
             if not missing_configs and has_safetensors: self.color.print('Mistral model downloaded & verified ✓', 'green'); return True
             else: self.color.print('Mistral download incomplete.', 'yellow'); return False
        except Exception as e: self.color.print(f"Error verifying Mistral download: {e}", 'red'); return False

    def _create_recipe_template(self)->bool:
        """Creates a default recipe template if it doesn't exist using python-docx."""
        # (Code remains the same as previous version)
        template_dir = self.base_dir / 'maggie/templates'; template_path = template_dir / 'recipe_template.docx'; template_dir.mkdir(parents=True, exist_ok=True)
        if template_path.exists(): self.color.print('Recipe template already exists ✓', 'green'); return True
        self.color.print("Creating default recipe template...", "cyan")
        template_path_str_escaped = str(template_path).replace('\\', '\\\\')
        create_script = f"""
import docx, os, sys
from pathlib import Path
template_path_str = r'{template_path_str_escaped}'
template_path = Path(template_path_str)
try:
    doc = docx.Document(); doc.add_heading("Recipe Name", level=1); doc.add_paragraph()
    doc.add_heading("Recipe Information", level=2); info_table = doc.add_table(rows=3, cols=2); info_table.style = 'Table Grid'
    info_table.cell(0, 0).text = "Preparation Time"; info_table.cell(0, 1).text = "00 minutes"
    info_table.cell(1, 0).text = "Cooking Time"; info_table.cell(1, 1).text = "00 minutes"
    info_table.cell(2, 0).text = "Servings"; info_table.cell(2, 1).text = "0 servings"; doc.add_paragraph()
    doc.add_heading("Ingredients", level=2); doc.add_paragraph("• Ingredient 1", style='List Bullet'); doc.add_paragraph("• Ingredient 2", style='List Bullet'); doc.add_paragraph("• Ingredient 3", style='List Bullet'); doc.add_paragraph()
    doc.add_heading("Instructions", level=2); doc.add_paragraph("Step 1", style='List Number'); doc.add_paragraph("Step 2", style='List Number'); doc.add_paragraph("Step 3", style='List Number'); doc.add_paragraph()
    doc.add_heading("Notes", level=2); doc.add_paragraph("Add any additional notes, tips, or variations here.")
    doc.save(template_path); print(f"Template saved to {{template_path}}"); sys.exit(0)
except ImportError: print("Error: python-docx not found.", file=sys.stderr); sys.exit(1)
except Exception as e: print(f"Error creating template: {{e}}", file=sys.stderr); sys.exit(1)
"""
        returncode, stdout, stderr = self._run_command(['poetry', 'run', 'python', '-c', create_script], check=False, capture_output=True)
        if self.verbose and stdout: self.color.print(f"Template stdout:\n{stdout}", "cyan")
        if stderr: self.color.print(f"Template stderr:\n{stderr}", "yellow" if returncode == 0 else "red")
        if returncode == 0: self.color.print('Recipe template created successfully ✓', 'green'); return True
        else: self.color.print('Failed to create recipe template.', 'red'); return False

    def _setup_config(self)->bool:
        """Creates or updates the config.yaml based on hardware detection."""
        # (Code remains the same as previous version)
        config_path = self.base_dir / 'config.yaml'; example_path = self.base_dir / 'config.yaml.example'
        if not example_path.exists(): example_path = self.base_dir / 'config.yaml.txt'
        if not example_path.exists(): example_path = self.base_dir / 'config-yaml-example.txt'
        if not config_path.exists() and not example_path.exists(): self.color.print("ERROR: Config/Example file not found.", 'red'); return False
        self.color.print("Setting up configuration file (config.yaml)...", "cyan")
        temp_hardware_file = self.base_dir / 'hardware_info.json.tmp'
        try:
            with open(temp_hardware_file, 'w') as f: json.dump(self.hardware_info, f, indent=2)
        except Exception as e: self.color.print(f"Error writing temp hardware info: {e}", 'red'); return False
        base_dir_str = repr(str(self.base_dir)); config_path_str_esc = str(config_path).replace('\\','\\\\'); example_path_str_esc = str(example_path).replace('\\','\\\\') if example_path.exists() else ''; hardware_file_str_esc = str(temp_hardware_file).replace('\\','\\\\'); cpu_only_flag = self.cpu_only
        config_script = f"""
import yaml, json, os, shutil, sys
from pathlib import Path
from collections.abc import MutableMapping
try: from yaml import CLoader as Loader, CDumper as Dumper
except ImportError: from yaml import Loader, Dumper
base_dir = Path({base_dir_str}); config_path = Path(r'{config_path_str_esc}'); example_path = Path(r'{example_path_str_esc}') if '{example_path_str_esc}' else None; hardware_file = Path(r'{hardware_file_str_esc}'); cpu_only = {cpu_only_flag}
def ensure_keys(d, keys):
    current = d
    for key in keys:
        if not isinstance(current, MutableMapping): return None
        current = current.setdefault(key, {{}})
    return current
config = {{}}; hardware_info = {{}}
try:
    with open(hardware_file, 'r', encoding='utf-8') as f: hardware_info = json.load(f)
except Exception as e: print(f"Error loading hardware info: {{e}}", file=sys.stderr)
if config_path.exists():
    print(f"Loading existing config: {{config_path}}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f: config = yaml.load(f, Loader=Loader)
        if not isinstance(config, MutableMapping): print("Warning: Existing config invalid.", file=sys.stderr); config = {{}}
    except Exception as e: print(f"Error loading config: {{e}}.", file=sys.stderr); config = {{}}
elif example_path and example_path.exists():
    print(f"Creating config from example: {{example_path}}")
    try:
        shutil.copyfile(example_path, config_path)
        with open(config_path, 'r', encoding='utf-8') as f: config = yaml.load(f, Loader=Loader)
        if not isinstance(config, MutableMapping): print("Warning: Example config invalid.", file=sys.stderr); config = {{}}
    except Exception as e: print(f"Error copying/loading example: {{e}}.", file=sys.stderr); config = {{}}
else: print("No config/example found. Starting fresh."); config = {{}}
llm_config = ensure_keys(config, ['llm']); gpu_config = ensure_keys(config, ['gpu']); cpu_config = ensure_keys(config, ['cpu']); mem_config = ensure_keys(config, ['memory']); stt_config = ensure_keys(config, ['stt']); stt_whisper_config = ensure_keys(stt_config, ['whisper']) if stt_config else None; tts_config = ensure_keys(config, ['tts'])
if llm_config is not None: llm_config.update({{'model_path': 'maggie/models/llm/mistral-7b-instruct-v0.3-GPTQ-4bit', 'model_type': 'mistral', 'use_autogptq': True}})
if stt_whisper_config is not None: stt_whisper_config['model_path'] = 'maggie/models/stt/whisper-base.en'
if tts_config is not None: tts_config.update({{'model_path': 'maggie/models/tts', 'voice_model': 'af_heart.pt'}})
gpu_hw = hardware_info.get('gpu', {{}}); cpu_hw = hardware_info.get('cpu', {{}}); mem_hw = hardware_info.get('memory', {{}})
if cpu_only:
    print("Applying CPU-only optimizations...")
    if llm_config: llm_config.update({{'gpu_layers': 0, 'gpu_layer_auto_adjust': False}}); llm_config.pop('rtx_3080_optimized', None)
    if gpu_config: gpu_config.update({{'max_percent': 0, 'model_unload_threshold': 0}}); gpu_config.pop('rtx_3080_optimized', None)
    if tts_config: tts_config['gpu_acceleration'] = False
    if stt_whisper_config: stt_whisper_config['compute_type'] = 'int8'
else:
    if gpu_hw.get('cuda_available'):
        print("Applying GPU optimizations...")
        if llm_config: llm_config.update({{'gpu_layer_auto_adjust': True, 'precision_type': 'float16', 'mixed_precision_enabled': True}})
        if tts_config: tts_config.update({{'gpu_acceleration': True}})
        if stt_whisper_config: stt_whisper_config['compute_type'] = 'float16'
        if gpu_hw.get('is_rtx_3080'):
            print("Applying RTX 3080 specific optimizations...")
            if llm_config: llm_config.update({{'gpu_layers': 32, 'rtx_3080_optimized': True}})
            if gpu_config: gpu_config.update({{'max_percent': 90, 'model_unload_threshold': 95, 'rtx_3080_optimized': True}})
            if tts_config: tts_config['gpu_precision'] = 'mixed_float16'
        else:
            print("Applying generic GPU optimizations...")
            if llm_config: llm_config.update({{'gpu_layers': -1}}); llm_config.pop('rtx_3080_optimized', None)
            if gpu_config: gpu_config.update({{'max_percent': 85, 'model_unload_threshold': 90}}); gpu_config.pop('rtx_3080_optimized', None)
            if tts_config: tts_config['gpu_precision'] = 'float16'
    else:
         print("CUDA not available via PyTorch. Forcing CPU settings.", file=sys.stderr)
         if llm_config: llm_config.update({{'gpu_layers': 0, 'gpu_layer_auto_adjust': False}}); llm_config.pop('rtx_3080_optimized', None)
         if gpu_config: gpu_config.update({{'max_percent': 0, 'model_unload_threshold': 0}}); gpu_config.pop('rtx_3080_optimized', None)
         if tts_config: tts_config['gpu_acceleration'] = False
         if stt_whisper_config: stt_whisper_config['compute_type'] = 'int8'
if cpu_hw.get('is_ryzen_9_5900x'):
    print("Applying Ryzen 9 5900X optimizations...")
    if cpu_config: cpu_config.update({{'max_threads': 8, 'thread_timeout': 30, 'ryzen_9_5900x_optimized': True}})
else:
    if cpu_config: cpu_config.pop('ryzen_9_5900x_optimized', None)
if mem_hw.get('is_32gb'):
    print("Applying 32GB+ RAM optimizations...")
    if mem_config: mem_config.update({{'max_percent': 75, 'model_unload_threshold': 85, 'xpg_d10_memory': True}})
else:
    if mem_config: mem_config.pop('xpg_d10_memory', None)
try:
    with open(config_path, 'w', encoding='utf-8') as f: yaml.dump(config, f, Dumper=Dumper, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Configuration saved to {{config_path}}")
except Exception as e: print(f"Error saving config: {{e}}", file=sys.stderr); sys.exit(1)
try: os.remove(hardware_file)
except Exception as e: print(f"Warning: Could not remove temp file: {{e}}", file=sys.stderr)
sys.exit(0)
"""
        returncode, stdout, stderr = self._run_command(['poetry', 'run', 'python', '-c', config_script], check=False, capture_output=True)
        if self.verbose and stdout: self.color.print(f"Config script stdout:\n{stdout}", "cyan")
        if stderr: self.color.print(f"Config script stderr:\n{stderr}", "yellow" if returncode == 0 else "red")
        if returncode != 0: self.color.print('Error setting up configuration file.', 'red'); return False
        else: self.color.print('Configuration file created/updated ✓', 'green'); self.color.print('NOTE: Edit config.yaml to add Picovoice Access Key.', 'yellow'); return True

    def verify_system(self)->bool:
        """Verifies Python version and essential tools (Poetry, Git, Compiler)."""
        self.progress.start_step('Verifying system & tools')
        python_compatible = self._verify_python_version()
        if not python_compatible:
            self.progress.complete_step(False,'Incompatible Python version')
            return False

        tools = self._check_tools() # This now attempts to install Poetry if missing
        if not tools['poetry']:
            # _check_tools already printed error messages
            self.progress.complete_step(False, 'Poetry setup failed.')
            return False
        # Warnings for optional tools were printed in _check_tools
        self.progress.complete_step(True)
        return True


    def install(self)->bool:
        """Runs the complete installation process using Poetry."""
        # (Code remains the same as previous version, calling the updated methods)
        self.color.print('\n=== Maggie AI Assistant Installation (using Poetry) ===', 'cyan', bold=True)
        self.color.print(f"Platform: {self.platform_system} ({platform.platform()})", 'cyan')
        self.color.print(f"Python: {platform.python_version()} (via Poetry environment)", 'cyan')
        if self.cpu_only: self.color.print('Mode: CPU-only installation requested', 'yellow')
        self.color.print(f"Project Root: {self.base_dir}", 'cyan')
        self.color.print(f"Non-Python Requirements: CUDA 11.8, cuDNN 8.9.7 (for GPU)", 'yellow')
        if not self.verify_system(): return False
        self.progress.start_step('Creating directory structure')
        if not self._create_directories(): self.progress.complete_step(False, 'Failed to create directories'); return False
        self.progress.complete_step(True)
        self.progress.start_step('Install dependencies & Detect Hardware')
        if not self._install_with_poetry(): self.progress.complete_step(False, 'Poetry dependency installation failed.'); return False
        self._detect_hardware()
        if not self.cpu_only and not self.hardware_info['gpu'].get('cuda_available'): self.color.print("  Warning: PyTorch installed, but CUDA unavailable.", 'yellow')
        self.progress.complete_step(True)
        self.progress.start_step('Setting up configuration file')
        if not self._setup_config(): self.progress.complete_step(False, 'Failed to set up configuration'); return False
        self.progress.complete_step(True)
        self.progress.start_step('Downloading models')
        models_ok = True
        if not self._download_af_heart_model(): self.color.print('Warning: Failed TTS voice download.', 'yellow')
        if not self._download_kokoro_onnx_models(): self.color.print('Warning: Failed kokoro-onnx download.', 'yellow')
        if not self._download_whisper_model(): self.color.print('Warning: Failed Whisper download.', 'yellow')
        if not self._download_mistral_model(): self.color.print('Warning: Failed Mistral download.', 'yellow')
        self.progress.complete_step(models_ok)
        self.progress.start_step('Setting up templates & completing installation')
        template_ok = self._create_recipe_template()
        if not template_ok: self.color.print('Warning: Failed recipe template creation.', 'yellow')
        self.progress.display_summary(True)
        self.color.print('\n--- Important Notes ---', 'cyan', bold=True)
        self.color.print('1. Dependencies installed via Poetry.', 'green')
        self.color.print(f"2. Non-Python Requirements: CUDA 11.8, cuDNN 8.9.7 (for GPU).", 'yellow')
        self.color.print(f"   (Detected CUDA: {self.hardware_info['gpu'].get('cuda_version', 'N/A')}, cuDNN: {self.hardware_info['gpu'].get('cudnn_version', 'N/A')})", 'yellow')
        self.color.print('3. Edit config.yaml for Picovoice Access Key.', 'yellow')
        if not self.has_git: self.color.print('4. Git not installed - model updates affected.', 'yellow')
        if not self.has_cpp_compiler: self.color.print('5. C++ Compiler not found - building packages may fail.', 'yellow')
        self.color.print('\n--- To start Maggie AI Assistant ---', 'cyan', bold=True)
        self.color.print('   Run: poetry run python main.py', 'green')
        self.progress.complete_step(True)
        return True

    def _print_hardware_summary(self):
        """Prints hardware summary previously gathered."""
        # (Code remains the same as previous version)
        self.color.print('Hardware Configuration Summary:', 'cyan', bold=True)
        cpu_info = self.hardware_info['cpu']; mem_info = self.hardware_info['memory']; gpu_info = self.hardware_info['gpu']
        self.color.print(f"  CPU: {cpu_info.get('model', 'Unknown')} ({cpu_info.get('cores', 'N/A')}c / {cpu_info.get('threads', 'N/A')}t)", 'green' if cpu_info.get('is_ryzen_9_5900x') else 'yellow')
        self.color.print(f"  RAM: {mem_info.get('total_gb', 0):.1f} GB", 'green' if mem_info.get('is_32gb') else 'yellow')
        if self.cpu_only: self.color.print('  GPU: CPU-only mode selected', 'yellow')
        elif gpu_info.get('cuda_available'):
             self.color.print(f"  GPU: {gpu_info.get('model', 'Unknown')} ✓", 'green')
             self.color.print(f"       {gpu_info.get('vram_gb', 0):.1f} GB VRAM", 'green')
             self.color.print(f"       CUDA {gpu_info.get('cuda_version', 'N/A')} | cuDNN {gpu_info.get('cudnn_version', 'N/A')}", 'green' if gpu_info.get('cudnn_available') else 'yellow')
        else: self.color.print('  GPU: No CUDA-capable GPU detected by PyTorch', 'red')

# --- Main Execution ---

def main() -> int:
    """Parses arguments and runs the installer."""
    # (Code remains the same as previous version)
    parser = argparse.ArgumentParser(description='Maggie AI Assistant Installer (Poetry Version)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--cpu-only', action='store_true', help='Install CPU-only version')
    parser.add_argument('--skip-models', action='store_true', help='Skip downloading AI models')
    args = parser.parse_args()
    script_dir = Path(__file__).parent.resolve(); os.chdir(script_dir)
    print(f"Running installer from directory: {script_dir}")
    installer = MaggieInstaller(verbose=args.verbose, cpu_only=args.cpu_only, skip_models=args.skip_models)
    try:
        success = installer.install()
        return 0 if success else 1
    except KeyboardInterrupt: print('\n\nInstallation cancelled by user.'); return 1
    except Exception as e:
        installer.color.print(f"\n\nAN UNEXPECTED ERROR OCCURRED:", 'red', bold=True)
        installer.color.print(f"Error Type: {type(e).__name__}", 'red')
        installer.color.print(f"Error Details: {e}", 'red')
        if args.verbose: import traceback; installer.color.print("Traceback:", 'red'); traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())