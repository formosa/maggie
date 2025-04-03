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
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from collections.abc import MutableMapping

# --- Helper Classes (ColorOutput, ProgressTracker) ---
# (Simplified - only ColorOutput needed for this bootstrap script)
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
                    if (mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING) == ENABLE_VIRTUAL_TERMINAL_PROCESSING: return True
                    mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
                    if kernel32.SetConsoleMode(stdout, mode): return True
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
            if bold and 'bold' in self.colors: formatted = f"{self.colors['bold']}{formatted}"
            if color and color in self.colors: formatted = f"{self.colors[color]}{formatted}"
            if (bold or color) and 'reset' in self.colors: formatted = f"{formatted}{self.colors['reset']}"
        print(formatted)

# --- Installer Class (Simplified Bootstrap) ---

class MaggieBootstrap:
    """Handles bootstrapping the Poetry environment for Maggie AI Assistant."""
    def __init__(self, verbose: bool = False, cpu_only: bool = False, skip_models: bool = False):
        self.verbose = verbose
        self.cpu_only = cpu_only # Pass this to post-install script
        self.skip_models = skip_models # Pass this to post-install script
        self.base_dir = Path(__file__).parent.resolve()
        self.platform_system = platform.system()
        self.color = ColorOutput()
        self.has_poetry = False

    def _run_command(self, command: List[str], check: bool = True, shell: bool = False, capture_output: bool = True, cwd: Optional[Union[str, Path]] = None, env: Optional[Dict] = None) -> Tuple[int, str, str]:
        """Runs a shell command."""
        # (Same as previous version)
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

    def _verify_python_version(self) -> bool:
        """Verifies the Python version is 3.10.x."""
        # (Same as previous version)
        version = platform.python_version_tuple()
        if int(version[0]) != 3 or int(version[1]) != 10:
            self.color.print(f"ERROR: Incompatible Python version: {platform.python_version()}", 'red', bold=True)
            self.color.print('Maggie requires Python 3.10.x specifically.', 'red')
            return False
        self.color.print(f"Python {platform.python_version()} - Compatible ✓", 'green')
        return True

    def _check_poetry(self) -> bool:
        """Checks for Poetry and attempts installation if missing."""
        # (Modified from previous _check_tools)
        self.color.print("Checking for Poetry...", "cyan")
        returncode_poetry, stdout_poetry, _ = self._run_command(['poetry', '--version'], check=False)
        if returncode_poetry == 0:
            self.has_poetry = True
            self.color.print(f"  Poetry found: {stdout_poetry.strip()} ✓", 'green')
            return True
        else:
            self.color.print('  Poetry not found.', 'yellow')
            self.color.print('  Attempting to install Poetry using "pip install --user poetry"...', 'blue')
            self.color.print('  NOTE: Using pip is not the official recommended method.', 'yellow')
            self.color.print('        See: https://python-poetry.org/docs/#installation', 'yellow')

            pip_cmd = [sys.executable, '-m', 'pip', 'install', '--user', 'poetry']
            rc_pip_install, out_pip, err_pip = self._run_command(pip_cmd, check=False, capture_output=True)

            if rc_pip_install == 0:
                self.color.print('  Poetry installed via pip successfully.', 'green')
                check_cmds = [['poetry', '--version'], [sys.executable, '-m', 'poetry', '--version']]
                verified_after_install = False
                for cmd in check_cmds:
                    returncode_poetry_after, stdout_poetry_after, _ = self._run_command(cmd, check=False)
                    if returncode_poetry_after == 0:
                        self.has_poetry = True
                        self.color.print(f"  Poetry verified using `{' '.join(cmd)}`: {stdout_poetry_after.strip()} ✓", 'green')
                        self.color.print('  IMPORTANT: May need terminal restart or PATH update for direct `poetry` command.', 'magenta')
                        verified_after_install = True
                        break
                if not verified_after_install:
                    self.color.print('  Poetry installed via pip, but command still fails.', 'red')
                    self.color.print('  Check PATH or install manually.', 'red')
                    return False
                return True # Successfully installed and verified
            else:
                self.color.print('  Failed to install Poetry using pip.', 'red', bold=True)
                if err_pip: self.color.print(f"  Pip Error: {err_pip}", 'red')
                self.color.print('  Install Poetry manually: https://python-poetry.org/docs/#installation', 'yellow')
                return False

    def _configure_poetry_venv(self) -> bool:
        """Configures poetry to create venv inside project."""
        if not self.has_poetry: return False
        self.color.print("Configuring Poetry to create virtualenv in project (.venv)...", "blue")
        config_cmd = ['poetry', 'config', 'virtualenvs.in-project', 'true', '--local']
        rc_config, out_config, err_config = self._run_command(config_cmd, check=False, capture_output=True)
        if rc_config == 0:
            self.color.print("  Poetry configured for in-project venv successfully.", "green")
            return True
        else:
            self.color.print("  Warning: Failed to set Poetry's virtualenvs.in-project config locally.", "yellow")
            if err_config: self.color.print(f"    Error: {err_config}", "yellow")
            return False # Treat config failure as potentially problematic

    def _generate_default_pyproject_toml(self) -> bool:
        """Generates a default pyproject.toml file if one does not exist."""
        # (Same as previous version)
        pyproject_path = self.base_dir / 'pyproject.toml'
        self.color.print(f"File 'pyproject.toml' not found.", 'yellow')
        self.color.print(f"Generating default '{pyproject_path}'...", 'blue')
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
python = "~3.10" # Requires Python 3.10.x

# --- Pinned Dependencies (Using specific source) ---
torch = {version = "2.1.2", source = "pytorch_cu118"}
torchvision = {version = "0.16.2", source = "pytorch_cu118"}
torchaudio = {version = "2.1.2", source = "pytorch_cu118"} # Corrected version

# --- Core Dependencies (Let Poetry Resolve) ---
PyYAML = "^6.0"
loguru = "^0.7"
numpy = "*"
tqdm = "^4.66"
requests = "^2.31"
psutil = "^5.9"
WMI = { version = "*", markers = "sys_platform == 'win32'" } # Added

# --- STT Dependencies ---
pvporcupine = "^3.0"
SpeechRecognition = "^3.10"
PyAudio = "*"
faster-whisper = ">=0.10.0"
sounddevice = "^0.4.6"

# --- TTS Dependencies ---
kokoro = {git = "https://github.com/hexgrad/kokoro.git"}
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
pipdeptree = "*"


# --- Build System Definition (Required for Poetry) ---
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
"""
        try:
            with open(pyproject_path, 'w', encoding='utf-8') as f: f.write(default_toml_content)
            self.color.print(f"Default 'pyproject.toml' created successfully.", 'green'); self.color.print("Please review the generated file.", 'yellow'); return True
        except Exception as e: self.color.print(f"ERROR: Failed to write default pyproject.toml: {e}", 'red'); return False


    def setup_environment(self) -> bool:
        """Ensures Poetry is installed, configured, and dependencies are installed."""
        self.color.print("--- Step 1: Verifying Python and Tools ---", 'magenta', bold=True)
        if not self._verify_python_version(): return False
        if not self._check_poetry(): return False # Exits if poetry check/install fails
        if not self._configure_poetry_venv(): return False # Exit if config fails

        self.color.print("--- Step 2: Checking/Updating Lock File ---", 'magenta', bold=True)
        pyproject_path = self.base_dir / 'pyproject.toml'
        if not pyproject_path.exists():
            if not self._generate_default_pyproject_toml(): return False
        if not pyproject_path.exists(): self.color.print("ERROR: pyproject.toml not found.", 'red'); return False
        try: # Sanity check TOML structure
            with open(pyproject_path, 'r', encoding='utf-8') as f: content = f.read()
            if '[tool.poetry.dependencies]' not in content or '[tool.poetry]' not in content:
                self.color.print("ERROR: pyproject.toml missing poetry sections.", 'red'); return False
        except Exception as e: self.color.print(f"Error reading pyproject.toml: {e}", 'red'); return False

        lock_cmd = ['poetry', 'lock'] # Removed --no-update
        if self.verbose: lock_cmd.append('-vvv')
        self.color.print(f"Running: {' '.join(lock_cmd)}", "blue")
        rc_lock, out_lock, err_lock = self._run_command(lock_cmd, check=False, capture_output=True)
        if out_lock: self.color.print(f"Poetry lock output:\n{out_lock}", 'cyan')
        if err_lock: self.color.print(f"Poetry lock errors/warnings:\n{err_lock}", 'red' if rc_lock != 0 else 'yellow')
        if rc_lock != 0: self.color.print("ERROR: Poetry failed to update lock file.", 'red'); return False
        self.color.print("Poetry lock file updated/verified successfully ✓", 'green')

        self.color.print("--- Step 3: Installing Dependencies ---", 'magenta', bold=True)
        install_cmd = ['poetry', 'install', '--no-interaction']
        if not self.cpu_only:
            # Check for GPU extras (same logic as before)
             has_gpu_extra = False
             try:
                 with open(pyproject_path, 'r', encoding='utf-8') as f: toml_content = f.read()
                 if '[tool.poetry.extras]' in toml_content and 'gpu = [' in toml_content: has_gpu_extra = True
                 if has_gpu_extra: self.color.print("Including [gpu] extras.", "blue"); install_cmd.extend(['--extras', 'gpu'])
                 else: self.color.print("No [gpu] extra found/defined, installing base dependencies.", "yellow")
             except Exception as e: self.color.print(f"Could not check GPU extras: {e}", "yellow"); self.color.print("Attempting base install.", "yellow")

        if self.verbose: install_cmd.append('-vvv')
        self.color.print(f"Running: {' '.join(install_cmd)}", "blue")
        rc_install, out_install, err_install = self._run_command(install_cmd, check=False, capture_output=True)
        if out_install: self.color.print(f"Poetry install output:\n{out_install}", 'cyan')
        if err_install: self.color.print(f"Poetry install errors/warnings:\n{err_install}", 'red' if rc_install != 0 else 'yellow')
        if rc_install != 0: self.color.print("ERROR: Poetry install command failed.", 'red'); return False
        self.color.print("Poetry install successful ✓", 'green')
        return True

    def run_post_install(self):
        """Runs the second script (_post_install_setup.py) using poetry run."""
        self.color.print("--- Step 4: Running Post-Installation Setup ---", 'magenta', bold=True)
        post_install_script = self.base_dir / '_post_install_setup.py'
        if not post_install_script.exists():
            self.color.print(f"ERROR: Post-install script not found: {post_install_script}", "red")
            return False

        cmd = ['poetry', 'run', 'python', str(post_install_script)]
        # Pass arguments through
        if self.verbose: cmd.append('--verbose')
        if self.cpu_only: cmd.append('--cpu-only')
        if self.skip_models: cmd.append('--skip-models')

        self.color.print(f"Running: {' '.join(cmd)}", "blue")
        # Run without capturing output by default, let the script print directly
        # Use check=True to raise error on failure
        try:
             # Use subprocess.run for simpler execution when not capturing detailed output
             result = subprocess.run(cmd, cwd=self.base_dir, check=True, text=True, encoding='utf-8')
             self.color.print("Post-installation setup completed successfully.", "green")
             return True
        except subprocess.CalledProcessError as e:
             self.color.print(f"ERROR: Post-installation script failed with exit code {e.returncode}.", "red")
             # Stderr/stdout might be useful but are captured by run by default
             # if e.stderr: self.color.print(f"Stderr:\n{e.stderr}", "red")
             # if e.stdout: self.color.print(f"Stdout:\n{e.stdout}", "red")
             return False
        except FileNotFoundError:
             self.color.print(f"ERROR: Command 'poetry' or 'python' not found when trying to run post-install script.", "red")
             return False
        except Exception as e:
             self.color.print(f"ERROR: Unexpected error running post-install script: {e}", "red")
             return False


# --- Main Execution ---

def main() -> int:
    """Parses arguments and runs the installer."""
    parser = argparse.ArgumentParser(description='Maggie AI Assistant Installer (Poetry Bootstrap)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--cpu-only', action='store_true', help='Install CPU-only version')
    parser.add_argument('--skip-models', action='store_true', help='Skip downloading AI models')
    args = parser.parse_args()
    script_dir = Path(__file__).parent.resolve(); os.chdir(script_dir)
    print(f"Running installer bootstrap from directory: {script_dir}")

    bootstrapper = MaggieBootstrap(
        verbose=args.verbose,
        cpu_only=args.cpu_only,
        skip_models=args.skip_models
    )
    installer_color = ColorOutput() # For final message

    try:
        if not bootstrapper.setup_environment():
             installer_color.print("\nBootstrap failed: Environment setup incomplete.", "red", bold=True)
             return 1

        if not bootstrapper.run_post_install():
             installer_color.print("\nInstallation failed during post-install setup.", "red", bold=True)
             return 1

        installer_color.print("\nInstallation process finished.", "green", bold=True)
        return 0

    except KeyboardInterrupt:
        print('\n\nInstallation cancelled by user.')
        return 1
    except Exception as e:
        installer_color.print(f"\n\nAN UNEXPECTED ERROR OCCURRED:", 'red', bold=True)
        installer_color.print(f"Error Type: {type(e).__name__}", 'red')
        installer_color.print(f"Error Details: {e}", 'red')
        if args.verbose: import traceback; installer_color.print("Traceback:", 'red'); traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
