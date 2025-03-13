#!/usr/bin/env python3
"""
Maggie AI Assistant - Main Script
===============================
Entry point for the Maggie AI Assistant application.

This script initializes and starts the Maggie AI Assistant with optimizations
for AMD Ryzen 9 5900X and NVIDIA GeForce RTX 3080 hardware.
It handles command-line arguments, logging configuration, system verification,
and application startup.

Examples
--------
Standard startup:
$ python main.py

Start with debug logging:
$ python main.py --debug

Verify system without starting:
$ python main.py --verify

Create recipe template:
$ python main.py --create-template

Apply hardware optimizations:
$ python main.py --optimize
"""

# Standard library imports
import os
import sys
import argparse
import platform
import multiprocessing
from typing import Dict, Any, Optional, List, Tuple

# Third-party imports
from loguru import logger

# Local imports (imported where needed to avoid circular imports)
# from maggie import MaggieAI
# from utils.config_validator import ConfigValidator

__all__ = ['main', 'parse_arguments', 'setup_logging', 'verify_system']

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments with the following fields:
        - config: Path to configuration file
        - debug: Boolean flag for debug logging
        - verify: Boolean flag for system verification
        - create_template: Boolean flag for template creation
        - optimize: Boolean flag for hardware optimization
    """
    parser = argparse.ArgumentParser(
        description="Maggie AI Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify system configuration without starting the assistant"
    )
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create the recipe template file if it doesn't exist"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize configuration for detected hardware"
    )
    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """
    Set up logging configuration.
    
    Parameters
    ----------
    debug : bool, optional
        Enable debug logging if True, by default False
        
    Notes
    -----
    This function configures the loguru logger with console and file
    handlers, and logs system information for diagnostic purposes.
    """
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Configure loguru
    log_level = "DEBUG" if debug else "INFO"
    logger.configure(
        handlers=[
            {"sink": sys.stdout, "level": log_level, "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"},
            {"sink": "logs/maggie.log", "rotation": "10 MB", "retention": "1 week", "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"}
        ]
    )
    
    # Log system info
    log_system_info()


def log_system_info() -> None:
    """
    Log detailed information about the system.
    
    Logs information about the operating system, CPU, RAM, and GPU
    to help with diagnostics and troubleshooting.
    """
    try:
        # System and Python info
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        
        # Log CPU info
        log_cpu_info()
        
        # Log RAM info
        log_ram_info()
        
        # Log GPU info
        log_gpu_info()
            
    except ImportError as e:
        logger.warning(f"System info modules not available: {e}")
    except Exception as e:
        logger.warning(f"Error gathering system information: {e}")


def log_cpu_info() -> None:
    """
    Log information about the CPU.
    
    Logs CPU model, cores, and other relevant information.
    """
    try:
        import psutil
        
        cpu_info = platform.processor()
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        
        logger.info(f"CPU: {cpu_info}")
        logger.info(f"CPU Cores: {cpu_cores} physical, {cpu_threads} logical")
    except ImportError:
        logger.warning("psutil not available for CPU information")
    except Exception as e:
        logger.warning(f"Error getting CPU information: {e}")


def log_ram_info() -> None:
    """
    Log information about system memory.
    
    Logs total RAM and available RAM.
    """
    try:
        import psutil
        
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        logger.info(f"RAM: {ram_gb:.2f} GB")
    except ImportError:
        logger.warning("psutil not available for RAM information")
    except Exception as e:
        logger.warning(f"Error getting RAM information: {e}")


def log_gpu_info() -> None:
    """
    Log information about the GPU.
    
    Logs GPU model, VRAM, and CUDA information if available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                logger.info(f"GPU {i}: {gpu_name}")
                logger.info(f"GPU Memory: {memory_gb:.2f} GB")
                
                # Log RTX 3080 specific capabilities
                if "3080" in gpu_name:
                    logger.info(f"RTX 3080 detected - Tensor Cores available")
                    logger.info(f"Optimizing for 10GB VRAM and Ampere architecture")
        else:
            logger.warning("CUDA not available, GPU acceleration disabled")
    except ImportError:
        logger.warning("PyTorch not installed, GPU detection skipped")
    except Exception as e:
        logger.warning(f"Error detecting GPU information: {e}")


def verify_system() -> bool:
    """
    Verify system meets requirements for Maggie.
    
    Returns
    -------
    bool
        True if system meets requirements, False otherwise
        
    Notes
    -----
    This function checks Python version, GPU availability, required
    directories, and critical dependencies to ensure the system is
    properly configured for running Maggie.
    """
    verification_issues = []
    logger.info("Verifying system configuration...")
    
    # Check Python version - require exactly 3.10.x
    python_version_valid = check_python_version()
    if not python_version_valid:
        verification_issues.append("Incompatible Python version")
    
    # Check for CUDA with better error handling
    cuda_valid = check_cuda_availability()
    if not cuda_valid:
        verification_issues.append("CUDA/GPU issues detected")
    
    # Check for required directories and create if necessary
    dirs_valid = check_required_directories()
    if not dirs_valid:
        verification_issues.append("Directory issues detected")
    
    # Check for required dependencies
    deps_valid = check_dependencies()
    if not deps_valid:
        verification_issues.append("Missing dependencies")
    
    # Final verification result
    if verification_issues:
        logger.error("System verification failed with the following issues:")
        for issue in verification_issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("System verification completed successfully")
    return True


def check_python_version() -> bool:
    """
    Check if Python version is compatible.
    
    Returns
    -------
    bool
        True if Python version is 3.10.x, False otherwise
    """
    python_version = platform.python_version_tuple()
    if int(python_version[0]) != 3 or int(python_version[1]) != 10:
        error_msg = f"Unsupported Python version: {platform.python_version()}"
        logger.error(error_msg)
        logger.error("Maggie requires Python 3.10.x specifically. Other versions are not compatible.")
        logger.error("Please install Python 3.10 and try again.")
        return False
    else:
        logger.info(f"Python version {platform.python_version()} is compatible")
        return True


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available and compatible.
    
    Returns
    -------
    bool
        True if CUDA is available or not required, False on critical issues
    """
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.version.cuda}")
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            
            # Check for RTX 3080 specifically
            is_rtx_3080 = "3080" in torch.cuda.get_device_name(0)
            if is_rtx_3080:
                logger.info("RTX 3080 detected - optimal hardware configuration")
            else:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory < 8:
                    logger.warning(f"GPU memory ({gpu_memory:.1f}GB) is less than recommended 10GB")
                    logger.warning("Performance may be degraded")
                    # Not a critical issue
        else:
            logger.warning("CUDA not available, GPU acceleration disabled")
            logger.warning("Performance may be significantly reduced without GPU acceleration")
            # Not a critical issue
        return True
    except ImportError:
        logger.error("PyTorch not installed, GPU acceleration unavailable")
        logger.error("Install PyTorch with: pip install torch")
        # Not a critical issue, can run without GPU
        return True
    except Exception as e:
        logger.error(f"Error checking CUDA: {e}")
        # Not a critical issue
        return True


def check_required_directories() -> bool:
    """
    Check if required directories exist and create them if necessary.
    
    Returns
    -------
    bool
        True if all directories exist or were created, False otherwise
    """
    required_dirs = ["models", "models/tts", "logs", "recipes", "templates"]
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                error_msg = f"Failed to create directory {directory}: {e}"
                logger.error(error_msg)
                return False
    return True


def check_dependencies() -> bool:
    """
    Check if required dependencies are installed.
    
    Returns
    -------
    bool
        True if all critical dependencies are installed, False otherwise
    """
    # Check for PyAudio (common source of issues)
    pyaudio_installed = check_pyaudio()
    
    # Check for other critical dependencies
    critical_deps = [
        "pvporcupine", "faster_whisper", "ctransformers", 
        "transitions", "docx", "PyQt6"
    ]
    missing_deps = []
    
    for dep in critical_deps:
        try:
            module_name = dep.replace("-", "_")
            if dep == "PyQt6":
                module_name = "PyQt6.QtCore"
            __import__(module_name)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        deps_str = ", ".join(missing_deps)
        logger.error(f"Missing critical dependencies: {deps_str}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_pyaudio() -> bool:
    """
    Check if PyAudio is installed.
    
    Returns
    -------
    bool
        True if PyAudio is installed, False otherwise
    """
    try:
        import pyaudio
        logger.info("PyAudio found")
        return True
    except ImportError:
        error_msg = "PyAudio not installed"
        logger.error(error_msg)
        if platform.system() == "Windows":
            logger.error("On Windows, install with: pipwin install pyaudio")
        else:
            logger.error("On Linux, first install portaudio19-dev, then install pyaudio")
        return False


def create_recipe_template() -> bool:
    """
    Create the recipe template file if it doesn't exist.
    
    Returns
    -------
    bool
        True if template created or already exists, False on error
        
    Notes
    -----
    This function creates a Microsoft Word document template for recipes,
    which is used by the recipe creator utility.
    """
    template_path = "templates/recipe_template.docx"
    
    if os.path.exists(template_path):
        logger.info(f"Recipe template already exists at {template_path}")
        return True
    
    try:
        from docx import Document
        
        os.makedirs("templates", exist_ok=True)
        
        doc = Document()
        
        # Create a more detailed template
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
        
        # Add nutrition section if available
        doc.add_heading("Nutrition Information (per serving)", level=2)
        doc.add_paragraph("Calories: 000")
        doc.add_paragraph("Protein: 00g")
        doc.add_paragraph("Carbohydrates: 00g")
        doc.add_paragraph("Fat: 00g")
        
        doc.save(template_path)
        logger.info(f"Created recipe template at {template_path}")
        return True
    except ImportError:
        logger.error("python-docx not installed, cannot create template")
        return False
    except Exception as e:
        logger.error(f"Failed to create recipe template: {e}")
        return False


def optimize_system() -> bool:
    """
    Optimize system settings for best performance.
    
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
        if platform.system() == "Windows":
            windows_optimizations()
        elif platform.system() == "Linux":
            linux_optimizations()
        
        # Apply common optimizations
        common_optimizations()
        
        return True
    except Exception as e:
        logger.error(f"Failed to optimize system: {e}")
        return False


def windows_optimizations() -> None:
    """
    Apply Windows-specific optimizations.
    
    Sets process priority and CPU affinity for optimal performance
    on Windows systems.
    """
    try:
        import psutil
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Set process priority to high
        process.nice(psutil.HIGH_PRIORITY_CLASS)
        logger.info("Set process priority to high")
        
        # CPU affinity optimization for Ryzen 9 5900X
        # Use the first 8 cores (16 threads for Ryzen)
        cpu_count = psutil.cpu_count(logical=True)
        if cpu_count >= 16:  # Likely Ryzen 9 with hyperthreading
            # Create affinity mask for first 8 physical cores (16 threads)
            # For Ryzen, logical processors are arranged as:
            # 0, 2, 4, ... are first 12 cores, 1, 3, 5, ... are their hyperthreaded pairs
            affinity = list(range(16))
            process.cpu_affinity(affinity)
            logger.info(f"Set CPU affinity to first 8 physical cores: {affinity}")
    except ImportError:
        logger.warning("psutil not available for Windows optimizations")
    except Exception as e:
        logger.warning(f"Failed to apply Windows performance optimizations: {e}")


def linux_optimizations() -> None:
    """
    Apply Linux-specific optimizations.
    
    Sets CPU governor and process nice level for optimal performance
    on Linux systems.
    """
    try:
        # Check if running as root (required for some optimizations)
        is_root = os.geteuid() == 0
        
        if is_root:
            # Set CPU governor to performance
            os.system("echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
            logger.info("Set CPU governor to performance mode")
        else:
            logger.warning("Not running as root, skipping some system optimizations")
            
        # Set process nice level
        os.nice(-10)  # Higher priority (lower nice value)
        logger.info("Set process nice level to -10")
    except Exception as e:
        logger.warning(f"Failed to apply Linux performance optimizations: {e}")


def common_optimizations() -> None:
    """
    Apply common optimizations for all platforms.
    
    Sets process name and Python-specific optimizations.
    """
    try:
        # Set process name for better identification
        import setproctitle
        setproctitle.setproctitle("maggie-ai-assistant")
        logger.info("Set process title to 'maggie-ai-assistant'")
    except ImportError:
        pass
    
    # Set Python-specific optimizations
    if hasattr(sys, 'set_int_max_str_digits'):
        # Limit maximum string conversion size (Python 3.10+)
        sys.set_int_max_str_digits(4096)


def main() -> int:
    """
    Main entry point for the application.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
        
    Notes
    -----
    This function handles the main program flow, including:
    - Parsing command-line arguments
    - Setting up logging
    - Verifying system configuration
    - Creating required directories and templates
    - Initializing and starting the Maggie AI Assistant
    """
    # Parse arguments
    args = parse_arguments()
    
    # Ensure necessary directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Set up logging
    setup_logging(args.debug)
    
    # Log startup and system information
    logger.info("Starting Maggie AI Assistant")
    logger.info(f"Running on Python {platform.python_version()}")
    logger.info(f"Process ID: {os.getpid()}")
    
    # Enable multiprocessing support
    setup_multiprocessing()
    
    # Optimize system if requested
    if args.optimize:
        if optimize_system():
            logger.info("System optimized for performance")
        else:
            logger.warning("System optimization failed")
    
    # Create recipe template if requested
    if args.create_template:
        create_recipe_template()
        if not args.verify:  # Exit if only create-template was specified
            return 0
    
    # Verify system if requested
    if args.verify:
        if verify_system():
            logger.info("System verification successful")
            return 0
        else:
            logger.error("System verification failed")
            return 1
    
    # Quick system check before starting
    if not verify_system():
        logger.warning("System verification failed, but attempting to start anyway")
    
    # Start the Maggie AI Assistant
    return start_maggie(args)


def setup_multiprocessing() -> None:
    """
    Set up multiprocessing support.
    
    Configures the multiprocessing module with appropriate settings.
    """
    try:
        multiprocessing.set_start_method('spawn')
        logger.info("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        # Already set
        pass


def start_maggie(args: argparse.Namespace) -> int:
    """
    Initialize and start the Maggie AI Assistant.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Import here to avoid circular imports
        from maggie import MaggieAI
        
        # Initialize and start Maggie AI
        maggie = MaggieAI(config_path=args.config)
        
        # Register signal handlers for graceful shutdown
        register_signal_handlers(maggie)
        
        # Start Maggie
        success = maggie.start()
        
        return 0 if success else 1
    except ImportError as e:
        logger.error(f"Failed to import required module: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Error starting Maggie AI: {e}")
        return 1


def register_signal_handlers(maggie) -> None:
    """
    Register signal handlers for graceful shutdown.
    
    Parameters
    ----------
    maggie : MaggieAI
        Instance of the Maggie AI Assistant
    """
    try:
        import signal
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down gracefully")
            maggie.shutdown()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Registered signal handlers for graceful shutdown")
    except Exception as e:
        logger.warning(f"Failed to register signal handlers: {e}")


if __name__ == "__main__":
    sys.exit(main())