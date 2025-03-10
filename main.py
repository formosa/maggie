#!/usr/bin/env python3
"""
Maggie AI Assistant - Main Script
===============================
Entry point for the Maggie AI Assistant application.

This script initializes and starts the Maggie AI Assistant.
It's optimized for AMD Ryzen 9 5900X and NVIDIA GeForce RTX 3080 hardware.
"""

import os
import sys
import argparse
import platform
from loguru import logger

# Import main application class
from maggie import MaggieAI


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Maggie AI Assistant")
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
    return parser.parse_args()


def setup_logging(debug=False):
    """
    Set up logging configuration.
    
    Parameters
    ----------
    debug : bool, optional
        Enable debug logging if True
    """
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Configure loguru
    log_level = "DEBUG" if debug else "INFO"
    logger.configure(
        handlers=[
            {"sink": sys.stdout, "level": log_level},
            {"sink": "logs/maggie.log", "rotation": "10 MB", "retention": "1 week"}
        ]
    )
    
    # Log system info
    try:
        import platform
        import psutil
        import torch
        
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"CPU: {platform.processor()}")
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        logger.info(f"RAM: {ram_gb:.2f} GB")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                logger.info(f"GPU Memory: {memory_gb:.2f} GB")
        else:
            logger.warning("CUDA not available, GPU acceleration disabled")
            
    except ImportError as e:
        logger.warning(f"System info modules not available: {e}")


def verify_system():
    """
    Verify system meets requirements for Maggie.
    
    Returns
    -------
    bool
        True if system meets requirements, False otherwise
    """
    logger.info("Verifying system configuration...")
    
    # Check Python version - require exactly 3.10.x
    python_version = platform.python_version_tuple()
    if int(python_version[0]) != 3 or int(python_version[1]) != 10:
        logger.error(f"Unsupported Python version: {platform.python_version()}")
        logger.error("Maggie requires Python 3.10.x specifically. Other versions are not compatible.")
        logger.error("Please install Python 3.10 and try again.")
        return False
    
    try:
        # Check for CUDA
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, GPU acceleration disabled")
            logger.warning("Performance may be significantly reduced without GPU acceleration")
        else:
            logger.info(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        logger.error("PyTorch not installed, GPU acceleration unavailable")
        logger.error("Install PyTorch with: pip install torch")
        return False
    
    # Check for required directories
    required_dirs = ["models", "models/tts", "logs", "recipes", "templates"]
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                return False
    
    # Check for PyAudio (common source of issues)
    try:
        import pyaudio
        logger.info("PyAudio found")
    except ImportError:
        logger.error("PyAudio not installed")
        if platform.system() == "Windows":
            logger.error("On Windows, install with: pipwin install pyaudio")
        else:
            logger.error("On Linux, first install portaudio19-dev, then install pyaudio")
        return False
    
    # Check for other critical dependencies
    critical_deps = [
        "pvporcupine", "faster_whisper", "ctransformers", 
        "transitions", "docx", "PyQt6"
    ]
    missing_deps = []
    
    for dep in critical_deps:
        try:
            __import__(dep.replace("-", "_").replace("PyQt6", "PyQt6.QtCore"))
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        logger.error(f"Missing critical dependencies: {', '.join(missing_deps)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("System verification completed successfully")
    return True


def create_recipe_template():
    """
    Create the recipe template file if it doesn't exist.
    
    Returns
    -------
    bool
        True if template created or already exists, False on error
    """
    template_path = "templates/recipe_template.docx"
    
    if os.path.exists(template_path):
        logger.info(f"Recipe template already exists at {template_path}")
        return True
    
    try:
        from docx import Document
        
        os.makedirs("templates", exist_ok=True)
        
        doc = Document()
        doc.add_heading("Recipe Name", level=1)
        doc.add_heading("Ingredients", level=2)
        doc.add_paragraph("• Ingredient 1", style='ListBullet')
        doc.add_paragraph("• Ingredient 2", style='ListBullet')
        doc.add_heading("Instructions", level=2)
        doc.add_paragraph("1. Step 1")
        doc.add_paragraph("2. Step 2")
        doc.add_heading("Notes", level=2)
        doc.add_paragraph("Add any additional notes here.")
        
        doc.save(template_path)
        logger.info(f"Created recipe template at {template_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create recipe template: {e}")
        return False


def main():
    """
    Main entry point for the application.
    
    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    args = parse_arguments()
    
    # Ensure necessary directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Set up logging
    setup_logging(args.debug)
    
    # Log startup
    logger.info("Starting Maggie AI Assistant")
    
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
    
    try:
        # Initialize and start Maggie AI
        maggie = MaggieAI(config_path=args.config)
        exit_code = maggie.start()
        
        return exit_code
    except Exception as e:
        logger.error(f"Error starting Maggie AI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())