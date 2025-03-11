#!/usr/bin/env python3
"""
Maggie AI Assistant - Comprehensive Setup and Resource Management

This module provides an advanced, automated installation and resource 
acquisition system for the Maggie AI Assistant project.

Key Features:
- Intelligent dependency management
- Automated model and resource downloads
- GPU and system optimization
- Comprehensive error handling
- Flexible configuration options

Dependencies:
- Python 3.10+
- setuptools
- wheel
- requests
- huggingface_hub
- torch
- transformers

Author: Maggie Development Team
Version: 0.2.0
"""

import os
import sys
import json
import shutil
import platform
import subprocess
import requests
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    from setuptools import setup, find_packages
    import torch
    import huggingface_hub
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                            'setuptools', 'wheel', 'requests', 
                            'huggingface_hub', 'torch'])

class MaggieResourceManager:
    """
    Comprehensive resource management and download utility for Maggie AI Assistant.

    Handles intelligent acquisition of AI models, TTS voices, and other 
    project-critical resources with robust error handling and optimization.

    Attributes
    ----------
    _base_dir : Path
        Base directory for resource downloads
    _config : Dict
        Configuration dictionary for resource locations
    _system_info : Dict
        Detected system hardware and configuration details
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the Maggie Resource Manager.

        Parameters
        ----------
        base_dir : Path, optional
            Base directory for resource downloads. 
            Defaults to './models' if not specified.
        """
        self._base_dir = base_dir or Path('./models')
        self._base_dir.mkdir(parents=True, exist_ok=True)
        
        self._config = self._load_resource_config()
        self._system_info = self._detect_system_configuration()

    def _load_resource_config(self) -> Dict:
        """
        Load resource configuration from a JSON file.

        Returns
        -------
        Dict
            Resource configuration dictionary
        """
        config_path = Path('resource_config.json')
        default_config = {
            "models": {
                "llm": {
                    "name": "TheBloke/Mistral-7B-Instruct-v0.3-GPTQ",
                    "path": "models/mistral-7b-instruct"
                },
                "tts": {
                    "name": "rhasspy/piper-voices",
                    "voice": "en_US/kathleen/medium",
                    "path": "models/tts/en_US-kathleen-medium"
                }
            }
        }

        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return default_config

    def _detect_system_configuration(self) -> Dict:
        """
        Detect and analyze system hardware and configuration.

        Returns
        -------
        Dict
            Comprehensive system configuration details
        """
        system_info = {
            "os": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine()
        }

        try:
            import torch
            system_info.update({
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cuda_device_capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None
            })
        except ImportError:
            system_info.update({
                "cuda_available": False,
                "cuda_device_name": None,
                "cuda_device_capability": None
            })

        return system_info

    def download_model(self, model_type: str = 'llm') -> Path:
        """
        Download and configure AI models with intelligent management.

        Parameters
        ----------
        model_type : str, optional
            Type of model to download (default: 'llm')

        Returns
        -------
        Path
            Path to the downloaded model
        
        Raises
        ------
        RuntimeError
            If model download fails
        """
        try:
            model_config = self._config['models'].get(model_type, {})
            model_name = model_config.get('name')
            model_path = self._base_dir / model_config.get('path', f'{model_type}_model')
            
            if not model_path.exists():
                print(f"Downloading {model_type} model: {model_name}")
                huggingface_hub.snapshot_download(
                    repo_id=model_name, 
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )
            
            return model_path
        
        except Exception as e:
            print(f"Error downloading {model_type} model: {e}")
            raise RuntimeError(f"Model download failed for {model_type}")

    def download_tts_voice(self) -> Path:
        """
        Download Text-to-Speech voice model with advanced configuration.

        Returns
        -------
        Path
            Path to the downloaded TTS voice model
        """
        tts_config = self._config['models'].get('tts', {})
        voice_name = tts_config.get('voice', 'en_US/kathleen/medium')
        tts_path = self._base_dir / tts_config.get('path', 'models/tts')
        
        # Construct download URLs dynamically
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
        
        files_to_download = [
            f"{base_url}/{voice_name}.onnx",
            f"{base_url}/{voice_name}.json"
        ]
        
        tts_path.mkdir(parents=True, exist_ok=True)
        
        for file_url in files_to_download:
            filename = Path(file_url).name
            target_path = tts_path / filename
            
            if not target_path.exists():
                print(f"Downloading TTS file: {filename}")
                response = requests.get(file_url, stream=True)
                response.raise_for_status()
                
                with open(target_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
        
        return tts_path

def get_long_description() -> str:
    """
    Read project long description from README.

    Returns
    -------
    str
        Long description for package metadata
    """
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Maggie AI Assistant - An advanced, intelligent assistant"

setup(
    name='maggie-ai',
    version='0.2.0',
    author='Maggie Development Team',
    author_email='contact@maggieai.com',
    description='An intelligent, configurable AI assistant',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/maggie',
    packages=find_packages(exclude=['tests*', 'docs*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.1',
        'transformers>=4.30.0',
        'huggingface_hub',
        'requests',
        'pyyaml',
        'numpy',
    ],
    extras_require={
        'gpu': [
            'nvidia-cudnn-cu11',
            'torchvision',
            'torchaudio'
        ],
        'dev': [
            'pytest',
            'sphinx',
            'sphinx-napoleon'
        ]
    },
    entry_points={
        'console_scripts': [
            'maggie=maggie.cli:main',
        ],
    },
)

def main():
    """
    Primary entry point for Maggie AI installation and configuration.
    Orchestrates resource management and system optimization.
    """
    resource_manager = MaggieResourceManager()
    
    print("🚀 Maggie AI Resource Configuration")
    print(f"System Configuration: {json.dumps(resource_manager._system_info, indent=2)}")
    
    try:
        # Automated resource downloads
        resource_manager.download_model('llm')
        resource_manager.download_model('tts')
        resource_manager.download_tts_voice()
        
        print("✅ All resources successfully downloaded and configured.")
    
    except Exception as e:
        print(f"❌ Resource configuration failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()