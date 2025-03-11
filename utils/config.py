"""
Maggie AI Assistant - Configuration Module
=======================================
Configuration handler for the Maggie AI Assistant.

This module provides functionality for loading, validating,
optimizing, and saving configuration from YAML files.
It includes special optimizations for AMD Ryzen 9 5900X
and NVIDIA GeForce RTX 3080 hardware.
"""

import os
import yaml
import platform
import psutil
from loguru import logger
from typing import Dict, Any, Optional, List, Tuple


class Config:
    """
    Configuration handler for the Maggie AI Assistant.
    
    This class handles loading, validating, and managing
    configuration settings for the Maggie AI Assistant,
    with special optimizations for Ryzen 9 5900X and RTX 3080.
    
    Attributes
    ----------
    config_path : str
        Path to the configuration file
    default_config : Dict[str, Any]
        Default configuration settings
    config : Dict[str, Any] or None
        Loaded configuration (None before load() is called)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration handler.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file, by default "config.yaml"
        """
        self.config_path = config_path
        self.default_config = {
            # Default configuration values
            "wake_word": {
                "sensitivity": 0.5,
                "keyword_path": None,
                "porcupine_access_key": ""
            },
            "speech": {
                "whisper": {
                    "model_size": "base",
                    "compute_type": "float16"  # Optimized for RTX 3080
                },
                "tts": {
                    "voice_model": "en_US-kathleen-medium",
                    "model_path": "models/tts",
                    "sample_rate": 22050
                }
            },
            "llm": {
                "model_path": "models/mistral-7b-instruct-v0.3-GPTQ-4bit",
                "model_type": "mistral",
                "gpu_layers": 32,  # Optimized for RTX 3080
                "precision": "float16"  # Best for tensor cores on RTX 3080
            },
            "inactivity_timeout": 300,  # 5 minutes
            "logging": {
                "path": "logs",
                "console_level": "INFO",
                "file_level": "DEBUG"
            },
            "utilities": {
                "recipe_creator": {
                    "output_dir": "recipes",
                    "template_path": "templates/recipe_template.docx"
                }
            },
            "threading": {
                "max_workers": 8,  # Optimized for Ryzen 9 5900X (12 cores)
                "thread_timeout": 30  # Timeout for thread operations
            },
            "memory": {
                "max_percent": 75,  # Use up to 75% of system memory for 32GB system
                "model_unload_threshold": 85  # Threshold to unload models
            }
        }
        self.config = None
        
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file or create with defaults.
        
        Returns
        -------
        Dict[str, Any]
            Loaded and optimized configuration dictionary
            
        Notes
        -----
        If the configuration file doesn't exist, it will be created
        with default values. The loaded configuration is automatically
        optimized for the detected hardware.
        """
        # Check if config file exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                logger.info("Using default configuration")
                self.config = self.default_config
        else:
            logger.info(f"Configuration file {self.config_path} not found, creating with defaults")
            self.config = self.default_config
            self.save()
            
        # Merge with defaults for any missing values
        self._merge_defaults(self.config, self.default_config)
        
        # Optimize config for hardware
        self._optimize_config()
        
        return self.config
        
    def save(self) -> bool:
        """
        Save current configuration to file.
        
        Returns
        -------
        bool
            True if saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            
            # Write config
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
                
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
            
    def _merge_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> None:
        """
        Recursively merge default values into configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to update
        defaults : Dict[str, Any]
            Default values dictionary
            
        Notes
        -----
        This method recursively merges default values into the configuration,
        preserving any existing values in the configuration.
        """
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                self._merge_defaults(config[key], value)
                
    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Detect hardware capabilities for optimization.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with hardware information
            
        Notes
        -----
        This method detects CPU, memory, and GPU information
        to optimize configuration accordingly.
        """
        hw_info = {}
        
        # Detect CPU
        hw_info["cpu"] = {
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
        }
        
        # Look for Ryzen 9 5900X
        cpu_model = platform.processor().lower()
        hw_info["cpu"]["is_ryzen_9"] = "ryzen 9" in cpu_model or "5900x" in cpu_model
        
        # Detect memory
        vm = psutil.virtual_memory()
        hw_info["memory"] = {
            "total_gb": vm.total / (1024**3),
            "is_32gb": 30 <= vm.total / (1024**3) <= 34  # 32GB with some tolerance
        }
        
        # Detect GPU
        hw_info["gpu"] = {"available": False, "is_rtx_3080": False}
        try:
            import torch
            if torch.cuda.is_available():
                hw_info["gpu"]["available"] = True
                hw_info["gpu"]["name"] = torch.cuda.get_device_name(0)
                hw_info["gpu"]["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                hw_info["gpu"]["is_rtx_3080"] = "3080" in torch.cuda.get_device_name(0)
        except ImportError:
            pass
            
        return hw_info
                
    def _optimize_config(self) -> None:
        """
        Optimize configuration for the detected hardware.
        
        This method applies specific optimizations for:
        - AMD Ryzen 9 5900X (threading)
        - 32GB RAM systems (memory allocation)
        - NVIDIA RTX 3080 (GPU acceleration)
        """
        # Detect hardware
        hw = self._detect_hardware()
        
        # Optimize for Ryzen 9 5900X
        if hw["cpu"]["is_ryzen_9"] or hw["cpu"]["cores_physical"] >= 12:
            logger.info("Applying optimizations for Ryzen 9 CPU")
            self.config.setdefault("threading", {})
            self.config["threading"]["max_workers"] = 8  # Use 8 of 12 cores for optimal performance
            self.config["threading"]["thread_timeout"] = 30  # 30-second timeout for safety
        
        # Optimize for 32GB RAM
        if hw["memory"]["is_32gb"] or hw["memory"]["total_gb"] >= 32:
            logger.info("Applying optimizations for 32GB system")
            self.config.setdefault("memory", {})
            self.config["memory"]["max_percent"] = 75  # Use up to 75% of memory (24GB)
            self.config["memory"]["model_unload_threshold"] = 85  # Unload at 85% (27GB)
        
        # Optimize for RTX 3080
        if hw["gpu"]["is_rtx_3080"] or (9.5 <= hw["gpu"].get("memory_gb", 0) <= 10.5):
            logger.info("Applying optimizations for RTX 3080 GPU")
            
            # LLM optimizations
            self.config.setdefault("llm", {})
            self.config["llm"]["gpu_layers"] = 32  # Optimal for 10GB VRAM
            self.config["llm"]["precision"] = "float16"  # Best performance on RTX 3080
            self.config["llm"]["gpu_layer_auto_adjust"] = True
            
            # Speech optimizations
            self.config.setdefault("speech", {})
            self.config["speech"].setdefault("whisper", {})
            self.config["speech"]["whisper"]["compute_type"] = "float16"
            # Small model is a good balance for RTX 3080
            self.config["speech"]["whisper"]["model_size"] = "small"
        
        # Set wake word detector to use minimal resources
        self.config["wake_word"].setdefault("cpu_threshold", 5.0)  # Max 5% CPU usage