"""
Maggie AI Assistant - Configuration Module
=======================================
Configuration handler for the Maggie AI Assistant.
Loads and validates configuration from YAML files.
"""

import os
import yaml
from loguru import logger
from typing import Dict, Any, Optional

class Config:
    """
    Configuration handler for the Maggie AI Assistant.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration handler.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file
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
                "gpu_layers": 32  # Optimized for RTX 3080
            },
            "inactivity_timeout": 300,  # 5 minutes
            "logging": {
                "path": "logs",
                "console_level": "INFO"
            },
            "utilities": {
                "recipe_creator": {
                    "output_dir": "recipes",
                    "template_path": "templates/recipe_template.docx"
                }
            }
        }
        self.config = None
        
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file or create with defaults.
        
        Returns
        -------
        Dict[str, Any]
            Loaded configuration dictionary
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
            True if saved successfully
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
        """
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                self._merge_defaults(config[key], value)
                
    def _optimize_config(self) -> None:
        """Optimize configuration for the hardware."""
        # Set optimal parameters for RTX 3080
        self.config["speech"]["whisper"]["compute_type"] = "float16"
        self.config["llm"]["gpu_layers"] = 32
        
        # Ensure wake word detector uses minimal resources
        self.config["wake_word"].setdefault("cpu_threshold", 5.0)  # Percent of CPU usage
        
        # Multi-threading settings for Ryzen 9 5900X
        self.config.setdefault("threading", {})
        self.config["threading"]["max_workers"] = 8  # Conservatively use 8 of the 12 available cores
        
        # Memory limits for 32GB system
        self.config.setdefault("memory", {})
        self.config["memory"]["max_percent"] = 75  # Use up to 75% of system memory (24GB)
