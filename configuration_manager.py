"""
Maggie AI Assistant - Configuration Manager
========================================
Unified configuration management for Maggie AI.

This module provides comprehensive configuration management including
loading, validation, and optimization, with specific enhancements
for AMD Ryzen 9 5900X and NVIDIA RTX 3080.
"""

import os
import yaml
import json
import shutil
import time
from typing import Dict, Any, Optional, List, Tuple, Set
from loguru import logger

class ConfigManager:
    """
    Unified configuration management system.
    
    Handles configuration loading, validation, optimization, 
    and persistence with robust error handling and recovery.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file, by default "config.yaml"
    backup_dir : str, optional
        Directory for configuration backups, by default "config_backups"
    
    Attributes
    ----------
    config : Dict[str, Any]
        Current configuration dictionary
    config_path : str
        Path to the configuration file
    backup_dir : str
        Directory for configuration backups
    validation_errors : List[str]
        List of validation errors
    validation_warnings : List[str]
        List of validation warnings
    """
    
    def __init__(self, config_path: str = "config.yaml", backup_dir: str = "config_backups"):
        """
        Initialize the configuration manager.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file, by default "config.yaml"
        backup_dir : str, optional
            Directory for configuration backups, by default "config_backups"
        """
        self.config_path = config_path
        self.backup_dir = backup_dir
        self.config = {}
        self.validation_errors = []
        self.validation_warnings = []
        
        # Default configuration with optimal settings for Ryzen 9 5900X and RTX 3080
        self.default_config = self._create_default_config()
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration with optimal settings.
        
        Returns
        -------
        Dict[str, Any]
            Default configuration dictionary
        """
        return {
            "wake_word": {
                "sensitivity": 0.5,
                "keyword_path": None,
                "porcupine_access_key": "",
                "cpu_threshold": 5.0  # Maximum CPU usage percentage
            },
            "speech": {
                "whisper": {
                    "model_size": "base",
                    "compute_type": "float16"  # Optimized for RTX 3080
                },
                "tts": {
                    "voice_model": "en_US-kathleen-medium",
                    "model_path": "models/tts",
                    "sample_rate": 22050,
                    "use_cache": True,
                    "cache_dir": "cache/tts",
                    "cache_size": 100
                }
            },
            "llm": {
                "model_path": "models/mistral-7b-instruct-v0.3-GPTQ-4bit",
                "model_type": "mistral",
                "gpu_layers": 32,  # Optimized for RTX 3080 10GB
                "gpu_layer_auto_adjust": True,
                "precision": "float16"  # Best for tensor cores on RTX 3080
            },
            "inactivity_timeout": 300,  # 5 minutes
            "logging": {
                "path": "logs",
                "console_level": "INFO",
                "file_level": "DEBUG",
                "max_size": "10MB",
                "retention": "1 week"
            },
            "utilities": {
                "recipe_creator": {
                    "output_dir": "recipes",
                    "template_path": "templates/recipe_template.docx"
                }
            },
            "threading": {
                "max_workers": 8,  # Optimized for Ryzen 9 5900X (12 cores)
                "thread_timeout": 30
            },
            "memory": {
                "max_percent": 75,  # Use up to 75% of system memory (24GB)
                "model_unload_threshold": 85
            },
            "gpu": {
                "enabled": True,
                "compute_type": "float16",
                "tensor_cores": True,
                "reserved_memory_mb": 512
            }
        }
        
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
                    self.config = yaml.safe_load(file) or {}
                logger.info(f"Configuration loaded from {self.config_path}")
                
                # Create backup of working configuration
                self._create_backup("loaded")
                
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                
                # Try to recover from backup
                backup_path = self._find_latest_backup()
                if backup_path:
                    logger.info(f"Attempting to recover from backup: {backup_path}")
                    try:
                        with open(backup_path, 'r') as file:
                            self.config = yaml.safe_load(file) or {}
                        logger.info(f"Configuration recovered from backup: {backup_path}")
                    except Exception as recover_e:
                        logger.error(f"Failed to recover from backup: {recover_e}")
                        self.config = self.default_config
                        logger.info("Using default configuration")
                else:
                    self.config = self.default_config
                    logger.info("Using default configuration")
        else:
            logger.info(f"Configuration file {self.config_path} not found, creating with defaults")
            self.config = self.default_config
            self.save()
        
        # Merge with defaults for any missing values
        self._merge_with_defaults()
        
        # Validate the configuration
        self.validate()
        
        # Return the config
        return self.config
        
    def _merge_with_defaults(self) -> None:
        """
        Recursively merge defaults into configuration to ensure completeness.
        """
        def deep_merge(target, source):
            """
            Recursively merge source dictionary into target dictionary.
            
            Parameters
            ----------
            target : Dict
                Target dictionary to merge into
            source : Dict
                Source dictionary with values to merge
            """
            for key, value in source.items():
                if key not in target:
                    target[key] = value
                elif isinstance(value, dict) and isinstance(target[key], dict):
                    deep_merge(target[key], value)
        
        deep_merge(self.config, self.default_config)
        
    def save(self) -> bool:
        """
        Save configuration to file.
        
        Returns
        -------
        bool
            True if saved successfully, False otherwise
        """
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            
            # Create backup before saving
            self._create_backup("before_save")
            
            # Write config
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
                
            logger.info(f"Configuration saved to {self.config_path}")
            
            # Create backup after successful save
            self._create_backup("after_save")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
            
    def _create_backup(self, reason: str) -> Optional[str]:
        """
        Create a backup of the current configuration.
        
        Parameters
        ----------
        reason : str
            Reason for creating the backup
            
        Returns
        -------
        Optional[str]
            Path to the created backup file, or None if backup failed
        """
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(
                self.backup_dir, 
                f"config_{timestamp}_{reason}.yaml"
            )
            
            # Ensure backup directory exists
            os.makedirs(self.backup_dir, exist_ok=True)
            
            # Write backup
            with open(backup_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
                
            logger.debug(f"Configuration backup created: {backup_path}")
            
            # Cleanup old backups (keep most recent 10)
            self._cleanup_old_backups()
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating configuration backup: {e}")
            return None
            
    def _find_latest_backup(self) -> Optional[str]:
        """
        Find the most recent backup file.
        
        Returns
        -------
        Optional[str]
            Path to the most recent backup file, or None if no backups found
        """
        try:
            if not os.path.exists(self.backup_dir):
                return None
                
            backup_files = [
                os.path.join(self.backup_dir, f) 
                for f in os.listdir(self.backup_dir) 
                if f.startswith("config_") and f.endswith(".yaml")
            ]
            
            if not backup_files:
                return None
                
            # Sort by modification time (most recent first)
            backup_files.sort(key=os.path.getmtime, reverse=True)
            
            return backup_files[0]
            
        except Exception as e:
            logger.error(f"Error finding latest backup: {e}")
            return None
            
    def _cleanup_old_backups(self, keep: int = 10) -> None:
        """
        Remove old backup files, keeping the most recent ones.
        
        Parameters
        ----------
        keep : int, optional
            Number of most recent backups to keep, by default 10
        """
        try:
            if not os.path.exists(self.backup_dir):
                return
                
            backup_files = [
                os.path.join(self.backup_dir, f) 
                for f in os.listdir(self.backup_dir) 
                if f.startswith("config_") and f.endswith(".yaml")
            ]
            
            if len(backup_files) <= keep:
                return
                
            # Sort by modification time (most recent first)
            backup_files.sort(key=os.path.getmtime, reverse=True)
            
            # Remove old backups
            for old_backup in backup_files[keep:]:
                os.remove(old_backup)
                logger.debug(f"Removed old backup: {old_backup}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
            
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Performs comprehensive validation of the configuration,
        checking required parameters, path existence, and value ranges.
        
        Returns
        -------
        bool
            True if configuration is valid (no errors), False otherwise
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check required parameters
        self._validate_required_params()
        
        # Check paths
        self._validate_paths()
        
        # Check value ranges
        self._validate_value_ranges()
        
        # Log validation results
        for error in self.validation_errors:
            logger.error(f"Configuration error: {error}")
            
        for warning in self.validation_warnings:
            logger.warning(f"Configuration warning: {warning}")
            
        return len(self.validation_errors) == 0
        
    def _validate_required_params(self) -> None:
        """
        Validate required configuration parameters.
        """
        required_params = [
            ("wake_word.porcupine_access_key", "Picovoice access key"),
            ("llm.model_path", "LLM model path"),
            ("speech.tts.voice_model", "TTS voice model")
        ]
        
        for param_path, param_name in required_params:
            value = self._get_nested_value(self.config, param_path)
            
            if value is None:
                self.validation_errors.append(f"Missing required configuration: {param_name} ({param_path})")
            elif isinstance(value, str) and not value:
                self.validation_errors.append(f"Empty required configuration: {param_name} ({param_path})")
                
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """
        Get a nested value from the configuration using dot notation.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        path : str
            Path to the value using dot notation (e.g., "wake_word.sensitivity")
            
        Returns
        -------
        Any
            Value at the specified path, or None if not found
        """
        parts = path.split(".")
        current = config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
            
        return current
        
    def _validate_paths(self) -> None:
        """
        Validate file and directory paths in the configuration.
        """
        # Check model paths
        model_paths = [
            (self._get_nested_value(self.config, "llm.model_path"), "LLM model directory", True),
            (os.path.join(
                self._get_nested_value(self.config, "speech.tts.model_path") or "models/tts",
                self._get_nested_value(self.config, "speech.tts.voice_model") or ""
            ), "TTS voice model directory", True)
        ]
        
        for path, name, is_dir in model_paths:
            if not path:
                continue
                
            if not os.path.exists(path):
                self.validation_warnings.append(f"{name} path does not exist: {path}")
                # Try to create directory if it's a directory path
                if is_dir:
                    try:
                        os.makedirs(path, exist_ok=True)
                        logger.info(f"Created directory for {name}: {path}")
                    except Exception as e:
                        self.validation_warnings.append(f"Could not create directory for {name}: {e}")
            elif is_dir and not os.path.isdir(path):
                self.validation_errors.append(f"{name} path is not a directory: {path}")
            elif not is_dir and os.path.isdir(path):
                self.validation_errors.append(f"{name} path is a directory, not a file: {path}")
                
        # Check utility paths
        utilities = self.config.get("utilities", {})
        for utility_name, utility_config in utilities.items():
            if "output_dir" in utility_config:
                output_dir = utility_config["output_dir"]
                if not os.path.exists(output_dir):
                    self.validation_warnings.append(f"{utility_name} output directory does not exist: {output_dir}")
                    # Try to create the directory
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                        logger.info(f"Created {utility_name} output directory: {output_dir}")
                    except Exception as e:
                        self.validation_errors.append(f"Failed to create {utility_name} output directory: {e}")
            
            if "template_path" in utility_config:
                template_path = utility_config["template_path"]
                template_dir = os.path.dirname(template_path)
                if not os.path.exists(template_dir):
                    self.validation_warnings.append(f"{utility_name} template directory does not exist: {template_dir}")
                    # Try to create the directory
                    try:
                        os.makedirs(template_dir, exist_ok=True)
                        logger.info(f"Created {utility_name} template directory: {template_dir}")
                    except Exception as e:
                        self.validation_errors.append(f"Failed to create {utility_name} template directory: {e}")
                        
    def _validate_value_ranges(self) -> None:
        """
        Validate that configuration values are within acceptable ranges.
        """
        # Wake word sensitivity (0.0-1.0)
        sensitivity = self._get_nested_value(self.config, "wake_word.sensitivity")
        if sensitivity is not None:
            if not isinstance(sensitivity, (int, float)):
                self.validation_errors.append(f"Wake word sensitivity must be a number, got {type(sensitivity).__name__}")
            elif sensitivity < 0.0 or sensitivity > 1.0:
                self.validation_errors.append(f"Wake word sensitivity must be between 0.0 and 1.0, got {sensitivity}")
                
        # Whisper model size
        model_size = self._get_nested_value(self.config, "speech.whisper.model_size")
        if model_size is not None:
            valid_sizes = ["tiny", "base", "small", "medium", "large"]
            if model_size not in valid_sizes:
                self.validation_errors.append(f"Invalid whisper model size: {model_size}. Valid values: {', '.join(valid_sizes)}")
                
        # Compute type
        compute_type = self._get_nested_value(self.config, "speech.whisper.compute_type")
        if compute_type is not None:
            valid_types = ["int8", "float16", "float32"]
            if compute_type not in valid_types:
                self.validation_errors.append(f"Invalid compute type: {compute_type}. Valid values: {', '.join(valid_types)}")
                
        # Threading max_workers
        max_workers = self._get_nested_value(self.config, "threading.max_workers")
        if max_workers is not None:
            import os
            cpu_count = os.cpu_count() or 4
            
            if not isinstance(max_workers, int):
                self.validation_errors.append(f"max_workers must be an integer, got {type(max_workers).__name__}")
            elif max_workers < 1:
                self.validation_errors.append(f"max_workers must be at least 1, got {max_workers}")
            elif max_workers > cpu_count * 2:
                self.validation_warnings.append(f"max_workers ({max_workers}) exceeds twice the number of CPU cores ({cpu_count})")
                
        # Memory max_percent
        max_percent = self._get_nested_value(self.config, "memory.max_percent")
        if max_percent is not None:
            if not isinstance(max_percent, (int, float)):
                self.validation_errors.append(f"memory.max_percent must be a number, got {type(max_percent).__name__}")
            elif max_percent < 10:
                self.validation_errors.append(f"memory.max_percent must be at least 10, got {max_percent}")
            elif max_percent > 95:
                self.validation_errors.append(f"memory.max_percent must be at most 95, got {max_percent}")
                
        # Memory unload_threshold
        unload_threshold = self._get_nested_value(self.config, "memory.model_unload_threshold")
        if unload_threshold is not None and max_percent is not None:
            if not isinstance(unload_threshold, (int, float)):
                self.validation_errors.append(f"memory.model_unload_threshold must be a number, got {type(unload_threshold).__name__}")
            elif unload_threshold <= max_percent:
                self.validation_errors.append(f"memory.model_unload_threshold ({unload_threshold}) must be greater than memory.max_percent ({max_percent})")
                
    def apply_hardware_optimizations(self, hardware_info: Dict[str, Any]) -> None:
        """
        Apply hardware-specific optimizations to the configuration.
        
        Parameters
        ----------
        hardware_info : Dict[str, Any]
            Hardware information dictionary
        """
        # CPU optimizations for Ryzen 9 5900X
        cpu_info = hardware_info.get("cpu", {})
        if cpu_info.get("is_ryzen_9_5900x", False):
            logger.info("Applying Ryzen 9 5900X optimizations")
            
            # Threading optimizations
            self.config.setdefault("threading", {})
            self.config["threading"]["max_workers"] = 8  # Optimal for 12-core Ryzen 9
            self.config["threading"]["thread_timeout"] = 30
            
            # Process priority optimizations
            self.config.setdefault("process", {})
            self.config["process"]["priority"] = "above_normal"
            self.config["process"]["affinity"] = "performance_cores"
            
        # Memory optimizations for 32GB RAM
        memory_info = hardware_info.get("memory", {})
        if memory_info.get("is_32gb", False):
            logger.info("Applying 32GB RAM optimizations")
            
            self.config.setdefault("memory", {})
            self.config["memory"]["max_percent"] = 75  # Use up to 75% of RAM (24GB)
            self.config["memory"]["model_unload_threshold"] = 85  # Unload at 85% (27GB)
            self.config["memory"]["cache_size_mb"] = 4096  # 4GB cache
            
        # GPU optimizations for RTX 3080
        gpu_info = hardware_info.get("gpu", {})
        if gpu_info.get("is_rtx_3080", False):
            logger.info("Applying RTX 3080 optimizations")
            
            # LLM optimizations
            self.config.setdefault("llm", {})
            self.config["llm"]["gpu_layers"] = 32  # Optimal for 10GB VRAM
            self.config["llm"]["precision"] = "float16"  # Best for Tensor Cores
            self.config["llm"]["gpu_layer_auto_adjust"] = True
            
            # Whisper optimizations
            self.config.setdefault("speech", {})
            self.config["speech"].setdefault("whisper", {})
            self.config["speech"]["whisper"]["compute_type"] = "float16"
            self.config["speech"]["whisper"]["model_size"] = "small"  # Balance of accuracy and speed
            
            # GPU-specific optimizations
            self.config.setdefault("gpu", {})
            self.config["gpu"]["enabled"] = True
            self.config["gpu"]["compute_type"] = "float16"
            self.config["gpu"]["tensor_cores"] = True
            self.config["gpu"]["reserved_memory_mb"] = 512