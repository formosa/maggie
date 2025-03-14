"""
Maggie AI Assistant - Configuration Validator
========================================
Configuration validation utilities for Maggie AI Assistant.

This module provides validation functionality for configuration files,
ensuring that all required settings are present and have valid values.

Examples
--------
>>> from utils.config_validator import ConfigValidator
>>> validator = ConfigValidator("config.yaml")
>>> valid = validator.validate()
>>> if not valid:
...     print("Configuration errors:", validator.errors)
>>> is_valid, errors, warnings = validator.get_validation_result()
"""

# Standard library imports
import os
import yaml
from typing import Dict, Any, List, Tuple, Optional

# Third-party imports
from loguru import logger

__all__ = ['ConfigValidator']

class ConfigValidator:
    """
    Configuration validator for Maggie AI Assistant.
    
    Validates configuration files to ensure they contain all required
    settings with appropriate values.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Attributes
    ----------
    config_path : str
        Path to the configuration file
    config : Dict[str, Any]
        Loaded configuration dictionary
    errors : List[str]
        List of validation errors
    warnings : List[str]
        List of validation warnings
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration validator.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.errors = []
        self.warnings = []
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns
        -------
        Dict[str, Any]
            The loaded configuration dictionary
            
        Raises
        ------
        FileNotFoundError
            If the configuration file doesn't exist
        yaml.YAMLError
            If the configuration file contains invalid YAML
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                if self.config is None:
                    self.config = {}
                return self.config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")
            
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Performs comprehensive validation checks on the configuration.
        
        Returns
        -------
        bool
            True if the configuration is valid, False otherwise
        """
        # Reset error and warning lists
        self.errors = []
        self.warnings = []
        
        # Load the configuration if it hasn't been loaded already
        if not self.config:
            try:
                self.load_config()
            except Exception as e:
                self.errors.append(str(e))
                return False
                
        # Validate required sections
        self._validate_required_sections()
        
        # Validate wake word settings
        self._validate_wake_word()
        
        # Validate speech settings
        self._validate_speech()
        
        # Validate LLM settings
        self._validate_llm()
        
        # Validate path existence
        self._validate_paths()
        
        # Validate system settings
        self._validate_system_settings()
        
        # Report errors and warnings
        for error in self.errors:
            logger.error(f"Configuration error: {error}")
            
        for warning in self.warnings:
            logger.warning(f"Configuration warning: {warning}")
            
        # Return True if no errors, False otherwise
        return len(self.errors) == 0
        
    def _validate_required_sections(self) -> None:
        """
        Validate that all required configuration sections are present.
        
        Adds errors to self.errors if required sections are missing.
        """
        required_sections = [
            "wake_word", "speech", "llm", "logging"
        ]
        
        for section in required_sections:
            if section not in self.config:
                self.errors.append(f"Missing required section: {section}")
                
    def _validate_wake_word(self) -> None:
        """
        Validate wake word configuration.
        
        Checks that required wake word settings are present and valid.
        Adds errors to self.errors for invalid settings.
        """
        if "wake_word" not in self.config:
            return
            
        wake_word = self.config["wake_word"]
        
        # Check for required settings
        if "porcupine_access_key" not in wake_word:
            self.errors.append("Missing required setting: wake_word.porcupine_access_key")
        elif not wake_word["porcupine_access_key"]:
            self.errors.append("Empty wake_word.porcupine_access_key - you must obtain a key from Picovoice console")
            
        # Check sensitivity range
        if "sensitivity" in wake_word:
            sensitivity = wake_word["sensitivity"]
            if not isinstance(sensitivity, (int, float)):
                self.errors.append(f"wake_word.sensitivity must be a number, got {type(sensitivity).__name__}")
            elif sensitivity < 0.0 or sensitivity > 1.0:
                self.errors.append(f"wake_word.sensitivity must be between 0.0 and 1.0, got {sensitivity}")
                
    def _validate_speech(self) -> None:
        """
        Validate speech configuration.
        
        Checks that speech settings like whisper model and TTS settings are valid.
        Adds errors to self.errors for invalid settings.
        """
        if "speech" not in self.config:
            return
            
        speech = self.config["speech"]
        
        # Validate whisper settings
        if "whisper" in speech:
            whisper = speech["whisper"]
            
            # Check model size
            if "model_size" in whisper:
                model_size = whisper["model_size"]
                valid_sizes = ["tiny", "base", "small", "medium", "large"]
                if model_size not in valid_sizes:
                    self.errors.append(f"Invalid speech.whisper.model_size: {model_size}. Valid values: {', '.join(valid_sizes)}")
                    
            # Check compute type
            if "compute_type" in whisper:
                compute_type = whisper["compute_type"]
                valid_types = ["int8", "float16", "float32"]
                if compute_type not in valid_types:
                    self.errors.append(f"Invalid speech.whisper.compute_type: {compute_type}. Valid values: {', '.join(valid_types)}")
                    
        # Validate TTS settings
        if "tts" in speech:
            tts = speech["tts"]
            
            # Check voice model
            if "voice_model" not in tts:
                self.errors.append("Missing required setting: speech.tts.voice_model")
                
            # Check model path
            if "model_path" in tts:
                model_path = tts["model_path"]
                if not os.path.exists(model_path):
                    self.warnings.append(f"TTS model path does not exist: {model_path}")
                    
    def _validate_llm(self) -> None:
        """
        Validate LLM configuration.
        
        Checks that LLM settings like model path and type are valid.
        Adds errors to self.errors for invalid settings.
        """
        if "llm" not in self.config:
            return
            
        llm = self.config["llm"]
        
        # Check for required settings
        if "model_path" not in llm:
            self.errors.append("Missing required setting: llm.model_path")
        elif not os.path.exists(llm["model_path"]):
            self.warnings.append(f"LLM model path does not exist: {llm['model_path']}")
            
        # Check model type
        if "model_type" in llm and llm["model_type"] not in ["mistral", "llama2", "phi"]:
            self.warnings.append(f"Unusual llm.model_type: {llm['model_type']}. Common values: mistral, llama2, phi")
            
        # Check GPU layers
        if "gpu_layers" in llm:
            gpu_layers = llm["gpu_layers"]
            if not isinstance(gpu_layers, int):
                self.errors.append(f"llm.gpu_layers must be an integer, got {type(gpu_layers).__name__}")
            elif gpu_layers < 0:
                self.errors.append(f"llm.gpu_layers must be non-negative, got {gpu_layers}")
                
        # Check precision
        if "precision" in llm and llm["precision"] not in ["float16", "float32", "int8", "int4"]:
            self.errors.append(f"Invalid llm.precision: {llm['precision']}. Valid values: float16, float32, int8, int4")
            
    def _validate_paths(self) -> None:
        """
        Validate that paths in the configuration exist or can be created.
        
        Checks paths for log directory and utility-specific directories.
        Attempts to create missing directories.
        """
        # Check logging path
        self._validate_logging_path()
                    
        # Check utility paths
        self._validate_utility_paths()
    
    def _validate_logging_path(self) -> None:
        """
        Validate logging path configuration.
        
        Checks if logging path exists and tries to create it if missing.
        """
        if "logging" in self.config and "path" in self.config["logging"]:
            log_path = self.config["logging"]["path"]
            if not os.path.exists(log_path):
                try:
                    os.makedirs(log_path, exist_ok=True)
                    logger.info(f"Created log directory: {log_path}")
                except Exception as e:
                    self.warnings.append(f"Could not create log directory {log_path}: {e}")
    
    def _validate_utility_paths(self) -> None:
        """
        Validate utility-specific path configuration.
        
        Checks if utility directories exist and tries to create them if missing.
        """
        if "utilities" not in self.config:
            return
            
        utilities = self.config["utilities"]
        
        for utility_name, utility_config in utilities.items():
            # Check output directory
            if "output_dir" in utility_config:
                output_dir = utility_config["output_dir"]
                if not os.path.exists(output_dir):
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                        logger.info(f"Created {utility_name} output directory: {output_dir}")
                    except Exception as e:
                        self.warnings.append(f"Could not create {utility_name} output directory: {e}")
                        
            # Check template path
            if "template_path" in utility_config:
                template_path = utility_config["template_path"]
                template_dir = os.path.dirname(template_path)
                if not os.path.exists(template_dir):
                    try:
                        os.makedirs(template_dir, exist_ok=True)
                        logger.info(f"Created {utility_name} template directory: {template_dir}")
                    except Exception as e:
                        self.warnings.append(f"Could not create {utility_name} template directory: {e}")
                        
    def _validate_system_settings(self) -> None:
        """
        Validate system settings.
        
        Checks threading, memory, and timeout settings.
        Adds errors to self.errors for invalid settings.
        """
        # Check threading settings
        self._validate_threading_settings()
                    
        # Check memory settings
        self._validate_memory_settings()
                    
        # Check inactivity timeout
        self._validate_timeout_settings()
    
    def _validate_threading_settings(self) -> None:
        """
        Validate threading configuration.
        
        Checks max_workers setting is valid.
        """
        if "threading" in self.config and "max_workers" in self.config["threading"]:
            max_workers = self.config["threading"]["max_workers"]
            if not isinstance(max_workers, int):
                self.errors.append(f"threading.max_workers must be an integer, got {type(max_workers).__name__}")
            elif max_workers < 1:
                self.errors.append(f"threading.max_workers must be at least 1, got {max_workers}")
                
            try:
                import os
                cpu_count = os.cpu_count() or 4
                if max_workers > cpu_count * 2:
                    self.warnings.append(f"max_workers ({max_workers}) exceeds twice the number of CPU cores ({cpu_count})")
            except ImportError:
                pass
    
    def _validate_memory_settings(self) -> None:
        """
        Validate memory configuration.
        
        Checks max_percent and model_unload_threshold settings are valid.
        """
        if "memory" not in self.config:
            return
            
        memory = self.config["memory"]
        
        if "max_percent" in memory:
            max_percent = memory["max_percent"]
            if not isinstance(max_percent, (int, float)):
                self.errors.append(f"memory.max_percent must be a number, got {type(max_percent).__name__}")
            elif max_percent < 10 or max_percent > 95:
                self.errors.append(f"memory.max_percent should be between 10 and 95, got {max_percent}")
                
        if "model_unload_threshold" in memory and "max_percent" in memory:
            unload_threshold = memory["model_unload_threshold"]
            max_percent = memory["max_percent"]
            
            if not isinstance(unload_threshold, (int, float)):
                self.errors.append(f"memory.model_unload_threshold must be a number, got {type(unload_threshold).__name__}")
            elif unload_threshold <= max_percent:
                self.errors.append(f"memory.model_unload_threshold ({unload_threshold}) must be greater than memory.max_percent ({max_percent})")
    
    def _validate_timeout_settings(self) -> None:
        """
        Validate timeout configuration.
        
        Checks inactivity_timeout setting is valid.
        """
        if "inactivity_timeout" in self.config:
            timeout = self.config["inactivity_timeout"]
            if not isinstance(timeout, (int, float)):
                self.errors.append(f"inactivity_timeout must be a number, got {type(timeout).__name__}")
            elif timeout < 10:
                self.warnings.append(f"inactivity_timeout is very short ({timeout} seconds), which may cause unwanted sleep transitions")
                
    def get_validation_result(self) -> Tuple[bool, List[str], List[str]]:
        """
        Get the validation result.
        
        Returns
        -------
        Tuple[bool, List[str], List[str]]
            A tuple containing (is_valid, errors, warnings)
        """
        is_valid = len(self.errors) == 0
        return (is_valid, self.errors, self.warnings)