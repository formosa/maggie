<<<<<<< HEAD
"""
Maggie AI Assistant - Configuration Validator
==========================================
Validates configuration parameters for the Maggie AI Assistant.
Ensures required parameters are present and values are within acceptable ranges.
"""

import os
from typing import Dict, Any, List, Tuple
from loguru import logger

class ConfigValidator:
    """
    Configuration validator for Maggie AI Assistant.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the configuration validator.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to validate
        """
        self.config = config
        self.validation_errors = []
        self.validation_warnings = []
        
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns
        -------
        bool
            True if configuration is valid, False otherwise
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Validate required parameters
        self._validate_required_params()
        
        # Validate paths
        self._validate_paths()
        
        # Validate parameter values
        self._validate_parameter_values()
        
        # Validate hardware compatibility
        self._validate_hardware_compatibility()
        
        # Log validation results
        for error in self.validation_errors:
            logger.error(f"Configuration error: {error}")
            
        for warning in self.validation_warnings:
            logger.warning(f"Configuration warning: {warning}")
            
        return len(self.validation_errors) == 0
        
    def _validate_required_params(self):
        """Validate required configuration parameters."""
        required_params = [
            ("wake_word.porcupine_access_key", "Picovoice access key"),
            ("llm.model_path", "LLM model path"),
            ("speech.tts.voice_model", "TTS voice model")
        ]
        
        for param_path, param_name in required_params:
            parts = param_path.split(".")
            config_part = self.config
            
            # Navigate through nested dictionaries
            missing = False
            for part in parts:
                if part not in config_part:
                    self.validation_errors.append(f"Missing required configuration: {param_name} ({param_path})")
                    missing = True
                    break
                config_part = config_part[part]
                
            # If not missing but empty string
            if not missing and isinstance(config_part, str) and not config_part:
                self.validation_errors.append(f"Empty required configuration: {param_name} ({param_path})")
    
    def _validate_paths(self):
        """Validate file and directory paths."""
        # Check model paths
        model_paths = [
            (self.config.get("llm", {}).get("model_path"), "LLM model directory", True),
            (os.path.join(
                self.config.get("speech", {}).get("tts", {}).get("model_path", "models/tts"),
                self.config.get("speech", {}).get("tts", {}).get("voice_model", "")
            ), "TTS voice model directory", True)
        ]
        
        for path, name, is_dir in model_paths:
            if not os.path.exists(path):
                self.validation_warnings.append(f"{name} path does not exist: {path}")
            elif is_dir and not os.path.isdir(path):
                self.validation_warnings.append(f"{name} path is not a directory: {path}")
        
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
                if not os.path.exists(template_path):
                    self.validation_warnings.append(f"{utility_name} template file does not exist: {template_path}")
    
    def _validate_parameter_values(self):
        """Validate parameter values are within acceptable ranges."""
        # Validate wake word sensitivity
        wake_word = self.config.get("wake_word", {})
        sensitivity = wake_word.get("sensitivity")
        if sensitivity is not None and (sensitivity < 0.0 or sensitivity > 1.0):
            self.validation_errors.append(f"Wake word sensitivity must be between 0.0 and 1.0, got {sensitivity}")
        
        # Validate whisper model size
        whisper_config = self.config.get("speech", {}).get("whisper", {})
        model_size = whisper_config.get("model_size")
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if model_size and model_size not in valid_sizes:
            self.validation_errors.append(f"Invalid whisper model size: {model_size}. Valid values: {valid_sizes}")
        
        # Validate compute type
        compute_type = whisper_config.get("compute_type")
        valid_types = ["int8", "float16", "float32"]
        if compute_type and compute_type not in valid_types:
            self.validation_errors.append(f"Invalid compute type: {compute_type}. Valid values: {valid_types}")
        
        # Validate threading configuration
        threading = self.config.get("threading", {})
        max_workers = threading.get("max_workers")
        if max_workers is not None and (not isinstance(max_workers, int) or max_workers < 1):
            self.validation_errors.append(f"max_workers must be a positive integer, got {max_workers}")
        
        # Validate memory configuration
        memory = self.config.get("memory", {})
        max_percent = memory.get("max_percent")
        if max_percent is not None and (not isinstance(max_percent, (int, float)) or max_percent < 10 or max_percent > 95):
            self.validation_errors.append(f"memory.max_percent must be between 10 and 95, got {max_percent}")
    
    def _validate_hardware_compatibility(self):
        """Validate configuration compatibility with hardware."""
        # Check GPU layers setting
        try:
            import torch
            
            llm_config = self.config.get("llm", {})
            gpu_layers = llm_config.get("gpu_layers", 0)
            
            if torch.cuda.is_available():
                # Get GPU memory information
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                
                # RTX 3080 has 10GB VRAM
                rtx_3080_vram_mb = 10 * 1024
                
                # If GPU has less memory than RTX 3080, reduce the layers proportionally
                if gpu_memory_mb < rtx_3080_vram_mb and gpu_layers > 0:
                    ratio = gpu_memory_mb / rtx_3080_vram_mb
                    recommended_layers = max(1, int(gpu_layers * ratio))
                    
                    if llm_config.get("gpu_layer_auto_adjust", False):
                        self.validation_warnings.append(
                            f"GPU has less memory than configured target. "
                            f"Auto-adjusting GPU layers from {gpu_layers} to {recommended_layers}"
                        )
                        # Update the config
                        llm_config["gpu_layers"] = recommended_layers
                    else:
                        self.validation_warnings.append(
                            f"GPU has less memory than configured target. "
                            f"Consider reducing gpu_layers from {gpu_layers} to {recommended_layers} "
                            f"or enable gpu_layer_auto_adjust"
                        )
            else:
                if gpu_layers > 0:
                    self.validation_warnings.append(
                        "No CUDA-capable GPU detected but gpu_layers > 0. "
                        "LLM will fall back to CPU execution which may be slow."
                    )
        except ImportError:
=======
"""
Maggie AI Assistant - Configuration Validator
==========================================
Validates configuration parameters for the Maggie AI Assistant.
Ensures required parameters are present and values are within acceptable ranges.
"""

import os
from typing import Dict, Any, List, Tuple
from loguru import logger

class ConfigValidator:
    """
    Configuration validator for Maggie AI Assistant.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the configuration validator.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to validate
        """
        self.config = config
        self.validation_errors = []
        self.validation_warnings = []
        
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns
        -------
        bool
            True if configuration is valid, False otherwise
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Validate required parameters
        self._validate_required_params()
        
        # Validate paths
        self._validate_paths()
        
        # Validate parameter values
        self._validate_parameter_values()
        
        # Validate hardware compatibility
        self._validate_hardware_compatibility()
        
        # Log validation results
        for error in self.validation_errors:
            logger.error(f"Configuration error: {error}")
            
        for warning in self.validation_warnings:
            logger.warning(f"Configuration warning: {warning}")
            
        return len(self.validation_errors) == 0
        
    def _validate_required_params(self):
        """Validate required configuration parameters."""
        required_params = [
            ("wake_word.porcupine_access_key", "Picovoice access key"),
            ("llm.model_path", "LLM model path"),
            ("speech.tts.voice_model", "TTS voice model")
        ]
        
        for param_path, param_name in required_params:
            parts = param_path.split(".")
            config_part = self.config
            
            # Navigate through nested dictionaries
            missing = False
            for part in parts:
                if part not in config_part:
                    self.validation_errors.append(f"Missing required configuration: {param_name} ({param_path})")
                    missing = True
                    break
                config_part = config_part[part]
                
            # If not missing but empty string
            if not missing and isinstance(config_part, str) and not config_part:
                self.validation_errors.append(f"Empty required configuration: {param_name} ({param_path})")
    
    def _validate_paths(self):
        """Validate file and directory paths."""
        # Check model paths
        model_paths = [
            (self.config.get("llm", {}).get("model_path"), "LLM model directory", True),
            (os.path.join(
                self.config.get("speech", {}).get("tts", {}).get("model_path", "models/tts"),
                self.config.get("speech", {}).get("tts", {}).get("voice_model", "")
            ), "TTS voice model directory", True)
        ]
        
        for path, name, is_dir in model_paths:
            if not os.path.exists(path):
                self.validation_warnings.append(f"{name} path does not exist: {path}")
            elif is_dir and not os.path.isdir(path):
                self.validation_warnings.append(f"{name} path is not a directory: {path}")
        
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
                if not os.path.exists(template_path):
                    self.validation_warnings.append(f"{utility_name} template file does not exist: {template_path}")
    
    def _validate_parameter_values(self):
        """Validate parameter values are within acceptable ranges."""
        # Validate wake word sensitivity
        wake_word = self.config.get("wake_word", {})
        sensitivity = wake_word.get("sensitivity")
        if sensitivity is not None and (sensitivity < 0.0 or sensitivity > 1.0):
            self.validation_errors.append(f"Wake word sensitivity must be between 0.0 and 1.0, got {sensitivity}")
        
        # Validate whisper model size
        whisper_config = self.config.get("speech", {}).get("whisper", {})
        model_size = whisper_config.get("model_size")
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if model_size and model_size not in valid_sizes:
            self.validation_errors.append(f"Invalid whisper model size: {model_size}. Valid values: {valid_sizes}")
        
        # Validate compute type
        compute_type = whisper_config.get("compute_type")
        valid_types = ["int8", "float16", "float32"]
        if compute_type and compute_type not in valid_types:
            self.validation_errors.append(f"Invalid compute type: {compute_type}. Valid values: {valid_types}")
        
        # Validate threading configuration
        threading = self.config.get("threading", {})
        max_workers = threading.get("max_workers")
        if max_workers is not None and (not isinstance(max_workers, int) or max_workers < 1):
            self.validation_errors.append(f"max_workers must be a positive integer, got {max_workers}")
        
        # Validate memory configuration
        memory = self.config.get("memory", {})
        max_percent = memory.get("max_percent")
        if max_percent is not None and (not isinstance(max_percent, (int, float)) or max_percent < 10 or max_percent > 95):
            self.validation_errors.append(f"memory.max_percent must be between 10 and 95, got {max_percent}")
    
    def _validate_hardware_compatibility(self):
        """Validate configuration compatibility with hardware."""
        # Check GPU layers setting
        try:
            import torch
            
            llm_config = self.config.get("llm", {})
            gpu_layers = llm_config.get("gpu_layers", 0)
            
            if torch.cuda.is_available():
                # Get GPU memory information
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                
                # RTX 3080 has 10GB VRAM
                rtx_3080_vram_mb = 10 * 1024
                
                # If GPU has less memory than RTX 3080, reduce the layers proportionally
                if gpu_memory_mb < rtx_3080_vram_mb and gpu_layers > 0:
                    ratio = gpu_memory_mb / rtx_3080_vram_mb
                    recommended_layers = max(1, int(gpu_layers * ratio))
                    
                    if llm_config.get("gpu_layer_auto_adjust", False):
                        self.validation_warnings.append(
                            f"GPU has less memory than configured target. "
                            f"Auto-adjusting GPU layers from {gpu_layers} to {recommended_layers}"
                        )
                        # Update the config
                        llm_config["gpu_layers"] = recommended_layers
                    else:
                        self.validation_warnings.append(
                            f"GPU has less memory than configured target. "
                            f"Consider reducing gpu_layers from {gpu_layers} to {recommended_layers} "
                            f"or enable gpu_layer_auto_adjust"
                        )
            else:
                if gpu_layers > 0:
                    self.validation_warnings.append(
                        "No CUDA-capable GPU detected but gpu_layers > 0. "
                        "LLM will fall back to CPU execution which may be slow."
                    )
        except ImportError:
>>>>>>> 6062514b96de23fbf6dcdbfd4420d6e2f22903ff
            self.validation_warnings.append("PyTorch not available, hardware compatibility check skipped")