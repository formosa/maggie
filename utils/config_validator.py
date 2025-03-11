"""
Maggie AI Assistant - Configuration Validator
==========================================
Validates configuration parameters for the Maggie AI Assistant.

This module provides validation functionality for configuration parameters,
ensuring all required parameters are present and values are within
acceptable ranges. It includes specific validation for hardware
compatibility with AMD Ryzen 9 5900X and NVIDIA RTX 3080.
"""

import os
import platform
from typing import Dict, Any, List, Tuple, Set
from loguru import logger

class ConfigValidator:
    """
    Configuration validator for Maggie AI Assistant.
    
    This class validates configuration settings, ensuring required
    parameters are present and have acceptable values, and checks
    compatibility with system hardware, particularly for AMD Ryzen 9
    and NVIDIA RTX 3080.
    
    Attributes
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate
    validation_errors : List[str]
        List of validation errors detected
    validation_warnings : List[str]
        List of validation warnings detected
    
    Methods
    -------
    validate()
        Validate the configuration and return result
    get_errors()
        Get the list of validation errors
    get_warnings()
        Get the list of validation warnings
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
        
        Performs comprehensive validation of the configuration,
        checking required parameters, path existence, parameter values,
        and hardware compatibility.
        
        Returns
        -------
        bool
            True if configuration is valid (no errors), False otherwise
            
        Notes
        -----
        Even if this method returns True, there may still be warnings.
        Check get_warnings() for any non-critical issues.
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
        
    def get_errors(self) -> List[str]:
        """
        Get the list of validation errors.
        
        Returns
        -------
        List[str]
            List of validation error messages
        """
        return self.validation_errors
        
    def get_warnings(self) -> List[str]:
        """
        Get the list of validation warnings.
        
        Returns
        -------
        List[str]
            List of validation warning messages
        """
        return self.validation_warnings
        
    def _validate_required_params(self) -> None:
        """
        Validate required configuration parameters.
        
        Checks for the presence of essential configuration parameters
        and ensures they have valid values.
        """
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
    
    def _validate_paths(self) -> None:
        """
        Validate file and directory paths.
        
        Checks if specified file and directory paths exist,
        and attempts to create missing directories when appropriate.
        """
        # Check model paths
        model_paths = [
            (self.config.get("llm", {}).get("model_path"), "LLM model directory", True),
            (os.path.join(
                self.config.get("speech", {}).get("tts", {}).get("model_path", "models/tts"),
                self.config.get("speech", {}).get("tts", {}).get("voice_model", "")
            ), "TTS voice model directory", True)
        ]
        
        # Check if paths exist and are appropriate type (file/directory)
        for path, name, is_dir in model_paths:
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
                if not os.path.exists(template_path):
                    self.validation_warnings.append(f"{utility_name} template file does not exist: {template_path}")
                    # Try to create the parent directory
                    try:
                        os.makedirs(os.path.dirname(template_path), exist_ok=True)
                        logger.info(f"Created directory for {utility_name} template: {os.path.dirname(template_path)}")
                    except Exception as e:
                        self.validation_warnings.append(f"Could not create directory for {utility_name} template: {e}")
    
    def _validate_parameter_values(self) -> None:
        """
        Validate parameter values are within acceptable ranges.
        
        Checks that configuration parameter values are appropriate
        types and fall within acceptable ranges.
        """
        # Validate wake word sensitivity (must be between 0.0 and 1.0)
        wake_word = self.config.get("wake_word", {})
        sensitivity = wake_word.get("sensitivity")
        if sensitivity is not None:
            if not isinstance(sensitivity, (int, float)):
                self.validation_errors.append(f"Wake word sensitivity must be a number, got {type(sensitivity).__name__}")
            elif sensitivity < 0.0 or sensitivity > 1.0:
                self.validation_errors.append(f"Wake word sensitivity must be between 0.0 and 1.0, got {sensitivity}")
        
        # Validate whisper model size (must be one of the valid sizes)
        whisper_config = self.config.get("speech", {}).get("whisper", {})
        model_size = whisper_config.get("model_size")
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if model_size and model_size not in valid_sizes:
            self.validation_errors.append(f"Invalid whisper model size: {model_size}. Valid values: {valid_sizes}")
        
        # Validate compute type (must be one of the valid types)
        compute_type = whisper_config.get("compute_type")
        valid_types = ["int8", "float16", "float32"]
        if compute_type and compute_type not in valid_types:
            self.validation_errors.append(f"Invalid compute type: {compute_type}. Valid values: {valid_types}")
        
        # Validate threading configuration
        threading = self.config.get("threading", {})
        max_workers = threading.get("max_workers")
        if max_workers is not None:
            if not isinstance(max_workers, int):
                self.validation_errors.append(f"max_workers must be an integer, got {type(max_workers).__name__}")
            elif max_workers < 1:
                self.validation_errors.append(f"max_workers must be at least 1, got {max_workers}")
            elif max_workers > os.cpu_count() * 2:
                self.validation_warnings.append(
                    f"max_workers ({max_workers}) is more than twice the number of CPU cores ({os.cpu_count()}). "
                    f"This may cause performance issues."
                )
        
        # Validate memory configuration
        memory = self.config.get("memory", {})
        max_percent = memory.get("max_percent")
        if max_percent is not None:
            if not isinstance(max_percent, (int, float)):
                self.validation_errors.append(f"memory.max_percent must be a number, got {type(max_percent).__name__}")
            elif max_percent < 10:
                self.validation_errors.append(f"memory.max_percent must be at least 10, got {max_percent}")
            elif max_percent > 95:
                self.validation_errors.append(f"memory.max_percent must be at most 95, got {max_percent}")
                
        # Validate model unload threshold
        unload_threshold = memory.get("model_unload_threshold")
        if unload_threshold is not None:
            if not isinstance(unload_threshold, (int, float)):
                self.validation_errors.append(
                    f"memory.model_unload_threshold must be a number, got {type(unload_threshold).__name__}"
                )
            elif unload_threshold <= max_percent:
                self.validation_errors.append(
                    f"memory.model_unload_threshold ({unload_threshold}) must be greater than "
                    f"memory.max_percent ({max_percent})"
                )
    
    def _validate_hardware_compatibility(self) -> None:
        """
        Validate configuration compatibility with the system hardware.
        
        Checks if the configuration is appropriate for the detected
        hardware, particularly focusing on AMD Ryzen 9 5900X CPU
        and NVIDIA RTX 3080 GPU compatibility.
        """
        # Check GPU layers setting
        try:
            import torch
            
            llm_config = self.config.get("llm", {})
            gpu_layers = llm_config.get("gpu_layers", 0)
            
            if torch.cuda.is_available():
                # Get GPU memory information
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                gpu_name = torch.cuda.get_device_name(0)
                
                # RTX 3080 has 10GB VRAM
                rtx_3080_vram_mb = 10 * 1024
                is_rtx_3080 = "3080" in gpu_name
                
                # If using RTX 3080, validate optimal settings
                if is_rtx_3080:
                    # Check if using optimal settings for RTX 3080
                    if gpu_layers != 32:
                        self.validation_warnings.append(
                            f"For RTX 3080, the optimal gpu_layers setting is 32, but found {gpu_layers}. "
                            f"Consider changing this for best performance."
                        )
                    
                    # Check whisper settings
                    whisper_config = self.config.get("speech", {}).get("whisper", {})
                    compute_type = whisper_config.get("compute_type")
                    if compute_type != "float16":
                        self.validation_warnings.append(
                            f"For RTX 3080, the optimal whisper compute_type is float16, but found {compute_type}. "
                            f"Consider changing this for best performance."
                        )
                        
                    # Check precision
                    precision = llm_config.get("precision")
                    if precision != "float16":
                        self.validation_warnings.append(
                            f"For RTX 3080, the optimal precision is float16, but found {precision}. "
                            f"Consider changing this for best performance with Tensor Cores."
                        )
                
                # If GPU has less memory than RTX 3080, reduce the layers proportionally
                elif gpu_memory_mb < rtx_3080_vram_mb and gpu_layers > 0:
                    ratio = gpu_memory_mb / rtx_3080_vram_mb
                    recommended_layers = max(1, int(gpu_layers * ratio))
                    
                    if llm_config.get("gpu_layer_auto_adjust", False):
                        self.validation_warnings.append(
                            f"GPU has less memory than RTX 3080. "
                            f"Auto-adjusting GPU layers from {gpu_layers} to {recommended_layers}"
                        )
                        # Update the config
                        llm_config["gpu_layers"] = recommended_layers
                    else:
                        self.validation_warnings.append(
                            f"GPU has less memory than RTX 3080. "
                            f"Consider reducing gpu_layers from {gpu_layers} to {recommended_layers} "
                            f"or enable gpu_layer_auto_adjust"
                        )
                            
                # If GPU has more memory than RTX 3080, could potentially increase layers
                elif gpu_memory_mb > rtx_3080_vram_mb * 1.5 and gpu_layers <= 32:
                    ratio = gpu_memory_mb / rtx_3080_vram_mb
                    recommended_layers = min(40, int(gpu_layers * ratio * 0.8))  # Conservative estimate
                    
                    if recommended_layers > gpu_layers:
                        self.validation_warnings.append(
                            f"GPU has more memory than RTX 3080. "
                            f"Consider increasing gpu_layers from {gpu_layers} to {recommended_layers} "
                            f"for better performance."
                        )
            else:
                if gpu_layers > 0:
                    self.validation_warnings.append(
                        "No CUDA-capable GPU detected but gpu_layers > 0. "
                        "LLM will fall back to CPU execution which may be slow."
                    )
        except ImportError:
            self.validation_warnings.append("PyTorch not available, hardware compatibility check skipped")
            
        # Check CPU compatibility (Ryzen 9 5900X has 12 cores, 24 threads)
        try:
            cpu_model = platform.processor().lower()
            is_ryzen_9 = "ryzen 9" in cpu_model or "5900x" in cpu_model
            physical_cores = os.cpu_count() // 2 if hasattr(os, 'sched_getaffinity') else os.cpu_count() // 2
            
            threading = self.config.get("threading", {})
            max_workers = threading.get("max_workers", 0)
            
            if is_ryzen_9 or physical_cores >= 12:  # Likely Ryzen 9 or similar high-core CPU
                if max_workers < 6 or max_workers > 10:
                    self.validation_warnings.append(
                        f"For Ryzen 9 5900X or similar CPU, the recommended max_workers setting is 8, "
                        f"but found {max_workers}. Consider adjusting for optimal performance."
                    )
            elif physical_cores >= 6:  # Mid-range CPU
                if max_workers > physical_cores * 0.75:
                    self.validation_warnings.append(
                        f"For your CPU with {physical_cores} cores, max_workers of {max_workers} may be too high. "
                        f"Consider reducing to {int(physical_cores * 0.75)} for system responsiveness."
                    )
        except Exception:
            self.validation_warnings.append("CPU compatibility check skipped due to error")