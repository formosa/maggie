"""
Maggie AI Assistant - Configuration Management Module
====================================================

This module provides comprehensive configuration management functionality for the Maggie AI 
Assistant, including configuration loading, validation, hardware optimization, and 
state-specific configuration handling.

The ConfigManager class serves as the central configuration hub, managing all aspects
of application configuration from initial loading through validation, hardware-specific
optimization, and state-based configuration transitions throughout the application lifecycle.

The module implements several key software engineering design patterns:
- Singleton pattern for global configuration access
- Observer pattern for configuration change events
- Strategy pattern for hardware-specific optimization strategies

Core features:
- YAML-based configuration with validation
- Automatic hardware detection and optimization
- Configuration adaptation for application state transitions
- Configuration backup and recovery mechanisms
- Comprehensive validation for required parameters

References
----------
.. [1] YAML specification: https://yaml.org/spec/1.2.2/
.. [2] Finite State Machine Pattern: https://en.wikipedia.org/wiki/Finite-state_machine
.. [3] PyYAML Documentation: https://pyyaml.org/wiki/PyYAMLDocumentation
"""

import os
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, cast

from maggie.utils.error_handling import with_error_handling, ErrorCategory, ErrorSeverity, record_error, safe_execute
from maggie.utils.logging import ComponentLogger, log_operation, logging_context
from maggie.utils.resource.detector import HardwareDetector
from maggie.service.locator import ServiceLocator

__all__: List[str] = ['ConfigManager']

class HardwareOptimizer:
    """Reference to the external HardwareOptimizer class from resource.optimizer"""
    ...

class ConfigManager:
    """
    Configuration Manager for Maggie AI Assistant.
    
    The ConfigManager serves as the central configuration hub for the Maggie AI Assistant,
    providing functionality for loading, validating, backing up, and optimizing configuration
    settings. It implements configuration loading from YAML files, hardware-specific
    optimization, state-specific configuration changes, and configuration validation.
    
    This class integrates with the Finite State Machine architecture of the Maggie AI 
    Assistant, allowing for dynamic configuration changes based on system state.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration YAML file, by default 'config.yaml'
    backup_dir : str, optional
        Directory where configuration backups are stored, by default 'config_backups'
    
    Attributes
    ----------
    config_path : str
        Path to the configuration file
    backup_dir : str
        Directory for storing configuration backups
    config : Dict[str, Any]
        The current loaded configuration dictionary
    validation_errors : List[str] 
        List of validation errors found in the configuration
    validation_warnings : List[str]
        List of validation warnings found in the configuration
    logger : ComponentLogger
        Logger for the ConfigManager
    default_config : Dict[str, Any]
        The default configuration used as a fallback
    hardware_detector : HardwareDetector
        Hardware detection utility for hardware-specific optimizations
    hardware_optimizer : Optional[HardwareOptimizer]
        Optimizer for hardware-specific configurations
    hardware_info : Optional[Dict[str, Any]]
        Information about detected hardware
    
    Notes
    -----
    The ConfigManager implements an extensive validation system to ensure the configuration
    meets all requirements for the application to function properly. It also provides
    automatic hardware detection to optimize configuration settings for the specific
    hardware on which the application is running.
    
    The design allows for graceful degradation when configuration files are missing or
    corrupted by falling back to default configuration values and attempting to recover
    from backup files.
    
    Examples
    --------
    Basic usage:
    
    >>> config_manager = ConfigManager('config.yaml')
    >>> config = config_manager.load()
    >>> print(config.get('llm', {}).get('model_path'))
    
    Hardware-specific optimization:
    
    >>> optimizations = config_manager.optimize_config_for_hardware()
    >>> print(optimizations.get('cpu', {}).get('max_threads'))
    
    State-specific configuration:
    
    >>> from maggie.core.state import State
    >>> config_manager.apply_state_specific_config(State.ACTIVE)
    """

    def __init__(self, config_path: str = 'config.yaml', backup_dir: str = 'config_backups') -> None:
        """
        Initialize the ConfigManager with specified paths.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration YAML file, by default 'config.yaml'
        backup_dir : str, optional
            Directory where configuration backups are stored, by default 'config_backups'
            
        Notes
        -----
        Creates the backup directory if it doesn't exist and initializes all necessary
        attributes for configuration management. The actual configuration loading is 
        performed separately by calling the `load()` method.
        """
        ...
        
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create the default configuration dictionary.
        
        Returns
        -------
        Dict[str, Any]
            The default configuration dictionary containing all required settings with
            sensible defaults for the Maggie AI Assistant.
            
        Notes
        -----
        This method serves as a fallback mechanism to ensure the application can start
        even if no configuration file exists. It provides a complete set of default values
        for all required configuration parameters, including FSM state settings, audio
        processing, LLM settings, TTS settings, logging, and hardware-specific settings.
        
        The default configuration is optimized for common hardware configurations and
        provides a balanced set of parameters that work on most systems, though not
        optimally for specific hardware.
        """
        ...
        
    @log_operation(component='ConfigManager')
    @with_error_handling(error_category=ErrorCategory.CONFIGURATION)
    def optimize_config_for_hardware(self) -> Dict[str, Any]:
        """
        Optimize the configuration for the detected hardware.
        
        This method detects the system hardware and applies optimizations to the configuration
        based on the specific hardware capabilities. It focuses primarily on optimizing for
        AMD Ryzen 9 5900X processors and NVIDIA RTX 3080 GPUs, as well as XPG D10 memory.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary containing the applied optimizations, organized by component type
            (cpu, gpu, memory, llm, stt, tts).
            
        Notes
        -----
        The method uses hardware detection to identify the CPU, GPU, and memory in the system
        and applies specific optimizations based on the detected hardware. These optimizations
        include:
        
        - For AMD Ryzen 9 5900X CPUs:
          - Thread affinity and core allocation settings
          - Optimized chunk sizes for audio processing
          - SIMD optimization flags
        
        - For NVIDIA RTX 3080 GPUs:
          - Tensor core and CUDA optimization settings
          - Mixed precision computation settings
          - Memory allocation and batch size optimizations
          - Float16 precision for LLM inference
        
        - For XPG D10 memory:
          - Large page and NUMA-aware memory settings
          - Optimized cache size settings
        
        The method automatically saves the configuration after applying optimizations.
        
        Examples
        --------
        >>> config_manager = ConfigManager()
        >>> config = config_manager.load()
        >>> optimizations = config_manager.optimize_config_for_hardware()
        >>> print(f"CPU optimizations: {optimizations.get('cpu', {})}")
        >>> print(f"GPU optimizations: {optimizations.get('gpu', {})}")
        """
        ...
        
    def _validate_required_params(self) -> None:
        """
        Validate that all required parameters are present in the configuration.
        
        This method checks for essential configuration parameters that must be present
        for the application to function correctly. It adds validation errors to the
        `validation_errors` list for any missing or empty required parameters.
        
        Required parameters include:
        - stt.wake_word.access_key: Picovoice access key for wake word detection
        - llm.model_path: Path to the language model
        - tts.voice_model: Voice model for text-to-speech
        
        Notes
        -----
        This validation occurs during the load() process. Errors found here don't 
        necessarily prevent the application from starting, but they indicate that
        certain functionality may not work correctly.
        """
        ...
        
    def _validate_fsm_config(self) -> None:
        """
        Validate the Finite State Machine (FSM) configuration.
        
        This method checks the configuration related to the application's Finite State
        Machine architecture, including state styles, valid transitions, and input field
        states. It adds validation warnings to the `validation_warnings` list for any
        missing or incomplete configuration elements.
        
        The validation includes:
        - Checking for style configurations for all required states
        - Verifying that all states have transition configurations
        - Ensuring that input field configurations exist for key states
        
        Notes
        -----
        The FSM configuration is crucial for the proper functioning of the application's
        state management system. Missing configurations may cause visual or behavioral
        issues in the user interface or state transitions, but typically won't prevent
        the application from starting.
        
        References
        ----------
        .. [1] Finite State Machine Pattern: https://en.wikipedia.org/wiki/Finite-state_machine
        """
        ...
        
    def _validate_hardware_specific_settings(self) -> None:
        """
        Validate hardware-specific configuration settings.
        
        This method checks for the completeness and correctness of hardware-specific
        configuration settings, such as those for Ryzen 9 5900X CPUs, RTX 3080 GPUs,
        and XPG D10 memory. It adds validation warnings to the `validation_warnings`
        list for any missing or potentially problematic settings.
        
        The validation includes:
        - For Ryzen 9 5900X settings: thread count, affinity, performance cores
        - For RTX 3080 settings: tensor cores, CUDA streams, batch sizes
        - For LLM with RTX 3080: GPU layers and optimization flags
        
        Notes
        -----
        Hardware-specific settings are critical for optimal performance but aren't
        always essential for basic functionality. Warnings from this validation indicate
        potential performance issues or suboptimal hardware utilization.
        """
        ...
        
    @log_operation(component='ConfigManager')
    @with_error_handling(error_category=ErrorCategory.CONFIGURATION)
    def load(self) -> Dict[str, Any]:
        """
        Load the configuration from the specified file.
        
        This method loads the configuration from the specified YAML file, merges it with
        default settings, performs hardware detection, runs validation, and optimizes
        the configuration for the detected hardware.
        
        Returns
        -------
        Dict[str, Any]
            The loaded and processed configuration dictionary.
            
        Notes
        -----
        The loading process follows these steps:
        1. Detect system hardware
        2. Attempt to load configuration from file
        3. Create a backup of the loaded configuration
        4. Fall back to default configuration if loading fails
        5. Merge loaded configuration with default values
        6. Validate the configuration
        7. Optimize for detected hardware
        
        If the configuration file doesn't exist, this method creates one with default
        settings. If loading fails due to YAML errors or I/O errors, it attempts to
        recover from a backup file.
        
        Examples
        --------
        >>> config_manager = ConfigManager()
        >>> config = config_manager.load()
        >>> llm_path = config.get('llm', {}).get('model_path')
        >>> print(f"Using LLM model from: {llm_path}")
        
        References
        ----------
        .. [1] PyYAML Documentation: https://pyyaml.org/wiki/PyYAMLDocumentation
        """
        ...
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Detect the system hardware and return hardware information.
        
        This method uses the HardwareDetector class to detect the CPU, GPU, and memory
        in the system. It logs information about the detected hardware and creates a
        HardwareOptimizer instance based on the detected hardware.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary containing information about the detected hardware, including
            details about the CPU, GPU, and memory.
            
        Notes
        -----
        The method specifically looks for:
        - AMD Ryzen 9 5900X CPUs
        - NVIDIA RTX 3080 GPUs
        - ADATA XPG D10 memory
        
        For each detected component, it logs appropriate information and prepares for
        hardware-specific optimizations. If hardware detection fails, an empty dictionary
        is returned, and the application will use default settings.
        """
        ...
        
    def _attempt_config_recovery(self, error: Exception) -> None:
        """
        Attempt to recover configuration from a backup after a loading error.
        
        This method is called when an error occurs during configuration loading. It
        attempts to find and load the most recent configuration backup file. If recovery
        fails, it falls back to using the default configuration.
        
        Parameters
        ----------
        error : Exception
            The exception that caused the loading failure.
            
        Notes
        -----
        The recovery process:
        1. Find the latest backup file using _find_latest_backup()
        2. Attempt to load the backup file
        3. If successful, use the recovered configuration
        4. If unsuccessful, fall back to default configuration
        
        This method is a key component of the application's resilience strategy,
        ensuring that configuration errors don't prevent the application from starting.
        """
        ...
        
    def _merge_with_defaults(self) -> None:
        """
        Merge the loaded configuration with default values.
        
        This method ensures that all required configuration parameters have values by
        merging the loaded configuration with the default configuration. It preserves
        hardware-specific settings during the merge.
        
        Notes
        -----
        The merging process:
        1. Extract hardware-specific settings from the current configuration
        2. Perform a deep merge of the default configuration with the loaded configuration
        3. Restore the hardware-specific settings
        
        This ensures that any missing configurations are filled with sensible defaults
        while preserving user customizations and hardware-specific optimizations.
        """
        ...
        
    def apply_state_specific_config(self, state) -> None:
        """
        Apply configuration settings specific to a particular application state.
        
        This method adjusts the configuration based on the current state of the
        application (e.g., INIT, STARTUP, IDLE, ACTIVE, BUSY). Different states
        may require different resource allocations or optimization strategies.
        
        Parameters
        ----------
        state : State
            The application state for which to apply specific configuration settings.
            
        Notes
        -----
        The state-specific configuration is a key part of the application's resource
        management strategy, allowing it to allocate resources differently based on
        the current operational state. This helps optimize performance and resource
        usage throughout the application lifecycle.
        
        For example:
        - IDLE state may use minimal GPU resources
        - ACTIVE state may allocate more memory to models
        - BUSY state may prioritize processing performance
        
        References
        ----------
        .. [1] State Pattern: https://refactoring.guru/design-patterns/state
        """
        ...
        
    def _get_nested_value(self, config_dict: Dict[str, Any], path_string: str) -> Any:
        """
        Get a nested value from a configuration dictionary using a dot-notation path.
        
        Parameters
        ----------
        config_dict : Dict[str, Any]
            The configuration dictionary to search.
        path_string : str
            A dot-notation path to the desired value (e.g., "llm.model_path").
            
        Returns
        -------
        Any
            The value at the specified path, or None if the path doesn't exist.
            
        Notes
        -----
        This utility method simplifies access to deeply nested configuration values
        by allowing dot-notation path specifications. It's primarily used for validation
        and value retrieval throughout the ConfigManager class.
        
        Examples
        --------
        >>> config = {"llm": {"model_path": "/path/to/model"}}
        >>> value = self._get_nested_value(config, "llm.model_path")
        >>> print(value)
        "/path/to/model"
        """
        ...
        
    def _extract_hardware_specific_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract hardware-specific settings from a configuration dictionary.
        
        Parameters
        ----------
        config : Dict[str, Any]
            The configuration dictionary from which to extract settings.
            
        Returns
        -------
        Dict[str, Any]
            A dictionary containing only the hardware-specific settings.
            
        Notes
        -----
        This method is used during configuration merging to preserve hardware-specific
        settings that should not be overwritten by default values. The extracted settings
        can later be restored to the merged configuration.
        """
        ...
        
    def _restore_hardware_specific_settings(self, settings: Dict[str, Any]) -> None:
        """
        Restore hardware-specific settings to the current configuration.
        
        Parameters
        ----------
        settings : Dict[str, Any]
            The hardware-specific settings to restore.
            
        Notes
        -----
        This method is used after configuration merging to restore hardware-specific
        settings that were preserved before the merge. This ensures that hardware
        optimizations are not lost during configuration processing.
        """
        ...
        
    def _deep_merge(self, default_dict: Dict[str, Any], user_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a deep merge of two configuration dictionaries.
        
        This method recursively merges the user dictionary into the default dictionary,
        preserving user settings while ensuring all required keys are present.
        
        Parameters
        ----------
        default_dict : Dict[str, Any]
            The default configuration dictionary (will be modified).
        user_dict : Dict[str, Any]
            The user configuration dictionary to merge into the default.
            
        Returns
        -------
        Dict[str, Any]
            The merged configuration dictionary.
            
        Notes
        -----
        The deep merge strategy ensures that:
        1. All keys from the default dictionary are present in the result
        2. User values override default values when both exist
        3. Nested dictionaries are recursively merged, not replaced
        
        This is crucial for maintaining a complete configuration while respecting
        user customizations.
        
        Examples
        --------
        >>> default = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> user = {"b": {"c": 4, "e": 5}}
        >>> result = self._deep_merge(default, user)
        >>> print(result)
        {"a": 1, "b": {"c": 4, "d": 3, "e": 5}}
        """
        ...
        
    def _find_latest_backup(self) -> Optional[str]:
        """
        Find the most recent configuration backup file.
        
        Returns
        -------
        Optional[str]
            Path to the most recent backup file, or None if no backups exist.
            
        Notes
        -----
        This method searches the backup directory for configuration backup files
        and returns the path to the most recent one based on file modification time.
        It's used during configuration recovery when the primary configuration file
        is corrupted or invalid.
        """
        ...
        
    def _create_backup(self, reason: str) -> None:
        """
        Create a backup of the current configuration.
        
        Parameters
        ----------
        reason : str
            The reason for creating the backup (e.g., 'loaded', 'modified').
            
        Notes
        -----
        This method creates a timestamped backup of the current configuration in the
        backup directory. The filename includes the timestamp and reason for the backup.
        These backups are used for recovery in case of configuration corruption or errors.
        
        The backup process ensures that configuration changes are never permanently lost,
        allowing recovery to a previous working state if needed.
        """
        ...
        
    def validate(self) -> None:
        """
        Validate the current configuration.
        
        This method runs all validation checks on the current configuration, including:
        - Required parameters validation
        - Finite State Machine configuration validation
        - Hardware-specific settings validation
        
        Notes
        -----
        Validation errors and warnings are stored in the `validation_errors` and
        `validation_warnings` lists, which can be checked after calling this method.
        Critical validation errors may prevent certain functionality from working
        correctly, while warnings indicate potential issues or suboptimal configurations.
        
        This validation process is a key part of ensuring the application's stability
        and correctness, as it can catch configuration issues early before they cause
        runtime problems.
        """
        ...
        
    def save(self) -> None:
        """
        Save the current configuration to the configuration file.
        
        This method writes the current configuration dictionary to the YAML configuration
        file and creates a backup of the saved configuration.
        
        Notes
        -----
        The save process:
        1. Create a backup of the current configuration file if it exists
        2. Write the current configuration to the file in YAML format
        3. Create a backup of the newly saved configuration
        
        This method is called automatically after certain operations (like hardware
        optimization) and can also be called manually to persist configuration changes.
        
        References
        ----------
        .. [1] PyYAML Documentation: https://pyyaml.org/wiki/PyYAMLDocumentation
        """
        ...
        
    def _get_hardware_optimizer(self, hardware_info: Dict[str, Any]) -> Any:
        """
        Get a hardware optimizer instance for the detected hardware.
        
        Parameters
        ----------
        hardware_info : Dict[str, Any]
            Information about the detected hardware.
            
        Returns
        -------
        Any
            A HardwareOptimizer instance configured for the detected hardware.
            
        Notes
        -----
        This method creates and returns a HardwareOptimizer instance based on the
        detected hardware information. The optimizer is used to apply hardware-specific
        optimizations to the configuration.
        """
        ...