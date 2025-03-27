from typing import Dict, Any, Optional, List, Tuple, Union

class ConfigManager:
    """
    A class to manage the configuration of the Maggie AI Assistant.

    This class provides functionality for loading, validating, optimizing, and saving
    configuration settings. It supports hardware-specific optimizations, state-specific
    configurations, and backup management.

    Attributes
    ----------
    config_path : str
        The path to the configuration file.
    backup_dir : str
        The directory where configuration backups are stored.
    config : Dict[str, Any]
        The current configuration loaded from the file or defaults.
    validation_errors : List[str]
        A list of validation errors encountered during configuration validation.
    validation_warnings : List[str]
        A list of validation warnings encountered during configuration validation.
    logger : ComponentLogger
        A logger instance for logging configuration-related events.
    default_config : Dict[str, Any]
        The default configuration settings.
    hardware_detector : HardwareDetector
        An instance of the hardware detector for system-specific optimizations.
    hardware_optimizer : Optional[HardwareOptimizer]
        An optional instance of the hardware optimizer.
    hardware_info : Optional[Dict[str, Any]]
        Information about the detected hardware.

    Notes
    -----
    This class uses the `yaml` library for reading and writing YAML files.
    For more information, see the official documentation:
    https://pyyaml.org/wiki/PyYAMLDocumentation
    """

    def __init__(self, config_path: str = 'config.yaml', backup_dir: str = 'config_backups') -> None:
        """
        Initialize the ConfigManager.

        Parameters
        ----------
        config_path : str, optional
            The path to the configuration file (default is 'config.yaml').
        backup_dir : str, optional
            The directory where configuration backups are stored (default is 'config_backups').
        """
        ...

    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create the default configuration settings.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the default configuration settings.
        """
        ...

    def optimize_config_for_hardware(self) -> Dict[str, Any]:
        """
        Optimize the configuration for the detected hardware.

        This method applies hardware-specific optimizations for CPU, GPU, memory,
        and other components based on the detected hardware.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the applied optimizations.

        Notes
        -----
        This method uses the `HardwareOptimizer` class to apply optimizations.
        """
        ...

    def load(self) -> Dict[str, Any]:
        """
        Load the configuration from the file.

        This method reads the configuration file, validates its contents, and applies
        hardware-specific optimizations if applicable.

        Returns
        -------
        Dict[str, Any]
            The loaded configuration.

        Raises
        ------
        yaml.YAMLError
            If there is an error parsing the YAML file.
        IOError
            If there is an error reading the configuration file.

        Examples
        --------
        >>> config_manager = ConfigManager()
        >>> config = config_manager.load()
        """
        ...

    def validate(self) -> None:
        """
        Validate the configuration settings.

        This method checks for missing or invalid configuration parameters and
        adds any errors or warnings to the `validation_errors` and `validation_warnings`
        attributes, respectively.
        """
        ...

    def save(self) -> None:
        """
        Save the current configuration to the file.

        This method writes the current configuration to the file specified by
        `config_path`.

        Raises
        ------
        IOError
            If there is an error writing to the configuration file.
        """
        ...

    def apply_state_specific_config(self, state: Any) -> None:
        """
        Apply state-specific configuration settings.

        Parameters
        ----------
        state : Any
            The state for which to apply the configuration.

        Notes
        -----
        This method is used to dynamically adjust the configuration based on
        the current state of the application.
        """
        ...

    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Detect the hardware configuration of the system.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing information about the detected hardware.

        Notes
        -----
        This method uses the `HardwareDetector` class to gather hardware information.
        """
        ...

    def _attempt_config_recovery(self, error: Exception) -> None:
        """
        Attempt to recover the configuration from a backup.

        Parameters
        ----------
        error : Exception
            The exception that triggered the recovery attempt.

        Notes
        -----
        If no backup is available, the default configuration is used.
        """
        ...

    def _merge_with_defaults(self) -> None:
        """
        Merge the current configuration with the default configuration.

        This method ensures that any missing parameters in the current configuration
        are filled in with default values.
        """
        ...

    def _validate_required_params(self) -> None:
        """
        Validate the presence of required configuration parameters.

        This method checks for the existence of critical configuration parameters
        and adds any missing parameters to the `validation_errors` attribute.
        """
        ...

    def _validate_hardware_specific_settings(self) -> None:
        """
        Validate hardware-specific configuration settings.

        This method checks for the presence and correctness of hardware-specific
        parameters, such as CPU and GPU optimizations.
        """
        ...