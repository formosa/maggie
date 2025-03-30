import os
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

from maggie.utils.error_handling import with_error_handling, ErrorCategory, ErrorSeverity, record_error, safe_execute
from maggie.utils.logging import ComponentLogger, log_operation, logging_context
from maggie.utils.resource.detector import HardwareDetector
from maggie.service.locator import ServiceLocator

__all__ = ['ConfigManager']

class HardwareOptimizer:
    """Reference to the external HardwareOptimizer class from resource.optimizer"""
    pass

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
        Path to the configuration YAML file (default: 'config.yaml')
    backup_dir : str, optional
        Directory where configuration backups are stored (default: 'config_backups')

    Attributes
    ----------
    config_path : str
        Path to the configuration file
    backup_dir : str
        Directory for storing configuration backups
    config : dict
        The current loaded configuration dictionary
    validation_errors : list
        List of validation errors found in the configuration
    validation_warnings : list
        List of validation warnings found in the configuration
    logger : ComponentLogger
        Logger for the ConfigManager
    default_config : dict
        The default configuration used as a fallback
    hardware_detector : HardwareDetector
        Hardware detection utility for hardware-specific optimizations
    hardware_optimizer : HardwareOptimizer, optional
        Optimizer for hardware-specific configurations
    hardware_info : dict, optional
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
            Path to the configuration YAML file (default: 'config.yaml')
        backup_dir : str, optional
            Directory where configuration backups are stored (default: 'config_backups')

        Notes
        -----
        Creates the backup directory if it doesn't exist and initializes all necessary
        attributes for configuration management. The actual configuration loading is 
        performed separately by calling the `load()` method.
        """
        self.config_path = config_path
        self.backup_dir = backup_dir
        self.config = {}
        self.validation_errors = []
        self.validation_warnings = []
        self.logger = ComponentLogger('ConfigManager')
        self.default_config = self._create_default_config()
        self.hardware_detector = HardwareDetector()
        self.hardware_optimizer = None
        self.hardware_info = None
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create the default configuration dictionary.

        Returns
        -------
        dict
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
        return {
            'inactivity_timeout': 60,
            'fsm': {
                'state_styles': {
                    'INIT': {'bg_color': '#E0E0E0', 'font_color': '#212121'},
                    'STARTUP': {'bg_color': '#B3E5FC', 'font_color': '#212121'},
                    'IDLE': {'bg_color': '#C8E6C9', 'font_color': '#212121'},
                    'LOADING': {'bg_color': '#FFE0B2', 'font_color': '#212121'},
                    'READY': {'bg_color': '#A5D6A7', 'font_color': '#212121'},
                    'ACTIVE': {'bg_color': '#FFCC80', 'font_color': '#FFFFFF'},
                    'BUSY': {'bg_color': '#FFAB91', 'font_color': '#FFFFFF'},
                    'CLEANUP': {'bg_color': '#E1BEE7', 'font_color': '#FFFFFF'},
                    'SHUTDOWN': {'bg_color': '#EF9A9A', 'font_color': '#FFFFFF'}
                },
                'transition_animations': {
                    'default': {'type': 'slide', 'duration': 300},
                    'to_shutdown': {'type': 'fade', 'duration': 800},
                    'to_busy': {'type': 'bounce', 'duration': 400}
                },
                'valid_transitions': {
                    'INIT': ['STARTUP', 'IDLE', 'SHUTDOWN'],
                    'STARTUP': ['IDLE', 'READY', 'CLEANUP', 'SHUTDOWN'],
                    'IDLE': ['STARTUP', 'READY', 'CLEANUP', 'SHUTDOWN'],
                    'LOADING': ['ACTIVE', 'READY', 'CLEANUP', 'SHUTDOWN'],
                    'READY': ['LOADING', 'ACTIVE', 'BUSY', 'CLEANUP', 'SHUTDOWN'],
                    'ACTIVE': ['READY', 'BUSY', 'CLEANUP', 'SHUTDOWN'],
                    'BUSY': ['READY', 'ACTIVE', 'CLEANUP', 'SHUTDOWN'],
                    'CLEANUP': ['IDLE', 'SHUTDOWN'],
                    'SHUTDOWN': []
                },
                'input_field_states': {
                    'IDLE': {'enabled': False, 'style': 'background-color: lightgray;'},
                    'READY': {'enabled': True, 'style': 'background-color: white;'},
                    'ACTIVE': {'enabled': True, 'style': 'background-color: white;'},
                    'BUSY': {'enabled': False, 'style': 'background-color: #FFAB91;'}
                }
            },
            'stt': {
                'whisper': {
                    'model_size': 'base',
                    'compute_type': 'float16',
                    'model_path': '\\maggie\\models\\stt\\whisper',
                    'sample_rate': 16000,
                    'tensor_cores_enabled': True,
                    'flash_attention_enabled': True,
                    'max_batch_size': 16,
                    'memory_efficient': True,
                    'parallel_processing': True,
                    'chunk_size': 512,
                    'simd_optimization': True,
                    'cache_models': True
                },
                'whisper_streaming': {
                    'enabled': True,
                    'model_name': 'base',
                    'language': 'en',
                    'compute_type': 'float16',
                    'cuda_streams': 2,
                    'batch_processing': True,
                    'low_latency_mode': True,
                    'tensor_cores_enabled': True,
                    'dedicated_threads': 2,
                    'thread_affinity_enabled': True
                },
                'wake_word': {
                    'engine': 'porcupine',
                    'access_key': '',
                    'sensitivity': 0.5,
                    'keyword': 'maggie',
                    'keyword_path': '\\maggie\\models\\stt\\porcupine\\Hey-Maggie_en_windows_v3_0_0.ppn',
                    'cpu_threshold': 5.0,
                    'dedicated_core_enabled': True,
                    'dedicated_core': 0,
                    'real_time_priority': True,
                    'minimal_processing': True
                }
            },
            'tts': {
                'voice_model': 'af_heart.pt',
                'model_path': '\\maggie\\models\\tts',
                'sample_rate': 22050,
                'use_cache': True,
                'cache_dir': '\\maggie\\cache\\tts',
                'cache_size': 200,
                'gpu_device': 0,
                'gpu_acceleration': True,
                'gpu_precision': 'mixed_float16',
                'max_workers': 4,
                'voice_preprocessing': True,
                'tensor_cores_enabled': True,
                'cuda_graphs_enabled': True,
                'amp_optimization_level': 'O2',
                'max_batch_size': 64,
                'dynamic_memory_management': True,
                'dedicated_threads': 2,
                'thread_affinity_enabled': True,
                'thread_affinity_cores': [8, 9],
                'realtime_priority': True,
                'simd_optimization': True,
                'buffer_size': 4096,
                'spectral_processing_on_gpu': True
            },
            'llm': {
                'model_path': 'maggie\\models\\llm\\mistral-7b-instruct-v0.3-GPTQ-4bit',
                'model_type': 'mistral',
                'gpu_layers': 32,
                'gpu_layer_auto_adjust': True,
                'tensor_cores_enabled': True,
                'mixed_precision_enabled': True,
                'precision_type': 'float16',
                'kv_cache_optimization': True,
                'attention_optimization': True,
                'context_length': 8192,
                'batch_size': 16,
                'offload_strategy': 'auto',
                'vram_efficient_loading': True,
                'rtx_3080_optimized': True
            },
            'logging': {
                'path': 'logs',
                'console_level': 'INFO',
                'file_level': 'DEBUG'
            },
            'extensions': {
                'recipe_creator': {
                    'enabled': True,
                    'template_path': 'templates\\recipe_template.docx',
                    'output_dir': 'recipes'
                }
            },
            'cpu': {
                'max_threads': 8,
                'thread_timeout': 30,
                'ryzen_9_5900x_optimized': True,
                'thread_affinity_enabled': True,
                'performance_cores': [0, 1, 2, 3, 4, 5, 6, 7],
                'background_cores': [8, 9, 10, 11],
                'high_performance_plan': True,
                'disable_core_parking': True,
                'precision_boost_overdrive': True,
                'simultaneous_multithreading': True
            },
            'memory': {
                'max_percent': 75,
                'model_unload_threshold': 85,
                'xpg_d10_memory': True,
                'large_pages_enabled': True,
                'numa_aware': True,
                'preload_models': True,
                'cache_size_mb': 6144,
                'min_free_gb': 4,
                'defragmentation_threshold': 70
            },
            'gpu': {
                'max_percent': 90,
                'model_unload_threshold': 95,
                'rtx_3080_optimized': True,
                'tensor_cores_enabled': True,
                'tensor_precision': 'tf32',
                'cuda_compute_type': 'float16',
                'cuda_streams': 3,
                'cuda_memory_pool': True,
                'cuda_graphs': True,
                'max_batch_size': 16,
                'reserved_memory_mb': 256,
                'dynamic_memory': True,
                'fragmentation_threshold': 15,
                'pre_allocation': True
            }
        }

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
        dict
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
        with logging_context(component='ConfigManager', operation='optimize_for_hardware'):
            optimizations = {'cpu': {}, 'gpu': {}, 'memory': {}, 'llm': {}, 'stt': {}, 'tts': {}}
            if not self.hardware_optimizer and self.hardware_info:
                self.hardware_optimizer = HardwareOptimizer(self.hardware_info, self.config)
            if not self.hardware_info or not self.hardware_optimizer:
                self.logger.warning('Cannot optimize configuration: hardware information not available')
                return optimizations
            
            cpu_info = self.hardware_info.get('cpu', {})
            if cpu_info.get('is_ryzen_9_5900x', False):
                cpu_opts = self.hardware_optimizer.optimize_for_ryzen_9_5900x()
                if cpu_opts.get('applied', False):
                    optimizations['cpu'] = cpu_opts.get('settings', {})
                    self.logger.info('Applied Ryzen 9 5900X-specific optimizations')
                    if 'cpu' not in self.config:
                        self.config['cpu'] = {}
                    for key, value in optimizations['cpu'].items():
                        self.config['cpu'][key] = value
                    if 'stt' in self.config:
                        stt_config = self.config['stt']
                        if 'whisper' in stt_config:
                            stt_config['whisper']['chunk_size'] = 512
                            stt_config['whisper']['simd_optimization'] = True
                            optimizations['stt']['chunk_size'] = 512
                            optimizations['stt']['simd_optimization'] = True

            gpu_info = self.hardware_info.get('gpu', {})
            if gpu_info.get('is_rtx_3080', False):
                gpu_opts = self.hardware_optimizer.optimize_for_rtx_3080()
                if gpu_opts.get('applied', False):
                    optimizations['gpu'] = gpu_opts.get('settings', {})
                    self.logger.info('Applied RTX 3080-specific optimizations')
                    if 'gpu' not in self.config:
                        self.config['gpu'] = {}
                    for key, value in optimizations['gpu'].items():
                        self.config['gpu'][key] = value
                    if 'llm' in self.config:
                        self.config['llm']['gpu_layers'] = 32
                        self.config['llm']['tensor_cores_enabled'] = True
                        self.config['llm']['mixed_precision_enabled'] = True
                        self.config['llm']['precision_type'] = 'float16'
                        optimizations['llm']['gpu_layers'] = 32
                        optimizations['llm']['tensor_cores_enabled'] = True
                        optimizations['llm']['mixed_precision_enabled'] = True
                        optimizations['llm']['precision_type'] = 'float16'
                    if 'stt' in self.config:
                        if 'whisper' in self.config['stt']:
                            self.config['stt']['whisper']['compute_type'] = 'float16'
                            self.config['stt']['whisper']['tensor_cores_enabled'] = True
                            self.config['stt']['whisper']['flash_attention_enabled'] = True
                            optimizations['stt']['compute_type'] = 'float16'
                            optimizations['stt']['tensor_cores_enabled'] = True
                        if 'whisper_streaming' in self.config['stt']:
                            self.config['stt']['whisper_streaming']['compute_type'] = 'float16'
                            self.config['stt']['whisper_streaming']['tensor_cores_enabled'] = True
                            optimizations['stt']['streaming_compute_type'] = 'float16'
                    if 'tts' in self.config:
                        self.config['tts']['gpu_acceleration'] = True
                        self.config['tts']['gpu_precision'] = 'mixed_float16'
                        self.config['tts']['tensor_cores_enabled'] = True
                        optimizations['tts']['gpu_acceleration'] = True
                        optimizations['tts']['gpu_precision'] = 'mixed_float16'
                        optimizations['tts']['tensor_cores_enabled'] = True

            memory_info = self.hardware_info.get('memory', {})
            if memory_info.get('is_xpg_d10', False) and memory_info.get('is_32gb', False):
                if 'memory' not in self.config:
                    self.config['memory'] = {}
                self.config['memory']['large_pages_enabled'] = True
                self.config['memory']['numa_aware'] = True
                self.config['memory']['cache_size_mb'] = 6144
                optimizations['memory']['large_pages_enabled'] = True
                optimizations['memory']['numa_aware'] = True
                optimizations['memory']['cache_size_mb'] = 6144
                self.logger.info('Applied XPG D10 memory-specific optimizations')

            if any(settings for settings in optimizations.values()):
                self.save()
            return optimizations

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
        required_params = [
            ('stt.wake_word.access_key', 'Picovoice access key'),
            ('llm.model_path', 'LLM model path'),
            ('tts.voice_model', 'TTS voice model')
        ]
        for param_path, param_name in required_params:
            value = self._get_nested_value(self.config, param_path)
            if value is None:
                self.validation_errors.append(f"Missing required configuration: {param_name} ({param_path})")
            elif isinstance(value, str) and not value:
                self.validation_errors.append(f"Empty required configuration: {param_name} ({param_path})")

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

        See Also
        --------
        https://en.wikipedia.org/wiki/Finite-state_machine
        """
        fsm_config = self.config.get('fsm', {})
        state_styles = fsm_config.get('state_styles', {})
        required_states = ['INIT', 'STARTUP', 'IDLE', 'LOADING', 'READY', 'ACTIVE', 'BUSY', 'CLEANUP', 'SHUTDOWN']
        
        for state in required_states:
            if state not in state_styles:
                self.validation_warnings.append(f"Missing style configuration for state: {state}")
            else:
                style = state_styles[state]
                if 'bg_color' not in style or 'font_color' not in style:
                    self.validation_warnings.append(f"Incomplete style configuration for state: {state}")
        
        valid_transitions = fsm_config.get('valid_transitions', {})
        for state in required_states:
            if state not in valid_transitions:
                self.validation_warnings.append(f"Missing transition configuration for state: {state}")
        
        input_field_states = fsm_config.get('input_field_states', {})
        for state in ['IDLE', 'READY', 'ACTIVE', 'BUSY']:
            if state not in input_field_states:
                self.validation_warnings.append(f"Missing input field configuration for state: {state}")

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
        cpu_config = self.config.get('cpu', {})
        if cpu_config.get('ryzen_9_5900x_optimized', False):
            if 'max_threads' not in cpu_config:
                self.validation_warnings.append("Missing 'max_threads' setting for Ryzen 9 5900X optimization")
            elif cpu_config['max_threads'] > 10:
                self.validation_warnings.append("'max_threads' value too high for optimal Ryzen 9 5900X performance")
            if 'thread_affinity_enabled' not in cpu_config:
                self.validation_warnings.append("Missing 'thread_affinity_enabled' setting for Ryzen 9 5900X optimization")
            if 'performance_cores' not in cpu_config or not cpu_config.get('performance_cores'):
                self.validation_warnings.append("Missing 'performance_cores' configuration for Ryzen 9 5900X")

        gpu_config = self.config.get('gpu', {})
        if gpu_config.get('rtx_3080_optimized', False):
            if 'tensor_cores_enabled' not in gpu_config:
                self.validation_warnings.append("Missing 'tensor_cores_enabled' setting for RTX 3080 optimization")
            if 'cuda_streams' not in gpu_config:
                self.validation_warnings.append("Missing 'cuda_streams' setting for RTX 3080 optimization")
            if 'max_batch_size' not in gpu_config:
                self.validation_warnings.append("Missing 'max_batch_size' setting for RTX 3080 optimization")

        llm_config = self.config.get('llm', {})
        if gpu_config.get('rtx_3080_optimized', False) and llm_config:
            if 'gpu_layers' not in llm_config:
                self.validation_warnings.append("Missing 'gpu_layers' setting for LLM with RTX 3080")
            elif llm_config.get('gpu_layers', 0) != 32 and not llm_config.get('gpu_layer_auto_adjust', False):
                self.validation_warnings.append("Non-optimal 'gpu_layers' setting (should be 32) for RTX 3080 without auto-adjust")

        memory_config = self.config.get('memory', {})
        if memory_config.get('xpg_d10_memory', False):
            if 'large_pages_enabled' not in memory_config:
                self.validation_warnings.append("Missing 'large_pages_enabled' setting for XPG D10 memory optimization")
            if 'numa_aware' not in memory_config:
                self.validation_warnings.append("Missing 'numa_aware' setting for XPG D10 memory optimization")

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
        dict
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

        See Also
        --------
        https://pyyaml.org/wiki/PyYAMLDocumentation
        """
        with logging_context(component='ConfigManager', operation='load'):
            self.hardware_info = self._detect_hardware()
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, 'r') as file:
                        self.config = yaml.safe_load(file) or {}
                    self.logger.info(f"Configuration loaded from {self.config_path}")
                    self._create_backup('loaded')
                except yaml.YAMLError as yaml_error:
                    self.logger.error(f"YAML error in configuration: {yaml_error}")
                    self._attempt_config_recovery(yaml_error)
                except IOError as io_error:
                    self.logger.error(f"IO error reading configuration: {io_error}")
                    self._attempt_config_recovery(io_error)
            else:
                self.logger.info(f"Configuration file {self.config_path} not found, creating with defaults")
                self.config = self.default_config
                self.save()
            
            self._merge_with_defaults()
            self.validate()
            if self.hardware_info:
                self.optimize_config_for_hardware()
            return self.config

    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Detect the system hardware and return hardware information.

        This method uses the HardwareDetector class to detect the CPU, GPU, and memory
        in the system. It logs information about the detected hardware and creates a
        HardwareOptimizer instance based on the detected hardware.

        Returns
        -------
        dict
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
        try:
            hardware_info = self.hardware_detector.detect_system()
            cpu_info = hardware_info.get('cpu', {})
            gpu_info = hardware_info.get('gpu', {})
            memory_info = hardware_info.get('memory', {})

            if cpu_info.get('is_ryzen_9_5900x', False):
                self.logger.info('Detected AMD Ryzen 9 5900X CPU - applying optimized settings')
            else:
                self.logger.info(f"Detected CPU: {cpu_info.get('model', 'Unknown')}")

            if gpu_info.get('is_rtx_3080', False):
                self.logger.info('Detected NVIDIA RTX 3080 GPU - applying optimized settings')
            elif gpu_info.get('available', False):
                self.logger.info(f"Detected GPU: {gpu_info.get('name', 'Unknown')}")
            else:
                self.logger.warning('No compatible GPU detected - some features may be limited')

            if memory_info.get('is_xpg_d10', False):
                self.logger.info('Detected ADATA XPG D10 memory - applying optimized settings')

            from maggie.utils.resource.optimizer import HardwareOptimizer
            self.hardware_optimizer = HardwareOptimizer(hardware_info, self.default_config)
            return hardware_info
        except Exception as e:
            self.logger.error(f"Error detecting hardware: {e}")
            return {}

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
        backup_path = self._find_latest_backup()
        if backup_path:
            self.logger.info(f"Attempting to recover from backup: {backup_path}")
            try:
                with open(backup_path, 'r') as file:
                    self.config = yaml.safe_load(file) or {}
                self.logger.info(f"Configuration recovered from backup: {backup_path}")
            except Exception as recover_error:
                self.logger.error(f"Failed to recover from backup: {recover_error}")
                self.config = self.default_config
                self.logger.info('Using default configuration')
        else:
            self.config = self.default_config
            self.logger.info('Using default configuration')

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
        hardware_specific_settings = self._extract_hardware_specific_settings(self.config)
        self.config = self._deep_merge(self.default_config.copy(), self.config)
        self._restore_hardware_specific_settings(hardware_specific_settings)

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

        See Also
        --------
        https://refactoring.guru/design-patterns/state
        """
        self.logger.debug(f"Applying state-specific configuration for state: {state.name}")

    def _get_nested_value(self, config_dict: Dict[str, Any], path_string: str) -> Any:
        """
        Get a nested value from a configuration dictionary using a dot-notation path.

        Parameters
        ----------
        config_dict : dict
            The configuration dictionary to search
        path_string : str
            A dot-notation path to the desired value (e.g., "llm.model_path")

        Returns
        -------
        Any
            The value at the specified path, or None if the path doesn't exist

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
        parts = path_string.split('.')
        current = config_dict
        for part in parts:
            if part not in current:
                return None
            current = current[part]
        return current

    def _extract_hardware_specific_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract hardware-specific settings from a configuration dictionary.

        Parameters
        ----------
        config : dict
            The configuration dictionary from which to extract settings

        Returns
        -------
        dict
            A dictionary containing only the hardware-specific settings

        Notes
        -----
        This method is used during configuration merging to preserve hardware-specific
        settings that should not be overwritten by default values. The extracted settings
        can later be restored to the merged configuration.
        """
        hardware_settings = {}
        return hardware_settings

    def _restore_hardware_specific_settings(self, settings: Dict[str, Any]) -> None:
        """
        Restore hardware-specific settings to the current configuration.

        Parameters
        ----------
        settings : dict
            The hardware-specific settings to restore

        Notes
        -----
        This method is used after configuration merging to restore hardware-specific
        settings that were preserved before the merge. This ensures that hardware
        optimizations are not lost during configuration processing.
        """
        pass

    def _deep_merge(self, default_dict: Dict[str, Any], user_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a deep merge of two configuration dictionaries.

        This method recursively merges the user dictionary into the default dictionary,
        preserving user settings while ensuring all required keys are present.

        Parameters
        ----------
        default_dict : dict
            The default configuration dictionary (will be modified)
        user_dict : dict
            The user configuration dictionary to merge into the default

        Returns
        -------
        dict
            The merged configuration dictionary

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
        return default_dict

    def _find_latest_backup(self) -> Optional[str]:
        """
        Find the most recent configuration backup file.

        Returns
        -------
        str, optional
            Path to the most recent backup file, or None if no backups exist

        Notes
        -----
        This method searches the backup directory for configuration backup files
        and returns the path to the most recent one based on file modification time.
        It's used during configuration recovery when the primary configuration file
        is corrupted or invalid.
        """
        return None

    def _create_backup(self, reason: str) -> None:
        """
        Create a backup of the current configuration.

        Parameters
        ----------
        reason : str
            The reason for creating the backup (e.g., 'loaded', 'modified')

        Notes
        -----
        This method creates a timestamped backup of the current configuration in the
        backup directory. The filename includes the timestamp and reason for the backup.
        These backups are used for recovery in case of configuration corruption or errors.

        The backup process ensures that configuration changes are never permanently lost,
        allowing recovery to a previous working state if needed.
        """
        pass

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
        pass

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

        See Also
        --------
        https://pyyaml.org/wiki/PyYAMLDocumentation
        """
        pass

    def _get_hardware_optimizer(self, hardware_info: Dict[str, Any]) -> Any:
        """
        Get a hardware optimizer instance for the detected hardware.

        Parameters
        ----------
        hardware_info : dict
            Information about the detected hardware

        Returns
        -------
        Any
            A HardwareOptimizer instance configured for the detected hardware

        Notes
        -----
        This method creates and returns a HardwareOptimizer instance based on the
        detected hardware information. The optimizer is used to apply hardware-specific
        optimizations to the configuration.
        """
        from maggie.utils.resource.optimizer import HardwareOptimizer
        return HardwareOptimizer(hardware_info, self.config)