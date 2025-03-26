"""
Maggie AI Assistant - Main Application Entry Point
================================================

This module serves as the primary entry point for the Maggie AI Assistant, 
implementing a comprehensive initialization and startup process for an 
intelligent, state-aware AI assistant application.

The application follows a modular, event-driven architecture with a 
Finite State Machine (FSM) design, optimized for high-performance 
computing environments, particularly systems with AMD Ryzen 9 5900X 
processors and NVIDIA RTX 3080 GPUs.

Key Architecture Concepts:
------------------------
- Modular Component Design
- Event-Driven State Management
- Configurable Hardware Optimization
- Graceful Error Handling
- Flexible Execution Modes (GUI and Headless)

External Dependencies:
--------------------
- Python 3.10+ (Strict version compatibility)
- PySide6 for GUI
- Custom Maggie AI Core Libraries

References:
----------
- Finite State Machine Patterns: https://python-patterns.guide/gang-of-four/state/
- Python Multiprocessing: https://docs.python.org/3/library/multiprocessing.html
- Signal Handling: https://docs.python.org/3/library/signal.html

.. moduleauthor:: Anthony Formosa <emailformosa@gmail.com>
"""

import os
import sys
import argparse
import signal
import platform
import multiprocessing
import time
from typing import Dict, Any, Optional, Tuple

from maggie.utils.logging import ComponentLogger
logger = ComponentLogger('Main')

from maggie.utils.error_handling import (
    safe_execute, 
    ErrorCategory, 
    ErrorSeverity, 
    record_error, 
    with_error_handling
)
# Initialize a component-specific logger for the main module
from maggie.core import MaggieAI, State
from maggie.utils.resource.detector import HardwareDetector
from maggie.utils.config.manager import ConfigManager



def parse_arguments() -> argparse.Namespace:
    """
    Parse and process command-line arguments for the Maggie AI Assistant.

    This function configures an argument parser to handle various runtime 
    configuration and execution modes for the application.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with the following potential options:
        - config: Path to configuration file
        - debug: Enable debug logging
        - verify: Verify system configuration
        - create-template: Create configuration template
        - headless: Run without GUI

    Examples
    --------
    >>> # Example CLI usage scenarios
    >>> # Run with custom config: python main.py --config custom_config.yaml
    >>> # Enable debug mode: python main.py --debug
    >>> # Run in headless mode: python main.py --headless
    """
    parser = argparse.ArgumentParser(
        description='Maggie AI Assistant',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml', 
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug logging'
    )
    parser.add_argument(
        '--verify', 
        action='store_true', 
        help='Verify system configuration without starting the assistant'
    )
    parser.add_argument(
        '--create-template', 
        action='store_true', 
        help="Create the recipe template file if it doesn't exist"
    )
    parser.add_argument(
        '--headless', 
        action='store_true', 
        help='Run in headless mode without GUI'
    )
    return parser.parse_args()

def initialize_logging(config: Dict[str, Any], debug: bool = False) -> None:
    """
    Initialize the logging system for the Maggie AI Assistant.

    This function sets up logging configurations, ensuring proper 
    log file creation and setting log levels based on debug mode.

    Parameters
    ----------
    config : Dict[str, Any]
        Application configuration dictionary
    debug : bool, optional
        Flag to enable debug-level logging, by default False

    Notes
    -----
    - Creates a 'logs' directory if it doesn't exist
    - Modifies logging configuration in debug mode
    - Sets up global exception handling

    References
    ----------
    - Logging Documentation: https://docs.python.org/3/library/logging.html
    - Loguru Documentation: https://loguru.readthedocs.io/
    """
    os.makedirs('logs', exist_ok=True)
    
    # Adjust log levels if debug mode is enabled
    if debug and 'logging' in config:
        config['logging']['console_level'] = 'DEBUG'
        config['logging']['file_level'] = 'DEBUG'
    
    LoggingManager.initialize(config)
    LoggingManager.get_instance().setup_global_exception_handler()
    logger.info('Logging system initialized')

def initialize_multiprocessing() -> None:
    """
    Configure multiprocessing start method for the application.

    Sets the multiprocessing start method to 'spawn', which is 
    safer and more compatible across different platforms, 
    especially for GUI and complex application architectures.

    Notes
    -----
    - Uses 'spawn' method to create new Python interpreter processes
    - Handles potential runtime errors if method is already set

    References
    ----------
    - Multiprocessing Documentation: https://docs.python.org/3/library/multiprocessing.html
    """
    try:
        multiprocessing.set_start_method('spawn')
        logger.info("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        # Method might already be set, which is acceptable
        pass

def setup_pyside6_paths() -> None:
    """
    Dynamically configure system paths for PySide6 Qt6 libraries.

    Searches for PySide6 installation paths and appends them to 
    sys.path to ensure proper library loading and compatibility.

    Notes
    -----
    - Discovers PySide6 site packages
    - Adds Qt6 library paths to system path
    - Handles potential path configuration errors gracefully

    References
    ----------
    - PySide6 Documentation: https://doc.qt.io/qtforpython-6/
    """
    try:
        import site
        for site_path in site.getsitepackages():
            pyside_path = os.path.join(site_path, 'PySide6')
            if os.path.exists(pyside_path):
                if pyside_path not in sys.path:
                    sys.path.append(pyside_path)
                    
                    # Add Qt6 specific paths
                    qt6_path = os.path.join(pyside_path, 'Qt6')
                    if os.path.exists(qt6_path) and qt6_path not in sys.path:
                        sys.path.append(qt6_path)
                    
                    # Add binary path
                    bin_path = os.path.join(qt6_path, 'bin')
                    if os.path.exists(bin_path) and bin_path not in sys.path:
                        sys.path.append(bin_path)
                    break
    except Exception as e:
        logger.warning(f"Error setting up PySide6 paths: {e}")

def register_signal_handlers(maggie: MaggieAI) -> None:
    """
    Register system signal handlers for graceful application shutdown.

    Configures handlers for SIGINT (Ctrl+C) and SIGTERM to ensure 
    the Maggie AI Assistant can perform clean shutdown procedures.

    Parameters
    ----------
    maggie : MaggieAI
        The main Maggie AI Assistant instance

    Notes
    -----
    - Handles keyboard interrupts and termination signals
    - Ensures orderly shutdown of all system components
    - Logs shutdown events

    References
    ----------
    - Signal Handling: https://docs.python.org/3/library/signal.html
    """
    try:
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down gracefully")
            maggie.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info('Registered signal handlers for graceful shutdown')
    except Exception as e:
        logger.warning(f"Failed to register signal handlers: {e}")

# @with_error_handling(error_category=ErrorCategory.CONFIGURATION)
def setup_application(args: argparse.Namespace) -> Tuple[Optional[MaggieAI], Dict[str, Any]]:
    """
    Initialize the core Maggie AI Assistant application.

    Performs comprehensive setup including:
    - Configuration loading
    - Logging initialization
    - Multiprocessing configuration
    - MaggieAI instance creation

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    Tuple[Optional[MaggieAI], Dict[str, Any]]
        A tuple containing the Maggie AI instance and configuration dictionary

    Notes
    -----
    - Handles potential import and initialization errors
    - Logs critical system information
    - Registers signal handlers for graceful shutdown

    Raises
    ------
    Various exceptions related to configuration and initialization
    """
    
	from maggie.utils.logging import ComponentLogger, LoggingManager


	config_manager = ConfigManager(args.config)
	config = config_manager.load()

	initialize_logging(config, args.debug)

	logger.info('Starting Maggie AI Assistant')
	logger.info(f"Running on Python {platform.python_version()}")
	logger.info(f"Process ID: {os.getpid()}")

	initialize_multiprocessing()

	try:
		maggie = MaggieAI(args.config)
		register_signal_handlers(maggie)
		
		logger.info('Application setup completed successfully')
		return maggie, config
	except ImportError as e:
		record_error(
			message=f"Failed to import required module: {e}",
			exception=e,
			category=ErrorCategory.SYSTEM,
			severity=ErrorSeverity.CRITICAL,
			source='main.setup_application'
		)
		return None, config
	except Exception as e:
		record_error(
			message=f"Error setting up application: {e}",
			exception=e,
			category=ErrorCategory.SYSTEM,
			severity=ErrorSeverity.CRITICAL,
			source='main.setup_application'
		)
		return None, config

@with_error_handling(error_category=ErrorCategory.SYSTEM)
def setup_gui(maggie: MaggieAI) -> Optional[Tuple[Any, Any]]:
    """
    Initialize the graphical user interface for Maggie AI Assistant.

    Configures PySide6 application and main window, integrating
    the Maggie AI core with the user interface.

    Parameters
    ----------
    maggie : MaggieAI
        The core Maggie AI Assistant instance

    Returns
    -------
    Optional[Tuple[Any, Any]]
        A tuple containing the main window and QApplication instances,
        or None if GUI setup fails

    Notes
    -----
    - Dynamically sets up PySide6 library paths
    - Creates QApplication and MainWindow
    - Handles potential import and initialization errors

    References
    ----------
    - PySide6 Documentation: https://doc.qt.io/qtforpython-6/
    """
    try:
        setup_pyside6_paths()
        from PySide6.QtWidgets import QApplication
        from maggie.utils import get_main_window
        
        MainWindow = get_main_window()
        app = QApplication(sys.argv)
        window = MainWindow(maggie)
        
        return window, app
    except ImportError as e:
        logger.error(f"Failed to import GUI modules: {e}")
        return None
    except Exception as e:
        logger.error(f"Error setting up GUI: {e}")
        return None

@with_error_handling(error_category=ErrorCategory.SYSTEM)
def start_maggie(args: argparse.Namespace, maggie: MaggieAI, config: Dict[str, Any]) -> int:
    """
    Start the Maggie AI Assistant core services and user interface.

    Manages the application startup process, handling both GUI and 
    headless execution modes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    maggie : MaggieAI
        The core Maggie AI Assistant instance
    config : Dict[str, Any]
        Application configuration dictionary

    Returns
    -------
    int
        Exit code (0 for successful execution, non-zero for errors)

    Notes
    -----
    - Starts core AI services
    - Configures GUI or headless execution
    - Handles potential startup and execution errors
    - Supports graceful shutdown

    Raises
    ------
    Various exceptions related to service startup and execution
    """
    logger.info('Starting Maggie AI core services')
    success = maggie.start()
    
    if not success:
        logger.error('Failed to start Maggie AI core services')
        return 1
    
    if not args.headless:
        try:
            gui_result = setup_gui(maggie)
            
            if gui_result is None:
                logger.error('GUI setup failed')
                maggie.shutdown()
                return 1
            
            window, app = gui_result
            
            if hasattr(maggie, 'set_gui') and callable(getattr(maggie, 'set_gui')):
                maggie.set_gui(window)
            
            window.show()
            return app.exec()
        
        except Exception as e:
            logger.error(f"Error starting GUI: {e}")
            maggie.shutdown()
            return 1
    
    else:
        logger.info('Running in headless mode')
        try:
            while maggie.state != State.SHUTDOWN:
                time.sleep(1)
            return 0
        
        except KeyboardInterrupt:
            logger.info('Keyboard interrupt received, shutting down')
            maggie.shutdown()
            return 0

def main() -> int:
    """
    Main entry point for the Maggie AI Assistant application.

    Orchestrates the entire application startup process, 
    handling command-line arguments, application setup, 
    and potential runtime errors.

    Returns
    -------
    int
        Exit code indicating the application's termination status
        (0 for successful execution, non-zero for errors)

    Notes
    -----
    - Parses command-line arguments
    - Sets up and starts the application
    - Provides top-level error handling
    - Supports various execution modes

    Raises
    ------
    Various exceptions that might occur during application startup
    """
    try:
        args = parse_arguments()
        maggie, config = setup_application(args)
        
        if maggie is None:
            logger.error('Failed to set up application')
            return 1
        
        return start_maggie(args, maggie, config)
    
    except KeyboardInterrupt:
        logger.info('\nApplication interrupted by user')
        return 1
    
    except Exception as e:
        logger.critical(f"Unexpected error in main: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())