"""
Maggie AI Assistant - Main Application Entry Point
================================================

This module serves as the primary entry point for the Maggie AI Assistant,
implementing a comprehensive initialization and startup process for an
intelligent, state-aware AI assistant application.

The application follows a modular, event-driven architecture with a
Finite State Machine (FSM) design pattern, optimized for high-performance
computing environments, particularly systems with AMD Ryzen 9 5900X
processors and NVIDIA RTX 3080 GPUs.

The architectural design implements:
- Event-driven programming with a centralized event bus
- State machine pattern for application lifecycle management
- Resource optimization with hardware-specific tuning
- Modular component initialization with dependency injection
- Signal handling for graceful application termination

Module Design
-------------
The main entry point orchestrates the following processes:
1. Command-line argument parsing
2. Component initialization through dependency injection
3. System configuration and hardware optimization
4. GUI setup (when not in headless mode)
5. Main application loop management
6. Graceful shutdown procedures

See Also
--------
- Event-driven architecture: https://en.wikipedia.org/wiki/Event-driven_architecture
- Finite State Machine: https://en.wikipedia.org/wiki/Finite-state_machine
- PySide6 Documentation: https://doc.qt.io/qtforpython-6/
"""

import os
import sys
import argparse
import signal
import platform
import multiprocessing
import time
from typing import Dict, Any, Optional, Tuple

# Import the modified initialization module
from maggie.core.initialization import initialize_components

def parse_arguments() -> argparse.Namespace:
    """
    Parse and process command-line arguments for the Maggie AI Assistant.

    The parser handles configuration file path, debug mode, verification mode,
    template creation, and headless operation flags. Argument parsing is implemented
    using Python's argparse library with appropriate defaults and help messages.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - config : str
            Path to configuration file (default: 'config.yaml')
        - debug : bool
            Flag indicating whether debug logging is enabled
        - verify : bool
            Flag to verify system configuration without starting the assistant
        - create_template : bool
            Flag to create recipe template file if it doesn't exist
        - headless : bool
            Flag to run in headless mode without GUI

    Notes
    -----
    The ArgumentParser is configured with descriptive help text and default values
    for improved user experience. The argparse.ArgumentDefaultsHelpFormatter is used
    to automatically append default values to the help text.

    Examples
    --------
    To start the assistant with default configuration:
    >>> python main.py

    To start in debug mode with a custom configuration file:
    >>> python main.py --config custom_config.yaml --debug

    To run in headless mode:
    >>> python main.py --headless
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

def initialize_multiprocessing() -> None:
    """
    Configure multiprocessing start method for the application.

    Sets the multiprocessing start method to 'spawn' for improved stability and
    compatibility across platforms. The 'spawn' method starts a fresh Python
    interpreter process, which avoids issues with forking on certain platforms,
    particularly macOS and Windows.

    Returns
    -------
    None

    Notes
    -----
    This function should be called early in the application startup sequence,
    before any multiprocessing contexts are created. If the start method is
    already set, a RuntimeError will be caught and ignored.

    The 'spawn' method is more resource-intensive than 'fork' but provides
    better isolation and compatibility. For more information, see the Python
    multiprocessing documentation:
    https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

    Examples
    --------
    >>> initialize_multiprocessing()
    Set multiprocessing start method to 'spawn'
    """
    try:
        multiprocessing.set_start_method('spawn')
        # Use basic logging here as logging system may not be initialized yet
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        # Method might already be set, which is acceptable
        pass

def setup_pyside6_paths() -> None:
    """
    Dynamically configure system paths for PySide6 Qt6 libraries.

    Identifies PySide6 installation directories from site-packages and
    adds the relevant paths to sys.path to ensure proper loading of
    Qt6 libraries and binary dependencies. This function helps resolve
    import and runtime issues related to PySide6 module discovery.

    Returns
    -------
    None

    Notes
    -----
    This function performs the following operations:
    1. Identifies site-packages directories using the site module
    2. Locates PySide6 installation directories
    3. Adds PySide6 directory to sys.path if not already present
    4. Adds Qt6 and bin subdirectories to sys.path if they exist

    The function specifically targets the following paths:
    - <site-packages>/PySide6
    - <site-packages>/PySide6/Qt6
    - <site-packages>/PySide6/Qt6/bin

    For more information on PySide6 installation structure, see:
    https://doc.qt.io/qtforpython-6/quickstart.html

    Examples
    --------
    >>> setup_pyside6_paths()
    # No output if successful
    # Error message if an exception occurs
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
        # Use basic print since logging might not be available
        print(f"Error setting up PySide6 paths: {e}")

def register_signal_handlers(maggie: Any) -> None:
    """
    Register system signal handlers for graceful application shutdown.

    Sets up signal handlers for SIGINT (Ctrl+C) and SIGTERM signals
    to ensure the Maggie AI Assistant can shut down gracefully when
    terminated. This prevents resource leaks and ensures proper cleanup
    of system resources.

    Parameters
    ----------
    maggie : MaggieAI
        The MaggieAI instance to shut down when signals are received

    Returns
    -------
    None

    Notes
    -----
    The signal handler performs the following actions:
    1. Logs the received signal
    2. Calls maggie.shutdown() to perform graceful cleanup
    3. Exits the process with a zero exit code

    Signal handling is an important aspect of proper application lifecycle
    management, especially for long-running processes or services. For more
    information on Python signal handling, see:
    https://docs.python.org/3/library/signal.html

    Examples
    --------
    >>> from maggie.core.app import MaggieAI
    >>> maggie = MaggieAI()
    >>> register_signal_handlers(maggie)
    Registered signal handlers for graceful shutdown
    """
    try:
        def signal_handler(sig, frame):
            print(f"Received signal {sig}, shutting down gracefully")
            maggie.shutdown()
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        # Use logger if available
        if 'logging' in sys.modules:
            import logging
            logging.getLogger('Main').info('Registered signal handlers for graceful shutdown')
    except Exception as e:
        print(f"Failed to register signal handlers: {e}")

def setup_application(args: argparse.Namespace) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Initialize the core Maggie AI Assistant application.

    This function orchestrates the initialization of all core application
    components, including configuration loading, multiprocessing setup,
    component initialization, and signal handler registration. It serves
    as the primary entry point for setting up the application framework.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing configuration options

    Returns
    -------
    Tuple[Optional[MaggieAI], Dict[str, Any]]
        A tuple containing:
        - MaggieAI instance or None if initialization failed
        - Configuration dictionary with application settings

    Notes
    -----
    The function performs the following initialization steps:
    1. Creates initial configuration dictionary from command-line arguments
    2. Initializes multiprocessing with 'spawn' method
    3. Initializes all application components using the dependency injection pattern
    4. Registers signal handlers for graceful termination
    5. Logs system information and initialization status

    The component initialization is handled by a dedicated module that implements
    a dependency injection pattern to resolve component dependencies automatically.
    This approach follows the Inversion of Control (IoC) design principle.

    For more on dependency injection and IoC, see:
    https://en.wikipedia.org/wiki/Dependency_injection

    Examples
    --------
    >>> args = parse_arguments()
    >>> maggie, config = setup_application(args)
    >>> if maggie is None:
    ...     print("Initialization failed")
    ... else:
    ...     print("Application setup completed successfully")
    """
    # Create initial config dictionary
    config = {
        'config_path': args.config,
        'debug': args.debug,
        'headless': args.headless,
        'create_template': args.create_template,
        'verify': args.verify
    }
    # Initialize multiprocessing first
    initialize_multiprocessing()
    # Initialize all components using the new initialization module
    components = initialize_components(config, args.debug)
    if not components:
        print("Failed to initialize components")
        return None, config
    # Extract the MaggieAI instance
    maggie = components.get('maggie_ai')
    if not maggie:
        print("Failed to create MaggieAI instance")
        return None, config
    # Register signal handlers
    register_signal_handlers(maggie)
    # Import logger now that it's initialized
    from maggie.utils.logging import ComponentLogger
    logger = ComponentLogger('Main')
    logger.info('Starting Maggie AI Assistant')
    logger.info(f"Running on Python {platform.python_version()}")
    logger.info(f"Process ID: {os.getpid()}")
    logger.info('Application setup completed successfully')
    return maggie, config

def setup_gui(maggie: Any) -> Optional[Tuple[Any, Any]]:
    """
    Initialize the graphical user interface for Maggie AI Assistant.

    Creates and configures the PySide6 Qt application and main window
    for the Maggie AI Assistant. This function handles the GUI initialization
    process, including PySide6 path setup and component instantiation.

    Parameters
    ----------
    maggie : MaggieAI
        The MaggieAI instance to connect to the GUI

    Returns
    -------
    Optional[Tuple[MainWindow, QApplication]]
        If successful, returns a tuple containing:
        - MainWindow instance representing the application's main window
        - QApplication instance representing the Qt application
        If unsuccessful, returns None

    Notes
    -----
    The function performs the following GUI initialization steps:
    1. Sets up PySide6 paths to ensure proper library discovery
    2. Imports required PySide6 modules
    3. Creates a QApplication instance
    4. Instantiates the MainWindow class with the MaggieAI instance

    This function implements a defensive programming approach with
    comprehensive exception handling to gracefully handle import errors
    or initialization failures.

    For more information on PySide6/Qt, see:
    https://doc.qt.io/qtforpython-6/quickstart.html

    Examples
    --------
    >>> from maggie.core.app import MaggieAI
    >>> maggie = MaggieAI()
    >>> gui_result = setup_gui(maggie)
    >>> if gui_result is not None:
    ...     window, app = gui_result
    ...     window.show()
    ...     app.exec()
    """
    try:
        # Import logger
        from maggie.utils.logging import ComponentLogger
        logger = ComponentLogger('Main')
        setup_pyside6_paths()
        try:
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
    except Exception as e:
        print(f"Error setting up GUI: {e}")
        return None

def start_maggie(args: argparse.Namespace, maggie: Any, config: Dict[str, Any]) -> int:
    """
    Start the Maggie AI Assistant core services and user interface.

    This function orchestrates the startup of all Maggie AI Assistant
    services, including core services and the graphical user interface
    (when not in headless mode). It handles the main execution flow and
    event loop management.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    maggie : MaggieAI
        The MaggieAI instance to start
    config : Dict[str, Any]
        Configuration dictionary with application settings

    Returns
    -------
    int
        Exit code indicating success (0) or failure (non-zero)

    Notes
    -----
    The function performs different startup sequences based on the
    execution mode:

    In GUI mode (default):
    1. Starts core Maggie AI services
    2. Sets up the graphical user interface
    3. Connects the GUI to the Maggie AI instance
    4. Shows the main window
    5. Enters the Qt application event loop

    In headless mode (--headless flag):
    1. Starts core Maggie AI services
    2. Enters a simple polling loop that checks for shutdown state
    3. Termin maintainers when shutdown is detected or interrupted

    The function implements proper error handling and cleanup procedures
    to ensure resources are released in case of failure.

    Examples
    --------
    >>> args = parse_arguments()
    >>> maggie, config = setup_application(args)
    >>> if maggie is not None:
    ...     exit_code = start_maggie(args, maggie, config)
    ...     sys.exit(exit_code)
    """
    # Import logger
    from maggie.utils.logging import ComponentLogger
    logger = ComponentLogger('Main')
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
            # Get State enum
            from maggie.core.state import State
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

	This function serves as the application's entry point, orchestrating
	the complete initialization, startup, execution, and shutdown process.
	It implements top-level exception handling to ensure clean termination
	under all circumstances.

	Returns
	-------
	int
		Exit code indicating success (0) or failure (non-zero)

	Parameters
	----------
	Command Line Arguments:
		--config : str, optional
			Path to the configuration YAML file (default: 'config.yaml')
			Specifies the location of the YAML configuration file containing all 
			application settings including hardware optimizations, speech processing 
			parameters, language model configurations, and system thresholds.
			
		--debug : flag, optional
			Enable detailed debug logging (default: False)
			When present, sets logging level to DEBUG for more verbose output,
			providing comprehensive diagnostics information throughout the application
			lifecycle. Useful for troubleshooting issues or development.
			
		--verify : flag, optional
			Verify system configuration without starting the assistant (default: False)
			Performs hardware detection and configuration validation only, reporting
			any issues without actually launching the assistant. Useful for checking
			if the system meets requirements and if the configuration is valid.
			
		--create-template : flag, optional
			Create the recipe template file if it doesn't exist (default: False)
			Generates the default template file for the recipe_creator extension
			specified in the configuration. This ensures the required template exists
			before attempting to use the extension.
			
		--headless : flag, optional
			Run in headless mode without GUI (default: False)
			Launches the assistant without the graphical user interface, operating
			through console and programmatic interactions only. Useful for server
			environments or when integrating with other systems.

	Notes
	-----
	The function implements the following execution flow:
	1. Parses command-line arguments
	2. Sets up the application environment and components
	3. Starts the Maggie AI Assistant services
	4. Handles execution based on mode (GUI or headless)
	5. Provides comprehensive exception handling

	The main function follows a defensive programming approach with
	structured exception handling to catch and report all errors that
	might occur during application execution. This ensures that the
	application can provide meaningful error messages and exit gracefully
	even in case of unexpected failures.

	This implementation follows the command pattern for application
	control flow, where each major step is encapsulated in its own
	function with clear responsibilities.

	Examples
	--------
	>>> # Standard execution as module entry point
	>>> if __name__ == '__main__':
	...     sys.exit(main())

	>>> # Running with specific config and in debug mode
	>>> # python main.py --config custom_config.yaml --debug

	>>> # Running in verification mode to check system compatibility
	>>> # python main.py --verify

	>>> # Running without GUI in server environment
	>>> # python main.py --headless
	"""
	try:
		args = parse_arguments()
		maggie, config = setup_application(args)
		if maggie is None:
			print('Failed to set up application')
			return 1
		return start_maggie(args, maggie, config)
	except KeyboardInterrupt:
		print('\nApplication interrupted by user')
		return 1
	except Exception as e:
		# Use basic print since logging might not be available
		print(f"Unexpected error in main: {e}")
		return 1

if __name__ == '__main__':
    sys.exit(main())