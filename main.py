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
    
    Returns:
        Parsed command-line arguments
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
    
    Args:
        maggie: MaggieAI instance
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
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple containing MaggieAI instance and configuration dictionary
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
    
    Args:
        maggie: MaggieAI instance
        
    Returns:
        Tuple containing the main window and QApplication instances
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
    
    Args:
        args: Parsed command-line arguments
        maggie: MaggieAI instance
        config: Configuration dictionary
        
    Returns:
        Exit code
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
    
    Returns:
        Exit code
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