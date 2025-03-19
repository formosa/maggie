# maggie/extensions/manager.py
"""
Maggie AI Assistant - Extensions Manager
=====================================

Command-line utility for managing Maggie AI extensions.

This module provides commands for creating, enabling, disabling, and
listing extensions for the Maggie AI Assistant.

Usage
-----
python -m maggie.utils.extension_manager create my_extension
python -m maggie.utils.extension_manager list
python -m maggie.utils.extension_manager enable my_extension
python -m maggie.utils.extension_manager disable my_extension
"""

import os
import sys
import argparse
import shutil
import yaml
from typing import List, Dict, Any
from pathlib import Path


def create_extension(name: str) -> bool:
    """
    Create a new extension with the given name.
    
    Parameters
    ----------
    name : str
        Name of the extension to create
        
    Returns
    -------
    bool
        True if extension was created successfully, False otherwise
    """
    # Convert to snake_case if needed
    name = name.lower().replace(' ', '_').replace('-', '_')
    
    # Get the package directory
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extensions_dir = os.path.join(package_dir, "maggie", "extensions")
    extension_dir = os.path.join(extensions_dir, name)
    
    # Check if extension already exists
    if os.path.exists(extension_dir):
        print(f"Error: Extension '{name}' already exists")
        return False
    
    # Create extension directory
    os.makedirs(extension_dir, exist_ok=True)
    
    # Create __init__.py
    with open(os.path.join(extension_dir, "__init__.py"), "w") as f:
        f.write(f'''"""
Maggie AI Extension - {name.title().replace('_', ' ')}
{"=" * (len(name) + 26)}

[Description of the extension]

This extension provides [functionality description] for the Maggie AI Assistant.
"""

from .{name} import {name.title().replace('_', '')}
__all__ = ['{name.title().replace('_', '')}']
''')
    
    # Create main module
    class_name = name.title().replace('_', '')
    with open(os.path.join(extension_dir, f"{name}.py"), "w") as f:
        f.write(f'''"""
Maggie AI Extension - {name.title().replace('_', ' ')} Implementation
{"=" * (len(name) + 38)}

[Detailed description of the extension]

Examples
--------
>>> from maggie.extensions.{name} import {class_name}
>>> from maggie import EventBus
>>> event_bus = EventBus()
>>> config = {{"param1": "value1"}}
>>> extension = {class_name}(event_bus, config)
>>> extension.initialize()
>>> extension.start()
"""

# Standard library imports
import os
import time
import threading
from typing import Dict, Any, Optional, List

# Third-party imports
from loguru import logger

# Local imports
from from maggie.extensions.base import ExtensionBase

class {class_name}(ExtensionBase):
    """
    {name.title().replace('_', ' ')} implementation for Maggie AI Assistant.
    
    [Detailed class description]
    
    Parameters
    ----------
    event_bus : EventBus
        Reference to the central event bus
    config : Dict[str, Any]
        Configuration dictionary with {name.replace('_', ' ')} settings
        
    Attributes
    ----------
    [List of important attributes with descriptions]
    """
    
    def __init__(self, event_bus, config: Dict[str, Any]):
        """
        Initialize the {name.replace('_', ' ')} utility.
        
        Parameters
        ----------
        event_bus : EventBus
            Reference to the central event bus
        config : Dict[str, Any]
            Configuration dictionary with {name.replace('_', ' ')} settings
        """
        super().__init__(event_bus, config)
        
        # Initialize attributes
        self.state = None
        
        # Parse configuration
        self.param1 = config.get("param1", "default_value")
        
    def get_trigger(self) -> str:
        """
        Get the trigger phrase for this utility.
        
        Returns
        -------
        str
            Trigger phrase that activates this utility
        """
        return "{name.replace('_', ' ')}"
        
    def initialize(self) -> bool:
        """
        Initialize the utility.
        
        Returns
        -------
        bool
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
            
        try:
            # Acquire required services
            self.stt_processor = self.get_service("stt_processor")
            self.llm_processor = self.get_service("llm_processor")
            
            if not self.stt_processor or not self.llm_processor:
                logger.error("Failed to acquire required services")
                return False
            
            # Initialize resources
            # ...
            
            self._initialized = True
            logger.info("{name.title().replace('_', ' ')} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {name.replace('_', ' ')}: {{e}}")
            return False
        
    def start(self) -> bool:
        """
        Start the utility.
        
        Returns
        -------
        bool
            True if started successfully, False otherwise
        """
        try:
            # Check if already running
            if self.running:
                logger.warning("{name.title().replace('_', ' ')} already running")
                return False
            
            # Initialize if needed
            if not self._initialized and not self.initialize():
                logger.error("Failed to initialize {name.replace('_', ' ')}")
                return False
            
            # Start workflow
            # ...
            
            self.running = True
            logger.info("{name.title().replace('_', ' ')} started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting {name.replace('_', ' ')}: {{e}}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the utility.
        
        Returns
        -------
        bool
            True if stopped successfully, False otherwise
        """
        if not self.running:
            return True
            
        try:
            # Stop workflow
            # ...
            
            self.running = False
            logger.info("{name.title().replace('_', ' ')} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping {name.replace('_', ' ')}: {{e}}")
            return False
    
    def process_command(self, command: str) -> bool:
        """
        Process a command directed to this utility.
        
        Parameters
        ----------
        command : str
            Command string to process
            
        Returns
        -------
        bool
            True if command was processed, False otherwise
        """
        if not self.running:
            return False
            
        # Process command
        # ...
        
        return False
''')
    
    # Create config.py
    with open(os.path.join(extension_dir, "config.py"), "w") as f:
        f.write(f'''"""
{name.title().replace('_', ' ')} Default Configuration
{"=" * (len(name) + 26)}

Default configuration values for the {name.title().replace('_', ' ')} extension.
"""

DEFAULT_CONFIG = {{
    "param1": "default_value",
    "enabled": True
}}
''')
    
    # Create requirements.txt
    with open(os.path.join(extension_dir, "requirements.txt"), "w") as f:
        f.write("# Required dependencies for the extension\n")
    
    print(f"Created extension: {name}")
    print(f"Location: {extension_dir}")
    print("Don't forget to add the extension to your config.yaml:")
    print(f"""
utilities:
  {name}:
    enabled: true
    param1: value1
""")
    
    return True

def list_extensions() -> List[str]:
    """
    List all available extensions.
    
    Returns
    -------
    List[str]
        List of extension names
    """
    # Get the package directory
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extensions_dir = os.path.join(package_dir, "maggie", "extensions")
    extension_templates_dir = os.path.join(package_dir, "templates", "extension")
    
    if not os.path.exists(extensions_dir):
        print("Extensions directory not found")
        return []
    
    # List subdirectories that contain __init__.py (proper packages)
    extensions = []
    for entry in os.scandir(extensions_dir):
        if entry.is_dir() and os.path.exists(os.path.join(entry.path, "__init__.py")):
            extensions.append(entry.name)
    
    # Print extensions
    if extensions:
        print(f"Found {len(extensions)} extensions:")
        for ext in extensions:
            # Check if enabled in config
            enabled = is_extension_enabled(ext)
            status = "enabled" if enabled else "disabled"
            print(f"  - {ext} ({status})")
    else:
        print("No extensions found")
    
    return extensions

def enable_extension(name: str) -> bool:
    """
    Enable an extension in the configuration.
    
    Parameters
    ----------
    name : str
        Name of the extension to enable
        
    Returns
    -------
    bool
        True if extension was enabled successfully, False otherwise
    """
    config_path = find_config_file()
    if not config_path:
        print("Configuration file not found")
        return False
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False
    
    # Ensure utilities section exists
    if "utilities" not in config:
        config["utilities"] = {}
    
    # Ensure extension config exists
    if name not in config["utilities"]:
        config["utilities"][name] = {}
    
    # Enable extension
    config["utilities"][name]["enabled"] = True
    
    # Save config
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False
    
    print(f"Enabled extension: {name}")
    return True

def disable_extension(name: str) -> bool:
    """
    Disable an extension in the configuration.
    
    Parameters
    ----------
    name : str
        Name of the extension to disable
        
    Returns
    -------
    bool
        True if extension was disabled successfully, False otherwise
    """
    config_path = find_config_file()
    if not config_path:
        print("Configuration file not found")
        return False
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False
    
    # Ensure utilities section exists
    if "utilities" not in config:
        config["utilities"] = {}
    
    # Ensure extension config exists
    if name not in config["utilities"]:
        config["utilities"][name] = {}
    
    # Disable extension
    config["utilities"][name]["enabled"] = False
    
    # Save config
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False
    
    print(f"Disabled extension: {name}")
    return True

def find_config_file() -> Optional[str]:
    """
    Find the configuration file.
    
    Returns
    -------
    Optional[str]
        Path to the configuration file if found, None otherwise
    """
    # Check common locations
    config_paths = [
        "config.yaml",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            return path
    
    return None

def is_extension_enabled(name: str) -> bool:
    """
    Check if an extension is enabled in the configuration.
    
    Parameters
    ----------
    name : str
        Name of the extension
        
    Returns
    -------
    bool
        True if extension is enabled, False otherwise
    """
    config_path = find_config_file()
    if not config_path:
        return False
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return False
    
    # Check if extension is enabled
    return config.get("utilities", {}).get(name, {}).get("enabled", True)

def main():
    """
    Main entry point for the extension manager.
    """
    parser = argparse.ArgumentParser(
        description="Maggie AI Extension Manager"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new extension")
    create_parser.add_argument("name", help="Name of the extension to create")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available extensions")
    
    # Enable command
    enable_parser = subparsers.add_parser("enable", help="Enable an extension")
    enable_parser.add_argument("name", help="Name of the extension to enable")
    
    # Disable command
    disable_parser = subparsers.add_parser("disable", help="Disable an extension")
    disable_parser.add_argument("name", help="Name of the extension to disable")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_extension(args.name)
    elif args.command == "list":
        list_extensions()
    elif args.command == "enable":
        enable_extension(args.name)
    elif args.command == "disable":
        disable_extension(args.name)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()