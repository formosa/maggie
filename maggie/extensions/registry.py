#  maggie/extensions/registry.py
"""
Maggie AI Assistant - Extension Registry
======================================

Dynamic registration and discovery system for Maggie AI extensions.

This module provides mechanisms for registering, discovering, and loading
extension modules at runtime, enabling a plugin architecture for extending
Maggie's functionality.

Examples
--------
>>> from maggie.extensions.registry import ExtensionRegistry
>>> registry = ExtensionRegistry()
>>> available_extensions = registry.discover_extensions()
>>> for name, cls in available_extensions.items():
...     print(f"Found extension: {name}")
>>> recipe_creator = registry.get_extension_class("recipe_creator")
"""

import os
import sys
import importlib
import importlib.util
import pkgutil
from typing import Dict, Type, Any, Optional, List

from loguru import logger
from .base import ExtensionBase

class ExtensionRegistry:
    """
    Registry for dynamically discovering and loading extension modules.
    
    Provides methods for scanning the extensions directory, importing
    extension modules, and instantiating extension classes.
    
    Attributes
    ----------
    _registry : Dict[str, Type[ExtensionBase]]
        Dictionary mapping extension names to their classes
    _extensions_path : str
        Path to the extensions directory
    """
    
    def __init__(self, extensions_path: Optional[str] = None):
        """
        Initialize the extension registry.
        
        Parameters
        ----------
        extensions_path : Optional[str], optional
            Path to the extensions directory, by default None which uses
            the default 'extensions' directory in the maggie package
        """
        self._registry = {}
        
        # Determine extensions path
        if extensions_path is None:
            # Get the package directory (where maggie is installed)
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self._extensions_path = os.path.join(package_dir, "extensions")
        else:
            self._extensions_path = extensions_path
            
        # Ensure extensions directory exists
        os.makedirs(self._extensions_path, exist_ok=True)
    
    def discover_extensions(self) -> Dict[str, Type[ExtensionBase]]:
        """
        Discover available extension modules in the extensions directory.
        
        Scans the extensions directory for packages that contain extension
        classes extending the ExtensionBase abstract class.
        
        Returns
        -------
        Dict[str, Type[ExtensionBase]]
            Dictionary mapping extension names to their classes
        """
        self._registry = {}
        
        # Add extensions directory to Python path if not already there
        if self._extensions_path not in sys.path:
            sys.path.append(self._extensions_path)
        
        # Scan for extension modules in extensions directory
        for _, module_name, is_pkg in pkgutil.iter_modules([self._extensions_path]):
            if is_pkg:
                try:
                    # Import the extension package
                    extension_module = importlib.import_module(module_name)
                    
                    # Look for main module with same name as package
                    main_module_name = f"{module_name}.{module_name}"
                    
                    try:
                        # Try importing the main module
                        extension_module = importlib.import_module(main_module_name)
                        
                        # Find all classes that inherit from ExtensionBase
                        for attr_name in dir(extension_module):
                            attr = getattr(extension_module, attr_name)
                            
                            # Check if it's a class that inherits from ExtensionBase
                            if (isinstance(attr, type) and 
                                issubclass(attr, ExtensionBase) and 
                                attr is not ExtensionBase):
                                
                                # Register the extension
                                self._registry[module_name] = attr
                                logger.info(f"Discovered extension: {module_name}")
                                break
                                
                    except ImportError:
                        logger.warning(f"Could not import main module for extension: {module_name}")
                        continue
                        
                except ImportError as e:
                    logger.warning(f"Failed to import extension package {module_name}: {e}")
                    continue
        
        return self._registry
    
    def get_extension_class(self, extension_name: str) -> Optional[Type[ExtensionBase]]:
        """
        Get the class for a specific extension by name.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension to get
            
        Returns
        -------
        Optional[Type[ExtensionBase]]
            The extension class if found, None otherwise
        """
        return self._registry.get(extension_name)
    
    def instantiate_extension(self, extension_name: str, event_bus: Any, config: Dict[str, Any]) -> Optional[ExtensionBase]:
        """
        Instantiate a extension by name with provided event bus and configuration.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension to instantiate
        event_bus : Any
            Event bus instance to pass to the extension
        config : Dict[str, Any]
            Configuration dictionary for the extension
            
        Returns
        -------
        Optional[ExtensionBase]
            Instantiated extension if successful, None otherwise
        """
        extension_class = self.get_extension_class(extension_name)
        
        if extension_class is None:
            logger.error(f"extension not found: {extension_name}")
            return None
            
        try:
            extension_instance = extension_class(event_bus, config)
            return extension_instance
        except Exception as e:
            logger.error(f"Error instantiating extension {extension_name}: {e}")
            return None
    
    def get_available_extensions(self) -> List[str]:
        """
        Get a list of available extension names.
        
        Returns
        -------
        List[str]
            List of available extension names
        """
        return list(self._registry.keys())