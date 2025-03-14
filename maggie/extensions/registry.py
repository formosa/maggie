# maggie/utils/utility_registry.py
"""
Maggie AI Assistant - Utility Registry
======================================

Dynamic registration and discovery system for Maggie AI utilities.

This module provides mechanisms for registering, discovering, and loading
utility modules at runtime, enabling a plugin architecture for extending
Maggie's functionality.

Examples
--------
>>> from maggie.utils.utility_registry import UtilityRegistry
>>> registry = UtilityRegistry()
>>> available_utilities = registry.discover_utilities()
>>> for name, cls in available_utilities.items():
...     print(f"Found utility: {name}")
>>> recipe_creator = registry.get_utility_class("recipe_creator")
"""

import os
import sys
import importlib
import importlib.util
import pkgutil
from typing import Dict, Type, Any, Optional, List

from loguru import logger
from .base import UtilityBase

class UtilityRegistry:
    """
    Registry for dynamically discovering and loading utility modules.
    
    Provides methods for scanning the extensions directory, importing
    utility modules, and instantiating utility classes.
    
    Attributes
    ----------
    _registry : Dict[str, Type[UtilityBase]]
        Dictionary mapping utility names to their classes
    _extensions_path : str
        Path to the extensions directory
    """
    
    def __init__(self, extensions_path: Optional[str] = None):
        """
        Initialize the utility registry.
        
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
    
    def discover_utilities(self) -> Dict[str, Type[UtilityBase]]:
        """
        Discover available utility modules in the extensions directory.
        
        Scans the extensions directory for packages that contain utility
        classes extending the UtilityBase abstract class.
        
        Returns
        -------
        Dict[str, Type[UtilityBase]]
            Dictionary mapping utility names to their classes
        """
        self._registry = {}
        
        # Add extensions directory to Python path if not already there
        if self._extensions_path not in sys.path:
            sys.path.append(self._extensions_path)
        
        # Scan for utility modules in extensions directory
        for _, module_name, is_pkg in pkgutil.iter_modules([self._extensions_path]):
            if is_pkg:
                try:
                    # Import the extension package
                    extension_module = importlib.import_module(module_name)
                    
                    # Look for main module with same name as package
                    main_module_name = f"{module_name}.{module_name}"
                    
                    try:
                        # Try importing the main module
                        utility_module = importlib.import_module(main_module_name)
                        
                        # Find all classes that inherit from UtilityBase
                        for attr_name in dir(utility_module):
                            attr = getattr(utility_module, attr_name)
                            
                            # Check if it's a class that inherits from UtilityBase
                            if (isinstance(attr, type) and 
                                issubclass(attr, UtilityBase) and 
                                attr is not UtilityBase):
                                
                                # Register the utility
                                self._registry[module_name] = attr
                                logger.info(f"Discovered utility: {module_name}")
                                break
                                
                    except ImportError:
                        logger.warning(f"Could not import main module for extension: {module_name}")
                        continue
                        
                except ImportError as e:
                    logger.warning(f"Failed to import extension package {module_name}: {e}")
                    continue
        
        return self._registry
    
    def get_utility_class(self, utility_name: str) -> Optional[Type[UtilityBase]]:
        """
        Get the class for a specific utility by name.
        
        Parameters
        ----------
        utility_name : str
            Name of the utility to get
            
        Returns
        -------
        Optional[Type[UtilityBase]]
            The utility class if found, None otherwise
        """
        return self._registry.get(utility_name)
    
    def instantiate_utility(self, utility_name: str, event_bus: Any, config: Dict[str, Any]) -> Optional[UtilityBase]:
        """
        Instantiate a utility by name with provided event bus and configuration.
        
        Parameters
        ----------
        utility_name : str
            Name of the utility to instantiate
        event_bus : Any
            Event bus instance to pass to the utility
        config : Dict[str, Any]
            Configuration dictionary for the utility
            
        Returns
        -------
        Optional[UtilityBase]
            Instantiated utility if successful, None otherwise
        """
        utility_class = self.get_utility_class(utility_name)
        
        if utility_class is None:
            logger.error(f"Utility not found: {utility_name}")
            return None
            
        try:
            utility_instance = utility_class(event_bus, config)
            return utility_instance
        except Exception as e:
            logger.error(f"Error instantiating utility {utility_name}: {e}")
            return None
    
    def get_available_utilities(self) -> List[str]:
        """
        Get a list of available utility names.
        
        Returns
        -------
        List[str]
            List of available utility names
        """
        return list(self._registry.keys())