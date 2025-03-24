import os
import sys
import importlib
import importlib.util
import pkgutil
from typing import Dict, Type, Any, Optional, List, Tuple, cast

from maggie.extensions.base import ExtensionBase
from maggie.utils.error_handling import (
    safe_execute, 
    with_error_handling, 
    ErrorCategory, 
    ErrorSeverity, 
    record_error
)
from maggie.utils.logging import ComponentLogger, log_operation, logging_context

class ExtensionRegistry:
    """
    Registry for discovering and managing extensions for the Maggie AI Assistant.
    
    This class handles dynamic discovery, instantiation, and management of extension 
    modules that add functionality to the core Maggie AI system.
    """
    
    def __init__(self, extensions_path: Optional[str] = None):
        """
        Initialize the extension registry.
        
        Parameters
        ----------
        extensions_path : Optional[str]
            Custom path to extensions directory. If None, uses the default path.
        """
        self._registry: Dict[str, Type[ExtensionBase]] = {}
        self.logger = ComponentLogger('ExtensionRegistry')
        
        if extensions_path is None:
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self._extensions_path = os.path.join(package_dir, 'extensions')
        else:
            self._extensions_path = extensions_path
        
        os.makedirs(self._extensions_path, exist_ok=True)
        self.event_bus = self._get_event_bus()
    
    def _get_event_bus(self) -> Optional[Any]:
        """
        Get a reference to the central event bus.
        
        Returns
        -------
        Optional[Any]
            Event bus instance or None if not available
        """
        try:
            from maggie.service.locator import ServiceLocator
            return ServiceLocator.get('event_bus')
        except (ImportError, Exception) as e:
            self.logger.warning(f"Could not access event bus: {e}")
            return None
    
    @log_operation(component='ExtensionRegistry')
    @with_error_handling(error_category=ErrorCategory.EXTENSION)
    def discover_extensions(self) -> Dict[str, Type[ExtensionBase]]:
        """
        Discover all available extensions in the extensions directory.
        
        Returns
        -------
        Dict[str, Type[ExtensionBase]]
            Dictionary mapping extension names to their classes
        """
        self._registry = {}
        
        if self._extensions_path not in sys.path:
            sys.path.append(self._extensions_path)
        
        with logging_context(component='ExtensionRegistry', operation='discover_extensions'):
            for (_, module_name, is_pkg) in pkgutil.iter_modules([self._extensions_path]):
                if is_pkg:
                    self._process_extension_package(module_name)
            
            extension_count = len(self._registry)
            self.logger.info(f"Discovered {extension_count} extensions: {', '.join(self._registry.keys())}")
            return self._registry
    
    @with_error_handling(error_category=ErrorCategory.EXTENSION, error_severity=ErrorSeverity.WARNING)
    def _process_extension_package(self, module_name: str) -> None:
        """
        Process a potential extension package to extract extension classes.
        
        Parameters
        ----------
        module_name : str
            Name of the module to process
        """
        try:
            # First try to import the package itself
            extension_package = importlib.import_module(module_name)
            
            # Then try to import the main module
            main_module_name = f"{module_name}.{module_name}"
            try:
                extension_module = importlib.import_module(main_module_name)
                self._extract_extension_classes(extension_module, module_name)
            except ImportError:
                self.logger.warning(f"Could not import main module for extension: {module_name}")
                
        except ImportError:
            self.logger.warning(f"Failed to import extension package {module_name}")
            self._publish_extension_error(module_name, "Failed to import extension package")
            
        except Exception as e:
            self.logger.error(f"Unexpected error processing extension package {module_name}: {e}")
            self._publish_extension_error(module_name, str(e))
    
    def _extract_extension_classes(self, module: Any, module_name: str) -> None:
        """
        Extract extension classes from a module.
        
        Parameters
        ----------
        module : Any
            Module to extract extension classes from
        module_name : str
            Name of the module
        """
        extension_found = False
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            if (isinstance(attr, type) and 
                issubclass(attr, ExtensionBase) and 
                attr is not ExtensionBase):
                
                self._registry[module_name] = attr
                self.logger.info(f"Discovered extension: {module_name}")
                extension_found = True
                break
        
        if not extension_found:
            self.logger.debug(f"No valid extension class found in module: {module_name}")
    
    def _publish_extension_error(self, extension_name: str, error_message: str) -> None:
        """
        Publish an extension error to the event bus.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension with the error
        error_message : str
            Error message to publish
        """
        if self.event_bus:
            event_data = {
                'source': 'ExtensionRegistry',
                'extension': extension_name,
                'message': f"Extension error: {error_message}",
                'category': ErrorCategory.EXTENSION.value,
                'severity': ErrorSeverity.WARNING.value
            }
            self.event_bus.publish('error_logged', event_data)
    
    @log_operation(component='ExtensionRegistry')
    def get_extension_class(self, extension_name: str) -> Optional[Type[ExtensionBase]]:
        """
        Get an extension class by name.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension
            
        Returns
        -------
        Optional[Type[ExtensionBase]]
            Extension class if found, None otherwise
        """
        return self._registry.get(extension_name)
    
    @log_operation(component='ExtensionRegistry')
    @with_error_handling(error_category=ErrorCategory.EXTENSION)
    def instantiate_extension(self, extension_name: str, event_bus: Any, config: Dict[str, Any]) -> Optional[ExtensionBase]:
        """
        Instantiate an extension by name.
        
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
            self.logger.error(f"Extension not found: {extension_name}")
            self._publish_extension_error(extension_name, 'Extension not found')
            return None
        
        try:
            extension_instance = extension_class(event_bus, config)
            self.logger.info(f"Instantiated extension: {extension_name}")
            return extension_instance
        except Exception as e:
            self.logger.error(f"Error instantiating extension {extension_name}: {e}")
            self._publish_extension_error(extension_name, str(e))
            return None
    
    def get_available_extensions(self) -> List[str]:
        """
        Get a list of all available extension names.
        
        Returns
        -------
        List[str]
            List of extension names
        """
        return list(self._registry.keys())
    
    @log_operation(component='ExtensionRegistry')
    @with_error_handling(error_category=ErrorCategory.EXTENSION, error_severity=ErrorSeverity.WARNING)
    def reload_extension(self, extension_name: str) -> bool:
        """
        Reload an extension module.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension to reload
            
        Returns
        -------
        bool
            True if reload successful, False otherwise
        """
        if extension_name not in self._registry:
            self.logger.warning(f"Cannot reload unknown extension: {extension_name}")
            return False
        
        try:
            module_name = f"{extension_name}.{extension_name}"
            
            if module_name in sys.modules:
                module = sys.modules[module_name]
                importlib.reload(module)
                self._extract_extension_classes(module, extension_name)
                self.logger.info(f"Reloaded extension: {extension_name}")
                return True
            else:
                self.logger.warning(f"Extension module {module_name} not loaded, cannot reload")
                return False
                
        except Exception as e:
            self.logger.error(f"Error reloading extension {extension_name}: {e}")
            self._publish_extension_error(extension_name, f"Failed to reload: {e}")
            return False