"""
Maggie AI Assistant - Extension Base Class
======================================

Abstract base class for all Maggie AI Assistant extension modules.

This module defines the common interface and baseline functionality 
that all extension modules must implement to work properly with the
Maggie AI Assistant architecture. Extensions use this base class to ensure
consistent behavior, proper resource management, and standardized
lifecycle management.

The extension framework provides:
1. Standardized lifecycle methods (initialize, start, stop)
2. Thread-safe event-driven communication
3. State management and command processing
4. Resource acquisition and management
5. Error handling and recovery mechanisms

Extensions can be dynamically loaded at runtime and respond to voice
commands through their trigger phrases.

Examples
--------
>>> from maggie.extensions.base import ExtensionBase
>>> class MyExtension(ExtensionBase):
...     def get_trigger(self):
...         return "my command"
...     def start(self):
...         print("Starting extension")
...         self.running = True
...         return True
...     def stop(self):
...         print("Stopping extension")
...         self.running = False
...         return True
...     def process_command(self, command):
...         print(f"Processing command: {command}")
...         return command in ["example", "test"]
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Callable

# Third-party imports
from loguru import logger

__all__ = ['ExtensionBase']

class ExtensionBase(ABC):
    """
    Abstract base class for all extension modules.
    
    Defines the standard interface that all extension modules must implement
    to properly integrate with the Maggie AI core system. Handles lifecycle
    management, state tracking, and provides common extension functionality.
    
    Parameters
    ----------
    event_bus : object
        Reference to the central event bus for event-driven communication.
        Must implement publish(), subscribe(), and unsubscribe() methods.
    config : Dict[str, Any]
        Configuration dictionary containing extension-specific settings such as:
        - enabled: Whether the extension is enabled (default: True)
        - output_dir: Directory for extension output files (if applicable)
        - custom parameters specific to each extension
        
    Attributes
    ----------
    event_bus : object
        Reference to the central event bus
    config : Dict[str, Any]
        Configuration parameters
    running : bool
        Whether the extension is currently running
    _initialized : bool
        Whether the extension has been initialized
        
    Examples
    --------
    >>> from maggie.extensions.base import ExtensionBase
    >>> from maggie.core import EventBus
    >>> class WeatherExtension(ExtensionBase):
    ...     def get_trigger(self):
    ...         return "weather forecast"
    ...     def start(self):
    ...         print("Starting weather extension")
    ...         self.running = True
    ...         return True
    ...     def stop(self):
    ...         print("Stopping weather extension")
    ...         self.running = False
    ...         return True
    ...     def process_command(self, command):
    ...         if "temperature" in command:
    ...             print("Getting temperature")
    ...             return True
    ...         return False
    >>> event_bus = EventBus()
    >>> config = {"api_key": "abc123", "units": "metric"}
    >>> weather = WeatherExtension(event_bus, config)
    >>> weather.initialize()
    >>> weather.start()
    Starting weather extension
    >>> weather.process_command("What's the temperature today?")
    Getting temperature
    True
    """
    
    def __init__(self, event_bus, config: Dict[str, Any]):
        """
        Initialize the extension module.
        
        Parameters
        ----------
        event_bus : object
            Reference to the central event bus
        config : Dict[str, Any]
            Configuration parameters for the extension
        """
        self.event_bus = event_bus
        self.config = config
        self.running = False
        self._initialized = False
        
    @property
    def initialized(self) -> bool:
        """
        Check if the extension is initialized.
        
        Returns
        -------
        bool
            True if the extension is initialized, False otherwise
        """
        return self._initialized
        
    def initialize(self) -> bool:
        """
        Initialize the extension module.
        
        Perform one-time initialization tasks that should happen before
        the extension is started for the first time.
        
        Returns
        -------
        bool
            True if initialized successfully, False otherwise
        
        Notes
        -----
        This method is called before start() and can be used for resource initialization
        that should happen only once, not every time the extension is started.
        """
        if self._initialized:
            return True
            
        try:
            success = self._initialize_resources()
            self._initialized = success
            return success
        except Exception as e:
            logger.error(f"Error initializing {self.__class__.__name__}: {e}")
            return False
    
    def _initialize_resources(self) -> bool:
        """
        Initialize resources needed by this extension.
        
        Returns
        -------
        bool
            True if resources were initialized successfully
        
        Notes
        -----
        Override this method in subclasses to implement custom initialization.
        Default implementation returns True without doing anything.
        """
        return True
        
    @abstractmethod
    def get_trigger(self) -> str:
        """
        Get the trigger phrase for this extension.
        
        Returns
        -------
        str
            Trigger phrase that activates this extension
            
        Notes
        -----
        This is the phrase that the user can say to activate this extension.
        For example, "new recipe" for a recipe creator extension.
        """
        pass
        
    @abstractmethod
    def start(self) -> bool:
        """
        Start the extension module.
        
        Begins the primary functionality of the extension. This is called
        when the extension is activated by a trigger phrase or direct request.
        
        Returns
        -------
        bool
            True if started successfully, False otherwise
        """
        pass
        
    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the extension module.
        
        Stops the extension's operations and performs any necessary cleanup.
        This is called when the extension needs to be deactivated.
        
        Returns
        -------
        bool
            True if stopped successfully, False otherwise
        """
        pass
        
    @abstractmethod
    def process_command(self, command: str) -> bool:
        """
        Process a command directed to this extension.
        
        Handles user input that is directed to this extension while it's active.
        
        Parameters
        ----------
        command : str
            The command string to process
            
        Returns
        -------
        bool
            True if command was processed, False if not applicable
        """
        pass
    
    def pause(self) -> bool:
        """
        Pause the extension if it supports pausing.
        
        Temporarily suspends the extension's operations without fully stopping.
        
        Returns
        -------
        bool
            True if paused successfully, False otherwise
        
        Notes
        -----
        Default implementation does nothing. Override in subclasses if needed.
        """
        return False
    
    def resume(self) -> bool:
        """
        Resume the extension if it was paused.
        
        Resumes operations after a pause.
        
        Returns
        -------
        bool
            True if resumed successfully, False otherwise
        
        Notes
        -----
        Default implementation does nothing. Override in subclasses if needed.
        """
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the extension.
        
        Provides detailed status information about the extension's current state.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with status information including running state,
            initialization status, and extension name
        
        Notes
        -----
        Default implementation returns running state. Override for more details.
        """
        return {
            "running": self.running,
            "initialized": self._initialized,
            "name": self.__class__.__name__
        }
    
    def get_service(self, name: str) -> Optional[Any]:
        """
        Get a service from the service locator.
        
        Parameters
        ----------
        name : str
            Name of the service
            
        Returns
        -------
        Optional[Any]
            Service instance if found, None otherwise
        """
        try:
            from maggie.service.service_locator import ServiceLocator
            return ServiceLocator.get(name)
        except ImportError:
            logger.error("Service locator not available")
            return None