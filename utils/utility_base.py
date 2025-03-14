"""
Maggie AI Assistant - Utility Base Class
======================================

Abstract base class for all Maggie AI Assistant utility modules.

This module defines the common interface and baseline functionality 
that all utility modules must implement to work properly with the
Maggie AI Assistant architecture.

Examples
--------
>>> from utils.utility_base import UtilityBase
>>> class MyUtility(UtilityBase):
...     def get_trigger(self):
...         return "my command"
...     def start(self):
...         print("Starting utility")
...         return True
...     def stop(self):
...         print("Stopping utility")
...         return True
...     def process_command(self, command):
...         print(f"Processing command: {command}")
...         return True
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Callable

# Third-party imports
from loguru import logger

__all__ = ['UtilityBase']

class UtilityBase(ABC):
    """
    Abstract base class for all utility modules.
    
    Defines the standard interface that all utility modules must implement
    to properly integrate with the Maggie AI core system. Handles lifecycle
    management, state tracking, and provides common utility functionality.
    
    Parameters
    ----------
    event_bus : object
        Reference to the central event bus for event-driven communication
    config : Dict[str, Any]
        Configuration parameters for the utility
        
    Attributes
    ----------
    event_bus : object
        Reference to the central event bus
    config : Dict[str, Any]
        Configuration parameters
    running : bool
        Whether the utility is currently running
    _initialized : bool
        Whether the utility has been initialized
    """
    
    def __init__(self, event_bus, config: Dict[str, Any]):
        """
        Initialize the utility module.
        
        Parameters
        ----------
        event_bus : object
            Reference to the central event bus
        config : Dict[str, Any]
            Configuration parameters for the utility
        """
        self.event_bus = event_bus
        self.config = config
        self.running = False
        self._initialized = False
        
    @property
    def initialized(self) -> bool:
        """
        Check if the utility is initialized.
        
        Returns
        -------
        bool
            True if the utility is initialized, False otherwise
        """
        return self._initialized
        
    def initialize(self) -> bool:
        """
        Initialize the utility module.
        
        Perform one-time initialization tasks that should happen before
        the utility is started for the first time.
        
        Returns
        -------
        bool
            True if initialized successfully, False otherwise
        
        Notes
        -----
        This method is called before start() and can be used for resource initialization
        that should happen only once, not every time the utility is started.
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
        Initialize resources needed by this utility.
        
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
        Get the trigger phrase for this utility.
        
        Returns
        -------
        str
            Trigger phrase that activates this utility
            
        Notes
        -----
        This is the phrase that the user can say to activate this utility.
        For example, "new recipe" for a recipe creator utility.
        """
        pass
        
    @abstractmethod
    def start(self) -> bool:
        """
        Start the utility module.
        
        Begins the primary functionality of the utility. This is called
        when the utility is activated by a trigger phrase or direct request.
        
        Returns
        -------
        bool
            True if started successfully, False otherwise
        """
        pass
        
    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the utility module.
        
        Stops the utility's operations and performs any necessary cleanup.
        This is called when the utility needs to be deactivated.
        
        Returns
        -------
        bool
            True if stopped successfully, False otherwise
        """
        pass
        
    @abstractmethod
    def process_command(self, command: str) -> bool:
        """
        Process a command directed to this utility.
        
        Handles user input that is directed to this utility while it's active.
        
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
        Pause the utility if it supports pausing.
        
        Temporarily suspends the utility's operations without fully stopping.
        
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
        Resume the utility if it was paused.
        
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
        Get the current status of the utility.
        
        Provides detailed status information about the utility's current state.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with status information including running state,
            initialization status, and utility name
        
        Notes
        -----
        Default implementation returns running state. Override for more details.
        """
        return {
            "running": self.running,
            "initialized": self._initialized,
            "name": self.__class__.__name__
        }