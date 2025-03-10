"""
Maggie AI Assistant - Utility Base Class
======================================
Abstract base class for all Maggie AI Assistant utility modules.
Provides a standard interface for utility modules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from loguru import logger


class UtilityBase(ABC):
    """
    Abstract base class for all utility modules.
    
    Parameters
    ----------
    event_bus : EventBus
        Reference to the central event bus
    config : Dict[str, Any]
        Configuration parameters for the utility
    """
    
    def __init__(self, event_bus, config: Dict[str, Any]):
        """
        Initialize the utility module.
        
        Parameters
        ----------
        event_bus : EventBus
            Reference to the central event bus
        config : Dict[str, Any]
            Configuration parameters for the utility
        """
        self.event_bus = event_bus
        self.config = config
        self.running = False
        self._initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize the utility module.
        
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
            self._initialize_resources()
            self._initialized = True
            return True
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
        """
        return True
        
    @abstractmethod
    def start(self) -> bool:
        """
        Start the utility module and return success status.
        
        Returns
        -------
        bool
            True if started successfully, False otherwise
        """
        pass
        
    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the utility module and return success status.
        
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
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with status information
        
        Notes
        -----
        Default implementation returns running state. Override for more details.
        """
        return {
            "running": self.running,
            "initialized": self._initialized
        }