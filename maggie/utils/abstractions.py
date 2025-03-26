"""
Maggie AI Assistant - Abstractions Module
=========================================

This module provides interfaces and a registry for component discovery,
enabling decoupling between core components.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable, Tuple, Union, Set, Type
import threading

# Interface definitions
class ILoggerProvider(ABC):
    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        pass
    
    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log an error message."""
        pass
    
    @abstractmethod
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log a critical message."""
        pass

class IErrorHandler(ABC):
    @abstractmethod
    def record_error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> Any:
        """Record an error."""
        pass
    
    @abstractmethod
    def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Safely execute a function."""
        pass

class IEventPublisher(ABC):
    @abstractmethod
    def publish(self, event_type: str, data: Any = None, **kwargs) -> None:
        """Publish an event."""
        pass

class IStateProvider(ABC):
    @abstractmethod
    def get_current_state(self) -> Any:
        """Get the current state."""
        pass

# Data containers for standalone use
class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class ErrorCategory(Enum):
    SYSTEM = auto()
    NETWORK = auto()
    RESOURCE = auto()
    PERMISSION = auto()
    CONFIGURATION = auto()
    INPUT = auto()
    PROCESSING = auto()
    MODEL = auto()
    EXTENSION = auto()
    STATE = auto()
    UNKNOWN = auto()

class ErrorSeverity(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

# Capability Registry
class CapabilityRegistry:
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = CapabilityRegistry()
        return cls._instance
    
    def __init__(self):
        self._registry = {}
    
    def register(self, capability_type: Type, instance: Any) -> None:
        """Register a capability instance."""
        self._registry[capability_type] = instance
    
    def get(self, capability_type: Type) -> Optional[Any]:
        """Get a capability instance."""
        return self._registry.get(capability_type)

# Helper functions
def get_logger_provider() -> Optional[ILoggerProvider]:
    """Get the registered logger provider."""
    return CapabilityRegistry.get_instance().get(ILoggerProvider)

def get_error_handler() -> Optional[IErrorHandler]:
    """Get the registered error handler."""
    return CapabilityRegistry.get_instance().get(IErrorHandler)

def get_event_publisher() -> Optional[IEventPublisher]:
    """Get the registered event publisher."""
    return CapabilityRegistry.get_instance().get(IEventPublisher)

def get_state_provider() -> Optional[IStateProvider]:
    """Get the registered state provider."""
    return CapabilityRegistry.get_instance().get(IStateProvider)