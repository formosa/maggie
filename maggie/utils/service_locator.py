# In maggie/utils/service_locator.py
"""
Maggie AI Assistant - Service Locator
====================================

Service locator pattern implementation for Maggie AI Assistant.

This module provides a central registry for component references,
enabling utilities to access shared services without direct dependencies
on the main application class.

Examples
--------
>>> from maggie.utils.service_locator import ServiceLocator
>>> # Register a service
>>> ServiceLocator.register("stt_processor", stt_processor_instance)
>>> # Retrieve a service
>>> stt_processor = ServiceLocator.get("stt_processor")
>>> stt_processor.speak("Hello, world!")
"""

from typing import Dict, Any, Optional, List, Callable, Type, TypeVar, Generic, cast
from loguru import logger

# Type variable for generic service types
T = TypeVar('T')

class ServiceLocator:
    """
    Service locator for component references.
    
    Provides a central registry for component references to decouple
    utilities from specific component implementations.
    
    Attributes
    ----------
    _services : Dict[str, Any]
        Dictionary mapping service names to their instances
    """
    
    _services: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, service: Any) -> None:
        """
        Register a service with the service locator.
        
        Parameters
        ----------
        name : str
            Name of the service
        service : Any
            Service instance
        """
        cls._services[name] = service
        logger.debug(f"Registered service: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """
        Get a service by name.
        
        Parameters
        ----------
        name : str
            Name of the service
            
        Returns
        -------
        Optional[Any]
            Service instance if found, None otherwise
        """
        service = cls._services.get(name)
        if service is None:
            logger.warning(f"Service not found: {name}")
        return service
    
    @classmethod
    def get_typed(cls, name: str, service_type: Type[T]) -> Optional[T]:
        """
        Get a service by name with type checking.
        
        Parameters
        ----------
        name : str
            Name of the service
        service_type : Type[T]
            Expected type of the service
            
        Returns
        -------
        Optional[T]
            Service instance if found and of correct type, None otherwise
            
        Notes
        -----
        This method provides additional type safety by ensuring the
        retrieved service is of the expected type.
        
        Examples
        --------
        >>> from maggie.utils.stt.processor import STTProcessor
        >>> stt_processor = ServiceLocator.get_typed("stt_processor", STTProcessor)
        >>> if stt_processor:
        ...     stt_processor.speak("Hello")
        """
        service = cls.get(name)
        if service is None:
            return None
            
        if not isinstance(service, service_type):
            logger.error(
                f"Service type mismatch: {name} is {type(service).__name__}, "
                f"expected {service_type.__name__}"
            )
            return None
            
        return cast(T, service)
    
    @classmethod
    def has_service(cls, name: str) -> bool:
        """
        Check if a service is registered.
        
        Parameters
        ----------
        name : str
            Name of the service
            
        Returns
        -------
        bool
            True if the service is registered, False otherwise
        """
        return name in cls._services
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered services.
        """
        cls._services.clear()
        logger.debug("Cleared all services")