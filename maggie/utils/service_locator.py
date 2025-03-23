# Updated maggie/utils/service_locator.py
"""
Maggie AI Assistant - Service Locator
====================================

Service locator pattern implementation for Maggie AI Assistant.

This module provides a central registry for component references,
enabling utilities to access shared services without direct dependencies
on the main application class.

The ServiceLocator pattern reduces coupling between components while
maintaining a centralized approach to service location and management.
"""

from typing import Dict, Any, Optional, List, Callable, Type, TypeVar, Generic, cast, Union
from loguru import logger

# Type variable for generic service types
T = TypeVar('T')

class ServiceLocator:
    """
    Service locator for component references.
    
    This class provides a central registry for component references to decouple
    utilities from specific component implementations. It uses a static registry
    pattern to store and retrieve service instances.
    
    Attributes
    ----------
    _services : Dict[str, Any]
        Dictionary mapping service names to their instances
    """
    
    # Static registry of services
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
            
        Notes
        -----
        This method registers a service under the given name. If a service
        with the same name already exists, it will be overwritten.
        
        Examples
        --------
        >>> from maggie.core import EventBus
        >>> event_bus = EventBus()
        >>> ServiceLocator.register("event_bus", event_bus)
        >>> 
        >>> from maggie.utils.stt.processor import STTProcessor
        >>> stt = STTProcessor(config)
        >>> ServiceLocator.register("stt_processor", stt)
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
            
        Notes
        -----
        This method retrieves a service by name. If the service is not found,
        it returns None and logs a warning.
        
        Examples
        --------
        >>> event_bus = ServiceLocator.get("event_bus")
        >>> if event_bus:
        ...     event_bus.publish("my_event", "event data")
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
        >>> 
        >>> from maggie.utils.llm.processor import LLMProcessor
        >>> llm = ServiceLocator.get_typed("llm_processor", LLMProcessor)
        >>> if llm:
        ...     response = llm.generate_text("Tell me a story")
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
            
        Examples
        --------
        >>> if ServiceLocator.has_service("event_bus"):
        ...     event_bus = ServiceLocator.get("event_bus")
        ...     event_bus.publish("my_event", data)
        ... else:
        ...     logger.warning("Event bus not available")
        """
        return name in cls._services
    
    @classmethod
    def get_or_create(cls, name: str, factory: Callable[[], T]) -> T:
        """
        Get a service by name or create it if not exists.
        
        Parameters
        ----------
        name : str
            Name of the service
        factory : Callable[[], T]
            Factory function to create the service if not found
            
        Returns
        -------
        T
            Existing or newly created service instance
            
        Notes
        -----
        This method provides a convenient way to implement the service
        locator pattern with lazy initialization. It registers the newly
        created service automatically.
        
        Examples
        --------
        >>> def create_event_bus():
        ...     from maggie.core import EventBus
        ...     return EventBus()
        >>> 
        >>> event_bus = ServiceLocator.get_or_create("event_bus", create_event_bus)
        >>> event_bus.publish("my_event", data)
        """
        service = cls.get(name)
        if service is None:
            service = factory()
            cls.register(name, service)
        return service
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered services.
        
        Notes
        -----
        This method removes all registered services from the registry.
        It's useful during shutdown or for testing.
        
        Examples
        --------
        >>> # During shutdown
        >>> ServiceLocator.clear()
        >>> 
        >>> # For testing isolation
        >>> def setUp():
        ...     ServiceLocator.clear()
        ...     # Register test services
        """
        cls._services.clear()
        logger.debug("Cleared all services")
        
    @classmethod
    def list_services(cls) -> List[str]:
        """
        Get a list of all registered service names.
        
        Returns
        -------
        List[str]
            List of registered service names
            
        Examples
        --------
        >>> services = ServiceLocator.list_services()
        >>> print("Registered services:")
        >>> for service in services:
        ...     print(f"  - {service}")
        """
        return list(cls._services.keys())