# maggie/utils/service_locator.py
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
>>> ServiceLocator.register("speech_processor", speech_processor_instance)
>>> # Retrieve a service
>>> speech_processor = ServiceLocator.get("speech_processor")
>>> speech_processor.speak("Hello, world!")
"""

from typing import Dict, Any, Optional, List
from loguru import logger

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
    
    _services = {}
    
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