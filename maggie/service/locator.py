from typing import Dict, Any, Optional, List, Callable, Type, TypeVar, Generic, cast, Union
from loguru import logger

T = TypeVar('T')

class ServiceLocator:
    _services: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, service: Any) -> None:
        """
        Register a service with the specified name.
        
        Parameters
        ----------
        name : str
            The name to register the service under
        service : Any
            The service instance to register
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
            The name of the service to retrieve
            
        Returns
        -------
        Optional[Any]
            The service instance if found, None otherwise
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
            The name of the service to retrieve
        service_type : Type[T]
            The expected type of the service
            
        Returns
        -------
        Optional[T]
            The service instance if found and of the correct type, None otherwise
        """
        service = cls.get(name)
        if service is None:
            return None
        if not isinstance(service, service_type):
            logger.error(f"Service type mismatch: {name} is {type(service).__name__}, expected {service_type.__name__}")
            return None
        return cast(T, service)
    
    @classmethod
    def has_service(cls, name: str) -> bool:
        """
        Check if a service with the specified name exists.
        
        Parameters
        ----------
        name : str
            The name of the service to check
            
        Returns
        -------
        bool
            True if the service exists, False otherwise
        """
        return name in cls._services
    
    @classmethod
    def get_or_create(cls, name: str, factory: Callable[[], T]) -> T:
        """
        Get a service by name, creating it if it doesn't exist.
        
        Parameters
        ----------
        name : str
            The name of the service to retrieve or create
        factory : Callable[[], T]
            Function to create the service if it doesn't exist
            
        Returns
        -------
        T
            The existing or newly created service
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
        """
        cls._services.clear()
        logger.debug('Cleared all services')
    
    @classmethod
    def list_services(cls) -> List[str]:
        """
        Get a list of all registered service names.
        
        Returns
        -------
        List[str]
            List of registered service names
        """
        return list(cls._services.keys())