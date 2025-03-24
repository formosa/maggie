from typing import Dict, Any, Optional, List, Callable, Type, TypeVar, Generic, cast, Union
from loguru import logger

T = TypeVar('T')

class ServiceLocator:
    _services: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, name: str, service: Any) -> None:
        cls._services[name] = service
        logger.debug(f"Registered service: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        service = cls._services.get(name)
        if service is None:
            logger.warning(f"Service not found: {name}")
        return service
    
    @classmethod
    def get_typed(cls, name: str, service_type: Type[T]) -> Optional[T]:
        service = cls.get(name)
        if service is None:
            return None
        if not isinstance(service, service_type):
            logger.error(f"Service type mismatch: {name} is {type(service).__name__}, expected {service_type.__name__}")
            return None
        return cast(T, service)
    
    @classmethod
    def has_service(cls, name: str) -> bool:
        return name in cls._services
    
    @classmethod
    def get_or_create(cls, name: str, factory: Callable[[], T]) -> T:
        service = cls.get(name)
        if service is None:
            service = factory()
            cls.register(name, service)
        return service
    
    @classmethod
    def clear(cls) -> None:
        cls._services.clear()
        logger.debug('Cleared all services')
    
    @classmethod
    def list_services(cls) -> List[str]:
        return list(cls._services.keys())