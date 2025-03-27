from abc import ABC,abstractmethod
from enum import Enum,auto
from typing import Dict,Any,Optional,List,Callable,Tuple,Union,Set,Type
import threading
class ILoggerProvider(ABC):
	@abstractmethod
	def debug(self,message:str,**kwargs)->None:pass
	@abstractmethod
	def info(self,message:str,**kwargs)->None:pass
	@abstractmethod
	def warning(self,message:str,**kwargs)->None:pass
	@abstractmethod
	def error(self,message:str,exception:Optional[Exception]=None,**kwargs)->None:pass
	@abstractmethod
	def critical(self,message:str,exception:Optional[Exception]=None,**kwargs)->None:pass
class IErrorHandler(ABC):
	@abstractmethod
	def record_error(self,message:str,exception:Optional[Exception]=None,**kwargs)->Any:pass
	@abstractmethod
	def safe_execute(self,func:Callable,*args,**kwargs)->Any:pass
class IEventPublisher(ABC):
	@abstractmethod
	def publish(self,event_type:str,data:Any=None,**kwargs)->None:pass
class IStateProvider(ABC):
	@abstractmethod
	def get_current_state(self)->Any:pass
class LogLevel(Enum):DEBUG=auto();INFO=auto();WARNING=auto();ERROR=auto();CRITICAL=auto()
class ErrorCategory(Enum):SYSTEM=auto();NETWORK=auto();RESOURCE=auto();PERMISSION=auto();CONFIGURATION=auto();INPUT=auto();PROCESSING=auto();MODEL=auto();EXTENSION=auto();STATE=auto();UNKNOWN=auto()
class ErrorSeverity(Enum):DEBUG=auto();INFO=auto();WARNING=auto();ERROR=auto();CRITICAL=auto()
class CapabilityRegistry:
	_instance=None;_lock=threading.RLock()
	@classmethod
	def get_instance(cls):
		if cls._instance is None:
			with cls._lock:
				if cls._instance is None:cls._instance=CapabilityRegistry()
		return cls._instance
	def __init__(self):self._registry={}
	def register(self,capability_type:Type,instance:Any)->None:self._registry[capability_type]=instance
	def get(self,capability_type:Type)->Optional[Any]:return self._registry.get(capability_type)
def get_logger_provider()->Optional[ILoggerProvider]:return CapabilityRegistry.get_instance().get(ILoggerProvider)
def get_error_handler()->Optional[IErrorHandler]:return CapabilityRegistry.get_instance().get(IErrorHandler)
def get_event_publisher()->Optional[IEventPublisher]:return CapabilityRegistry.get_instance().get(IEventPublisher)
def get_state_provider()->Optional[IStateProvider]:return CapabilityRegistry.get_instance().get(IStateProvider)