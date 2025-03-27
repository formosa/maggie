from typing import Dict,Any,Optional,Callable
from maggie.utils.abstractions import ILoggerProvider,IErrorHandler,IEventPublisher,IStateProvider,CapabilityRegistry
class LoggingManagerAdapter(ILoggerProvider):
	def __init__(self,logging_manager):self.logging_manager=logging_manager;registry=CapabilityRegistry.get_instance();registry.register(ILoggerProvider,self)
	def debug(self,message:str,**kwargs)->None:from maggie.utils.logging import LogLevel;self.logging_manager.log(LogLevel.DEBUG,message,**kwargs)
	def info(self,message:str,**kwargs)->None:from maggie.utils.logging import LogLevel;self.logging_manager.log(LogLevel.INFO,message,**kwargs)
	def warning(self,message:str,**kwargs)->None:from maggie.utils.logging import LogLevel;self.logging_manager.log(LogLevel.WARNING,message,**kwargs)
	def error(self,message:str,exception:Optional[Exception]=None,**kwargs)->None:from maggie.utils.logging import LogLevel;self.logging_manager.log(LogLevel.ERROR,message,exception=exception,**kwargs)
	def critical(self,message:str,exception:Optional[Exception]=None,**kwargs)->None:from maggie.utils.logging import LogLevel;self.logging_manager.log(LogLevel.CRITICAL,message,exception=exception,**kwargs)
class ErrorHandlerAdapter(IErrorHandler):
	def __init__(self):registry=CapabilityRegistry.get_instance();registry.register(IErrorHandler,self)
	def record_error(self,message:str,exception:Optional[Exception]=None,**kwargs)->Any:from maggie.utils.error_handling import record_error as do_record_error,ErrorCategory,ErrorSeverity;category=kwargs.pop('category',ErrorCategory.UNKNOWN);severity=kwargs.pop('severity',ErrorSeverity.ERROR);source=kwargs.pop('source','');details=kwargs.pop('details',None);publish=kwargs.pop('publish',True);return do_record_error(message=message,exception=exception,category=category,severity=severity,source=source,details=details,publish=publish,**kwargs)
	def safe_execute(self,func:Callable,*args,**kwargs)->Any:from maggie.utils.error_handling import safe_execute as do_safe_execute,ErrorCategory,ErrorSeverity;error_code=kwargs.pop('error_code',None);default_return=kwargs.pop('default_return',None);error_details=kwargs.pop('error_details',None);error_category=kwargs.pop('error_category',ErrorCategory.UNKNOWN);error_severity=kwargs.pop('error_severity',ErrorSeverity.ERROR);publish_error=kwargs.pop('publish_error',True);include_state_info=kwargs.pop('include_state_info',True);return do_safe_execute(func,*args,error_code=error_code,default_return=default_return,error_details=error_details,error_category=error_category,error_severity=error_severity,publish_error=publish_error,include_state_info=include_state_info,**kwargs)
class EventBusAdapter(IEventPublisher):
	def __init__(self,event_bus):self.event_bus=event_bus;registry=CapabilityRegistry.get_instance();registry.register(IEventPublisher,self)
	def publish(self,event_type:str,data:Any=None,**kwargs)->None:
		priority=kwargs.pop('priority',None)
		if priority is not None:self.event_bus.publish(event_type,data,priority)
		else:self.event_bus.publish(event_type,data)
class StateManagerAdapter(IStateProvider):
	def __init__(self,state_manager):self.state_manager=state_manager;registry=CapabilityRegistry.get_instance();registry.register(IStateProvider,self)
	def get_current_state(self)->Any:return self.state_manager.get_current_state()