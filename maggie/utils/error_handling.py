import sys,traceback,logging,time,enum
from typing import Any,Callable,Optional,TypeVar,Dict,Union,List,Tuple,cast
from functools import wraps
T=TypeVar('T')
logger=logging.getLogger(__name__)
class ErrorSeverity(enum.Enum):DEBUG=0;INFO=1;WARNING=2;ERROR=3;CRITICAL=4
class ErrorCategory(enum.Enum):SYSTEM='system';NETWORK='network';RESOURCE='resource';PERMISSION='permission';CONFIGURATION='configuration';INPUT='input';PROCESSING='processing';MODEL='model';EXTENSION='extension';UNKNOWN='unknown'
class ErrorContext:
	def __init__(self,message:str,exception:Optional[Exception]=None,category:ErrorCategory=ErrorCategory.UNKNOWN,severity:ErrorSeverity=ErrorSeverity.ERROR,source:str='',details:Dict[str,Any]=None):
		self.message=message;self.exception=exception;self.category=category;self.severity=severity;self.source=source;self.details=details or{};self.timestamp=time.time()
		if exception:
			self.exception_type=type(exception).__name__;self.exception_msg=str(exception);exc_type,exc_value,exc_traceback=sys.exc_info()
			if exc_traceback:
				tb=traceback.extract_tb(exc_traceback)
				if tb:frame=tb[-1];self.filename=frame.filename;self.line=frame.lineno;self.function=frame.name;self.code=frame.line
	def to_dict(self)->Dict[str,Any]:
		result={'message':self.message,'category':self.category.value,'severity':self.severity.value,'source':self.source,'timestamp':self.timestamp}
		if hasattr(self,'exception_type'):result['exception']={'type':self.exception_type,'message':self.exception_msg}
		if hasattr(self,'filename'):result['location']={'file':self.filename,'line':self.line,'function':self.function,'code':self.code}
		if self.details:result['details']=self.details
		return result
def safe_execute(func:Callable[...,T],*args:Any,default_return:Optional[T]=None,error_message:str='Error executing function',error_category:ErrorCategory=ErrorCategory.UNKNOWN,error_severity:ErrorSeverity=ErrorSeverity.ERROR,publish_error:bool=True,**kwargs:Any)->T:
	try:return func(*args,**kwargs)
	except Exception as e:
		context=ErrorContext(message=error_message,exception=e,category=error_category,severity=error_severity,source=func.__module__+'.'+func.__name__,details={'args':str(args),'kwargs':str(kwargs)})
		if error_severity==ErrorSeverity.CRITICAL:logger.critical(f"{error_message}: {e}",exc_info=True)
		elif error_severity==ErrorSeverity.ERROR:logger.error(f"{error_message}: {e}",exc_info=True)
		elif error_severity==ErrorSeverity.WARNING:logger.warning(f"{error_message}: {e}")
		else:logger.debug(f"{error_message}: {e}")
		if publish_error:
			try:
				event_bus=get_event_bus()
				if event_bus:event_bus.publish('error_logged',context.to_dict())
			except Exception as event_error:logger.error(f"Failed to publish error event: {event_error}")
		return default_return if default_return is not None else cast(T,None)
def retry_operation(max_attempts:int=3,retry_delay:float=1.,exponential_backoff:bool=True,jitter:bool=True,allowed_exceptions:Tuple[Exception,...]=(Exception,),on_retry_callback:Optional[Callable[[Exception,int],None]]=None)->Callable:
	def decorator(func:Callable)->Callable:
		@wraps(func)
		def wrapper(*args:Any,**kwargs:Any)->Any:
			import random;last_exception=None
			for attempt in range(1,max_attempts+1):
				try:return func(*args,**kwargs)
				except allowed_exceptions as e:
					last_exception=e
					if attempt==max_attempts:logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}");raise
					delay=retry_delay
					if exponential_backoff:delay=retry_delay*2**(attempt-1)
					if jitter:delay=delay*(.5+random.random())
					if on_retry_callback:
						try:on_retry_callback(e,attempt)
						except Exception as callback_error:logger.warning(f"Error in retry callback: {callback_error}")
					logger.warning(f"Attempt {attempt}/{max_attempts} for {func.__name__} failed: {e}. Retrying in {delay:.2f}s");time.sleep(delay)
			if last_exception:raise last_exception
			return None
		return wrapper
	return decorator
def get_event_bus() -> Optional[Any]:
    try:
        from maggie.service.locator import ServiceLocator
        return ServiceLocator.get('event_bus')
    except ImportError:
        logger.warning("ServiceLocator not available, can't get event_bus")
        return None
    except Exception as e:
        logger.error(f"Error getting event_bus: {e}")
        return None
def record_error(message:str,exception:Optional[Exception]=None,category:ErrorCategory=ErrorCategory.UNKNOWN,severity:ErrorSeverity=ErrorSeverity.ERROR,source:str='',details:Dict[str,Any]=None,publish:bool=True)->ErrorContext:
	context=ErrorContext(message=message,exception=exception,category=category,severity=severity,source=source,details=details)
	if severity==ErrorSeverity.CRITICAL:logger.critical(message,exc_info=bool(exception))
	elif severity==ErrorSeverity.ERROR:logger.error(message,exc_info=bool(exception))
	elif severity==ErrorSeverity.WARNING:logger.warning(message)
	else:logger.debug(message)
	if publish:
		try:
			event_bus=get_event_bus()
			if event_bus:event_bus.publish('error_logged',context.to_dict())
		except Exception as e:logger.error(f"Failed to publish error event: {e}")
	return context