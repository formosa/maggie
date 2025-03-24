import sys,traceback,logging,time,enum,uuid,functools
from typing import Any,Callable,Optional,TypeVar,Dict,Union,List,Tuple,cast
from functools import wraps
T=TypeVar('T')
logger=logging.getLogger(__name__)
class ErrorSeverity(enum.Enum):DEBUG=0;INFO=1;WARNING=2;ERROR=3;CRITICAL=4
class ErrorCategory(enum.Enum):SYSTEM='system';NETWORK='network';RESOURCE='resource';PERMISSION='permission';CONFIGURATION='configuration';INPUT='input';PROCESSING='processing';MODEL='model';EXTENSION='extension';UNKNOWN='unknown'
class ErrorContext:
	def __init__(self,message:str,exception:Optional[Exception]=None,category:ErrorCategory=ErrorCategory.UNKNOWN,severity:ErrorSeverity=ErrorSeverity.ERROR,source:str='',details:Dict[str,Any]=None,correlation_id:Optional[str]=None):
		self.message=message;self.exception=exception;self.category=category;self.severity=severity;self.source=source;self.details=details or{};self.correlation_id=correlation_id or str(uuid.uuid4());self.timestamp=time.time()
		if exception:
			self.exception_type=type(exception).__name__;self.exception_msg=str(exception);exc_type,exc_value,exc_traceback=sys.exc_info()
			if exc_traceback:
				tb=traceback.extract_tb(exc_traceback)
				if tb:frame=tb[-1];self.filename=frame.filename;self.line=frame.lineno;self.function=frame.name;self.code=frame.line
	def to_dict(self)->Dict[str,Any]:
		result={'message':self.message,'category':self.category.value,'severity':self.severity.value,'source':self.source,'timestamp':self.timestamp,'correlation_id':self.correlation_id}
		if hasattr(self,'exception_type'):result['exception']={'type':self.exception_type,'message':self.exception_msg}
		if hasattr(self,'filename'):result['location']={'file':self.filename,'line':self.line,'function':self.function,'code':self.code}
		if self.details:result['details']=self.details
		return result
class ErrorRegistry:
	_instance=None
	@classmethod
	def get_instance(cls)->'ErrorRegistry':
		if cls._instance is None:cls._instance=ErrorRegistry()
		return cls._instance
	def __init__(self):self.errors={};self.register_error('CONFIG_LOAD_ERROR','Failed to load configuration: {details}',ErrorCategory.CONFIGURATION,ErrorSeverity.ERROR);self.register_error('RESOURCE_UNAVAILABLE','Required resource unavailable: {resource}',ErrorCategory.RESOURCE,ErrorSeverity.ERROR);self.register_error('MODEL_LOAD_ERROR','Failed to load model: {model_name}',ErrorCategory.MODEL,ErrorSeverity.ERROR);self.register_error('EXTENSION_INIT_ERROR','Failed to initialize extension: {extension_name}',ErrorCategory.EXTENSION,ErrorSeverity.ERROR);self.register_error('NETWORK_ERROR','Network error occurred: {details}',ErrorCategory.NETWORK,ErrorSeverity.ERROR);self.register_error('STATE_TRANSITION_ERROR','Invalid state transition: {from_state} -> {to_state}',ErrorCategory.SYSTEM,ErrorSeverity.ERROR);self.register_error('SPEECH_RECOGNITION_ERROR','Speech recognition failed: {details}',ErrorCategory.PROCESSING,ErrorSeverity.ERROR);self.register_error('TEXT_GENERATION_ERROR','Text generation failed: {details}',ErrorCategory.PROCESSING,ErrorSeverity.ERROR)
	def register_error(self,code:str,message_template:str,category:ErrorCategory=ErrorCategory.UNKNOWN,severity:ErrorSeverity=ErrorSeverity.ERROR)->None:
		if code in self.errors:logger.warning(f"Error code '{code}' already registered, overwriting")
		self.errors[code]={'message_template':message_template,'category':category,'severity':severity}
	def create_error(self,code:str,details:Dict[str,Any]=None,exception:Optional[Exception]=None,source:str='',severity_override:Optional[ErrorSeverity]=None)->ErrorContext:
		if code not in self.errors:logger.warning(f"Unknown error code: {code}, using generic error");return ErrorContext(message=f"Unknown error: {code}",exception=exception,source=source,details=details)
		error_def=self.errors[code];message=error_def['message_template']
		if details:
			try:message=message.format(**details)
			except KeyError as e:logger.warning(f"Missing key in error details: {e}")
		return ErrorContext(message=message,exception=exception,category=error_def['category'],severity=severity_override or error_def['severity'],source=source,details=details)
def get_event_bus()->Optional[Any]:
	try:from maggie.service.locator import ServiceLocator;return ServiceLocator.get('event_bus')
	except ImportError:logger.warning("ServiceLocator not available, can't get event_bus");return None
	except Exception as e:logger.error(f"Error getting event_bus: {e}");return None
def safe_execute(func:Callable[...,T],*args:Any,error_code:Optional[str]=None,default_return:Optional[T]=None,error_details:Dict[str,Any]=None,error_category:ErrorCategory=ErrorCategory.UNKNOWN,error_severity:ErrorSeverity=ErrorSeverity.ERROR,publish_error:bool=True,**kwargs:Any)->T:
	try:return func(*args,**kwargs)
	except Exception as e:
		details=error_details or{}
		if not details:details={'args':str(args),'kwargs':str(kwargs)}
		source=f"{func.__module__}.{func.__name__}"
		if error_code and ErrorRegistry.get_instance().errors.get(error_code):context=ErrorRegistry.get_instance().create_error(code=error_code,details=details,exception=e,source=source)
		else:message=error_code if error_code else f"Error executing {func.__name__}: {e}";context=ErrorContext(message=message,exception=e,category=error_category,severity=error_severity,source=source,details=details)
		if context.severity==ErrorSeverity.CRITICAL:logger.critical(context.message,exc_info=True)
		elif context.severity==ErrorSeverity.ERROR:logger.error(context.message,exc_info=True)
		elif context.severity==ErrorSeverity.WARNING:logger.warning(context.message)
		else:logger.debug(context.message)
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
class LLMError(Exception):pass
class ModelLoadError(LLMError):pass
class GenerationError(LLMError):pass
class STTError(Exception):pass
class TTSError(Exception):pass
class ExtensionError(Exception):pass
ERROR_EVENT_LOGGED='error_logged'
ERROR_EVENT_COMPONENT_FAILURE='component_failure'
ERROR_EVENT_RESOURCE_WARNING='resource_warning'
ERROR_EVENT_SYSTEM_ERROR='system_error'
def with_error_handling(error_code:Optional[str]=None,error_category:ErrorCategory=ErrorCategory.UNKNOWN,error_severity:ErrorSeverity=ErrorSeverity.ERROR,publish_error:bool=True):
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):return safe_execute(func,*args,error_code=error_code,error_category=error_category,error_severity=error_severity,publish_error=publish_error,**kwargs)
		return wrapper
	return decorator