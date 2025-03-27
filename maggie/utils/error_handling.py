import sys,traceback,logging,time,enum,uuid,functools,threading
from typing import Any,Callable,Optional,TypeVar,Dict,Union,List,Tuple,cast,Type
T=TypeVar('T')
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger('maggie.error_handling')
class ErrorSeverity(enum.Enum):DEBUG=0;INFO=1;WARNING=2;ERROR=3;CRITICAL=4
class ErrorCategory(enum.Enum):SYSTEM='system';NETWORK='network';RESOURCE='resource';PERMISSION='permission';CONFIGURATION='configuration';INPUT='input';PROCESSING='processing';MODEL='model';EXTENSION='extension';STATE='state';UNKNOWN='unknown'
ERROR_EVENT_LOGGED='error_logged'
ERROR_EVENT_COMPONENT_FAILURE='component_failure'
ERROR_EVENT_RESOURCE_WARNING='resource_warning'
ERROR_EVENT_SYSTEM_ERROR='system_error'
ERROR_EVENT_STATE_TRANSITION='state_transition_error'
ERROR_EVENT_RESOURCE_MANAGEMENT='resource_management_error'
ERROR_EVENT_INPUT_PROCESSING='input_processing_error'
class MaggieError(Exception):pass
class LLMError(MaggieError):pass
class ModelLoadError(LLMError):pass
class GenerationError(LLMError):pass
class STTError(MaggieError):pass
class TTSError(MaggieError):pass
class ExtensionError(MaggieError):pass
class StateTransitionError(MaggieError):
	def __init__(self,message:str,from_state:Any=None,to_state:Any=None,trigger:str=None,details:Dict[str,Any]=None):self.from_state=from_state;self.to_state=to_state;self.trigger=trigger;self.details=details or{};super().__init__(message)
class ResourceManagementError(MaggieError):
	def __init__(self,message:str,resource_type:str=None,resource_name:str=None,details:Dict[str,Any]=None):self.resource_type=resource_type;self.resource_name=resource_name;self.details=details or{};super().__init__(message)
class InputProcessingError(MaggieError):
	def __init__(self,message:str,input_type:str=None,input_source:str=None,details:Dict[str,Any]=None):self.input_type=input_type;self.input_source=input_source;self.details=details or{};super().__init__(message)
class ErrorContext:
	def __init__(self,message:str,exception:Optional[Exception]=None,category:ErrorCategory=ErrorCategory.UNKNOWN,severity:ErrorSeverity=ErrorSeverity.ERROR,source:str='',details:Dict[str,Any]=None,correlation_id:Optional[str]=None,state_info:Optional[Dict[str,Any]]=None):
		self.message=message;self.exception=exception;self.category=category;self.severity=severity;self.source=source;self.details=details or{};self.correlation_id=correlation_id or str(uuid.uuid4());self.timestamp=time.time();self.state_info=state_info or{}
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
		if self.state_info:result['state']=self.state_info
		return result
	def log(self,publish:bool=True)->None:
		if self.severity==ErrorSeverity.CRITICAL:logger.critical(self.message,exc_info=bool(self.exception))
		elif self.severity==ErrorSeverity.ERROR:logger.error(self.message,exc_info=bool(self.exception))
		elif self.severity==ErrorSeverity.WARNING:logger.warning(self.message)
		else:logger.debug(self.message)
		try:
			from maggie.utils.abstractions import get_event_publisher;publisher=get_event_publisher()
			if publish and publisher:
				publisher.publish(ERROR_EVENT_LOGGED,self.to_dict())
				if self.category==ErrorCategory.STATE:publisher.publish(ERROR_EVENT_STATE_TRANSITION,self.to_dict())
				elif self.category==ErrorCategory.RESOURCE:publisher.publish(ERROR_EVENT_RESOURCE_MANAGEMENT,self.to_dict())
				elif self.category==ErrorCategory.INPUT:publisher.publish(ERROR_EVENT_INPUT_PROCESSING,self.to_dict())
		except ImportError:pass
		except Exception as e:logger.debug(f"Failed to publish error event: {e}")
def safe_execute(func:Callable[...,T],*args:Any,error_code:Optional[str]=None,default_return:Optional[T]=None,error_details:Dict[str,Any]=None,error_category:ErrorCategory=ErrorCategory.UNKNOWN,error_severity:ErrorSeverity=ErrorSeverity.ERROR,publish_error:bool=True,include_state_info:bool=True,**kwargs:Any)->T:
	try:return func(*args,**kwargs)
	except Exception as e:
		details=error_details or{}
		if not details:details={'args':str(args),'kwargs':str(kwargs)}
		source=f"{func.__module__}.{func.__name__}";current_state=None
		if include_state_info:
			try:
				from maggie.utils.abstractions import get_state_provider;state_provider=get_state_provider()
				if state_provider:current_state=state_provider.get_current_state()
			except ImportError:pass
			except Exception:pass
		context=ErrorContext(message=f"Error executing {func.__name__}: {e}",exception=e,category=error_category,severity=error_severity,source=source,details=details)
		if current_state is not None:context.state_info['current_state']=current_state.name if hasattr(current_state,'name')else str(current_state)
		context.log(publish=publish_error);return default_return if default_return is not None else cast(T,None)
def record_error(message:str,exception:Optional[Exception]=None,category:ErrorCategory=ErrorCategory.UNKNOWN,severity:ErrorSeverity=ErrorSeverity.ERROR,source:str='',details:Dict[str,Any]=None,publish:bool=True,state_object:Any=None,from_state:Any=None,to_state:Any=None,trigger:str=None)->ErrorContext:
	context=ErrorContext(message=message,exception=exception,category=category,severity=severity,source=source,details=details or{})
	if state_object is not None:state_name=state_object.name if hasattr(state_object,'name')else str(state_object);context.state_info['current_state']=state_name
	if from_state is not None and to_state is not None:from_name=from_state.name if hasattr(from_state,'name')else str(from_state);to_name=to_state.name if hasattr(to_state,'name')else str(to_state);context.state_info['transition']={'from':from_name,'to':to_name,'trigger':trigger}
	context.log(publish=publish);return context
def with_error_handling(error_code:Optional[str]=None,error_category:ErrorCategory=ErrorCategory.UNKNOWN,error_severity:ErrorSeverity=ErrorSeverity.ERROR,publish_error:bool=True,include_state_info:bool=True):
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):return safe_execute(func,*args,error_code=error_code,error_category=error_category,error_severity=error_severity,publish_error=publish_error,include_state_info=include_state_info,**kwargs)
		return wrapper
	return decorator
def retry_operation(max_attempts:int=3,retry_delay:float=1.,exponential_backoff:bool=True,jitter:bool=True,allowed_exceptions:Tuple[Type[Exception],...]=(Exception,),on_retry_callback:Optional[Callable[[Exception,int],None]]=None,error_category:ErrorCategory=ErrorCategory.UNKNOWN):
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
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
def create_state_transition_error(from_state:Any,to_state:Any,trigger:str,details:Dict[str,Any]=None)->StateTransitionError:from_name=from_state.name if hasattr(from_state,'name')else str(from_state);to_name=to_state.name if hasattr(to_state,'name')else str(to_state);message=f"Invalid state transition: {from_name} -> {to_name} (trigger: {trigger})";record_error(message=message,category=ErrorCategory.STATE,severity=ErrorSeverity.ERROR,source='StateManager.transition_to',details=details or{},from_state=from_state,to_state=to_state,trigger=trigger);return StateTransitionError(message=message,from_state=from_state,to_state=to_state,trigger=trigger,details=details)