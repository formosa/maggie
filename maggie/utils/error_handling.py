import sys,traceback,logging,time,enum,uuid,functools,threading
from typing import Any,Callable,Optional,TypeVar,Dict,Union,List,Tuple,cast,Type
from functools import wraps
T=TypeVar('T')
logger=logging.getLogger(__name__)
class ErrorSeverity(enum.Enum):DEBUG=0;INFO=1;WARNING=2;ERROR=3;CRITICAL=4
class ErrorCategory(enum.Enum):SYSTEM='system';NETWORK='network';RESOURCE='resource';PERMISSION='permission';CONFIGURATION='configuration';INPUT='input';PROCESSING='processing';MODEL='model';EXTENSION='extension';STATE='state';UNKNOWN='unknown'
ERROR_EVENT_LOGGED='error_logged'
ERROR_EVENT_COMPONENT_FAILURE='component_failure'
ERROR_EVENT_RESOURCE_WARNING='resource_warning'
ERROR_EVENT_SYSTEM_ERROR='system_error'
ERROR_EVENT_STATE_TRANSITION='state_transition_error'
ERROR_EVENT_RESOURCE_MANAGEMENT='resource_management_error'
ERROR_EVENT_INPUT_PROCESSING='input_processing_error'
class LLMError(Exception):pass
class ModelLoadError(LLMError):pass
class GenerationError(LLMError):pass
class STTError(Exception):pass
class TTSError(Exception):pass
class ExtensionError(Exception):pass
class StateTransitionError(Exception):
	def __init__(self,message:str,from_state:Any=None,to_state:Any=None,trigger:str=None,details:Dict[str,Any]=None):self.from_state=from_state;self.to_state=to_state;self.trigger=trigger;self.details=details or{};super().__init__(message)
class ResourceManagementError(Exception):
	def __init__(self,message:str,resource_type:str=None,resource_name:str=None,details:Dict[str,Any]=None):self.resource_type=resource_type;self.resource_name=resource_name;self.details=details or{};super().__init__(message)
class InputProcessingError(Exception):
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
		if publish:
			try:
				event_bus=get_event_bus()
				if event_bus:event_bus.publish('error_logged',self.to_dict())
			except Exception as e:logger.error(f"Failed to publish error event: {e}")
	def add_state_info(self,state_object:Any)->None:
		if state_object is not None:
			state_name=state_object.name if hasattr(state_object,'name')else str(state_object);self.state_info['current_state']=state_name
			if hasattr(state_object,'get_style'):self.state_info['style']=state_object.get_style()
	def add_transition_info(self,from_state:Any,to_state:Any,trigger:str)->None:
		if from_state is not None and to_state is not None:from_name=from_state.name if hasattr(from_state,'name')else str(from_state);to_name=to_state.name if hasattr(to_state,'name')else str(to_state);self.state_info['transition']={'from':from_name,'to':to_name,'trigger':trigger}
	def add_resource_info(self,resource_info:Dict[str,Any])->None:
		if resource_info:self.details['resources']=resource_info
class ErrorRegistry:
	_instance=None;_lock=threading.RLock()
	@classmethod
	def get_instance(cls)->'ErrorRegistry':
		if cls._instance is None:
			with cls._lock:
				if cls._instance is None:cls._instance=ErrorRegistry()
		return cls._instance
	def __init__(self):self.errors={};self.register_error('CONFIG_LOAD_ERROR','Failed to load configuration: {details}',ErrorCategory.CONFIGURATION,ErrorSeverity.ERROR);self.register_error('RESOURCE_UNAVAILABLE','Required resource unavailable: {resource}',ErrorCategory.RESOURCE,ErrorSeverity.ERROR);self.register_error('MODEL_LOAD_ERROR','Failed to load model: {model_name}',ErrorCategory.MODEL,ErrorSeverity.ERROR);self.register_error('EXTENSION_INIT_ERROR','Failed to initialize extension: {extension_name}',ErrorCategory.EXTENSION,ErrorSeverity.ERROR);self.register_error('NETWORK_ERROR','Network error occurred: {details}',ErrorCategory.NETWORK,ErrorSeverity.ERROR);self.register_error('STATE_TRANSITION_ERROR','Invalid state transition: {from_state} -> {to_state}',ErrorCategory.STATE,ErrorSeverity.ERROR);self.register_error('STATE_HANDLER_ERROR','Error in state handler for {state}: {details}',ErrorCategory.STATE,ErrorSeverity.ERROR);self.register_error('STATE_EVENT_ERROR','Failed to process state event: {event_type}',ErrorCategory.STATE,ErrorSeverity.ERROR);self.register_error('RESOURCE_ALLOCATION_ERROR','Failed to allocate {resource_type} resource: {details}',ErrorCategory.RESOURCE,ErrorSeverity.ERROR);self.register_error('RESOURCE_DEALLOCATION_ERROR','Failed to deallocate {resource_type} resource: {details}',ErrorCategory.RESOURCE,ErrorSeverity.ERROR);self.register_error('RESOURCE_LIMIT_EXCEEDED','{resource_type} limit exceeded: {details}',ErrorCategory.RESOURCE,ErrorSeverity.WARNING);self.register_error('INPUT_VALIDATION_ERROR','Input validation failed: {details}',ErrorCategory.INPUT,ErrorSeverity.ERROR);self.register_error('INPUT_PROCESSING_ERROR','Failed to process {input_type} input: {details}',ErrorCategory.INPUT,ErrorSeverity.ERROR);self.register_error('INPUT_FIELD_ERROR','Error in input field: {field_name}',ErrorCategory.INPUT,ErrorSeverity.ERROR);self.register_error('SPEECH_RECOGNITION_ERROR','Speech recognition failed: {details}',ErrorCategory.PROCESSING,ErrorSeverity.ERROR);self.register_error('TEXT_GENERATION_ERROR','Text generation failed: {details}',ErrorCategory.PROCESSING,ErrorSeverity.ERROR)
	def register_error(self,code:str,message_template:str,category:ErrorCategory=ErrorCategory.UNKNOWN,severity:ErrorSeverity=ErrorSeverity.ERROR)->None:
		with self._lock:
			if code in self.errors:logger.warning(f"Error code '{code}' already registered, overwriting")
			self.errors[code]={'message_template':message_template,'category':category,'severity':severity}
	def create_error(self,code:str,details:Dict[str,Any]=None,exception:Optional[Exception]=None,source:str='',severity_override:Optional[ErrorSeverity]=None,state_object:Any=None,from_state:Any=None,to_state:Any=None,trigger:str=None)->ErrorContext:
		with self._lock:
			if code not in self.errors:logger.warning(f"Unknown error code: {code}, using generic error");return ErrorContext(message=f"Unknown error: {code}",exception=exception,source=source,details=details)
			error_def=self.errors[code];message=error_def['message_template']
			if details:
				try:message=message.format(**details)
				except KeyError as e:logger.warning(f"Missing key in error details: {e}")
			context=ErrorContext(message=message,exception=exception,category=error_def['category'],severity=severity_override or error_def['severity'],source=source,details=details)
			if state_object is not None:context.add_state_info(state_object)
			if from_state is not None and to_state is not None:context.add_transition_info(from_state,to_state,trigger)
			return context
def get_event_bus()->Optional[Any]:
	try:from maggie.service.locator import ServiceLocator;return ServiceLocator.get('event_bus')
	except ImportError:logger.warning("ServiceLocator not available, can't get event_bus");return None
	except Exception as e:logger.error(f"Error getting event_bus: {e}");return None
def get_state_manager()->Optional[Any]:
	try:from maggie.service.locator import ServiceLocator;return ServiceLocator.get('state_manager')
	except ImportError:logger.warning("ServiceLocator not available, can't get state_manager");return None
	except Exception as e:logger.error(f"Error getting state_manager: {e}");return None
def get_resource_manager()->Optional[Any]:
	try:from maggie.service.locator import ServiceLocator;return ServiceLocator.get('resource_manager')
	except ImportError:logger.warning("ServiceLocator not available, can't get resource_manager");return None
	except Exception as e:logger.error(f"Error getting resource_manager: {e}");return None
def get_current_state()->Optional[Any]:
	state_manager=get_state_manager()
	if state_manager:
		try:return state_manager.get_current_state()
		except Exception as e:logger.error(f"Error getting current state: {e}")
	return None
def safe_execute(func:Callable[...,T],*args:Any,error_code:Optional[str]=None,default_return:Optional[T]=None,error_details:Dict[str,Any]=None,error_category:ErrorCategory=ErrorCategory.UNKNOWN,error_severity:ErrorSeverity=ErrorSeverity.ERROR,publish_error:bool=True,include_state_info:bool=True,**kwargs:Any)->T:
	try:return func(*args,**kwargs)
	except Exception as e:
		details=error_details or{}
		if not details:details={'args':str(args),'kwargs':str(kwargs)}
		source=f"{func.__module__}.{func.__name__}";current_state=get_current_state()if include_state_info else None
		if error_code and ErrorRegistry.get_instance().errors.get(error_code):context=ErrorRegistry.get_instance().create_error(code=error_code,details=details,exception=e,source=source,state_object=current_state)
		else:
			message=error_code if error_code else f"Error executing {func.__name__}: {e}";context=ErrorContext(message=message,exception=e,category=error_category,severity=error_severity,source=source,details=details)
			if current_state is not None:context.add_state_info(current_state)
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
def retry_operation(max_attempts:int=3,retry_delay:float=1.,exponential_backoff:bool=True,jitter:bool=True,allowed_exceptions:Tuple[Type[Exception],...]=(Exception,),on_retry_callback:Optional[Callable[[Exception,int],None]]=None,error_category:ErrorCategory=ErrorCategory.UNKNOWN)->Callable:
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
def record_error(message:str,exception:Optional[Exception]=None,category:ErrorCategory=ErrorCategory.UNKNOWN,severity:ErrorSeverity=ErrorSeverity.ERROR,source:str='',details:Dict[str,Any]=None,publish:bool=True,state_object:Any=None,from_state:Any=None,to_state:Any=None,trigger:str=None)->ErrorContext:
	context=ErrorContext(message=message,exception=exception,category=category,severity=severity,source=source,details=details)
	if state_object is not None:context.add_state_info(state_object)
	if from_state is not None and to_state is not None:context.add_transition_info(from_state,to_state,trigger)
	if severity==ErrorSeverity.CRITICAL:logger.critical(message,exc_info=bool(exception))
	elif severity==ErrorSeverity.ERROR:logger.error(message,exc_info=bool(exception))
	elif severity==ErrorSeverity.WARNING:logger.warning(message)
	else:logger.debug(message)
	if publish:
		try:
			event_bus=get_event_bus()
			if event_bus:
				event_data=context.to_dict()
				if category==ErrorCategory.STATE and from_state and to_state:event_bus.publish(ERROR_EVENT_STATE_TRANSITION,event_data)
				elif category==ErrorCategory.RESOURCE:event_bus.publish(ERROR_EVENT_RESOURCE_MANAGEMENT,event_data)
				elif category==ErrorCategory.INPUT:event_bus.publish(ERROR_EVENT_INPUT_PROCESSING,event_data)
				event_bus.publish(ERROR_EVENT_LOGGED,event_data)
		except Exception as e:logger.error(f"Failed to publish error event: {e}")
	return context
def with_error_handling(error_code:Optional[str]=None,error_category:ErrorCategory=ErrorCategory.UNKNOWN,error_severity:ErrorSeverity=ErrorSeverity.ERROR,publish_error:bool=True,include_state_info:bool=True):
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):return safe_execute(func,*args,error_code=error_code,error_category=error_category,error_severity=error_severity,publish_error=publish_error,include_state_info=include_state_info,**kwargs)
		return wrapper
	return decorator
def create_state_transition_error(from_state:Any,to_state:Any,trigger:str,details:Dict[str,Any]=None)->StateTransitionError:from_name=from_state.name if hasattr(from_state,'name')else str(from_state);to_name=to_state.name if hasattr(to_state,'name')else str(to_state);message=f"Invalid state transition: {from_name} -> {to_name} (trigger: {trigger})";record_error(message=message,category=ErrorCategory.STATE,severity=ErrorSeverity.ERROR,source='StateManager.transition_to',details=details or{},from_state=from_state,to_state=to_state,trigger=trigger);return StateTransitionError(message=message,from_state=from_state,to_state=to_state,trigger=trigger,details=details)