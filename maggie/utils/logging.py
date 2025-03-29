import os,sys,logging,time,uuid,inspect,functools,threading,queue,traceback
from enum import Enum,auto
from pathlib import Path
from typing import Dict,Any,Optional,List,Union,Set,Callable,Generator,TypeVar,cast
from contextlib import contextmanager
T=TypeVar('T')
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger('maggie.error_handling')
class LogLevel(Enum):DEBUG=auto();INFO=auto();WARNING=auto();ERROR=auto();CRITICAL=auto()
class LogDestination(Enum):CONSOLE='CONSOLE';FILE='FILE';EVENT_BUS='EVENT_BUS'
class LoggingManager:
	_instance=None;_lock=threading.RLock()
	@classmethod
	def get_instance(cls)->'LoggingManager':
		if cls._instance is None:raise RuntimeError('LoggingManager not initialized')
		return cls._instance
	@classmethod
	def initialize(cls,config:Dict[str,Any])->'LoggingManager':
		if cls._instance is not None:logger.warning('LoggingManager already initialized');return cls._instance
		with cls._lock:
			if cls._instance is None:cls._instance=LoggingManager(config)
		return cls._instance
	def __init__(self,config:Dict[str,Any])->None:self.config=config.get('logging',{});self.log_dir=Path(self.config.get('path','logs')).resolve();self.log_dir.mkdir(exist_ok=True,parents=True);self.console_level=self.config.get('console_level','INFO');self.file_level=self.config.get('file_level','DEBUG');self._enhanced_logging=False;self._event_publisher=None;self._error_handler=None;self._state_provider=None;self.log_batch_size=self.config.get('batch_size',50);self.log_batch_timeout=self.config.get('batch_timeout',5.);self.async_logging=self.config.get('async_enabled',True);self.correlation_id=None;self._configure_basic_logging()
	def _configure_basic_logging(self)->None:console_handler=logging.StreamHandler(sys.stdout);console_handler.setLevel(getattr(logging,self.console_level));console_formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s');console_handler.setFormatter(console_formatter);log_file=self.log_dir/'maggie.log';file_handler=logging.FileHandler(log_file);file_handler.setLevel(getattr(logging,self.file_level));file_formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s');file_handler.setFormatter(file_formatter);root_logger=logging.getLogger();root_logger.handlers.clear();root_logger.addHandler(console_handler);root_logger.addHandler(file_handler);root_logger.setLevel(logging.DEBUG)
	def enhance_with_event_publisher(self,event_publisher:Any)->None:self._event_publisher=event_publisher;self._enhanced_logging=True
	def enhance_with_error_handler(self,error_handler:Any)->None:self._error_handler=error_handler;self._enhanced_logging=True
	def enhance_with_state_provider(self,state_provider:Any)->None:self._state_provider=state_provider;self._enhanced_logging=True
	def log(self,level:LogLevel,message:str,*args:Any,exception:Optional[Exception]=None,**kwargs:Any)->None:
		log_method={LogLevel.DEBUG:logger.debug,LogLevel.INFO:logger.info,LogLevel.WARNING:logger.warning,LogLevel.ERROR:logger.error,LogLevel.CRITICAL:logger.critical}.get(level)
		if log_method:log_method(message,*args,exc_info=exception,**kwargs)
		if self._enhanced_logging and self._event_publisher and level in[LogLevel.ERROR,LogLevel.CRITICAL]:
			try:
				event_data={'message':message,'level':level.name,'timestamp':time.time(),'correlation_id':self.correlation_id}
				if self._state_provider:
					try:
						current_state=self._state_provider.get_current_state()
						if hasattr(current_state,'name'):event_data['state']=current_state.name
					except Exception:pass
				if exception:event_data['exception']=str(exception)
				self._event_publisher.publish('error_logged',event_data)
			except Exception as e:logger.warning(f"Failed to publish error event: {e}")
	def set_correlation_id(self,correlation_id:str)->None:self.correlation_id=correlation_id
	def get_correlation_id(self)->Optional[str]:return self.correlation_id
	def clear_correlation_id(self)->None:self.correlation_id=None
	def log_performance(self,component:str,operation:str,elapsed:float,details:Optional[Dict[str,Any]]=None)->None:
		log_entry=f"{operation} took {elapsed:.3f}s"
		if details:detail_str=', '.join(f"{k}={v}"for(k,v)in details.items());log_entry+=f" ({detail_str})"
		logger.debug(f"Performance: {component}/{operation} - {log_entry}")
		if self._enhanced_logging and self._event_publisher:
			try:event_data={'component':component,'operation':operation,'elapsed_time':elapsed,'details':details or{},'timestamp':time.time(),'correlation_id':self.correlation_id};self._event_publisher.publish('performance_metric',event_data)
			except Exception as e:logger.warning(f"Failed to publish performance metric: {e}")
	def log_state_transition(self,from_state:Any,to_state:Any,trigger:str)->None:
		from_name=from_state.name if hasattr(from_state,'name')else str(from_state);to_name=to_state.name if hasattr(to_state,'name')else str(to_state);logger.info(f"State transition: {from_name} -> {to_name} (trigger: {trigger})")
		if self._enhanced_logging and self._event_publisher:
			try:event_data={'from_state':from_name,'to_state':to_name,'trigger':trigger,'timestamp':time.time(),'correlation_id':self.correlation_id};self._event_publisher.publish('state_transition_logged',event_data)
			except Exception as e:logger.warning(f"Failed to publish state transition: {e}")
	def setup_global_exception_handler(self)->None:
		def global_exception_handler(exc_type,exc_value,exc_traceback):
			if issubclass(exc_type,KeyboardInterrupt):sys.__excepthook__(exc_type,exc_value,exc_traceback);return
			logger.critical('Unhandled exception:',exc_info=(exc_type,exc_value,exc_traceback))
			if self._enhanced_logging and self._event_publisher:
				try:event_data={'type':str(exc_type.__name__),'message':str(exc_value),'traceback':''.join(traceback.format_tb(exc_traceback)),'is_unhandled':True,'timestamp':time.time(),'correlation_id':self.correlation_id};self._event_publisher.publish('unhandled_exception',event_data)
				except Exception:pass
		sys.excepthook=global_exception_handler
class ComponentLogger:
	def __init__(self,component_name:str)->None:self.component=component_name;self.logger=logging.getLogger(component_name)
	def debug(self,message:str,**kwargs:Any)->None:
		exception=kwargs.pop('exception',None)
		if exception:self.logger.debug(message,exc_info=exception,**kwargs)
		else:self.logger.debug(message,**kwargs)
		try:manager=LoggingManager.get_instance();manager.log(LogLevel.DEBUG,message,exception=exception,**kwargs)
		except Exception:pass
	def info(self,message:str,**kwargs:Any)->None:
		exception=kwargs.pop('exception',None)
		if exception:self.logger.info(message,exc_info=exception,**kwargs)
		else:self.logger.info(message,**kwargs)
		try:manager=LoggingManager.get_instance();manager.log(LogLevel.INFO,message,exception=exception,**kwargs)
		except Exception:pass
	def warning(self,message:str,**kwargs:Any)->None:
		exception=kwargs.pop('exception',None)
		if exception:self.logger.warning(message,exc_info=exception,**kwargs)
		else:self.logger.warning(message,**kwargs)
		try:manager=LoggingManager.get_instance();manager.log(LogLevel.WARNING,message,exception=exception,**kwargs)
		except Exception:pass
	def error(self,message:str,exception:Optional[Exception]=None,**kwargs:Any)->None:
		if exception:self.logger.error(message,exc_info=exception,**kwargs)
		else:self.logger.error(message,**kwargs)
		try:manager=LoggingManager.get_instance();manager.log(LogLevel.ERROR,message,exception=exception,**kwargs)
		except Exception:pass
	def critical(self,message:str,exception:Optional[Exception]=None,**kwargs:Any)->None:
		if exception:self.logger.critical(message,exc_info=exception,**kwargs)
		else:self.logger.critical(message,**kwargs)
		try:manager=LoggingManager.get_instance();manager.log(LogLevel.CRITICAL,message,exception=exception,**kwargs)
		except Exception:pass
	def log_state_change(self,old_state:Any,new_state:Any,trigger:str)->None:
		old_name=old_state.name if hasattr(old_state,'name')else str(old_state);new_name=new_state.name if hasattr(new_state,'name')else str(new_state);self.info(f"State change: {old_name} -> {new_name} (trigger: {trigger})")
		try:manager=LoggingManager.get_instance();manager.log_state_transition(old_state,new_state,trigger)
		except Exception:pass
	def log_performance(self,operation:str,elapsed:float,details:Optional[Dict[str,Any]]=None)->None:
		message=f"Performance: {operation} took {elapsed:.3f}s"
		if details:message+=f" ({', '.join(f'{k}={v}'for(k,v)in details.items())})"
		self.debug(message)
		try:manager=LoggingManager.get_instance();manager.log_performance(self.component,operation,elapsed,details)
		except Exception:pass
@contextmanager
def logging_context(correlation_id:Optional[str]=None,component:str='',operation:str='',state:Any=None)->Generator[Dict[str,Any],None,None]:
	ctx_id=correlation_id or str(uuid.uuid4());context={'correlation_id':ctx_id,'component':component,'operation':operation,'start_time':time.time()}
	if state is not None:context['state']=state.name if hasattr(state,'name')else str(state)
	try:manager=LoggingManager.get_instance();old_correlation_id=manager.get_correlation_id();manager.set_correlation_id(ctx_id)
	except Exception:old_correlation_id=None
	logger_instance=logging.getLogger(component or'context')
	try:yield context
	except Exception as e:logger_instance.error(f"Error in {component}/{operation}: {e}",exc_info=True);raise
	finally:
		elapsed=time.time()-context['start_time'];logger_instance.debug(f"{component}/{operation} completed in {elapsed:.3f}s")
		try:
			manager=LoggingManager.get_instance()
			if old_correlation_id is not None:manager.set_correlation_id(old_correlation_id)
			if component and operation:manager.log_performance(component,operation,elapsed)
		except Exception:pass
def log_operation(component:str='',log_args:bool=True,log_result:bool=False,include_state:bool=True)->Callable[[Callable[...,T]],Callable[...,T]]:
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
			operation=func.__name__;args_str=''
			if log_args:
				sig=inspect.signature(func);arg_names=list(sig.parameters.keys());pos_args=[]
				for(i,arg)in enumerate(args):
					if i<len(arg_names)and i>0:pos_args.append(f"{arg_names[i]}={repr(arg)}")
					elif i>=len(arg_names):pos_args.append(repr(arg))
				kw_args=[f"{k}={repr(v)}"for(k,v)in kwargs.items()];all_args=pos_args+kw_args;args_str=', '.join(all_args)
				if len(args_str)>200:args_str=args_str[:197]+'...'
			state=None;logger_instance=logging.getLogger(component or func.__module__)
			if log_args and args_str:logger_instance.debug(f"{operation} called with args: {args_str}")
			with logging_context(component=component,operation=operation,state=state)as ctx:
				start_time=time.time();result=func(*args,**kwargs);elapsed=time.time()-start_time
				if log_result:
					if isinstance(result,(str,int,float,bool,type(None))):logger_instance.debug(f"{operation} returned: {result}")
					else:logger_instance.debug(f"{operation} returned: {type(result).__name__}")
				try:manager=LoggingManager.get_instance();manager.log_performance(component or func.__module__,operation,elapsed)
				except Exception:logger_instance.debug(f"{operation} completed in {elapsed:.3f}s")
				return result
		return wrapper
	return decorator
if __name__=='__main__':config={'logging':{'path':'logs','console_level':'INFO','file_level':'DEBUG'}};logging_manager=LoggingManager.initialize(config);logger=ComponentLogger('TestComponent');logger.info('Testing logging system')