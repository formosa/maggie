import os,sys,logging,time,uuid,inspect,functools
from enum import Enum,auto
from pathlib import Path
from typing import Dict,Any,Optional,List,Union,Set,Callable,Generator,TypeVar,cast
from contextlib import contextmanager
from loguru import logger
from maggie.utils.resource.detector import HardwareDetector
T=TypeVar('T')
class LogLevel(str,Enum):DEBUG='DEBUG';INFO='INFO';WARNING='WARNING';ERROR='ERROR';CRITICAL='CRITICAL'
class LogDestination(str,Enum):CONSOLE='console';FILE='file';EVENT_BUS='event_bus'
class LoggingManager:
	_instance=None
	@classmethod
	def get_instance(cls)->'LoggingManager':
		if cls._instance is None:raise RuntimeError('LoggingManager not initialized')
		return cls._instance
	@classmethod
	def initialize(cls,config:Dict[str,Any])->'LoggingManager':
		if cls._instance is not None:logger.warning('LoggingManager already initialized');return cls._instance
		cls._instance=LoggingManager(config);return cls._instance
	def __init__(self,config:Dict[str,Any]):self.config=config.get('logging',{});self.log_dir=Path(self.config.get('path','logs')).resolve();self.log_dir.mkdir(exist_ok=True,parents=True);self.console_level=self.config.get('console_level','INFO');self.file_level=self.config.get('file_level','DEBUG');(self.enabled_destinations):Set[LogDestination]={LogDestination.CONSOLE,LogDestination.FILE};self._hardware_detector=HardwareDetector();self._configure_logging();self._log_system_info();self.correlation_id=None
	def _configure_logging(self)->None:
		logger.remove()
		if LogDestination.CONSOLE in self.enabled_destinations:logger.add(sys.stdout,level=self.console_level,format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',colorize=True,backtrace=True,diagnose=True)
		if LogDestination.FILE in self.enabled_destinations:log_file=self.log_dir/'maggie.log';logger.add(log_file,level=self.file_level,format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',rotation='10 MB',retention='1 week',compression='zip',backtrace=True,diagnose=True);error_log=self.log_dir/'errors.log';logger.add(error_log,level='ERROR',format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',rotation='5 MB',retention='1 month',compression='zip',backtrace=True,diagnose=True);perf_log=self.log_dir/'performance.log';logger.add(perf_log,level='DEBUG',format='{time:YYYY-MM-DD HH:mm:ss} | PERFORMANCE | {extra[component]}:{extra[operation]} | {message}',filter=lambda record:'performance'in record['extra'],rotation='5 MB',retention='1 week',compression='zip')
	def _log_system_info(self)->None:
		try:
			system_info=self._hardware_detector.detect_system();logger.info(f"System: {system_info['os']['system']} {system_info['os']['release']}");cpu_info=system_info['cpu'];logger.info(f"CPU: {cpu_info['model']}");logger.info(f"CPU Cores: {cpu_info['physical_cores']} physical, {cpu_info['logical_cores']} logical");memory_info=system_info['memory'];logger.info(f"RAM: {memory_info['total_gb']:.2f} GB (Available: {memory_info['available_gb']:.2f} GB)");gpu_info=system_info['gpu']
			if gpu_info.get('available',False):
				logger.info(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.2f} GB VRAM)");logger.info(f"CUDA available: {gpu_info['cuda_version']}")
				if gpu_info.get('is_rtx_3080',False):logger.info('RTX 3080 detected - applying optimal configurations')
			else:logger.warning('CUDA not available, GPU acceleration disabled')
		except Exception as e:logger.warning(f"Error logging system information: {e}")
	def set_correlation_id(self,correlation_id:str)->None:self.correlation_id=correlation_id;logger.configure(extra={'correlation_id':correlation_id})
	def get_correlation_id(self)->Optional[str]:return self.correlation_id
	def clear_correlation_id(self)->None:self.correlation_id=None;logger.configure(extra={'correlation_id':None})
	def add_event_bus_handler(self,event_bus:Any)->None:
		if LogDestination.EVENT_BUS not in self.enabled_destinations:
			self.enabled_destinations.add(LogDestination.EVENT_BUS)
			def handle_log(record):
				if record['level'].name in('ERROR','CRITICAL'):data={'message':record['message'],'level':record['level'].name,'time':record['time'].isoformat(),'name':record['name'],'function':record['function'],'line':record['line'],'module':record['module'],'correlation_id':record.get('extra',{}).get('correlation_id')};event_bus.publish('error_logged',data)
			logger.configure(handlers=[{'sink':handle_log,'level':'ERROR'}],extra=logger._core.extra)
	def setup_global_exception_handler(self)->None:
		import traceback
		def global_exception_handler(exc_type,exc_value,exc_traceback):
			if issubclass(exc_type,KeyboardInterrupt):sys.__excepthook__(exc_type,exc_value,exc_traceback);return
			logger.opt(exception=(exc_type,exc_value,exc_traceback)).critical('Unhandled exception:')
			try:
				from maggie.service.locator import ServiceLocator;event_bus=ServiceLocator.get('event_bus')
				if event_bus:error_data={'type':str(exc_type.__name__),'message':str(exc_value),'traceback':''.join(traceback.format_tb(exc_traceback)),'is_unhandled':True,'correlation_id':self.correlation_id};event_bus.publish('error_logged',error_data)
			except Exception:pass
		sys.excepthook=global_exception_handler
	def get_logger(self,name:str)->logger:return logger.bind(name=name)
	def log_performance(self,component:str,operation:str,elapsed:float,details:Dict[str,Any]=None)->None:
		log_entry=f"{operation} took {elapsed:.3f}s"
		if details:detail_str=', '.join(f"{k}={v}"for(k,v)in details.items());log_entry+=f" ({detail_str})"
		perf_logger=logger.bind(performance=True,component=component,operation=operation);perf_logger.debug(log_entry)
		if elapsed>1.:logger.debug(f"Performance: {component}/{operation} - {log_entry}")
@contextmanager
def logging_context(correlation_id:Optional[str]=None,component:str='',operation:str='')->Generator[Dict[str,Any],None,None]:
	ctx_id=correlation_id or str(uuid.uuid4());context={'correlation_id':ctx_id,'component':component,'operation':operation,'start_time':time.time()}
	try:logging_mgr=LoggingManager.get_instance();prev_id=logging_mgr.get_correlation_id();logging_mgr.set_correlation_id(ctx_id)
	except RuntimeError:prev_id=None
	with logger.contextualize(correlation_id=ctx_id,component=component,operation=operation):
		try:yield context
		except Exception as e:logger.error(f"Error in {component}/{operation}: {e}");raise
		finally:
			try:
				if prev_id is not None:logging_mgr=LoggingManager.get_instance();logging_mgr.set_correlation_id(prev_id)
			except RuntimeError:pass
			if component and operation:elapsed=time.time()-context['start_time'];logger.debug(f"{component}/{operation} completed in {elapsed:.3f}s")
def log_operation(component:str='',log_args:bool=True,log_result:bool=False):
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
			with logging_context(component=component,operation=operation)as ctx:
				if log_args and args_str:logger.debug(f"{operation} called with args: {args_str}")
				start_time=time.time();result=func(*args,**kwargs);elapsed=time.time()-start_time
				if log_result:
					if isinstance(result,(str,int,float,bool,type(None))):logger.debug(f"{operation} returned: {result}")
					else:logger.debug(f"{operation} returned: {type(result).__name__}")
				try:logging_mgr=LoggingManager.get_instance();logging_mgr.log_performance(component or func.__module__,operation,elapsed)
				except RuntimeError:pass
				return result
		return wrapper
	return decorator
class ComponentLogger:
	def __init__(self,component_name:str):self.component=component_name;self.logger=logger.bind(component=component_name)
	def debug(self,message:str,**kwargs)->None:self.logger.debug(message,**kwargs)
	def info(self,message:str,**kwargs)->None:self.logger.info(message,**kwargs)
	def warning(self,message:str,**kwargs)->None:self.logger.warning(message,**kwargs)
	def error(self,message:str,exception:Optional[Exception]=None,**kwargs)->None:
		if exception:self.logger.error(f"{message}: {exception}",exc_info=exception,**kwargs)
		else:self.logger.error(message,**kwargs)
	def critical(self,message:str,exception:Optional[Exception]=None,**kwargs)->None:
		if exception:self.logger.critical(f"{message}: {exception}",exc_info=exception,**kwargs)
		else:self.logger.critical(message,**kwargs)
	def log_state_change(self,old_state:Any,new_state:Any,trigger:str)->None:self.logger.info(f"State change: {old_state} -> {new_state} (trigger: {trigger})")
	def log_performance(self,operation:str,elapsed:float,details:Dict[str,Any]=None)->None:
		try:logging_mgr=LoggingManager.get_instance();logging_mgr.log_performance(self.component,operation,elapsed,details)
		except RuntimeError:
			message=f"Performance: {operation} took {elapsed:.3f}s"
			if details:message+=f" ({', '.join(f'{k}={v}'for(k,v)in details.items())})"
			self.logger.debug(message)