import os,sys,logging,time,uuid,inspect,functools,threading,queue
from enum import Enum,auto
from pathlib import Path
from typing import Dict,Any,Optional,List,Union,Set,Callable,Generator,TypeVar,cast
from contextlib import contextmanager
from loguru import logger

# Remove the direct import of HardwareDetector
# from maggie.utils.resource.detector import HardwareDetector

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
	def __init__(self,config:Dict[str,Any]):
		self.config=config.get('logging',{});self.log_dir=Path(self.config.get('path','logs')).resolve();self.log_dir.mkdir(exist_ok=True,parents=True);self.console_level=self.config.get('console_level','INFO');self.file_level=self.config.get('file_level','DEBUG');(self.enabled_destinations):Set[LogDestination]={LogDestination.CONSOLE,LogDestination.FILE};
		
		# Use lazy import for HardwareDetector to avoid circular reference
		self._hardware_detector = None
		
		self.log_batch_size=self.config.get('batch_size',50);self.log_batch_timeout=self.config.get('batch_timeout',5.);self.log_batch=[];self.log_batch_lock=threading.RLock();self.log_batch_timer=None;self.log_batch_enabled=self.config.get('batch_enabled',True);self.async_logging=self.config.get('async_enabled',True);self.log_queue=queue.Queue()if self.async_logging else None;self.log_worker=None;self._configure_logging();self._log_system_info()
		if self.log_batch_enabled:self._initialize_log_batching()
		if self.async_logging:self._initialize_async_logging()
		self.correlation_id=None
	
	def _get_hardware_detector(self):
		# Lazy import of HardwareDetector
		if self._hardware_detector is None:
			try:
				from maggie.utils.resource.detector import HardwareDetector
				self._hardware_detector = HardwareDetector()
			except ImportError:
				logger.warning("Failed to import HardwareDetector, system info may be limited")
				self._hardware_detector = None
		return self._hardware_detector
		
	def _configure_logging(self)->None:
		logger.remove()
		if LogDestination.CONSOLE in self.enabled_destinations:logger.add(sys.stdout,level=self.console_level,format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',colorize=True,backtrace=True,diagnose=True)
		if LogDestination.FILE in self.enabled_destinations:log_file=self.log_dir/'maggie.log';logger.add(log_file,level=self.file_level,format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',rotation='10 MB',retention='1 week',compression='zip',backtrace=True,diagnose=True);error_log=self.log_dir/'errors.log';logger.add(error_log,level='ERROR',format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',rotation='5 MB',retention='1 month',compression='zip',backtrace=True,diagnose=True);perf_log=self.log_dir/'performance.log';logger.add(perf_log,level='DEBUG',format='{time:YYYY-MM-DD HH:mm:ss} | PERFORMANCE | {extra[component]}:{extra[operation]} | {message}',filter=lambda record:'performance'in record['extra'],rotation='5 MB',retention='1 week',compression='zip');fsm_log=self.log_dir/'fsm.log';logger.add(fsm_log,level='DEBUG',format='{time:YYYY-MM-DD HH:mm:ss} | FSM | {extra[component]} | {message}',filter=lambda record:'fsm'in record['extra'],rotation='5 MB',retention='1 week',compression='zip');resource_log=self.log_dir/'resources.log';logger.add(resource_log,level='DEBUG',format='{time:YYYY-MM-DD HH:mm:ss} | RESOURCE | {extra[component]}:{extra[resource_type]} | {message}',filter=lambda record:'resource_operation'in record['extra'],rotation='5 MB',retention='1 week',compression='zip');input_log=self.log_dir/'input.log';logger.add(input_log,level='DEBUG',format='{time:YYYY-MM-DD HH:mm:ss} | INPUT | {extra[component]}:{extra[input_type]} | {message}',filter=lambda record:'input_operation'in record['extra'],rotation='5 MB',retention='1 week',compression='zip')
	def _initialize_log_batching(self)->None:self.log_batch_timer=threading.Timer(self.log_batch_timeout,self._flush_log_batch);self.log_batch_timer.daemon=True;self.log_batch_timer.start()
	def _initialize_async_logging(self)->None:self.log_worker=threading.Thread(target=self._process_log_queue,name='AsyncLogWorker',daemon=True);self.log_worker.start()
	def _process_log_queue(self)->None:
		while True:
			try:
				log_record=self.log_queue.get(timeout=1.)
				if log_record is None:break
				level,message,args,kwargs=log_record
				if level==LogLevel.DEBUG:logger.debug(message,*args,**kwargs)
				elif level==LogLevel.INFO:logger.info(message,*args,**kwargs)
				elif level==LogLevel.WARNING:logger.warning(message,*args,**kwargs)
				elif level==LogLevel.ERROR:logger.error(message,*args,**kwargs)
				elif level==LogLevel.CRITICAL:logger.critical(message,*args,**kwargs)
				self.log_queue.task_done()
			except queue.Empty:continue
			except Exception as e:sys.stderr.write(f"Error in async log worker: {e}\n")
	def _flush_log_batch(self)->None:
		with self.log_batch_lock:
			if not self.log_batch:return
			for log_record in self.log_batch:
				level,message,args,kwargs=log_record
				if level==LogLevel.DEBUG:logger.debug(message,*args,**kwargs)
				elif level==LogLevel.INFO:logger.info(message,*args,**kwargs)
				elif level==LogLevel.WARNING:logger.warning(message,*args,**kwargs)
				elif level==LogLevel.ERROR:logger.error(message,*args,**kwargs)
				elif level==LogLevel.CRITICAL:logger.critical(message,*args,**kwargs)
			self.log_batch.clear();self.log_batch_timer=threading.Timer(self.log_batch_timeout,self._flush_log_batch);self.log_batch_timer.daemon=True;self.log_batch_timer.start()
	def _log_system_info(self)->None:
		try:
			# Use the lazy-loaded hardware detector
			hardware_detector = self._get_hardware_detector()
			if hardware_detector:
				system_info = hardware_detector.detect_system()
				logger.info(f"System: {system_info['os']['system']} {system_info['os']['release']}")
				cpu_info = system_info['cpu']
				logger.info(f"CPU: {cpu_info['model']}")
				logger.info(f"CPU Cores: {cpu_info['physical_cores']} physical, {cpu_info['logical_cores']} logical")
				memory_info = system_info['memory']
				logger.info(f"RAM: {memory_info['total_gb']:.2f} GB (Available: {memory_info['available_gb']:.2f} GB)")
				gpu_info = system_info['gpu']
				if gpu_info.get('available', False):
					logger.info(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.2f} GB VRAM)")
					logger.info(f"CUDA available: {gpu_info['cuda_version']}")
					if gpu_info.get('is_rtx_3080', False):
						logger.info('RTX 3080 detected - applying optimal configurations')
				else:
					logger.warning('CUDA not available, GPU acceleration disabled')
			else:
				logger.warning("Hardware detector not available, skipping detailed system info")
				logger.info(f"System: {sys.platform}")
		except Exception as e:
			logger.warning(f"Error logging system information: {e}")
	def log(self,level:LogLevel,message:str,*args,**kwargs)->None:
		if self.async_logging and self.log_queue is not None:self.log_queue.put((level,message,args,kwargs))
		elif self.log_batch_enabled:
			with self.log_batch_lock:
				self.log_batch.append((level,message,args,kwargs))
				if len(self.log_batch)>=self.log_batch_size:self._flush_log_batch()
		elif level==LogLevel.DEBUG:logger.debug(message,*args,**kwargs)
		elif level==LogLevel.INFO:logger.info(message,*args,**kwargs)
		elif level==LogLevel.WARNING:logger.warning(message,*args,**kwargs)
		elif level==LogLevel.ERROR:logger.error(message,*args,**kwargs)
		elif level==LogLevel.CRITICAL:logger.critical(message,*args,**kwargs)
	def set_correlation_id(self,correlation_id:str)->None:self.correlation_id=correlation_id;logger.configure(extra={'correlation_id':correlation_id})
	def get_correlation_id(self)->Optional[str]:return self.correlation_id
	def clear_correlation_id(self)->None:self.correlation_id=None;logger.configure(extra={'correlation_id':None})
	def add_event_bus_handler(self,event_bus:Any)->None:
		if LogDestination.EVENT_BUS not in self.enabled_destinations:
			self.enabled_destinations.add(LogDestination.EVENT_BUS)
			def handle_log(record):
				if record['level'].name in('ERROR','CRITICAL'):
					data={'message':record['message'],'level':record['level'].name,'time':record['time'].isoformat(),'name':record['name'],'function':record['function'],'line':record['line'],'module':record['module'],'correlation_id':record.get('extra',{}).get('correlation_id')}
					try:
						from maggie.service.locator import ServiceLocator;state_manager=ServiceLocator.get('state_manager')
						if state_manager:current_state=state_manager.get_current_state();data['state']=current_state.name;data['state_style']=current_state.get_style()
					except Exception:pass
					event_bus.publish('error_logged',data)
			logger.configure(handlers=[{'sink':handle_log,'level':'ERROR'}],extra=logger._core.extra)
	def setup_global_exception_handler(self)->None:
		import traceback
		def global_exception_handler(exc_type,exc_value,exc_traceback):
			if issubclass(exc_type,KeyboardInterrupt):sys.__excepthook__(exc_type,exc_value,exc_traceback);return
			logger.opt(exception=(exc_type,exc_value,exc_traceback)).critical('Unhandled exception:')
			try:
				from maggie.service.locator import ServiceLocator;event_bus=ServiceLocator.get('event_bus')
				if event_bus:
					state_manager=ServiceLocator.get('state_manager');state_info={}
					if state_manager:current_state=state_manager.get_current_state();state_info={'current_state':current_state.name,'style':current_state.get_style()}
					error_data={'type':str(exc_type.__name__),'message':str(exc_value),'traceback':''.join(traceback.format_tb(exc_traceback)),'is_unhandled':True,'correlation_id':self.correlation_id,'state_info':state_info};event_bus.publish('error_logged',error_data)
			except Exception:pass
		sys.excepthook=global_exception_handler
	def get_logger(self,name:str)->logger:return logger.bind(name=name)
	def log_performance(self,component:str,operation:str,elapsed:float,details:Dict[str,Any]=None)->None:
		log_entry=f"{operation} took {elapsed:.3f}s"
		if details:detail_str=', '.join(f"{k}={v}"for(k,v)in details.items());log_entry+=f" ({detail_str})"
		perf_logger=logger.bind(performance=True,component=component,operation=operation);perf_logger.debug(log_entry)
		if elapsed>1.:logger.debug(f"Performance: {component}/{operation} - {log_entry}")
	def log_state_transition(self,from_state:Any,to_state:Any,trigger:str)->None:from_name=from_state.name if hasattr(from_state,'name')else str(from_state);to_name=to_state.name if hasattr(to_state,'name')else str(to_state);fsm_logger=logger.bind(fsm=True,component='StateManager');fsm_logger.info(f"State transition: {from_name} -> {to_name} (trigger: {trigger})");logger.info(f"State transition: {from_name} -> {to_name} (trigger: {trigger})")
	def log_resource_allocation(self,resource_type:str,resource_name:str,state:Any,details:Dict[str,Any]=None)->None:
		state_name=state.name if hasattr(state,'name')else str(state);log_msg=f"Allocated {resource_type}/{resource_name} in state {state_name}"
		if details:detail_str=', '.join(f"{k}={v}"for(k,v)in details.items());log_msg+=f" ({detail_str})"
		resource_logger=logger.bind(resource_operation=True,component='ResourceManager',resource_type=resource_type);resource_logger.info(log_msg)
	def log_resource_deallocation(self,resource_type:str,resource_name:str,state:Any,details:Dict[str,Any]=None)->None:
		state_name=state.name if hasattr(state,'name')else str(state);log_msg=f"Deallocated {resource_type}/{resource_name} in state {state_name}"
		if details:detail_str=', '.join(f"{k}={v}"for(k,v)in details.items());log_msg+=f" ({detail_str})"
		resource_logger=logger.bind(resource_operation=True,component='ResourceManager',resource_type=resource_type);resource_logger.info(log_msg)
	def log_input_processing(self,input_type:str,state:Any,details:Dict[str,Any]=None)->None:
		state_name=state.name if hasattr(state,'name')else str(state);log_msg=f"Processing {input_type} in state {state_name}"
		if details:detail_str=', '.join(f"{k}={v}"for(k,v)in details.items());log_msg+=f" ({detail_str})"
		input_logger=logger.bind(input_operation=True,component='InputProcessor',input_type=input_type);input_logger.info(log_msg)
	def shutdown(self)->None:
		if self.log_batch_enabled:
			self._flush_log_batch()
			if self.log_batch_timer:self.log_batch_timer.cancel()
		if self.async_logging and self.log_queue:
			self.log_queue.put(None)
			if self.log_worker:self.log_worker.join(timeout=2.)
		logger.info('Logging system shutdown')

@contextmanager
def logging_context(correlation_id:Optional[str]=None,component:str='',operation:str='',state:Any=None)->Generator[Dict[str,Any],None,None]:
	ctx_id=correlation_id or str(uuid.uuid4());context={'correlation_id':ctx_id,'component':component,'operation':operation,'start_time':time.time()}
	if state is not None:context['state']=state.name if hasattr(state,'name')else str(state)
	if state is None:
		try:
			from maggie.service.locator import ServiceLocator;state_manager=ServiceLocator.get('state_manager')
			if state_manager:current_state=state_manager.get_current_state();context['state']=current_state.name if hasattr(current_state,'name')else str(current_state)
		except Exception:pass
	try:logging_mgr=LoggingManager.get_instance();prev_id=logging_mgr.get_correlation_id();logging_mgr.set_correlation_id(ctx_id)
	except RuntimeError:prev_id=None
	extra={'correlation_id':ctx_id,'component':component,'operation':operation}
	if'state'in context:extra['state']=context['state']
	with logger.contextualize(**extra):
		try:yield context
		except Exception as e:logger.error(f"Error in {component}/{operation}: {e}");raise
		finally:
			try:
				if prev_id is not None:logging_mgr=LoggingManager.get_instance();logging_mgr.set_correlation_id(prev_id)
			except RuntimeError:pass
			if component and operation:
				elapsed=time.time()-context['start_time'];logger.debug(f"{component}/{operation} completed in {elapsed:.3f}s")
				if elapsed>.1:
					try:logging_mgr=LoggingManager.get_instance();logging_mgr.log_performance(component,operation,elapsed)
					except RuntimeError:pass

def log_operation(component:str='',log_args:bool=True,log_result:bool=False,include_state:bool=True):
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
			state=None
			if include_state:
				try:
					from maggie.service.locator import ServiceLocator;state_manager=ServiceLocator.get('state_manager')
					if state_manager:state=state_manager.get_current_state()
				except Exception:pass
			with logging_context(component=component,operation=operation,state=state)as ctx:
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
	def __init__(self,component_name:str):
		self.component=component_name;self.logger=logger.bind(component=component_name);self._state_manager=None;self._resource_manager=None
		try:from maggie.service.locator import ServiceLocator;self._state_manager=ServiceLocator.get('state_manager');self._resource_manager=ServiceLocator.get('resource_manager')
		except Exception:pass
	def _get_state_context(self)->Dict[str,Any]:
		context={}
		if self._state_manager:
			try:current_state=self._state_manager.get_current_state();context['state']=current_state.name
			except Exception:pass
		return context
	def _get_resource_context(self)->Dict[str,Any]:
		context={}
		if self._resource_manager:
			try:
				resource_status=self._resource_manager.get_resource_status()
				if'memory'in resource_status:context['memory_percent']=resource_status['memory'].get('used_percent',0)
				if'gpu'in resource_status and resource_status['gpu'].get('available',False):context['gpu_percent']=resource_status['gpu'].get('memory_percent',0)
			except Exception:pass
		return context
	def debug(self,message:str,**kwargs)->None:context=self._get_state_context();self.logger.bind(**context).debug(message,**kwargs)
	def info(self,message:str,**kwargs)->None:context=self._get_state_context();self.logger.bind(**context).info(message,**kwargs)
	def warning(self,message:str,**kwargs)->None:context=self._get_state_context();self.logger.bind(**context).warning(message,**kwargs)
	def error(self,message:str,exception:Optional[Exception]=None,**kwargs)->None:
		context=self._get_state_context()
		if exception:self.logger.bind(**context).error(f"{message}: {exception}",exc_info=exception,**kwargs)
		else:self.logger.bind(**context).error(message,**kwargs)
	def critical(self,message:str,exception:Optional[Exception]=None,**kwargs)->None:
		context=self._get_state_context();resource_context=self._get_resource_context();context.update(resource_context)
		if exception:self.logger.bind(**context).critical(f"{message}: {exception}",exc_info=exception,**kwargs)
		else:self.logger.bind(**context).critical(message,**kwargs)
	def log_state_change(self,old_state:Any,new_state:Any,trigger:str)->None:
		old_name=old_state.name if hasattr(old_state,'name')else str(old_state);new_name=new_state.name if hasattr(new_state,'name')else str(new_state);context={'fsm':True,'from_state':old_name,'to_state':new_name,'trigger':trigger};self.logger.bind(**context).info(f"State change: {old_name} -> {new_name} (trigger: {trigger})")
		try:logging_mgr=LoggingManager.get_instance();logging_mgr.log_state_transition(old_state,new_state,trigger)
		except RuntimeError:pass
	def log_performance(self,operation:str,elapsed:float,details:Dict[str,Any]=None)->None:
		message=f"Performance: {operation} took {elapsed:.3f}s"
		if details:message+=f" ({', '.join(f'{k}={v}'for(k,v)in details.items())})"
		self.logger.debug(message)
		try:logging_mgr=LoggingManager.get_instance();logging_mgr.log_performance(self.component,operation,elapsed,details)
		except RuntimeError:pass