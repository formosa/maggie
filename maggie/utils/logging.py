import os,json,uuid,platform,logging,threading,traceback,sys,time
from pathlib import Path
from typing import Dict,Any,Optional,List,Union,Set
from enum import Enum
import psutil
from loguru import logger
class LogLevel(str,Enum):DEBUG='DEBUG';INFO='INFO';WARNING='WARNING';ERROR='ERROR';CRITICAL='CRITICAL'
class LogDestination(str,Enum):CONSOLE='console';FILE='file';EVENT_BUS='event_bus';TELEMETRY='telemetry'
class LoggingManager:
	def __init__(self,config:Dict[str,Any]):self.config=config.get('logging',{});self.log_dir=Path(self.config.get('path','logs')).resolve();self.log_dir.mkdir(exist_ok=True,parents=True);self.console_level=self.config.get('console_level','INFO');self.file_level=self.config.get('file_level','DEBUG');(self.enabled_destinations):Set[LogDestination]={LogDestination.CONSOLE,LogDestination.FILE};self._configure_logging();self._log_system_info()
	def _configure_logging(self)->None:
		logger.remove()
		if LogDestination.CONSOLE in self.enabled_destinations:logger.add(sys.stdout,level=self.console_level,format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',colorize=True,backtrace=True,diagnose=True)
		if LogDestination.FILE in self.enabled_destinations:log_file=self.log_dir/'maggie.log';logger.add(log_file,level=self.file_level,format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',rotation='10 MB',retention='1 week',compression='zip',backtrace=True,diagnose=True);error_log=self.log_dir/'errors.log';logger.add(error_log,level='ERROR',format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',rotation='5 MB',retention='1 month',compression='zip',backtrace=True,diagnose=True)
	def _log_system_info(self)->None:
		try:logger.info(f"System: {platform.system()} {platform.release()}");logger.info(f"Python: {platform.python_version()}");self._log_cpu_info();self._log_ram_info();self._log_gpu_info()
		except Exception as e:logger.warning(f"Error logging system information: {e}")
	def _log_cpu_info(self)->None:
		try:cpu_info=platform.processor();cpu_cores=psutil.cpu_count(logical=False);cpu_threads=psutil.cpu_count(logical=True);logger.info(f"CPU: {cpu_info}");logger.info(f"CPU Cores: {cpu_cores} physical, {cpu_threads} logical")
		except Exception as e:logger.warning(f"Error getting CPU information: {e}")
	def _log_ram_info(self)->None:
		try:memory=psutil.virtual_memory();logger.info(f"RAM: {memory.total/1024**3:.2f} GB (Available: {memory.available/1024**3:.2f} GB)")
		except Exception as e:logger.warning(f"Error getting RAM information: {e}")
	def _log_gpu_info(self)->None:
		try:
			import torch
			if torch.cuda.is_available():
				device_count=torch.cuda.device_count()
				for i in range(device_count):
					device_name=torch.cuda.get_device_name(i);memory_gb=torch.cuda.get_device_properties(i).total_memory/1024**3;logger.info(f"GPU {i}: {device_name} ({memory_gb:.2f} GB VRAM)")
					if'3080'in device_name:logger.info('RTX 3080 detected - applying optimal configurations')
			else:logger.warning('CUDA not available, GPU acceleration disabled')
		except ImportError:logger.warning('PyTorch not installed, GPU detection skipped')
		except Exception as e:logger.warning(f"Error detecting GPU: {e}")
	def add_event_bus_handler(self,event_bus:Any)->None:
		if LogDestination.EVENT_BUS not in self.enabled_destinations:
			self.enabled_destinations.add(LogDestination.EVENT_BUS)
			def handle_log(record):
				if record['level'].name in('ERROR','CRITICAL'):data={'message':record['message'],'level':record['level'].name,'time':record['time'].isoformat(),'name':record['name'],'function':record['function'],'line':record['line'],'module':record['module']};event_bus.publish('error_logged',data)
			logger.configure(handlers=[{'sink':handle_log,'level':'ERROR'}],extra=logger._core.extra)
	def setup_global_exception_handler(self)->None:
		def global_exception_handler(exc_type,exc_value,exc_traceback):
			if issubclass(exc_type,KeyboardInterrupt):sys.__excepthook__(exc_type,exc_value,exc_traceback);return
			logger.opt(exception=(exc_type,exc_value,exc_traceback)).critical('Unhandled exception:')
			try:
				from maggie.utils.service_locator import ServiceLocator;event_bus=ServiceLocator.get('event_bus')
				if event_bus:error_data={'type':str(exc_type.__name__),'message':str(exc_value),'traceback':''.join(traceback.format_tb(exc_traceback)),'is_unhandled':True};event_bus.publish('error_logged',error_data)
			except:pass
		sys.excepthook=global_exception_handler
	def get_logger(self,name:str)->logger:return logger.bind(name=name)
class TelemetryManager:
	def __init__(self,config_path:Optional[str]=None):self._instance_id=str(uuid.uuid4());self._config=self._load_telemetry_config(config_path);self._logger=logging.getLogger('MaggieTelemetry');self._setup_telemetry_logging()
	def _load_telemetry_config(self,config_path:Optional[str]=None)->Dict[str,Any]:
		default_config={'logging':{'level':'INFO','filepath':'logs/maggie_telemetry.log'},'telemetry':{'opt_in':False,'endpoint':'https://telemetry.maggieai.com/report','include_system_info':True,'include_error_reports':True,'include_performance_metrics':True}}
		if config_path and os.path.exists(config_path):
			try:
				with open(config_path,'r')as f:user_config=json.load(f);return{**default_config,**user_config}
			except Exception:self._logger.warning('Invalid telemetry config. Using defaults.')
		return default_config
	def _setup_telemetry_logging(self)->None:log_config=self._config.get('logging',{});log_level=getattr(logging,log_config.get('level','INFO').upper());log_filepath=log_config.get('filepath','logs/maggie_telemetry.log');os.makedirs(os.path.dirname(log_filepath),exist_ok=True);logging.basicConfig(level=log_level,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',handlers=[logging.FileHandler(log_filepath),logging.StreamHandler()])
	def is_telemetry_enabled(self)->bool:return self._config.get('telemetry',{}).get('opt_in',False)
	def capture_system_snapshot(self)->Dict[str,Any]:
		if not self.is_telemetry_enabled():return{}
		system_info={'instance_id':self._instance_id,'timestamp':time.time(),'system':{'os':platform.system(),'release':platform.release(),'architecture':platform.machine(),'python_version':platform.python_version()},'hardware':{'cpu':{'name':platform.processor(),'physical_cores':os.cpu_count()},'memory':{'total_gb':round(psutil.virtual_memory().total/1024**3,2)}}}
		try:import torch;system_info['hardware']['gpu']={'cuda_available':torch.cuda.is_available(),'device_name':torch.cuda.get_device_name(0)if torch.cuda.is_available()else None,'cuda_version':torch.version.cuda if torch.cuda.is_available()else None}
		except ImportError:system_info['hardware']['gpu']={'cuda_available':False}
		return system_info
	def log_installation_event(self,event_type:str,details:Dict[str,Any])->None:
		if not self.is_telemetry_enabled():return
		event_log={'event_type':event_type,'timestamp':time.time(),**details};self._logger.info(f"Event Logged: {json.dumps(event_log)}");self.submit_anonymous_telemetry(event_log)
	def submit_anonymous_telemetry(self,event_data:Dict[str,Any])->bool:
		if not self.is_telemetry_enabled():return False
		try:import requests;response=requests.post(self._config['telemetry']['endpoint'],json=event_data,timeout=5);return response.status_code==200
		except Exception as e:self._logger.error(f"Telemetry submission failed: {e}");return False
	def set_opt_in_status(self,opt_in:bool)->bool:
		try:
			self._config['telemetry']['opt_in']=opt_in;config_path=os.path.join('config','telemetry.json');os.makedirs(os.path.dirname(config_path),exist_ok=True)
			with open(config_path,'w')as f:json.dump(self._config,f,indent=2)
			self._logger.info(f"Telemetry opt-in status set to: {opt_in}");return True
		except Exception as e:self._logger.error(f"Failed to update telemetry opt-in status: {e}");return False