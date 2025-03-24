import os,sys,logging
from enum import Enum,auto
from pathlib import Path
from typing import Dict,Any,Optional,List,Union,Set,Callable
from loguru import logger
from maggie.utils.resource.detector import HardwareDetector
class LogLevel(str,Enum):DEBUG='DEBUG';INFO='INFO';WARNING='WARNING';ERROR='ERROR';CRITICAL='CRITICAL'
class LogDestination(str,Enum):CONSOLE='console';FILE='file';EVENT_BUS='event_bus'
class LoggingManager:
	def __init__(self,config:Dict[str,Any]):self.config=config.get('logging',{});self.log_dir=Path(self.config.get('path','logs')).resolve();self.log_dir.mkdir(exist_ok=True,parents=True);self.console_level=self.config.get('console_level','INFO');self.file_level=self.config.get('file_level','DEBUG');(self.enabled_destinations):Set[LogDestination]={LogDestination.CONSOLE,LogDestination.FILE};self._hardware_detector=HardwareDetector();self._configure_logging();self._log_system_info()
	def _configure_logging(self)->None:
		logger.remove()
		if LogDestination.CONSOLE in self.enabled_destinations:logger.add(sys.stdout,level=self.console_level,format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',colorize=True,backtrace=True,diagnose=True)
		if LogDestination.FILE in self.enabled_destinations:log_file=self.log_dir/'maggie.log';logger.add(log_file,level=self.file_level,format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',rotation='10 MB',retention='1 week',compression='zip',backtrace=True,diagnose=True);error_log=self.log_dir/'errors.log';logger.add(error_log,level='ERROR',format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',rotation='5 MB',retention='1 month',compression='zip',backtrace=True,diagnose=True)
	def _log_system_info(self)->None:
		try:
			system_info=self._hardware_detector.detect_system();logger.info(f"System: {system_info['os']['system']} {system_info['os']['release']}");cpu_info=system_info['cpu'];logger.info(f"CPU: {cpu_info['model']}");logger.info(f"CPU Cores: {cpu_info['physical_cores']} physical, {cpu_info['logical_cores']} logical");memory_info=system_info['memory'];logger.info(f"RAM: {memory_info['total_gb']:.2f} GB (Available: {memory_info['available_gb']:.2f} GB)");gpu_info=system_info['gpu']
			if gpu_info.get('available',False):
				logger.info(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.2f} GB VRAM)");logger.info(f"CUDA available: {gpu_info['cuda_version']}")
				if gpu_info.get('is_rtx_3080',False):logger.info('RTX 3080 detected - applying optimal configurations')
			else:logger.warning('CUDA not available, GPU acceleration disabled')
		except Exception as e:logger.warning(f"Error logging system information: {e}")
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
				from maggie.service.locator import ServiceLocator;event_bus=ServiceLocator.get('event_bus')
				if event_bus:error_data={'type':str(exc_type.__name__),'message':str(exc_value),'traceback':''.join(traceback.format_tb(exc_traceback)),'is_unhandled':True};event_bus.publish('error_logged',error_data)
			except:pass
		sys.excepthook=global_exception_handler
	def get_logger(self,name:str)->logger:return logger.bind(name=name)
import traceback