"""
Maggie AI Assistant - Logging Utility
================================================

This module provides a comprehensive logging utility for managing and structuring logs in a Python application. 
It includes support for multiple logging destinations, asynchronous logging, batched logging, and contextual logging.
Classes:
	LogLevel (Enum):
		Enumeration representing different levels of logging severity.
	LogDestination (Enum):
		Enumeration representing different destinations where log messages can be sent.
	LoggingManager:
		A singleton class responsible for managing the logging system in the application. 
	ComponentLogger:
		A utility class for structured logging with additional context related to the component's state and resource usage.
Functions:
	logging_context(correlation_id: Optional[str] = None, component: str = '', operation: str = '', state: Any = None) -> Generator[Dict[str, Any], None, None]:
	log_operation(component: str = '', log_args: bool = True, log_result: bool = False, include_state: bool = True):
	- Supports logging to console, files, and event buses.
	- Provides detailed system information logging.
	- Includes performance metrics logging for operations.
	- Allows for state transition and resource allocation/deallocation logging.
	- Integrates with a state manager and resource manager for additional context.
	- Offers global exception handling for unhandled exceptions.
	- Enables asynchronous and batched logging for performance optimization.
	- Provides decorators for function-level logging with argument/result tracking.
"""
import os,sys,logging,time,uuid,inspect,functools,threading,queue
from enum import Enum,auto
from pathlib import Path
from typing import Dict,Any,Optional,List,Union,Set,Callable,Generator,TypeVar,cast
from contextlib import contextmanager
from loguru import logger

# Remove the direct import of HardwareDetector
from maggie.utils.resource.detector import HardwareDetector

T=TypeVar('T')
class LogLevel(str,Enum):
	"""
	LogLevel is an enumeration that represents different levels of logging severity.

	Attributes:
		DEBUG (str): Detailed information, typically of interest only when diagnosing problems.
		INFO (str): Confirmation that things are working as expected.
		WARNING (str): An indication that something unexpected happened, or indicative of some problem in the near future.
		ERROR (str): A more serious problem, the software has not been able to perform some function.
		CRITICAL (str): A very serious error, indicating that the program itself may be unable to continue running.
	"""
	DEBUG='DEBUG'
	INFO='INFO'
	WARNING='WARNING'
	ERROR='ERROR'
	CRITICAL='CRITICAL'

class LogDestination(str,Enum):
	"""
	LogDestination is an enumeration that represents different destinations
	where log messages can be sent.

	Attributes:
		CONSOLE (str): Represents logging to the console.
		FILE (str): Represents logging to a file.
		EVENT_BUS (str): Represents logging to an event bus.
	"""
	CONSOLE='CONSOLE'
	FILE='FILE'
	EVENT_BUS='EVENT_BUS'
class LoggingManager:
	"""
	LoggingManager is a singleton class responsible for managing the logging system in the application. 
	It provides features such as asynchronous logging, batched logging, and support for multiple logging destinations.
	Attributes:
		_instance (LoggingManager): The singleton instance of the LoggingManager.
		config (Dict[str, Any]): Configuration dictionary for logging settings.
		log_dir (Path): Directory where log files are stored.
		console_level (str): Logging level for console output.
		file_level (str): Logging level for file output.
		enabled_destinations (Set[LogDestination]): Set of enabled logging destinations.
		log_batch_size (int): Maximum size of the log batch before flushing.
		log_batch_timeout (float): Timeout in seconds for flushing the log batch.
		log_batch (List): List of batched log records.
		log_batch_lock (threading.RLock): Lock for synchronizing access to the log batch.
		log_batch_timer (threading.Timer): Timer for flushing the log batch.
		log_batch_enabled (bool): Whether batched logging is enabled.
		async_logging (bool): Whether asynchronous logging is enabled.
		log_queue (queue.Queue): Queue for asynchronous logging.
		log_worker (threading.Thread): Worker thread for processing asynchronous log records.
		correlation_id (Optional[str]): Correlation ID for tracking logs across requests.
	Methods:
		get_instance() -> 'LoggingManager':
			Returns the singleton instance of LoggingManager. Raises an error if not initialized.
		initialize(config: Dict[str, Any]) -> 'LoggingManager':
			Initializes the LoggingManager with the given configuration. Returns the instance.
		__init__(config: Dict[str, Any]):
			Initializes the LoggingManager instance with the provided configuration.
		_get_hardware_detector():
			Lazily imports and returns the HardwareDetector for system information.
		_configure_logging() -> None:
			Configures the logging system, including console and file handlers.
		_initialize_log_batching() -> None:
			Initializes the batching mechanism for logs.
		_initialize_async_logging() -> None:
			Initializes the asynchronous logging mechanism.
		_process_log_queue() -> None:
			Processes log records from the asynchronous logging queue.
		_flush_log_batch() -> None:
			Flushes the batched log records to the logging system.
		_log_system_info() -> None:
			Logs system information such as OS, CPU, memory, and GPU details.
		log(level: LogLevel, message: str, *args, **kwargs) -> None:
			Logs a message at the specified log level.
		set_correlation_id(correlation_id: str) -> None:
			Sets the correlation ID for tracking logs.
		get_correlation_id() -> Optional[str]:
			Retrieves the current correlation ID.
		clear_correlation_id() -> None:
			Clears the correlation ID.
		add_event_bus_handler(event_bus: Any) -> None:
			Adds a handler to publish error logs to an event bus.
		setup_global_exception_handler() -> None:
			Sets up a global exception handler to log unhandled exceptions.
		get_logger(name: str) -> logger:
			Returns a logger instance bound to the specified name.
		log_performance(component: str, operation: str, elapsed: float, details: Dict[str, Any] = None) -> None:
			Logs performance metrics for a specific operation.
		log_state_transition(from_state: Any, to_state: Any, trigger: str) -> None:
			Logs state transitions in the application.
		log_resource_allocation(resource_type: str, resource_name: str, state: Any, details: Dict[str, Any] = None) -> None:
			Logs resource allocation details.
		log_resource_deallocation(resource_type: str, resource_name: str, state: Any, details: Dict[str, Any] = None) -> None:
			Logs resource deallocation details.
		log_input_processing(input_type: str, state: Any, details: Dict[str, Any] = None) -> None:
			Logs input processing details.
		shutdown() -> None:
			Shuts down the logging system, flushing any remaining logs and stopping background threads.
	"""
	_instance=None
	_configured=False
	
	@classmethod
	def get_instance(cls)->'LoggingManager':
		"""
		Retrieve the singleton instance of the LoggingManager.

		Returns:
			LoggingManager: The singleton instance of the LoggingManager.

		Raises:
			RuntimeError: If the LoggingManager has not been initialized.
		"""
		if cls._instance is None:
			raise RuntimeError('LoggingManager not initialized')
		return cls._instance
	
	@classmethod
	def initialize(cls,config:Dict[str,Any])->'LoggingManager':
		"""
		Initializes the LoggingManager singleton instance with the provided configuration.

		Args:
			config (Dict[str, Any]): A dictionary containing configuration settings for the LoggingManager.

		Returns:
			LoggingManager: The singleton instance of the LoggingManager.

		Notes:
			If the LoggingManager has already been initialized, a warning is logged, and the existing instance is returned.
		"""
		if cls._instance is not None:
			logger.warning('LoggingManager already initialized')
			return cls._instance
		cls._instance=LoggingManager(config)
		return cls._instance
	
	def __init__(self,config:Dict[str,Any]):
		"""
		Initialize the logging utility with the provided configuration.
		Args:
			config (Dict[str, Any]): A dictionary containing logging configuration options. 
				Expected keys include:
					- 'logging': A dictionary with logging-specific settings.
					- 'path' (str): Directory path for log files. Defaults to 'logs'.
					- 'console_level' (str): Logging level for console output. Defaults to 'INFO'.
					- 'file_level' (str): Logging level for file output. Defaults to 'DEBUG'.
					- 'batch_size' (int): Number of log entries to batch before writing. Defaults to 50.
					- 'batch_timeout' (float): Timeout in seconds for batching logs. Defaults to 5.0.
					- 'batch_enabled' (bool): Whether log batching is enabled. Defaults to True.
					- 'async_enabled' (bool): Whether asynchronous logging is enabled. Defaults to True.
		Attributes:
			config (Dict[str, Any]): The logging configuration dictionary.
			log_dir (Path): The resolved directory path for log files.
			console_level (str): Logging level for console output.
			file_level (str): Logging level for file output.
			enabled_destinations (Set[LogDestination]): Set of enabled logging destinations.
			_hardware_detector (Optional[Any]): Lazy-loaded hardware detector instance.
			log_batch_size (int): Number of log entries to batch before writing.
			log_batch_timeout (float): Timeout in seconds for batching logs.
			log_batch (List[Any]): List to store batched log entries.
			log_batch_lock (threading.RLock): Lock for thread-safe access to log batching.
			log_batch_timer (Optional[threading.Timer]): Timer for managing log batch timeouts.
			log_batch_enabled (bool): Whether log batching is enabled.
			async_logging (bool): Whether asynchronous logging is enabled.
			log_queue (Optional[queue.Queue]): Queue for asynchronous logging.
			log_worker (Optional[threading.Thread]): Worker thread for asynchronous logging.
			correlation_id (Optional[Any]): Correlation ID for tracking log entries.
		Raises:
			OSError: If the log directory cannot be created.
		"""
		self.config=config.get('logging',{})
		self.log_dir=Path(self.config.get('path','logs')).resolve()
		self.log_dir.mkdir(exist_ok=True,parents=True)
		self.console_level=self.config.get('console_level','INFO')
		self.file_level=self.config.get('file_level','DEBUG')
		(self.enabled_destinations):Set[LogDestination]={LogDestination.CONSOLE,LogDestination.FILE}
		
		# Use lazy import for HardwareDetector to avoid circular reference
		self._hardware_detector = None
		
		self.log_batch_size=self.config.get('batch_size',50)
		self.log_batch_timeout=self.config.get('batch_timeout',5.)
		self.log_batch=[]
		self.log_batch_lock=threading.RLock()
		self.log_batch_timer=None
		self.log_batch_enabled=self.config.get('batch_enabled',True)
		self.async_logging=self.config.get('async_enabled',True)
		self.log_queue=queue.Queue()if self.async_logging else None
		self.log_worker=None
		self._configure_logging()
		self._log_system_info()
		if self.log_batch_enabled:self._initialize_log_batching()
		if self.async_logging:self._initialize_async_logging()
		self.correlation_id=None
	
	def _get_hardware_detector(self):
		"""
		Lazily initializes and returns the HardwareDetector instance.

		This method attempts to import the `HardwareDetector` class from 
		`maggie.utils.resource.detector` and instantiate it. If the import 
		fails, a warning is logged, and the `_hardware_detector` attribute 
		is set to `None`.

		Returns:
			HardwareDetector or None: An instance of `HardwareDetector` if 
			the import and instantiation succeed, otherwise `None`.
		"""
		# Lazy import of HardwareDetector
		if self._hardware_detector is None:
			try:
				from maggie.utils.resource.detector import HardwareDetector
				self._hardware_detector = HardwareDetector()
			except ImportError:
				logger.warning("Failed to import HardwareDetector, system info may be limited")
				self._hardware_detector = None
		return self._hardware_detector
		

	def _configure_logging(self) -> None:
		"""
		Configures the logging system for the application.

		This method removes any existing loggers and sets up new loggers based on the
		enabled destinations and logging levels. It supports logging to the console,
		as well as to multiple log files with specific formats, rotations, and retention
		policies. The log files include:

		- General log file (`maggie.log`): Logs all messages with a specified file logging level.
		- Error log file (`errors.log`): Logs messages with level `ERROR` or higher.
		- Performance log file (`performance.log`): Logs performance-related messages with
			additional `component` and `operation` metadata.
		- FSM log file (`fsm.log`): Logs finite state machine (FSM) related messages with
			additional `component` metadata.
		- Resource log file (`resources.log`): Logs resource operation messages with
			additional `component` and `resource_type` metadata.
		- Input log file (`input.log`): Logs input operation messages with additional
			`component` and `input_type` metadata.

		Each log file has specific configurations for rotation, retention, and compression.
		The method also supports colorized console output for better readability.

		Raises:
			Any exceptions related to file handling or logger configuration.
		"""
		logger.remove()
		if LogDestination.CONSOLE in self.enabled_destinations:
			logger.add(
				sys.stdout,
				level=self.console_level,
				format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
				colorize=True,
				backtrace=True,
				diagnose=True
			)
		if LogDestination.FILE in self.enabled_destinations:
			log_file=self.log_dir/'maggie.log'
			logger.add(
				log_file,
				level=self.file_level,
				format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',
				rotation='10 MB',
				retention='1 week',
				compression='zip',
				backtrace=True,
				diagnose=True
			)
			error_log=self.log_dir/'errors.log'
			logger.add(
				error_log,
				level='ERROR',
				format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',
				rotation='5 MB',
				retention='1 month',
				compression='zip',
				backtrace=True,
				diagnose=True
			)
			perf_log=self.log_dir/'performance.log'
			logger.add(
				perf_log,
				level='DEBUG',
				format='{time:YYYY-MM-DD HH:mm:ss} | PERFORMANCE | {extra[component]}:{extra[operation]} | {message}',
				filter=lambda record:'performance'in record['extra'],
				rotation='5 MB',
				retention='1 week',
				compression='zip'
			)
			fsm_log=self.log_dir/'fsm.log'
			logger.add(
				fsm_log,
				level='DEBUG',
				format='{time:YYYY-MM-DD HH:mm:ss} | FSM | {extra[component]} | {message}',
				filter=lambda record:'fsm'in record['extra'],
				rotation='5 MB',
				retention='1 week',
				compression='zip'
			)
			resource_log=self.log_dir/'resources.log'
			logger.add(
				resource_log,
				level='DEBUG',
				format='{time:YYYY-MM-DD HH:mm:ss} | RESOURCE | {extra[component]}:{extra[resource_type]} | {message}',
				filter=lambda record:'resource_operation'in record['extra'],
				rotation='5 MB',
				retention='1 week',
				compression='zip'
			)
			input_log=self.log_dir/'input.log'
			logger.add(
				input_log,level='DEBUG',
				format='{time:YYYY-MM-DD HH:mm:ss} | INPUT | {extra[component]}:{extra[input_type]} | {message}',
				filter=lambda record:'input_operation'in record['extra'],
				rotation='5 MB',
				retention='1 week',
				compression='zip'
			)
	
	def _initialize_log_batching(self) -> None:
		"""
		Initializes the log batching mechanism by setting up a timer that triggers
		the flushing of the log batch after a specified timeout. The timer runs
		as a daemon thread to ensure it does not block program termination.

		Attributes:
			self.log_batch_timer (threading.Timer): A timer object that schedules
				the periodic flushing of the log batch.
		"""
	
	def _initialize_async_logging(self) -> None:
		"""
		Initializes asynchronous logging by creating and starting a background
		thread to process log messages from a queue.

		This method sets up a daemon thread named 'AsyncLogWorker' that runs
		the `_process_log_queue` method. The thread ensures that log messages
		are handled asynchronously without blocking the main application flow.

		Returns:
			None
		"""
		self.log_worker=threading.Thread(target=self._process_log_queue,name='AsyncLogWorker',daemon=True)
		self.log_worker.start()

	def _process_log_queue(self) -> None:
		"""
		Processes log records from a queue in an asynchronous manner.

		This method continuously retrieves log records from the `log_queue` and 
		processes them based on their log level. It supports the following log levels:
		DEBUG, INFO, WARNING, ERROR, and CRITICAL. Each log record is expected to 
		be a tuple containing the log level, message, arguments, and keyword arguments.

		If the queue is empty, it waits for a short timeout before retrying. If a 
		`None` value is retrieved from the queue, the loop breaks, signaling the 
		termination of the logging process.

		Exceptions during processing are caught and written to `sys.stderr`.

		Raises:
			queue.Empty: If the queue is empty for the specified timeout.
			Exception: For any other unexpected errors during log processing.
		"""
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
		"""
		Flushes the current batch of log records.

		This method processes all log records in the `log_batch` list, sending them
		to the appropriate logging method based on their log level. After processing,
		the `log_batch` is cleared, and a new timer is started to schedule the next
		flush.

		The method is thread-safe and ensures that only one thread can access the
		`log_batch` at a time by using a lock.

		Log levels handled:
			- DEBUG: Calls `logger.debug`
			- INFO: Calls `logger.info`
			- WARNING: Calls `logger.warning`
			- ERROR: Calls `logger.error`
			- CRITICAL: Calls `logger.critical`

		Side Effects:
			- Clears the `log_batch` list.
			- Resets and starts a new timer (`log_batch_timer`) for the next flush.

		Note:
			This method is intended to be used internally and is not meant to be
			called directly by external code.
		"""
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
	
	def _log_system_info(self) -> None:
		"""
		Logs detailed system information including operating system, CPU, memory, and GPU details.

		This method uses a hardware detector to gather and log system information. If the hardware
		detector is unavailable, it logs a warning and provides basic system information.

		Logs include:
		- Operating system name and release version.
		- CPU model, physical cores, and logical cores.
		- Total and available RAM in GB.
		- GPU name, VRAM in GB, and CUDA availability (if applicable).

		Special handling:
		- If an RTX 3080 GPU is detected, a specific log message is added for optimal configurations.
		- Logs a warning if CUDA is not available or if the hardware detector is missing.

		Exceptions:
		- Catches and logs any exceptions that occur during the logging process.

		Returns:
			None
		"""
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
		"""
		Logs a message with the specified log level.

		This method supports multiple logging mechanisms:
		- Asynchronous logging using a queue.
		- Batch logging with a configurable batch size.
		- Direct logging to the appropriate log level.

		Args:
			level (LogLevel): The severity level of the log message.
			message (str): The log message to be recorded.
			*args: Additional positional arguments to be passed to the logger.
			**kwargs: Additional keyword arguments to be passed to the logger.

		Behavior:
			- If `async_logging` is enabled and `log_queue` is not None, the log
			  message is added to the queue for asynchronous processing.
			- If batch logging is enabled (`log_batch_enabled`), the log message
			  is added to a batch. When the batch size reaches `log_batch_size`,
			  the batch is flushed.
			- If neither of the above is enabled, the log message is logged
			  directly to the appropriate log level (DEBUG, INFO, WARNING, ERROR,
			  CRITICAL).

		Note:
			Ensure that the `log_queue` and `log_batch` are properly initialized
			if using asynchronous or batch logging, respectively.
		"""
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
	
	def set_correlation_id(self,correlation_id:str)->None:
		"""
		Sets the correlation ID for the current instance and configures the logger
		to include the correlation ID in its extra context.

		Args:
			correlation_id (str): The correlation ID to associate with the current instance.
		"""
		self.correlation_id=correlation_id
		logger.configure(extra={'correlation_id':correlation_id})
	
	def get_correlation_id(self)->Optional[str]:
		"""
		Retrieve the correlation ID associated with the current context.

		Returns:
			Optional[str]: The correlation ID if set, otherwise None.
		"""
		return self.correlation_id
	
	def clear_correlation_id(self) -> None:
		"""
		Clears the correlation ID for the current context.

		This method sets the `correlation_id` attribute to `None` and updates
		the logger configuration to remove the correlation ID from the logging
		context.

		Returns:
			None
		"""
		self.correlation_id=None
		logger.configure(extra={'correlation_id':None})
	
	def add_event_bus_handler(self, event_bus: Any) -> None:
		"""
		Adds an event bus handler to the logger for handling log records with levels
		'ERROR' and 'CRITICAL'. If the event bus destination is not already enabled,
		it is added to the enabled destinations.

		Args:
			event_bus (Any): The event bus instance to which error log events will be published.

		Behavior:
			- For log records with levels 'ERROR' or 'CRITICAL', a dictionary containing
				log details (e.g., message, level, time, name, function, line, module, and
				correlation ID) is created.
			- If a state manager is available via the service locator, the current state
				and its style are added to the log data.
			- The log data is published to the event bus under the topic 'error_logged'.
			- Configures the logger to use the custom handler for processing error-level logs.

		Raises:
			Exception: If an error occurs while retrieving the state manager or current state,
			it is silently ignored.
		"""
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

	def setup_global_exception_handler(self) -> None:
		"""
		Sets up a global exception handler to catch and log unhandled exceptions.

		This method overrides the default `sys.excepthook` with a custom handler that:
		- Logs unhandled exceptions using the `logger` with detailed traceback information.
		- Ignores `KeyboardInterrupt` exceptions to allow graceful termination of the program.
		- Publishes error details to an event bus if available, including:
			- Exception type and message.
			- Formatted traceback.
			- Correlation ID for tracking.
			- Current state information (if a state manager is available).

		Note:
			- The method attempts to retrieve and use the `event_bus` and `state_manager` 
				from the `ServiceLocator`. If these services are unavailable or an error 
				occurs during this process, it fails silently.
		"""
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

	def get_logger(self,name:str)->logger:
		"""
		Retrieves a logger instance with a specified name.

		Args:
			name (str): The name to bind to the logger instance.

		Returns:
			logger: A logger instance with the specified name bound to it.
		"""
		return logger.bind(name=name)
	
	def log_performance(self,component:str,operation:str,elapsed:float,details:Dict[str,Any]=None)->None:
		"""
		Logs the performance of a specific operation within a component.

		Args:
			component (str): The name of the component where the operation is performed.
			operation (str): The name of the operation being logged.
			elapsed (float): The time taken to complete the operation, in seconds.
			details (Dict[str, Any], optional): Additional details about the operation 
				as key-value pairs. Defaults to None.

		Returns:
			None
		"""
		log_entry=f"{operation} took {elapsed:.3f}s"
		if details:detail_str=', '.join(f"{k}={v}"for(k,v)in details.items());log_entry+=f" ({detail_str})"
		perf_logger=logger.bind(performance=True,component=component,operation=operation);perf_logger.debug(log_entry)
		if elapsed>1.:logger.debug(f"Performance: {component}/{operation} - {log_entry}")
	
	def log_state_transition(self, from_state: Any, to_state: Any, trigger: str) -> None:
		"""
		Logs a state transition event for a finite state machine (FSM).

		This method logs the transition from one state to another, including the trigger
		that caused the transition. It uses a specialized logger for FSM-related events
		and also logs the transition using the general logger.

		Args:
			from_state (Any): The state the FSM is transitioning from. If the state has
								a 'name' attribute, it will be used; otherwise, the state
								will be converted to a string.
			to_state (Any): The state the FSM is transitioning to. If the state has
							a 'name' attribute, it will be used; otherwise, the state
							will be converted to a string.
			trigger (str): The name of the event or action that triggered the state transition.

		Returns:
			None
		"""
		from_name=from_state.name if hasattr(from_state,'name')else str(from_state)
		to_name=to_state.name if hasattr(to_state,'name')else str(to_state)
		fsm_logger=logger.bind(fsm=True,component='StateManager')
		fsm_logger.info(f"State transition: {from_name} -> {to_name} (trigger: {trigger})")
		logger.info(f"State transition: {from_name} -> {to_name} (trigger: {trigger})")
	
	def log_resource_allocation(self,resource_type:str,resource_name:str,state:Any,details:Dict[str,Any]=None)->None:
		"""
		Logs the allocation of a resource with relevant details.

		Args:
			resource_type (str): The type of the resource being allocated (e.g., "CPU", "Memory").
			resource_name (str): The name of the resource being allocated.
			state (Any): The state of the resource, which can be an object with a 'name' attribute or any other type.
			details (Dict[str, Any], optional): Additional details about the resource allocation as key-value pairs. Defaults to None.

		Returns:
			None
		"""
		state_name=state.name if hasattr(state,'name')else str(state);log_msg=f"Allocated {resource_type}/{resource_name} in state {state_name}"
		if details:detail_str=', '.join(f"{k}={v}"for(k,v)in details.items());log_msg+=f" ({detail_str})"
		resource_logger=logger.bind(resource_operation=True,component='ResourceManager',resource_type=resource_type);resource_logger.info(log_msg)
	
	def log_resource_deallocation(self,resource_type:str,resource_name:str,state:Any,details:Dict[str,Any]=None)->None:
		"""
		Logs the deallocation of a resource with relevant details.

		Args:
			resource_type (str): The type of the resource being deallocated.
			resource_name (str): The name of the resource being deallocated.
			state (Any): The state of the resource during deallocation. If the state has a 'name' attribute, it will be used; otherwise, the string representation of the state will be logged.
			details (Dict[str, Any], optional): Additional details about the deallocation process. Defaults to None.

		Returns:
			None
		"""
		state_name=state.name if hasattr(state,'name')else str(state);log_msg=f"Deallocated {resource_type}/{resource_name} in state {state_name}"
		if details:detail_str=', '.join(f"{k}={v}"for(k,v)in details.items());log_msg+=f" ({detail_str})"
		resource_logger=logger.bind(resource_operation=True,component='ResourceManager',resource_type=resource_type);resource_logger.info(log_msg)
	
	def log_input_processing(self,input_type:str,state:Any,details:Dict[str,Any]=None)->None:
		"""
		Logs the processing of an input with details about its type, state, and additional information.

		Args:
			input_type (str): The type of input being processed.
			state (Any): The current state of the system or process. If the state has a 'name' attribute, it will be used; otherwise, the string representation of the state is used.
			details (Dict[str, Any], optional): Additional details about the input processing as key-value pairs. Defaults to None.

		Returns:
			None
		"""
		state_name=state.name if hasattr(state,'name')else str(state);log_msg=f"Processing {input_type} in state {state_name}"
		if details:detail_str=', '.join(f"{k}={v}"for(k,v)in details.items());log_msg+=f" ({detail_str})"
		input_logger=logger.bind(input_operation=True,component='InputProcessor',input_type=input_type);input_logger.info(log_msg)
	
	def shutdown(self)->None:
		"""
		Shuts down the logging system, ensuring all pending logs are processed and resources are cleaned up.

		This method performs the following actions:
		- If batch logging is enabled, flushes the log batch and cancels the batch timer.
		- If asynchronous logging is enabled, signals the logging queue to stop and waits for the logging worker to terminate.
		- Logs a message indicating that the logging system has been shut down.

		Returns:
			None
		"""
		if self.log_batch_enabled:
			self._flush_log_batch()
			if self.log_batch_timer:self.log_batch_timer.cancel()
		if self.async_logging and self.log_queue:
			self.log_queue.put(None)
			if self.log_worker:self.log_worker.join(timeout=2.)
		logger.info('Logging system shutdown')

@contextmanager
def logging_context(correlation_id:Optional[str]=None,component:str='',operation:str='',state:Any=None)->Generator[Dict[str,Any],None,None]:
	"""
	Provides a context manager for logging operations with contextual information.

	This context manager sets up a logging context with details such as correlation ID, 
	component name, operation name, and optional state. It ensures that the logging 
	context is properly managed and cleaned up after the operation completes.

	Args:
		correlation_id (Optional[str]): A unique identifier for correlating logs. 
			If not provided, a new UUID will be generated.
		component (str): The name of the component performing the operation.
		operation (str): The name of the operation being performed.
		state (Any, optional): The current state of the operation. If not provided, 
			attempts to retrieve the state from a state manager.

	Yields:
		Dict[str, Any]: A dictionary containing the logging context, including 
		correlation ID, component, operation, start time, and optionally the state.

	Raises:
		Exception: If an error occurs within the context, it is logged and re-raised.

	Notes:
		- The context manager integrates with a `LoggingManager` to manage correlation IDs.
		- Performance metrics (execution time) are logged if the operation takes longer 
		than 0.1 seconds.
		- If a `state_manager` is available, it attempts to retrieve the current state 
		dynamically when `state` is not provided.
	"""
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

def log_operation(component: str = '', log_args: bool = True, log_result: bool = False, include_state: bool = True):
	"""
	A decorator to log the execution of a function, including its arguments, result, and execution time.
	Optionally includes the current application state in the log context.

	Args:
		component (str): The name of the component or module where the function resides. Defaults to an empty string.
		log_args (bool): Whether to log the arguments passed to the function. Defaults to True.
		log_result (bool): Whether to log the result returned by the function. Defaults to False.
		include_state (bool): Whether to include the current application state in the logging context. Defaults to True.

	Returns:
		Callable: A decorator that wraps the target function with logging functionality.

	Notes:
		- If `log_args` is True, the arguments are logged up to a maximum length of 200 characters.
		- If `log_result` is True, the result is logged only if it is a primitive type (str, int, float, bool, or None).
		- The decorator measures and logs the execution time of the function.
		- If `include_state` is True, the current application state is retrieved using the `ServiceLocator` and included in the logging context.
		- Performance metrics are logged using the `LoggingManager`.

	Example:
		@log_operation(component="example_component", log_args=True, log_result=True)
		def example_function(arg1, arg2):
			return arg1 + arg2
	"""
	def decorator(func):
		"""
		A decorator that wraps a function to provide enhanced logging and performance tracking.

		The decorator logs the function's arguments, execution time, and result based on the provided
		configuration. It also integrates with a state manager and logging context for additional
		contextual information.

		Args:
			func (Callable): The function to be wrapped by the decorator.

		Returns:
			Callable: The wrapped function with added logging and performance tracking.

		Features:
			- Logs function arguments if `log_args` is enabled.
			- Truncates argument strings longer than 200 characters.
			- Logs the function's execution time and result if `log_result` is enabled.
			- Integrates with a state manager to include the current state in the logging context.
			- Tracks performance metrics using the `LoggingManager`.

		Notes:
			- Requires `functools`, `inspect`, and `time` modules.
			- Assumes the presence of `logging_context`, `logger`, and `LoggingManager` for logging.
			- Relies on `ServiceLocator` to fetch the state manager if `include_state` is enabled.
			- Handles exceptions gracefully to avoid disrupting the wrapped function's execution.
		"""
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			"""
			A decorator function that wraps another function to provide enhanced logging and performance tracking.

			Args:
				*args: Positional arguments passed to the wrapped function.
				**kwargs: Keyword arguments passed to the wrapped function.

			Keyword Arguments:
				log_args (bool): If True, logs the arguments passed to the function. Defaults to False.
				include_state (bool): If True, includes the current application state in the logging context. Defaults to False.
				log_result (bool): If True, logs the result returned by the function. Defaults to False.
				component (str): The name of the component to include in the logging context. Defaults to None.

			Behavior:
				- Logs the function name and its arguments (if `log_args` is True).
				- Truncates the logged arguments string if it exceeds 200 characters.
				- Attempts to retrieve the current application state (if `include_state` is True).
				- Logs the execution time of the wrapped function.
				- Logs the result of the function (if `log_result` is True).
				- Handles exceptions gracefully during state retrieval and performance logging.

			Returns:
				The result of the wrapped function.

			Note:
				- This decorator uses `logging_context` for structured logging.
				- Performance metrics are logged using `LoggingManager`.
				- The decorator is designed to work with the `maggie` framework and its service locator.
			"""
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
	"""
	ComponentLogger is a utility class for structured logging with additional context 
	related to the component's state and resource usage. It provides methods for 
	logging messages at various levels (debug, info, warning, error, critical) and 
	supports logging state transitions and performance metrics.

	Attributes:
		component (str): The name of the component associated with this logger.
		logger: The underlying logger instance with component-specific binding.
		_state_manager: A reference to the state manager service, if available.
		_resource_manager: A reference to the resource manager service, if available.

	Methods:
		__init__(component_name: str):
			Initializes the ComponentLogger with the given component name and 
			attempts to retrieve state and resource managers.

		_get_state_context() -> Dict[str, Any]:
			Retrieves the current state context from the state manager, if available.

		_get_resource_context() -> Dict[str, Any]:
			Retrieves the current resource usage context from the resource manager, 
			if available.

		debug(message: str, **kwargs) -> None:
			Logs a debug-level message with optional additional context.

		info(message: str, **kwargs) -> None:
			Logs an info-level message with optional additional context.

		warning(message: str, **kwargs) -> None:
			Logs a warning-level message with optional additional context.

		error(message: str, exception: Optional[Exception] = None, **kwargs) -> None:
			Logs an error-level message with optional exception details and context.

		critical(message: str, exception: Optional[Exception] = None, **kwargs) -> None:
			Logs a critical-level message with optional exception details and 
			additional resource context.

		log_state_change(old_state: Any, new_state: Any, trigger: str) -> None:
			Logs a state transition with details about the old state, new state, 
			and the trigger that caused the transition.

		log_performance(operation: str, elapsed: float, details: Dict[str, Any] = None) -> None:
			Logs performance metrics for a specific operation, including elapsed 
			time and optional additional details.
	"""
	def __init__(self,component_name:str):
		"""
		Initializes an instance of the class with the specified component name.

		Args:
			component_name (str): The name of the component to associate with this instance.

		Attributes:
			component (str): The name of the component.
			logger (Logger): A logger instance bound to the specified component.
			_state_manager (Optional[Any]): The state manager instance, if available.
			_resource_manager (Optional[Any]): The resource manager instance, if available.

		Notes:
			Attempts to retrieve the 'state_manager' and 'resource_manager' from the
			ServiceLocator. If unsuccessful, these attributes will remain as None.
		"""
		self.component=component_name
		self.logger=logger.bind(component=component_name)
		self._state_manager=None;self._resource_manager=None
		try:
			from maggie.service.locator import ServiceLocator
			self._state_manager=ServiceLocator.get('state_manager')
			self._resource_manager=ServiceLocator.get('resource_manager')
		except Exception:pass

	def _get_state_context(self)->Dict[str,Any]:
		"""
		Retrieve the current state context as a dictionary.

		This method attempts to fetch the current state from the state manager
		(if available) and includes it in the returned context dictionary.

		Returns:
			Dict[str, Any]: A dictionary containing the current state context.
							If the state manager is not available or an error
							occurs, the dictionary will be empty or partially populated.
		"""
		context={}
		if self._state_manager:
			try:current_state=self._state_manager.get_current_state();context['state']=current_state.name
			except Exception:pass
		return context
	
	def _get_resource_context(self)->Dict[str,Any]:
		def _get_resource_context(self) -> Dict[str, Any]:
			"""
			Retrieves the resource usage context from the resource manager.

			This method collects information about memory and GPU usage from the 
			resource manager, if available, and returns it as a dictionary.

			Returns:
				Dict[str, Any]: A dictionary containing resource usage information:
					- 'memory_percent' (float): The percentage of memory used, if available.
					- 'gpu_percent' (float): The percentage of GPU memory used, if available 
					  and the GPU is accessible.
			"""
		context={}
		if self._resource_manager:
			try:
				resource_status=self._resource_manager.get_resource_status()
				if'memory'in resource_status:context['memory_percent']=resource_status['memory'].get('used_percent',0)
				if'gpu'in resource_status and resource_status['gpu'].get('available',False):context['gpu_percent']=resource_status['gpu'].get('memory_percent',0)
			except Exception:pass
		return context
	
	def debug(self,message:str,**kwargs)->None:
		"""
		Logs a debug-level message with additional contextual information.

		Args:
			message (str): The debug message to log.
			**kwargs: Additional keyword arguments to include in the log entry.

		Returns:
			None
		"""
		context=self._get_state_context()
		self.logger.bind(**context).debug(message,**kwargs)

	def info(self,message:str,**kwargs)->None:
		"""
		Logs an informational message with additional context.

		Args:
			message (str): The informational message to log.
			**kwargs: Additional keyword arguments to include in the log entry.

		Returns:
			None
		"""
		context=self._get_state_context()
		self.logger.bind(**context).info(message,**kwargs)

	def warning(self,message:str,**kwargs)->None:
		"""
		Logs a warning message with additional contextual information.

		Args:
			message (str): The warning message to log.
			**kwargs: Additional keyword arguments to pass to the logger.

		Returns:
			None
		"""
		context=self._get_state_context()
		self.logger.bind(**context).warning(message,**kwargs)

	def error(self,message:str,exception:Optional[Exception]=None,**kwargs)->None:
		"""
		Logs an error message with optional exception details and additional context.

		Args:
			message (str): The error message to log.
			exception (Optional[Exception]): An optional exception to include in the log.
			**kwargs: Additional keyword arguments to include in the log.

		Returns:
			None
		"""
		context=self._get_state_context()
		if exception:
			self.logger.bind(**context).error(f"{message}: {exception}",exc_info=exception,**kwargs)
		else:
			self.logger.bind(**context).error(message,**kwargs)

	def critical(self,message:str,exception:Optional[Exception]=None,**kwargs)->None:
		"""
		Logs a critical message with optional exception details and additional context.

		Args:
			message (str): The critical message to log.
			exception (Optional[Exception], optional): An exception to include in the log. Defaults to None.
			**kwargs: Additional keyword arguments to pass to the logger.

		Behavior:
			- Retrieves the current state and resource context.
			- If an exception is provided, logs the message along with the exception details and stack trace.
			- If no exception is provided, logs the message with the context.

		"""
		context=self._get_state_context();resource_context=self._get_resource_context();context.update(resource_context)
		if exception:
			self.logger.bind(**context).critical(f"{message}: {exception}",exc_info=exception,**kwargs)
		else:self.logger.bind(**context).critical(message,**kwargs)

	def log_state_change(self,old_state:Any,new_state:Any,trigger:str)->None:
		"""
		Logs a state transition and triggers additional logging through a logging manager.

		Args:
			old_state (Any): The previous state before the transition. If it has a 'name' attribute, it will be used; otherwise, its string representation will be used.
			new_state (Any): The new state after the transition. If it has a 'name' attribute, it will be used; otherwise, its string representation will be used.
			trigger (str): The event or action that caused the state transition.

		Behavior:
			- Logs the state transition using the logger with additional context.
			- Attempts to log the state transition through a singleton LoggingManager instance.
			- Silently handles `RuntimeError` exceptions raised during the LoggingManager operation.
		"""
		old_name=old_state.name if hasattr(old_state,'name')else str(old_state)
		new_name=new_state.name if hasattr(new_state,'name')else str(new_state)
		context={'fsm':True,'from_state':old_name,'to_state':new_name,'trigger':trigger}
		self.logger.bind(**context).info(f"State change: {old_name} -> {new_name} (trigger: {trigger})")
		try:
			logging_mgr=LoggingManager.get_instance()
			logging_mgr.log_state_transition(old_state,new_state,trigger)
		except RuntimeError:
			pass

	def log_performance(self,operation:str,elapsed:float,details:Dict[str,Any]=None)->None:
		"""
		Logs the performance of a specific operation, including its elapsed time and optional details.

		Args:
			operation (str): The name of the operation being logged.
			elapsed (float): The time taken to complete the operation, in seconds.
			details (Dict[str, Any], optional): Additional details about the operation as key-value pairs. Defaults to None.

		Returns:
			None
		"""
		message=f"Performance: {operation} took {elapsed:.3f}s"
		if details:
			message+=f" ({', '.join(f'{k}={v}'for(k,v)in details.items())})"
			self.logger.debug(message)
		try:
			logging_mgr=LoggingManager.get_instance()
			logging_mgr.log_performance(self.component,operation,elapsed,details)
		except RuntimeError:
			pass