import os
import sys
import logging
import time
import uuid
import inspect
import functools
import threading
import queue
import traceback
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set, Callable, Generator, TypeVar, cast
from contextlib import contextmanager

T = TypeVar('T')

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('maggie.error_handling')


class LogLevel(Enum):
    """Enumeration of log severity levels.

    This enum provides a type-safe way to specify log levels throughout the application,
    serving as a wrapper around the standard logging levels from Python's logging module.

    Attributes
    ----------
    DEBUG : Enum
        Detailed information, typically of interest only when diagnosing problems.
    INFO : Enum
        Confirmation that things are working as expected.
    WARNING : Enum
        An indication that something unexpected happened, or may happen in the near future.
    ERROR : Enum
        Due to a more serious problem, the software has not been able to perform some function.
    CRITICAL : Enum
        A serious error, indicating that the program itself may be unable to continue running.

    See Also
    --------
    logging : Python's logging module documentation
    """
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class LogDestination(Enum):
    """Enumeration of possible log destinations.

    This enum provides a type-safe way to specify where log messages should be sent.
    The logging system supports multiple destinations simultaneously.

    Attributes
    ----------
    CONSOLE : str
        Output logs to the console (stdout).
    FILE : str
        Output logs to a file on disk.
    EVENT_BUS : str
        Publish logs as events on the application's event bus.

    Notes
    -----
    The EVENT_BUS destination requires the event bus component to be available
    and properly initialized. If it's not available, logs to this destination
    will be silently dropped.
    """
    CONSOLE = 'CONSOLE'
    FILE = 'FILE'
    EVENT_BUS = 'EVENT_BUS'


class LoggingManager:
    """Manages the logging system with enhanced capabilities via dependency injection.

    This class follows the singleton pattern and provides methods for logging messages
    with various levels of severity. It can be enhanced with event publishing, error handling,
    and state information capabilities when those dependencies become available.

    Attributes
    ----------
    config : Dict[str, Any]
        Configuration dictionary for the logging system.
    log_dir : Path
        Directory where log files are stored.
    console_level : str
        Minimum log level for console output (e.g., 'INFO', 'DEBUG').
    file_level : str
        Minimum log level for file output.
    log_batch_size : int
        Number of log messages to batch together for asynchronous logging.
    log_batch_timeout : float
        Maximum time (in seconds) to wait before flushing a partial batch.
    async_logging : bool
        Whether to use asynchronous logging.
    correlation_id : Optional[str]
        Current correlation ID for tracking related logs.

    Notes
    -----
    This class implements the Singleton pattern to ensure that there's only one instance
    of the logging manager throughout the application.
    """
    
    _instance = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> 'LoggingManager':
        """Get the singleton instance of LoggingManager.

        Returns
        -------
        LoggingManager
            The singleton instance of LoggingManager.

        Raises
        ------
        RuntimeError
            If LoggingManager has not been initialized yet.

        Examples
        --------
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.log(LogLevel.INFO, "Using existing logging manager")
        """
        if cls._instance is None:
            raise RuntimeError('LoggingManager not initialized')
        return cls._instance

    @classmethod
    def initialize(cls, config: Dict[str, Any]) -> 'LoggingManager':
        """Initialize the LoggingManager with configuration settings.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing logging settings, including:
            - logging.path: Directory where log files will be stored
            - logging.console_level: Minimum log level for console output (default: 'INFO')
            - logging.file_level: Minimum log level for file output (default: 'DEBUG')
            - logging.batch_size: Number of log messages to batch (default: 50)
            - logging.batch_timeout: Timeout for log batching in seconds (default: 5.0)
            - logging.async_enabled: Whether to use asynchronous logging (default: True)

        Returns
        -------
        LoggingManager
            The initialized singleton instance of LoggingManager.

        Examples
        --------
        >>> config = {
        ...     'logging': {
        ...         'path': 'logs',
        ...         'console_level': 'INFO',
        ...         'file_level': 'DEBUG'
        ...     }
        ... }
        >>> logging_manager = LoggingManager.initialize(config)
        """
        if cls._instance is not None:
            logger.warning('LoggingManager already initialized')
            return cls._instance
        with cls._lock:
            if cls._instance is None:
                cls._instance = LoggingManager(config)
        return cls._instance

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the logging utility with the provided configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing logging settings.

        Notes
        -----
        This constructor should not be called directly. Use LoggingManager.initialize()
        instead to properly set up the singleton instance.
        """
        self.config = config.get('logging', {})
        self.log_dir = Path(self.config.get('path', 'logs')).resolve()
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.console_level = self.config.get('console_level', 'INFO')
        self.file_level = self.config.get('file_level', 'DEBUG')
        self._enhanced_logging = False
        self._event_publisher = None
        self._error_handler = None
        self._state_provider = None
        self.log_batch_size = self.config.get('batch_size', 50)
        self.log_batch_timeout = self.config.get('batch_timeout', 5.0)
        self.async_logging = self.config.get('async_enabled', True)
        self.correlation_id = None
        self._configure_basic_logging()

    def _configure_basic_logging(self) -> None:
        """Configure basic logging with standard Python logging module.

        Notes
        -----
        This is an internal method and should not be called directly.
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.console_level))
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        log_file = self.log_dir / 'maggie.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, self.file_level))
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)

    def enhance_with_event_publisher(self, event_publisher: Any) -> None:
        """Enhance logging with an event publisher.

        Parameters
        ----------
        event_publisher : Any
            An object implementing the IEventPublisher interface with a publish() method.

        Examples
        --------
        >>> from maggie.core.event import EventBus
        >>> from maggie.utils.adapters import EventBusAdapter
        >>> event_bus = EventBus()
        >>> event_bus_adapter = EventBusAdapter(event_bus)
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.enhance_with_event_publisher(event_bus_adapter)
        """
        self._event_publisher = event_publisher
        self._enhanced_logging = True

    def enhance_with_error_handler(self, error_handler: Any) -> None:
        """Enhance logging with an error handler.

        Parameters
        ----------
        error_handler : Any
            An object implementing the IErrorHandler interface for error handling.

        Examples
        --------
        >>> from maggie.utils.error_handling import ErrorHandler
        >>> error_handler = ErrorHandler()
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.enhance_with_error_handler(error_handler)
        """
        self._error_handler = error_handler
        self._enhanced_logging = True

    def enhance_with_state_provider(self, state_provider: Any) -> None:
        """Enhance logging with a state provider.

        Parameters
        ----------
        state_provider : Any
            An object implementing the IStateProvider interface with get_current_state().

        Examples
        --------
        >>> from maggie.core.state import StateManager
        >>> state_manager = StateManager()
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.enhance_with_state_provider(state_manager)
        """
        self._state_provider = state_provider
        self._enhanced_logging = True

    def log(self, level: LogLevel, message: str, *args: Any, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log a message with the specified severity level.

        Parameters
        ----------
        level : LogLevel
            The severity level of the log message.
        message : str
            The message to log, can contain format specifiers.
        *args : Any
            Positional arguments for message formatting.
        exception : Exception, optional
            An exception to include with the log message.
        **kwargs : Any
            Additional key-value pairs for structured logging.

        Examples
        --------
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.log(LogLevel.INFO, "Processing %s items", 5)
        >>> try:
        ...     result = 1 / 0
        ... except Exception as e:
        ...     logging_manager.log(LogLevel.ERROR, "Division error", exception=e)
        """
        log_method = {
            LogLevel.DEBUG: logger.debug,
            LogLevel.INFO: logger.info,
            LogLevel.WARNING: logger.warning,
            LogLevel.ERROR: logger.error,
            LogLevel.CRITICAL: logger.critical
        }.get(level)
        if log_method:
            log_method(message, *args, exc_info=exception, **kwargs)
        
        if self._enhanced_logging and self._event_publisher and level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            try:
                event_data = {
                    'message': message,
                    'level': level.name,
                    'timestamp': time.time(),
                    'correlation_id': self.correlation_id
                }
                if self._state_provider:
                    try:
                        current_state = self._state_provider.get_current_state()
                        if hasattr(current_state, 'name'):
                            event_data['state'] = current_state.name
                    except Exception:
                        pass
                if exception:
                    event_data['exception'] = str(exception)
                self._event_publisher.publish('error_logged', event_data)
            except Exception as e:
                logger.warning(f"Failed to publish error event: {e}")

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for tracking related logs.

        Parameters
        ----------
        correlation_id : str
            The correlation ID to set, typically a UUID.

        Examples
        --------
        >>> logging_manager = LoggingManager.get_instance()
        >>> correlation_id = str(uuid.uuid4())
        >>> logging_manager.set_correlation_id(correlation_id)
        """
        self.correlation_id = correlation_id

    def get_correlation_id(self) -> Optional[str]:
        """Get the current correlation ID.

        Returns
        -------
        Optional[str]
            The current correlation ID, or None if not set.

        Examples
        --------
        >>> logging_manager = LoggingManager.get_instance()
        >>> current_id = logging_manager.get_correlation_id()
        """
        return self.correlation_id

    def clear_correlation_id(self) -> None:
        """Clear the correlation ID.

        Examples
        --------
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.clear_correlation_id()
        """
        self.correlation_id = None

    def log_performance(self, component: str, operation: str, elapsed: float, details: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metrics.

        Parameters
        ----------
        component : str
            The name of the component performing the operation.
        operation : str
            The name of the operation performed.
        elapsed : float
            The elapsed time in seconds for the operation.
        details : Dict[str, Any], optional
            Additional details about the operation.

        Examples
        --------
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.log_performance("DataProcessor", "process_data", 0.532)
        """
        log_entry = f"{operation} took {elapsed:.3f}s"
        if details:
            detail_str = ', '.join(f"{k}={v}" for k, v in details.items())
            log_entry += f" ({detail_str})"
        logger.debug(f"Performance: {component}/{operation} - {log_entry}")
        
        if self._enhanced_logging and self._event_publisher:
            try:
                event_data = {
                    'component': component,
                    'operation': operation,
                    'elapsed_time': elapsed,
                    'details': details or {},
                    'timestamp': time.time(),
                    'correlation_id': self.correlation_id
                }
                self._event_publisher.publish('performance_metric', event_data)
            except Exception as e:
                logger.warning(f"Failed to publish performance metric: {e}")

    def log_state_transition(self, from_state: Any, to_state: Any, trigger: str) -> None:
        """Log state transition.

        Parameters
        ----------
        from_state : Any
            The source state of the transition.
        to_state : Any
            The destination state of the transition.
        trigger : str
            The event or action that triggered the transition.

        Examples
        --------
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.log_state_transition("IDLE", "ACTIVE", "user_command")
        """
        from_name = from_state.name if hasattr(from_state, 'name') else str(from_state)
        to_name = to_state.name if hasattr(to_state, 'name') else str(to_state)
        logger.info(f"State transition: {from_name} -> {to_name} (trigger: {trigger})")
        
        if self._enhanced_logging and self._event_publisher:
            try:
                event_data = {
                    'from_state': from_name,
                    'to_state': to_name,
                    'trigger': trigger,
                    'timestamp': time.time(),
                    'correlation_id': self.correlation_id
                }
                self._event_publisher.publish('state_transition_logged', event_data)
            except Exception as e:
                logger.warning(f"Failed to publish state transition: {e}")

    def setup_global_exception_handler(self) -> None:
        """Set up global exception handler.

        Examples
        --------
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.setup_global_exception_handler()
        """
        def global_exception_handler(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logger.critical('Unhandled exception:', exc_info=(exc_type, exc_value, exc_traceback))
            if self._enhanced_logging and self._event_publisher:
                try:
                    event_data = {
                        'type': str(exc_type.__name__),
                        'message': str(exc_value),
                        'traceback': ''.join(traceback.format_tb(exc_traceback)),
                        'is_unhandled': True,
                        'timestamp': time.time(),
                        'correlation_id': self.correlation_id
                    }
                    self._event_publisher.publish('unhandled_exception', event_data)
                except Exception:
                    pass
        sys.excepthook = global_exception_handler


class ComponentLogger:
    """A simplified component logger that doesn't depend on other modules.

    Attributes
    ----------
    component : str
        The name of the component this logger is associated with.
    logger : logging.Logger
        The underlying Python logger instance.
    """

    def __init__(self, component_name: str) -> None:
        """Initialize a component logger.

        Parameters
        ----------
        component_name : str
            The name of the component this logger is associated with.

        Examples
        --------
        >>> logger = ComponentLogger("UserService")
        >>> logger.info("Service initialized")
        """
        self.component = component_name
        self.logger = logging.getLogger(component_name)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message.

        Parameters
        ----------
        message : str
            The debug message to log.
        **kwargs : Any
            Additional key-value pairs, including 'exception' for exception info.

        Examples
        --------
        >>> logger = ComponentLogger("DataProcessor")
        >>> logger.debug("Processing file: %s", "data.txt")
        """
        exception = kwargs.pop('exception', None)
        if exception:
            self.logger.debug(message, exc_info=exception, **kwargs)
        else:
            self.logger.debug(message, **kwargs)
        try:
            manager = LoggingManager.get_instance()
            manager.log(LogLevel.DEBUG, message, exception=exception, **kwargs)
        except Exception:
            pass

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message.

        Parameters
        ----------
        message : str
            The info message to log.
        **kwargs : Any
            Additional key-value pairs, including 'exception' for exception info.

        Examples
        --------
        >>> logger = ComponentLogger("UserService")
        >>> logger.info("User logged in", user_id=123)
        """
        exception = kwargs.pop('exception', None)
        if exception:
            self.logger.info(message, exc_info=exception, **kwargs)
        else:
            self.logger.info(message, **kwargs)
        try:
            manager = LoggingManager.get_instance()
            manager.log(LogLevel.INFO, message, exception=exception, **kwargs)
        except Exception:
            pass

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message.

        Parameters
        ----------
        message : str
            The warning message to log.
        **kwargs : Any
            Additional key-value pairs, including 'exception' for exception info.

        Examples
        --------
        >>> logger = ComponentLogger("NetworkService")
        >>> logger.warning("Connection timeout")
        """
        exception = kwargs.pop('exception', None)
        if exception:
            self.logger.warning(message, exc_info=exception, **kwargs)
        else:
            self.logger.warning(message, **kwargs)
        try:
            manager = LoggingManager.get_instance()
            manager.log(LogLevel.WARNING, message, exception=exception, **kwargs)
        except Exception:
            pass

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log an error message.

        Parameters
        ----------
        message : str
            The error message to log.
        exception : Exception, optional
            An exception to include with the log.
        **kwargs : Any
            Additional key-value pairs for structured logging.

        Examples
        --------
        >>> logger = ComponentLogger("FileService")
        >>> try:
        ...     open("nonexistent.txt")
        ... except Exception as e:
        ...     logger.error("File not found", exception=e)
        """
        if exception:
            self.logger.error(message, exc_info=exception, **kwargs)
        else:
            self.logger.error(message, **kwargs)
        try:
            manager = LoggingManager.get_instance()
            manager.log(LogLevel.ERROR, message, exception=exception, **kwargs)
        except Exception:
            pass

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log a critical message.

        Parameters
        ----------
        message : str
            The critical message to log.
        exception : Exception, optional
            An exception to include with the log.
        **kwargs : Any
            Additional key-value pairs for structured logging.

        Examples
        --------
        >>> logger = ComponentLogger("SystemService")
        >>> logger.critical("System failure")
        """
        if exception:
            self.logger.critical(message, exc_info=exception, **kwargs)
        else:
            self.logger.critical(message, **kwargs)
        try:
            manager = LoggingManager.get_instance()
            manager.log(LogLevel.CRITICAL, message, exception=exception, **kwargs)
        except Exception:
            pass

    def log_state_change(self, old_state: Any, new_state: Any, trigger: str) -> None:
        """Log a state transition.

        Parameters
        ----------
        old_state : Any
            The source state of the transition.
        new_state : Any
            The destination state of the transition.
        trigger : str
            The event or action that triggered the transition.

        Examples
        --------
        >>> logger = ComponentLogger("StateMachine")
        >>> logger.log_state_change("IDLE", "ACTIVE", "start")
        """
        old_name = old_state.name if hasattr(old_state, 'name') else str(old_state)
        new_name = new_state.name if hasattr(new_state, 'name') else str(new_state)
        self.info(f"State change: {old_name} -> {new_name} (trigger: {trigger})")
        try:
            manager = LoggingManager.get_instance()
            manager.log_state_transition(old_state, new_state, trigger)
        except Exception:
            pass

    def log_performance(self, operation: str, elapsed: float, details: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metrics.

        Parameters
        ----------
        operation : str
            The name of the operation performed.
        elapsed : float
            The elapsed time in seconds for the operation.
        details : Dict[str, Any], optional
            Additional details about the operation.

        Examples
        --------
        >>> logger = ComponentLogger("DatabaseService")
        >>> logger.log_performance("query", 0.123, {"rows": 100})
        """
        message = f"Performance: {operation} took {elapsed:.3f}s"
        if details:
            message += f" ({', '.join(f'{k}={v}' for k, v in details.items())})"
        self.debug(message)
        try:
            manager = LoggingManager.get_instance()
            manager.log_performance(self.component, operation, elapsed, details)
        except Exception:
            pass


@contextmanager
def logging_context(correlation_id: Optional[str] = None, component: str = '', operation: str = '', state: Any = None) -> Generator[Dict[str, Any], None, None]:
    """A context manager for structured logging with correlation tracking.

    Parameters
    ----------
    correlation_id : str, optional
        A unique identifier for correlating related log messages.
    component : str, optional
        The name of the component performing the operation.
    operation : str, optional
        The name of the operation being performed.
    state : Any, optional
        The current application state, if relevant.

    Yields
    ------
    Dict[str, Any]
        A context dictionary with correlation_id, component, operation, and start_time.

    Examples
    --------
    >>> with logging_context(correlation_id="req-123", component="Processor", operation="process") as ctx:
    ...     ctx['step'] = 'start'
    ...     # processing code here
    """
    ctx_id = correlation_id or str(uuid.uuid4())
    context = {
        'correlation_id': ctx_id,
        'component': component,
        'operation': operation,
        'start_time': time.time()
    }
    if state is not None:
        context['state'] = state.name if hasattr(state, 'name') else str(state)
    
    try:
        manager = LoggingManager.get_instance()
        old_correlation_id = manager.get_correlation_id()
        manager.set_correlation_id(ctx_id)
    except Exception:
        old_correlation_id = None
    
    logger_instance = logging.getLogger(component or 'context')
    
    try:
        yield context
    except Exception as e:
        logger_instance.error(f"Error in {component}/{operation}: {e}", exc_info=True)
        raise
    finally:
        elapsed = time.time() - context['start_time']
        logger_instance.debug(f"{component}/{operation} completed in {elapsed:.3f}s")
        try:
            manager = LoggingManager.get_instance()
            if old_correlation_id is not None:
                manager.set_correlation_id(old_correlation_id)
            if component and operation:
                manager.log_performance(component, operation, elapsed)
        except Exception:
            pass


def log_operation(component: str = '', log_args: bool = True, log_result: bool = False, include_state: bool = True) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """A decorator for logging function operations with detailed information.

    Parameters
    ----------
    component : str, optional
        The name of the component containing the function.
    log_args : bool, optional
        Whether to log the function arguments (default: True).
    log_result : bool, optional
        Whether to log the function return value (default: False).
    include_state : bool, optional
        Whether to include application state in logs (default: True).

    Returns
    -------
    Callable
        A decorator function wrapping the target with logging capabilities.

    Examples
    --------
    >>> @log_operation(component="UserService")
    ... def get_user(user_id):
    ...     return {"id": user_id}
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation = func.__name__
            args_str = ''
            if log_args:
                sig = inspect.signature(func)
                arg_names = list(sig.parameters.keys())
                pos_args = []
                for i, arg in enumerate(args):
                    if i < len(arg_names) and i > 0:
                        pos_args.append(f"{arg_names[i]}={repr(arg)}")
                    elif i >= len(arg_names):
                        pos_args.append(repr(arg))
                kw_args = [f"{k}={repr(v)}" for k, v in kwargs.items()]
                all_args = pos_args + kw_args
                args_str = ', '.join(all_args)
                if len(args_str) > 200:
                    args_str = args_str[:197] + '...'
            
            state = None
            logger_instance = logging.getLogger(component or func.__module__)
            
            if log_args and args_str:
                logger_instance.debug(f"{operation} called with args: {args_str}")
            
            with logging_context(component=component, operation=operation, state=state) as ctx:
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                if log_result:
                    if isinstance(result, (str, int, float, bool, type(None))):
                        logger_instance.debug(f"{operation} returned: {result}")
                    else:
                        logger_instance.debug(f"{operation} returned: {type(result).__name__}")
                
                try:
                    manager = LoggingManager.get_instance()
                    manager.log_performance(component or func.__module__, operation, elapsed)
                except Exception:
                    logger_instance.debug(f"{operation} completed in {elapsed:.3f}s")
                
                return result
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    config = {
        'logging': {
            'path': 'logs',
            'console_level': 'INFO',
            'file_level': 'DEBUG'
        }
    }
    logging_manager = LoggingManager.initialize(config)
    logger = ComponentLogger("TestComponent")
    logger.info("Testing logging system")