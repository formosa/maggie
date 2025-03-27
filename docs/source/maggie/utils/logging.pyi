"""
Maggie AI Assistant - Logging Utility
====================================

This module provides a comprehensive logging utility for managing and structuring logs in the Maggie AI Assistant.
It implements advanced logging patterns including hierarchical logging, contextual logging, performance tracking,
and integrates with the application's state management and event systems.

The module is designed around several key software engineering patterns:
- Singleton Pattern: LoggingManager implements a singleton to provide a central logging service
- Decorator Pattern: log_operation provides a clean interface for function instrumentation
- Context Manager Pattern: logging_context provides scoped logging contexts
- Observer Pattern: Integration with the event system for log event propagation

Features
--------
- Multiple logging destinations (console, file, event bus)
- Asynchronous and batched logging capabilities
- Contextual logging with correlation IDs and state awareness
- Performance tracking and metrics collection
- Component-based logging with hierarchical organization
- Integration with application state management
- Global exception handling and reporting

Examples
--------
Basic usage with ComponentLogger:
```python
from maggie.utils.logging import ComponentLogger

logger = ComponentLogger('MyComponent')
logger.info('This is an info message')
logger.error('An error occurred', exception=ValueError('Invalid value'))
```

Using the logging context manager:
```python
from maggie.utils.logging import logging_context

with logging_context(component='MyComponent', operation='important_task') as ctx:
    # Do some work
    ctx['status'] = 'in_progress'
    # More work
    # Context automatically logs duration when exiting
```

Using the operation decorator:
```python
from maggie.utils.logging import log_operation

@log_operation(component='MyComponent', log_args=True)
def my_function(arg1, arg2):
    # Function code
    return result
```

Initializing the LoggingManager (typically done in application startup):
```python
from maggie.utils.logging import LoggingManager

config = {
    'logging': {
        'path': 'logs',
        'console_level': 'INFO',
        'file_level': 'DEBUG'
    }
}
logging_manager = LoggingManager.initialize(config)
```

Notes
-----
The logging system has dependencies on other components of the Maggie AI Assistant,
but is designed to gracefully degrade if those components are not available. This makes
the logging system resilient to initialization order issues during application startup.

See Also
--------
- Python's logging module: https://docs.python.org/3/library/logging.html
- Context managers: https://docs.python.org/3/reference/datamodel.html#context-managers
- Decorators: https://docs.python.org/3/glossary.html#term-decorator
- Singleton pattern: https://en.wikipedia.org/wiki/Singleton_pattern
"""

import logging
import queue
import sys
import threading
import time
import uuid
import inspect
import functools
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set, Callable, Generator, TypeVar, cast, ContextManager, Protocol, overload, Literal

T = TypeVar('T')


class LogLevel(Enum):
    """
    Enumeration of log severity levels.
    
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
    Python's logging levels: https://docs.python.org/3/library/logging.html#logging-levels
    """
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class LogDestination(Enum):
    """
    Enumeration of possible log destinations.
    
    This enum provides a type-safe way to specify where log messages should be sent.
    The logging system supports multiple destinations simultaneously.
    
    Attributes
    ----------
    CONSOLE : Enum
        Output logs to the console (stdout).
    FILE : Enum
        Output logs to a file on disk.
    EVENT_BUS : Enum
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
    """
    Manages the logging system with enhanced capabilities via dependency injection.
    
    This class follows the singleton pattern and provides methods for logging messages
    with various levels of severity. It can be enhanced with event publishing, error handling,
    and state information capabilities when those dependencies become available.
    
    The class handles the initialization and configuration of Python's standard logging
    system, adding additional features like performance logging, correlation tracking,
    and state awareness.
    
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
        
    Methods
    -------
    get_instance()
        Get the singleton instance of LoggingManager.
    initialize(config)
        Initialize the LoggingManager with configuration settings.
    log(level, message, *args, exception=None, **kwargs)
        Log a message with the specified severity level.
    set_correlation_id(correlation_id)
        Set correlation ID for tracking related logs.
    get_correlation_id()
        Get the current correlation ID.
    clear_correlation_id()
        Clear the correlation ID.
    log_performance(component, operation, elapsed, details=None)
        Log performance metrics.
    log_state_transition(from_state, to_state, trigger)
        Log state transition.
    setup_global_exception_handler()
        Set up global exception handler.
        
    Examples
    --------
    >>> from maggie.utils.logging import LoggingManager
    >>> config = {'logging': {'path': 'logs', 'console_level': 'INFO'}}
    >>> logging_manager = LoggingManager.initialize(config)
    >>> logging_manager.log(LogLevel.INFO, "Application started")
    >>> logging_manager.set_correlation_id("user-request-123")
    >>> logging_manager.log(LogLevel.DEBUG, "Processing user request")
    >>> elapsed_time = 0.532  # seconds
    >>> logging_manager.log_performance("RequestHandler", "process_request", elapsed_time)
    >>> logging_manager.clear_correlation_id()
    
    Notes
    -----
    This class implements the Singleton pattern to ensure that there's only one instance
    of the logging manager throughout the application. This is important to maintain
    consistent logging behavior and configuration.
    
    See Also
    --------
    Singleton pattern: https://en.wikipedia.org/wiki/Singleton_pattern
    Python's logging module: https://docs.python.org/3/library/logging.html
    """
    
    _instance: Optional['LoggingManager'] = None
    _lock: threading.RLock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> 'LoggingManager':
        """
        Get the singleton instance of LoggingManager.
        
        This method implements the Singleton pattern, ensuring that only one instance
        of LoggingManager exists in the application.
        
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
        
        Notes
        -----
        This method should be called after LoggingManager.initialize() has been called.
        If get_instance() is called before initialize(), a RuntimeError is raised.
        
        See Also
        --------
        initialize : Initialize the LoggingManager with configuration settings
        """
        ...
    
    @classmethod
    def initialize(cls, config: Dict[str, Any]) -> 'LoggingManager':
        """
        Initialize the LoggingManager with configuration settings.
        
        This method configures and sets up the logging system according to the provided
        configuration dictionary. It creates the necessary log directories, sets up
        log handlers, and returns the singleton instance of LoggingManager.
        
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
        
        Notes
        -----
        If LoggingManager has already been initialized, this method returns the
        existing instance without reinitializing it, after logging a warning.
        """
        ...
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the logging utility with the provided configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing logging settings.
            
        Notes
        -----
        This constructor should not be called directly. Use LoggingManager.initialize()
        instead to properly set up the singleton instance.
        """
        ...
    
    def _configure_basic_logging(self) -> None:
        """
        Configure basic logging with standard Python logging module.
        
        This method sets up console and file logging handlers with appropriate
        formatters and attaches them to the root logger.
        
        Notes
        -----
        This is an internal method and should not be called directly.
        """
        ...
    
    def enhance_with_event_publisher(self, event_publisher: Any) -> None:
        """
        Enhance logging with an event publisher.
        
        This method adds event publishing capabilities to the logging system,
        allowing log messages to be published as events on the application's
        event bus.
        
        Parameters
        ----------
        event_publisher : Any
            An object implementing the IEventPublisher interface, which provides
            a publish() method for sending events.
            
        Examples
        --------
        >>> from maggie.core.event import EventBus
        >>> from maggie.utils.adapters import EventBusAdapter
        >>> event_bus = EventBus()
        >>> event_bus_adapter = EventBusAdapter(event_bus)
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.enhance_with_event_publisher(event_bus_adapter)
        """
        ...
    
    def enhance_with_error_handler(self, error_handler: Any) -> None:
        """
        Enhance logging with an error handler.
        
        This method adds error handling capabilities to the logging system,
        allowing errors to be properly recorded and tracked.
        
        Parameters
        ----------
        error_handler : Any
            An object implementing the IErrorHandler interface, which provides
            methods for recording and handling errors.
            
        Examples
        --------
        >>> from maggie.utils.error_handling import ErrorHandler
        >>> from maggie.utils.adapters import ErrorHandlerAdapter
        >>> error_handler = ErrorHandlerAdapter()
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.enhance_with_error_handler(error_handler)
        """
        ...
    
    def enhance_with_state_provider(self, state_provider: Any) -> None:
        """
        Enhance logging with a state provider.
        
        This method adds state awareness to the logging system, allowing
        log messages to include information about the current application state.
        
        Parameters
        ----------
        state_provider : Any
            An object implementing the IStateProvider interface, which provides
            a get_current_state() method for retrieving the current application state.
            
        Examples
        --------
        >>> from maggie.core.state import StateManager
        >>> from maggie.utils.adapters import StateManagerAdapter
        >>> state_manager = StateManager()
        >>> state_adapter = StateManagerAdapter(state_manager)
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.enhance_with_state_provider(state_adapter)
        """
        ...
    
    def log(self, level: LogLevel, message: str, *args: Any, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """
        Log a message with the specified severity level.
        
        This method logs a message with the given log level and optional exception
        information. The message can include format specifiers that will be replaced
        by the positional arguments.
        
        Parameters
        ----------
        level : LogLevel
            The severity level of the log message.
        message : str
            The message to log. Can contain format specifiers (%s, %d, etc.)
            that will be replaced by the positional arguments.
        *args : Any
            Positional arguments for message formatting.
        exception : Exception, optional
            An exception to include with the log message. If provided, the exception
            information (type, message, and traceback) will be included in the log.
        **kwargs : Any
            Additional key-value pairs to include with the log message. These can
            be used for structured logging or additional context.
            
        Examples
        --------
        >>> from maggie.utils.logging import LoggingManager, LogLevel
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.log(LogLevel.INFO, "Processing %s items", 5)
        >>> try:
        ...     result = 1 / 0
        ... except Exception as e:
        ...     logging_manager.log(LogLevel.ERROR, "Division error", exception=e)
        >>> logging_manager.log(LogLevel.DEBUG, "User context", user_id=123, action="login")
        """
        ...
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """
        Set correlation ID for tracking related logs.
        
        A correlation ID is a unique identifier that can be used to group related
        log messages together, making it easier to trace the flow of operations
        across different components or even different services.
        
        Parameters
        ----------
        correlation_id : str
            The correlation ID to set. This should be a unique identifier,
            typically a UUID or other string that can be used to correlate
            log messages.
            
        Examples
        --------
        >>> from maggie.utils.logging import LoggingManager, LogLevel
        >>> import uuid
        >>> logging_manager = LoggingManager.get_instance()
        >>> correlation_id = str(uuid.uuid4())
        >>> logging_manager.set_correlation_id(correlation_id)
        >>> logging_manager.log(LogLevel.INFO, "Starting request processing")
        >>> # All subsequent logs will include this correlation ID
        >>> logging_manager.log(LogLevel.DEBUG, "Processing request details")
        >>> logging_manager.clear_correlation_id()  # Don't forget to clear when done
        
        Notes
        -----
        The correlation ID is stored in the LoggingManager instance and will be
        included in all log messages until it is cleared or changed. It's important
        to clear the correlation ID when the related operation is complete to avoid
        incorrectly correlating unrelated log messages.
        
        See Also
        --------
        get_correlation_id : Get the current correlation ID
        clear_correlation_id : Clear the correlation ID
        """
        ...
    
    def get_correlation_id(self) -> Optional[str]:
        """
        Get the current correlation ID.
        
        Returns
        -------
        Optional[str]
            The current correlation ID, or None if no correlation ID is set.
            
        Examples
        --------
        >>> from maggie.utils.logging import LoggingManager
        >>> logging_manager = LoggingManager.get_instance()
        >>> current_id = logging_manager.get_correlation_id()
        >>> if current_id:
        ...     print(f"Current correlation ID: {current_id}")
        ... else:
        ...     print("No correlation ID is currently set")
        
        See Also
        --------
        set_correlation_id : Set correlation ID for tracking related logs
        clear_correlation_id : Clear the correlation ID
        """
        ...
    
    def clear_correlation_id(self) -> None:
        """
        Clear the correlation ID.
        
        This method removes the current correlation ID, if any. Subsequent log
        messages will not have a correlation ID until a new one is set.
        
        Examples
        --------
        >>> from maggie.utils.logging import LoggingManager
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.set_correlation_id("request-123")
        >>> # Do some logging with this correlation ID
        >>> logging_manager.clear_correlation_id()
        >>> # Subsequent logs will not have a correlation ID
        
        See Also
        --------
        set_correlation_id : Set correlation ID for tracking related logs
        get_correlation_id : Get the current correlation ID
        """
        ...
    
    def log_performance(self, component: str, operation: str, elapsed: float, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log performance metrics.
        
        This method logs performance information about an operation, including
        the elapsed time and optional details. This is useful for monitoring
        and profiling application performance.
        
        Parameters
        ----------
        component : str
            The name of the component or module that performed the operation.
        operation : str
            The name of the operation that was performed.
        elapsed : float
            The elapsed time in seconds for the operation.
        details : Dict[str, Any], optional
            Additional details about the operation, such as input parameters,
            result sizes, or other contextual information.
            
        Examples
        --------
        >>> from maggie.utils.logging import LoggingManager
        >>> import time
        >>> logging_manager = LoggingManager.get_instance()
        >>> 
        >>> # Measure performance of an operation
        >>> start_time = time.time()
        >>> result = process_large_dataset(dataset)
        >>> elapsed = time.time() - start_time
        >>> 
        >>> # Log the performance metrics
        >>> logging_manager.log_performance(
        ...     "DataProcessor",
        ...     "process_large_dataset",
        ...     elapsed,
        ...     details={
        ...         "dataset_size": len(dataset),
        ...         "result_items": len(result),
        ...         "batch_size": 1000
        ...     }
        ... )
        """
        ...
    
    def log_state_transition(self, from_state: Any, to_state: Any, trigger: str) -> None:
        """
        Log state transition.
        
        This method logs information about a state transition in the application,
        including the source state, destination state, and trigger that caused
        the transition.
        
        Parameters
        ----------
        from_state : Any
            The source state of the transition.
        to_state : Any
            The destination state of the transition.
        trigger : str
            The event or action that triggered the state transition.
            
        Examples
        --------
        >>> from maggie.utils.logging import LoggingManager
        >>> from maggie.core.state import State
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.log_state_transition(
        ...     State.IDLE,
        ...     State.ACTIVE,
        ...     "user_command"
        ... )
        
        Notes
        -----
        The state objects can be of any type, but they should have a `name` attribute
        or a reasonable string representation for proper logging.
        """
        ...
    
    def setup_global_exception_handler(self) -> None:
        """
        Set up global exception handler.
        
        This method replaces the default Python exception handler with a custom one
        that logs unhandled exceptions through the logging system. This ensures that
        all exceptions, even those not caught by try-except blocks, are properly logged.
        
        Examples
        --------
        >>> from maggie.utils.logging import LoggingManager
        >>> logging_manager = LoggingManager.get_instance()
        >>> logging_manager.setup_global_exception_handler()
        >>> # Now all unhandled exceptions will be logged via the logging system
        
        Notes
        -----
        This method should typically be called early in the application's initialization
        process to ensure that all unhandled exceptions are caught. It does not affect
        KeyboardInterrupt exceptions, which are passed through to the default handler.
        
        See Also
        --------
        Python's sys.excepthook: https://docs.python.org/3/library/sys.html#sys.excepthook
        """
        ...


class ComponentLogger:
    """
    A simplified component logger that doesn't depend on other modules.
    
    This class provides a component-specific logger that can be used independently
    of the main LoggingManager. It's designed to be easy to use and to gracefully
    fall back to basic logging when advanced features are not available.
    
    The ComponentLogger attempts to use LoggingManager if it's available, but
    falls back to Python's standard logging module if not. This makes it suitable
    for use in components that may be initialized before the LoggingManager.
    
    Attributes
    ----------
    component : str
        The name of the component this logger is associated with.
    logger : logging.Logger
        The underlying Python logger instance.
        
    Methods
    -------
    debug(message, **kwargs)
        Log a debug message.
    info(message, **kwargs)
        Log an info message.
    warning(message, **kwargs)
        Log a warning message.
    error(message, exception=None, **kwargs)
        Log an error message.
    critical(message, exception=None, **kwargs)
        Log a critical message.
    log_state_change(old_state, new_state, trigger)
        Log a state transition.
    log_performance(operation, elapsed, details=None)
        Log performance metrics.
        
    Examples
    --------
    >>> from maggie.utils.logging import ComponentLogger
    >>> 
    >>> # Create a logger for a specific component
    >>> logger = ComponentLogger("MyComponent")
    >>> 
    >>> # Log messages at different levels
    >>> logger.debug("Detailed information for debugging")
    >>> logger.info("General information about what's happening")
    >>> logger.warning("Something unexpected happened")
    >>> 
    >>> try:
    ...     result = 1 / 0
    ... except Exception as e:
    ...     logger.error("Division by zero", exception=e)
    >>> 
    >>> # Log performance metrics
    >>> import time
    >>> start_time = time.time()
    >>> result = process_data()
    >>> elapsed = time.time() - start_time
    >>> logger.log_performance("process_data", elapsed, {"data_points": 1000})
    
    Notes
    -----
    This class is designed to be used early in the application lifecycle, when
    the full logging infrastructure may not yet be available. It gracefully adds
    features as dependencies become available.
    """
    
    def __init__(self, component_name: str) -> None:
        """
        Initialize a component logger.
        
        Parameters
        ----------
        component_name : str
            The name of the component this logger is associated with. This will
            be included in all log messages to identify the source component.
            
        Examples
        --------
        >>> from maggie.utils.logging import ComponentLogger
        >>> logger = ComponentLogger("UserService")
        >>> logger.info("Service initialized")
        """
        ...
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log a debug message.
        
        Debug messages contain detailed information that is primarily useful for
        diagnosing problems. They are typically not shown in production environments
        unless debug mode is enabled.
        
        Parameters
        ----------
        message : str
            The debug message to log.
        **kwargs : Any
            Additional key-value pairs to include with the log message. Common
            keyword arguments include:
            - exception: An exception object to include with the log
            - correlation_id: A specific correlation ID for this log entry
            - user_id, session_id, etc.: Contextual information relevant to the log
            
        Examples
        --------
        >>> from maggie.utils.logging import ComponentLogger
        >>> logger = ComponentLogger("DataProcessor")
        >>> logger.debug("Processing file: %s", filename)
        >>> logger.debug("Parser state", parser_state="READING_HEADER", line_number=10)
        """
        ...
    
    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log an info message.
        
        Info messages confirm that things are working as expected. They provide
        general information about the application's operation.
        
        Parameters
        ----------
        message : str
            The info message to log.
        **kwargs : Any
            Additional key-value pairs to include with the log message. Common
            keyword arguments include:
            - exception: An exception object to include with the log
            - correlation_id: A specific correlation ID for this log entry
            - user_id, session_id, etc.: Contextual information relevant to the log
            
        Examples
        --------
        >>> from maggie.utils.logging import ComponentLogger
        >>> logger = ComponentLogger("UserService")
        >>> logger.info("User %s logged in successfully", username)
        >>> logger.info("Processing complete", items_processed=100, elapsed_time=1.5)
        """
        ...
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log a warning message.
        
        Warning messages indicate that something unexpected happened, or that a
        problem might occur in the future if action is not taken.
        
        Parameters
        ----------
        message : str
            The warning message to log.
        **kwargs : Any
            Additional key-value pairs to include with the log message. Common
            keyword arguments include:
            - exception: An exception object to include with the log
            - correlation_id: A specific correlation ID for this log entry
            - user_id, session_id, etc.: Contextual information relevant to the log
            
        Examples
        --------
        >>> from maggie.utils.logging import ComponentLogger
        >>> logger = ComponentLogger("NetworkService")
        >>> logger.warning("Connection attempt timed out, retrying...")
        >>> logger.warning("Memory usage high", usage_percent=85, action="Consider restarting")
        """
        ...
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """
        Log an error message.
        
        Error messages indicate that the software has not been able to perform
        some function due to a problem. This is more serious than a warning.
        
        Parameters
        ----------
        message : str
            The error message to log.
        exception : Exception, optional
            An exception object to include with the log. If provided, its type,
            message, and traceback will be included in the log.
        **kwargs : Any
            Additional key-value pairs to include with the log message. Common
            keyword arguments include:
            - correlation_id: A specific correlation ID for this log entry
            - user_id, session_id, etc.: Contextual information relevant to the log
            
        Examples
        --------
        >>> from maggie.utils.logging import ComponentLogger
        >>> logger = ComponentLogger("FileService")
        >>> try:
        ...     with open(filename, 'r') as f:
        ...         data = f.read()
        ... except FileNotFoundError as e:
        ...     logger.error("File not found", exception=e, filename=filename)
        >>> logger.error("Database connection failed", attempts=3, next_retry=30)
        """
        ...
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs: Any) -> None:
        """
        Log a critical message.
        
        Critical messages indicate serious errors where the program itself may be
        unable to continue running. This is the most severe type of log message.
        
        Parameters
        ----------
        message : str
            The critical message to log.
        exception : Exception, optional
            An exception object to include with the log. If provided, its type,
            message, and traceback will be included in the log.
        **kwargs : Any
            Additional key-value pairs to include with the log message. Common
            keyword arguments include:
            - correlation_id: A specific correlation ID for this log entry
            - user_id, session_id, etc.: Contextual information relevant to the log
            
        Examples
        --------
        >>> from maggie.utils.logging import ComponentLogger
        >>> logger = ComponentLogger("SystemService")
        >>> try:
        ...     # Critical operation
        ...     pass
        ... except Exception as e:
        ...     logger.critical("System failure", exception=e)
        ...     # Initiate system shutdown or failover
        >>> logger.critical("Database corrupted, initiating recovery", backup_used=True)
        """
        ...
    
    def log_state_change(self, old_state: Any, new_state: Any, trigger: str) -> None:
        """
        Log a state transition.
        
        This method logs information about a state transition related to this component,
        including the source state, destination state, and trigger that caused
        the transition.
        
        Parameters
        ----------
        old_state : Any
            The source state of the transition.
        new_state : Any
            The destination state of the transition.
        trigger : str
            The event or action that triggered the state transition.
            
        Examples
        --------
        >>> from maggie.utils.logging import ComponentLogger
        >>> from maggie.core.state import State
        >>> logger = ComponentLogger("StateMachine")
        >>> logger.log_state_change(
        ...     State.IDLE,
        ...     State.ACTIVE,
        ...     "user_command"
        ... )
        
        Notes
        -----
        The state objects can be of any type, but they should have a `name` attribute
        or a reasonable string representation for proper logging.
        """
        ...
    
    def log_performance(self, operation: str, elapsed: float, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log performance metrics.
        
        This method logs performance information about an operation performed by
        this component, including the elapsed time and optional details.
        
        Parameters
        ----------
        operation : str
            The name of the operation that was performed.
        elapsed : float
            The elapsed time in seconds for the operation.
        details : Dict[str, Any], optional
            Additional details about the operation, such as input parameters,
            result sizes, or other contextual information.
            
        Examples
        --------
        >>> from maggie.utils.logging import ComponentLogger
        >>> import time
        >>> logger = ComponentLogger("DatabaseService")
        >>> 
        >>> # Measure performance of a database query
        >>> start_time = time.time()
        >>> results = db.execute_query(query)
        >>> elapsed = time.time() - start_time
        >>> 
        >>> # Log the performance metrics
        >>> logger.log_performance(
        ...     "execute_query",
        ...     elapsed,
        ...     details={
        ...         "query_type": "SELECT",
        ...         "result_count": len(results),
        ...         "with_joins": True
        ...     }
        ... )
        """
        ...


@contextmanager
def logging_context(correlation_id: Optional[str] = None, component: str = '', operation: str = '', state: Any = None) -> Generator[Dict[str, Any], None, None]:
    """
    A context manager for structured logging with correlation tracking.
    
    This context manager creates a logging context with a correlation ID, component name,
    operation name, and optional state information. It tracks the start and end time of
    the context, logs performance metrics when the context exits, and ensures that the
    correlation ID is properly managed.
    
    Parameters
    ----------
    correlation_id : str, optional
        A unique identifier for correlating related log messages. If not provided,
        a new UUID will be generated.
    component : str, optional
        The name of the component or module performing the operation.
    operation : str, optional
        The name of the operation being performed.
    state : Any, optional
        The current application state, if relevant. If provided, state information
        will be included in log messages.
        
    Yields
    ------
    Dict[str, Any]
        A context dictionary that can be used to store additional information about
        the operation. This dictionary includes the following keys:
        - correlation_id: The correlation ID for this context
        - component: The component name
        - operation: The operation name
        - start_time: The time when the context was entered
        State information will be added if state is provided.
        
    Examples
    --------
    >>> from maggie.utils.logging import logging_context
    >>> 
    >>> def process_request(request_id, data):
    ...     with logging_context(
    ...         correlation_id=request_id,
    ...         component="RequestProcessor",
    ...         operation="process_request"
    ...     ) as ctx:
    ...         # First step
    ...         ctx['step'] = 'validate'
    ...         validate_data(data)
    ...         
    ...         # Second step
    ...         ctx['step'] = 'transform'
    ...         result = transform_data(data)
    ...         
    ...         # Third step
    ...         ctx['step'] = 'store'
    ...         store_result(result)
    ...         
    ...         return result
    
    Notes
    -----
    This context manager automatically logs performance metrics when the context exits,
    including the elapsed time for the entire operation. It also properly manages the
    correlation ID, restoring the previous correlation ID when the context exits.
    
    If an exception occurs within the context, it will be logged with the appropriate
    context information and then re-raised.
    
    See Also
    --------
    LoggingManager.set_correlation_id : Set correlation ID for tracking related logs
    LoggingManager.log_performance : Log performance metrics
    """
    ...


def log_operation(component: str = '', log_args: bool = True, log_result: bool = False, include_state: bool = True) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator for logging function operations with detailed information.
    
    This decorator logs information about function calls, including arguments,
    return values, and performance metrics. It automatically creates a logging
    context for the duration of the function call.
    
    Parameters
    ----------
    component : str, optional
        The name of the component or module that contains the decorated function.
        If not provided, the module name of the decorated function will be used.
    log_args : bool, optional
        Whether to log the function arguments. Default is True.
    log_result : bool, optional
        Whether to log the function return value. Default is False.
    include_state : bool, optional
        Whether to include application state information in the logs. Default is True.
        
    Returns
    -------
    Callable[[Callable[..., T]], Callable[..., T]]
        A decorator function that will wrap the target function with logging capabilities.
        
    Examples
    --------
    >>> from maggie.utils.logging import log_operation
    >>> 
    >>> @log_operation(component="UserService", log_args=True, log_result=True)
    ... def get_user(user_id):
    ...     # Function implementation
    ...     return user
    >>> 
    >>> # Using the decorator without arguments uses default values
    >>> @log_operation()
    ... def update_user(user_id, data):
    ...     # Function implementation
    ...     return success
    
    Notes
    -----
    This decorator uses the logging_context context manager internally to create
    a structured logging context for the function call. It also automatically logs
    performance metrics when the function completes.
    
    If the decorated function raises an exception, it will be logged with the
    appropriate context information and then re-raised.
    
    The decorator attempts to be intelligent about logging arguments and return values:
    - For primitive types (str, int, float, bool, None), it logs the actual value
    - For complex types, it logs the type name and sometimes basic information
    - It truncates long argument strings to avoid overwhelming the logs
    
    See Also
    --------
    logging_context : A context manager for structured logging
    """
    ...