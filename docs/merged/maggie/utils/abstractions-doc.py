from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable, Tuple, Union, Set, Type, TypeVar, cast
import threading

T = TypeVar('T')

class ILoggerProvider(ABC):
    """Abstract interface for logging providers.

    This interface defines the contract that any logging implementation
    must satisfy to be used within the Maggie AI system. It ensures that
    any logger can be used interchangeably, regardless of implementation.

    Notes
    -----
    The ILoggerProvider follows both the Strategy and Adapter patterns,
    allowing different logging backends (like standard library logging,
    loguru, etc.) to be used without affecting the rest of the codebase.

    See Also
    --------
    LoggingManagerAdapter : A concrete implementation adapter in maggie.utils.adapters

    Examples
    --------
    >>> class CustomLogger(ILoggerProvider):
    ...     def debug(self, message: str, **kwargs) -> None:
    ...         print(f"DEBUG: {message}")
    ...     
    ...     def info(self, message: str, **kwargs) -> None:
    ...         print(f"INFO: {message}")
    ...     
    ...     # Implement other required methods...
    """

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message.

        Parameters
        ----------
        message : str
            The debug message to log.
        **kwargs : dict, optional
            Additional parameters to include with the log message.
            Common parameters include:
            - exception: Exception object
            - component: str, name of the component logging the message
            - operation: str, specific operation being performed
            - correlation_id: str, ID for tracing related log entries
        """
        pass
    
    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log an informational message.

        Parameters
        ----------
        message : str
            The informational message to log.
        **kwargs : dict, optional
            Additional parameters to include with the log message.
            Common parameters include:
            - exception: Exception object
            - component: str, name of the component logging the message
            - operation: str, specific operation being performed
            - correlation_id: str, ID for tracing related log entries
        """
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message.

        Parameters
        ----------
        message : str
            The warning message to log.
        **kwargs : dict, optional
            Additional parameters to include with the log message.
            Common parameters include:
            - exception: Exception object
            - component: str, name of the component logging the message
            - operation: str, specific operation being performed
            - correlation_id: str, ID for tracing related log entries
        """
        pass
    
    @abstractmethod
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log an error message.

        Parameters
        ----------
        message : str
            The error message to log.
        exception : Exception, optional
            Associated exception object to include details from.
        **kwargs : dict, optional
            Additional parameters to include with the log message.
            Common parameters include:
            - component: str, name of the component logging the message
            - operation: str, specific operation being performed
            - correlation_id: str, ID for tracing related log entries
        """
        pass
    
    @abstractmethod
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log a critical message.

        Parameters
        ----------
        message : str
            The critical message to log.
        exception : Exception, optional
            Associated exception object to include details from.
        **kwargs : dict, optional
            Additional parameters to include with the log message.
            Common parameters include:
            - component: str, name of the component logging the message
            - operation: str, specific operation being performed
            - correlation_id: str, ID for tracing related log entries
        """
        pass

class IErrorHandler(ABC):
    """Abstract interface for error handling.

    This interface defines the contract that error handling components must
    implement. It provides a centralized way to record errors and safely execute
    functions with error handling, separating error handling logic from business logic.

    Notes
    -----
    The pattern used here is a combination of the Observer pattern (for error events)
    and the Command pattern (for safe execution).

    See Also
    --------
    ErrorHandlerAdapter : A concrete implementation adapter in maggie.utils.adapters

    Examples
    --------
    >>> class SimpleErrorHandler(IErrorHandler):
    ...     def record_error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> Any:
    ...         print(f"ERROR: {message}")
    ...         if exception:
    ...             print(f"Exception: {exception}")
    ...         return {"recorded": True}
    ...     
    ...     def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
    ...         try:
    ...             return func(*args, **kwargs)
    ...         except Exception as e:
    ...             self.record_error(f"Error executing function: {e}", e)
    ...             return kwargs.get('default_return', None)
    """

    @abstractmethod
    def record_error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> Any:
        """Record an error.

        Creates a structured error record and handles appropriate logging,
        event emission, and tracking as configured.

        Parameters
        ----------
        message : str
            Error message describing what went wrong.
        exception : Exception, optional
            Exception object associated with the error.
        **kwargs : dict, optional
            Additional error context parameters, commonly including:
            - category: ErrorCategory, type of error
            - severity: ErrorSeverity, how severe the error is
            - source: str, component or function that generated the error
            - details: dict, additional structured data about the error
            - publish: bool, whether to publish an error event
            - correlation_id: str, for correlating related errors

        Returns
        -------
        Any
            Implementation-specific return value, usually an error context object
            or error ID for referencing this error.
        """
        pass
    
    @abstractmethod
    def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Safely execute a function handling exceptions.

        Executes the provided function, handling any exceptions that might occur
        and returning a default value in case of failure.

        Parameters
        ----------
        func : callable
            The function to execute safely.
        *args : tuple
            Positional arguments to pass to the function.
        **kwargs : dict, optional
            Keyword arguments to pass to the function, including special
            error handling parameters:
            - error_code: str, identifier for this error scenario
            - default_return: Any, value to return if execution fails
            - error_details: dict, additional error context
            - error_category: ErrorCategory, type of error
            - error_severity: ErrorSeverity, how severe the error is
            - publish_error: bool, whether to publish an error event
            - include_state_info: bool, whether to include app state info

        Returns
        -------
        Any
            The result of the function call if successful, or the default_return
            value if an exception occurs.
        """
        pass

class IEventPublisher(ABC):
    """Abstract interface for event publishing.

    This interface defines the contract that event publishing components must
    implement. It enables pub/sub patterns throughout the application, promoting
    loose coupling between components.

    Notes
    -----
    The pattern used here is primarily the Observer pattern, allowing components
    to communicate without direct dependencies.

    See Also
    --------
    EventBusAdapter : A concrete implementation adapter in maggie.utils.adapters
    EventBus : Main implementation in maggie.core.event

    Examples
    --------
    >>> class SimpleEventPublisher(IEventPublisher):
    ...     def __init__(self):
    ...         self.subscribers = {}
    ...     
    ...     def publish(self, event_type: str, data: Any = None, **kwargs) -> None:
    ...         if event_type in self.subscribers:
    ...             for callback in self.subscribers[event_type]:
    ...                 callback(data)
    """

    @abstractmethod
    def publish(self, event_type: str, data: Any = None, **kwargs) -> None:
        """Publish an event.

        Distributes an event to all subscribers of the specified event type.

        Parameters
        ----------
        event_type : str
            Type/category of event to publish.
        data : Any, optional
            Event payload data to be sent to subscribers.
        **kwargs : dict, optional
            Additional parameters for event processing, including:
            - priority: int/enum, priority level for event processing
            - async: bool, whether to process event asynchronously
            - timeout: float, maximum time to wait for processing
            - headers: dict, metadata about the event
        """
        pass

class IStateProvider(ABC):
    """Abstract interface for state information.

    This interface defines the contract for components that provide state
    information about the application. It enables components to react to
    state changes without direct dependencies on the state management system.

    Notes
    -----
    The pattern used is a combination of the State pattern and Observer pattern,
    allowing components to query current state and react to state changes.

    See Also
    --------
    StateManagerAdapter : A concrete implementation adapter in maggie.utils.adapters
    StateManager : Main implementation in maggie.core.state

    Examples
    --------
    >>> class SimpleStateProvider(IStateProvider):
    ...     def __init__(self):
    ...         self.current_state = "IDLE"
    ...     
    ...     def get_current_state(self) -> Any:
    ...         return self.current_state
    """

    @abstractmethod
    def get_current_state(self) -> Any:
        """Get the current state.

        Retrieves the current state of the application or component.

        Returns
        -------
        Any
            The current state, typically a State enum value in Maggie's core system.
            The exact type depends on the implementation, but will typically be
            one of the State enum values defined in maggie.core.state.
        """
        pass

class LogLevel(Enum):
    """Enumeration of log severity levels.

    Defines standard log levels used throughout the Maggie AI system.
    These levels match the common logging levels in most logging systems
    and are organized in increasing order of severity.

    Attributes
    ----------
    DEBUG
        Detailed information, typically of interest only when diagnosing problems.
    INFO
        Confirmation that things are working as expected.
    WARNING
        Indication that something unexpected happened, or may happen in the near future.
    ERROR
        Due to a more serious problem, the software has not been able to perform a function.
    CRITICAL
        A serious error, indicating that the program itself may be unable to continue running.

    See Also
    --------
    logging.LogRecord : Standard library logging levels

    Examples
    --------
    >>> level = LogLevel.INFO
    >>> if level == LogLevel.INFO:
    ...     print("This is an informational message")
    """
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class ErrorCategory(Enum):
    """Enumeration of error categories.

    Categorizes errors by domain or subsystem to help with error grouping,
    filtering, and handling throughout the Maggie AI system.

    Attributes
    ----------
    SYSTEM
        Errors related to system operations, typically OS or environment issues.
    NETWORK
        Errors related to network connectivity, requests, or related operations.
    RESOURCE
        Errors related to resource allocation, like memory, CPU, GPU, etc.
    PERMISSION
        Errors related to insufficient permissions or authorization issues.
    CONFIGURATION
        Errors related to configuration issues, like invalid settings.
    INPUT
        Errors related to user input or data validation.
    PROCESSING
        Errors during data processing operations.
    MODEL
        Errors specific to AI model operations, like loading or inference.
    EXTENSION
        Errors from extension modules or plugins.
    STATE
        Errors related to application state management.
    UNKNOWN
        Default category for uncategorized errors.

    Examples
    --------
    >>> def classify_error(error: Exception) -> ErrorCategory:
    ...     if isinstance(error, ConnectionError):
    ...         return ErrorCategory.NETWORK
    ...     elif isinstance(error, MemoryError):
    ...         return ErrorCategory.RESOURCE
    ...     else:
    ...         return ErrorCategory.UNKNOWN
    """
    SYSTEM = auto()
    NETWORK = auto()
    RESOURCE = auto()
    PERMISSION = auto()
    CONFIGURATION = auto()
    INPUT = auto()
    PROCESSING = auto()
    MODEL = auto()
    EXTENSION = auto()
    STATE = auto()
    UNKNOWN = auto()

class ErrorSeverity(Enum):
    """Enumeration of error severity levels.

    Classifies errors by their impact level, helping to prioritize responses
    and determine appropriate handling strategies.

    Attributes
    ----------
    DEBUG
        Minor issues that don't affect operation but might be useful for debugging.
    INFO
        Informational events that don't require action but should be noted.
    WARNING
        Issues that don't prevent operation but might indicate problems.
    ERROR
        Significant problems that prevent specific operations from completing.
    CRITICAL
        Severe issues that might prevent the application from functioning.

    Examples
    --------
    >>> def determine_severity(error: Exception) -> ErrorSeverity:
    ...     if isinstance(error, KeyboardInterrupt):
    ...         return ErrorSeverity.INFO
    ...     elif isinstance(error, ValueError):
    ...         return ErrorSeverity.WARNING
    ...     elif isinstance(error, IOError):
    ...         return ErrorSeverity.ERROR
    ...     elif isinstance(error, MemoryError):
    ...         return ErrorSeverity.CRITICAL
    ...     else:
    ...         return ErrorSeverity.ERROR
    """
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class CapabilityRegistry:
    """Registry for application capabilities and services.

    This class implements the Service Locator pattern, providing a central registry
    for capability discovery and dependency injection. It helps to decouple
    component implementations from their consumers.

    Notes
    -----
    The registry is implemented as a thread-safe singleton with lazy initialization,
    ensuring consistent access throughout the application. This pattern is particularly
    useful in architectures where:
    - Components need to be swapped at runtime
    - Dependencies are resolved dynamically
    - Cross-cutting concerns (like logging) need consistent access

    See Also
    --------
    get_logger_provider : Helper function that uses this registry
    get_error_handler : Helper function that uses this registry
    get_event_publisher : Helper function that uses this registry
    get_state_provider : Helper function that uses this registry
    ServiceLocator : A more extensive implementation in maggie.service.locator

    Examples
    --------
    >>> registry = CapabilityRegistry.get_instance()
    >>> registry.register(ILoggerProvider, custom_logger)
    >>> logger = registry.get(ILoggerProvider)
    >>> logger.info("Using the registered logger")
    """
    _instance: Optional['CapabilityRegistry'] = None
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def get_instance(cls) -> 'CapabilityRegistry':
        """Get the singleton instance of the registry.

        This method implements the lazy initialization pattern, creating the
        singleton instance only when first needed and then reusing it for
        subsequent calls.

        Returns
        -------
        CapabilityRegistry
            The singleton instance of the capability registry.

        Notes
        -----
        This method is thread-safe due to the use of a reentrant lock during
        the singleton instance creation.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = CapabilityRegistry()
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the registry.

        Creates an empty registry. This should not be called directly;
        use get_instance() instead to access the singleton.
        """
        self._registry: Dict[Type, Any] = {}
    
    def register(self, capability_type: Type, instance: Any) -> None:
        """Register a capability instance.

        Associates a capability implementation with its interface type for later retrieval.
        If a capability of the same type was already registered, it will be replaced.

        Parameters
        ----------
        capability_type : Type
            The interface type (usually an ABC) that the instance implements.
        instance : Any
            The implementation instance to register.

        Examples
        --------
        >>> registry = CapabilityRegistry.get_instance()
        >>> registry.register(ILoggerProvider, my_logger_impl)
        >>> # Later, another component can retrieve this logger:
        >>> logger = registry.get(ILoggerProvider)
        """
        self._registry[capability_type] = instance
    
    def get(self, capability_type: Type[T]) -> Optional[T]:
        """Get a capability instance.

        Retrieves the registered implementation for the specified capability type.

        Parameters
        ----------
        capability_type : Type[T]
            The interface type to retrieve. This should be the same type that
            was used when registering the capability.

        Returns
        -------
        Optional[T]
            The registered instance if found, None otherwise.
            The return type T is determined by the capability_type parameter.

        Examples
        --------
        >>> registry = CapabilityRegistry.get_instance()
        >>> logger = registry.get(ILoggerProvider)
        >>> if logger:
        ...     logger.info("Using registered logger")
        ... else:
        ...     print("No logger registered")
        """
        return self._registry.get(capability_type)

def get_logger_provider() -> Optional[ILoggerProvider]:
    """Get the registered logger provider.

    Retrieves the currently registered logger provider from the capability registry.
    This is a convenience function that simplifies access to the logger provider
    without directly interacting with the registry.

    Returns
    -------
    Optional[ILoggerProvider]
        The registered logger provider if available, None otherwise.

    Examples
    --------
    >>> logger = get_logger_provider()
    >>> if logger:
    ...     logger.info("System initialized successfully")
    ... else:
    ...     print("Logging not available")
    """
    return CapabilityRegistry.get_instance().get(ILoggerProvider)

def get_error_handler() -> Optional[IErrorHandler]:
    """Get the registered error handler.

    Retrieves the currently registered error handler from the capability registry.
    This is a convenience function that simplifies access to the error handler
    without directly interacting with the registry.

    Returns
    -------
    Optional[IErrorHandler]
        The registered error handler if available, None otherwise.

    Examples
    --------
    >>> handler = get_error_handler()
    >>> if handler:
    ...     result = handler.safe_execute(risky_function, arg1, arg2)
    ... else:
    ...     try:
    ...         result = risky_function(arg1, arg2)
    ...     except Exception as e:
    ...         print(f"Error: {e}")
    ...         result = None
    """
    return CapabilityRegistry.get_instance().get(IErrorHandler)

def get_event_publisher() -> Optional[IEventPublisher]:
    """Get the registered event publisher.

    Retrieves the currently registered event publisher from the capability registry.
    This is a convenience function that simplifies access to the event publisher
    without directly interacting with the registry.

    Returns
    -------
    Optional[IEventPublisher]
        The registered event publisher if available, None otherwise.

    Examples
    --------
    >>> publisher = get_event_publisher()
    >>> if publisher:
    ...     publisher.publish("user_logged_in", {"user_id": "123"})
    """
    return CapabilityRegistry.get_instance().get(IEventPublisher)

def get_state_provider() -> Optional[IStateProvider]:
    """Get the registered state provider.

    Retrieves the currently registered state provider from the capability registry.
    This is a convenience function that simplifies access to the state provider
    without directly interacting with the registry.

    Returns
    -------
    Optional[IStateProvider]
        The registered state provider if available, None otherwise.

    Examples
    --------
    >>> state_provider = get_state_provider()
    >>> if state_provider:
    ...     current_state = state_provider.get_current_state()
    ...     print(f"Current application state: {current_state}")
    """
    return CapabilityRegistry.get_instance().get(IStateProvider)