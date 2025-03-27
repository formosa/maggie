"""
Maggie AI Assistant - Error Handling Utility
============================================

This module provides comprehensive error handling utilities for the Maggie AI Assistant.

The error handling system implements several design patterns to create a robust error
management framework:

- **Chain of Responsibility Pattern**: Errors are passed through a chain of handlers
  (logging, event publishing, fallback mechanisms) until properly addressed.
- **Decorator Pattern**: Functions like `with_error_handling` and `retry_operation` 
  decorate other functions with error handling capabilities.
- **Observer Pattern**: Errors trigger events that notify subscribers via the event bus.
- **Context Object Pattern**: The `ErrorContext` class encapsulates all information 
  about an error occurrence.

The module provides several key features:

1. Custom exception hierarchy for different error types
2. Error context object for capturing detailed error information
3. Utility functions for safe execution and error recording
4. Function decorators for automatic error handling and retry logic
5. Integration with the event bus system for error event propagation

References
----------
- Exception handling best practices: https://docs.python.org/3/tutorial/errors.html
- Decorator pattern: https://refactoring.guru/design-patterns/decorator/python/example
- Context managers: https://docs.python.org/3/reference/datamodel.html#context-managers
"""

import sys
import traceback
import logging
import time
import enum
import uuid
import functools
import threading
from typing import Any, Callable, Optional, TypeVar, Dict, Union, List, Tuple, cast, Type

T = TypeVar('T')

# Setup basic logging as fallback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('maggie.error_handling')

# Event constants for integration with event bus
ERROR_EVENT_LOGGED = 'error_logged'
"""Event name for when any error is logged."""

ERROR_EVENT_COMPONENT_FAILURE = 'component_failure'
"""Event name for when a component fails."""

ERROR_EVENT_RESOURCE_WARNING = 'resource_warning'
"""Event name for resource-related warnings."""

ERROR_EVENT_SYSTEM_ERROR = 'system_error'
"""Event name for system-level errors."""

ERROR_EVENT_STATE_TRANSITION = 'state_transition_error'
"""Event name for state transition errors."""

ERROR_EVENT_RESOURCE_MANAGEMENT = 'resource_management_error'
"""Event name for resource management errors."""

ERROR_EVENT_INPUT_PROCESSING = 'input_processing_error'
"""Event name for input processing errors."""

class ErrorSeverity(enum.Enum):
    """
    Enumeration of error severity levels.
    
    This enum provides standardized severity levels for classifying errors
    throughout the application, enabling consistent error handling, logging,
    and reporting.

    Attributes
    ----------
    DEBUG : int
        Low-level debug information about errors that is primarily useful for developers.
    INFO : int
        Informational messages about errors that don't require immediate attention.
    WARNING : int
        Errors that might cause issues but don't prevent the system from functioning.
    ERROR : int
        Significant errors that impact functionality but don't require system shutdown.
    CRITICAL : int
        Severe errors that may require immediate attention or system restart.

    Examples
    --------
    >>> severity = ErrorSeverity.WARNING
    >>> if severity == ErrorSeverity.WARNING:
    ...     print("This is a warning level error")
    This is a warning level error
    """
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class ErrorCategory(enum.Enum):
    """
    Enumeration of error categories.
    
    This enum provides standardized categories for classifying errors based on
    their source or nature, enabling more precise error handling, routing, and reporting.

    Attributes
    ----------
    SYSTEM : str
        Errors related to system-level operations (process, OS interactions).
    NETWORK : str
        Errors related to network operations (connections, API calls).
    RESOURCE : str
        Errors related to resource management (memory, CPU, GPU).
    PERMISSION : str
        Errors related to security, permissions, or authorization.
    CONFIGURATION : str
        Errors related to system configuration or settings.
    INPUT : str
        Errors related to user input or data validation.
    PROCESSING : str
        Errors related to data processing or transformation.
    MODEL : str
        Errors related to AI model loading or inference.
    EXTENSION : str
        Errors related to extensions or plugins.
    STATE : str
        Errors related to state management or transitions.
    UNKNOWN : str
        Errors that don't fit into any other category.

    Examples
    --------
    >>> error_type = ErrorCategory.NETWORK
    >>> if error_type == ErrorCategory.NETWORK:
    ...     print("This is a network-related error")
    This is a network-related error
    """
    SYSTEM = 'system'
    NETWORK = 'network'
    RESOURCE = 'resource'
    PERMISSION = 'permission'
    CONFIGURATION = 'configuration'
    INPUT = 'input'
    PROCESSING = 'processing'
    MODEL = 'model'
    EXTENSION = 'extension'
    STATE = 'state'
    UNKNOWN = 'unknown'

class MaggieError(Exception):
    """
    Base exception for all Maggie AI Assistant specific errors.
    
    This class serves as the root of the Maggie exception hierarchy, providing
    a way to catch all application-specific exceptions with a single except clause.
    All other custom exceptions in the application should inherit from this class.

    Parameters
    ----------
    *args : tuple
        Variable length argument list passed to the exception constructor.

    Examples
    --------
    >>> try:
    ...     raise MaggieError("Something went wrong")
    ... except MaggieError as e:
    ...     print(f"Caught error: {e}")
    Caught error: Something went wrong

    >>> class MyCustomError(MaggieError):
    ...     pass
    >>> try:
    ...     raise MyCustomError("Custom error")
    ... except MaggieError as e:
    ...     print(f"Caught custom error: {e}")
    Caught custom error: Custom error
    """
    pass

class LLMError(MaggieError):
    """
    Error related to language model operations.
    
    This exception is raised for errors that occur during language model
    interactions, such as text generation, model loading, or inference failures.

    Parameters
    ----------
    *args : tuple
        Variable length argument list passed to the exception constructor.

    Examples
    --------
    >>> try:
    ...     raise LLMError("Failed to generate text")
    ... except LLMError as e:
    ...     print(f"LLM error: {e}")
    LLM error: Failed to generate text
    """
    pass

class ModelLoadError(LLMError):
    """
    Error raised when loading a language model fails.
    
    This exception is specifically for errors that occur during model loading,
    such as file not found, invalid model format, or insufficient resources.

    Parameters
    ----------
    *args : tuple
        Variable length argument list passed to the exception constructor.

    Examples
    --------
    >>> try:
    ...     raise ModelLoadError("Failed to load model: insufficient VRAM")
    ... except ModelLoadError as e:
    ...     print(f"Model loading error: {e}")
    Model loading error: Failed to load model: insufficient VRAM
    """
    pass

class GenerationError(LLMError):
    """
    Error raised during text generation.
    
    This exception is specifically for errors that occur during text generation,
    such as context length exceeded, token limits, or inference failures.

    Parameters
    ----------
    *args : tuple
        Variable length argument list passed to the exception constructor.

    Examples
    --------
    >>> try:
    ...     raise GenerationError("Token limit exceeded during generation")
    ... except GenerationError as e:
    ...     print(f"Text generation error: {e}")
    Text generation error: Token limit exceeded during generation
    """
    pass

class STTError(MaggieError):
    """
    Error related to speech-to-text operations.
    
    This exception is raised for errors that occur during speech recognition,
    audio processing, or transcription operations.

    Parameters
    ----------
    *args : tuple
        Variable length argument list passed to the exception constructor.

    Examples
    --------
    >>> try:
    ...     raise STTError("Failed to transcribe audio")
    ... except STTError as e:
    ...     print(f"STT error: {e}")
    STT error: Failed to transcribe audio
    """
    pass

class TTSError(MaggieError):
    """
    Error related to text-to-speech operations.
    
    This exception is raised for errors that occur during speech synthesis,
    voice model loading, or audio playback operations.

    Parameters
    ----------
    *args : tuple
        Variable length argument list passed to the exception constructor.

    Examples
    --------
    >>> try:
    ...     raise TTSError("Failed to synthesize speech")
    ... except TTSError as e:
    ...     print(f"TTS error: {e}")
    TTS error: Failed to synthesize speech
    """
    pass

class ExtensionError(MaggieError):
    """
    Error related to extensions.
    
    This exception is raised for errors that occur within extension modules,
    such as initialization failures, runtime errors, or resource issues.

    Parameters
    ----------
    *args : tuple
        Variable length argument list passed to the exception constructor.

    Examples
    --------
    >>> try:
    ...     raise ExtensionError("Extension 'my_extension' failed to initialize")
    ... except ExtensionError as e:
    ...     print(f"Extension error: {e}")
    Extension error: Extension 'my_extension' failed to initialize
    """
    pass

class StateTransitionError(MaggieError):
    """
    Exception for state transition errors.
    
    This exception is raised when a state transition fails, such as when
    attempting to transition to an invalid state or with invalid parameters.

    Parameters
    ----------
    message : str
        The error message.
    from_state : Any, optional
        The current state before the attempted transition.
    to_state : Any, optional
        The target state of the attempted transition.
    trigger : str, optional
        The event or action that triggered the attempted transition.
    details : Dict[str, Any], optional
        Additional details about the transition error.

    Attributes
    ----------
    from_state : Any
        The current state before the attempted transition.
    to_state : Any
        The target state of the attempted transition.
    trigger : str
        The event or action that triggered the attempted transition.
    details : Dict[str, Any]
        Additional details about the transition error.

    Examples
    --------
    >>> try:
    ...     raise StateTransitionError(
    ...         "Invalid transition",
    ...         from_state="IDLE",
    ...         to_state="BUSY",
    ...         trigger="user_command"
    ...     )
    ... except StateTransitionError as e:
    ...     print(f"Transition error: {e}")
    ...     print(f"From: {e.from_state}, To: {e.to_state}")
    Transition error: Invalid transition
    From: IDLE, To: BUSY
    """
    def __init__(self, message: str, from_state: Any = None, to_state: Any = None, trigger: str = None,
                 details: Dict[str, Any] = None):
        self.from_state = from_state
        self.to_state = to_state
        self.trigger = trigger
        self.details = details or {}
        super().__init__(message)

class ResourceManagementError(MaggieError):
    """
    Exception for resource management errors.
    
    This exception is raised when there are issues managing system resources,
    such as memory allocation failures, GPU memory issues, or resource limits.

    Parameters
    ----------
    message : str
        The error message.
    resource_type : str, optional
        The type of resource involved (e.g., 'memory', 'gpu', 'cpu').
    resource_name : str, optional
        The specific resource name or identifier.
    details : Dict[str, Any], optional
        Additional details about the resource error.

    Attributes
    ----------
    resource_type : str
        The type of resource involved (e.g., 'memory', 'gpu', 'cpu').
    resource_name : str
        The specific resource name or identifier.
    details : Dict[str, Any]
        Additional details about the resource error.

    Examples
    --------
    >>> try:
    ...     raise ResourceManagementError(
    ...         "Insufficient resources",
    ...         resource_type="gpu",
    ...         resource_name="cuda:0",
    ...         details={"available_mb": 1024, "requested_mb": 2048}
    ...     )
    ... except ResourceManagementError as e:
    ...     print(f"Resource error: {e}")
    ...     print(f"Resource type: {e.resource_type}")
    Resource error: Insufficient resources
    Resource type: gpu
    """
    def __init__(self, message: str, resource_type: str = None, resource_name: str = None,
                 details: Dict[str, Any] = None):
        self.resource_type = resource_type
        self.resource_name = resource_name
        self.details = details or {}
        super().__init__(message)

class InputProcessingError(MaggieError):
    """
    Exception for input processing errors.
    
    This exception is raised when there are issues processing user input,
    such as invalid input format, failed parsing, or validation errors.

    Parameters
    ----------
    message : str
        The error message.
    input_type : str, optional
        The type of input being processed (e.g., 'voice', 'text', 'command').
    input_source : str, optional
        The source of the input (e.g., 'user', 'system', 'extension').
    details : Dict[str, Any], optional
        Additional details about the input processing error.

    Attributes
    ----------
    input_type : str
        The type of input being processed (e.g., 'voice', 'text', 'command').
    input_source : str
        The source of the input (e.g., 'user', 'system', 'extension').
    details : Dict[str, Any]
        Additional details about the input processing error.

    Examples
    --------
    >>> try:
    ...     raise InputProcessingError(
    ...         "Failed to parse input",
    ...         input_type="command",
    ...         input_source="user",
    ...         details={"raw_input": "invalid command"}
    ...     )
    ... except InputProcessingError as e:
    ...     print(f"Input error: {e}")
    ...     print(f"Input type: {e.input_type}")
    Input error: Failed to parse input
    Input type: command
    """
    def __init__(self, message: str, input_type: str = None, input_source: str = None,
                 details: Dict[str, Any] = None):
        self.input_type = input_type
        self.input_source = input_source
        self.details = details or {}
        super().__init__(message)

class ErrorContext:
    """
    Context information for errors.
    
    This class encapsulates all information related to an error occurrence,
    including the error message, exception object, category, severity, source,
    and other contextual details. It provides methods for converting to a
    dictionary representation and logging the error.

    Parameters
    ----------
    message : str
        The error message.
    exception : Exception, optional
        The exception object, if applicable.
    category : ErrorCategory, optional
        The error category. Defaults to ErrorCategory.UNKNOWN.
    severity : ErrorSeverity, optional
        The error severity level. Defaults to ErrorSeverity.ERROR.
    source : str, optional
        The source of the error (e.g., component or function name). Defaults to ''.
    details : Dict[str, Any], optional
        Additional details about the error.
    correlation_id : str, optional
        A unique ID for correlating related errors.
    state_info : Dict[str, Any], optional
        Information about the system state when the error occurred.

    Attributes
    ----------
    message : str
        The error message.
    exception : Exception, optional
        The exception object, if applicable.
    category : ErrorCategory
        The error category.
    severity : ErrorSeverity
        The error severity level.
    source : str
        The source of the error (e.g., component or function name).
    details : Dict[str, Any]
        Additional details about the error.
    correlation_id : str
        A unique ID for correlating related errors.
    timestamp : float
        The Unix timestamp when the error occurred.
    state_info : Dict[str, Any]
        Information about the system state when the error occurred.
    exception_type : str
        The type of the exception (if provided).
    exception_msg : str
        The exception message (if provided).
    filename : str
        The filename where the exception occurred (if provided).
    line : int
        The line number where the exception occurred (if provided).
    function : str
        The function name where the exception occurred (if provided).
    code : str
        The code snippet where the exception occurred (if provided).

    Examples
    --------
    >>> try:
    ...     1/0
    ... except Exception as e:
    ...     context = ErrorContext(
    ...         message="Division by zero error",
    ...         exception=e,
    ...         category=ErrorCategory.PROCESSING,
    ...         severity=ErrorSeverity.ERROR,
    ...         source="example_function",
    ...         details={"operation": "division"}
    ...     )
    ...     error_dict = context.to_dict()
    ...     print(f"Error message: {error_dict['message']}")
    ...     print(f"Error category: {error_dict['category']}")
    Error message: Division by zero error
    Error category: processing
    """
    def __init__(self, message: str, exception: Optional[Exception] = None, 
                 category: ErrorCategory = ErrorCategory.UNKNOWN, 
                 severity: ErrorSeverity = ErrorSeverity.ERROR, source: str = '',
                 details: Dict[str, Any] = None, correlation_id: Optional[str] = None,
                 state_info: Optional[Dict[str, Any]] = None):
        self.message = message
        self.exception = exception
        self.category = category
        self.severity = severity
        self.source = source
        self.details = details or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = time.time()
        self.state_info = state_info or {}
        
        if exception:
            self.exception_type = type(exception).__name__
            self.exception_msg = str(exception)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_traceback:
                tb = traceback.extract_tb(exc_traceback)
                if tb:
                    frame = tb[-1]
                    self.filename = frame.filename
                    self.line = frame.lineno
                    self.function = frame.name
                    self.code = frame.line
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error context to a dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing all error context information.

        Notes
        -----
        The 'exception' and 'location' keys are only included if an exception
        was provided. The 'details' and 'state' keys are only included if
        those attributes contain data.
        """
        result = {
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'source': self.source,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id
        }
        
        if hasattr(self, 'exception_type'):
            result['exception'] = {
                'type': self.exception_type,
                'message': self.exception_msg
            }
        
        if hasattr(self, 'filename'):
            result['location'] = {
                'file': self.filename,
                'line': self.line,
                'function': self.function,
                'code': self.code
            }
        
        if self.details:
            result['details'] = self.details
        
        if self.state_info:
            result['state'] = self.state_info
        
        return result
    
    def log(self, publish: bool = True) -> None:
        """
        Log the error.

        Parameters
        ----------
        publish : bool, optional
            Whether to publish an error event via the event bus. Defaults to True.

        Notes
        -----
        The method uses different logging levels based on severity and publishes
        to specific error channels based on category if publish is True.
        """
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(self.message, exc_info=bool(self.exception))
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(self.message, exc_info=bool(self.exception))
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(self.message)
        else:
            logger.debug(self.message)
        
        try:
            from maggie.utils.abstractions import get_event_publisher
            publisher = get_event_publisher()
            if publish and publisher:
                publisher.publish(ERROR_EVENT_LOGGED, self.to_dict())
                
                if self.category == ErrorCategory.STATE:
                    publisher.publish(ERROR_EVENT_STATE_TRANSITION, self.to_dict())
                elif self.category == ErrorCategory.RESOURCE:
                    publisher.publish(ERROR_EVENT_RESOURCE_MANAGEMENT, self.to_dict())
                elif self.category == ErrorCategory.INPUT:
                    publisher.publish(ERROR_EVENT_INPUT_PROCESSING, self.to_dict())
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to publish error event: {e}")

def safe_execute(func: Callable[..., T], *args: Any, error_code: Optional[str] = None,
                default_return: Optional[T] = None, error_details: Dict[str, Any] = None,
                error_category: ErrorCategory = ErrorCategory.UNKNOWN,
                error_severity: ErrorSeverity = ErrorSeverity.ERROR,
                publish_error: bool = True, include_state_info: bool = True, **kwargs: Any) -> T:
    """
    Safely execute a function, handling exceptions.

    Parameters
    ----------
    func : Callable[..., T]
        The function to execute.
    *args : Any
        Arguments to pass to the function.
    error_code : str, optional
        Optional error code for categorization.
    default_return : T, optional
        Value to return if an exception occurs.
    error_details : Dict[str, Any], optional
        Additional error details.
    error_category : ErrorCategory, optional
        Error category. Defaults to ErrorCategory.UNKNOWN.
    error_severity : ErrorSeverity, optional
        Error severity. Defaults to ErrorSeverity.ERROR.
    publish_error : bool, optional
        Whether to publish error events. Defaults to True.
    include_state_info : bool, optional
        Whether to include state information. Defaults to True.
    **kwargs : Any
        Keyword arguments to pass to the function.

    Returns
    -------
    T
        The result of the function or the default return value if an exception occurs.

    Examples
    --------
    >>> def divide(a, b):
    ...     return a / b
    >>> result = safe_execute(divide, 10, 0, default_return=0)
    >>> print(result)
    0
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        details = error_details or {}
        if not details:
            details = {'args': str(args), 'kwargs': str(kwargs)}
        
        source = f"{func.__module__}.{func.__name__}"
        
        current_state = None
        if include_state_info:
            try:
                from maggie.utils.abstractions import get_state_provider
                state_provider = get_state_provider()
                if state_provider:
                    current_state = state_provider.get_current_state()
            except ImportError:
                pass
            except Exception:
                pass
        
        context = ErrorContext(
            message=f"Error executing {func.__name__}: {e}",
            exception=e,
            category=error_category,
            severity=error_severity,
            source=source,
            details=details
        )
        
        if current_state is not None:
            context.state_info['current_state'] = current_state.name if hasattr(current_state, 'name') else str(current_state)
        
        context.log(publish=publish_error)
        return default_return if default_return is not None else cast(T, None)

def record_error(message: str, exception: Optional[Exception] = None, 
                category: ErrorCategory = ErrorCategory.UNKNOWN, 
                severity: ErrorSeverity = ErrorSeverity.ERROR, source: str = '', 
                details: Dict[str, Any] = None, publish: bool = True, state_object: Any = None,
                from_state: Any = None, to_state: Any = None, trigger: str = None) -> ErrorContext:
    """
    Record an error.

    Parameters
    ----------
    message : str
        Error message.
    exception : Exception, optional
        Exception object.
    category : ErrorCategory, optional
        Error category. Defaults to ErrorCategory.UNKNOWN.
    severity : ErrorSeverity, optional
        Error severity. Defaults to ErrorSeverity.ERROR.
    source : str, optional
        Error source. Defaults to ''.
    details : Dict[str, Any], optional
        Additional details.
    publish : bool, optional
        Whether to publish the error event. Defaults to True.
    state_object : Any, optional
        Current state object.
    from_state : Any, optional
        From state for transitions.
    to_state : Any, optional
        To state for transitions.
    trigger : str, optional
        Transition trigger.

    Returns
    -------
    ErrorContext
        The created ErrorContext object.

    Examples
    --------
    >>> context = record_error(
    ...     message="Failed to process input",
    ...     category=ErrorCategory.INPUT,
    ...     severity=ErrorSeverity.WARNING,
    ...     source="input_processor.validate"
    ... )
    """
    context = ErrorContext(
        message=message,
        exception=exception,
        category=category,
        severity=severity,
        source=source,
        details=details or {}
    )
    
    if state_object is not None:
        state_name = state_object.name if hasattr(state_object, 'name') else str(state_object)
        context.state_info['current_state'] = state_name
    
    if from_state is not None and to_state is not None:
        from_name = from_state.name if hasattr(from_state, 'name') else str(from_state)
        to_name = to_state.name if hasattr(to_state, 'name') else str(to_state)
        context.state_info['transition'] = {
            'from': from_name,
            'to': to_name,
            'trigger': trigger
        }
    
    context.log(publish=publish)
    return context

def with_error_handling(error_code: Optional[str] = None, 
                       error_category: ErrorCategory = ErrorCategory.UNKNOWN,
                       error_severity: ErrorSeverity = ErrorSeverity.ERROR,
                       publish_error: bool = True, include_state_info: bool = True):
    """
    Decorator for error handling.

    Parameters
    ----------
    error_code : str, optional
        Optional error code for categorization.
    error_category : ErrorCategory, optional
        Default error category. Defaults to ErrorCategory.UNKNOWN.
    error_severity : ErrorSeverity, optional
        Default error severity. Defaults to ErrorSeverity.ERROR.
    publish_error : bool, optional
        Whether to publish error events. Defaults to True.
    include_state_info : bool, optional
        Whether to include state information. Defaults to True.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @with_error_handling()
    ... def risky_function(a, b):
    ...     return a / b
    >>> result = risky_function(10, 0)  # Returns None instead of raising an exception
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return safe_execute(
                func, *args,
                error_code=error_code,
                error_category=error_category,
                error_severity=error_severity,
                publish_error=publish_error,
                include_state_info=include_state_info,
                **kwargs
            )
        return wrapper
    return decorator

def retry_operation(max_attempts: int = 3, retry_delay: float = 1.0, 
                   exponential_backoff: bool = True, jitter: bool = True,
                   allowed_exceptions: Tuple[Type[Exception], ...] = (Exception,),
                   on_retry_callback: Optional[Callable[[Exception, int], None]] = None,
                   error_category: ErrorCategory = ErrorCategory.UNKNOWN):
    """
    Decorator for retrying operations.

    Parameters
    ----------
    max_attempts : int, optional
        Maximum number of attempts. Defaults to 3.
    retry_delay : float, optional
        Base delay between retries in seconds. Defaults to 1.0.
    exponential_backoff : bool, optional
        Whether to use exponential backoff. Defaults to True.
    jitter : bool, optional
        Whether to add random jitter to retry delay. Defaults to True.
    allowed_exceptions : Tuple[Type[Exception], ...], optional
        Exceptions to retry on. Defaults to (Exception,).
    on_retry_callback : Callable[[Exception, int], None], optional
        Callback function for retry events.
    error_category : ErrorCategory, optional
        Error category for failures. Defaults to ErrorCategory.UNKNOWN.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @retry_operation(max_attempts=3)
    ... def fetch_data(url):
    ...     # Simulate network call
    ...     raise Exception("Network error")
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import random
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                        raise
                    
                    delay = retry_delay
                    if exponential_backoff:
                        delay = retry_delay * (2 ** (attempt - 1))
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    if on_retry_callback:
                        try:
                            on_retry_callback(e, attempt)
                        except Exception as callback_error:
                            logger.warning(f"Error in retry callback: {callback_error}")
                    
                    logger.warning(f"Attempt {attempt}/{max_attempts} for {func.__name__} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
            
            if last_exception:
                raise last_exception
            return None
        return wrapper
    return decorator

def create_state_transition_error(from_state: Any, to_state: Any, 
                                 trigger: str, details: Dict[str, Any] = None) -> StateTransitionError:
    """
    Create a state transition error.

    Parameters
    ----------
    from_state : Any
        Current state.
    to_state : Any
        Target state.
    trigger : str
        Transition trigger.
    details : Dict[str, Any], optional
        Additional details.

    Returns
    -------
    StateTransitionError
        StateTransitionError object.

    Examples
    --------
    >>> error = create_state_transition_error("IDLE", "BUSY", "user_command")
    >>> print(error)
    Invalid state transition: IDLE -> BUSY (trigger: user_command)
    """
    from_name = from_state.name if hasattr(from_state, 'name') else str(from_state)
    to_name = to_state.name if hasattr(to_state, 'name') else str(to_state)
    message = f"Invalid state transition: {from_name} -> {to_name} (trigger: {trigger})"
    
    record_error(
        message=message,
        category=ErrorCategory.STATE,
        severity=ErrorSeverity.ERROR,
        source='StateManager.transition_to',
        details=details or {},
        from_state=from_state,
        to_state=to_state,
        trigger=trigger
    )
    
    return StateTransitionError(
        message=message,
        from_state=from_state,
        to_state=to_state,
        trigger=trigger,
        details=details
    )