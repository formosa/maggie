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

# Error severity levels
class ErrorSeverity(enum.Enum):
    """
    Enumeration of error severity levels.
    
    This enum provides standardized severity levels for classifying errors
    throughout the application, enabling consistent error handling, logging,
    and reporting.
    
    Attributes
    ----------
    DEBUG : enum.auto
        Low-level debug information about errors that is primarily useful for developers.
    INFO : enum.auto
        Informational messages about errors that don't require immediate attention.
    WARNING : enum.auto
        Errors that might cause issues but don't prevent the system from functioning.
    ERROR : enum.auto
        Significant errors that impact functionality but don't require system shutdown.
    CRITICAL : enum.auto
        Severe errors that may require immediate attention or system restart.
        
    Examples
    --------
    >>> from maggie.utils.error_handling import ErrorSeverity
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
    >>> from maggie.utils.error_handling import ErrorCategory
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

# Event constants for integration with event bus
ERROR_EVENT_LOGGED: str
"""Event name for when any error is logged."""

ERROR_EVENT_COMPONENT_FAILURE: str
"""Event name for when a component fails."""

ERROR_EVENT_RESOURCE_WARNING: str
"""Event name for resource-related warnings."""

ERROR_EVENT_SYSTEM_ERROR: str
"""Event name for system-level errors."""

ERROR_EVENT_STATE_TRANSITION: str
"""Event name for state transition errors."""

ERROR_EVENT_RESOURCE_MANAGEMENT: str
"""Event name for resource management errors."""

ERROR_EVENT_INPUT_PROCESSING: str
"""Event name for input processing errors."""

# Base exception classes
class MaggieError(Exception):
    """
    Base exception for all Maggie AI Assistant specific errors.
    
    This class serves as the root of the Maggie exception hierarchy, providing
    a way to catch all application-specific exceptions with a single except clause.
    All other custom exceptions in the application should inherit from this class.
    
    Attributes
    ----------
    args : tuple
        Arguments passed to the exception constructor.
    
    Examples
    --------
    >>> from maggie.utils.error_handling import MaggieError
    >>> try:
    ...     raise MaggieError("Something went wrong")
    ... except MaggieError as e:
    ...     print(f"Caught error: {e}")
    Caught error: Something went wrong
    
    >>> # Create a custom error
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
    
    Attributes
    ----------
    args : tuple
        Arguments passed to the exception constructor.
    
    Examples
    --------
    >>> from maggie.utils.error_handling import LLMError
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
    
    Attributes
    ----------
    args : tuple
        Arguments passed to the exception constructor.
    
    Examples
    --------
    >>> from maggie.utils.error_handling import ModelLoadError
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
    
    Attributes
    ----------
    args : tuple
        Arguments passed to the exception constructor.
    
    Examples
    --------
    >>> from maggie.utils.error_handling import GenerationError
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
    
    Attributes
    ----------
    args : tuple
        Arguments passed to the exception constructor.
    
    Examples
    --------
    >>> from maggie.utils.error_handling import STTError
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
    
    Attributes
    ----------
    args : tuple
        Arguments passed to the exception constructor.
    
    Examples
    --------
    >>> from maggie.utils.error_handling import TTSError
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
    
    Attributes
    ----------
    args : tuple
        Arguments passed to the exception constructor.
    
    Examples
    --------
    >>> from maggie.utils.error_handling import ExtensionError
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
    >>> from maggie.utils.error_handling import StateTransitionError
    >>> from maggie.core.state import State
    >>> try:
    ...     raise StateTransitionError(
    ...         "Invalid transition",
    ...         from_state=State.IDLE,
    ...         to_state=State.BUSY,
    ...         trigger="user_command"
    ...     )
    ... except StateTransitionError as e:
    ...     print(f"Transition error: {e}")
    ...     print(f"From: {e.from_state}, To: {e.to_state}")
    Transition error: Invalid transition
    From: State.IDLE, To: State.BUSY
    """
    def __init__(self, message: str, from_state: Any = None, to_state: Any = None, trigger: str = None,
                 details: Dict[str, Any] = None) -> None: 
        """
        Initialize a StateTransitionError.
        
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
        """
        ...

class ResourceManagementError(MaggieError):
    """
    Exception for resource management errors.
    
    This exception is raised when there are issues managing system resources,
    such as memory allocation failures, GPU memory issues, or resource limits.
    
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
    >>> from maggie.utils.error_handling import ResourceManagementError
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
                 details: Dict[str, Any] = None) -> None: 
        """
        Initialize a ResourceManagementError.
        
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
        """
        ...

class InputProcessingError(MaggieError):
    """
    Exception for input processing errors.
    
    This exception is raised when there are issues processing user input,
    such as invalid input format, failed parsing, or validation errors.
    
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
    >>> from maggie.utils.error_handling import InputProcessingError
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
                 details: Dict[str, Any] = None) -> None: 
        """
        Initialize an InputProcessingError.
        
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
        """
        ...

class ErrorContext:
    """
    Context information for errors.
    
    This class encapsulates all information related to an error occurrence,
    including the error message, exception object, category, severity, source,
    and other contextual details. It provides methods for converting to a
    dictionary representation and logging the error.
    
    The ErrorContext implements a structured approach to error handling by:
    1. Capturing comprehensive error details in a standardized format
    2. Providing consistent conversion to dictionary format for serialization
    3. Integrating with the event bus for error event publishing
    4. Supporting correlation IDs for tracking related errors
    
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
        The type of the exception (available if exception is provided).
    exception_msg : str
        The exception message (available if exception is provided).
    filename : str
        The filename where the exception occurred (available if exception is provided).
    line : int
        The line number where the exception occurred (available if exception is provided).
    function : str
        The function name where the exception occurred (available if exception is provided).
    code : str
        The code snippet where the exception occurred (available if exception is provided).
    
    Examples
    --------
    >>> from maggie.utils.error_handling import ErrorContext, ErrorCategory, ErrorSeverity
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
                state_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize an ErrorContext.
        
        Parameters
        ----------
        message : str
            The error message.
        exception : Exception, optional
            The exception object, if applicable.
        category : ErrorCategory, default=ErrorCategory.UNKNOWN
            The error category.
        severity : ErrorSeverity, default=ErrorSeverity.ERROR
            The error severity level.
        source : str, default=''
            The source of the error (e.g., component or function name).
        details : Dict[str, Any], optional
            Additional details about the error.
        correlation_id : str, optional
            A unique ID for correlating related errors. If not provided,
            a new UUID will be generated.
        state_info : Dict[str, Any], optional
            Information about the system state when the error occurred.
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error context to a dictionary.
        
        This method creates a serializable dictionary representation of the
        error context, suitable for logging, event publishing, or persistence.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary containing all error context information, with the
            following structure:
            {
                'message': str,
                'category': str,
                'severity': str,
                'source': str,
                'timestamp': float,
                'correlation_id': str,
                'exception': {
                    'type': str,
                    'message': str
                },
                'location': {
                    'file': str,
                    'line': int,
                    'function': str,
                    'code': str
                },
                'details': Dict[str, Any],
                'state': Dict[str, Any]
            }
            
        Notes
        -----
        The 'exception' and 'location' keys are only included if an exception
        was provided. The 'details' and 'state' keys are only included if
        those attributes contain data.
        """
        ...
    
    def log(self, publish: bool = True) -> None:
        """
        Log the error.
        
        This method logs the error using the appropriate logging level based on
        the error severity, and optionally publishes an error event using the
        event bus if available.
        
        Parameters
        ----------
        publish : bool, default=True
            Whether to publish an error event via the event bus, if available.
            
        Notes
        -----
        The method attempts to publish the error event using the event bus if
        available, but gracefully handles the case where the event bus is not
        available or publishing fails.
        
        The method uses the following mapping from severity to logging level:
        - CRITICAL: logger.critical
        - ERROR: logger.error
        - WARNING: logger.warning
        - INFO/DEBUG: logger.debug
        
        In addition to publishing to the general ERROR_EVENT_LOGGED event,
        the method also publishes to specific error channels based on the
        error category:
        - ErrorCategory.STATE: ERROR_EVENT_STATE_TRANSITION
        - ErrorCategory.RESOURCE: ERROR_EVENT_RESOURCE_MANAGEMENT
        - ErrorCategory.INPUT: ERROR_EVENT_INPUT_PROCESSING
        """
        ...

def safe_execute(func: Callable[..., T], *args: Any, error_code: Optional[str] = None,
                default_return: Optional[T] = None, error_details: Dict[str, Any] = None,
                error_category: ErrorCategory = ErrorCategory.UNKNOWN,
                error_severity: ErrorSeverity = ErrorSeverity.ERROR,
                publish_error: bool = True, include_state_info: bool = True, **kwargs: Any) -> T:
    """
    Safely execute a function, handling exceptions.
    
    This utility function provides a standardized way to execute a function
    with comprehensive error handling. If an exception occurs, it captures
    detailed error information, logs the error, optionally publishes an error
    event, and returns a default value if provided.
    
    The function implements the "Circuit Breaker" design pattern by:
    1. Attempting to execute the wrapped function
    2. Catching and handling any exceptions that occur
    3. Returning a safe default value if execution fails
    4. Recording detailed error information for diagnosis
    
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
    error_category : ErrorCategory, default=ErrorCategory.UNKNOWN
        Error category.
    error_severity : ErrorSeverity, default=ErrorSeverity.ERROR
        Error severity.
    publish_error : bool, default=True
        Whether to publish error events.
    include_state_info : bool, default=True
        Whether to include state information.
    **kwargs : Any
        Keyword arguments to pass to the function.
        
    Returns
    -------
    T
        The result of the function or the default return value if an exception occurs.
        
    Examples
    --------
    >>> from maggie.utils.error_handling import safe_execute, ErrorCategory, ErrorSeverity
    >>> def divide(a, b):
    ...     return a / b
    >>> 
    >>> # Safe execution with default return value
    >>> result = safe_execute(
    ...     divide, 10, 0,
    ...     default_return=0,
    ...     error_category=ErrorCategory.PROCESSING,
    ...     error_severity=ErrorSeverity.WARNING
    ... )
    >>> print(result)
    0
    
    >>> # Safe execution with custom error details
    >>> result = safe_execute(
    ...     divide, 10, 0,
    ...     default_return=0,
    ...     error_details={"operation": "division", "inputs": {"a": 10, "b": 0}},
    ...     error_category=ErrorCategory.PROCESSING
    ... )
    >>> print(result)
    0
    
    Notes
    -----
    This function is particularly useful for operations that should not cause
    the application to crash, such as UI operations, non-critical background
    tasks, or operations with reasonable default behaviors.
    """
    ...

def record_error(message: str, exception: Optional[Exception] = None, 
                category: ErrorCategory = ErrorCategory.UNKNOWN, 
                severity: ErrorSeverity = ErrorSeverity.ERROR, source: str = '', 
                details: Dict[str, Any] = None, publish: bool = True, state_object: Any = None,
                from_state: Any = None, to_state: Any = None, trigger: str = None) -> ErrorContext:
    """
    Record an error.
    
    This function creates an ErrorContext for an error occurrence, logs the error,
    and optionally publishes an error event. It provides a standardized way to
    record errors throughout the application.
    
    Parameters
    ----------
    message : str
        Error message.
    exception : Exception, optional
        Exception object.
    category : ErrorCategory, default=ErrorCategory.UNKNOWN
        Error category.
    severity : ErrorSeverity, default=ErrorSeverity.ERROR
        Error severity.
    source : str, default=''
        Error source.
    details : Dict[str, Any], optional
        Additional details.
    publish : bool, default=True
        Whether to publish the error event.
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
    >>> from maggie.utils.error_handling import record_error, ErrorCategory, ErrorSeverity
    >>> 
    >>> # Basic error recording
    >>> context = record_error(
    ...     message="Failed to process input",
    ...     category=ErrorCategory.INPUT,
    ...     severity=ErrorSeverity.WARNING,
    ...     source="input_processor.validate"
    ... )
    >>> 
    >>> # Recording an exception
    >>> try:
    ...     1/0
    ... except Exception as e:
    ...     context = record_error(
    ...         message="Division operation failed",
    ...         exception=e,
    ...         category=ErrorCategory.PROCESSING,
    ...         source="calculator.divide"
    ...     )
    >>> 
    >>> # Recording a state transition error
    >>> from maggie.core.state import State
    >>> context = record_error(
    ...     message="Invalid state transition",
    ...     category=ErrorCategory.STATE,
    ...     source="state_manager.transition_to",
    ...     from_state=State.IDLE,
    ...     to_state=State.BUSY,
    ...     trigger="user_command"
    ... )
    
    Notes
    -----
    The function automatically enhances the error context with state information
    if provided, either via the state_object parameter or the from_state/to_state
    parameters for transition errors.
    """
    ...

def with_error_handling(error_code: Optional[str] = None, 
                       error_category: ErrorCategory = ErrorCategory.UNKNOWN,
                       error_severity: ErrorSeverity = ErrorSeverity.ERROR,
                       publish_error: bool = True, include_state_info: bool = True):
    """
    Decorator for error handling.
    
    This decorator wraps a function with standardized error handling using
    `safe_execute`. It provides a convenient way to add consistent error
    handling to multiple functions without duplicating code.
    
    The decorator pattern applied here allows for:
    1. Separation of error handling logic from business logic
    2. Consistent error handling across multiple functions
    3. Centralized configuration of error handling parameters
    
    Parameters
    ----------
    error_code : str, optional
        Optional error code for categorization.
    error_category : ErrorCategory, default=ErrorCategory.UNKNOWN
        Default error category.
    error_severity : ErrorSeverity, default=ErrorSeverity.ERROR
        Default error severity.
    publish_error : bool, default=True
        Whether to publish error events.
    include_state_info : bool, default=True
        Whether to include state information.
        
    Returns
    -------
    Callable
        Decorator function.
        
    Examples
    --------
    >>> from maggie.utils.error_handling import with_error_handling, ErrorCategory, ErrorSeverity
    >>> 
    >>> # Basic usage
    >>> @with_error_handling()
    ... def risky_function(a, b):
    ...     return a / b
    >>> 
    >>> result = risky_function(10, 0)  # Returns None instead of raising an exception
    >>> 
    >>> # With custom parameters
    >>> @with_error_handling(
    ...     error_code="MATH_ERROR",
    ...     error_category=ErrorCategory.PROCESSING,
    ...     error_severity=ErrorSeverity.WARNING
    ... )
    ... def divide(a, b):
    ...     return a / b
    >>> 
    >>> result = divide(10, 0)  # Returns None and logs a warning
    >>> 
    >>> # With default return value (via wrapper function)
    >>> @with_error_handling(error_category=ErrorCategory.PROCESSING)
    ... def safe_divide(a, b, default=0):
    ...     try:
    ...         return a / b
    ...     except ZeroDivisionError:
    ...         return default
    >>> 
    >>> result = safe_divide(10, 0, default=0)  # Returns 0
    >>> print(result)
    0
    
    Notes
    -----
    The wrapped function will return None if an exception occurs and no default
    return value is specified. To provide a default return value, you can either:
    1. Handle the exception within the wrapped function and return a default value
    2. Use `safe_execute` directly with the `default_return` parameter
    """
    ...

def retry_operation(max_attempts: int = 3, retry_delay: float = 1.0, 
                   exponential_backoff: bool = True, jitter: bool = True,
                   allowed_exceptions: Tuple[Type[Exception], ...] = (Exception,),
                   on_retry_callback: Optional[Callable[[Exception, int], None]] = None,
                   error_category: ErrorCategory = ErrorCategory.UNKNOWN):
    """
    Decorator for retrying operations.
    
    This decorator adds retry logic to a function, allowing it to automatically
    retry upon failure up to a specified number of times. It supports exponential
    backoff, jitter, and filtering of allowed exceptions.
    
    The retry mechanism implements best practices for resilient systems:
    1. Exponential backoff to avoid overloading systems under stress
    2. Jitter to prevent synchronized retry storms
    3. Selective retry based on exception types
    4. Callback hooks for monitoring and logging retry attempts
    
    Parameters
    ----------
    max_attempts : int, default=3
        Maximum number of attempts.
    retry_delay : float, default=1.0
        Base delay between retries in seconds.
    exponential_backoff : bool, default=True
        Whether to use exponential backoff.
    jitter : bool, default=True
        Whether to add random jitter to retry delay.
    allowed_exceptions : Tuple[Type[Exception], ...], default=(Exception,)
        Exceptions to retry on.
    on_retry_callback : Callable[[Exception, int], None], optional
        Callback function for retry events.
    error_category : ErrorCategory, default=ErrorCategory.UNKNOWN
        Error category for failures.
        
    Returns
    -------
    Callable
        Decorator function.
        
    Examples
    --------
    >>> import requests
    >>> from maggie.utils.error_handling import retry_operation, ErrorCategory
    >>> 
    >>> # Basic usage with default parameters (retry up to 3 times)
    >>> @retry_operation()
    ... def fetch_data(url):
    ...     response = requests.get(url)
    ...     response.raise_for_status()
    ...     return response.json()
    >>> 
    >>> # Custom retry parameters
    >>> @retry_operation(
    ...     max_attempts=5,
    ...     retry_delay=2.0,
    ...     exponential_backoff=True,
    ...     jitter=True,
    ...     allowed_exceptions=(requests.RequestException,),
    ...     error_category=ErrorCategory.NETWORK
    ... )
    ... def fetch_api_data(url):
    ...     response = requests.get(url, timeout=5)
    ...     response.raise_for_status()
    ...     return response.json()
    >>> 
    >>> # With retry callback
    >>> def log_retry(exception, attempt):
    ...     print(f"Retry attempt {attempt} after error: {exception}")
    >>> 
    >>> @retry_operation(
    ...     max_attempts=3, 
    ...     on_retry_callback=log_retry
    ... )
    ... def connect_to_database():
    ...     # Database connection code
    ...     pass
    
    Notes
    -----
    For exponential backoff, the delay between retries is calculated as:
    delay = retry_delay * (2 ** (attempt - 1))
    
    If jitter is enabled, the delay is further randomized by multiplying by
    a random factor between 0.5 and 1.5, which helps prevent retry storms
    when multiple clients are retrying simultaneously.
    
    References
    ----------
    - Exponential backoff: https://en.wikipedia.org/wiki/Exponential_backoff
    - Jitter for distributed systems: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    """
    ...

def create_state_transition_error(from_state: Any, to_state: Any, 
                                 trigger: str, details: Dict[str, Any] = None) -> StateTransitionError:
    """
    Create a state transition error.
    
    This function creates a StateTransitionError with the given parameters,
    records the error using `record_error`, and returns the error object.
    
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
    >>> from maggie.utils.error_handling import create_state_transition_error
    >>> from maggie.core.state import State
    >>> 
    >>> # Create a state transition error
    >>> error = create_state_transition_error(
    ...     from_state=State.IDLE,
    ...     to_state=State.BUSY,
    ...     trigger="user_command",
    ...     details={"user_id": "12345", "command": "process_data"}
    ... )
    >>> 
    >>> # Raise the error
    >>> try:
    ...     raise error
    ... except StateTransitionError as e:
    ...     print(f"Error: {e}")
    ...     print(f"From: {e.from_state}, To: {e.to_state}, Trigger: {e.trigger}")
    Error: Invalid state transition: IDLE -> BUSY (trigger: user_command)
    From: State.IDLE, To: State.BUSY, Trigger: user_command
    
    Notes
    -----
    This function automatically records the error using `record_error` with
    the appropriate category (ErrorCategory.STATE) and severity (ErrorSeverity.ERROR).
    
    The error message is automatically generated as:
    "Invalid state transition: {from_name} -> {to_name} (trigger: {trigger})"
    """
    ...