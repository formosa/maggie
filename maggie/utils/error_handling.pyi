from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

T = TypeVar('T')

class ErrorSeverity(Enum):
    """
    Enumeration of error severity levels for categorizing and handling errors.

    Notes
    -----
    The severity levels are hierarchical, with higher levels indicating more critical issues:
    - DEBUG: Lowest severity, used for detailed diagnostic information
    - INFO: Informational messages about system state
    - WARNING: Potential issues that don't prevent operation but require attention
    - ERROR: Significant problems that may impact functionality
    - CRITICAL: Severe errors that may cause system failure or require immediate intervention

    Examples
    --------
    >>> severity = ErrorSeverity.ERROR
    >>> print(severity.name)  # Outputs: 'ERROR'
    >>> print(severity.value)  # Outputs: 3
    """
    DEBUG: int
    INFO: int
    WARNING: int
    ERROR: int
    CRITICAL: int

class ErrorCategory(Enum):
    """
    Enumeration of error categories to classify different types of errors in the system.

    Notes
    -----
    Provides a standardized way to categorize errors across the Maggie AI Assistant:
    - SYSTEM: Fundamental system-level errors
    - NETWORK: Connectivity or communication-related errors
    - RESOURCE: Resource allocation, management, or exhaustion errors
    - PERMISSION: Access control or authorization issues
    - CONFIGURATION: Configuration-related errors
    - INPUT: User input or data input errors
    - PROCESSING: Errors during data or computational processing
    - MODEL: Machine learning model-related errors
    - EXTENSION: Errors in extension modules
    - UNKNOWN: Catch-all for unclassified errors

    Examples
    --------
    >>> category = ErrorCategory.NETWORK
    >>> print(category.name)  # Outputs: 'NETWORK'
    >>> print(category.value)  # Outputs: 'network'
    """
    SYSTEM: str
    NETWORK: str
    RESOURCE: str
    PERMISSION: str
    CONFIGURATION: str
    INPUT: str
    PROCESSING: str
    MODEL: str
    EXTENSION: str
    UNKNOWN: str

class ErrorContext:
    """
    Comprehensive error context container for detailed error information and tracking.

    Parameters
    ----------
    message : str
        A human-readable description of the error
    exception : Optional[Exception], optional
        The original exception object, if applicable
    category : ErrorCategory, optional
        Categorization of the error type (default: ErrorCategory.UNKNOWN)
    severity : ErrorSeverity, optional
        Severity level of the error (default: ErrorSeverity.ERROR)
    source : str, optional
        Source component or module where the error occurred
    details : Dict[str, Any], optional
        Additional contextual details about the error
    correlation_id : Optional[str], optional
        Unique identifier for tracing and correlating error events

    Attributes
    ----------
    message : str
        Error description
    exception : Optional[Exception]
        Original exception object
    category : ErrorCategory
        Error categorization
    severity : ErrorSeverity
        Error severity level
    source : str
        Error source component
    details : Dict[str, Any]
        Additional error context
    correlation_id : str
        Unique error event identifier
    timestamp : float
        Timestamp of error creation

    Notes
    -----
    Provides a comprehensive error context for logging, monitoring, and debugging.
    Supports rich error information capture and event correlation.

    Examples
    --------
    >>> error = ErrorContext(
    ...     message="Database connection failed",
    ...     category=ErrorCategory.NETWORK,
    ...     severity=ErrorSeverity.CRITICAL
    ... )
    >>> print(error.message)
    'Database connection failed'
    """
    message: str
    exception: Optional[Exception]
    category: ErrorCategory
    severity: ErrorSeverity
    source: str
    details: Dict[str, Any]
    correlation_id: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error context to a dictionary representation.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing error context details

        Examples
        --------
        >>> error = ErrorContext(message="Test error")
        >>> error_dict = error.to_dict()
        >>> isinstance(error_dict, dict)
        True
        """
        ...

class ErrorRegistry:
    """
    Centralized registry for managing and creating standardized error definitions.

    Notes
    -----
    Provides a mechanism to:
    - Register predefined error types
    - Create consistent error contexts
    - Manage error code and message templates

    Methods allow dynamic error registration and context creation with 
    standardized categorization and severity levels.

    Examples
    --------
    >>> registry = ErrorRegistry.get_instance()
    >>> registry.register_error(
    ...     'CONFIG_LOAD_ERROR', 
    ...     'Failed to load configuration: {details}',
    ...     ErrorCategory.CONFIGURATION
    ... )
    """
    @classmethod
    def get_instance(cls) -> 'ErrorRegistry':
        """
        Retrieve the singleton instance of ErrorRegistry.

        Returns
        -------
        ErrorRegistry
            The singleton ErrorRegistry instance
        """
        ...

    def register_error(
        self, 
        code: str, 
        message_template: str, 
        category: ErrorCategory = ErrorCategory.UNKNOWN, 
        severity: ErrorSeverity = ErrorSeverity.ERROR
    ) -> None:
        """
        Register a new error type in the error registry.

        Parameters
        ----------
        code : str
            Unique error code identifier
        message_template : str
            Error message template with placeholders
        category : ErrorCategory, optional
            Error categorization (default: UNKNOWN)
        severity : ErrorSeverity, optional
            Error severity level (default: ERROR)
        """
        ...

    def create_error(
        self, 
        code: str, 
        details: Optional[Dict[str, Any]] = None, 
        exception: Optional[Exception] = None, 
        source: str = '', 
        severity_override: Optional[ErrorSeverity] = None
    ) -> ErrorContext:
        """
        Create an error context based on a registered error code.

        Parameters
        ----------
        code : str
            Registered error code
        details : Optional[Dict[str, Any]], optional
            Additional error details for message formatting
        exception : Optional[Exception], optional
            Original exception object
        source : str, optional
            Source component of the error
        severity_override : Optional[ErrorSeverity], optional
            Override the default severity level

        Returns
        -------
        ErrorContext
            Constructed error context object
        """
        ...

def get_event_bus() -> Optional[Any]:
    """
    Retrieve the global event bus for error event publishing.

    Returns
    -------
    Optional[Any]
        Event bus instance if available, None otherwise

    Notes
    -----
    Attempts to retrieve the event bus through the ServiceLocator.
    Used for publishing error events across the application.
    """
    ...

def safe_execute(
    func: Callable[..., T],
    *args: Any,
    error_code: Optional[str] = None,
    default_return: Optional[T] = None,
    error_details: Dict[str, Any] = None,
    error_category: ErrorCategory = ErrorCategory.UNKNOWN,
    error_severity: ErrorSeverity = ErrorSeverity.ERROR,
    publish_error: bool = True,
    **kwargs: Any
) -> T:
    """
    Execute a function with comprehensive error handling and recovery.

    Parameters
    ----------
    func : Callable[..., T]
        Function to be executed
    *args : Any
        Positional arguments for the function
    error_code : Optional[str], optional
        Specific error code for registered error types
    default_return : Optional[T], optional
        Default return value if execution fails
    error_details : Dict[str, Any], optional
        Additional error context details
    error_category : ErrorCategory, optional
        Error categorization (default: UNKNOWN)
    error_severity : ErrorSeverity, optional
        Error severity level (default: ERROR)
    publish_error : bool, optional
        Whether to publish error events (default: True)
    **kwargs : Any
        Keyword arguments for the function

    Returns
    -------
    T
        Function return value or default_return on failure

    Notes
    -----
    Provides a robust error handling wrapper that:
    - Catches and logs exceptions
    - Publishes error events
    - Allows configurable error handling behavior
    - Supports default return values

    Examples
    --------
    >>> def risky_function(x, y):
    ...     return x / y
    >>> result = safe_execute(
    ...     risky_function, 
    ...     10, 
    ...     0, 
    ...     error_code='DIVISION_ERROR',
    ...     default_return=0
    ... )
    >>> result  # Will be 0 due to division by zero
    0
    """
    ...

def retry_operation(
    max_attempts: int = 3,
    retry_delay: float = 1.,
    exponential_backoff: bool = True,
    jitter: bool = True,
    allowed_exceptions: Tuple[Exception, ...] = (Exception,),
    on_retry_callback: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    Decorator for implementing retry logic with configurable retry strategies.

    Parameters
    ----------
    max_attempts : int, optional
        Maximum number of execution attempts (default: 3)
    retry_delay : float, optional
        Base delay between retry attempts (default: 1 second)
    exponential_backoff : bool, optional
        Whether to use exponential backoff (default: True)
    jitter : bool, optional
        Add randomness to retry delays (default: True)
    allowed_exceptions : Tuple[Exception, ...], optional
        Exceptions that trigger retry (default: all Exceptions)
    on_retry_callback : Optional[Callable[[Exception, int], None]], optional
        Callback function for retry events

    Returns
    -------
    Callable
        Decorated function with retry mechanism

    Notes
    -----
    Provides a flexible retry mechanism with features:
    - Exponential backoff
    - Jitter for distributed retry timing
    - Configurable exception handling
    - Optional retry callbacks

    Examples
    --------
    >>> @retry_operation(max_attempts=3, retry_delay=2)
    ... def unstable_network_call():
    ...     # Function with potential transient failures
    ...     pass
    """
    ...

def record_error(
    message: str,
    exception: Optional[Exception] = None,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    source: str = '',
    details: Optional[Dict[str, Any]] = None,
    publish: bool = True
) -> ErrorContext:
    """
    Record and optionally publish an error context.

    Parameters
    ----------
    message : str
        Error description message
    exception : Optional[Exception], optional
        Original exception object
    category : ErrorCategory, optional
        Error categorization (default: UNKNOWN)
    severity : ErrorSeverity, optional
        Error severity level (default: ERROR)
    source : str, optional
        Source component of the error
    details : Optional[Dict[str, Any]], optional
        Additional error context details
    publish : bool, optional
        Whether to publish the error event (default: True)

    Returns
    -------
    ErrorContext
        Created error context object

    Notes
    -----
    Provides a convenient method to:
    - Create error contexts
    - Log errors
    - Optionally publish error events

    Examples
    --------
    >>> try:
    ...     # Some risky operation
    ...     raise ValueError("Example error")
    ... except ValueError as e:
    ...     error_ctx = record_error(
    ...         "Operation failed", 
    ...         e, 
    ...         category=ErrorCategory.PROCESSING
    ...     )
    """
    ...

# Custom Exception Classes
class LLMError(Exception):
    """Base exception for Language Model related errors."""
    ...

class ModelLoadError(LLMError):
    """Exception raised for errors during model loading."""
    ...

class GenerationError(LLMError):
    """Exception raised for errors during text generation."""
    ...

class STTError(Exception):
    """Base exception for Speech-to-Text related errors."""
    ...

class TTSError(Exception):
    """Base exception for Text-to-Speech related errors."""
    ...

class ExtensionError(Exception):
    """Base exception for extension-related errors."""
    ...

# Error Event Constants
ERROR_EVENT_LOGGED: str
ERROR_EVENT_COMPONENT_FAILURE: str
ERROR_EVENT_RESOURCE_WARNING: str
ERROR_EVENT_SYSTEM_ERROR: str

def with_error_handling(
    error_code: Optional[str] = None,
    error_category: ErrorCategory = ErrorCategory.UNKNOWN,
    error_severity: ErrorSeverity = ErrorSeverity.ERROR,
    publish_error: bool = True
) -> Callable:
    """
    Decorator for adding standardized error handling to functions.

    Parameters
    ----------
    error_code : Optional[str], optional
        Specific error code for registered error types
    error_category : ErrorCategory, optional
        Error categorization (default: UNKNOWN)
    error_severity : ErrorSeverity, optional
        Error severity level (default: ERROR)
    publish_error : bool, optional
        Whether to publish error events (default: True)

    Returns
    -------
    Callable
        Decorated function with standardized error handling

    Notes
    -----
    Provides a decorator that wraps function calls with:
    - Standardized error context creation
    - Logging
    - Optional error event publishing

    Examples
    --------
    >>> @with_error_handling(
    ...     error_code='MODEL_INIT_ERROR',
    ...     error_category=ErrorCategory.MODEL
    ... )
    ... def initialize_model():
    ...     # Model initialization logic
    ...     pass
    """
    ...