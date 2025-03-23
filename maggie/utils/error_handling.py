# Updated maggie/utils/error_handling.py
"""
Maggie AI Assistant - Error Handling Utilities
=============================================

Standardized error handling utilities for Maggie AI Assistant.

This module provides consistent patterns for error handling, logging,
recovery, and event publication to improve robustness across the application.
"""

import sys
import traceback
import logging
from typing import Any, Callable, Optional, TypeVar, Dict, Union, List, Tuple, cast

# Type variable for generic function return type
T = TypeVar('T')

# Configure module logger
logger = logging.getLogger(__name__)

def safe_execute(func: Callable[..., T], 
                 *args: Any, 
                 default_return: Optional[T] = None,
                 error_message: str = "Error executing function",
                 critical: bool = False,
                 event_bus: Any = None,
                 **kwargs: Any) -> T:
    """
    Execute a function with standardized error handling.
    
    This function wraps another function call with try/except, logging,
    and optional event publication to provide consistent error handling.
    
    Parameters
    ----------
    func : Callable[..., T]
        Function to execute safely
    *args : Any
        Positional arguments to pass to the function
    default_return : Optional[T], optional
        Value to return on error, by default None
    error_message : str, optional
        Message prefix for error logs, by default "Error executing function"
    critical : bool, optional
        Whether this is a critical operation where errors should be elevated,
        by default False
    event_bus : Any, optional
        Event bus to publish errors to, by default None
    **kwargs : Any
        Keyword arguments to pass to the function
        
    Returns
    -------
    T
        Return value from function or default_return on error
        
    Notes
    -----
    This utility ensures consistent error handling patterns across
    the codebase, including proper logging of exceptions with source
    context and optional error event publication.
    
    Examples
    --------
    >>> # Simple usage
    >>> result = safe_execute(
    ...     complex_function, arg1, arg2, 
    ...     default_return=[], 
    ...     error_message="Error in data processing"
    ... )
    >>> 
    >>> # With event bus publication
    >>> from maggie.utils.service_locator import ServiceLocator
    >>> event_bus = ServiceLocator.get("event_bus")
    >>> result = safe_execute(
    ...     api_call, payload,
    ...     default_return={"status": "error"},
    ...     error_message="API call failed",
    ...     event_bus=event_bus
    ... )
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Get exception details
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Extract source file and line number
        tb = traceback.extract_tb(exc_traceback)
        if tb:
            filename, line, func_name, text = tb[-1]
            
            # Log the error with context
            error_detail = f"{error_message}: {e} in {filename}:{line} (function: {func_name})"
            
            if critical:
                logger.critical(error_detail)
            else:
                logger.error(error_detail)
            
            # Publish error event if event bus provided
            if event_bus:
                error_data = {
                    "message": str(e),
                    "source": filename,
                    "line": line,
                    "function": func_name,
                    "type": exc_type.__name__,
                    "context": error_message
                }
                try:
                    event_bus.publish("error_logged", error_data)
                except Exception as event_error:
                    logger.error(f"Failed to publish error event: {event_error}")
        else:
            # Simplified error logging if stack trace unavailable
            if critical:
                logger.critical(f"{error_message}: {e}")
            else:
                logger.error(f"{error_message}: {e}")
        
        return default_return if default_return is not None else cast(T, None)

def retry_operation(max_attempts: int = 3, 
                   retry_delay: float = 1.0,
                   exponential_backoff: bool = True,
                   allowed_exceptions: Tuple[Exception, ...] = (Exception,)) -> Callable:
    """
    Decorator for retrying operations that may fail temporarily.
    
    Parameters
    ----------
    max_attempts : int, optional
        Maximum number of attempts, by default 3
    retry_delay : float, optional
        Base delay between retries in seconds, by default 1.0
    exponential_backoff : bool, optional
        Whether to use exponential backoff for delays, by default True
    allowed_exceptions : Tuple[Exception, ...], optional
        Exception types to catch and retry, by default (Exception,)
        
    Returns
    -------
    Callable
        Decorated function with retry logic
        
    Notes
    -----
    This decorator is useful for operations that may fail due to
    temporary issues like network problems or resource contention.
    
    Examples
    --------
    >>> @retry_operation(max_attempts=3, retry_delay=2.0)
    >>> def download_model(url: str) -> bytes:
    ...     # Function that might fail temporarily
    ...     return requests.get(url).content
    >>>
    >>> @retry_operation(
    ...     max_attempts=5,
    ...     exponential_backoff=True,
    ...     allowed_exceptions=(ConnectionError, TimeoutError)
    ... )
    >>> def api_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    ...     response = requests.post(endpoint, json=data, timeout=10)
    ...     response.raise_for_status()
    ...     return response.json()
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        # Last attempt failed, re-raise or log
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                        raise
                    
                    # Calculate delay for next attempt
                    delay = retry_delay
                    if exponential_backoff:
                        delay = retry_delay * (2 ** (attempt - 1))
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} for {func.__name__} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    import time
                    time.sleep(delay)
            
            # This should never happen, but just in case
            if last_exception:
                raise last_exception
            return None
                
        return wrapper
    return decorator

def get_event_bus() -> Optional[Any]:
    """
    Get the event bus from ServiceLocator.
    
    This utility function provides a convenient way to get the event bus
    for error reporting from anywhere in the codebase.
    
    Returns
    -------
    Optional[Any]
        Event bus instance or None if not available
        
    Notes
    -----
    This function uses the ServiceLocator pattern to find the event bus
    instance, avoiding circular imports by importing the ServiceLocator
    only when the function is called.
    
    Examples
    --------
    >>> event_bus = get_event_bus()
    >>> if event_bus:
    ...     event_bus.publish("error_logged", {"message": "Something went wrong"})
    """
    try:
        from maggie.utils.service_locator import ServiceLocator
        return ServiceLocator.get("event_bus")
    except ImportError:
        logger.warning("ServiceLocator not available, can't get event_bus")
        return None
    except Exception as e:
        logger.error(f"Error getting event_bus: {e}")
        return None