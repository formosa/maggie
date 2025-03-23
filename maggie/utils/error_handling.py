# New file: maggie/utils/error_handling.py
"""
Maggie AI Assistant - Error Handling Utilities
=============================================

Standardized error handling utilities for Maggie AI Assistant.

This module provides consistent patterns for error handling, logging,
and recovery to improve robustness across the application.
"""

import sys
import traceback
import logging
from typing import Any, Callable, Optional, TypeVar, Dict

# Type variable for generic function return type
T = TypeVar('T')

# Configure module logger
logger = logging.getLogger(__name__)

def safe_execute(func: Callable[..., T], 
                 *args: Any, 
                 default_return: Optional[T] = None,
                 error_message: str = "Error executing function",
                 critical: bool = False,
                 event_bus=None,
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
    event_bus : EventBus, optional
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
    >>> result = safe_execute(
    ...     complex_function, arg1, arg2, 
    ...     default_return=[], 
    ...     error_message="Error in data processing"
    ... )
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Get exception details
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Extract source file and line number
        tb = traceback.extract_tb(exc_traceback)
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
            event_bus.publish("error_logged", error_data)
        
        return default_return

def retry_operation(max_attempts: int = 3, 
                   retry_delay: float = 1.0,
                   exponential_backoff: bool = True,
                   allowed_exceptions: tuple = (Exception,)) -> Callable:
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
    allowed_exceptions : tuple, optional
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
    >>> @retry_operation(max_attempts=3)
    >>> def download_model(url):
    ...     # Function that might fail temporarily
    ...     return requests.get(url)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempt = 1
            last_exception = None
            
            while attempt <= max_attempts:
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
                    attempt += 1
                    
        return wrapper
    return decorator