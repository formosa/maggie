"""
Maggie AI Assistant - Adapter Classes
=====================================

This module provides adapter implementations that bridge concrete implementations 
with abstract interfaces from the abstractions module, applying the Adapter design 
pattern to maintain loose coupling between system components.

The adapter pattern allows classes with incompatible interfaces to work together by 
wrapping an instance of one class with a new adapter class that implements the interface 
expected by clients. This pattern is essential for:

1. Decoupling system components by removing direct dependencies
2. Enabling modular architecture with interchangeable components
3. Facilitating unit testing through interface-based design
4. Supporting gradual evolution of the codebase by adapting legacy code

The adapters in this module connect core system components to the capability registry
system, allowing these components to be discovered and used through their respective
interfaces rather than concrete implementations.

See Also
--------
maggie.utils.abstractions : Contains the interface definitions adapted by these classes
https://refactoring.guru/design-patterns/adapter : Detailed explanation of the Adapter pattern

Notes
-----
Each adapter automatically registers itself with the CapabilityRegistry upon initialization,
making the adapted functionality available through the central registry system.
"""

from typing import Dict, Any, Optional, Callable, Type, TypeVar

# Import interfaces from abstractions
from maggie.utils.abstractions import (
    ILoggerProvider, 
    IErrorHandler, 
    IEventPublisher, 
    IStateProvider, 
    CapabilityRegistry,
    ErrorCategory,
    ErrorSeverity
)

# For typing
from maggie.utils.logging import LoggingManager, LogLevel
from maggie.core.state import State, StateManager


class LoggingManagerAdapter(ILoggerProvider):
    """
    Adapter that converts a LoggingManager instance to the ILoggerProvider interface.
    
    This adapter enables loose coupling between components that need logging 
    functionality and the concrete LoggingManager implementation. It forwards
    log calls to the appropriate methods on the wrapped LoggingManager instance
    and handles automatic registration with the capability registry.
    
    Parameters
    ----------
    logging_manager : LoggingManager
        The LoggingManager instance to adapt to the ILoggerProvider interface.
        
    Attributes
    ----------
    logging_manager : LoggingManager
        Reference to the adapted LoggingManager instance.
        
    Examples
    --------
    >>> from maggie.utils.logging import LoggingManager
    >>> from maggie.utils.adapters import LoggingManagerAdapter
    >>> logging_mgr = LoggingManager.initialize(config)
    >>> logging_adapter = LoggingManagerAdapter(logging_mgr)
    >>> # Now the adapter can be used through the ILoggerProvider interface
    >>> logging_adapter.info("This message will be logged")
    
    Notes
    -----
    This adapter automatically registers itself with the CapabilityRegistry
    upon initialization, making it discoverable through the ILoggerProvider
    interface.
    """
    
    def __init__(self, logging_manager: LoggingManager) -> None:
        """
        Initialize adapter with LoggingManager instance.
        
        Parameters
        ----------
        logging_manager : LoggingManager
            The LoggingManager instance to adapt to the ILoggerProvider interface.
        """
        self.logging_manager = logging_manager
        
        # Register self with the capability registry
        registry = CapabilityRegistry.get_instance()
        registry.register(ILoggerProvider, self)
    
    def debug(self, message: str, **kwargs) -> None:
        """
        Log a debug message.
        
        Parameters
        ----------
        message : str
            The debug message to log.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the logging system.
            Common keyword arguments include:
            - exception: Exception instance to include in the log
            - component: Component name for the log entry
            - operation: Operation name for the log entry
            
        Notes
        -----
        This method converts the call to a LogLevel.DEBUG log entry
        in the underlying LoggingManager.
        """
        from maggie.utils.logging import LogLevel
        self.logging_manager.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """
        Log an info message.
        
        Parameters
        ----------
        message : str
            The info message to log.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the logging system.
            See debug() method for common keyword arguments.
            
        Notes
        -----
        This method converts the call to a LogLevel.INFO log entry
        in the underlying LoggingManager.
        """
        from maggie.utils.logging import LogLevel
        self.logging_manager.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message.
        
        Parameters
        ----------
        message : str
            The warning message to log.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the logging system.
            See debug() method for common keyword arguments.
            
        Notes
        -----
        This method converts the call to a LogLevel.WARNING log entry
        in the underlying LoggingManager.
        """
        from maggie.utils.logging import LogLevel
        self.logging_manager.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """
        Log an error message.
        
        Parameters
        ----------
        message : str
            The error message to log.
        exception : Exception, optional
            Exception instance to include in the log.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the logging system.
            See debug() method for common keyword arguments.
            
        Notes
        -----
        This method converts the call to a LogLevel.ERROR log entry
        in the underlying LoggingManager. If an exception is provided,
        its traceback will be included in the log.
        """
        from maggie.utils.logging import LogLevel
        self.logging_manager.log(LogLevel.ERROR, message, exception=exception, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """
        Log a critical message.
        
        Parameters
        ----------
        message : str
            The critical message to log.
        exception : Exception, optional
            Exception instance to include in the log.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the logging system.
            See debug() method for common keyword arguments.
            
        Notes
        -----
        This method converts the call to a LogLevel.CRITICAL log entry
        in the underlying LoggingManager. Critical logs often trigger
        additional alerting mechanisms in the system.
        """
        from maggie.utils.logging import LogLevel
        self.logging_manager.log(LogLevel.CRITICAL, message, exception=exception, **kwargs)


class ErrorHandlerAdapter(IErrorHandler):
    """
    Adapter that implements the IErrorHandler interface.
    
    This adapter provides a standalone implementation of the IErrorHandler
    interface without requiring any wrapped object. It uses the error handling
    functions from maggie.utils.error_handling to implement the interface
    methods and registers itself with the capability registry.
    
    Unlike other adapters in this module, ErrorHandlerAdapter doesn't wrap
    an existing object but instead creates a new error handling capability
    by directly implementing the required interface methods.
    
    Examples
    --------
    >>> from maggie.utils.adapters import ErrorHandlerAdapter
    >>> error_handler = ErrorHandlerAdapter()
    >>> # Now the adapter can be used through the IErrorHandler interface
    >>> def risky_function():
    ...     # Some code that might raise an exception
    ...     raise ValueError("Something went wrong")
    >>> result = error_handler.safe_execute(
    ...     risky_function, 
    ...     error_category=ErrorCategory.PROCESSING,
    ...     default_return="Default value"
    ... )
    >>> print(result)  # Will print "Default value"
    
    Notes
    -----
    This adapter automatically registers itself with the CapabilityRegistry
    upon initialization, making it discoverable through the IErrorHandler
    interface.
    """
    
    def __init__(self) -> None:
        """
        Initialize adapter and register with capability registry.
        
        Notes
        -----
        This constructor doesn't take any parameters since it doesn't
        wrap an existing object. Instead, it directly implements the
        IErrorHandler interface using utility functions.
        """
        # Register self with the capability registry
        registry = CapabilityRegistry.get_instance()
        registry.register(IErrorHandler, self)
    
    def record_error(
        self, 
        message: str, 
        exception: Optional[Exception] = None, 
        **kwargs
    ) -> Any:
        """
        Record an error.
        
        Parameters
        ----------
        message : str
            Error message describing what went wrong.
        exception : Exception, optional
            Exception object if an exception was caught.
        **kwargs : dict, optional
            Additional parameters for error context:
            - category : ErrorCategory
                Type of error (system, network, etc.)
            - severity : ErrorSeverity
                How severe the error is (warning, error, critical)
            - source : str
                Source of the error (component/function name)
            - details : dict
                Additional contextual information about the error
            - publish : bool
                Whether to publish the error as an event
            
        Returns
        -------
        Any
            Return value from the error handling system, typically
            an ErrorContext object.
            
        Notes
        -----
        This method forwards the call to the record_error function in
        maggie.utils.error_handling. It extracts known parameters
        from kwargs and passes them as positional arguments.
        """
        from maggie.utils.error_handling import record_error as do_record_error
        from maggie.utils.error_handling import ErrorCategory, ErrorSeverity
        
        # Extract known kwargs
        category = kwargs.pop('category', ErrorCategory.UNKNOWN)
        severity = kwargs.pop('severity', ErrorSeverity.ERROR)
        source = kwargs.pop('source', '')
        details = kwargs.pop('details', None)
        publish = kwargs.pop('publish', True)
        
        return do_record_error(
            message=message,
            exception=exception,
            category=category,
            severity=severity,
            source=source,
            details=details,
            publish=publish,
            **kwargs
        )
    
    def safe_execute(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Safely execute a function, handling exceptions.
        
        Parameters
        ----------
        func : callable
            Function to execute safely.
        *args : tuple
            Positional arguments to pass to the function.
        **kwargs : dict
            Keyword arguments which may include:
            - error_code : str, optional
                Custom error code for tracking.
            - default_return : Any, optional
                Value to return if an exception occurs.
            - error_details : dict, optional
                Additional details about the error context.
            - error_category : ErrorCategory, optional
                Type of error (system, network, etc.)
            - error_severity : ErrorSeverity, optional
                How severe the error is (warning, error, critical)
            - publish_error : bool, optional
                Whether to publish error events.
            - include_state_info : bool, optional
                Whether to include state information in error.
            
        Returns
        -------
        Any
            Return value from the function or the default return value
            if an exception occurs.
            
        Notes
        -----
        This method forwards the call to the safe_execute function in
        maggie.utils.error_handling. It extracts known parameters
        from kwargs and passes the remaining kwargs to the function.
        """
        from maggie.utils.error_handling import safe_execute as do_safe_execute
        from maggie.utils.error_handling import ErrorCategory, ErrorSeverity
        
        # Extract known kwargs
        error_code = kwargs.pop('error_code', None)
        default_return = kwargs.pop('default_return', None)
        error_details = kwargs.pop('error_details', None)
        error_category = kwargs.pop('error_category', ErrorCategory.UNKNOWN)
        error_severity = kwargs.pop('error_severity', ErrorSeverity.ERROR)
        publish_error = kwargs.pop('publish_error', True)
        include_state_info = kwargs.pop('include_state_info', True)
        
        return do_safe_execute(
            func, *args,
            error_code=error_code,
            default_return=default_return,
            error_details=error_details,
            error_category=error_category,
            error_severity=error_severity,
            publish_error=publish_error,
            include_state_info=include_state_info,
            **kwargs
        )


class EventBusAdapter(IEventPublisher):
    """
    Adapter that converts an EventBus instance to the IEventPublisher interface.
    
    This adapter enables loose coupling between components that need to publish
    events and the concrete EventBus implementation. It forwards publish calls
    to the appropriate methods on the wrapped EventBus instance and handles 
    automatic registration with the capability registry.
    
    Parameters
    ----------
    event_bus : EventBus
        The EventBus instance to adapt to the IEventPublisher interface.
        
    Attributes
    ----------
    event_bus : EventBus
        Reference to the adapted EventBus instance.
        
    Examples
    --------
    >>> from maggie.core.event import EventBus
    >>> from maggie.utils.adapters import EventBusAdapter
    >>> event_bus = EventBus()
    >>> event_adapter = EventBusAdapter(event_bus)
    >>> # Now the adapter can be used through the IEventPublisher interface
    >>> event_adapter.publish("component_started", {"name": "MyComponent"})
    
    Notes
    -----
    This adapter automatically registers itself with the CapabilityRegistry
    upon initialization, making it discoverable through the IEventPublisher
    interface.
    """
    
    def __init__(self, event_bus) -> None:
        """
        Initialize adapter with EventBus instance.
        
        Parameters
        ----------
        event_bus : EventBus
            The EventBus instance to adapt to the IEventPublisher interface.
        """
        self.event_bus = event_bus
        
        # Register self with the capability registry
        registry = CapabilityRegistry.get_instance()
        registry.register(IEventPublisher, self)
    
    def publish(self, event_type: str, data: Any = None, **kwargs) -> None:
        """
        Publish an event.
        
        Parameters
        ----------
        event_type : str
            Type of event being published (e.g., "state_changed").
        data : Any, optional
            Data associated with the event.
        **kwargs : dict, optional
            Additional parameters for the event:
            - priority : EventPriority, optional
                Priority level for the event (HIGH, NORMAL, LOW).
            
        Notes
        -----
        This method forwards the publish call to the underlying EventBus.
        If a priority is specified in kwargs, it's passed to the EventBus
        publish method as a separate parameter.
        """
        priority = kwargs.pop('priority', None)
        if priority is not None:
            self.event_bus.publish(event_type, data, priority)
        else:
            self.event_bus.publish(event_type, data)


class StateManagerAdapter(IStateProvider):
    """
    Adapter that converts a StateManager instance to the IStateProvider interface.
    
    This adapter enables loose coupling between components that need to query
    the current system state and the concrete StateManager implementation. It 
    forwards state queries to the appropriate methods on the wrapped StateManager
    instance and handles automatic registration with the capability registry.
    
    Parameters
    ----------
    state_manager : StateManager
        The StateManager instance to adapt to the IStateProvider interface.
        
    Attributes
    ----------
    state_manager : StateManager
        Reference to the adapted StateManager instance.
        
    Examples
    --------
    >>> from maggie.core.state import StateManager, State
    >>> from maggie.utils.adapters import StateManagerAdapter
    >>> state_manager = StateManager(State.INIT, event_bus)
    >>> state_adapter = StateManagerAdapter(state_manager)
    >>> # Now the adapter can be used through the IStateProvider interface
    >>> current_state = state_adapter.get_current_state()
    >>> print(current_state.name)  # "INIT"
    
    Notes
    -----
    This adapter automatically registers itself with the CapabilityRegistry
    upon initialization, making it discoverable through the IStateProvider
    interface.
    """
    
    def __init__(self, state_manager: StateManager) -> None:
        """
        Initialize adapter with StateManager instance.
        
        Parameters
        ----------
        state_manager : StateManager
            The StateManager instance to adapt to the IStateProvider interface.
        """
        self.state_manager = state_manager
        
        # Register self with the capability registry
        registry = CapabilityRegistry.get_instance()
        registry.register(IStateProvider, self)
    
    def get_current_state(self) -> State:
        """
        Get the current state of the system.
        
        Returns
        -------
        State
            Current state of the system, as an enum value from the State
            enumeration defined in maggie.core.state.
            
        Notes
        -----
        This method simply forwards the call to the state_manager's
        get_current_state method, providing a level of indirection that
        allows the StateManager implementation to change without affecting
        components that use the IStateProvider interface.
        """
        return self.state_manager.get_current_state()