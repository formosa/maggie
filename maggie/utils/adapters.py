"""
Maggie AI Assistant - Adapter Classes
=====================================

This module provides adapter classes to bridge between concrete implementations
and abstract interfaces without creating direct dependencies.
"""

from typing import Dict, Any, Optional, Callable

# Import interfaces from abstractions
from maggie.utils.abstractions import (
    ILoggerProvider, 
    IErrorHandler, 
    IEventPublisher, 
    IStateProvider, 
    CapabilityRegistry
)

class LoggingManagerAdapter(ILoggerProvider):
    """Adapter for LoggingManager to implement ILoggerProvider interface."""
    
    def __init__(self, logging_manager):
        """Initialize adapter with LoggingManager instance."""
        self.logging_manager = logging_manager
        
        # Register self with the capability registry
        registry = CapabilityRegistry.get_instance()
        registry.register(ILoggerProvider, self)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        from maggie.utils.logging import LogLevel
        self.logging_manager.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        from maggie.utils.logging import LogLevel
        self.logging_manager.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        from maggie.utils.logging import LogLevel
        self.logging_manager.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log an error message."""
        from maggie.utils.logging import LogLevel
        self.logging_manager.log(LogLevel.ERROR, message, exception=exception, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log a critical message."""
        from maggie.utils.logging import LogLevel
        self.logging_manager.log(LogLevel.CRITICAL, message, exception=exception, **kwargs)

class ErrorHandlerAdapter(IErrorHandler):
    """Adapter to convert error handling functions to IErrorHandler interface."""
    
    def __init__(self):
        """Initialize adapter and register with capability registry."""
        # Register self with the capability registry
        registry = CapabilityRegistry.get_instance()
        registry.register(IErrorHandler, self)
    
    def record_error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> Any:
        """Record an error."""
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
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Safely execute a function."""
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
    """Adapter for EventBus to implement IEventPublisher interface."""
    
    def __init__(self, event_bus):
        """Initialize adapter with EventBus instance."""
        self.event_bus = event_bus
        
        # Register self with the capability registry
        registry = CapabilityRegistry.get_instance()
        registry.register(IEventPublisher, self)
    
    def publish(self, event_type: str, data: Any = None, **kwargs) -> None:
        """Publish an event."""
        priority = kwargs.pop('priority', None)
        if priority is not None:
            self.event_bus.publish(event_type, data, priority)
        else:
            self.event_bus.publish(event_type, data)

class StateManagerAdapter(IStateProvider):
    """Adapter for StateManager to implement IStateProvider interface."""
    
    def __init__(self, state_manager):
        """Initialize adapter with StateManager instance."""
        self.state_manager = state_manager
        
        # Register self with the capability registry
        registry = CapabilityRegistry.get_instance()
        registry.register(IStateProvider, self)
    
    def get_current_state(self) -> Any:
        """Get the current state."""
        return self.state_manager.get_current_state()