"""
Maggie AI Assistant - Event Bus Module
=====================================

This module provides an event bus system for the Maggie AI Assistant,
enabling event-driven architecture and decoupled communication between components.
"""

import queue
import threading
import time
import uuid
import sys
import logging
from typing import Dict, Any, Optional, List, Callable, Set, Tuple, Union, cast

from maggie.utils.abstractions import IEventPublisher

# Setup basic logging as fallback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('maggie.core.event')

# Define event constants
STATE_CHANGED_EVENT = 'state_changed'
STATE_ENTRY_EVENT = 'state_entry'
STATE_EXIT_EVENT = 'state_exit'
TRANSITION_COMPLETED_EVENT = 'transition_completed'
TRANSITION_FAILED_EVENT = 'transition_failed'
UI_STATE_UPDATE_EVENT = 'ui_state_update'
INPUT_ACTIVATION_EVENT = 'input_activation'
INPUT_DEACTIVATION_EVENT = 'input_deactivation'

class EventPriority:
    """Priority levels for event processing."""
    HIGH = 0
    NORMAL = 10
    LOW = 20
    BACKGROUND = 30

class EventBus(IEventPublisher):
    """
    Central event bus for publishing and subscribing to events.
    Implements IEventPublisher interface for decoupled communication.
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Tuple[int, Callable]]] = {}
        self.queue = queue.PriorityQueue()
        self.running = False
        self._worker_thread = None
        self._lock = threading.RLock()
        self.logger = logging.getLogger('maggie.core.event.EventBus')
        self._correlation_id = None
        self._event_filters = {}
    
    def subscribe(self, event_type: str, callback: Callable, priority: int = EventPriority.NORMAL) -> None:
        """Subscribe to an event type."""
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            
            self.subscribers[event_type].append((priority, callback))
            self.subscribers[event_type].sort(key=lambda x: x[0])
            self.logger.debug(f"Subscription added for event type: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """Unsubscribe from an event type."""
        with self._lock:
            if event_type not in self.subscribers:
                return False
            
            for (i, (_, cb)) in enumerate(self.subscribers[event_type]):
                if cb == callback:
                    self.subscribers[event_type].pop(i)
                    self.logger.debug(f"Unsubscribed from event type: {event_type}")
                    return True
            
            return False
    
    def publish(self, event_type: str, data: Any = None, **kwargs) -> None:
        """Publish an event. Implements IEventPublisher interface."""
        # Add correlation ID if data is a dictionary
        if isinstance(data, dict) and self._correlation_id:
            data = data.copy()
            if 'correlation_id' not in data:
                data['correlation_id'] = self._correlation_id
        
        # Extract priority from kwargs or use default
        priority = kwargs.get('priority', EventPriority.NORMAL)
        self.queue.put((priority, (event_type, data)))
        self.logger.debug(f"Event published: {event_type}")
    
    def start(self) -> bool:
        """Start the event processing loop."""
        with self._lock:
            if self.running:
                return False
            
            self.running = True
            self._worker_thread = threading.Thread(
                target=self._process_events, 
                name='EventBusThread', 
                daemon=True
            )
            self._worker_thread.start()
            self.logger.info('Event bus started')
            return True
    
    def stop(self) -> bool:
        """Stop the event processing loop."""
        with self._lock:
            if not self.running:
                return False
            
            self.running = False
            self.queue.put((0, None))
            
            if self._worker_thread:
                self._worker_thread.join(timeout=2.0)
            
            self.logger.info('Event bus stopped')
            return True
    
    def set_correlation_id(self, correlation_id: Optional[str]) -> None:
        """Set the correlation ID for event tracking."""
        self._correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get the current correlation ID."""
        return self._correlation_id
    
    def add_event_filter(self, event_type: str, filter_func: Callable[[Any], bool]) -> str:
        """Add a filter for an event type."""
        filter_id = str(uuid.uuid4())
        
        with self._lock:
            if event_type not in self._event_filters:
                self._event_filters[event_type] = {}
            
            self._event_filters[event_type][filter_id] = filter_func
        
        return filter_id
    
    def remove_event_filter(self, event_type: str, filter_id: str) -> bool:
        """Remove a filter for an event type."""
        with self._lock:
            if event_type in self._event_filters and filter_id in self._event_filters[event_type]:
                del self._event_filters[event_type][filter_id]
                
                if not self._event_filters[event_type]:
                    del self._event_filters[event_type]
                
                return True
        
        return False
    
    def _process_events(self) -> None:
        """Process events from the queue."""
        while self.running:
            try:
                # Get events from queue (batch processing)
                events_batch = []
                
                try:
                    priority, event = self.queue.get(timeout=0.05)
                    if event is None:
                        break
                    
                    events_batch.append((priority, event))
                    self.queue.task_done()
                except queue.Empty:
                    time.sleep(0.001)
                    continue
                
                # Process additional events in batch
                batch_size = 10
                while len(events_batch) < batch_size:
                    try:
                        priority, event = self.queue.get(block=False)
                        if event is None:
                            break
                        
                        events_batch.append((priority, event))
                        self.queue.task_done()
                    except queue.Empty:
                        break
                
                # Dispatch events
                for (priority, event) in events_batch:
                    if event is None:
                        continue
                    
                    event_type, data = event
                    self._dispatch_event(event_type, data)
            
            except Exception as e:
                error_msg = f"Error processing events: {e}"
                self.logger.error(error_msg)
                
                # Try to report error via abstraction layer instead of direct dependency
                try:
                    from maggie.utils.abstractions import get_error_handler
                    error_handler = get_error_handler()
                    if error_handler:
                        error_handler.record_error(
                            message=error_msg,
                            exception=e,
                            category='SYSTEM',
                            severity='ERROR',
                            source='EventBus._process_events'
                        )
                except ImportError:
                    # Abstractions module not available, skip error handling
                    pass
                except Exception:
                    # Error handler not available or failed
                    pass
    
    def _dispatch_event(self, event_type: str, data: Any) -> None:
        """Dispatch an event to subscribers."""
        with self._lock:
            # Apply filters
            if event_type in self._event_filters:
                should_process = True
                
                for filter_func in self._event_filters[event_type].values():
                    try:
                        if not filter_func(data):
                            should_process = False
                            break
                    except Exception as e:
                        self.logger.error(f"Error in event filter for {event_type}: {e}")
                
                if not should_process:
                    return
            
            # Dispatch to subscribers
            if event_type in self.subscribers:
                for (_, callback) in self.subscribers[event_type]:
                    try:
                        callback(data)
                    except Exception as e:
                        error_msg = f"Error in event handler for {event_type}: {e}"
                        self.logger.error(error_msg)
                        
                        # Publish error event without direct dependency
                        self.publish(
                            'error_logged',
                            {
                                'message': error_msg,
                                'event_type': event_type,
                                'source': 'event_bus'
                            },
                            priority=EventPriority.HIGH
                        )
    
    def is_empty(self) -> bool:
        """Check if the event queue is empty."""
        return self.queue.empty()


class EventEmitter:
    """Base class for objects that emit events."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.logger = logging.getLogger(self.__class__.__name__)
        self._correlation_id = None
    
    def emit(self, event_type: str, data: Any = None, priority: int = EventPriority.NORMAL) -> None:
        """Emit an event through the event bus."""
        if self._correlation_id and self.event_bus:
            old_correlation_id = self.event_bus.get_correlation_id()
            self.event_bus.set_correlation_id(self._correlation_id)
            
            try:
                self.event_bus.publish(event_type, data, priority=priority)
            finally:
                self.event_bus.set_correlation_id(old_correlation_id)
        else:
            self.event_bus.publish(event_type, data, priority=priority)
    
    def set_correlation_id(self, correlation_id: Optional[str]) -> None:
        """Set the correlation ID for event tracking."""
        self._correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get the current correlation ID."""
        return self._correlation_id
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._correlation_id = None


class EventListener:
    """Base class for objects that listen for events."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.logger = logging.getLogger(self.__class__.__name__)
        self.subscriptions: Set[Tuple[str, Callable]] = set()
    
    def listen(self, event_type: str, callback: Callable, priority: int = EventPriority.NORMAL) -> None:
        """Subscribe to an event type."""
        self.event_bus.subscribe(event_type, callback, priority)
        self.subscriptions.add((event_type, callback))
    
    def stop_listening(self, event_type: str = None, callback: Callable = None) -> None:
        """Unsubscribe from events."""
        if event_type is None and callback is None:
            # Unsubscribe from all events
            for (evt_type, cb) in list(self.subscriptions):
                self.event_bus.unsubscribe(evt_type, cb)
                self.subscriptions.remove((evt_type, cb))
        
        elif callback is None:
            # Unsubscribe from a specific event type
            for (evt_type, cb) in list(self.subscriptions):
                if evt_type == event_type:
                    self.event_bus.unsubscribe(evt_type, cb)
                    self.subscriptions.remove((evt_type, cb))
        
        elif event_type is None:
            # Unsubscribe a specific callback
            for (evt_type, cb) in list(self.subscriptions):
                if cb == callback:
                    self.event_bus.unsubscribe(evt_type, cb)
                    self.subscriptions.remove((evt_type, cb))
        
        else:
            # Unsubscribe a specific event/callback pair
            self.event_bus.unsubscribe(event_type, callback)
            self.subscriptions.remove((event_type, callback))
    
    def add_filter(self, event_type: str, filter_func: Callable[[Any], bool]) -> str:
        """Add a filter for an event type."""
        return self.event_bus.add_event_filter(event_type, filter_func)
    
    def remove_filter(self, event_type: str, filter_id: str) -> bool:
        """Remove a filter for an event type."""
        return self.event_bus.remove_event_filter(event_type, filter_id)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_listening()


class CompositeEventListener(EventListener):
    """A container for multiple event listeners."""
    
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)
        self.children: List[EventListener] = []
    
    def add_listener(self, listener: EventListener) -> None:
        """Add a child listener."""
        if listener not in self.children:
            self.children.append(listener)
    
    def remove_listener(self, listener: EventListener) -> bool:
        """Remove a child listener."""
        if listener in self.children:
            self.children.remove(listener)
            return True
        
        return False
    
    def _forward_event(self, event_type: str, data: Any) -> None:
        """Forward an event to all child listeners."""
        for child in self.children:
            if hasattr(child, 'handle_event') and callable(getattr(child, 'handle_event')):
                try:
                    child.handle_event(event_type, data)
                except Exception as e:
                    self.logger.error(f"Error forwarding event to child: {e}")
    
    def listen_and_forward(self, event_type: str, priority: int = EventPriority.NORMAL) -> None:
        """Listen for an event and forward it to all child listeners."""
        self.listen(
            event_type, 
            lambda data: self._forward_event(event_type, data),
            priority
        )
    
    def cleanup(self) -> None:
        """Clean up resources including child listeners."""
        super().cleanup()
        
        for child in self.children:
            child.cleanup()
        
        self.children.clear()