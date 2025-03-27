"""
Maggie AI Assistant - Event Bus Module
=====================================

This module provides an event bus system for the Maggie AI Assistant,
enabling event-driven architecture and decoupled communication between components.

The event bus architecture implemented in this module follows the publish-subscribe pattern
(also known as pub/sub), a messaging pattern where senders (publishers) categorize messages
into classes and send them without knowledge of which receivers (subscribers) will receive them.
Similarly, receivers express interest in one or more classes and only receive messages of interest,
without knowledge of which senders exist.

Key features:
    - Decoupled communication between components
    - Priority-based event processing
    - Correlation tracking for related events
    - Event filtering capabilities 
    - Batch processing for performance optimization
    - Thread-safe implementations
    - Hierarchical event listener composition

See Also
--------
https://en.wikipedia.org/wiki/Publish%E2%80%93subscribe_pattern : Details on the publish-subscribe pattern
https://martinfowler.com/articles/201701-event-driven.html : Martin Fowler's article on event-driven architecture
https://docs.python.org/3/library/queue.html : Python's queue module used for event queueing
https://docs.python.org/3/library/threading.html : Python's threading module used for concurrent event processing
"""

import queue
import threading
import time
import uuid
import sys
import logging
from typing import Dict, Any, Optional, List, Callable, Set, Tuple, Union, cast

from maggie.utils.abstractions import IEventPublisher

# State-related event constants
STATE_CHANGED_EVENT: str
"""Event type fired when a state change occurs in the system."""

STATE_ENTRY_EVENT: str 
"""Event type fired when a new state is entered."""

STATE_EXIT_EVENT: str
"""Event type fired when a state is exited."""

TRANSITION_COMPLETED_EVENT: str
"""Event type fired when a state transition is completed."""

TRANSITION_FAILED_EVENT: str
"""Event type fired when a state transition fails."""

# UI-related event constants
UI_STATE_UPDATE_EVENT: str
"""Event type fired when the UI state needs to be updated."""

# Input-related event constants
INPUT_ACTIVATION_EVENT: str
"""Event type fired when input is activated."""

INPUT_DEACTIVATION_EVENT: str
"""Event type fired when input is deactivated."""

class EventPriority:
    """
    Priority levels for event processing.
    
    This class defines constants that represent different levels of priority for event
    processing. Events with higher priority (lower numerical value) are processed before
    events with lower priority. This allows critical system events to be handled promptly
    even when the system is under load.
    
    Attributes
    ----------
    HIGH : int
        Highest priority (0). Used for critical system events that must be processed immediately.
    NORMAL : int
        Normal priority (10). Used for regular events that should be processed in a timely manner.
    LOW : int
        Low priority (20). Used for non-urgent events that can be delayed if the system is busy.
    BACKGROUND : int
        Lowest priority (30). Used for background tasks that should only be processed when
        the system is idle.
        
    Notes
    -----
    When designing a system using these priorities, it's important to consider potential
    starvation of lower priority events during high system load. Proper monitoring and
    resource management should be implemented to ensure all events eventually get processed.
    """
    
    HIGH: int
    """Highest priority (0). Used for critical system events."""
    
    NORMAL: int
    """Normal priority (10). Used for regular events."""
    
    LOW: int
    """Low priority (20). Used for non-urgent events."""
    
    BACKGROUND: int
    """Lowest priority (30). Used for background tasks."""

class EventBus(IEventPublisher):
    """
    Central event bus for publishing and subscribing to events.
    
    The EventBus is the core component of the event system, responsible for
    managing subscriptions and publishing events to interested subscribers.
    It implements the IEventPublisher interface for decoupled communication
    between components.
    
    The EventBus maintains a thread-safe queue of events and processes them
    in a separate thread, allowing for asynchronous event handling. Events
    are dispatched to subscribers based on their priority, with higher priority
    subscribers receiving events first.
    
    Attributes
    ----------
    subscribers : Dict[str, List[Tuple[int, Callable]]]
        A dictionary mapping event types to lists of subscribers.
        Each subscriber is represented as a tuple of (priority, callback function).
    queue : queue.PriorityQueue
        A thread-safe priority queue for storing events to be processed.
    running : bool
        A flag indicating whether the event bus is currently running.
    _worker_thread : Optional[threading.Thread]
        The worker thread that processes events from the queue.
    _lock : threading.RLock
        A reentrant lock for thread-safe operations on shared data.
    logger : logging.Logger
        Logger for the EventBus.
    _correlation_id : Optional[str]
        The current correlation ID for event tracking.
    _event_filters : Dict[str, Dict[str, Callable[[Any], bool]]]
        A dictionary mapping event types to dictionaries of filter functions.
        
    Examples
    --------
    >>> from maggie.core.event import EventBus
    >>> event_bus = EventBus()
    >>> event_bus.start()
    
    >>> # Subscribe to an event
    >>> def handle_state_changed(data):
    ...     print(f"State changed: {data}")
    >>> event_bus.subscribe("state_changed", handle_state_changed)
    
    >>> # Publish an event
    >>> event_bus.publish("state_changed", {"from": "INIT", "to": "READY"})
    
    >>> # Unsubscribe from an event
    >>> event_bus.unsubscribe("state_changed", handle_state_changed)
    
    >>> # Stop the event bus
    >>> event_bus.stop()
    """
    
    def __init__(self) -> None:
        """
        Initialize the EventBus.
        
        Sets up the necessary attributes and initializes internal structures
        for event handling and subscription management.
        """
        ...
    
    def subscribe(self, event_type: str, callback: Callable, priority: int = ...) -> None:
        """
        Subscribe to an event type.
        
        Registers a callback function to be called when events of the specified
        type are published. Multiple callbacks can be registered for the same
        event type, and they will be called in order of priority (higher priority
        callbacks are called first).
        
        Parameters
        ----------
        event_type : str
            The type of event to subscribe to.
        callback : Callable
            The function to call when events of this type are published.
            The function should accept a single parameter for the event data.
        priority : int, optional
            The priority of this subscription. Higher priority (lower numerical value)
            subscriptions are called first. Default is EventPriority.NORMAL.
            
        Notes
        -----
        Subscriptions are stored in a list sorted by priority, so subscribers
        with higher priority (lower numerical value) will be notified first.
        
        Examples
        --------
        >>> def handle_state_changed(data):
        ...     print(f"State changed: {data}")
        >>> event_bus.subscribe("state_changed", handle_state_changed, EventPriority.HIGH)
        """
        ...
    
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Unsubscribe from an event type.
        
        Removes a previously registered callback function from the subscribers
        for the specified event type.
        
        Parameters
        ----------
        event_type : str
            The type of event to unsubscribe from.
        callback : Callable
            The callback function to remove.
            
        Returns
        -------
        bool
            True if the callback was found and removed, False otherwise.
            
        Examples
        --------
        >>> def handle_state_changed(data):
        ...     print(f"State changed: {data}")
        >>> event_bus.subscribe("state_changed", handle_state_changed)
        >>> event_bus.unsubscribe("state_changed", handle_state_changed)  # Returns True
        >>> event_bus.unsubscribe("unknown_event", handle_state_changed)  # Returns False
        """
        ...
    
    def publish(self, event_type: str, data: Any = None, **kwargs) -> None:
        """
        Publish an event.
        
        Adds an event to the processing queue. The event will be dispatched
        to all subscribers of the specified event type when it is processed.
        Implements the IEventPublisher interface.
        
        Parameters
        ----------
        event_type : str
            The type of event to publish.
        data : Any, optional
            The data to include with the event. Default is None.
        **kwargs : dict
            Additional keyword arguments.
            
            - priority : int, optional
                The priority of this event. Higher priority (lower numerical value)
                events are processed first. Default is EventPriority.NORMAL.
                
        Notes
        -----
        If the data is a dictionary and a correlation ID is set, it will be added
        to the data under the 'correlation_id' key, unless it already exists.
        
        Examples
        --------
        >>> event_bus.publish("state_changed", {"from": "INIT", "to": "READY"})
        >>> event_bus.publish("error", {"message": "Connection failed"}, priority=EventPriority.HIGH)
        """
        ...
    
    def start(self) -> bool:
        """
        Start the event processing loop.
        
        Creates and starts a worker thread that processes events from the queue.
        Events are dispatched to subscribers in order of priority.
        
        Returns
        -------
        bool
            True if the event bus was started, False if it was already running.
            
        Notes
        -----
        This method uses a daemon thread for event processing, which means the
        thread will be automatically terminated when the main program exits.
        
        Examples
        --------
        >>> event_bus = EventBus()
        >>> event_bus.start()  # Returns True
        >>> event_bus.start()  # Returns False (already running)
        """
        ...
    
    def stop(self) -> bool:
        """
        Stop the event processing loop.
        
        Stops the worker thread that processes events from the queue.
        Any events that have not been processed will remain in the queue.
        
        Returns
        -------
        bool
            True if the event bus was stopped, False if it was not running.
            
        Notes
        -----
        This method adds a special event to the queue to signal the worker
        thread to stop, and then waits for the thread to terminate with a
        timeout of 2 seconds.
        
        Examples
        --------
        >>> event_bus = EventBus()
        >>> event_bus.start()
        >>> event_bus.stop()  # Returns True
        >>> event_bus.stop()  # Returns False (not running)
        """
        ...
    
    def set_correlation_id(self, correlation_id: Optional[str]) -> None:
        """
        Set the correlation ID for event tracking.
        
        The correlation ID is used to track related events across different
        components of the system. It can be useful for debugging and tracing
        the flow of events.
        
        Parameters
        ----------
        correlation_id : Optional[str]
            The correlation ID to set, or None to clear the correlation ID.
            
        Notes
        -----
        The correlation ID is automatically added to event data if the data
        is a dictionary and it doesn't already have a 'correlation_id' key.
        
        Examples
        --------
        >>> event_bus.set_correlation_id("request-123")
        >>> event_bus.publish("state_changed", {"from": "INIT", "to": "READY"})
        # The event data will include 'correlation_id': 'request-123'
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
        >>> event_bus.set_correlation_id("request-123")
        >>> event_bus.get_correlation_id()
        'request-123'
        >>> event_bus.set_correlation_id(None)
        >>> event_bus.get_correlation_id() is None
        True
        """
        ...
    
    def add_event_filter(self, event_type: str, filter_func: Callable[[Any], bool]) -> str:
        """
        Add a filter for an event type.
        
        Event filters allow for conditional event processing. An event will only
        be dispatched to subscribers if all filters for its type return True.
        
        Parameters
        ----------
        event_type : str
            The type of event to filter.
        filter_func : Callable[[Any], bool]
            The filter function. It should accept the event data as a parameter
            and return a boolean indicating whether the event should be processed.
            
        Returns
        -------
        str
            A unique ID for the filter, which can be used to remove it later.
            
        Notes
        -----
        Event filters are applied before dispatching an event to subscribers.
        If any filter returns False, the event will not be dispatched.
        
        Examples
        --------
        >>> def only_errors(data):
        ...     return data.get('level') == 'ERROR'
        >>> filter_id = event_bus.add_event_filter("log", only_errors)
        # Only log events with level='ERROR' will be dispatched
        """
        ...
    
    def remove_event_filter(self, event_type: str, filter_id: str) -> bool:
        """
        Remove a filter for an event type.
        
        Parameters
        ----------
        event_type : str
            The type of event the filter was added to.
        filter_id : str
            The unique ID of the filter to remove, as returned by add_event_filter.
            
        Returns
        -------
        bool
            True if the filter was found and removed, False otherwise.
            
        Examples
        --------
        >>> def only_errors(data):
        ...     return data.get('level') == 'ERROR'
        >>> filter_id = event_bus.add_event_filter("log", only_errors)
        >>> event_bus.remove_event_filter("log", filter_id)  # Returns True
        >>> event_bus.remove_event_filter("log", "invalid-id")  # Returns False
        """
        ...
    
    def _process_events(self) -> None:
        """
        Process events from the queue.
        
        This method is called by the worker thread started by the start() method.
        It continuously retrieves events from the queue and dispatches them to
        subscribers until the event bus is stopped.
        
        Events are processed in batches for better performance, with higher
        priority events processed first. If an error occurs during event
        processing, it is logged and an error event is published.
        
        Notes
        -----
        This is a private method that should not be called directly.
        """
        ...
    
    def _dispatch_event(self, event_type: str, data: Any) -> None:
        """
        Dispatch an event to subscribers.
        
        This method is called by _process_events() to dispatch an event to all
        subscribers of the specified event type. It applies event filters and
        calls subscriber callbacks with the event data.
        
        Parameters
        ----------
        event_type : str
            The type of event to dispatch.
        data : Any
            The event data to pass to subscribers.
            
        Notes
        -----
        This is a private method that should not be called directly.
        
        If an error occurs while calling a subscriber callback, it is logged
        and an error event is published, but processing continues for other
        subscribers.
        """
        ...
    
    def is_empty(self) -> bool:
        """
        Check if the event queue is empty.
        
        Returns
        -------
        bool
            True if the event queue is empty, False otherwise.
            
        Examples
        --------
        >>> event_bus = EventBus()
        >>> event_bus.start()
        >>> event_bus.is_empty()
        True
        >>> event_bus.publish("test_event")
        >>> event_bus.is_empty()
        False
        """
        ...


class EventEmitter:
    """
    Base class for objects that emit events.
    
    This class provides a common interface for objects that need to publish
    events through an event bus. It maintains a reference to the event bus
    and provides methods for emitting events and tracking correlation IDs.
    
    Attributes
    ----------
    event_bus : EventBus
        The event bus to publish events through.
    logger : logging.Logger
        Logger for the EventEmitter.
    _correlation_id : Optional[str]
        The current correlation ID for event tracking.
        
    Examples
    --------
    >>> class MyComponent(EventEmitter):
    ...     def __init__(self, event_bus):
    ...         super().__init__(event_bus)
    ...         
    ...     def do_something(self):
    ...         # Do something
    ...         self.emit("something_done", {"status": "success"})
    ...         
    >>> event_bus = EventBus()
    >>> component = MyComponent(event_bus)
    >>> component.do_something()  # Emits 'something_done' event
    """
    
    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialize the EventEmitter.
        
        Parameters
        ----------
        event_bus : EventBus
            The event bus to publish events through.
        """
        ...
    
    def emit(self, event_type: str, data: Any = None, priority: int = ...) -> None:
        """
        Emit an event through the event bus.
        
        This method publishes an event through the event bus, optionally with
        a correlation ID to track related events.
        
        Parameters
        ----------
        event_type : str
            The type of event to emit.
        data : Any, optional
            The data to include with the event. Default is None.
        priority : int, optional
            The priority of this event. Higher priority (lower numerical value)
            events are processed first. Default is EventPriority.NORMAL.
            
        Notes
        -----
        If a correlation ID is set on this emitter, it will be set on the event bus
        before publishing the event, and then restored to its previous value.
        
        Examples
        --------
        >>> component = MyComponent(event_bus)
        >>> component.emit("status_update", {"status": "ready"})
        >>> component.emit("error", {"message": "Failed"}, priority=EventPriority.HIGH)
        """
        ...
    
    def set_correlation_id(self, correlation_id: Optional[str]) -> None:
        """
        Set the correlation ID for event tracking.
        
        The correlation ID is used to track related events across different
        components of the system. It can be useful for debugging and tracing
        the flow of events.
        
        Parameters
        ----------
        correlation_id : Optional[str]
            The correlation ID to set, or None to clear the correlation ID.
            
        Examples
        --------
        >>> component = MyComponent(event_bus)
        >>> component.set_correlation_id("request-123")
        >>> component.emit("status_update", {"status": "ready"})
        # The event will include 'correlation_id': 'request-123'
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
        >>> component = MyComponent(event_bus)
        >>> component.set_correlation_id("request-123")
        >>> component.get_correlation_id()
        'request-123'
        """
        ...
    
    def cleanup(self) -> None:
        """
        Clean up resources.
        
        This method should be called when the emitter is no longer needed
        to release any resources it has acquired.
        
        Examples
        --------
        >>> component = MyComponent(event_bus)
        >>> # Use component
        >>> component.cleanup()
        """
        ...


class EventListener:
    """
    Base class for objects that listen for events.
    
    This class provides a common interface for objects that need to subscribe
    to events from an event bus. It maintains a reference to the event bus
    and keeps track of subscriptions to simplify cleanup.
    
    Attributes
    ----------
    event_bus : EventBus
        The event bus to subscribe to events from.
    logger : logging.Logger
        Logger for the EventListener.
    subscriptions : Set[Tuple[str, Callable]]
        A set of (event_type, callback) tuples representing active subscriptions.
        
    Examples
    --------
    >>> class MyComponent(EventListener):
    ...     def __init__(self, event_bus):
    ...         super().__init__(event_bus)
    ...         self.listen("state_changed", self.handle_state_change)
    ...         
    ...     def handle_state_change(self, data):
    ...         print(f"State changed: {data}")
    ...         
    >>> event_bus = EventBus()
    >>> component = MyComponent(event_bus)
    >>> event_bus.publish("state_changed", {"from": "INIT", "to": "READY"})
    # Output: State changed: {'from': 'INIT', 'to': 'READY'}
    """
    
    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialize the EventListener.
        
        Parameters
        ----------
        event_bus : EventBus
            The event bus to subscribe to events from.
        """
        ...
    
    def listen(self, event_type: str, callback: Callable, priority: int = ...) -> None:
        """
        Subscribe to an event type.
        
        This method subscribes to events of the specified type through the event bus
        and keeps track of the subscription for later cleanup.
        
        Parameters
        ----------
        event_type : str
            The type of event to listen for.
        callback : Callable
            The function to call when events of this type are published.
            The function should accept a single parameter for the event data.
        priority : int, optional
            The priority of this subscription. Higher priority (lower numerical value)
            subscriptions are called first. Default is EventPriority.NORMAL.
            
        Examples
        --------
        >>> class MyComponent(EventListener):
        ...     def __init__(self, event_bus):
        ...         super().__init__(event_bus)
        ...         self.listen("state_changed", self.handle_state_change, EventPriority.HIGH)
        ...         
        ...     def handle_state_change(self, data):
        ...         print(f"State changed: {data}")
        """
        ...
    
    def stop_listening(self, event_type: str = None, callback: Callable = None) -> None:
        """
        Unsubscribe from events.
        
        This method unsubscribes from events based on the provided parameters.
        If both event_type and callback are None, it unsubscribes from all events.
        If only event_type is provided, it unsubscribes from all events of that type.
        If only callback is provided, it unsubscribes that callback from all event types.
        If both are provided, it unsubscribes that specific callback from that specific event type.
        
        Parameters
        ----------
        event_type : str, optional
            The type of event to stop listening for, or None to apply to all event types.
            Default is None.
        callback : Callable, optional
            The callback function to remove, or None to remove all callbacks.
            Default is None.
            
        Examples
        --------
        >>> class MyComponent(EventListener):
        ...     def __init__(self, event_bus):
        ...         super().__init__(event_bus)
        ...         self.listen("state_changed", self.handle_state_change)
        ...         self.listen("error", self.handle_error)
        ...         
        ...     def handle_state_change(self, data):
        ...         print(f"State changed: {data}")
        ...         
        ...     def handle_error(self, data):
        ...         print(f"Error: {data}")
        ...         
        ...     def stop_all(self):
        ...         self.stop_listening()  # Unsubscribe from all events
        ...         
        ...     def stop_state_events(self):
        ...         self.stop_listening("state_changed")  # Unsubscribe from state_changed events
        ...         
        ...     def stop_error_handler(self):
        ...         self.stop_listening(callback=self.handle_error)  # Unsubscribe handle_error from all events
        ...         
        ...     def stop_specific(self):
        ...         self.stop_listening("state_changed", self.handle_state_change)  # Unsubscribe specific callback from specific event
        """
        ...
    
    def add_filter(self, event_type: str, filter_func: Callable[[Any], bool]) -> str:
        """
        Add a filter for an event type.
        
        This method adds a filter for events of the specified type through the event bus.
        The filter will be applied before the event is dispatched to any subscribers.
        
        Parameters
        ----------
        event_type : str
            The type of event to filter.
        filter_func : Callable[[Any], bool]
            The filter function. It should accept the event data as a parameter
            and return a boolean indicating whether the event should be processed.
            
        Returns
        -------
        str
            A unique ID for the filter, which can be used to remove it later.
            
        Examples
        --------
        >>> class MyComponent(EventListener):
        ...     def __init__(self, event_bus):
        ...         super().__init__(event_bus)
        ...         self.listen("log", self.handle_log)
        ...         self.filter_id = self.add_filter("log", self.only_errors)
        ...         
        ...     def handle_log(self, data):
        ...         print(f"Log: {data}")
        ...         
        ...     def only_errors(self, data):
        ...         return data.get('level') == 'ERROR'
        """
        ...
    
    def remove_filter(self, event_type: str, filter_id: str) -> bool:
        """
        Remove a filter for an event type.
        
        This method removes a filter previously added with add_filter.
        
        Parameters
        ----------
        event_type : str
            The type of event the filter was added to.
        filter_id : str
            The unique ID of the filter to remove, as returned by add_filter.
            
        Returns
        -------
        bool
            True if the filter was found and removed, False otherwise.
            
        Examples
        --------
        >>> class MyComponent(EventListener):
        ...     def __init__(self, event_bus):
        ...         super().__init__(event_bus)
        ...         self.filter_id = self.add_filter("log", self.only_errors)
        ...         
        ...     def only_errors(self, data):
        ...         return data.get('level') == 'ERROR'
        ...         
        ...     def remove_error_filter(self):
        ...         self.remove_filter("log", self.filter_id)
        """
        ...
    
    def cleanup(self) -> None:
        """
        Clean up resources.
        
        This method unsubscribes from all events and should be called when
        the listener is no longer needed to avoid memory leaks.
        
        Examples
        --------
        >>> component = MyComponent(event_bus)
        >>> # Use component
        >>> component.cleanup()  # Unsubscribes from all events
        """
        ...


class CompositeEventListener(EventListener):
    """
    A container for multiple event listeners.
    
    This class allows for hierarchical organization of event listeners,
    making it easier to manage complex event handling logic. It can
    forward events to all child listeners, providing a way to broadcast
    events to multiple components.
    
    This implementation follows the Composite design pattern, where the
    CompositeEventListener acts as a composite node that can contain leaf
    nodes (regular EventListeners) or other composite nodes.
    
    Attributes
    ----------
    children : List[EventListener]
        The list of child listeners.
        
    See Also
    --------
    https://refactoring.guru/design-patterns/composite : Description of the Composite design pattern
    
    Examples
    --------
    >>> class ParentComponent(CompositeEventListener):
    ...     def __init__(self, event_bus):
    ...         super().__init__(event_bus)
    ...         self.child1 = ChildComponent1(event_bus)
    ...         self.child2 = ChildComponent2(event_bus)
    ...         self.add_listener(self.child1)
    ...         self.add_listener(self.child2)
    ...         self.listen_and_forward("state_changed")  # Forward state_changed events to all children
    ...         
    >>> class ChildComponent1(EventListener):
    ...     def __init__(self, event_bus):
    ...         super().__init__(event_bus)
    ...         
    ...     def handle_event(self, event_type, data):
    ...         if event_type == "state_changed":
    ...             print(f"Child1 received state change: {data}")
    ...             
    >>> class ChildComponent2(EventListener):
    ...     def __init__(self, event_bus):
    ...         super().__init__(event_bus)
    ...         
    ...     def handle_event(self, event_type, data):
    ...         if event_type == "state_changed":
    ...             print(f"Child2 received state change: {data}")
    """
    
    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialize the CompositeEventListener.
        
        Parameters
        ----------
        event_bus : EventBus
            The event bus to subscribe to events from.
        """
        ...
    
    def add_listener(self, listener: EventListener) -> None:
        """
        Add a child listener.
        
        This method adds a child listener to the composite. Events forwarded
        to the composite will be forwarded to all child listeners.
        
        Parameters
        ----------
        listener : EventListener
            The child listener to add.
            
        Examples
        --------
        >>> parent = ParentComponent(event_bus)
        >>> child = ChildComponent(event_bus)
        >>> parent.add_listener(child)
        """
        ...
    
    def remove_listener(self, listener: EventListener) -> bool:
        """
        Remove a child listener.
        
        This method removes a child listener from the composite.
        
        Parameters
        ----------
        listener : EventListener
            The child listener to remove.
            
        Returns
        -------
        bool
            True if the listener was found and removed, False otherwise.
            
        Examples
        --------
        >>> parent = ParentComponent(event_bus)
        >>> child = ChildComponent(event_bus)
        >>> parent.add_listener(child)
        >>> parent.remove_listener(child)  # Returns True
        >>> parent.remove_listener(child)  # Returns False (already removed)
        """
        ...
    
    def _forward_event(self, event_type: str, data: Any) -> None:
        """
        Forward an event to all child listeners.
        
        This method is called when an event is received that needs to be
        forwarded to all child listeners.
        
        Parameters
        ----------
        event_type : str
            The type of event being forwarded.
        data : Any
            The event data.
            
        Notes
        -----
        This is a private method that should not be called directly.
        Child listeners must have a handle_event method to receive forwarded events.
        """
        ...
    
    def listen_and_forward(self, event_type: str, priority: int = ...) -> None:
        """
        Listen for an event and forward it to all child listeners.
        
        This method subscribes to events of the specified type and forwards
        them to all child listeners when they are received.
        
        Parameters
        ----------
        event_type : str
            The type of event to listen for and forward.
        priority : int, optional
            The priority of this subscription. Higher priority (lower numerical value)
            subscriptions are called first. Default is EventPriority.NORMAL.
            
        Examples
        --------
        >>> parent = ParentComponent(event_bus)
        >>> child1 = ChildComponent1(event_bus)
        >>> child2 = ChildComponent2(event_bus)
        >>> parent.add_listener(child1)
        >>> parent.add_listener(child2)
        >>> parent.listen_and_forward("state_changed")
        >>> event_bus.publish("state_changed", {"from": "INIT", "to": "READY"})
        # Output:
        # Child1 received state change: {'from': 'INIT', 'to': 'READY'}
        # Child2 received state change: {'from': 'INIT', 'to': 'READY'}
        """
        ...
    
    def cleanup(self) -> None:
        """
        Clean up resources including child listeners.
        
        This method calls cleanup on all child listeners and then on itself.
        It should be called when the composite listener is no longer needed.
        
        Examples
        --------
        >>> parent = ParentComponent(event_bus)
        >>> child1 = ChildComponent1(event_bus)
        >>> child2 = ChildComponent2(event_bus)
        >>> parent.add_listener(child1)
        >>> parent.add_listener(child2)
        >>> # Use components
        >>> parent.cleanup()  # Cleans up parent and all children
        """
        ...