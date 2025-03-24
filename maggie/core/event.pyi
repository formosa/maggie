from typing import Dict, Any, Optional, List, Tuple, Callable, Set, Union

class EventPriority:
    """
    Predefined priority levels for event handling in the Maggie AI Assistant.

    Attributes
    ----------
    HIGH : int
        Highest priority level for critical events (0)
    NORMAL : int
        Standard priority level for most events (10)
    LOW : int
        Lower priority for less critical events (20)
    BACKGROUND : int
        Lowest priority for background or non-essential events (30)

    Notes
    -----
    Priority levels are used to determine the order of event processing 
    in the EventBus. Lower numeric values indicate higher priority.
    """
    HIGH: int
    NORMAL: int
    LOW: int
    BACKGROUND: int

class EventBus:
    """
    A centralized event management system for the Maggie AI Assistant.

    This class provides a robust publish-subscribe mechanism for 
    inter-component communication with prioritized event handling.

    Attributes
    ----------
    subscribers : Dict[str, List[Tuple[int, Callable]]]
        A dictionary mapping event types to lists of (priority, callback) tuples
    queue : queue.PriorityQueue
        A thread-safe priority queue for event processing
    running : bool
        Indicates whether the event bus is currently active
    logger : ComponentLogger
        Logging instance for tracking event bus operations

    Examples
    --------
    >>> event_bus = EventBus()
    >>> def example_handler(data):
    ...     print(f"Received event with data: {data}")
    >>> event_bus.subscribe('user_action', example_handler)
    >>> event_bus.publish('user_action', {'action': 'login'})
    >>> event_bus.start()  # Start processing events
    >>> event_bus.stop()   # Stop event processing

    Notes
    -----
    - Thread-safe event processing
    - Supports prioritized event subscriptions
    - Provides centralized event management for the AI assistant
    """
    def __init__(self) -> None:
        """
        Initialize the EventBus with empty subscribers and event queue.
        """
        ...

    def subscribe(self, 
                  event_type: str, 
                  callback: Callable, 
                  priority: int = EventPriority.NORMAL) -> None:
        """
        Subscribe a callback function to a specific event type.

        Parameters
        ----------
        event_type : str
            The type of event to subscribe to
        callback : Callable
            The function to be called when the event is published
        priority : int, optional
            Priority of the event handler (default is EventPriority.NORMAL)

        Notes
        -----
        - Callbacks are sorted by priority (lower numeric value = higher priority)
        - Multiple callbacks can be registered for the same event type
        """
        ...

    def unsubscribe(self, 
                    event_type: str, 
                    callback: Callable) -> bool:
        """
        Remove a specific callback from an event type's subscribers.

        Parameters
        ----------
        event_type : str
            The event type to unsubscribe from
        callback : Callable
            The specific callback function to remove

        Returns
        -------
        bool
            True if the callback was successfully unsubscribed, False otherwise
        """
        ...

    def publish(self, 
                event_type: str, 
                data: Any = None, 
                priority: int = EventPriority.NORMAL) -> None:
        """
        Publish an event to all subscribed handlers.

        Parameters
        ----------
        event_type : str
            The type of event to publish
        data : Any, optional
            Payload data associated with the event
        priority : int, optional
            Priority of the event (default is EventPriority.NORMAL)

        Notes
        -----
        - Events are queued and processed asynchronously
        - Handlers are called in order of their registered priority
        """
        ...

    def start(self) -> bool:
        """
        Start the event processing loop.

        Returns
        -------
        bool
            True if the event bus was successfully started, 
            False if it was already running

        Notes
        -----
        - Launches a daemon thread for event processing
        - Enables asynchronous event handling
        """
        ...

    def stop(self) -> bool:
        """
        Stop the event processing loop.

        Returns
        -------
        bool
            True if the event bus was successfully stopped, 
            False if it was not running

        Notes
        -----
        - Gracefully terminates the event processing thread
        - Prevents further event processing
        """
        ...

class EventEmitter:
    """
    A utility class for emitting events through an EventBus.

    Provides a simplified interface for publishing events with 
    associated logging capabilities.

    Attributes
    ----------
    event_bus : EventBus
        The event bus used for publishing events
    logger : ComponentLogger
        Logging instance for tracking event emissions

    Examples
    --------
    >>> emitter = EventEmitter(event_bus)
    >>> emitter.emit('system_status', {'status': 'online'})
    """
    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialize the EventEmitter with a specific event bus.

        Parameters
        ----------
        event_bus : EventBus
            The event bus to use for publishing events
        """
        ...

    def emit(self, 
             event_type: str, 
             data: Any = None, 
             priority: int = EventPriority.NORMAL) -> None:
        """
        Emit an event through the associated event bus.

        Parameters
        ----------
        event_type : str
            The type of event to emit
        data : Any, optional
            Payload data associated with the event
        priority : int, optional
            Priority of the event (default is EventPriority.NORMAL)
        """
        ...

class EventListener:
    """
    A utility class for managing event subscriptions.

    Provides methods to listen for and unsubscribe from events 
    with built-in tracking of subscriptions.

    Attributes
    ----------
    event_bus : EventBus
        The event bus used for listening to events
    logger : ComponentLogger
        Logging instance for tracking event listening activities
    subscriptions : Set[Tuple[str, Callable]]
        Tracked set of current event subscriptions

    Examples
    --------
    >>> listener = EventListener(event_bus)
    >>> def handle_event(data):
    ...     print(f"Received event: {data}")
    >>> listener.listen('system_update', handle_event)
    >>> # Later, to stop listening
    >>> listener.stop_listening()
    """
    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialize the EventListener with a specific event bus.

        Parameters
        ----------
        event_bus : EventBus
            The event bus to use for listening to events
        """
        ...

    def listen(self, 
               event_type: str, 
               callback: Callable, 
               priority: int = EventPriority.NORMAL) -> None:
        """
        Subscribe to a specific event type.

        Parameters
        ----------
        event_type : str
            The type of event to listen for
        callback : Callable
            The function to be called when the event is published
        priority : int, optional
            Priority of the event handler (default is EventPriority.NORMAL)
        """
        ...

    def stop_listening(self, 
                       event_type: Optional[str] = None, 
                       callback: Optional[Callable] = None) -> None:
        """
        Unsubscribe from events.

        Parameters
        ----------
        event_type : str, optional
            Specific event type to unsubscribe from
        callback : Callable, optional
            Specific callback to unsubscribe

        Notes
        -----
        - If no arguments are provided, unsubscribes from all events
        - If only event_type is provided, unsubscribes all callbacks for that type
        - If only callback is provided, unsubscribes that callback from all events
        - If both are provided, unsubscribes the specific callback from the specific event type
        """
        ...