"""
Maggie AI Assistant - Finite State Machine Module
================================================

This module defines a state management system for handling state transitions,
state-specific behaviors, and associated metadata. It implements a finite state machine
pattern to manage application states and transitions between them, with support for
validation, handlers, animation properties, and styling.

The module follows the State pattern, one of the Gang of Four design patterns described in
'Design Patterns: Elements of Reusable Object-Oriented Software'. The State pattern allows
an object to alter its behavior when its internal state changes, appearing as if the object
changed its class. In this implementation, the state machine centralizes transition logic
and provides a clean separation of concerns between state representation and state-dependent
behaviors.

Classes
-------
- State: An enumeration of possible states with associated properties for styling and display.
- StateTransition: Represents a transition between two states, including metadata and animation properties.
- StateManager: Manages the current state, valid transitions, and handlers for state-specific and transition-specific logic.
- StateAwareComponent: A base class for components that need to respond to state changes.

References
----------
.. [1] Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). Design Patterns: Elements of
       Reusable Object-Oriented Software. Addison-Wesley Professional.
       https://en.wikipedia.org/wiki/Design_Patterns
.. [2] State Pattern - https://refactoring.guru/design-patterns/state
.. [3] Finite State Machines - https://en.wikipedia.org/wiki/Finite-state_machine
"""

import threading
import time
import sys
import logging
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable, Tuple, Set, Union, Type
from dataclasses import dataclass, field

from maggie.utils.abstractions import IStateProvider

# Setup basic logging as fallback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('maggie.core.state')


class State(Enum):
    """
    Enumeration representing various states with associated properties and methods for styling.

    The State enum represents distinct application states in the Maggie AI Assistant's
    finite state machine. Each state has associated properties for visual representation
    and user interface styling. The states define the different operational modes of the
    application and the valid transitions between them.

    Attributes
    ----------
    INIT : Enum
        Initial state when the application starts up. This is a frozen state where minimal
        resources are allocated, and the system is preparing for initialization.
    STARTUP : Enum
        Startup state during which the system loads essential components. This is also a
        frozen state where user interaction is limited.
    IDLE : Enum
        Idle state where the system is waiting for activation. The system runs with minimal
        resource usage in this state.
    LOADING : Enum
        Loading state where the system is actively loading resources or models. This is a
        frozen state where user interaction is limited.
    READY : Enum
        Ready state where the system is prepared to receive and process commands.
    ACTIVE : Enum
        Active state where the system is actively interacting with the user.
    BUSY : Enum
        Busy state where the system is processing a command or performing a resource-intensive
        operation. This is a frozen state where user interaction is limited.
    CLEANUP : Enum
        Cleanup state where the system is releasing resources and preparing for shutdown or
        a state transition. This is a frozen state.
    SHUTDOWN : Enum
        Shutdown state where the system is in the process of terminating. This is a frozen state.

    Notes
    -----
    Frozen states are states where user interaction is limited or disabled. These states
    typically represent transitional phases where the system is busy performing internal
    operations.

    The visual representation of states helps users understand the current system status.
    Each state has associated colors, styling, and animation properties for the user
    interface.

    Examples
    --------
    >>> state = State.READY
    >>> print(state.display_name)
    'Ready'
    >>> print(state.bg_color)
    '#A5D6A7'
    >>> style = state.get_style()
    >>> print(style['background'])
    '#A5D6A7'
    """
    INIT = auto()
    STARTUP = auto()
    IDLE = auto()
    LOADING = auto()
    READY = auto()
    ACTIVE = auto()
    BUSY = auto()
    CLEANUP = auto()
    SHUTDOWN = auto()

    @property
    def bg_color(self) -> str:
        """
        Get the background color associated with the state.

        Returns
        -------
        str
            Hexadecimal color code (e.g., '#A5D6A7') for the state's background color.

        Notes
        -----
        Colors are chosen to provide visual cues about the state's nature:
        - Cool colors (blues, greens) for stable states
        - Warm colors (oranges, reds) for active or busy states
        - Neutral colors for transitional states
        """
        colors = {
            State.INIT: '#E0E0E0',
            State.STARTUP: '#B3E5FC',
            State.IDLE: '#C8E6C9',
            State.LOADING: '#FFE0B2',
            State.READY: '#A5D6A7',
            State.ACTIVE: '#FFCC80',
            State.BUSY: '#FFAB91',
            State.CLEANUP: '#E1BEE7',
            State.SHUTDOWN: '#EF9A9A'
        }
        return colors.get(self, '#FFFFFF')

    @property
    def font_color(self) -> str:
        """
        Get the font color associated with the state.

        Returns
        -------
        str
            Hexadecimal color code (e.g., '#212121') for the state's font color.
            Dark text ('#212121') is used for states with light backgrounds, and
            light text ('#FFFFFF') is used for states with dark backgrounds.

        Notes
        -----
        This ensures proper contrast for readability in the UI.
        """
        dark_text_states = {State.INIT, State.STARTUP, State.IDLE, State.LOADING, State.READY}
        return '#212121' if self in dark_text_states else '#FFFFFF'

    @property
    def display_name(self) -> str:
        """
        Get the human-readable name of the state.

        Returns
        -------
        str
            Capitalized name of the state (e.g., 'Ready' for State.READY).
        """
        return self.name.capitalize()

    def get_style(self) -> Dict[str, str]:
        """
        Get a dictionary of CSS style properties for the state.

        Returns
        -------
        Dict[str, str]
            Dictionary containing CSS style properties for the state, including
            background color, font color, border, font weight, padding, and border radius.

        Examples
        --------
        >>> state = State.READY
        >>> style = state.get_style()
        >>> print(style)
        {'background': '#A5D6A7', 'color': '#212121', 'border': '1px solid #424242',
         'font-weight': 'bold', 'padding': '4px 8px', 'border-radius': '4px'}
        """
        return {
            'background': self.bg_color,
            'color': self.font_color,
            'border': '1px solid #424242',
            'font-weight': 'bold',
            'padding': '4px 8px',
            'border-radius': '4px'
        }


@dataclass
class StateTransition:
    """
    Represents a transition between two states in a state machine.

    This class encapsulates all metadata associated with a state transition,
    including the source and target states, the trigger that caused the transition,
    and the time when the transition occurred. It also provides methods for animation
    properties and serialization.

    Parameters
    ----------
    from_state : State
        The initial state before the transition.
    to_state : State
        The target state after the transition.
    trigger : str
        The event or action that triggered the transition.
    timestamp : float
        The time at which the transition occurred (as returned by time.time()).
    metadata : Dict[str, Any], optional
        Additional metadata associated with the transition, by default {}.
        This can include user-specific data, context information, or any other
        relevant data for the transition.

    Attributes
    ----------
    from_state : State
        The initial state before the transition.
    to_state : State
        The target state after the transition.
    trigger : str
        The event or action that triggered the transition.
    timestamp : float
        The time at which the transition occurred.
    metadata : Dict[str, Any]
        Additional metadata associated with the transition.

    Examples
    --------
    >>> transition = StateTransition(
    ...     from_state=State.IDLE,
    ...     to_state=State.READY,
    ...     trigger="user_activation",
    ...     timestamp=time.time(),
    ...     metadata={"user_id": "12345"}
    ... )
    >>> print(transition.animation_type)
    'slide'
    >>> print(transition.animation_duration)
    300
    >>> props = transition.get_animation_properties()
    >>> print(props)
    {'type': 'slide', 'duration': 300, 'easing': 'ease-in-out'}
    """
    from_state: State
    to_state: State
    trigger: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: 'StateTransition') -> bool:
        """
        Compare two StateTransition objects based on their timestamps.

        This enables sorting of transitions by time.

        Parameters
        ----------
        other : StateTransition
            Another StateTransition object to compare against.

        Returns
        -------
        bool
            True if this transition occurred before the other, False otherwise.

        Examples
        --------
        >>> t1 = StateTransition(State.IDLE, State.READY, "trigger1", 100.0)
        >>> t2 = StateTransition(State.READY, State.ACTIVE, "trigger2", 200.0)
        >>> t1 < t2
        True
        """
        return self.timestamp < other.timestamp

    @property
    def animation_type(self) -> str:
        """
        Get the type of animation associated with the transition based on the target state.

        Returns
        -------
        str
            Animation type, one of:
            - 'fade': For transitions to SHUTDOWN state
            - 'bounce': For transitions to BUSY state
            - 'slide': For all other transitions

        Notes
        -----
        These animation types are used in the UI to provide visual feedback about
        the nature of the state change. For example, a 'fade' animation for shutdown
        gives a sense of the system powering down.
        """
        if self.to_state == State.SHUTDOWN:
            return 'fade'
        elif self.to_state == State.BUSY:
            return 'bounce'
        else:
            return 'slide'

    @property
    def animation_duration(self) -> int:
        """
        Get the duration of the animation in milliseconds based on the target state.

        Returns
        -------
        int
            Animation duration in milliseconds:
            - 800ms for transitions to SHUTDOWN or CLEANUP states
            - 400ms for transitions to BUSY or LOADING states
            - 300ms for all other transitions

        Notes
        -----
        Longer durations are used for significant state changes (like shutdown),
        while shorter durations are used for more common transitions to maintain
        responsiveness.
        """
        if self.to_state in {State.SHUTDOWN, State.CLEANUP}:
            return 800
        elif self.to_state in {State.BUSY, State.LOADING}:
            return 400
        else:
            return 300

    def get_animation_properties(self) -> Dict[str, Any]:
        """
        Get a dictionary of animation properties for the transition.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing animation properties including:
            - type: Animation type (fade, bounce, slide)
            - duration: Animation duration in milliseconds
            - easing: CSS easing function ('ease-in-out')

        Examples
        --------
        >>> transition = StateTransition(State.IDLE, State.BUSY, "process_start", time.time())
        >>> props = transition.get_animation_properties()
        >>> print(props)
        {'type': 'bounce', 'duration': 400, 'easing': 'ease-in-out'}
        """
        return {
            'type': self.animation_type,
            'duration': self.animation_duration,
            'easing': 'ease-in-out'
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the StateTransition object into a dictionary representation.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the transition data, with keys:
            - from_state: Name of the source state
            - to_state: Name of the target state
            - trigger: The trigger that caused the transition
            - timestamp: The time when the transition occurred
            - metadata: Any additional metadata associated with the transition

        Notes
        -----
        This method is useful for serializing transitions for logging,
        event publishing, or saving to persistent storage.

        Examples
        --------
        >>> transition = StateTransition(State.IDLE, State.READY, "user_input", time.time())
        >>> data = transition.to_dict()
        >>> print(data['from_state'], data['to_state'])
        'IDLE' 'READY'
        """
        return {
            'from_state': self.from_state.name,
            'to_state': self.to_state.name,
            'trigger': self.trigger,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class StateManager(IStateProvider):
    """
    Manages state transitions in a finite state machine.

    StateManager is responsible for managing the current state of the system,
    controlling valid transitions between states, and executing registered handlers
    for state entries, exits, and transitions. It implements the IStateProvider
    interface, allowing other components to query the current state.

    The StateManager implements a modified State design pattern with additional
    support for transition validation, history tracking, and event publication.
    It serves as the central coordinator for the application's state machine.

    Parameters
    ----------
    initial_state : State, optional
        The initial state of the system, by default State.INIT
    event_bus : Any, optional
        An optional event bus for publishing state change events, by default None.
        If provided, the manager will publish events when state changes occur.

    Attributes
    ----------
    current_state : State
        The current state of the system.
    state_handlers : Dict[State, List[Tuple[Callable, bool]]]
        Handlers for entry and exit actions for each state.
        Each handler is associated with a state and a flag indicating
        whether it's an entry (True) or exit (False) handler.
    transition_handlers : Dict[Tuple[State, State], List[Callable]]
        Handlers for specific state transitions.
    valid_transitions : Dict[State, List[State]]
        A mapping of valid transitions for each state.
    transition_history : List[StateTransition]
        A history of state transitions.
    max_history_size : int
        The maximum size of the transition history.

    Examples
    --------
    >>> manager = StateManager(initial_state=State.INIT)
    >>> manager.register_state_handler(State.READY, lambda t: print("Entered READY state"), True)
    >>> manager.register_transition_handler(State.INIT, State.READY, lambda t: print("INIT to READY"))
    >>> manager.transition_to(State.READY, "startup_complete")
    INIT to READY
    Entered READY state
    True

    Notes
    -----
    The StateManager handles thread safety internally using a reentrant lock to ensure
    that state transitions are atomic operations, even in multi-threaded environments.

    See Also
    --------
    State : Enumeration of possible states
    StateTransition : Represents a transition between states
    IStateProvider : Interface implemented by StateManager
    """
    def __init__(self, initial_state: State = State.INIT, event_bus: Any = None) -> None:
        """
        Initialize a StateManager instance.

        Parameters
        ----------
        initial_state : State, optional
            The initial state of the system, by default State.INIT
        event_bus : Any, optional
            An optional event bus for publishing state change events, by default None.
            If provided, the manager will publish events when state changes occur.
        """
        self.current_state = initial_state
        self.event_bus = event_bus
        self.state_handlers: Dict[State, List[Tuple[Callable, bool]]] = {state: [] for state in State}
        self.transition_handlers: Dict[Tuple[State, State], List[Callable]] = {}
        self.logger = logging.getLogger('maggie.core.state.StateManager')
        self._lock = threading.RLock()

        # Define valid transitions
        self.valid_transitions = {
            State.INIT: [State.STARTUP, State.IDLE, State.SHUTDOWN],
            State.STARTUP: [State.IDLE, State.READY, State.CLEANUP, State.SHUTDOWN],
            State.IDLE: [State.STARTUP, State.READY, State.CLEANUP, State.SHUTDOWN],
            State.LOADING: [State.ACTIVE, State.READY, State.CLEANUP, State.SHUTDOWN],
            State.READY: [State.LOADING, State.ACTIVE, State.BUSY, State.CLEANUP, State.SHUTDOWN],
            State.ACTIVE: [State.READY, State.BUSY, State.CLEANUP, State.SHUTDOWN],
            State.BUSY: [State.READY, State.ACTIVE, State.CLEANUP, State.SHUTDOWN],
            State.CLEANUP: [State.IDLE, State.SHUTDOWN],
            State.SHUTDOWN: []
        }

        self.transition_history: List[StateTransition] = []
        self.max_history_size = 100
        self.logger.info(f"StateManager initialized with state: {initial_state.name}")

    def transition_to(self, new_state: State, trigger: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Transition to a new state if the transition is valid.

        This method attempts to transition from the current state to the specified
        new state. It validates the transition, executes appropriate handlers, and
        updates the transition history.

        Parameters
        ----------
        new_state : State
            The target state to transition to.
        trigger : str
            The event or action that triggered the transition.
        metadata : Dict[str, Any], optional
            Additional metadata to associate with the transition, by default None.

        Returns
        -------
        bool
            True if the transition was successful, False otherwise (e.g., if the
            transition is invalid or an error occurred during handler execution).

        Notes
        -----
        This method performs several steps:
        1. Validates that the transition from current_state to new_state is allowed
        2. Creates a StateTransition object with metadata
        3. Executes exit handlers for the current state
        4. Executes transition handlers for the specific transition
        5. Updates the current state
        6. Executes entry handlers for the new state
        7. Publishes a state change event if an event bus is available

        If any step fails, error information is logged and False is returned.

        Examples
        --------
        >>> manager = StateManager(initial_state=State.INIT)
        >>> manager.transition_to(State.STARTUP, "system_start", {"user_id": "12345"})
        True
        >>> manager.get_current_state()
        <State.STARTUP: 2>
        """
        with self._lock:
            # Check if already in the target state
            if new_state == self.current_state:
                self.logger.debug(f"Already in state {new_state.name}, ignoring transition")
                return True

            # Check if transition is valid
            if not self.is_valid_transition(self.current_state, new_state):
                error_message = f"Invalid transition from {self.current_state.name} to {new_state.name} (trigger: {trigger})"
                self.logger.warning(error_message)

                # Try to use error handling if available through abstraction layer
                try:
                    from maggie.utils.abstractions import get_error_handler
                    error_handler = get_error_handler()
                    if error_handler:
                        error_handler.record_error(
                            message=error_message,
                            category='STATE',
                            severity='ERROR',
                            source='StateManager.transition_to'
                        )
                except ImportError:
                    # Abstractions module not available
                    pass
                except Exception:
                    # Error handler not available or failed
                    pass

                return False

            # Perform the transition
            old_state = self.current_state
            self.current_state = new_state

            self.logger.info(f"State transition: {old_state.name} -> {new_state.name} (trigger: {trigger})")

            # Create transition object
            transition = StateTransition(
                from_state=old_state,
                to_state=new_state,
                trigger=trigger,
                timestamp=time.time(),
                metadata=metadata or {}
            )

            # Update history
            self.transition_history.append(transition)
            if len(self.transition_history) > self.max_history_size:
                self.transition_history = self.transition_history[-self.max_history_size:]

            try:
                # Execute handlers
                self._execute_state_handlers(old_state, 'exit', transition)
                self._execute_transition_handlers(old_state, new_state, transition)
                self._execute_state_handlers(new_state, 'entry', transition)

                # Publish state change event if event bus available
                if self.event_bus:
                    self.event_bus.publish('state_changed', transition)

                return True
            except Exception as e:
                self.logger.error(f"Error during state transition: {e}")
                return False

    def register_state_handler(self, state: State, handler: Callable, is_entry: bool = True) -> None:
        """
        Register a handler for a specific state.

        This method registers a callback function to be executed either when
        entering or exiting a specific state.

        Parameters
        ----------
        state : State
            The state to associate the handler with.
        handler : Callable
            The callback function to execute. It should accept a StateTransition
            object as its parameter.
        is_entry : bool, optional
            If True, the handler is executed when entering the state.
            If False, the handler is executed when exiting the state.
            By default True.

        Notes
        -----
        State handlers are useful for setting up or tearing down resources,
        updating UI elements, or performing other actions specific to a state.

        Examples
        --------
        >>> def on_enter_ready(transition):
        ...     print(f"Entered READY state from {transition.from_state.name}")
        >>> manager = StateManager()
        >>> manager.register_state_handler(State.READY, on_enter_ready, True)
        """
        with self._lock:
            if state not in self.state_handlers:
                self.state_handlers[state] = []

            self.state_handlers[state].append((handler, is_entry))
            self.logger.debug(f"Registered {'entry' if is_entry else 'exit'} handler for state: {state.name}")

    def register_transition_handler(self, from_state: State, to_state: State, handler: Callable) -> None:
        """
        Register a handler for a specific state transition.

        This method registers a callback function to be executed when
        transitioning from one specific state to another.

        Parameters
        ----------
        from_state : State
            The source state of the transition.
        to_state : State
            The target state of the transition.
        handler : Callable
            The callback function to execute. It should accept a StateTransition
            object as its parameter.

        Notes
        -----
        Transition handlers are useful for performing actions specific to a
        particular transition, such as resource allocation, validation, or
        specialized UI updates.

        Examples
        --------
        >>> def on_idle_to_ready(transition):
        ...     print(f"Transitioning from IDLE to READY (trigger: {transition.trigger})")
        >>> manager = StateManager()
        >>> manager.register_transition_handler(State.IDLE, State.READY, on_idle_to_ready)
        """
        with self._lock:
            transition_key = (from_state, to_state)
            if transition_key not in self.transition_handlers:
                self.transition_handlers[transition_key] = []

            self.transition_handlers[transition_key].append(handler)
            self.logger.debug(f"Registered transition handler for {from_state.name} -> {to_state.name}")

    def get_current_state(self) -> State:
        """
        Get the current state of the system.

        This method implements the IStateProvider interface.

        Returns
        -------
        State
            The current state of the system.

        Examples
        --------
        >>> manager = StateManager(initial_state=State.READY)
        >>> manager.get_current_state()
        <State.READY: 5>
        """
        with self._lock:
            return self.current_state

    def get_style_for_current_state(self) -> Dict[str, str]:
        """
        Get the style associated with the current state.

        Returns
        -------
        Dict[str, str]
            Dictionary containing CSS style properties for the current state.

        Examples
        --------
        >>> manager = StateManager(initial_state=State.READY)
        >>> style = manager.get_style_for_current_state()
        >>> print(style['background'])
        '#A5D6A7'
        """
        with self._lock:
            return self.current_state.get_style()

    def is_in_state(self, state: State) -> bool:
        """
        Check if the system is in a specific state.

        Parameters
        ----------
        state : State
            The state to check.

        Returns
        -------
        bool
            True if the current state matches the specified state, False otherwise.

        Examples
        --------
        >>> manager = StateManager(initial_state=State.READY)
        >>> manager.is_in_state(State.READY)
        True
        >>> manager.is_in_state(State.ACTIVE)
        False
        """
        with self._lock:
            return self.current_state == state

    def is_valid_transition(self, from_state: State, to_state: State) -> bool:
        """
        Check if a transition from one state to another is valid.

        Parameters
        ----------
        from_state : State
            The source state.
        to_state : State
            The target state.

        Returns
        -------
        bool
            True if the transition is valid, False otherwise.

        Notes
        -----
        Valid transitions are defined in the valid_transitions dictionary,
        which maps each state to a list of states it can transition to.

        Examples
        --------
        >>> manager = StateManager()
        >>> manager.is_valid_transition(State.IDLE, State.READY)
        True
        >>> manager.is_valid_transition(State.IDLE, State.BUSY)
        False
        """
        return to_state in self.valid_transitions.get(from_state, [])

    def get_valid_transitions(self, from_state: Optional[State] = None) -> List[State]:
        """
        Get a list of valid states that can be transitioned to from a specific state.

        Parameters
        ----------
        from_state : Optional[State], optional
            The source state, by default None.
            If None, the current state is used.

        Returns
        -------
        List[State]
            List of states that can be transitioned to from the specified state.

        Examples
        --------
        >>> manager = StateManager(initial_state=State.IDLE)
        >>> valid_transitions = manager.get_valid_transitions()
        >>> State.READY in valid_transitions
        True
        >>> State.BUSY in valid_transitions
        False
        """
        source_state = from_state if from_state is not None else self.current_state
        return self.valid_transitions.get(source_state, [])

    def get_transition_history(self, limit: int = 10) -> List[StateTransition]:
        """
        Get the most recent state transitions.

        Parameters
        ----------
        limit : int, optional
            Maximum number of transitions to return, by default 10.

        Returns
        -------
        List[StateTransition]
            List of the most recent state transitions, limited to the specified number.

        Notes
        -----
        The transitions are returned in chronological order, with the most recent
        transitions at the end of the list.

        Examples
        --------
        >>> manager = StateManager(initial_state=State.INIT)
        >>> manager.transition_to(State.STARTUP, "system_start")
        True
        >>> manager.transition_to(State.IDLE, "startup_complete")
        True
        >>> history = manager.get_transition_history()
        >>> len(history)
        2
        >>> history[0].from_state, history[0].to_state
        (<State.INIT: 1>, <State.STARTUP: 2>)
        """
        with self._lock:
            return self.transition_history[-limit:]

    def _execute_state_handlers(self, state: State, handler_type: str, transition: StateTransition) -> None:
        """Execute entry or exit handlers for a specific state."""
        is_entry = handler_type == 'entry'

        for (handler, handler_is_entry) in self.state_handlers.get(state, []):
            if handler_is_entry == is_entry:
                try:
                    handler(transition)
                except Exception as e:
                    self.logger.error(f"Error in state {handler_type} handler for {state.name}: {e}")

    def _execute_transition_handlers(self, from_state: State, to_state: State, transition: StateTransition) -> None:
        """Execute handlers for a specific state transition."""
        transition_key = (from_state, to_state)

        for handler in self.transition_handlers.get(transition_key, []):
            try:
                handler(transition)
            except Exception as e:
                self.logger.error(f"Error in transition handler for {from_state.name} -> {to_state.name}: {e}")

    def reset(self, new_state: State = State.INIT) -> None:
        """
        Reset the state manager to a new state.

        This method sets the current state to the specified new state without
        going through the normal transition process. It also clears the transition
        history.

        Parameters
        ----------
        new_state : State, optional
            The state to reset to, by default State.INIT.

        Notes
        -----
        This method should only be used in exceptional circumstances, such as
        during system initialization or recovery from an error state, as it
        bypasses the normal transition validation and handler execution.

        If an event bus is available, a 'state_manager_reset' event is published.

        Examples
        --------
        >>> manager = StateManager(initial_state=State.ACTIVE)
        >>> manager.reset(State.INIT)
        >>> manager.get_current_state()
        <State.INIT: 1>
        """
        with self._lock:
            old_state = self.current_state
            self.current_state = new_state
            self.logger.info(f"State manager reset: {old_state.name} -> {new_state.name}")
            self.transition_history = []

            if self.event_bus:
                self.event_bus.publish('state_manager_reset', new_state)


class StateAwareComponent:
    """
    Base class for components that need to respond to state changes.

    This class provides a framework for components that need to be aware of
    and respond to state changes in the system. It automatically registers
    handlers for state transitions and provides methods for updating UI elements
    based on the current state.

    StateAwareComponent follows the Observer pattern, observing state changes
    in the StateManager and responding appropriately. It also provides a convenient
    way to register state-specific handlers using naming conventions.

    Parameters
    ----------
    state_manager : StateManager
        The state manager to observe and register handlers with.

    Attributes
    ----------
    state_manager : StateManager
        The state manager being observed.
    logger : Logger
        Logger instance for the component.

    Notes
    -----
    Components that extend this class can define methods with specific naming
    patterns to automatically register them as state handlers:
    - `on_enter_<state>`: Called when entering a state
    - `on_exit_<state>`: Called when exiting a state

    These methods should accept a StateTransition object as their parameter.

    Examples
    --------
    >>> class MyComponent(StateAwareComponent):
    ...     def __init__(self, state_manager):
    ...         super().__init__(state_manager)
    ...
    ...     def on_enter_ready(self, transition):
    ...         print(f"MyComponent: Entered READY state from {transition.from_state.name}")
    ...
    ...     def on_exit_active(self, transition):
    ...         print(f"MyComponent: Exited ACTIVE state to {transition.to_state.name}")

    See Also
    --------
    StateManager : Manages state transitions
    State : Enumeration of possible states
    StateTransition : Represents a transition between states
    """
    def __init__(self, state_manager: StateManager) -> None:
        """
        Initialize a StateAwareComponent instance.

        Parameters
        ----------
        state_manager : StateManager
            The state manager to observe and register handlers with.
        """
        self.state_manager = state_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self._registered_handlers = []
        self._register_state_handlers()

    def _register_state_handlers(self) -> None:
        """
        Register state entry and exit handlers.

        This method automatically registers methods named `on_enter_<state>` and
        `on_exit_<state>` as handlers for the corresponding states.

        Notes
        -----
        For each state in the State enum, this method looks for methods in the
        component class with names following the patterns:
        - `on_enter_<state>`: Registered as an entry handler for the state
        - `on_exit_<state>`: Registered as an exit handler for the state

        For example, a method named `on_enter_ready` would be registered as an
        entry handler for the READY state.
        """
        for state in State:
            method_name = f"on_enter_{state.name.lower()}"
            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                handler = getattr(self, method_name)
                self.state_manager.register_state_handler(state, handler, True)
                self._registered_handlers.append((state, handler, True))

            method_name = f"on_exit_{state.name.lower()}"
            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                handler = getattr(self, method_name)
                self.state_manager.register_state_handler(state, handler, False)
                self._registered_handlers.append((state, handler, False))

    def handle_state_change(self, transition: StateTransition) -> None:
        """
        Handle a state change.

        This method is called whenever the state changes. It can be overridden
        in subclasses to provide custom behavior during state transitions.

        Parameters
        ----------
        transition : StateTransition
            The state transition that occurred.

        Notes
        -----
        This method is intended to be overridden in subclasses to provide
        state-specific behavior that doesn't fit neatly into the on_enter/on_exit
        pattern.
        """
        pass

    def on_state_entry(self, state: State, transition: StateTransition) -> None:
        """
        Handle state entry.

        This method is called when entering a state. It can be overridden
        in subclasses to define behavior upon state entry.

        Parameters
        ----------
        state : State
            The state being entered.
        transition : StateTransition
            The state transition that led to this state.

        Notes
        -----
        This method provides a more flexible alternative to the naming convention
        approach for handling state entries.
        """
        pass

    def on_state_exit(self, state: State, transition: StateTransition) -> None:
        """
        Handle state exit.

        This method is called when exiting a state. It can be overridden
        in subclasses to define behavior upon state exit.

        Parameters
        ----------
        state : State
            The state being exited.
        transition : StateTransition
            The state transition that is causing the exit.

        Notes
        -----
        This method provides a more flexible alternative to the naming convention
        approach for handling state exits.
        """
        pass

    def get_component_state(self) -> Dict[str, Any]:
        """
        Get the current component state.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing state information for the component, including:
            - system_state: The current system state
            - component_type: The type of the component

        Notes
        -----
        This method is useful for serializing the component's state for logging,
        persistence, or UI updates.

        Examples
        --------
        >>> component = MyComponent(state_manager)
        >>> state_info = component.get_component_state()
        >>> print(state_info)
        {'system_state': 'READY', 'component_type': 'MyComponent'}
        """
        return {
            'system_state': self.state_manager.get_current_state().name,
            'component_type': self.__class__.__name__
        }

    def update_ui(self, state: State) -> Dict[str, Any]:
        """
        Update the UI for a specific state.

        This method returns a dictionary of UI properties for the specified state.

        Parameters
        ----------
        state : State
            The state to get UI properties for.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing UI properties for the state, including:
            - state: The state name
            - display_name: The human-readable state name
            - style: CSS style properties for the state

        Notes
        -----
        This method is intended to be used by UI components to update their
        appearance based on the current state.

        Examples
        --------
        >>> component = MyComponent(state_manager)
        >>> ui_props = component.update_ui(State.READY)
        >>> print(ui_props)
        {'state': 'READY', 'display_name': 'Ready', 'style': {'background': '#A5D6A7', ...}}
        """
        return {
            'state': state.name,
            'display_name': state.display_name,
            'style': state.get_style()
        }