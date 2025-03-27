from typing import Dict, Any, Optional, List, Callable, Type, TypeVar, Set, Tuple, cast

T = TypeVar('T')

class ServiceLocator:
    """
    Service Locator implementation for the Maggie AI Assistant.
    
    This class implements the Service Locator pattern, a design pattern used to encapsulate
    the processes involved in obtaining service objects with a strong abstraction layer.
    It provides a centralized registry of services that components can access without
    tight coupling to concrete implementations.
    
    The implementation provides additional features beyond a basic service locator:
    
    1. State-aware service access - services can be constrained to specific application states
    2. Transition-specific services - services can be registered for specific state transitions
    3. Type-safe service retrieval - services can be retrieved with type checking
    
    References
    ----------
    .. [1] Fowler, Martin. "Inversion of Control Containers and the Dependency Injection Pattern"
           https://martinfowler.com/articles/injection.html
    .. [2] Microsoft patterns & practices. "Service Locator Pattern"
           https://learn.microsoft.com/en-us/previous-versions/msp-n-p/ff648968(v=pandp.10)
    
    Notes
    -----
    While the Service Locator pattern provides flexibility and loose coupling, it has been
    criticized for potentially hiding dependencies, making it harder to trace dependencies 
    in the codebase, and complicating unit testing. Consider these trade-offs when using 
    this pattern versus Dependency Injection.
    
    In the Maggie AI system, this pattern is particularly valuable due to:
    - Its ability to handle state-dependent service availability
    - Support for dynamic service creation and replacement
    - Centralized service management across a complex, state-driven application
    
    Examples
    --------
    Basic service registration and retrieval:
    
    >>> from maggie.service.locator import ServiceLocator
    >>> # Register a service
    >>> ServiceLocator.register("config_manager", config_manager)
    >>> # Retrieve a service
    >>> config = ServiceLocator.get("config_manager")
    
    State-aware service registration:
    
    >>> from maggie.core.state import State
    >>> # Register service only available in ACTIVE and READY states
    >>> ServiceLocator.register("tts_processor", tts_processor, 
    ...                         available_states=[State.ACTIVE, State.READY])
    >>> # Service will only be accessible when current state is ACTIVE or READY
    """
    
    _services: Dict[str, Any] = {}
    _state_constraints: Dict[str, Set] = {}
    _transition_constraints: Dict[str, List[Tuple]] = {}
    _current_state: Optional[Any] = None
    _last_transition: Optional[Tuple] = None
    _logger: Any

    @classmethod
    def register(cls, name: str, service: Any, available_states: Optional[List] = None) -> None:
        """
        Register a service with the locator.
        
        Registers a service instance with the given name. Optionally, the service
        can be constrained to be available only in specific application states.
        
        Parameters
        ----------
        name : str
            The name to register the service under. This name will be used to
            retrieve the service later.
        service : Any
            The service instance to register.
        available_states : Optional[List], default=None
            If provided, restricts the service to only be available when the 
            application is in one of these states. If None, the service is 
            available in all states.
            
        Returns
        -------
        None
        
        Examples
        --------
        >>> # Register a service available in all states
        >>> ServiceLocator.register("database", db_connection)
        >>> 
        >>> # Register a service only available in ACTIVE and READY states
        >>> from maggie.core.state import State
        >>> ServiceLocator.register("speech_recognizer", speech_recognizer, 
        ...                         available_states=[State.ACTIVE, State.READY])
        """
        ...
    
    @classmethod
    def register_for_transition(cls, name: str, service: Any, transitions: List[Tuple]) -> None:
        """
        Register a service for specific state transitions.
        
        Registers a service that will only be available during specific state transitions,
        such as when moving from one application state to another.
        
        Parameters
        ----------
        name : str
            The name to register the service under.
        service : Any
            The service instance to register.
        transitions : List[Tuple]
            List of state transition tuples (from_state, to_state) during which 
            the service should be available.
            
        Returns
        -------
        None
        
        Examples
        --------
        >>> from maggie.core.state import State
        >>> # Register a logger service specifically for transitions from IDLE to ACTIVE
        >>> transition_logger = TransitionLogger()
        >>> ServiceLocator.register_for_transition(
        ...     "transition_logger", 
        ...     transition_logger,
        ...     [(State.IDLE, State.ACTIVE)]
        ... )
        """
        ...
    
    @classmethod
    def update_state(cls, new_state) -> None:
        """
        Update the current state of the service locator.
        
        This method should be called whenever the application state changes, 
        to ensure that state-constrained services are properly managed.
        
        Parameters
        ----------
        new_state : Any
            The new state of the application. Expected to be an instance of a State enum.
            
        Returns
        -------
        None
        
        Notes
        -----
        This method updates both the current state and records the last transition 
        (from previous state to new state), which is used for transition-constrained services.
        
        Examples
        --------
        >>> from maggie.core.state import State
        >>> # When application transitions to a new state
        >>> ServiceLocator.update_state(State.ACTIVE)
        """
        ...
    
    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """
        Get a service by name.
        
        Retrieves a registered service by its name. The service will only be returned
        if it is available in the current state, or if it has no state constraints.
        
        Parameters
        ----------
        name : str
            The name of the service to retrieve.
            
        Returns
        -------
        Optional[Any]
            The service instance if found and available in the current state, 
            otherwise None.
            
        Notes
        -----
        This method checks both state constraints and transition constraints before
        returning a service. If a service is constrained to specific states or transitions
        and the current state/transition doesn't match, None will be returned even if
        the service is registered.
        
        Examples
        --------
        >>> # Get a service
        >>> tts_processor = ServiceLocator.get("tts_processor")
        >>> if tts_processor:
        ...     tts_processor.speak("Hello")
        ... else:
        ...     print("TTS processor not available in current state")
        """
        ...
    
    @classmethod
    def get_typed(cls, name: str, service_type: Type[T]) -> Optional[T]:
        """
        Get a service by name with type checking.
        
        Similar to `get()`, but performs type checking to ensure the returned service
        is of the expected type.
        
        Parameters
        ----------
        name : str
            The name of the service to retrieve.
        service_type : Type[T]
            The expected type of the service.
            
        Returns
        -------
        Optional[T]
            The service instance if found, available in the current state, and of the
            correct type. Otherwise, None.
            
        Examples
        --------
        >>> from maggie.service.tts.processor import TTSProcessor
        >>> # Get a service with type checking
        >>> tts = ServiceLocator.get_typed("tts_processor", TTSProcessor)
        >>> if tts:
        ...     tts.speak("Hello with type safety")
        """
        ...
    
    @classmethod
    def has_service(cls, name: str) -> bool:
        """
        Check if a service exists in the registry.
        
        Checks if a service with the given name is registered, regardless of
        state constraints.
        
        Parameters
        ----------
        name : str
            The name of the service to check.
            
        Returns
        -------
        bool
            True if the service is registered, False otherwise.
            
        Examples
        --------
        >>> # Check if a service exists
        >>> if ServiceLocator.has_service("config_manager"):
        ...     config = ServiceLocator.get("config_manager")
        ... else:
        ...     config = default_config
        """
        ...
    
    @classmethod
    def get_or_create(cls, name: str, factory: Callable[[], T], available_states: Optional[List] = None) -> T:
        """
        Get a service or create it if it doesn't exist.
        
        Attempts to retrieve a service by name. If the service doesn't exist,
        creates it using the provided factory function and registers it.
        
        Parameters
        ----------
        name : str
            The name of the service to retrieve or create.
        factory : Callable[[], T]
            A factory function that creates the service if it doesn't exist.
        available_states : Optional[List], default=None
            If provided and the service needs to be created, restricts the new
            service to only be available in these states.
            
        Returns
        -------
        T
            The existing or newly created service.
            
        Examples
        --------
        >>> # Get a database connection or create a new one
        >>> db = ServiceLocator.get_or_create(
        ...     "database", 
        ...     lambda: Database.connect("localhost", "mydb")
        ... )
        """
        ...
    
    @classmethod
    def get_available_services(cls, state=None) -> List[str]:
        """
        Get a list of services available in a specific state.
        
        Returns the names of services that are available in the given state,
        or in the current state if no state is specified.
        
        Parameters
        ----------
        state : Any, default=None
            The state to check availability for. If None, uses the current state.
            
        Returns
        -------
        List[str]
            A list of service names available in the specified or current state.
            
        Examples
        --------
        >>> from maggie.core.state import State
        >>> # Get services available in the ACTIVE state
        >>> active_services = ServiceLocator.get_available_services(State.ACTIVE)
        >>> print(f"Available services in ACTIVE state: {active_services}")
        """
        ...
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered services.
        
        Removes all services, state constraints, and transition constraints from
        the registry, effectively resetting the service locator to its initial state.
        
        Returns
        -------
        None
        
        Examples
        --------
        >>> # Clear all services during application shutdown
        >>> ServiceLocator.clear()
        """
        ...
    
    @classmethod
    def list_services(cls) -> List[str]:
        """
        List all registered services.
        
        Returns a list of names of all registered services, regardless of state
        constraints.
        
        Returns
        -------
        List[str]
            A list of all service names registered in the locator.
            
        Examples
        --------
        >>> # Get all registered services
        >>> all_services = ServiceLocator.list_services()
        >>> print(f"All registered services: {all_services}")
        """
        ...