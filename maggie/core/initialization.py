"""
Maggie AI Assistant - Component Initialization
================================================

This module manages the careful initialization of core system components to
avoid circular dependencies and enforce proper initialization order. It follows
the Dependency Inversion Principle (DIP) and implements a variant of the
Initialization-on-demand design pattern.

The initialization process creates and connects system components through
a capability registry mechanism, allowing them to discover each other at runtime
without creating hard dependencies. This architecture enables a modular and
extensible system where components remain decoupled but can still interact
when needed.
"""


from typing import Dict, Any, Optional

from maggie.utils.logging import LoggingManager
from maggie.utils.error_handling import ErrorHandler

from maggie.utils.abstractions import (
    ILoggerProvider, 
    IErrorHandler, 
    IEventPublisher, 
    IStateProvider,
    CapabilityRegistry
)
from maggie.utils.adapters import (
    LoggingManagerAdapter,
    ErrorHandlerAdapter,
    EventBusAdapter,
    StateManagerAdapter
)

def initialize_components(config: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    """
    Initialize all components in the correct order to prevent circular dependencies.

    This function implements a deterministic initialization sequence that ensures
    components are created and interconnected in the proper order. The initialization
    follows these steps:
    
    1. Initialize the capability registry (service locator pattern)
    2. Set up the logging system (LoggingManager)
    3. Create error handling infrastructure
    4. Initialize event bus for system-wide messaging
    5. Set up state management system
    6. Create adapter layers to implement abstract interfaces
    7. Register adapters with capability registry
    8. Initialize the main MaggieAI instance
    9. Start the event processing loop

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing system-wide settings for all components.
        Must include sections for logging, state management, error handling, etc.
        Example keys: 'logging.path', 'logging.level', 'cpu.max_threads', etc.
    debug : bool, optional
        Whether to enable debug mode, which provides more detailed logging,
        additional error information, and potentially slower but more traceable
        execution. Default is False.

    Returns
    -------
    Dict[str, Any]
        Dictionary of initialized components where keys are component names
        and values are the actual component instances. Typical keys include:
        - 'registry': The CapabilityRegistry instance
        - 'logging_manager': LoggingManager instance
        - 'error_handler': ErrorHandlerAdapter instance
        - 'event_bus': EventBus instance
        - 'state_manager': StateManager instance
        - various adapters: 'logging_adapter', 'event_bus_adapter', etc.
        - 'maggie_ai': The main MaggieAI instance

    Raises
    ------
    RuntimeError
        If component initialization fails due to missing dependencies
        or configuration errors.
    ImportError
        If required modules cannot be imported.

    See Also
    --------
    * `Dependency Inversion Principle <https://en.wikipedia.org/wiki/Dependency_inversion_principle>`_
    * `Capability-based security <https://en.wikipedia.org/wiki/Capability-based_security>`_
    * `Initialization-on-demand holder idiom <https://en.wikipedia.org/wiki/Initialization-on-demand_holder_idiom>`_

    Notes
    -----
    This function must be called before any other system operations.
    Components are initialized in a specific order to resolve dependencies:
    base services first (logging, error handling), then communication layers
    (event bus), then state management, and finally application components.
    
    The capability registry pattern (implemented via the CapabilityRegistry class)
    allows components to discover and use each other at runtime without creating
    compile-time dependencies. This approach implements the "dependency inversion
    principle" from SOLID design principles.
    
    See: https://en.wikipedia.org/wiki/SOLID
    
    This module is critical for the proper startup of the Maggie AI system and must be
    carefully maintained. Altering the order of component initialization may lead to
    circular dependencies, deadlocks, or system instability.

    Examples
    --------
    >>> from maggie.core.initialization import initialize_components
    >>> config = {
    ...     "logging": {"path": "logs", "level": "INFO"},
    ...     "cpu": {"max_threads": 8},
    ...     "config_path": "config.yaml"
    ... }
    >>> components = initialize_components(config, debug=True)
    >>> maggie_ai = components['maggie_ai']
    >>> maggie_ai.start()
    """
    components = {}
    logger = logging.getLogger('maggie.initialization')
    
    try:
        # Step 1: Initialize capability registry
        registry = CapabilityRegistry.get_instance()
        components['registry'] = registry
        
        # Step 2: Initialize LoggingManager
        from maggie.utils.logging import LoggingManager
        logging_mgr = LoggingManager.initialize(config)
        components['logging_manager'] = logging_mgr
        
        # Step 3: Initialize ErrorHandlerAdapter
        error_handler = ErrorHandlerAdapter()
        components['error_handler'] = error_handler
        
        # Step 4: Initialize EventBus
        from maggie.core.event import EventBus
        event_bus = EventBus()
        components['event_bus'] = event_bus
        
        # Step 5: Initialize StateManager
        from maggie.core.state import StateManager, State
        state_manager = StateManager(State.INIT, event_bus)
        components['state_manager'] = state_manager
        
        # Step 6: Create and register adapters
        logging_adapter = LoggingManagerAdapter(logging_mgr)
        components['logging_adapter'] = logging_adapter
        
        event_bus_adapter = EventBusAdapter(event_bus)
        components['event_bus_adapter'] = event_bus_adapter
        
        state_manager_adapter = StateManagerAdapter(state_manager)
        components['state_manager_adapter'] = state_manager_adapter
        
        # Step 7: Enhance LoggingManager with event publisher and state provider
        logging_mgr.enhance_with_event_publisher(event_bus_adapter)
        logging_mgr.enhance_with_state_provider(state_manager_adapter)
        
        # Step 8: Initialize MaggieAI
        from maggie.core.app import MaggieAI
        config_path = config.get('config_path', 'config.yaml')
        maggie_ai = MaggieAI(config_path)
        components['maggie_ai'] = maggie_ai
        
        # Step 9: Start the event bus
        event_bus.start()
        
        logger.info("All components initialized successfully")
        return components
    
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return {}