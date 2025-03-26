"""
Maggie AI Assistant - Component Initialization
================================================

This module handles proper initialization of core components to
avoid circular dependencies and ensure correct order of operations.
"""

import logging
from typing import Dict, Any, Optional

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
    Initialize all components in the correct order.
    
    Args:
        config: Configuration dictionary
        debug: Whether to enable debug mode
    
    Returns:
        Dictionary of initialized components
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
        
        # Start the event bus
        event_bus.start()
        
        logger.info("All components initialized successfully")
        return components
    
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return {}