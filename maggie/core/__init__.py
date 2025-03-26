"""
Maggie AI Assistant Core Module
===============================

Core module exporting the main classes for the Maggie AI Assistant.
"""

# Import core components individually to avoid circular imports
from maggie.core.state import State, StateTransition
from maggie.core.event import EventBus, EventEmitter, EventListener

# Only import MaggieAI after other components to prevent circular references
from maggie.core.app import MaggieAI

__all__ = ['MaggieAI', 'State', 'StateTransition', 'EventBus', 'EventEmitter', 'EventListener']