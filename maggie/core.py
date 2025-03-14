"""
Maggie AI Assistant Core Module
===============================

Core module exporting the main classes for the Maggie AI Assistant.

This module re-exports the core classes from the maggie module for 
easier imports throughout the application, preventing circular imports
and providing a clean public API.

Classes
-------
MaggieAI : class
    Main class implementing the Maggie AI Assistant.
State : enum
    Enumeration of possible states in the FSM.
StateTransition : dataclass
    Data structure for state transitions.
EventBus : class
    Event management system for component communication.
"""

from ._maggie import MaggieAI, State, StateTransition, EventBus