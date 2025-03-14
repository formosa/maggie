"""
Maggie AI Assistant Package
===========================

Core package for the Maggie AI Assistant, implementing a Finite State Machine
architecture with event-driven state transitions.

This package provides an intelligent assistant with voice capabilities,
LLM integration, and extensible utility framework. It's specifically
optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.

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

from .core import MaggieAI, State, StateTransition, EventBus
__all__ = ['MaggieAI', 'State', 'StateTransition', 'EventBus']