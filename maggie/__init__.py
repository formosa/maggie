"""
Maggie AI Assistant Package
===========================

Core package for the Maggie AI Assistant, implementing a Finite State Machine
architecture with event-driven state transitions.

This package provides an intelligent assistant with voice capabilities,
LLM integration, and modular extensions framework. It's specifically
optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.
"""

from maggie.core import MaggieAI, State, StateTransition, EventBus
__all__ = ['MaggieAI', 'State', 'StateTransition', 'EventBus']