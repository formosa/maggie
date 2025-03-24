"""
Maggie AI Assistant Utilities Package
====================================

Core utilities package for the Maggie AI Assistant, providing support 
modules for resource management, configuration, and user interface.

This package offers hardware-aware optimization, configuration management,
and utility functions specifically tuned for AMD Ryzen 9 5900X and 
NVIDIA RTX 3080 hardware, enabling efficient resource usage across the system.
"""

from maggie.utils.resource.manager import ResourceManager
from maggie.utils.config.manager import ConfigManager
from maggie.utils.gui import MainWindow

__all__ = ['ResourceManager', 'ConfigManager', 'MainWindow']