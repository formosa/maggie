"""
Maggie AI Assistant Utilities Package
====================================

Core utilities package for the Maggie AI Assistant, providing support 
modules for, audio attention, hardware management, configuration, 
logging, and services.

This package offers hardware-aware optimization, configuration management,
and service locator utilities specifically tuned for AMD Ryzen 9 5900X and 
NVIDIA RTX 3080 hardware, enabling efficient resource usage across the system.
"""

from maggie.utils.service_locator import ServiceLocator
from maggie.utils.resource.manager import ResourceManager
from maggie.utils.hardware.manager import HardwareManager
from maggie.utils.config.manager import ConfigManager
from maggie.utils.stt import WakeWordDetector
from maggie.utils.gui import MainWindow

__all__ = ['ServiceLocator', 'ResourceManager', 'HardwareManager', 'ConfigManager','WakeWordDetector', 'MainWindow']