"""
Maggie AI Assistant Resource Management Package
============================================

Comprehensive resource management for the Maggie AI Assistant.

This package provides hardware detection, optimization, and monitoring
capabilities optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.
"""

from maggie.utils.resource.manager import ResourceManager
from maggie.utils.resource.detector import HardwareDetector
from maggie.utils.resource.optimizer import HardwareOptimizer
from maggie.utils.resource.monitor import ResourceMonitor

__all__ = ['ResourceManager', 'HardwareDetector', 'HardwareOptimizer', 'ResourceMonitor']