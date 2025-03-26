"""
Maggie AI Assistant Resource Management Package
============================================

Comprehensive resource management for the Maggie AI Assistant.

This package provides hardware detection, optimization, and monitoring
capabilities optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.
"""

# Import detector first as others depend on it
from maggie.utils.resource.detector import HardwareDetector

# Import optimizer and monitor which depend on detector
from maggie.utils.resource.optimizer import HardwareOptimizer
from maggie.utils.resource.monitor import ResourceMonitor

# Import manager last as it depends on all other components
from maggie.utils.resource.manager import ResourceManager

__all__ = [
    'HardwareDetector',
    'HardwareOptimizer', 
    'ResourceMonitor',
    'ResourceManager'
]