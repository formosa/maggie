"""
Maggie AI Assistant Resource Management Package
============================================

Comprehensive resource management for the Maggie AI Assistant.

This package provides hardware detection, optimization, and monitoring
capabilities optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.
"""

# Import detector first as others depend on it
from maggie.utils.resource.detector import HardwareDetector

# Use lazy imports for modules that may create circular dependencies
def get_hardware_optimizer():
    from maggie.utils.resource.optimizer import HardwareOptimizer
    return HardwareOptimizer

def get_resource_monitor():
    from maggie.utils.resource.monitor import ResourceMonitor
    return ResourceMonitor

def get_resource_manager():
    from maggie.utils.resource.manager import ResourceManager
    return ResourceManager

__all__ = [
    'HardwareDetector',
    'get_hardware_optimizer', 
    'get_resource_monitor',
    'get_resource_manager'
]