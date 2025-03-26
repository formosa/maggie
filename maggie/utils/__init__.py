"""
Maggie AI Assistant Utilities Package
====================================

Core utilities package for the Maggie AI Assistant, providing support 
modules for resource management, configuration, and user interface.

This package offers hardware-aware optimization, configuration management,
and utility functions specifically tuned for AMD Ryzen 9 5900X and 
NVIDIA RTX 3080 hardware, enabling efficient resource usage across the system.
"""

# Use lazy imports to prevent circular dependencies
def get_resource_manager():
    """
    Returns the ResourceManager class, not an instance.
    This prevents circular dependency issues during initialization.
    """
    try:
        from maggie.utils.resource.manager import ResourceManager
        return ResourceManager
    except ImportError:
        import logging
        logging.getLogger(__name__).error("Failed to import ResourceManager")
        return None

# Use lazy import for MainWindow to avoid circular dependency
def get_main_window():
    from maggie.utils.gui import MainWindow
    return MainWindow

# Direct imports for modules that don't create circular dependencies
from maggie.utils.config.manager import ConfigManager

__all__ = ['get_resource_manager', 'ConfigManager', 'get_main_window']