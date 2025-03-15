"""
Maggie AI Assistant - Configuration Module
=========================================

Core configuration management module for the Maggie AI Assistant, providing
comprehensive configuration loading, validation, and optimization capabilities.

This module offers a unified approach to configuration management with hardware-specific
optimization profiles for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.
"""

from maggie.utils.config.manager import ConfigManager
from maggie.utils.config.validator import ConfigValidator

__all__ = ['ConfigManager', 'ConfigValidator']