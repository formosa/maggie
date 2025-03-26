# maggie/extensions/__init__.py
"""
Maggie AI Extensions Package
============================

This package contains extension modules that add functionality to the core
Maggie AI Assistant. Each subpackage is a self-contained extension that
implements the extension interface defined in maggie.extensions.base.

Extensions can be enabled or disabled through configuration and are loaded
dynamically at runtime.
"""

from maggie.extensions.base import ExtensionBase
from maggie.extensions.registry import ExtensionRegistry

__all__ = ['ExtensionBase', 'ExtensionRegistry']