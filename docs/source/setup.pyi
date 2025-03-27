#!/usr/bin/env python3
"""
Maggie AI Assistant - Setup Script
==================================

Defines the package metadata and installation requirements for the Maggie AI Assistant,
a voice-controlled AI assistant application that integrates speech recognition, text-to-speech,
and language model capabilities optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.

This setup script follows the setuptools framework to provide a standardized way to build,
distribute, and install the Maggie AI Assistant package. It defines the package metadata,
required dependencies, optional development dependencies, and Python version requirements.

The package uses a finite state machine architecture with event-driven communication and
hardware-specific optimizations for high-performance voice assistant functionality.

References
----------
.. [1] setuptools documentation
       https://setuptools.pypa.io/en/latest/setuptools.html
.. [2] Python Packaging User Guide
       https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/
.. [3] PyPI Classifiers
       https://pypi.org/classifiers/
.. [4] Semantic Versioning
       https://semver.org/

Examples
--------
Install in development mode:

>>> pip install -e .

Build distribution package:

>>> python setup.py sdist bdist_wheel

Install from source with development dependencies:

>>> pip install -e .[dev]

Notes
-----
The package requires Python 3.10 and is not compatible with Python 3.11+.
This is due to specific version requirements of some dependencies and
the use of Python 3.10-specific language features.

The hardware-specific optimizations are implemented through configuration
profiles in the resource management module, providing automatic detection
and configuration for AMD Ryzen 9 5900X CPUs and NVIDIA RTX 3080 GPUs.
"""

from setuptools import setup as _setup
from setuptools import find_packages as _find_packages
from typing import List, Dict, Optional, Union, Any, Callable

def setup(
    name: str = "maggie",
    version: str = "0.1.0",
    description: str = "Maggie AI Assistant optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080",
    author: str = "Maggie Development Team",
    packages: List[str] = None,
    install_requires: List[str] = None,
    extras_require: Dict[str, List[str]] = None,
    python_requires: str = ">=3.10, <3.11",
    classifiers: List[str] = None
) -> None:
    """
    Configure and register the Maggie AI Assistant package with setuptools.
    
    This function encapsulates the setuptools.setup() call with predefined
    parameters for the Maggie AI project. It defines metadata about the package,
    its dependencies, and installation requirements.
    
    The setup function implements the Python packaging standard defined in PEP 517/PEP 518,
    using setuptools as the build backend. This enables installation through pip,
    building wheel and source distributions, and managing dependencies.
    
    Parameters
    ----------
    name : str, optional
        The name of the package as it will appear in the Python Package Index (PyPI).
        Default is "maggie".
    version : str, optional
        The package version following semantic versioning (major.minor.patch).
        Default is "0.1.0".
    description : str, optional
        A brief description of the package's purpose and features.
        Default is "Maggie AI Assistant optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080".
    author : str, optional
        The package author or organization name.
        Default is "Maggie Development Team".
    packages : List[str], optional
        List of package names to include, or None to use find_packages() to auto-discover.
        Default is None, which will use find_packages() with include=["maggie", "maggie.*"].
    install_requires : List[str], optional
        List of package dependencies required for installation.
        Default is a predefined list of specific package versions.
    extras_require : Dict[str, List[str]], optional
        Dictionary mapping extra feature names to their additional dependencies.
        Default includes a "dev" extra with testing and linting packages.
    python_requires : str, optional
        Python version specification string.
        Default is ">=3.10, <3.11" as the package requires Python 3.10.
    classifiers : List[str], optional
        List of PyPI classifiers for categorizing the package.
        Default includes development status, audience, license, and Python version.
    
    Returns
    -------
    None
        The function calls setuptools.setup() which registers the package
        with the Python packaging system but does not return a value.
    
    Notes
    -----
    The default dependencies include:
    
    - PyYAML: For configuration file parsing
    - torch and torchaudio: For neural network inference and audio processing
    - numpy: For numerical computations
    - loguru: For structured logging
    - PySide6: For GUI components
    - faster-whisper: For speech recognition
    - pvporcupine: For wake word detection
    - ctransformers: For language model inference
    - transitions: For finite state machine implementation
    - python-docx: For document generation in the recipe creator extension
    
    Examples
    --------
    Basic usage with defaults:
    
    >>> from setuptools import setup, find_packages
    >>> setup()
    
    Custom version and additional classifiers:
    
    >>> setup(
    ...     version="0.2.0",
    ...     classifiers=[
    ...         "Development Status :: 4 - Beta",
    ...         "Environment :: Desktop Environment",
    ...         "Topic :: Text Processing :: Linguistic"
    ...     ]
    ... )
    """
    ...

def find_packages(
    where: str = ".",
    exclude: List[str] = None,
    include: List[str] = None
) -> List[str]:
    """
    Discover and return a list of all Python packages in the specified directory.
    
    This function is a proxy to setuptools.find_packages() which auto-discovers
    Python packages by searching for __init__.py files. It's used to automatically
    build the list of packages to include in the distribution without manually
    specifying each one.
    
    Parameters
    ----------
    where : str, optional
        Root directory to search for packages.
        Default is "." (the current directory).
    exclude : List[str], optional
        List of package names or patterns to exclude from the result.
        Default is None, which excludes no packages.
    include : List[str], optional
        List of package names or patterns that must be included in the result.
        Default is None. In the Maggie setup, this defaults to ["maggie", "maggie.*"].
    
    Returns
    -------
    List[str]
        List of package names discovered in the specified directory,
        filtered by the include and exclude patterns.
    
    Notes
    -----
    Package discovery follows these rules:
    
    1. Search for directories containing an __init__.py file
    2. Include only directories matching patterns in the `include` parameter
    3. Remove directories matching patterns in the `exclude` parameter
    
    The patterns can use glob-style wildcards where * matches any number of characters.
    For example, "maggie.*" will match "maggie.core", "maggie.utils", etc.
    
    Examples
    --------
    Basic usage to find all packages:
    
    >>> find_packages()
    ['maggie', 'maggie.core', 'maggie.utils', ...]
    
    Find only specific subpackages:
    
    >>> find_packages(include=["maggie.core.*"])
    ['maggie.core', 'maggie.core.event', 'maggie.core.state', ...]
    
    Exclude test packages:
    
    >>> find_packages(exclude=["*.tests", "*.tests.*"])
    ['maggie', 'maggie.core', 'maggie.utils', ...]
    """
    ...