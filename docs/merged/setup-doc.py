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
- setuptools documentation: https://setuptools.pypa.io/en/latest/setuptools.html
- Python Packaging User Guide: https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/
- PyPI Classifiers: https://pypi.org/classifiers/
- Semantic Versioning: https://semver.org/

Examples
--------
Install in development mode:

    $ pip install -e .

Build distribution package:

    $ python setup.py sdist bdist_wheel

Install from source with development dependencies:

    $ pip install -e .[dev]

Notes
-----
The package requires Python 3.10 and is not compatible with Python 3.11+ due to specific
version requirements of some dependencies and the use of Python 3.10-specific language features.

The hardware-specific optimizations are implemented through configuration profiles in the
resource management module, providing automatic detection and configuration for AMD Ryzen 9
5900X CPUs and NVIDIA RTX 3080 GPUs.
"""

from setuptools import setup, find_packages


def setup(
    name: str = "maggie",
    version: str = "0.1.0",
    description: str = "Maggie AI Assistant optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080",
    author: str = "Maggie Development Team",
    packages: list[str] = None,
    install_requires: list[str] = None,
    extras_require: dict[str, list[str]] = None,
    python_requires: str = ">=3.10, <3.11",
    classifiers: list[str] = None
) -> None:
    """
    Configure and register the Maggie AI Assistant package with setuptools.

    This function encapsulates the setuptools.setup() call with predefined parameters for
    the Maggie AI project. It defines metadata about the package, its dependencies, and
    installation requirements. The setup function implements the Python packaging standard
    defined in PEP 517/PEP 518, using setuptools as the build backend.

    Parameters
    ----------
    name : str, optional
        The name of the package as it will appear in PyPI. Default is "maggie".
    version : str, optional
        The package version following semantic versioning (major.minor.patch).
        Default is "0.1.0".
    description : str, optional
        A brief description of the package's purpose and features.
        Default is "Maggie AI Assistant optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080".
    author : str, optional
        The package author or organization name. Default is "Maggie Development Team".
    packages : list[str], optional
        List of package names to include, or None to use find_packages().
        Default is None, which uses find_packages() with include=["maggie", "maggie.*"].
    install_requires : list[str], optional
        List of package dependencies required for installation.
        Default is a predefined list of specific package versions.
    extras_require : dict[str, list[str]], optional
        Dictionary mapping extra feature names to their additional dependencies.
        Default includes a "dev" extra with testing and linting packages.
    python_requires : str, optional
        Python version specification string. Default is ">=3.10, <3.11".
    classifiers : list[str], optional
        List of PyPI classifiers for categorizing the package.
        Default includes development status, audience, license, and Python version.

    Returns
    -------
    None
        The function calls setuptools.setup() which registers the package with the Python
        packaging system but does not return a value.

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
    if packages is None:
        packages = find_packages(include=["maggie", "maggie.*"])
    
    if install_requires is None:
        install_requires = [
            "PyYAML>=6.0",
            "torch>=2.0.1",
            "torchaudio>=2.0.2",
            "numpy>=1.24.0",
            "loguru>=0.7.0",
            "PySide6>=6.5.0",
            "faster-whisper>=0.9.0",
            "pvporcupine>=2.2.0",
            "ctransformers>=0.2.0",
            "transitions>=0.9.0",
            "python-docx>=0.8.11"
        ]
    
    if extras_require is None:
        extras_require = {
            "dev": [
                "pytest>=7.0.0",
                "flake8>=6.0.0",
                "mypy>=1.0.0"
            ]
        }
    
    if classifiers is None:
        classifiers = [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: End Users/Desktop",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ]
    
    _setup(
        name=name,
        version=version,
        description=description,
        author=author,
        packages=packages,
        install_requires=install_requires,
        extras_require=extras_require,
        python_requires=python_requires,
        classifiers=classifiers
    )


def find_packages(
    where: str = ".",
    exclude: list[str] = None,
    include: list[str] = None
) -> list[str]:
    """
    Discover and return a list of all Python packages in the specified directory.

    This function is a proxy to setuptools.find_packages() which auto-discovers Python
    packages by searching for __init__.py files. It's used to automatically build the
    list of packages to include in the distribution without manually specifying each one.

    Parameters
    ----------
    where : str, optional
        Root directory to search for packages. Default is "." (current directory).
    exclude : list[str], optional
        List of package names or patterns to exclude from the result. Default is None.
    include : list[str], optional
        List of package names or patterns that must be included in the result.
        Default is None. In Maggie setup, defaults to ["maggie", "maggie.*"] in setup().

    Returns
    -------
    list[str]
        List of package names discovered in the specified directory, filtered by the
        include and exclude patterns.

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
    return _find_packages(where=where, exclude=exclude, include=include)


if __name__ == "__main__":
    setup()