#!/usr/bin/env python3
"""
Maggie AI Assistant - Setup Script
=================================

Installation and packaging configuration for Maggie AI Assistant.
This setup script configures the package metadata and dependencies
for installation via pip or setuptools.

Examples
--------
Install in development mode:
    $ pip install -e .

Build distribution package:
    $ python setup.py sdist bdist_wheel
"""

from setuptools import setup, find_packages

setup(
    name="maggie",
    version="0.1.0",
    description="Maggie AI Assistant optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080",
    author="Maggie Development Team",
    packages=find_packages(include=["maggie", "maggie.*"]),
    install_requires=[
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
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0"
        ]
    },
    python_requires=">=3.10, <3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)