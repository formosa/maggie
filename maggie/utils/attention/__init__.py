"""
Maggie AI Assistant Attention Management Package
===============================================

Provides components for detecting and managing user attention signals,
including wake word detection and user presence awareness, optimized
for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.

This package implements low-resource background monitoring with minimal CPU
usage while in idle state, providing immediate system responsiveness when
user attention is detected.
"""

from maggie.utils.attention.wake_word import WakeWordDetector
__all__ = ['WakeWordDetector']