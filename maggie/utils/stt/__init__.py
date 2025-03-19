"""
Maggie AI Assistant Speech-to-Text Package
=========================================

Provides components for speech recognition and wake word detection, 
optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.

This package implements low-resource background monitoring with minimal CPU
usage while in idle state, providing immediate system responsiveness when
user attention is detected.
"""

from maggie.utils.stt.wake_word import WakeWordDetector
from maggie.utils.stt.processor import STTProcessor

__all__ = ['WakeWordDetector', 'STTProcessor']