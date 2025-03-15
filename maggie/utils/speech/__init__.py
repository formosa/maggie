"""
Maggie AI Assistant Speech Utilities
====================================

Speech processing utilities for the Maggie AI Assistant, providing
text-to-speech and speech recognition capabilities optimized for
AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.

This package includes:
- TTS (Text-to-Speech) implementation using Kokoro for high-quality voice synthesis
- Speech recognition utilities for processing voice input
- Audio processing optimizations for low-latency interactions
"""

from maggie.utils.speech.tts import KokoroTTS
__all__ = ['KokoroTTS']