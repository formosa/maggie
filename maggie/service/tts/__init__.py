"""
Maggie AI Assistant Text-to-Speech Services
===========================================

Text-to-Speech services for the Maggie AI Assistant, providing
voice synthesis capabilities optimized for AMD Ryzen 9 5900X 
and NVIDIA RTX 3080 hardware.

This package includes:
- TTS (Text-to-Speech) implementation using Kokoro for high-quality voice synthesis
- Audio processing optimizations for low-latency interactions
"""

from maggie.service.tts.processor import TTSProcessor

__all__ = ['TTSProcessor']