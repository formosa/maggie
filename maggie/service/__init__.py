"""
Maggie AI Assistant Services Package
===================================

Core services for the Maggie AI Assistant.

This package provides the core AI service implementations including speech recognition,
voice synthesis, language model processing, and service discovery mechanisms.
"""

from maggie.service.service_locator import ServiceLocator
from maggie.service.llm.processor import LLMProcessor
from maggie.service.tts.processor import TTSProcessor
from maggie.service.stt.processor import STTProcessor
from maggie.service.stt.wake_word import WakeWordDetector

__all__ = ['ServiceLocator', 'LLMProcessor', 'TTSProcessor', 'STTProcessor', 'WakeWordDetector']