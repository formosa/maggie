"""
Maggie AI Assistant Services Package
===================================

Core services for the Maggie AI Assistant.

This package provides the core AI service implementations including speech recognition,
voice synthesis, language model processing, and service discovery mechanisms.
"""

# Import service locator first as it's used by other services
from maggie.service.locator import ServiceLocator

# Lazy imports for processors to avoid circular dependencies
def get_llm_processor():
    from maggie.service.llm.processor import LLMProcessor
    return LLMProcessor

def get_tts_processor():
    from maggie.service.tts.processor import TTSProcessor
    return TTSProcessor

def get_stt_processor():
    from maggie.service.stt.processor import STTProcessor
    return STTProcessor

def get_wake_word_detector():
    from maggie.service.stt.wake_word import WakeWordDetector
    return WakeWordDetector

__all__ = [
    'ServiceLocator', 
    'get_llm_processor',
    'get_tts_processor', 
    'get_stt_processor',
    'get_wake_word_detector'
]