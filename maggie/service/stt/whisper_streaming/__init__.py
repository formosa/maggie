"""
Whisper Streaming Package
=========================

A package for real-time audio transcription using OpenAI's Whisper model with support
for various backends, streaming processing, and networked operation.
"""

# Package version
__version__ = "0.1.0"

# Import line_packet in isolation - needed by other modules
from .line_packet import PACKET_SIZE, send_one_line, receive_one_line, receive_lines

# Import VAD components
from .silero_vad_iterator import VADIterator, FixedVADIterator

# Import main whisper components that the application relies on
from .whisper_online import (
    # ASR Classes
    ASRBase, 
    FasterWhisperASR,
    WhisperTimestampedASR,
    MLXWhisper,
    OpenaiApiASR,
    
    # Processing Classes
    HypothesisBuffer,
    OnlineASRProcessor, 
    VACOnlineASRProcessor,
    
    # Utility Functions
    asr_factory,
    create_tokenizer,
    
    # Constants
    WHISPER_LANG_CODES
)

# Import server components
from .whisper_online_server import Connection, ServerProcessor

# Define public exports - minimized to reduce initialization overhead
__all__ = [
    # Package metadata
    "__version__",
    
    # ASR Classes
    "ASRBase", 
    "FasterWhisperASR",
    "WhisperTimestampedASR", 
    "MLXWhisper",
    "OpenaiApiASR",
    
    # Processing Classes
    "OnlineASRProcessor", 
    "VACOnlineASRProcessor",
    
    # VAD Classes
    "VADIterator", 
    "FixedVADIterator",
    
    # Utility Functions
    "asr_factory",
    "create_tokenizer",
    
    # Server Components
    "Connection", 
    "ServerProcessor",
    
    # Constants and utilities
    "PACKET_SIZE", 
    "WHISPER_LANG_CODES"
]