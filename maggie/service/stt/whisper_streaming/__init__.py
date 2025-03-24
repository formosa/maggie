#!/usr/bin/env python3

"""
Whisper Streaming Package
=========================

A package for real-time audio transcription using OpenAI's Whisper model with support
for various backends, streaming processing, and networked operation.

This package provides tools for streaming audio transcription with various backends,
including real-time processing, voice activity detection, and server-client communication.

Features
--------
* Multiple ASR backends (faster-whisper, whisper_timestamped, MLX Whisper, OpenAI API)
* Voice Activity Detection using Silero VAD
* Real-time audio streaming and processing
* Networked client-server communication
* Support for multiple languages
* Support for transcription and translation

Components
---------
* ASR Base Classes:
  - ASRBase: Abstract base class for all ASR implementations
  - WhisperTimestampedASR: Implementation using whisper_timestamped
  - FasterWhisperASR: Implementation using faster-whisper (recommended)
  - MLXWhisper: Implementation optimized for Apple Silicon
  - OpenaiApiASR: Implementation using OpenAI's Whisper API

* Processing:
  - OnlineASRProcessor: Handles real-time audio processing and transcription
  - VACOnlineASRProcessor: Adds Voice Activity Control to OnlineASRProcessor
  - HypothesisBuffer: Manages transcription hypotheses

* Voice Activity Detection:
  - VADIterator: Base iterator for voice activity detection
  - FixedVADIterator: Extended VAD iterator supporting variable-length inputs

* Networking:
  - Connection: Wrapper for socket connections
  - ServerProcessor: Processes audio chunks in server context

* Utilities:
  - load_audio: Load an audio file
  - load_audio_chunk: Load a specific segment of an audio file
  - create_tokenizer: Create a language-specific sentence tokenizer
  - asr_factory: Create ASR and processor instances
  - add_shared_args: Add common command-line arguments
  - set_logging: Configure logging
"""

# Package version
__version__ = "0.1.0"

# Import from line_packet.py
from .line_packet import PACKET_SIZE, send_one_line, receive_one_line, receive_lines

# Import from silero_vad_iterator.py
from .silero_vad_iterator import VADIterator, FixedVADIterator

# Import from whisper_online.py
from .whisper_online import (
    # ASR Classes
    ASRBase, WhisperTimestampedASR, FasterWhisperASR, MLXWhisper, OpenaiApiASR,
    
    # Processing Classes
    HypothesisBuffer, OnlineASRProcessor, VACOnlineASRProcessor,
    
    # Utility Functions
    load_audio, load_audio_chunk, create_tokenizer, asr_factory, add_shared_args, set_logging,
    
    # Constants
    WHISPER_LANG_CODES
)

# Import from whisper_online_server.py
from .whisper_online_server import Connection, ServerProcessor

# Define public exports
__all__ = [
    # Package metadata
    "__version__",
    
    # line_packet
    "PACKET_SIZE", "send_one_line", "receive_one_line", "receive_lines",
    
    # silero_vad_iterator
    "VADIterator", "FixedVADIterator",
    
    # whisper_online - ASR Classes
    "ASRBase", "WhisperTimestampedASR", "FasterWhisperASR", "MLXWhisper", "OpenaiApiASR",
    
    # whisper_online - Processing Classes
    "HypothesisBuffer", "OnlineASRProcessor", "VACOnlineASRProcessor",
    
    # whisper_online - Utility Functions
    "load_audio", "load_audio_chunk", "create_tokenizer", "asr_factory", 
    "add_shared_args", "set_logging",
    
    # whisper_online - Constants
    "WHISPER_LANG_CODES",
    
    # whisper_online_server
    "Connection", "ServerProcessor",
]

# Add detailed docstrings to the imported objects
ASRBase.__doc__ = """
Base class for ASR (Automatic Speech Recognition) implementations.

Parameters
----------
lan : str
    Language code for transcription.
modelsize : str, optional
    Size of the model (e.g., 'tiny', 'base', 'small', 'medium', 'large').
cache_dir : str, optional
    Directory for caching models.
model_dir : str, optional
    Directory containing a pre-downloaded model.
logfile : file, optional
    File for logging, default is sys.stderr.

Methods
-------
load_model(modelsize, cache_dir, model_dir)
    Load the ASR model.
    
    Parameters
    ----------
    modelsize : str
        Size of the model to load.
    cache_dir : str or None
        Directory for caching models.
    model_dir : str or None
        Directory containing a pre-downloaded model.
        
    Returns
    -------
    model
        The loaded model object.
        
transcribe(audio, init_prompt="")
    Transcribe audio with optional initial prompt.
    
    Parameters
    ----------
    audio : numpy.ndarray
        Audio data as float32 array.
    init_prompt : str, optional
        Initial prompt to guide transcription.
        
    Returns
    -------
    object
        Transcription result object.
        
use_vad()
    Enable Voice Activity Detection.
"""

WhisperTimestampedASR.__doc__ = """
ASR implementation using whisper_timestamped library.

This class provides timestamped transcription using the whisper_timestamped backend.
Initially tested with this backend, but slower than faster-whisper.

Parameters
----------
lan : str
    Language code for transcription.
modelsize : str, optional
    Size of the model (e.g., 'tiny', 'base', 'small', 'medium', 'large').
cache_dir : str, optional
    Directory for caching models.
model_dir : str, optional
    Directory containing a pre-downloaded model.
logfile : file, optional
    File for logging, default is sys.stderr.

Methods
-------
load_model(modelsize, cache_dir, model_dir)
    Load the whisper_timestamped model.
    
transcribe(audio, init_prompt="")
    Transcribe audio with optional initial prompt.
    
    Parameters
    ----------
    audio : numpy.ndarray
        Audio data as float32 array.
    init_prompt : str, optional
        Initial prompt to guide transcription.
        
    Returns
    -------
    dict
        Transcription result dictionary.
        
ts_words(r)
    Convert transcription result to timestamped words.
    
    Parameters
    ----------
    r : dict
        Transcription result from transcribe method.
        
    Returns
    -------
    list
        List of tuples (start_time, end_time, text).
        
segments_end_ts(res)
    Get end timestamps of segments.
    
    Parameters
    ----------
    res : dict
        Transcription result from transcribe method.
        
    Returns
    -------
    list
        List of segment end timestamps.
        
use_vad()
    Enable Voice Activity Detection.
    
set_translate_task()
    Set the task to translation instead of transcription.
"""

FasterWhisperASR.__doc__ = """
ASR implementation using faster-whisper library.

This class provides transcription using the faster-whisper backend,
which offers significantly improved performance (approximately 4x faster).
For GPU usage, requires specific CUDNN version.

Parameters
----------
lan : str
    Language code for transcription.
modelsize : str, optional
    Size of the model (e.g., 'tiny', 'base', 'small', 'medium', 'large').
cache_dir : str, optional
    Directory for caching models.
model_dir : str, optional
    Directory containing a pre-downloaded model.
logfile : file, optional
    File for logging, default is sys.stderr.

Methods
-------
load_model(modelsize, cache_dir, model_dir)
    Load the faster-whisper model.
    
transcribe(audio, init_prompt="")
    Transcribe audio with optional initial prompt.
    
    Parameters
    ----------
    audio : numpy.ndarray
        Audio data as float32 array.
    init_prompt : str, optional
        Initial prompt to guide transcription.
        
    Returns
    -------
    list
        List of segment objects.
        
ts_words(segments)
    Convert transcription segments to timestamped words.
    
    Parameters
    ----------
    segments : list
        Segments from transcribe method.
        
    Returns
    -------
    list
        List of tuples (start_time, end_time, text).
        
segments_end_ts(res)
    Get end timestamps of segments.
    
    Parameters
    ----------
    res : list
        Segments from transcribe method.
        
    Returns
    -------
    list
        List of segment end timestamps.
        
use_vad()
    Enable Voice Activity Detection.
    
set_translate_task()
    Set the task to translation instead of transcription.
"""

MLXWhisper.__doc__ = """
ASR implementation using MLX Whisper library for Apple Silicon.

This class provides optimized transcription for Apple Silicon processors.
Significantly faster than faster-whisper (without CUDA) on Apple M1.

Parameters
----------
lan : str
    Language code for transcription.
modelsize : str, optional
    Size of the model (e.g., 'tiny', 'base', 'small', 'medium', 'large').
cache_dir : str, optional
    Directory for caching models (not used by MLX Whisper).
model_dir : str, optional
    Directory containing a pre-downloaded model.
logfile : file, optional
    File for logging, default is sys.stderr.

Methods
-------
load_model(modelsize, cache_dir, model_dir)
    Load the MLX Whisper model.
    
    Parameters
    ----------
    modelsize : str
        Size of the model to load.
    cache_dir : str or None
        Directory for caching models.
    model_dir : str or None
        Directory containing a pre-downloaded model.
        
    Returns
    -------
    function
        The transcribe function from MLX Whisper.
        
translate_model_name(model_name)
    Translate model name to MLX-compatible format.
    
    Parameters
    ----------
    model_name : str
        Name of the model to translate.
        
    Returns
    -------
    str
        MLX-compatible model path.
        
transcribe(audio, init_prompt="")
    Transcribe audio with optional initial prompt.
    
    Parameters
    ----------
    audio : numpy.ndarray
        Audio data as float32 array.
    init_prompt : str, optional
        Initial prompt to guide transcription.
        
    Returns
    -------
    list
        List of segment dictionaries.
        
ts_words(segments)
    Extract timestamped words from segments.
    
    Parameters
    ----------
    segments : list
        Segments from transcribe method.
        
    Returns
    -------
    list
        List of tuples (start_time, end_time, text).
        
segments_end_ts(res)
    Get end timestamps of segments.
    
    Parameters
    ----------
    res : list
        Segments from transcribe method.
        
    Returns
    -------
    list
        List of segment end timestamps.
        
use_vad()
    Enable Voice Activity Detection.
    
set_translate_task()
    Set the task to translation instead of transcription.
"""

OpenaiApiASR.__doc__ = """
ASR implementation using OpenAI's Whisper API.

Parameters
----------
lan : str, optional
    ISO-639-1 language code for transcription.
temperature : float, optional
    Temperature for sampling. Default is 0.
logfile : file, optional
    File for logging, default is sys.stderr.

Methods
-------
load_model()
    Initialize the OpenAI client.
    
transcribe(audio_data, prompt=None, *args, **kwargs)
    Transcribe audio using OpenAI API.
    
    Parameters
    ----------
    audio_data : numpy.ndarray
        Audio data as float32 array.
    prompt : str, optional
        Prompt to guide transcription.
    *args, **kwargs
        Additional arguments for OpenAI API.
        
    Returns
    -------
    object
        OpenAI API response object.
        
ts_words(segments)
    Convert API response to timestamped words.
    
    Parameters
    ----------
    segments : object
        API response from transcribe method.
        
    Returns
    -------
    list
        List of tuples (start_time, end_time, text).
        
segments_end_ts(res)
    Get end timestamps of words.
    
    Parameters
    ----------
    res : object
        API response from transcribe method.
        
    Returns
    -------
    list
        List of word end timestamps.
        
use_vad()
    Enable Voice Activity Detection filtering.
    
set_translate_task()
    Set the task to translation instead of transcription.
"""

HypothesisBuffer.__doc__ = """
Buffer for managing transcription hypotheses.

This class maintains and processes interim transcription outputs.

Parameters
----------
logfile : file, optional
    File for logging, default is sys.stderr.

Methods
-------
insert(new, offset)
    Insert new transcription hypotheses with time offset.
    
    Parameters
    ----------
    new : list
        List of tuples (start_time, end_time, text).
    offset : float
        Time offset to apply to timestamps.
        
flush()
    Flush and return confirmed transcription chunks.
    
    Returns
    -------
    list
        List of confirmed chunks as tuples (start_time, end_time, text).
        
pop_commited(time)
    Remove committed chunks up to the specified time.
    
    Parameters
    ----------
    time : float
        Time threshold for removal.
        
complete()
    Return the complete buffer content.
    
    Returns
    -------
    list
        List of buffer content as tuples (start_time, end_time, text).
"""

OnlineASRProcessor.__doc__ = """
Processor for online (real-time) audio transcription.

This class manages the audio buffer and transcription process for streaming audio.

Parameters
----------
asr : ASRBase
    ASR implementation to use for transcription.
tokenizer : object, optional
    Sentence tokenizer for the target language.
buffer_trimming : tuple, optional
    Tuple of (strategy, seconds) for buffer trimming.
    Strategy can be 'sentence' or 'segment'. Default is ("segment", 15).
logfile : file, optional
    File for logging, default is sys.stderr.

Attributes
----------
SAMPLING_RATE : int
    Audio sampling rate, fixed at 16000 Hz.

Methods
-------
init(offset=None)
    Initialize or reset the processor.
    
    Parameters
    ----------
    offset : float, optional
        Initial time offset.
        
insert_audio_chunk(audio)
    Add an audio chunk to the buffer.
    
    Parameters
    ----------
    audio : numpy.ndarray
        Audio data as float32 array.
        
prompt()
    Generate prompt from committed text.
    
    Returns
    -------
    tuple
        (prompt, context) where prompt is for ASR and context is for debugging.
        
process_iter()
    Process the current audio buffer and return transcription.
    
    Returns
    -------
    tuple
        (start_time, end_time, text) or (None, None, "").
        
chunk_at(time)
    Trim the buffer at the specified time.
    
    Parameters
    ----------
    time : float
        Time threshold for trimming.
        
finish()
    Finalize processing and return any remaining transcription.
    
    Returns
    -------
    tuple
        (start_time, end_time, text) or (None, None, "").
"""

VACOnlineASRProcessor.__doc__ = """
Online ASR processor with Voice Activity Control.

This class wraps OnlineASRProcessor with VAC (Voice Activity Controller),
detecting speech segments and processing them in real-time.

Parameters
----------
online_chunk_size : float
    Size of online chunks in seconds.
*a : positional arguments
    Arguments to pass to OnlineASRProcessor.
**kw : keyword arguments
    Keyword arguments to pass to OnlineASRProcessor.

Methods
-------
init()
    Initialize the processor.
    
insert_audio_chunk(audio)
    Add an audio chunk and process it with VAD.
    
    Parameters
    ----------
    audio : numpy.ndarray
        Audio data as float32 array.
        
process_iter()
    Process the current audio buffer and return transcription.
    
    Returns
    -------
    tuple
        (start_time, end_time, text) or (None, None, "").
        
finish()
    Finalize processing and return any remaining transcription.
    
    Returns
    -------
    tuple
        (start_time, end_time, text) or (None, None, "").
"""

load_audio.__doc__ = """
Load an audio file into memory.

Parameters
----------
fname : str
    Path to the audio file.

Returns
-------
numpy.ndarray
    Audio data as a float32 array sampled at 16kHz.

Notes
-----
Results are cached using lru_cache for efficiency.
"""

load_audio_chunk.__doc__ = """
Load a specific chunk of an audio file.

Parameters
----------
fname : str
    Path to the audio file.
beg : float
    Beginning time in seconds.
end : float
    Ending time in seconds.

Returns
-------
numpy.ndarray
    Audio chunk data as a float32 array.
"""

create_tokenizer.__doc__ = """
Create a sentence tokenizer for a specific language.

Parameters
----------
lan : str
    Language code (must be in WHISPER_LANG_CODES).

Returns
-------
object
    A tokenizer object with a 'split' method for sentence segmentation.
    
Notes
-----
Depending on the language, uses MosesTokenizer, WtP, or a custom tokenizer.
"""

asr_factory.__doc__ = """
Create ASR and OnlineASR instances based on command-line arguments.

Parameters
----------
args : argparse.Namespace
    Command-line arguments including:
    - backend: ASR backend to use
    - model: Model size/name
    - lan: Language code
    - model_cache_dir: Optional cache directory
    - model_dir: Optional model directory
    - vad: Whether to use VAD
    - vac: Whether to use VAC
    - vac_chunk_size: VAC sample size
    - min_chunk_size: Minimum chunk size
    - buffer_trimming: Buffer trimming strategy
    - buffer_trimming_sec: Buffer trimming threshold
    - task: 'transcribe' or 'translate'
logfile : file, optional
    File for logging, default is sys.stderr.

Returns
-------
tuple
    (asr, online) - ASR backend and online processor instances.
"""

add_shared_args.__doc__ = """
Add shared command-line arguments to an argument parser.

Parameters
----------
parser : argparse.ArgumentParser
    The argument parser to add arguments to.
    
Notes
-----
Adds arguments for model size, language, ASR backend, chunk size,
buffer trimming, VAD/VAC settings, and logging.
"""

set_logging.__doc__ = """
Configure logging for the whisper_streaming package.

Parameters
----------
args : argparse.Namespace
    Command-line arguments containing log_level.
logger : logging.Logger
    Logger instance to configure.
other : str, optional
    Additional string for logger name, default is "_server".
"""

Connection.__doc__ = """
Wrapper for socket connections in the transcription server.

This class handles sending and receiving text lines over sockets.

Parameters
----------
conn : socket.socket
    Socket connection object.

Attributes
----------
PACKET_SIZE : int
    Size of the packet buffer (32000*5*60).

Methods
-------
send(line)
    Send a line of text, avoiding duplicates.
    
    Parameters
    ----------
    line : str
        Line of text to send.
        
receive_lines()
    Receive multiple lines of text.
    
    Returns
    -------
    list or None
        List of received lines or None.
        
non_blocking_receive_audio()
    Receive audio data without blocking.
    
    Returns
    -------
    bytes or None
        Received audio data or None.
"""

ServerProcessor.__doc__ = """
Processor for audio chunks in server context.

This class handles receiving audio, processing it, and sending
transcription results to clients.

Parameters
----------
c : Connection
    Connection wrapper for socket communication.
online_asr_proc : OnlineASRProcessor
    Processor for online audio transcription.
min_chunk : float
    Minimum chunk size in seconds.

Methods
-------
receive_audio_chunk()
    Receive and process audio chunks from the client.
    
    Returns
    -------
    numpy.ndarray or None
        Audio data as float32 array or None.
        
format_output_transcript(o)
    Format transcription output for sending to client.
    
    Parameters
    ----------
    o : tuple
        (start_time, end_time, text) from ASR processor.
        
    Returns
    -------
    str or None
        Formatted output string or None.
        
send_result(o)
    Send transcription result to the client.
    
    Parameters
    ----------
    o : tuple
        (start_time, end_time, text) from ASR processor.
        
process()
    Handle one client connection lifecycle.
"""

WHISPER_LANG_CODES.__doc__ = """
List of language codes supported by Whisper models.

This is a comma-separated string of all supported language codes.
"""
