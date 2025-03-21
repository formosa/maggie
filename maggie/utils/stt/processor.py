"""
Maggie AI Assistant - Enhanced Speech Recognition Module
===============================================
Speech recognition implementation with real-time transcription using whisper_streaming.

This module provides enhanced speech recognition functionality for Maggie AI Assistant
with real-time streaming transcription capabilities. It integrates with the whisper_streaming
library to provide immediate transcription feedback while speaking, with both intermediate
and final transcription results.

Examples
--------
>>> from maggie.utils.stt.processor import STTProcessor
>>> config = {"whisper_streaming": {"enabled": True, "model_name": "base"}}
>>> stt_processor = STTProcessor(config)
>>> stt_processor.start_listening()
>>> stt_processor.start_streaming(on_intermediate=lambda text: print(f"Intermediate: {text}"), 
...                             on_final=lambda text: print(f"Final: {text}"))
>>> # Later:
>>> stt_processor.stop_streaming()
>>> stt_processor.stop_listening()
"""

# Standard library imports
import os
import io
import tempfile
import threading
import time
import wave
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

# Third-party imports
import numpy as np
import pyaudio
from loguru import logger

# Local imports
from maggie.utils.tts.processor import TTSProcessor

class STTProcessor:
    """
    Enhanced speech recognition and processing with real-time transcription.
    
    This class provides speech recognition capabilities with Whisper models,
    now enhanced with real-time streaming transcription using whisper_streaming.
    It supports different model sizes and compute types, with optimizations for
    RTX 3080 GPUs when available.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing speech processing settings
        
    Attributes
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    whisper_config : Dict[str, Any]
        Whisper-specific configuration
    model_size : str
        Size of the Whisper model to use (tiny, base, small, medium)
    compute_type : str
        Computation type (int8, float16, float32)
    whisper_model : Optional[Any]
        Loaded Whisper model instance
    audio_stream : Optional[Any]
        PyAudio stream for audio capture
    pyaudio_instance : Optional[Any]
        PyAudio instance
    tts_processor : Optional[Any]
        Reference to the TTS processor for speech output
    listening : bool
        Whether currently listening for speech
    streaming_active : bool
        Whether real-time streaming transcription is active
    streaming_paused : bool
        Whether streaming transcription is temporarily paused
    on_intermediate_result : Optional[Callable[[str], None]]
        Callback for intermediate transcription results
    on_final_result : Optional[Callable[[str], None]]
        Callback for final transcription results
    streaming_client : Optional[Any]
        Whisper streaming client for real-time transcription
    streaming_server : Optional[Any]
        Whisper streaming server for processing audio
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the enhanced speech processor with configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing speech processing settings
        """
        # Existing initialization code...
        self.config = config
        self.whisper_config = config.get("whisper", {})
        self.model_size = self.whisper_config.get("model_size", "base")
        self.compute_type = self.whisper_config.get("compute_type", "float16")
        
        # Initialize attributes
        self.whisper_model = None
        self.audio_stream = None
        self.pyaudio_instance = None

        # Whisper streaming configuration
        self.streaming_config = config.get("whisper_streaming", {})
        self.use_streaming = self.streaming_config.get("enabled", False)
        self.streaming_server = None
        self.streaming_client = None
        
        # Reference to TTS processor
        self.tts_processor = None
        
        # Audio configuration
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.chunk_size = 1024    # Audio buffer size
        self.channels = 1         # Mono audio
        
        # State management
        self.listening = False
        self.streaming_active = False
        self.streaming_paused = False
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Real-time streaming attributes
        self.on_intermediate_result = None
        self.on_final_result = None
        self._streaming_thread = None
        self._streaming_stop_event = threading.Event()
        
        # Buffer for continuous audio
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Performance optimization
        self.use_gpu = not config.get("cpu_only", False)
        self.vad_enabled = config.get("vad_enabled", True)
        self.vad_threshold = config.get("vad_threshold", 0.5)
        
        # Cached model info for faster reloading
        self._model_info = {
            "size": None,
            "compute_type": None
        }
        
        # Initialize in a lazy manner
        logger.info(f"Enhanced speech processor initialized with model: {self.model_size}, compute type: {self.compute_type}")
        logger.info(f"Streaming mode: {'enabled' if self.use_streaming else 'disabled'}")

    def speak(self, text: str) -> bool:
        """
        Text-to-speech synthesis via TTS processor.
        
        This is a delegation method that forwards speak requests to the TTS processor.
        It retrieves the TTS processor from the ServiceLocator if not already provided.
        
        Parameters
        ----------
        text : str
            Text to be spoken
            
        Returns
        -------
        bool
            True if speech was successfully generated and played, False otherwise
            
        Notes
        -----
        This method facilitates proper separation of concerns by delegating
        text-to-speech functionality to the appropriate component.
        """
        try:
            # If tts_processor not set, try to get it from ServiceLocator
            if self.tts_processor is None:
                from maggie.utils.service_locator import ServiceLocator
                self.tts_processor = ServiceLocator.get("tts_processor")
                
                if self.tts_processor is None:
                    logger.error("TTS processor not found in ServiceLocator")
                    return False
            
            # Delegate the speech task to the TTS processor
            return self.tts_processor.speak(text)
        except Exception as e:
            logger.error(f"Error in STTProcessor.speak(): {e}")
            return False

    def start_listening(self) -> bool:
        """
        Start listening for audio input.
        
        This method initializes audio capture and begins listening for user speech.
        It must be called before streaming or speech recognition can work.
        
        Returns
        -------
        bool
            True if listening started successfully, False otherwise
            
        Notes
        -----
        This method initializes the PyAudio instance and audio stream for
        capturing microphone input. It sets the listening flag to True,
        which allows other components to check if audio capture is active.
        
        The method uses a thread lock to prevent race conditions when
        starting audio capture from multiple threads.
        
        Example
        -------
        >>> processor.start_listening()
        >>> # Now the processor is listening for audio input
        >>> success, text = processor.recognize_speech()
        """
        with self.lock:
            # Check if already listening
            if self.listening:
                logger.debug("Already listening")
                return True
                
            try:
                # Initialize PyAudio if not already initialized
                if self.pyaudio_instance is None:
                    self.pyaudio_instance = pyaudio.PyAudio()
                    
                # Open audio stream with optimal settings
                self.audio_stream = self.pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback
                )
                
                # Reset stop event
                self._stop_event.clear()
                
                # Set listening flag
                self.listening = True
                
                # Clear audio buffer
                with self.buffer_lock:
                    self.audio_buffer = []
                    
                logger.info("Audio listening started")
                return True
                
            except Exception as e:
                logger.error(f"Error starting audio listening: {e}")
                self._cleanup_audio_resources()
                return False
                
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback for audio stream data.
        
        Parameters
        ----------
        in_data : bytes
            Raw audio data
        frame_count : int
            Number of frames
        time_info : dict
            Timing information
        status : int
            Status flag
            
        Returns
        -------
        tuple
            (data, flag) tuple for PyAudio
        """
        if self._stop_event.is_set():
            return (in_data, pyaudio.paComplete)
            
        try:
            # Add audio data to buffer with lock
            with self.buffer_lock:
                self.audio_buffer.append(in_data)
                
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            
        return (in_data, pyaudio.paContinue)
        
    def stop_listening(self) -> bool:
        """
        Stop listening for audio input.
        
        This method stops audio capture and releases audio resources.
        
        Returns
        -------
        bool
            True if listening stopped successfully, False otherwise
            
        Notes
        -----
        This method stops the audio stream, releases PyAudio resources,
        and sets the listening flag to False. It uses a thread lock to
        prevent race conditions when stopping audio capture from multiple
        threads.
        
        Example
        -------
        >>> processor.stop_listening()
        >>> # Audio capture is now stopped
        """
        with self.lock:
            # Check if not listening
            if not self.listening:
                logger.debug("Not listening")
                return True
                
            try:
                # Signal stop
                self._stop_event.set()
                
                # Clean up audio resources
                self._cleanup_audio_resources()
                
                # Set listening flag
                self.listening = False
                
                logger.info("Audio listening stopped")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping audio listening: {e}")
                return False
                
    def _cleanup_audio_resources(self) -> None:
        """
        Clean up audio resources.
        
        This method releases PyAudio and audio stream resources
        to prevent resource leaks.
        
        Returns
        -------
        None
        """
        # Stop and close audio stream
        if self.audio_stream is not None:
            try:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
            
        # Release PyAudio
        if self.pyaudio_instance is not None:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass
            self.pyaudio_instance = None
            
    def recognize_speech(self, timeout: float = 10.0) -> Tuple[bool, str]:
        """
        Recognize speech from audio input.
        
        This method listens for speech and returns the recognized text.
        
        Parameters
        ----------
        timeout : float, optional
            Maximum time to listen for speech in seconds, by default 10.0
            
        Returns
        -------
        Tuple[bool, str]
            Tuple containing success flag and recognized text
            
        Notes
        -----
        This method uses the Whisper model to recognize speech from
        the audio buffer. It returns a tuple with a success flag and
        the recognized text if successful.
        
        The method requires that listening is already started with
        start_listening() before it can recognize speech.
        
        Example
        -------
        >>> processor.start_listening()
        >>> success, text = processor.recognize_speech(timeout=5.0)
        >>> if success:
        ...     print(f"Recognized: {text}")
        ... else:
        ...     print("Recognition failed")
        """
        if not self.listening:
            logger.error("Cannot recognize speech - not listening")
            return False, ""
            
        try:
            # Get start time
            start_time = time.time()
            
            # Wait for audio data with timeout
            audio_data = None
            while time.time() - start_time < timeout:
                # Get audio data from buffer with lock
                with self.buffer_lock:
                    if self.audio_buffer:
                        audio_data = b''.join(self.audio_buffer)
                        self.audio_buffer = []
                        break
                
                # Sleep briefly to prevent tight loop
                time.sleep(0.05)
                
            # Check if we have audio data
            if audio_data is None:
                logger.warning("No audio data received within timeout")
                return False, ""
                
            # Convert to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Initialize whisper model if needed
            if self.whisper_model is None:
                self._load_whisper_model()
                
            # Use Whisper model for transcription
            if self.whisper_model is not None:
                result = self.whisper_model.transcribe(audio_np)
                if isinstance(result, dict) and "text" in result:
                    return True, result["text"].strip()
                elif hasattr(result, "text"):
                    return True, result.text.strip()
                else:
                    logger.warning("Unexpected result format from Whisper model")
                    return False, ""
            else:
                logger.error("Whisper model not initialized")
                return False, ""
                
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            return False, ""
            
    def _load_whisper_model(self) -> bool:
        """
        Load the Whisper model for speech recognition.
        
        Returns
        -------
        bool
            True if model loaded successfully, False otherwise
            
        Notes
        -----
        This method loads the Whisper model based on the configuration
        settings. It supports different backend implementations and
        compute types.
        """
        try:
            from faster_whisper import WhisperModel
            
            # Get model parameters from config
            model_size = self.model_size
            compute_type = self.compute_type
            
            # Determine device based on GPU availability
            device = "cuda" if self.use_gpu else "cpu"
            
            # Load model with appropriate settings
            logger.info(f"Loading Whisper model: {model_size} on {device} with {compute_type}")
            self.whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            
            return True
            
        except ImportError as e:
            logger.error(f"Error importing WhisperModel: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            return False

    def start_streaming(self, on_intermediate: Optional[Callable[[str], None]] = None, 
                        on_final: Optional[Callable[[str], None]] = None) -> bool:
        """
        Start real-time streaming transcription with callback handlers.
        
        This method initiates the real-time transcription process using whisper_streaming,
        allowing for immediate feedback during speech. It requires that listening has
        already been started with start_listening().
        
        Parameters
        ----------
        on_intermediate : Optional[Callable[[str], None]], optional
            Callback function for intermediate transcription results, by default None
        on_final : Optional[Callable[[str], None]], optional
            Callback function for final transcription results, by default None
            
        Returns
        -------
        bool
            True if streaming started successfully, False otherwise
            
        Notes
        -----
        This method requires the listening mode to be active first via start_listening().
        It initializes the whisper_streaming components and sets up a continuous
        processing thread to handle real-time transcription.
        
        Example
        -------
        >>> processor.start_listening()
        >>> def show_interim(text):
        ...     print(f"In progress: {text}")
        >>> def process_final(text):
        ...     print(f"Final result: {text}")
        >>> processor.start_streaming(on_intermediate=show_interim, on_final=process_final)
        """
        with self.lock:
            # Check if already streaming
            if self.streaming_active:
                logger.warning("Already streaming")
                return True
                
            # Check if listening is active
            if not self.listening:
                logger.error("Must start listening before streaming can begin")
                return False
                
            # Store callback functions
            self.on_intermediate_result = on_intermediate
            self.on_final_result = on_final
            
            # Initialize streaming components if not already initialized
            if not self._init_streaming():
                logger.error("Failed to initialize streaming components")
                return False
                
            # Reset stop event
            self._streaming_stop_event.clear()
            
            # Start streaming thread
            self._streaming_thread = threading.Thread(
                target=self._streaming_process_loop,
                name="StreamingTranscriptionThread",
                daemon=True
            )
            self._streaming_thread.start()
            
            self.streaming_active = True
            self.streaming_paused = False
            logger.info("Real-time transcription streaming started")
            return True

    def stop_streaming(self) -> bool:
        """
        Stop real-time streaming transcription.
        
        This method halts the real-time transcription process but does not stop
        the audio listening. To completely stop audio capture, use stop_listening().
        
        Returns
        -------
        bool
            True if streaming stopped successfully, False otherwise
            
        Notes
        -----
        This method signals the streaming thread to stop and waits for it to finish
        with a reasonable timeout. It does not affect the audio capture process,
        which continues to run until stop_listening() is called.
        
        Example
        -------
        >>> processor.stop_streaming()  # Stop real-time transcription
        >>> # Audio capture is still active
        >>> processor.stop_listening()  # Stop audio capture completely
        """
        with self.lock:
            # Check if not streaming
            if not self.streaming_active:
                logger.debug("Not streaming, nothing to stop")
                return True
                
            try:
                # Signal streaming thread to stop
                self._streaming_stop_event.set()
                
                # Wait for thread to finish with timeout
                if self._streaming_thread and self._streaming_thread.is_alive():
                    self._streaming_thread.join(timeout=2.0)
                
                self.streaming_active = False
                self.streaming_paused = False
                logger.info("Streaming transcription stopped")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping streaming: {e}")
                return False

    def pause_streaming(self) -> bool:
        """
        Pause real-time streaming transcription temporarily.
        
        This method temporarily pauses the streaming transcription process without
        stopping it completely. This is useful when the user wants to type manually
        or when the application needs to pause transcription briefly.
        
        Returns
        -------
        bool
            True if streaming paused successfully, False otherwise
            
        Notes
        -----
        When paused, audio is still being captured but not processed for transcription.
        This maintains the audio buffer but prevents updates to the transcription text.
        
        Example
        -------
        >>> processor.pause_streaming()  # User clicked on input field
        >>> # Later
        >>> processor.resume_streaming()  # User stopped typing
        """
        with self.lock:
            # Check if not streaming
            if not self.streaming_active:
                logger.debug("Not streaming, nothing to pause")
                return False
                
            # Already paused
            if self.streaming_paused:
                return True
                
            self.streaming_paused = True
            logger.info("Streaming transcription paused")
            return True

    def resume_streaming(self) -> bool:
        """
        Resume previously paused streaming transcription.
        
        This method resumes the streaming transcription process after it has been
        paused with pause_streaming(). This is useful when the user finishes typing
        manually and wants to continue with voice input.
        
        Returns
        -------
        bool
            True if streaming resumed successfully, False otherwise
            
        Notes
        -----
        When resumed, the transcription process continues from where it left off,
        processing any audio that was captured during the pause.
        
        Example
        -------
        >>> processor.pause_streaming()  # User started typing
        >>> # User finished typing
        >>> processor.resume_streaming()  # Continue with voice input
        """
        with self.lock:
            # Check if not streaming
            if not self.streaming_active:
                logger.debug("Not streaming, nothing to resume")
                return False
                
            # Not paused
            if not self.streaming_paused:
                return True
                
            self.streaming_paused = False
            logger.info("Streaming transcription resumed")
            return True

    def _init_streaming(self) -> bool:
        """
        Initialize whisper_streaming components for real-time speech recognition.
        
        Returns
        -------
        bool
            True if initialization successful, False otherwise
        """
        if self.streaming_server is not None and self.streaming_client is not None:
            return True
            
        try:
            from whisper_streaming.whisper_live import Client, TranscriptionServer
            
            # Initialize the transcription server
            model_name = self.streaming_config.get("model_name", self.model_size)
            language = self.streaming_config.get("language", "en")
            
            # Use compute_type from config if available
            compute_type = self.streaming_config.get("compute_type", self.compute_type)
            
            # Create server with optimized settings for RTX 3080
            server_options = {
                "language": language,
                "use_gpu": self.use_gpu,
                "compute_type": compute_type
            }
            
            # RTX 3080 specific optimizations
            if self.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available() and "3080" in torch.cuda.get_device_name(0):
                        logger.info("Applying RTX 3080 optimizations for whisper_streaming")
                        server_options["gpu_thread_count"] = 2
                        server_options["buffer_size_seconds"] = 30.0
                        server_options["vad_threshold"] = 0.6
                except:
                    pass
            
            # Create server
            self.streaming_server = TranscriptionServer(
                model=model_name,
                **server_options
            )
            
            # Start server in a background thread
            server_thread = threading.Thread(
                target=self.streaming_server.run, 
                daemon=True
            )
            server_thread.start()
            
            # Wait a moment for server to start
            time.sleep(2)
            
            # Create client
            self.streaming_client = Client()
            
            logger.info(f"Whisper streaming initialized with model: {model_name}")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import whisper_streaming: {e}")
            logger.error("Please install with: pip install git+https://github.com/ufal/whisper_streaming.git")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing whisper_streaming: {e}")
            return False

    def _streaming_process_loop(self) -> None:
        """
        Main processing loop for streaming transcription.
        
        This method continuously processes audio input and generates real-time
        transcription results with both intermediate and final outputs.
        
        Notes
        -----
        This method runs in a separate thread and processes audio chunks from
        the buffer, sending them to the whisper_streaming client for transcription.
        It calls the appropriate callback functions for intermediate and final
        transcription results.
        """
        logger.debug("Streaming transcription processing thread started")
        
        last_result = ""
        intermediate_timeout = self.streaming_config.get("result_timeout", 0.5)
        commit_timeout = self.streaming_config.get("commit_timeout", 2.0)
        last_commit_time = time.time()
        
        try:
            while not self._streaming_stop_event.is_set():
                # Skip processing if paused
                if self.streaming_paused:
                    time.sleep(0.1)
                    continue
                
                # Get audio data from buffer with lock
                audio_data = None
                with self.buffer_lock:
                    if self.audio_buffer:
                        audio_data = b''.join(self.audio_buffer)
                        self.audio_buffer = []
                
                if audio_data is not None:
                    # Convert to int16 for streaming
                    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Add to streaming client
                    self.streaming_client.add_frames(audio_int16)
                    
                    # Get current transcription
                    current_result = self.streaming_client.get_transcription(timeout=intermediate_timeout)
                    current_time = time.time()
                    
                    # If we have a new result, send it as intermediate
                    if current_result and current_result != last_result:
                        last_result = current_result
                        
                        # Call intermediate result callback
                        if self.on_intermediate_result:
                            cleaned_text = self._clean_recognized_text(current_result)
                            self.on_intermediate_result(cleaned_text)
                            
                        # Reset commit timer when we get new content
                        last_commit_time = current_time
                    
                    # Check if we should commit the current result as final
                    # This happens when no new content has been received for commit_timeout seconds
                    if current_result and (current_time - last_commit_time) > commit_timeout:
                        # Call final result callback
                        if self.on_final_result:
                            final_text = self._clean_recognized_text(current_result)
                            self.on_final_result(final_text)
                            
                        # Reset last result after committing
                        last_result = ""
                        last_commit_time = current_time
                        
                        # Clear client buffer after committing
                        self.streaming_client.clear_buffer()
                
                # Small sleep to prevent tight loop
                time.sleep(0.05)
                
        except Exception as e:
            logger.error(f"Error in streaming transcription loop: {e}")
        finally:
            logger.debug("Streaming transcription processing thread stopped")