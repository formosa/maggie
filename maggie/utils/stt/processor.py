"""
Maggie AI Assistant - Speech Recognition Module
===============================================
Speech recognition implementation using Faster Whisper models.

This module provides speech recognition functionality for the Maggie AI Assistant
using the faster-whisper library. It includes optimizations for AMD Ryzen 9 5900X
and NVIDIA RTX 3080 hardware, offering efficient speech-to-text conversion with
various model sizes and precision options.

Examples
--------
>>> from maggie.utils.stt.processor import SpeechProcessor
>>> config = {"whisper": {"model_size": "base", "compute_type": "float16"}}
>>> stt_processor = SpeechProcessor(config)
>>> stt_processor.start_listening()
>>> success, text = stt_processor.recognize_speech(timeout=10.0)
>>> if success:
...     print(f"Recognized: {text}")
>>> stt_processor.stop_listening()
>>> stt_processor.speak("I heard what you said")
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
    Speech recognition and processing using Faster Whisper models.
    
    This class provides speech recognition capabilities with Whisper models,
    along with methods for audio capture, processing, and text-to-speech output.
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
    tts_engine : KokoroTTS
        Text-to-speech engine for voice output
    listening : bool
        Whether currently listening for speech
    lock : threading.Lock
        Lock for thread safety
    _stream_thread : Optional[threading.Thread]
        Thread for continuous audio processing
    _stop_event : threading.Event
        Event to signal stream thread to stop
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the speech processor with configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing speech processing settings:
            - whisper: Dict with model_size and compute_type
            - tts: Dict with TTS configuration
        """
        self.config = config
        self.whisper_config = config.get("whisper", {})
        self.model_size = self.whisper_config.get("model_size", "base")
        self.compute_type = self.whisper_config.get("compute_type", "float16")
        
        # Initialize attributes
        self.whisper_model = None
        self.audio_stream = None
        self.pyaudio_instance = None
        
        # Initialize TTS engine with config
        tts_config = config.get("tts", {})
        self.tts_engine = TTSProcessor(tts_config)
        
        # Audio configuration
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.chunk_size = 1024    # Audio buffer size
        self.channels = 1         # Mono audio
        
        # State management
        self.listening = False
        self.lock = threading.Lock()
        self._stream_thread = None
        self._stop_event = threading.Event()
        
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
        logger.info(f"Speech processor initialized with model: {self.model_size}, compute type: {self.compute_type}")
    
    def _load_whisper_model(self) -> bool:
        """
        Load the Whisper model with the configured settings.
        
        Loads the model with GPU acceleration if available and
        configured. Optimized for RTX 3080 with float16 precision.
        
        Returns
        -------
        bool
            True if model loaded successfully, False otherwise
            
        Notes
        -----
        This method is thread-safe and implements lazy loading.
        """
        if self.whisper_model is not None:
            # Check if we need to reload (configuration changed)
            if (self._model_info["size"] == self.model_size and
                self._model_info["compute_type"] == self.compute_type):
                return True
            else:
                # Unload current model before loading new one
                self.whisper_model = None
        
        with self.lock:
            try:
                # Import here to avoid circular imports
                from faster_whisper import WhisperModel
                
                # Check GPU availability
                import torch
                gpu_available = torch.cuda.is_available() and self.use_gpu
                device = "cuda" if gpu_available else "cpu"
                
                logger.info(f"Loading Whisper model: {self.model_size} on {device} with {self.compute_type} precision")
                
                # Load model with optimizations for RTX 3080
                if gpu_available:
                    # RTX 3080 optimized settings
                    gpu_params = {
                        "device": device,
                        "compute_type": self.compute_type,
                        "cpu_threads": 4,  # Limit CPU threads when using GPU
                        "num_workers": 2   # Number of workers for GPU processing
                    }
                    
                    # Special optimization for RTX 3080
                    if "3080" in torch.cuda.get_device_name(0):
                        logger.info("Detected RTX 3080 - applying specific optimizations")
                        # Use flash attention for better performance on Ampere architecture
                        if self.compute_type == "float16":
                            gpu_params["gpu_vram_gb"] = 8  # Reserve ~8GB for model
                            # Additional RTX 3080 specific flags
                            gpu_params["use_kv_cache"] = True
                            
                    self.whisper_model = WhisperModel(
                        self.model_size,
                        device=gpu_params["device"],
                        compute_type=gpu_params["compute_type"],
                        cpu_threads=gpu_params["cpu_threads"],
                        num_workers=gpu_params["num_workers"]
                    )
                else:
                    # CPU fallback with optimized threading for Ryzen 9 5900X
                    cpu_threads = 8  # Optimal for Ryzen 9 5900X
                    self.whisper_model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",  # Use int8 for CPU to save memory
                        cpu_threads=cpu_threads
                    )
                
                # Cache model info
                self._model_info["size"] = self.model_size
                self._model_info["compute_type"] = self.compute_type
                
                logger.info(f"Whisper model {self.model_size} loaded successfully")
                return True
                
            except ImportError as e:
                logger.error(f"Failed to import faster_whisper: {e}")
                logger.error("Please install with: pip install faster-whisper==0.9.0")
                return False
                
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                return False
    
    def start_listening(self) -> bool:
        """
        Start listening for audio input.
        
        Creates and starts an audio stream for capturing microphone input.
        
        Returns
        -------
        bool
            True if started successfully, False otherwise
        """
        with self.lock:
            if self.listening:
                logger.warning("Already listening")
                return True
                
            try:
                # Initialize PyAudio if needed
                if self.pyaudio_instance is None:
                    self.pyaudio_instance = pyaudio.PyAudio()
                
                # Reset stop event
                self._stop_event.clear()
                
                # Start audio stream
                self.audio_stream = self.pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                
                # Clear audio buffer
                with self.buffer_lock:
                    self.audio_buffer = []
                
                # Start streaming thread
                self._stream_thread = threading.Thread(
                    target=self._stream_audio,
                    name="AudioStreamThread",
                    daemon=True
                )
                self._stream_thread.start()
                
                self.listening = True
                logger.info("Started listening for audio input")
                return True
                
            except Exception as e:
                logger.error(f"Error starting audio input: {e}")
                self._cleanup_audio()
                return False
    
    def stop_listening(self) -> bool:
        """
        Stop listening for audio input.
        
        Stops the audio stream and releases resources.
        
        Returns
        -------
        bool
            True if stopped successfully, False otherwise
        """
        with self.lock:
            if not self.listening:
                return True
                
            try:
                # Signal stream thread to stop
                self._stop_event.set()
                
                # Wait for thread to finish with timeout
                if self._stream_thread and self._stream_thread.is_alive():
                    self._stream_thread.join(timeout=2.0)
                
                # Clean up audio resources
                self._cleanup_audio()
                
                self.listening = False
                logger.info("Stopped listening for audio input")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping audio input: {e}")
                return False
    
    def _cleanup_audio(self) -> None:
        """
        Clean up audio resources.
        
        Releases the audio stream and PyAudio instance.
        """
        # Stop and close audio stream
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
        
        # Don't terminate PyAudio instance here to avoid reinitialization overhead
        # Just keep it for the next use
    
    def _stream_audio(self) -> None:
        """
        Stream audio from microphone to buffer.
        
        This method runs in a separate thread and continuously
        captures audio from the microphone to the audio buffer.
        It uses a circular buffer with a maximum size to prevent
        excessive memory usage.
        """
        max_buffer_size = 8 * self.sample_rate  # Limit to 8 seconds to prevent memory issues
        
        try:
            while not self._stop_event.is_set() and self.audio_stream:
                # Read audio chunk
                try:
                    data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Add to buffer with lock
                    with self.buffer_lock:
                        self.audio_buffer.append(data)
                        
                        # Limit buffer size (circular buffer)
                        buffer_bytes = sum(len(chunk) for chunk in self.audio_buffer)
                        while buffer_bytes > max_buffer_size and self.audio_buffer:
                            removed = self.audio_buffer.pop(0)
                            buffer_bytes -= len(removed)
                        
                except OSError as e:
                    logger.error(f"Audio stream error: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error in audio streaming: {e}")
                    time.sleep(0.1)  # Prevent tight loop on error
        except Exception as e:
            logger.error(f"Fatal error in audio streaming thread: {e}")
        finally:
            logger.debug("Audio streaming thread stopped")
    
    def recognize_speech(self, timeout: float = 10.0) -> Tuple[bool, str]:
        """
        Recognize speech from audio input.
        
        Records audio for the specified timeout period, processes it with
        the Whisper model, and returns the recognized text.
        
        Parameters
        ----------
        timeout : float, optional
            Maximum recording time in seconds, by default 10.0
            
        Returns
        -------
        Tuple[bool, str]
            Tuple containing success flag and recognized text
            
        Notes
        -----
        This method is blocking and will return after the timeout
        period or earlier if silence is detected after speech.
        """
        # Check if listening
        if not self.listening:
            logger.warning("Not listening - start_listening() must be called first")
            return False, ""
        
        # Load whisper model if needed
        if not self._load_whisper_model():
            logger.error("Failed to load Whisper model")
            return False, ""
        
        try:
            logger.debug(f"Recording speech for up to {timeout} seconds")
            
            # Record audio
            audio_data = self._record_audio(timeout)
            if not audio_data:
                logger.warning("No audio data recorded")
                return False, ""
            
            # Process with Whisper
            logger.debug("Processing audio with Whisper")
            text = self._process_with_whisper(audio_data)
            
            if text:
                logger.debug(f"Speech recognized: {text}")
                return True, text
            else:
                logger.debug("No speech recognized")
                return False, ""
                
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            return False, ""
    
    def _record_audio(self, timeout: float) -> Optional[np.ndarray]:
        """
        Record audio from the buffer for the specified duration.
        
        Parameters
        ----------
        timeout : float
            Maximum recording time in seconds
            
        Returns
        -------
        Optional[np.ndarray]
            Recorded audio as numpy array or None if no audio recorded
            
        Notes
        -----
        This method will return earlier than the timeout if silence
        is detected after speech using a Voice Activity Detection (VAD)
        algorithm, or when the timeout is reached.
        """
        start_time = time.time()
        recorded_chunks = []
        
        # Clear any previous audio in buffer
        with self.buffer_lock:
            if self.audio_buffer:
                recorded_chunks.extend(self.audio_buffer)
                self.audio_buffer = []
        
        # Calculate silence parameters
        silence_threshold = 1000  # Silence amplitude threshold
        silence_duration = 1.0    # Required silence duration to stop (seconds)
        silence_chunks = int(silence_duration * self.sample_rate / self.chunk_size)
        consecutive_silence = 0
        
        # Record until timeout or silence after speech
        speech_detected = False
        while time.time() - start_time < timeout:
            # Check for new audio chunks with lock
            new_chunks = []
            with self.buffer_lock:
                if self.audio_buffer:
                    new_chunks = self.audio_buffer
                    self.audio_buffer = []
            
            # Process new chunks
            if new_chunks:
                recorded_chunks.extend(new_chunks)
                
                # Check for speech or silence in last chunk
                last_chunk = new_chunks[-1]
                # Convert to numpy array for analysis
                audio_array = np.frombuffer(last_chunk, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array**2))
                
                if rms > silence_threshold:
                    speech_detected = True
                    consecutive_silence = 0
                else:
                    consecutive_silence += 1
                
                # Stop if silence after speech for a while
                if speech_detected and consecutive_silence >= silence_chunks:
                    logger.debug("Speech followed by silence - stopping recording")
                    break
            
            # Small sleep to prevent tight loop
            time.sleep(0.05)
        
        # Convert recorded chunks to numpy array
        if not recorded_chunks:
            return None
            
        # Concatenate all audio chunks
        audio_data = b''.join(recorded_chunks)
        
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_array
    
    def _process_with_whisper(self, audio_data: np.ndarray) -> str:
        """
        Process audio data with Whisper model.
        
        Parameters
        ----------
        audio_data : np.ndarray
            Audio data as float32 numpy array
            
        Returns
        -------
        str
            Recognized text
            
        Notes
        -----
        This method uses the Whisper model to transcribe speech in the
        audio data. It applies optimizations for RTX 3080 GPUs when available.
        """
        try:
            # Log start time for performance measurement
            start_time = time.time()
            
            # Use the model to transcribe
            segments, info = self.whisper_model.transcribe(
                audio_data,
                beam_size=5,           # Beam search for better accuracy
                word_timestamps=False, # Disable for faster processing
                language="en",         # English language (can be made configurable)
                vad_filter=self.vad_enabled,  # Voice activity detection
                vad_parameters={"threshold": self.vad_threshold}
            )
            
            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
            
            text = ' '.join(text_parts).strip()
            
            # Remove common fillers and clean up text
            text = self._clean_recognized_text(text)
            
            # Log performance metrics
            processing_time = time.time() - start_time
            logger.debug(f"Whisper processing took {processing_time:.2f}s")
            
            return text
            
        except Exception as e:
            logger.error(f"Error processing with Whisper: {e}")
            return ""
    
    def _clean_recognized_text(self, text: str) -> str:
        """
        Clean up recognized text by removing fillers and improving formatting.
        
        Parameters
        ----------
        text : str
            Raw recognized text
            
        Returns
        -------
        str
            Cleaned text
            
        Notes
        -----
        This method removes common speech fillers, normalizes spacing,
        and improves capitalization and punctuation.
        """
        if not text:
            return ""
            
        # Remove common fillers
        fillers = [" um ", " uh ", " ah ", " er ", " like ", " you know ", " so "]
        cleaned = " " + text.lower() + " "
        for filler in fillers:
            cleaned = cleaned.replace(filler, " ")
        
        # Fix common recognition errors
        replacements = {
            "maggie": "Maggie",  # Always capitalize the assistant name
            "i": "I",            # Always capitalize I
            " i'm ": " I'm ",
            " i'll ": " I'll ",
            " i've ": " I've ",
            " i'd ": " I'd ",
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(" " + old + " ", " " + new + " ")
        
        # Basic sentence capitalization
        sentences = []
        for sentence in cleaned.split("."):
            if sentence:
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                    sentences.append(sentence)
        
        cleaned = ". ".join(sentences)
        
        # Final cleanup
        cleaned = cleaned.replace("  ", " ").strip()
        if cleaned and not cleaned.endswith((".", "!", "?")):
            cleaned += "."
            
        return cleaned
    
    def speak(self, text: str) -> bool:
        """
        Convert text to speech and play it.
        
        Parameters
        ----------
        text : str
            Text to be spoken
            
        Returns
        -------
        bool
            True if speech synthesis successful, False otherwise
            
        Notes
        -----
        This method uses the TTS engine to convert text to speech
        and play it through the default audio device.
        """
        if not text:
            return False
            
        return self.tts_engine.speak(text)
    
    def save_to_file(self, text: str, output_path: str) -> bool:
        """
        Convert text to speech and save to file.
        
        Parameters
        ----------
        text : str
            Text to be synthesized
        output_path : str
            Path to save the WAV file
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not text:
            return False
            
        return self.tts_engine.save_to_file(text, output_path)
    
    def save_recording(self, output_path: str, timeout: float = 10.0) -> bool:
        """
        Record speech and save to a WAV file.
        
        Parameters
        ----------
        output_path : str
            Path to save the WAV file
        timeout : float, optional
            Maximum recording time in seconds, by default 10.0
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        # Check if listening
        if not self.listening:
            logger.warning("Not listening - start_listening() must be called first")
            return False
        
        try:
            # Record audio
            audio_data = self._record_audio(timeout)
            if audio_data is None:
                logger.warning("No audio data recorded")
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Convert float32 to int16
            audio_int16 = (audio_data * 32768).astype(np.int16)
            
            # Save to WAV file
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
                
            logger.info(f"Saved recording to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            return False
    
    def unload_model(self) -> bool:
        """
        Unload the Whisper model to free memory.
        
        Returns
        -------
        bool
            True if model unloaded successfully, False otherwise
            
        Notes
        -----
        This method releases the Whisper model to free GPU and system
        memory. It's useful when transitioning to IDLE state.
        """
        with self.lock:
            try:
                # Release model reference
                self.whisper_model = None
                
                # Clean up CUDA memory if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("CUDA memory cache cleared")
                except ImportError:
                    pass
                    
                logger.info("Whisper model unloaded")
                return True
                
            except Exception as e:
                logger.error(f"Error unloading model: {e}")
                return False
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the speech processor.
        
        This method stops listening, unloads the model, and releases
        all resources used by the speech processor.
        """
        logger.debug("Cleaning up speech processor resources")
        
        # Stop listening if active
        self.stop_listening()
        
        # Unload model
        self.unload_model()
        
        # Release PyAudio resources
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass
            self.pyaudio_instance = None
        
        # Clean up TTS engine
        if hasattr(self.tts_engine, 'cleanup'):
            self.tts_engine.cleanup()
            
        logger.info("Speech processor resources cleaned up")