"""
Maggie AI Assistant - TTS Utility
================================
Text-to-Speech module using Piper TTS for local, high-quality speech synthesis.

This module provides Text-to-Speech functionality for the Maggie AI Assistant
using the Piper TTS library. It includes optimizations for AMD Ryzen 9 5900X
and NVIDIA RTX 3080 hardware, offering low-latency speech synthesis with
high-quality voice output.

Examples
--------
>>> from utils.tts import PiperTTS
>>> config = {"voice_model": "en_US-kathleen-medium", "model_path": "models/tts"}
>>> tts = PiperTTS(config)
>>> tts.speak("Hello, I am Maggie AI Assistant")
>>> # Save to file
>>> tts.save_to_file("This is a test", "test_output.wav")
"""

# Standard library imports
import io
import os
import time
import threading
import wave
import hashlib
from typing import Dict, Any, Optional, Union, Tuple

# Third-party imports
import numpy as np
import soundfile as sf
from loguru import logger

__all__ = ['PiperTTS']

class PiperTTS:
    """
    Text-to-Speech implementation using Piper TTS.
    
    This class provides a simple interface to Piper TTS for high-quality
    speech synthesis. It supports hardware acceleration through ONNX runtime
    when available, particularly optimized for RTX 3080 GPUs.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary for TTS settings
        
    Attributes
    ----------
    voice_model : str
        Name of the voice model to use
    model_path : str
        Path to the directory containing TTS models
    sample_rate : int
        Sample rate for audio output (Hz)
    piper_instance : Optional[Any]
        Loaded Piper TTS model instance
    lock : threading.Lock
        Lock for thread-safe operations
    cache_dir : str
        Directory for TTS audio caching
    use_cache : bool
        Whether to use caching for repeated phrases
    cache_size : int
        Maximum number of cached phrases
    cache : Dict[str, np.ndarray]
        In-memory cache of synthesized audio
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TTS module with configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary for TTS containing:
            - voice_model: Name of voice to use (default: "en_US-kathleen-medium")
            - model_path: Path to TTS models (default: "models/tts")
            - sample_rate: Sample rate in Hz (default: 22050)
            - use_cache: Whether to cache TTS results (default: True)
            - cache_size: Maximum number of cached utterances (default: 100)
        """
        self.config = config
        self.voice_model = config.get("voice_model", "en_US-kathleen-medium")
        self.model_path = config.get("model_path", "models/tts")
        self.sample_rate = config.get("sample_rate", 22050)
        self.piper_instance = None
        self.lock = threading.Lock()
        
        # Audio caching for repeated phrases (optimizes performance)
        self.use_cache = config.get("use_cache", True)
        self.cache_dir = config.get("cache_dir", "cache/tts")
        self.cache_size = config.get("cache_size", 100)
        self.cache = {}
        
        # Create cache directory if needed
        if self.use_cache and not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info(f"Created TTS cache directory: {self.cache_dir}")
            except Exception as e:
                logger.error(f"Failed to create TTS cache directory: {e}")
                self.use_cache = False
        
        # Lazy initialization - will load when first needed
        # This speeds up startup time and reduces memory usage when TTS is not used
        
    def _init_piper(self) -> bool:
        """
        Initialize the Piper TTS model.
        
        Loads the Piper TTS voice model using ONNX runtime with GPU
        acceleration when available, particularly optimized for RTX 3080.
        
        Returns
        -------
        bool
            True if initialization was successful, False otherwise
            
        Notes
        -----
        This method is called automatically when needed. It's thread-safe
        and will only initialize the model once.
        """
        if self.piper_instance is not None:
            return True
            
        try:
            # Using GPU acceleration through ONNX runtime (optimized for RTX 3080)
            from piper import PiperVoice
            
            voice_dir = os.path.join(self.model_path, self.voice_model)
            onnx_path = os.path.join(voice_dir, f"{self.voice_model}.onnx")
            config_path = os.path.join(voice_dir, f"{self.voice_model}.json")
            
            # Check if model files exist
            if not os.path.exists(onnx_path):
                logger.error(f"TTS ONNX model file not found: {onnx_path}")
                return False
                
            if not os.path.exists(config_path):
                logger.error(f"TTS config file not found: {config_path}")
                return False
            
            # Load the model with CUDA acceleration if available
            # This is optimized for RTX 3080 using ONNX runtime
            start_time = time.time()
            self.piper_instance = PiperVoice.load(
                onnx_path, 
                config_path,
                use_cuda=True  # Will use GPU if available, fallback to CPU
            )
            load_time = time.time() - start_time
            logger.info(f"Initialized Piper TTS with voice {self.voice_model} in {load_time:.2f}s")
            
            # Check CUDA availability for logging
            self._log_cuda_status()
                
            return True
            
        except ImportError as import_error:
            logger.error(f"Failed to import Piper TTS module: {import_error}")
            logger.error("Please install piper-tts with: pip install piper-tts==1.2.0")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}")
            return False
    
    def _log_cuda_status(self) -> None:
        """
        Log CUDA availability and GPU information.
        
        Provides diagnostic information about the GPU acceleration
        capabilities for TTS.
        """
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"TTS using GPU acceleration on {gpu_name}")
            else:
                logger.info("TTS using CPU (CUDA not available)")
        except ImportError:
            logger.debug("PyTorch not available for GPU detection")
            
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for the given text.
        
        Parameters
        ----------
        text : str
            Text to generate a cache key for
            
        Returns
        -------
        str
            Cache key (MD5 hash of the text)
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
            
    def _get_cached_audio(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached audio data for the given text if available.
        
        Parameters
        ----------
        text : str
            Text to get cached audio for
            
        Returns
        -------
        Optional[np.ndarray]
            Cached audio data if available, None otherwise
        """
        if not self.use_cache:
            return None
            
        cache_key = self._get_cache_key(text)
        
        # Check memory cache first (faster)
        if cache_key in self.cache:
            logger.debug(f"TTS cache hit (memory): {text[:30]}...")
            return self.cache[cache_key]
            
        # Check file cache
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
        if os.path.exists(cache_path):
            try:
                audio_data = np.load(cache_path)
                # Add to memory cache
                self.cache[cache_key] = audio_data
                logger.debug(f"TTS cache hit (disk): {text[:30]}...")
                return audio_data
            except Exception as e:
                logger.warning(f"Failed to load cached audio: {e}")
                
        return None
        
    def _save_audio_to_cache(self, text: str, audio_data: np.ndarray) -> None:
        """
        Save audio data to cache.
        
        Parameters
        ----------
        text : str
            Text that was synthesized
        audio_data : np.ndarray
            Audio data to cache
        """
        if not self.use_cache:
            return
            
        try:
            cache_key = self._get_cache_key(text)
            
            # Save to memory cache
            self.cache[cache_key] = audio_data
            
            # Limit memory cache size
            if len(self.cache) > self.cache_size:
                # Remove oldest item (first item in dict)
                self.cache.pop(next(iter(self.cache)))
                
            # Save to file cache
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
            np.save(cache_path, audio_data)
            
            logger.debug(f"Saved TTS output to cache: {text[:30]}...")
            
        except Exception as e:
            logger.warning(f"Failed to save audio to cache: {e}")
            
    def speak(self, text: str) -> bool:
        """
        Convert text to speech and play it.
        
        This method synthesizes speech from the given text and plays it
        through the default audio device. It uses caching when possible
        to improve performance for repeated phrases.
        
        Parameters
        ----------
        text : str
            Text to be spoken
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if not text:
            return False
            
        with self.lock:
            try:
                # Check cache first
                cached_audio = self._get_cached_audio(text)
                if cached_audio is not None:
                    logger.debug(f"Using cached audio for: {text[:30]}...")
                    self._play_audio(cached_audio)
                    return True
                
                # Initialize if needed
                if not self._init_piper():
                    return False
                    
                # Generate audio data
                start_time = time.time()
                audio_data = self._synthesize(text)
                synth_time = time.time() - start_time
                
                if audio_data is None:
                    logger.error("Failed to synthesize speech")
                    return False
                
                # Log synthesis time for performance monitoring
                logger.debug(f"Synthesized {len(text)} chars in {synth_time:.2f}s ({len(text)/synth_time:.1f} chars/s)")
                
                # Save to cache
                self._save_audio_to_cache(text, audio_data)
                    
                # Play audio
                self._play_audio(audio_data)
                return True
                
            except Exception as e:
                logger.error(f"Error in TTS: {e}")
                return False
                
    def _synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        Synthesize speech from text.
        
        Parameters
        ----------
        text : str
            Text to synthesize
            
        Returns
        -------
        Optional[np.ndarray]
            Audio data as numpy array or None if error
        """
        try:
            # Use piper for synthesis
            audio_data = self.piper_instance.synthesize_stream(text)
            return np.array(audio_data)
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None
            
    def _play_audio(self, audio_data: np.ndarray) -> None:
        """
        Play audio data.
        
        This method plays the synthesized audio through the default audio device.
        It's optimized for low latency playback on Windows 11.
        
        Parameters
        ----------
        audio_data : np.ndarray
            Audio data to play
        """
        try:
            import pyaudio
            
            # Convert float32 to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Set up PyAudio with optimized buffer size for low latency
            # Smaller chunks reduce latency but increase CPU usage
            # Using 1024 as a balance for Ryzen 9 5900X
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=1024,  # Optimized for low latency
                output_device_index=None  # Use default device
            )
            
            # Play audio in chunks
            self._play_audio_chunks(stream, audio_int16)
                
            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except ImportError as import_error:
            logger.error(f"Failed to import PyAudio: {import_error}")
            logger.error("Please install PyAudio with: pip install PyAudio==0.2.13")
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def _play_audio_chunks(self, stream, audio_int16: np.ndarray, chunk_size: int = 1024) -> None:
        """
        Play audio data in chunks.
        
        Parameters
        ----------
        stream : pyaudio.Stream
            PyAudio stream to play through
        audio_int16 : np.ndarray
            Audio data in int16 format
        chunk_size : int, optional
            Size of each audio chunk, by default 1024
        """
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size].tobytes()
            stream.write(chunk)
    
    def save_to_file(self, text: str, output_path: str) -> bool:
        """
        Convert text to speech and save to a WAV file.
        
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
            
        with self.lock:
            try:
                # Initialize if needed
                if not self._init_piper():
                    return False
                    
                # Check cache first
                cached_audio = self._get_cached_audio(text)
                if cached_audio is not None:
                    audio_data = cached_audio
                else:
                    # Generate audio data
                    audio_data = self._synthesize(text)
                    if audio_data is None:
                        return False
                    
                    # Save to cache
                    self._save_audio_to_cache(text, audio_data)
                
                # Ensure output directory exists
                output_dir = os.path.dirname(os.path.abspath(output_path))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                # Save to WAV file
                self._save_audio_to_wav(audio_data, output_path)
                    
                logger.info(f"Saved TTS audio to {output_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error saving TTS to file: {e}")
                return False
    
    def _save_audio_to_wav(self, audio_data: np.ndarray, output_path: str) -> None:
        """
        Save audio data to a WAV file.
        
        Parameters
        ----------
        audio_data : np.ndarray
            Audio data to save
        output_path : str
            Path to save the WAV file
        """
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            # Convert float32 to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
                
    def cleanup(self) -> None:
        """
        Clean up resources used by the TTS module.
        
        This method frees resources used by the TTS module,
        including unloading the model from memory.
        """
        with self.lock:
            # Free the Piper instance
            self.piper_instance = None
            
            # Clear cache
            self.cache.clear()
            
            logger.info("TTS resources cleaned up")