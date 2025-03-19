"""
Maggie AI Assistant - TTS Utility using Kokoro
==============================================
Text-to-Speech module using Kokoro for local, high-quality speech synthesis.

This module provides Text-to-Speech functionality for the Maggie AI Assistant
using the Kokoro library. It includes optimizations for AMD Ryzen 9 5900X
and NVIDIA RTX 3080 hardware, offering low-latency speech synthesis with
high-quality voice output.

Examples
--------
>>> from utils.kokoro_tts import KokoroTTS
>>> config = {"voice_model": "af_heart.pt", "model_path": "models/tts"}
>>> tts = KokoroTTS(config)
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
import concurrent.futures
from typing import Dict, Any, Optional, Union, Tuple

# Third-party imports
import numpy as np
import soundfile as sf
from loguru import logger

__all__ = ['TTSProcessor']

class TTSProcessor:
    """
    Text-to-Speech implementation using Kokoro.
    
    This class provides a simple interface to Kokoro for high-quality
    speech synthesis. It supports optional CUDA acceleration when available,
    particularly optimized for RTX 3080 GPUs.
    
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
    kokoro_instance : Optional[Any]
        Loaded Kokoro TTS model instance
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
            - voice_model: Name of voice to use (default: "af_heart.pt")
            - model_path: Path to TTS models (default: "models/tts")
            - sample_rate: Sample rate in Hz (default: 22050)
            - use_cache: Whether to cache TTS results (default: True)
            - cache_size: Maximum number of cached utterances (default: 100)
        """
        self.config = config
        self.voice_model = config.get("voice_model", "af_heart.pt")
        self.model_path = config.get("model_path", "models/tts")
        self.sample_rate = config.get("sample_rate", 22050)
        self.kokoro_instance = None
        self.lock = threading.Lock()
        
        # Audio caching for repeated phrases (optimized performance)
        self.use_cache = config.get("use_cache", True)
        self.cache_dir = config.get("cache_dir", "cache/tts")
        self.cache_size = config.get("cache_size", 200)  # Increased from 100 to 200
        self.cache = {}
        
        # Enhanced thread pool for audio processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,  # 2 workers for audio tasks
            thread_name_prefix="tts_worker"
        )
        
        # Model-specific options for RTX 3080
        self.gpu_acceleration = config.get("gpu_acceleration", True)
        self.gpu_precision = config.get("gpu_precision", "float16")
        
        # Voice preprocessing for more natural speech
        self.voice_preprocessing = config.get("voice_preprocessing", True)
        
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
        
    def _init_kokoro(self) -> bool:
        """
        Initialize the Kokoro TTS model.
        
        Loads the Kokoro TTS voice model with CUDA acceleration
        for RTX 3080 if available.
        
        Returns
        -------
        bool
            True if initialization was successful, False otherwise
            
        Notes
        -----
        This method is called automatically when needed. It's thread-safe
        and will only initialize the model once.
        """
        if self.kokoro_instance is not None:
            return True
            
        try:
            # Import Kokoro library
            import kokoro
            
            voice_path = os.path.join(self.model_path, self.voice_model)

            # Check if model exists with improved error reporting
            if not os.path.exists(voice_path):
                error_msg = f"TTS voice model not found: {voice_path}"
                logger.error(error_msg)
                
                # Publish detailed error to event bus
                try:
                    from maggie.utils.service_locator import ServiceLocator
                    event_bus = ServiceLocator.get("event_bus")
                    if event_bus:
                        event_bus.publish("error_logged", {
                            "source": "tts",
                            "message": f"Voice model not found: {voice_path}",
                            "path": voice_path
                        })
                except ImportError:
                    pass  # Service locator not available
                    
                # Suggest model download
                logger.info(f"Attempting to download missing voice model...")
                if self._download_voice_model():
                    logger.info(f"Successfully downloaded voice model")
                    # Try again with the downloaded model
                    if os.path.exists(voice_path):
                        return self._initialize_kokoro_engine(voice_path)
                return False
                
            # UPDATED: Fix for Kokoro 0.8.4 API
            # Check available methods in the kokoro module
            logger.debug(f"Available kokoro methods: {dir(kokoro)}")
            
            # Try the correct API - several alternatives based on common TTS patterns
            if hasattr(kokoro, 'TTS'):
                # Method 1: kokoro.TTS(model_path)
                self.kokoro_instance = kokoro.TTS(voice_path)
            elif hasattr(kokoro, 'Model'):
                # Method 2: kokoro.Model.load(model_path)
                self.kokoro_instance = kokoro.Model.load(voice_path)
            else:
                # Method 3: Direct constructor with attributes
                self.kokoro_instance = kokoro.TTSModel(voice_path, use_cuda=self.gpu_acceleration)

            # Initialize the Kokoro TTS engine
            return self._initialize_kokoro_engine(voice_path)
        
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS engine: {e}")
            return False
    
    def _initialize_kokoro_engine(self, voice_path: str) -> bool:
        """
        Initialize the Kokoro TTS engine with the specified voice model.
        
        Parameters
        ----------
        voice_path : str
            Path to the voice model file
            
        Returns
        -------
        bool
            True if initialization was successful, False otherwise
        """
        try:
            import kokoro
            
            # Configure GPU options for Kokoro
            gpu_options = {}
            if self.gpu_acceleration:
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        # Enhanced options specific to RTX 3080
                        if "3080" in gpu_name:
                            gpu_options = {
                                "precision": self.gpu_precision,
                                "cuda_graphs": True,
                                "max_batch_size": 64,
                                "mixed_precision": True,
                                "tensor_cores": True,
                                "stream_buffer_size": 8
                            }
                except ImportError:
                    pass
            
            # Initialize Kokoro with enhanced settings
            start_time = time.time()
            
            # Configure Kokoro to use CUDA if available
            self.kokoro_instance = kokoro.load_tts_model(
                voice_path,
                use_cuda=self.gpu_acceleration,
                sample_rate=self.sample_rate,
                **gpu_options
            )
            
            load_time = time.time() - start_time
            logger.info(f"Initialized Kokoro TTS with voice {self.voice_model} in {load_time:.2f}s")
            
            # Check CUDA availability for logging
            self._log_cuda_status()
            
            # Warmup the model with a short text to initialize the CUDA kernels
            if self.gpu_acceleration:
                self._warm_up_model()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS engine: {e}")
            return False

    def _warm_up_model(self) -> None:
        """
        Warm up the TTS model to initialize CUDA kernels.
        
        Performs a synthesis of a short text to reduce latency
        of the first real synthesis request.
        """
        try:
            logger.debug("Warming up TTS model...")
            _ = self.kokoro_instance.synthesize("Warming up the model.")
            logger.debug("TTS model warm-up complete")
        except Exception as e:
            logger.warning(f"Failed to warm up TTS model: {e}")
    
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
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                
                logger.info(f"TTS using GPU acceleration on {gpu_name} with {vram_total:.2f}GB VRAM")
                logger.debug(f"Current VRAM usage: {vram_allocated:.2f}GB")
                
                # Log specific RTX 3080 optimizations
                if "3080" in gpu_name:
                    logger.info("Applied RTX 3080 specific optimizations for TTS")
            else:
                logger.info("TTS using CPU (CUDA not available)")
        except ImportError:
            logger.debug("PyTorch not available for GPU detection")

    def _download_voice_model(self) -> bool:
        """
        Download the TTS voice model if missing.
        
        Returns
        -------
        bool
            True if download was successful, False otherwise
        """
        try:
            # Ensure model directory exists
            os.makedirs(self.model_path, exist_ok=True)
            
            model_filename = self.voice_model
            target_path = os.path.join(self.model_path, model_filename)
            
            # URL for the voice model
            model_url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt"
            
            logger.info(f"Downloading voice model from {model_url}")
            
            # Use requests to download the file
            import requests
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            # Write the file
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Voice model downloaded to {target_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download voice model: {e}")
            return False

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
        # Add voice model name to the hash to avoid conflicts when changing voices
        cache_text = f"{self.voice_model}:{text}"
        return hashlib.md5(cache_text.encode('utf-8')).hexdigest()
            
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
            
        Notes
        -----
        Implements a two-level cache system (memory and disk) for optimal performance.
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
        
        Notes
        -----
        Saves to both memory and disk cache for optimal performance.
        Implements LRU (Least Recently Used) caching policy.
        """
        if not self.use_cache:
            return
            
        try:
            cache_key = self._get_cache_key(text)
            
            # Save to memory cache
            self.cache[cache_key] = audio_data
            
            # Limit memory cache size (LRU eviction)
            if len(self.cache) > self.cache_size:
                # Remove oldest items (first items in dict)
                to_remove = list(self.cache.keys())[0:len(self.cache) - self.cache_size]
                for key in to_remove:
                    self.cache.pop(key, None)
                    
            # Save to file cache asynchronously
            self.thread_pool.submit(self._save_to_disk_cache, cache_key, audio_data)
            
        except Exception as e:
            logger.warning(f"Failed to save audio to cache: {e}")
            
    def _save_to_disk_cache(self, cache_key: str, audio_data: np.ndarray) -> None:
        """
        Save audio data to disk cache asynchronously.
        
        Parameters
        ----------
        cache_key : str
            Cache key for the audio data
        audio_data : np.ndarray
            Audio data to save
        """
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
            np.save(cache_path, audio_data)
            logger.debug(f"Saved TTS output to disk cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save audio to disk cache: {e}")
            
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
                if not self._init_kokoro():
                    return False
                    
                # Preprocess text for better speech quality if enabled
                if self.voice_preprocessing:
                    text = self._preprocess_text(text)
                
                # Generate audio data
                start_time = time.time()
                audio_data = self._synthesize(text)
                synth_time = time.time() - start_time
                
                if audio_data is None:
                    logger.error("Failed to synthesize speech")
                    return False
                
                # Log synthesis performance metrics
                chars_per_second = len(text) / synth_time if synth_time > 0 else 0
                logger.debug(f"Synthesized {len(text)} chars in {synth_time:.2f}s ({chars_per_second:.1f} chars/s)")
                
                # Save to cache
                self._save_audio_to_cache(text, audio_data)
                    
                # Play audio
                self._play_audio(audio_data)
                return True
                
            except Exception as e:
                logger.error(f"Error in TTS: {e}")
                return False
                
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better speech synthesis.
        
        Parameters
        ----------
        text : str
            Original text
            
        Returns
        -------
        str
            Preprocessed text
            
        Notes
        -----
        Applies various text transformations to improve synthesis quality:
        - Expands common abbreviations
        - Normalizes punctuation
        - Handles numbers and special characters
        """
        # Common abbreviations expansion
        abbreviations = {
            "Dr.": "Doctor",
            "Mr.": "Mister",
            "Mrs.": "Misses",
            "Ms.": "Miss",
            "Prof.": "Professor",
            "e.g.": "for example",
            "i.e.": "that is",
            "vs.": "versus"
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
            
        # Add slight pauses with commas for better rhythm
        text = text.replace(" - ", ", ")
        
        # Ensure sentence endings have proper spacing
        for punct in ['.', '!', '?']:
            text = text.replace(f"{punct}", f"{punct} ")
            text = text.replace(f"{punct}  ", f"{punct} ")
            
        return text
                
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
            
        Notes
        -----
        Optimized for RTX 3080 with GPU acceleration and enhanced
        memory management.
        """
        try:
            # Free unnecessary CUDA memory before synthesis
            if self.gpu_acceleration:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # Clear CUDA cache before synthesis
                except ImportError:
                    pass
            
            # Use Kokoro for synthesis
            audio_data = self.kokoro_instance.synthesize(text)
            return np.array(audio_data)
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None
            
    def _play_audio(self, audio_data: np.ndarray) -> None:
        """
        Play audio data.
        
        This method plays the synthesized audio through the default audio device.
        It's optimized for low latency playback on Windows 11 with Ryzen 9 5900X.
        
        Parameters
        ----------
        audio_data : np.ndarray
            Audio data to play
        """
        try:
            import pyaudio
            
            # Convert float32 to int16 with proper scaling and clipping
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            
            # Set up PyAudio with optimized buffer size for low latency
            # Smaller chunks reduce latency but increase CPU usage
            # 512 is optimal for Ryzen 9 5900X with its high single-thread performance
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=512,  # Reduced from 1024 for lower latency on Ryzen 9 5900X
                output_device_index=None  # Use default device
            )
            
            # Play audio in chunks with optimized playback
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
    
    def _play_audio_chunks(self, stream, audio_int16: np.ndarray, chunk_size: int = 512) -> None:
        """
        Play audio data in chunks.
        
        Parameters
        ----------
        stream : pyaudio.Stream
            PyAudio stream to play through
        audio_int16 : np.ndarray
            Audio data in int16 format
        chunk_size : int, optional
            Size of each audio chunk, by default 512
            
        Notes
        -----
        Uses smaller chunk size (512 vs 1024) for lower latency on Ryzen 9 5900X.
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
                if not self._init_kokoro():
                    return False
                    
                # Check cache first
                cached_audio = self._get_cached_audio(text)
                if cached_audio is not None:
                    audio_data = cached_audio
                else:
                    # Preprocess text for better speech quality if enabled
                    if self.voice_preprocessing:
                        text = self._preprocess_text(text)
                        
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
                
                # Save to WAV file with enhanced quality
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
            
        Notes
        -----
        Uses 24-bit depth for higher quality output when saving to file.
        """
        try:
            # Use soundfile for better quality control
            sf.write(
                output_path, 
                audio_data, 
                self.sample_rate,
                subtype='PCM_24'  # Use 24-bit for better quality in saved files
            )
        except Exception as e:
            logger.error(f"Error saving audio with soundfile: {e}")
            # Fallback to wave module
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
        including unloading the model from memory and clearing GPU resources.
        """
        with self.lock:
            # Free the Kokoro instance
            self.kokoro_instance = None
            
            # Clear cache
            self.cache.clear()
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=False)
            
            # Free GPU resources
            if self.gpu_acceleration:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            
            logger.info("TTS resources cleaned up")