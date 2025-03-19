"""
Maggie AI Assistant - Wake Word Detection Module
==============================================
Wake word detection functionality using Picovoice Porcupine.

This module provides a thread-safe interface to the Porcupine wake word detection
engine, with specific optimizations for AMD Ryzen 9 5900X CPU to minimize
idle resource usage while maintaining detection accuracy.

Examples
--------
>>> from maggie.utils.stt.wake_word import WakeWordDetector
>>> config = {"sensitivity": 0.5, "porcupine_access_key": "YOUR_KEY_HERE"}
>>> detector = WakeWordDetector(config)
>>> detector.on_detected = lambda: print("Wake word detected!")
>>> detector.start()
>>> # Later, when done:
>>> detector.stop()
"""

# Standard library imports
import os
import threading
import time
import queue
from typing import Dict, Any, Optional, Callable, List, Union

# Third-party imports
import pvporcupine
import pyaudio
import numpy as np
from loguru import logger

__all__ = ['WakeWordDetector']

class WakeWordDetector:
    """
    Wake word detection using Picovoice Porcupine.
    
    Implements continuous background monitoring for the wake word "Maggie"
    with minimal CPU usage and configurable sensitivity. Optimized for
    AMD Ryzen 9 5900X CPU with thread management and resource efficiency.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing wake word settings:
        - sensitivity: float, detection sensitivity (0.0-1.0), default: 0.5
        - keyword_path: str, path to custom keyword model file, default: None (uses "maggie")
        - porcupine_access_key: str, Picovoice access key (required)
        - cpu_threshold: float, maximum CPU usage percentage, default: 5.0
        
    Attributes
    ----------
    on_detected : Optional[Callable]
        Callback function invoked when wake word is detected
    running : bool
        Whether the detector is currently running
    sensitivity : float
        Detection sensitivity (0.0-1.0)
    _porcupine : Optional[pvporcupine.Porcupine]
        Porcupine engine instance
    _pyaudio : Optional[pyaudio.PyAudio]
        PyAudio instance for audio input
    _audio_stream : Optional[pyaudio.Stream]
        PyAudio stream for continuous audio input
    _detection_thread : Optional[threading.Thread]
        Thread for continuous wake word detection
    _stop_event : threading.Event
        Event to signal thread termination
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the wake word detector with configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary for wake word detection settings
        
        Raises
        ------
        ValueError
            If required configuration parameters are missing or invalid
        """
        self.config = config
        config_wake_word = self.config.get("stt", {}).get("wake_word", {})
        self.on_detected = None
        self.running = False
        

        print(f"\n\n-------------------config-------------------\n{config}\n")
        print(f"\n\n-------------------self.config-------------------\n{self.config}\n")
        print(f"\n\n-------------------config_wake_word-------------------\n{config_wake_word}\n")
        print(f"\n\n-------------------self.config.stt-------------------\n{self.config.stt}\n")
        
        
        
        # Parse configuration
        self.sensitivity = config.get("sensitivity", 0.5)
        self.keyword_path = config.get("keyword_path", None)  # Uses "maggie" if None
        self.access_key = self.config.get("stt", {}).get("wake_word",{}).get("porcupine_access_key", None)
        self.cpu_threshold = config.get("cpu_threshold", 5.0)
        
        # Validate configuration
        if not self.access_key:
            logger.error("Missing Porcupine access key in configuration")
            raise ValueError("Porcupine access key is required")
            
        if not 0.0 <= self.sensitivity <= 1.0:
            logger.warning(f"Invalid sensitivity value: {self.sensitivity}, using default: 0.5")
            self.sensitivity = 0.5
        
        # Initialize private attributes
        self._porcupine = None
        self._pyaudio = None
        self._audio_stream = None
        self._detection_thread = None
        self._stop_event = threading.Event()
        self._audio_queue = queue.Queue(maxsize=3)  # Buffer only a few frames
        
        # Thread lock for state changes
        self._lock = threading.Lock()
        
        logger.info(f"Wake word detector initialized with sensitivity: {self.sensitivity}")
        
    def start(self) -> bool:
        """
        Start wake word detection.
        
        Initializes the Porcupine engine, audio input, and starts the
        detection thread for continuous monitoring.
        
        Returns
        -------
        bool
            True if started successfully, False otherwise
        """
        with self._lock:
            if self.running:
                logger.warning("Wake word detector already running")
                return True
                
            try:
                # Initialize Porcupine for wake word detection
                keywords = []
                sensitivities = []
                
                # UPDATED: Fix for keyword availability
                if self.keyword_path and os.path.exists(self.keyword_path):
                    # Use custom keyword model
                    keywords = [self.keyword_path]
                    sensitivities = [self.sensitivity]
                    logger.info(f"Using custom keyword model: {self.keyword_path}")
                else:
                    # Use a default keyword that's available
                    # Fallback to "computer" which is in the default list
                    default_keyword = "computer"  # Change to preferred default
                    keywords = [default_keyword]
                    sensitivities = [self.sensitivity]
                    logger.warning(f"Custom keyword not found, using default: '{default_keyword}'")
                    logger.warning(f"To use 'maggie', create a custom keyword at console.picovoice.ai")
                
                # Create Porcupine instance with proper error handling
                try:
                    self._porcupine = pvporcupine.create(
                        access_key=self.access_key,
                        keywords=keywords,
                        sensitivities=sensitivities
                    )
                except ValueError as e:
                    # Handle case where keywords aren't available
                    logger.error(f"Keyword error: {e}")
                    logger.info("Falling back to 'computer' keyword")
                    self._porcupine = pvporcupine.create(
                        access_key=self.access_key,
                        keywords=["computer"],
                        sensitivities=[self.sensitivity]
                    )
                
                # Initialize PyAudio
                self._pyaudio = pyaudio.PyAudio()
                
                # Configure and open audio stream
                self._audio_stream = self._pyaudio.open(
                    rate=self._porcupine.sample_rate,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=self._porcupine.frame_length,
                    stream_callback=self._audio_callback
                )
                
                # Reset stop event
                self._stop_event.clear()
                
                # Start detection thread
                self._detection_thread = threading.Thread(
                    target=self._detection_loop,
                    name="WakeWordThread",
                    daemon=True
                )
                self._detection_thread.start()
                
                self.running = True
                logger.info("Wake word detection started")
                return True
                
            except pvporcupine.PorcupineError as e:
                logger.error(f"Porcupine error: {e}")
                self._cleanup_resources()
                return False
            except pyaudio.PyAudioError as e:
                logger.error(f"PyAudio error: {e}")
                self._cleanup_resources()
                return False
            except Exception as e:
                logger.error(f"Error starting wake word detection: {e}")
                self._cleanup_resources()
                return False
    
    def stop(self) -> bool:
        """
        Stop wake word detection.
        
        Stops the detection thread and releases resources.
        
        Returns
        -------
        bool
            True if stopped successfully, False otherwise
        """
        with self._lock:
            if not self.running:
                logger.debug("Wake word detector already stopped")
                return True
                
            try:
                # Signal thread to stop
                self._stop_event.set()
                
                # Wait for thread to finish with timeout
                if self._detection_thread and self._detection_thread.is_alive():
                    self._detection_thread.join(timeout=2.0)
                
                # Clean up resources
                self._cleanup_resources()
                
                self.running = False
                logger.info("Wake word detection stopped")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping wake word detection: {e}")
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
            # Add audio data to queue, non-blocking to prevent backlog
            self._audio_queue.put(in_data, block=False)
        except queue.Full:
            # Skip frame if queue is full (better than blocking)
            pass
            
        return (in_data, pyaudio.paContinue)
    
    def _detection_loop(self) -> None:
        """
        Main detection loop running in a separate thread.
        
        Continuously processes audio frames and checks for wake word.
        Optimized for low CPU usage on Ryzen 9 5900X.
        """
        logger.debug("Wake word detection thread started")
        
        import psutil
        process = psutil.Process()
        
        # Set thread to below normal priority to minimize system impact
        if hasattr(process, "nice"):
            try:
                # Lower priority on Linux/macOS
                process.nice(10)
            except:
                pass
        
        # Main detection loop
        while not self._stop_event.is_set():
            try:
                # Get audio data from queue with timeout
                audio_data = self._audio_queue.get(timeout=0.1)
                
                # Convert bytes to int16 array
                pcm = np.frombuffer(audio_data, dtype=np.int16)
                
                # Process audio frame with Porcupine
                keyword_index = self._porcupine.process(pcm)
                
                # Check for wake word detection
                if keyword_index >= 0:
                    logger.info("Wake word detected!")
                    
                    # Call the detection callback if set
                    if self.on_detected:
                        # Call in main thread to avoid potential issues
                        threading.Thread(target=self.on_detected).start()
                
                # CPU usage throttling
                current_cpu = process.cpu_percent(interval=None)
                if current_cpu > self.cpu_threshold:
                    # Briefly sleep to reduce CPU usage
                    time.sleep(0.01)
                    
            except queue.Empty:
                # No audio data available, continue
                continue
            except Exception as e:
                logger.error(f"Error in wake word detection loop: {e}")
                # Brief pause to prevent error spam
                time.sleep(0.1)
        
        logger.debug("Wake word detection thread stopped")
    
    def _cleanup_resources(self) -> None:
        """
        Clean up resources used by the wake word detector.
        
        Releases PyAudio stream, Porcupine instance, and other resources.
        """
        # Stop audio stream
        if self._audio_stream:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except:
                pass
            self._audio_stream = None
        
        # Release PyAudio
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except:
                pass
            self._pyaudio = None
        
        # Delete Porcupine instance
        if self._porcupine:
            try:
                self._porcupine.delete()
            except:
                pass
            self._porcupine = None
            
        # Clear audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except:
                pass