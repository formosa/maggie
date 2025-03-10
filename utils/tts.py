"""
Maggie AI Assistant - TTS Utility
================================
Text-to-Speech module using Piper TTS for local, high-quality speech synthesis.
Optimized for the AMD Ryzen 9 5900X and Windows 11 Pro.
"""

import io
import os
import time
import threading
import wave
import numpy as np
import soundfile as sf
from loguru import logger

class PiperTTS:
    """
    Text-to-Speech implementation using Piper TTS.
    """
    
    def __init__(self, config):
        """
        Initialize the TTS module with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary for TTS
        """
        self.config = config
        self.voice_model = config.get("voice_model", "en_US-kathleen-medium")
        self.model_path = config.get("model_path", "models/tts")
        self.sample_rate = config.get("sample_rate", 22050)
        self.piper_instance = None
        self.lock = threading.Lock()
        
        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)
        
        # Lazy initialization - will load when first needed
        
    def _init_piper(self):
        """Initialize the Piper TTS model."""
        if self.piper_instance is not None:
            return True
            
        try:
            # Using GPU acceleration through ONNX runtime
            from piper import PiperVoice
            
            voice_dir = os.path.join(self.model_path, self.voice_model)
            onnx_path = os.path.join(voice_dir, f"{self.voice_model}.onnx")
            config_path = os.path.join(voice_dir, f"{self.voice_model}.json")
            
            # Check if model files exist
            if not os.path.exists(onnx_path) or not os.path.exists(config_path):
                logger.error(f"TTS model files not found at {voice_dir}")
                return False
                
            self.piper_instance = PiperVoice.load(onnx_path, config_path)
            logger.info(f"Initialized Piper TTS with voice {self.voice_model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}")
            return False
            
    def speak(self, text):
        """
        Convert text to speech and play it.
        
        Parameters
        ----------
        text : str
            Text to be spoken
            
        Returns
        -------
        bool
            True if successful
        """
        if not text:
            return False
            
        with self.lock:
            try:
                # Initialize if needed
                if not self._init_piper():
                    return False
                    
                # Generate audio data
                audio_data = self._synthesize(text)
                if audio_data is None:
                    return False
                    
                # Play audio
                self._play_audio(audio_data)
                return True
                
            except Exception as e:
                logger.error(f"Error in TTS: {e}")
                return False
                
    def _synthesize(self, text):
        """
        Synthesize speech from text.
        
        Parameters
        ----------
        text : str
            Text to synthesize
            
        Returns
        -------
        np.ndarray or None
            Audio data as numpy array or None if error
        """
        try:
            # Use piper for synthesis
            audio_data = self.piper_instance.synthesize_stream(text)
            return np.array(audio_data)
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return None
            
    def _play_audio(self, audio_data):
        """
        Play audio data.
        
        Parameters
        ----------
        audio_data : np.ndarray
            Audio data to play
        """
        try:
            import pyaudio
            
            # Convert float32 to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Set up PyAudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True
            )
            
            # Play audio
            chunk_size = 1024
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i + chunk_size].tobytes()
                stream.write(chunk)
                
            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
