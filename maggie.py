"""
Maggie AI Assistant - Main Application
=====================================
Core implementation of the Maggie AI Assistant, implementing a Finite State Machine
architecture with event-driven state transitions and modular utility objects.

Optimized for:
- CPU: AMD Ryzen 9 5900X (12-core)
- GPU: NVIDIA GeForce RTX 3080 (10GB GDDR6X)
- RAM: 32GB DDR4-3200
- OS: Windows 11 Pro
"""

import os
import time
import threading
import queue
import yaml
import numpy as np
import sys  # Added missing import
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum, auto
from dataclasses import dataclass
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor  # Added for thread management

# Core dependencies
import pvporcupine
import pyaudio
import speech_recognition as sr
from faster_whisper import WhisperModel
from ctransformers import AutoModelForCausalLM
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QThread, pyqtSignal
from transitions import Machine
from loguru import logger

# Import utility modules
from utils.tts import PiperTTS
from utils.gui import MainWindow
from utils.config import Config
from utils.recipe_creator import RecipeCreator  # Fixed import path


class MaggieState(Enum):
    """Enumeration of possible states for the Maggie AI Assistant."""
    IDLE = auto()
    STARTUP = auto()
    READY = auto()
    ACTIVE = auto()
    BUSY = auto()
    CLEANUP = auto()
    SHUTDOWN = auto()


@dataclass
class StateTransitionEvent:
    """Data structure for state transition events."""
    from_state: MaggieState
    to_state: MaggieState
    trigger: str
    timestamp: float


class EventBus:
    """
    Central event bus for handling system-wide events and communication between components.
    """
    
    def __init__(self):
        """Initialize the event bus with empty subscribers dictionary."""
        self.subscribers = {}
        self.event_queue = queue.Queue()
        self._running = False
        self._thread = None
        
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe a callback function to a specific event type.
        
        Parameters
        ----------
        event_type : str
            The type of event to subscribe to
        callback : Callable
            Function to be called when event is published
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe a callback function from a specific event type.
        
        Parameters
        ----------
        event_type : str
            The type of event to unsubscribe from
        callback : Callable
            Function to be removed from subscribers
        """
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            
    def publish(self, event_type: str, data: Any = None) -> None:
        """
        Publish an event to all subscribers.
        
        Parameters
        ----------
        event_type : str
            The type of event being published
        data : Any, optional
            Data payload to send with the event
        """
        self.event_queue.put((event_type, data))
        
    def start(self) -> None:
        """Start the event processing thread."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._process_events, daemon=True)
        self._thread.start()
        
    def stop(self) -> None:
        """Stop the event processing thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self.event_queue.put(None)  # Signal to stop
            self._thread.join(timeout=2.0)
            
    def _process_events(self) -> None:
        """Process events from the queue and dispatch to subscribers."""
        while self._running:
            try:
                event = self.event_queue.get(timeout=0.1)
                if event is None:  # Stop signal
                    break
                    
                event_type, data = event
                if event_type in self.subscribers:
                    for callback in self.subscribers[event_type]:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in event handler for {event_type}: {e}")
                            
                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing events: {e}")


class UtilityBase(ABC):
    """
    Abstract base class for all utility modules.
    """
    
    def __init__(self, event_bus: EventBus, config: Dict[str, Any]):
        """
        Initialize the utility module.
        
        Parameters
        ----------
        event_bus : EventBus
            Reference to the central event bus
        config : Dict[str, Any]
            Configuration parameters for the utility
        """
        self.event_bus = event_bus
        self.config = config
        self.running = False
        
    @abstractmethod
    def start(self) -> bool:
        """
        Start the utility module and return success status.
        
        Returns
        -------
        bool
            True if started successfully, False otherwise
        """
        pass
        
    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the utility module and return success status.
        
        Returns
        -------
        bool
            True if stopped successfully, False otherwise
        """
        pass
        
    @abstractmethod
    def process_command(self, command: str) -> bool:
        """
        Process a command directed to this utility.
        
        Parameters
        ----------
        command : str
            The command string to process
            
        Returns
        -------
        bool
            True if command was processed, False if not applicable
        """
        pass


class WakeWordDetector(QThread):
    """
    Responsible for detecting the wake word using Porcupine.
    Optimized for low CPU usage in idle state.
    """
    
    detected = pyqtSignal()
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the wake word detector with configuration parameters.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration including sensitivity and keyword paths
        """
        super().__init__()
        self.config = config
        self.porcupine = None
        self.pa = None
        self.audio_stream = None
        self.running = False
        self.sensitivity = config.get("sensitivity", 0.5)
        self.keyword_path = config.get("keyword_path", None)
        self.access_key = config.get("porcupine_access_key", "")
        
    def run(self):
        """Main execution loop for wake word detection."""
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[self.keyword_path] if self.keyword_path else None,
                keywords=["maggie"] if not self.keyword_path else None,
                sensitivities=[self.sensitivity]
            )
            
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            self.running = True
            logger.info("Wake word detector started")
            
            while self.running:
                pcm = self.audio_stream.read(self.porcupine.frame_length)
                pcm = np.frombuffer(pcm, dtype=np.int16)
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    logger.info("Wake word detected!")
                    self.detected.emit()
                    
                # Add a small sleep to reduce CPU usage further
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources used by wake word detector."""
        if self.audio_stream:
            self.audio_stream.close()
            self.audio_stream = None
            
        if self.pa:
            self.pa.terminate()
            self.pa = None
            
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
            
    def stop(self):
        """Stop the wake word detection thread."""
        self.running = False
        self.cleanup()


class SpeechProcessor:
    """
    Handles speech recognition and text-to-speech conversions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the speech processor with configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration for speech recognition and TTS
        """
        self.config = config
        self.whisper_model = None
        self.tts = PiperTTS(config.get("tts", {}))
        self.sr_recognizer = sr.Recognizer()
        self.sr_mic = None
        
        # Initialize whisper model based on hardware capabilities
        whisper_config = config.get("whisper", {})
        self.init_whisper_model(whisper_config)
        
    def init_whisper_model(self, whisper_config: Dict[str, Any]):
        """
        Initialize the Whisper model for speech recognition.
        
        Parameters
        ----------
        whisper_config : Dict[str, Any]
            Configuration for Whisper model
        """
        model_size = whisper_config.get("model_size", "base")
        compute_type = whisper_config.get("compute_type", "float16")  # Use float16 for RTX 3080
        
        # Load the model with GPU acceleration
        try:
            # Use CUDA for RTX 3080
            self.whisper_model = WhisperModel(
                model_size,
                device="cuda",
                compute_type=compute_type
            )
            logger.info(f"Loaded Whisper model {model_size} on CUDA with {compute_type}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model on GPU: {e}")
            # Fallback to CPU
            self.whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info(f"Loaded Whisper model {model_size} on CPU (fallback)")
            
    def start_listening(self) -> bool:
        """
        Start the microphone for continuous speech recognition.
        
        Returns
        -------
        bool
            True if microphone started successfully
        """
        try:
            self.sr_mic = sr.Microphone()
            return True
        except Exception as e:
            logger.error(f"Failed to start microphone: {e}")
            return False
            
    def stop_listening(self) -> bool:
        """
        Stop the microphone.
        
        Returns
        -------
        bool
            True if stopped successfully
        """
        self.sr_mic = None
        return True
        
    def recognize_speech(self, audio_data=None, timeout=None) -> Tuple[bool, str]:
        """
        Recognize speech from microphone or provided audio data.
        
        Parameters
        ----------
        audio_data : Optional
            Audio data to recognize, if None will listen from microphone
        timeout : Optional[float]
            Maximum time to listen for in seconds
            
        Returns
        -------
        Tuple[bool, str]
            Success status and recognized text
        """
        try:
            if not audio_data and self.sr_mic:
                with self.sr_mic as source:
                    logger.info("Adjusting for ambient noise...")
                    self.sr_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    logger.info("Listening...")
                    audio_data = self.sr_recognizer.listen(source, timeout=timeout)
                    
            if audio_data:
                # Extract audio data as numpy array
                audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
                
                # Use faster-whisper for transcription
                segments, _ = self.whisper_model.transcribe(audio_np, beam_size=5)
                text = " ".join([segment.text for segment in segments])
                
                return True, text.strip()
            else:
                return False, "No audio data available"
                
        except sr.WaitTimeoutError:
            return False, "Listening timed out"
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return False, f"Error: {str(e)}"
            
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
            True if successful
        """
        try:
            return self.tts.speak(text)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False


class LLMProcessor:
    """
    Handles interactions with the LLM for text generation.
    Optimized for the RTX 3080 GPU.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM processor with configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration for LLM including model path and parameters
        """
        self.config = config
        self.model = None
        self.model_path = config.get("model_path", "")
        self.model_type = config.get("model_type", "mistral")
        self.loaded = False
        
    def load_model(self) -> bool:
        """
        Load the LLM model into memory.
        
        Returns
        -------
        bool
            True if loaded successfully
        """
        try:
            # Using ctransformers for optimized inference
            gpu_layers = self.config.get("gpu_layers", 32)  # Use most layers on GPU for RTX 3080
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_type=self.model_type,
                gpu_layers=gpu_layers
            )
            
            self.loaded = True
            logger.info(f"Loaded LLM model from {self.model_path}, GPU layers: {gpu_layers}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            return False
            
    def unload_model(self) -> bool:
        """
        Unload the model from memory.
        
        Returns
        -------
        bool
            True if unloaded or was not loaded
        """
        if self.model:
            self.model = None
            self.loaded = False
            return True
        return True
            
    def generate_text(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Generate text using the LLM model.
        
        Parameters
        ----------
        prompt : str
            Input prompt for the LLM
        max_tokens : int, optional
            Maximum number of tokens to generate
            
        Returns
        -------
        str
            Generated text
        """
        if not self.loaded:
            logger.warning("LLM model not loaded, loading now...")
            if not self.load_model():
                return "Error: Failed to load LLM model"
                
        try:
            # Configure generation parameters optimized for RTX 3080
            response = self.model(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            return response
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            return f"Error in text generation: {str(e)}"


class MaggieAI:
    """
    Main class for the Maggie AI Assistant implementing a Finite State Machine.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Maggie AI Assistant with configuration.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file
        """
        # Load configuration
        self.config = Config(config_path).load()
        self.config_validator = ConfigValidator(self.config)
        
        if not self.config_validator.validate():
            logger.warning("Configuration validation failed, some features may not work correctly")
        
        # Set up thread pool for controlled concurrency
        threading_config = self.config.get("threading", {})
        self.thread_pool = ThreadPoolExecutor(
            max_workers=threading_config.get("max_workers", 8),
            thread_name_prefix="maggie_worker"
        )
        
        # Use pool for tasks
        self.thread_pool.submit(self._initialize_components)

        # Set up logging
        log_config = self.config.get("logging", {})
        log_path = log_config.get("path", "logs")
        os.makedirs(log_path, exist_ok=True)
        logger.configure(
            handlers=[
                {"sink": os.path.join(log_path, "maggie.log"), "rotation": "10 MB", "retention": "1 week"},
                {"sink": os.sys.stdout, "level": log_config.get("console_level", "INFO")}
            ]
        )
        
        # Initialize core components
        self.event_bus = EventBus()
        self.wake_word_detector = WakeWordDetector(self.config.get("wake_word", {}))
        self.speech_processor = SpeechProcessor(self.config.get("speech", {}))
        self.llm_processor = LLMProcessor(self.config.get("llm", {}))
        
        # Initialize FSM
        self.machine = Machine(
            model=self,
            states=[state.name for state in MaggieState],
            initial=MaggieState.IDLE.name,
            transitions=[
                {'trigger': 'wake_up', 'source': MaggieState.IDLE.name, 'dest': MaggieState.STARTUP.name},
                {'trigger': 'initialized', 'source': MaggieState.STARTUP.name, 'dest': MaggieState.READY.name},
                {'trigger': 'process_command', 'source': MaggieState.READY.name, 'dest': MaggieState.ACTIVE.name},
                {'trigger': 'heavy_processing', 'source': MaggieState.ACTIVE.name, 'dest': MaggieState.BUSY.name},
                {'trigger': 'finish_processing', 'source': MaggieState.BUSY.name, 'dest': MaggieState.ACTIVE.name},
                {'trigger': 'task_complete', 'source': MaggieState.ACTIVE.name, 'dest': MaggieState.READY.name},
                {'trigger': 'timeout', 'source': MaggieState.READY.name, 'dest': MaggieState.CLEANUP.name},
                {'trigger': 'shutdown', 'source': '*', 'dest': MaggieState.CLEANUP.name},
                {'trigger': 'cleaned_up', 'source': MaggieState.CLEANUP.name, 'dest': MaggieState.SHUTDOWN.name},
                {'trigger': 'sleep', 'source': MaggieState.CLEANUP.name, 'dest': MaggieState.IDLE.name},
            ],
            send_event=True,
            auto_transitions=False
        )
        
        # Utilities
        self.utilities = {}
        self.command_mapping = {}
        self.initialize_utilities()
        
        # Event handlers
        self.inactivity_timer = None
        self.inactivity_timeout = self.config.get("inactivity_timeout", 300)  # 5 minutes default
        
        # GUI
        self.app = None
        self.window = None
        
        # Connect wake word detection
        self.wake_word_detector.detected.connect(self.on_wake_word_detected)
        
        # Set up event bus listeners
        self.event_bus.subscribe("command_detected", self.handle_command)
        self.event_bus.subscribe("state_changed", self.handle_state_change)
        
    def initialize_utilities(self):
        """Initialize utility modules based on configuration."""
        utility_configs = self.config.get("utilities", {})
        
        if "recipe_creator" in utility_configs:
            recipe_creator = RecipeCreator(self.event_bus, utility_configs["recipe_creator"])
            self.utilities["recipe_creator"] = recipe_creator
            self.command_mapping["new recipe"] = recipe_creator
        
        # Add more utilities as needed
        
    def on_wake_word_detected(self):
        """Handle wake word detection."""
        if self.state == MaggieState.IDLE.name:
            self.wake_up()
        
    def on_enter_STARTUP(self, event):
        """
        Handle actions when entering the STARTUP state.
        
        Parameters
        ----------
        event : EventData
            Event data from the state machine
        """
        logger.info("Entering STARTUP state")
        self.event_bus.publish("state_changed", StateTransitionEvent(
            from_state=MaggieState.IDLE, 
            to_state=MaggieState.STARTUP,
            trigger="wake_up",
            timestamp=time.time()
        ))
        
        # Initialize components that should be ready in READY state
        threading.Thread(target=self._initialize_components).start()
        
    def _initialize_components(self):
        """Initialize components needed for READY state."""
        try:
            # Start TTS
            success = self.speech_processor.speak("Initializing Maggie")
            if not success:
                logger.error("Failed to initialize TTS")
                
            # Start speech recognition
            if not self.speech_processor.start_listening():
                logger.error("Failed to initialize speech recognition")
                
            # Transition to READY state
            self.initialized()
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            # Fall back to IDLE state
            self.cleanup()
            self.sleep()
            
    def on_enter_READY(self, event):
        """
        Handle actions when entering the READY state.
        
        Parameters
        ----------
        event : EventData
            Event data from the state machine
        """
        logger.info("Entering READY state")
        self.event_bus.publish("state_changed", StateTransitionEvent(
            from_state=MaggieState.STARTUP,
            to_state=MaggieState.READY,
            trigger="initialized",
            timestamp=time.time()
        ))
        
        # Speak ready message
        self.speech_processor.speak("Ready for your command")
        
        # Start inactivity timer
        self.start_inactivity_timer()
        
        # Start listening for commands
        threading.Thread(target=self._listen_for_commands).start()
        
    def start_inactivity_timer(self):
        """Start or reset the inactivity timer."""
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
            
        self.inactivity_timer = threading.Timer(self.inactivity_timeout, self.handle_timeout)
        self.inactivity_timer.daemon = True
        self.inactivity_timer.start()
        
    def handle_timeout(self):
        """Handle inactivity timeout."""
        if self.state == MaggieState.READY.name:
            logger.info("Inactivity timeout reached")
            self.timeout()
            
    def _listen_for_commands(self):
        """Listen for commands in the READY state."""
        if self.state == MaggieState.READY.name:
            try:
                success, text = self.speech_processor.recognize_speech(timeout=10.0)
                if success and text:
                    logger.info(f"Recognized: {text}")
                    self.event_bus.publish("command_detected", text)
                    # Reset inactivity timer
                    self.start_inactivity_timer()
                else:
                    # Continue listening
                    threading.Thread(target=self._listen_for_commands).start()
            except Exception as e:
                logger.error(f"Error listening for commands: {e}")
                # Continue listening
                threading.Thread(target=self._listen_for_commands).start()
                
    def handle_command(self, command: str):
        """
        Handle a detected command.
        
        Parameters
        ----------
        command : str
            The detected command
        """
        if self.state != MaggieState.READY.name:
            logger.warning(f"Command received in non-READY state: {self.state}")
            return
            
        command = command.lower().strip()
        
        # Handle core commands
        if command in ["sleep", "go to sleep"]:
            self.speech_processor.speak("Going to sleep")
            self.shutdown()
            return
        elif command in ["shutdown", "turn off"]:
            self.speech_processor.speak("Shutting down")
            self.shutdown()
            return
            
        # Check for utility commands
        for cmd_phrase, utility in self.command_mapping.items():
            if cmd_phrase in command:
                self.process_command(utility=utility)
                return
                
        # Unknown command
        self.speech_processor.speak("I didn't understand that command")
        
    def on_enter_ACTIVE(self, event):
        """
        Handle actions when entering the ACTIVE state.
        
        Parameters
        ----------
        event : EventData
            Event data from the state machine
        """
        logger.info("Entering ACTIVE state")
        self.event_bus.publish("state_changed", StateTransitionEvent(
            from_state=MaggieState.READY,
            to_state=MaggieState.ACTIVE,
            trigger="process_command",
            timestamp=time.time()
        ))
        
        # Process the utility command
        utility = event.kwargs.get('utility')
        if utility:
            threading.Thread(target=self._process_utility, args=(utility,)).start()
        else:
            logger.error("No utility specified for ACTIVE state")
            self.task_complete()
            
    def _process_utility(self, utility: UtilityBase):
        """
        Process a utility in the ACTIVE state.
        
        Parameters
        ----------
        utility : UtilityBase
            The utility to process
        """
        try:
            utility.start()
            # The utility will handle its own completion and state transitions
        except Exception as e:
            logger.error(f"Error processing utility: {e}")
            self.task_complete()
            
    def on_enter_BUSY(self, event):
        """
        Handle actions when entering the BUSY state.
        
        Parameters
        ----------
        event : EventData
            Event data from the state machine
        """
        logger.info("Entering BUSY state")
        self.event_bus.publish("state_changed", StateTransitionEvent(
            from_state=MaggieState.ACTIVE,
            to_state=MaggieState.BUSY,
            trigger="heavy_processing",
            timestamp=time.time()
        ))
        
    def on_enter_CLEANUP(self, event):
        """
        Handle actions when entering the CLEANUP state.
        
        Parameters
        ----------
        event : EventData
            Event data from the state machine
        """
        logger.info("Entering CLEANUP state")
        from_state = MaggieState[self.state] if self.state != MaggieState.CLEANUP.name else None
        trigger = event.event.name if hasattr(event, 'event') and hasattr(event.event, 'name') else "unknown"
        
        self.event_bus.publish("state_changed", StateTransitionEvent(
            from_state=from_state,
            to_state=MaggieState.CLEANUP,
            trigger=trigger,
            timestamp=time.time()
        ))
        
        # Clean up resources
        threading.Thread(target=self._cleanup_resources, args=(event,)).start()
        
    def _cleanup_resources(self, event):
        """
        Clean up resources before transitioning to IDLE or SHUTDOWN.
        
        Parameters
        ----------
        event : EventData
            Event data from the state machine
        """
        try:
            # Cancel inactivity timer
            if self.inactivity_timer:
                self.inactivity_timer.cancel()
                self.inactivity_timer = None
                
            # Stop all utilities
            for utility in self.utilities.values():
                utility.stop()
                
            # Stop speech processor
            self.speech_processor.stop_listening()
            
            # Unload LLM model if loaded
            self.llm_processor.unload_model()
            
            # Determine next state
            trigger = event.event.name if hasattr(event, 'event') and hasattr(event.event, 'name') else ""
            if trigger == "shutdown":
                self.cleaned_up()
            else:
                self.sleep()
                
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
            # Force transition to appropriate state
            if trigger == "shutdown":
                self.cleaned_up()
            else:
                self.sleep()
                
    def on_enter_SHUTDOWN(self, event):
        """
        Handle actions when entering the SHUTDOWN state.
        
        Parameters
        ----------
        event : EventData
            Event data from the state machine
        """
        logger.info("Entering SHUTDOWN state")
        self.event_bus.publish("state_changed", StateTransitionEvent(
            from_state=MaggieState.CLEANUP,
            to_state=MaggieState.SHUTDOWN,
            trigger="cleaned_up",
            timestamp=time.time()
        ))
        
        # Stop the event bus
        self.event_bus.stop()
        
        # Stop the application
        if self.app:
            self.app.quit()
            
    def on_enter_IDLE(self, event):
        """
        Handle actions when entering the IDLE state.
        
        Parameters
        ----------
        event : EventData
            Event data from the state machine
        """
        logger.info("Entering IDLE state")
        self.event_bus.publish("state_changed", StateTransitionEvent(
            from_state=MaggieState.CLEANUP,
            to_state=MaggieState.IDLE,
            trigger="sleep",
            timestamp=time.time()
        ))
        
        # Start wake word detector
        self.wake_word_detector.start()
        
    def handle_state_change(self, event: StateTransitionEvent):
        """
        Handle state change events.
        
        Parameters
        ----------
        event : StateTransitionEvent
            The state transition event
        """
        if self.window:
            self.window.update_state(event.to_state.name)
            self.window.log_event(f"State change: {event.from_state.name} -> {event.to_state.name}")
            
    def start(self):
        """Start the Maggie AI Assistant."""
        try:
            # Start event bus
            self.event_bus.start()
            
            # Create and show GUI
            self.app = QApplication([])
            self.window = MainWindow(self)
            self.window.show()
            
            # Start in IDLE state
            if self.state != MaggieState.IDLE.name:
                self.to_IDLE()
                
            # Start application loop
            logger.info("Maggie AI starting...")
            return self.app.exec()
            
        except Exception as e:
            logger.error(f"Error starting Maggie AI: {e}")
            return 1
        finally:
            # Clean up any remaining resources
            self.cleanup_all()
            
    def cleanup_all(self):
        """Clean up all resources."""
        try:
            # Stop wake word detector
            if hasattr(self, 'wake_word_detector'):
                self.wake_word_detector.stop()
            
            # Stop speech processor
            if hasattr(self, 'speech_processor'):
                self.speech_processor.stop_listening()
            
            # Stop all utilities
            if hasattr(self, 'utilities'):
                for utility in self.utilities.values():
                    utility.stop()
            
            # Stop event bus
            if hasattr(self, 'event_bus'):
                self.event_bus.stop()
            
            # Shut down thread pool
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

    def to_IDLE(self):
        """
        Transition to IDLE state from any state.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        # Cancel inactivity timer
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
            self.inactivity_timer = None
            
        # Clean up resources
        for utility in self.utilities.values():
            utility.stop()
            
        # Stop speech processor
        self.speech_processor.stop_listening()
        
        # Unload LLM model if loaded
        self.llm_processor.unload_model()
        
        # Set state to IDLE
        self.state = MaggieState.IDLE.name
        self.event_bus.publish("state_changed", StateTransitionEvent(
            from_state=None,
            to_state=MaggieState.IDLE,
            trigger="direct_transition",
            timestamp=time.time()
        ))
        
        # Start wake word detector
        self.wake_word_detector.start()


if __name__ == "__main__":
    maggie = MaggieAI()
    exit_code = maggie.start()
    sys.exit(exit_code)
