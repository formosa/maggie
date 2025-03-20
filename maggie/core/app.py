"""
Maggie AI Assistant - Core Implementation
=======================================
Core FSM implementation of the Maggie AI Assistant.

This module implements a simplified Finite State Machine (FSM) architecture
with event-driven state transitions and optimized resource management.
Specifically tuned for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.

The architecture consists of:
1. State - Enum defining possible states (IDLE, READY, ACTIVE, CLEANUP, SHUTDOWN)
2. StateTransition - Data class for state transition events with metadata
3. EventBus - Centralized event management with publisher-subscriber pattern
4. MaggieAI - Main class implementing the FSM with optimized threading

The implementation leverages hardware-specific optimizations including:
- Thread pool sizing optimized for Ryzen 9 5900X's 12 cores
- GPU layer allocation for RTX 3080's 10GB VRAM
- Memory management strategies for 32GB system configurations
- Dynamic resource allocation based on system state

Examples
--------
>>> from maggie import MaggieAI
>>> config = {"threading": {"max_workers": 10}, "inactivity_timeout": 300}
>>> maggie = MaggieAI(config)
>>> maggie.initialize_components()
>>> maggie.start()
>>> # Later, to stop
>>> maggie.stop()
"""

# Standard library imports
import argparse
import sys
import os
import threading
import queue
import time
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
from loguru import logger

__all__ = ['State', 'StateTransition', 'EventBus', 'MaggieAI']

class State(Enum):
    """
    Simplified state enumeration for Maggie AI Assistant.
    
    Reduces the previous state machine to five essential states
    for improved clarity and manageability.
    
    Attributes
    ----------
    IDLE : enum
        Waiting for wake word, minimal resource usage. Only wake word detection
        is active, models are unloaded, CPU usage <5%.
    READY : enum
        Listening for commands, resources initialized. Speech recognition is
        active, models are loaded, waiting for user command.
    ACTIVE : enum
        Processing commands and running extensions. Full system engagement
        with maximum resource utilization.
    CLEANUP : enum
        Cleaning up resources. Releasing memory, stopping components, and
        preparing for state transition.
    SHUTDOWN : enum
        Final state before application exit. All resources released, threads
        terminated, application preparing to exit.
    """
    IDLE = auto()      # Waiting for wake word, minimal resource usage
    READY = auto()     # Listening for commands, resources initialized
    ACTIVE = auto()    # Processing commands and running extensions
    CLEANUP = auto()   # Cleaning up resources
    SHUTDOWN = auto()  # Final state before application exit

@dataclass
class StateTransition:
    """
    Data structure for state transition events.
    
    Parameters
    ----------
    from_state : State
        Previous state the system is transitioning from
    to_state : State
        New state the system is transitioning to
    trigger : str
        Event that triggered the transition (e.g., "wake_word_detected", "timeout")
    timestamp : float
        Unix timestamp of the transition for logging and debugging
        
    Examples
    --------
    >>> transition = StateTransition(
    ...     from_state=State.IDLE,
    ...     to_state=State.READY,
    ...     trigger="wake_word_detected",
    ...     timestamp=time.time()
    ... )
    >>> print(f"Transition from {transition.from_state.name} to {transition.to_state.name}")
    Transition from IDLE to READY
    """
    from_state: State
    to_state: State
    trigger: str
    timestamp: float
    
    def __lt__(self, other):
        """
        Compare transitions for priority queue ordering.
        
        Parameters
        ----------
        other : StateTransition
            Another transition to compare with
            
        Returns
        -------
        bool
            True if self has higher priority than other
        """
        # Compare by timestamp for ordering in priority queue
        return self.timestamp < other.timestamp

class EventBus:
    """
    Centralized event management system.
    
    Handles event publication, subscription, and dispatching with
    thread-safety and prioritization.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    subscribers : Dict[str, List[Tuple[int, Callable]]]
        Event subscribers mapped by event type with priority (lower values have higher priority)
    queue : queue.PriorityQueue
        Priority queue for event processing
    running : bool
        Whether the event bus is currently running
    _worker_thread : Optional[threading.Thread]
        Thread for event processing
        
    Examples
    --------
    >>> event_bus = EventBus()
    >>> def handle_wake_word(data):
    ...     print(f"Wake word detected: {data}")
    >>> event_bus.subscribe("wake_word_detected", handle_wake_word)
    >>> event_bus.start()
    >>> event_bus.publish("wake_word_detected", "Maggie")
    Wake word detected: Maggie
    >>> event_bus.stop()
    """
    
    def __init__(self):
        """
        Initialize the event bus with empty subscribers dictionary.
        """
        self.subscribers = {}
        self.queue = queue.PriorityQueue()
        self.running = False
        self._worker_thread = None
        
    def subscribe(self, event_type: str, callback: Callable, priority: int = 0) -> None:
        """
        Subscribe to an event type with priority support.
        
        Parameters
        ----------
        event_type : str
            Event type to subscribe to
        callback : Callable
            Function to call when event is published
        priority : int, optional
            Subscription priority (lower is higher priority), by default 0
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
            
        self.subscribers[event_type].append((priority, callback))
        self.subscribers[event_type].sort(key=lambda x: x[0])  # Sort by priority
        
    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Unsubscribe from an event type.
        
        Parameters
        ----------
        event_type : str
            Event type to unsubscribe from
        callback : Callable
            Function to unsubscribe
            
        Returns
        -------
        bool
            True if unsubscribed successfully, False otherwise
        """
        if event_type not in self.subscribers:
            return False
            
        for i, (_, cb) in enumerate(self.subscribers[event_type]):
            if cb == callback:
                self.subscribers[event_type].pop(i)
                return True
                
        return False
        
    def publish(self, event_type: str, data: Any = None, priority: int = 0) -> None:
        """
        Publish an event with priority support.
        
        Parameters
        ----------
        event_type : str
            Type of event to publish
        data : Any, optional
            Event data payload, by default None
        priority : int, optional
            Event priority (lower is higher priority), by default 0            

        """
        self.queue.put((priority, (event_type, data)))
        
    def start(self) -> bool:
        """
        Start the event processing thread.
        
        Returns
        -------
        bool
            True if started successfully, False if already running
        """
        if self.running:
            return False
            
        self.running = True
        self._worker_thread = threading.Thread(
            target=self._process_events,
            name="EventBusThread",
            daemon=True
        )
        self._worker_thread.start()
        
        logger.info("Event bus started")
        return True
        
    def stop(self) -> bool:
        """
        Stop the event processing thread.
        
        Returns
        -------
        bool
            True if stopped successfully, False if not running
        """
        if not self.running:
            return False
            
        self.running = False
        self.queue.put((0, None))  # Signal to stop
        
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            
        logger.info("Event bus stopped")
        return True
        
    def _process_events(self) -> None:
        """
        Process events from the queue and dispatch to subscribers.
        
        Optimized for efficient event processing with better queue handling
        for Ryzen 9 5900X.
        """
        while self.running:
            try:
                # Use shorter timeout for more responsive event handling
                priority, event = self.queue.get(timeout=0.05)
                
                if event is None:  # Stop signal
                    break
                    
                event_type, data = event
                
                if event_type in self.subscribers:
                    for _, callback in self.subscribers[event_type]:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in event handler for {event_type}: {e}")
                            
                self.queue.task_done()
                
            except queue.Empty:
                # More efficient CPU usage during idle periods
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error processing events: {e}")

class MaggieAI:
    """
    Core implementation of the Maggie AI Assistant.
    
    Implements a simplified Finite State Machine architecture with
    event-driven state transitions and resource management optimized
    for AMD Ryzen 9 5900X and NVIDIA RTX 3080.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
        
    Attributes
    ----------
    state : State
        Current state of the assistant
    event_bus : EventBus
        Central event bus for component communication
    config : Dict[str, Any]
        Configuration dictionary
    extensions : Dict[str, Any]
        Loaded extension modules
    hardware_manager : Optional[Any]
        Hardware management component
    wake_word_detector : Optional[Any]
        Wake word detection component
    stt_processor : Optional[Any]
        Speech processing component
    llm_processor : Optional[Any]
        LLM processing component
    gui : Optional[Any]
        GUI component
    thread_pool : ThreadPoolExecutor
        Worker thread pool for async tasks
    inactivity_timer : Optional[threading.Timer]
        Timer for tracking inactivity
    transition_handlers : Dict[State, Callable]
        State transition handler functions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Maggie AI Core.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        """
        # Set configuration dictionary
        self.config = config

        # Set initial state to IDLE and prepare for state transitions
        self.state = State.IDLE

        # Initialize EventBus for event management
        self.event_bus = EventBus()

        # Initialize extension dictionary for loaded extensions
        self.extensions = {}

        # Inactivity timer for tracking user inactivity
        self.inactivity_timer = None
        # Inactivity timeout in seconds (default: 5 minutes)
        self.inactivity_timeout = self.config.get("inactivity_timeout", 300)

        # Utility References - will be initialized during startup
        self.hardware_manager = None
        self.wake_word_detector = None
        self.stt_processor = None
        self.llm_processor = None
        self.gui = None
        
        # CPU management
        # Worker thread pool for async tasks (default: 10 workers)
        self.thread_pool = ThreadPoolExecutor(
            max_workers= self.config.get("cpu", {}).get("max_threads", 10),
            thread_name_prefix="maggie_thread_"
        )
         
        # Setup state transition handlers
        self.transition_handlers = {
            State.IDLE: self._on_enter_idle,
            State.READY: self._on_enter_ready,
            State.ACTIVE: self._on_enter_active,
            State.CLEANUP: self._on_enter_cleanup,
            State.SHUTDOWN: self._on_enter_shutdown
        }
        
        # Register event handlers
        self._register_event_handlers()
        
        # Create GPU resource management
        self._setup_gpu_resource_management()
        
    def _setup_gpu_resource_management(self) -> None:
        """
        GPU Resource Management: Setup for the RTX 3080.
        
        Configures PyTorch for optimal GPU memory management with the 
        RTX 3080's 10GB VRAM, enabling efficient memory allocation and
        garbage collection.
        """
        try:
            import torch
            
            # Check for CUDA availability
            if torch.cuda.is_available():

                # Set up GPU resource management for RTX 3080
                logger.info(f"GPU Resource Management: Setup for the {torch.cuda.get_device_name(0)}")

                # Enable CUDA caching allocator for better memory efficiency
                torch.cuda.empty_cache()
                logger.info("PyTorch CUDA caching allocator enabled")
                
                # Enable anomaly detection (DEBUG mode only)
                if self.config.get("logging", {}).get("console_level", "INFO") == "DEBUG":
                    torch.autograd.set_detect_anomaly(True)
                    logger.info("PyTorch anomaly detection enabled")
                else:
                    torch.autograd.set_detect_anomaly(False)
                    logger.info("PyTorch anomaly detection disabled")
                    
                # Set optimal memory management for RTX 3080 (10GB VRAM)
                if hasattr(torch.cuda, 'memory_reserved'):

                    # Get total VRAM
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    logger.info(f"Total GPU memory == {total_memory/1024**3:.2f}GB")

                    # Get reserved VRAM
                    memory_reserved = torch.cuda.memory_reserved(0)
                    logger.info(f"memory_reserved == {memory_reserved/1024**3:.2f}GB")

        except ImportError:
            logger.debug("PyTorch not available for GPU resource management")
        except Exception as e:
            logger.error(f"Error setting up GPU resource management: {e}")
        
    def _register_event_handlers(self) -> None:
        """
        Register event handlers for the event bus.
        
        Sets up handlers for wake word detection, commands, inactivity timeout,
        and extension completion events.
        """
        self.event_bus.subscribe("wake_word_detected", self._handle_wake_word)
        self.event_bus.subscribe("command_detected", self._handle_command)
        self.event_bus.subscribe("inactivity_timeout", self._handle_timeout)
        self.event_bus.subscribe("extension_completed", self._handle_extension_completed)
        
    def initialize_components(self) -> bool:
        """
        Initialize all required components.

        The initialization process begins with the application
        utilities; ServiceLocator, logging, HardwareManager, 
        WakeWordDetector, SpeechProcessor, STT processing, 
        and LLMProcessor.

        The process also initializes extensions based on the
        configuration and registers them for global access.
        
        Returns
        -------
        bool
            True if all components initialized successfully
            
        Raises
        ------
        ImportError
            If required modules cannot be imported
        """
        try:
            # Import: Service Locator
            logger.info("Initializing components...")
            logger.info("self.config")
            logger.info(f"{self.config}")
            from maggie.utils.service_locator import ServiceLocator
            # Register event bus for global access
            ServiceLocator.register("event_bus", self.event_bus)
            
            # Import: Hardware Manager
            from maggie.utils.hardware.manager import HardwareManager
            # Create hardware manager and provide configuration
            self.hardware_manager = HardwareManager(self.config)
            # Apply hardware optimizations to configuration
            # self.config = self.hardware_manager.optimize_config(self.config)
            # Register hardware manager for global access
            ServiceLocator.register("hardware_manager", self.hardware_manager)

            
            # Import: WakeWordDetector
            from maggie.utils.stt.wake_word import WakeWordDetector
            # Create wake word detector with configured wake word
            self.wake_word_detector = WakeWordDetector(self.config.get("stt", {}).get("wake_word", {}))
            # Set wake word detection callback
            # to trigger state transition to READY state when wake word is detected
            self.wake_word_detector.on_detected = lambda: self.event_bus.publish("wake_word_detected")
            # Register wake word detector for global access
            ServiceLocator.register("wake_word_detector", self.wake_word_detector)
            
            # Import: STTProcessor
            from maggie.utils.stt.processor import STTProcessor
            # Create stt processor with configuration 
            self.stt_processor = STTProcessor(self.config.get("stt", {}))
            # Register stt processor for global access
            ServiceLocator.register("stt_processor", self.stt_processor)
            
            # Import: TTSProcessor
            from maggie.utils.tts.processor import TTSProcessor
            # Create TTS processor with configuration
            self.tts_processor = TTSProcessor(self.config.get("tts", {}))
            # Register TTS processor for global access
            ServiceLocator.register("tts_processor", self.tts_processor)

            # Import: LLMProcessor
            from maggie.utils.llm.processor import LLMProcessor
            # Create LLM processor with configuration
            self.llm_processor = LLMProcessor(self.config.get("llm", {}))
            # Register LLM processor for global access
            ServiceLocator.register("llm_processor", self.llm_processor)
                       
            # Register self in service locator for GUI access
            ServiceLocator.register("maggie_ai", self)
            
            # Initialize extensions
            self._initialize_extensions()
            
            # Start event bus
            self.event_bus.start()
            
            logger.info("All components initialized successfully")
            return True
            
        except ImportError as import_error:
            logger.error(f"Failed to import required module: {import_error}")
            return False
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False

    def _initialize_extensions(self) -> None:
        """
        Initialize extension modules based on configuration.
        
        Loads and initializes extension modules specified in the configuration
        using the extension registry for dynamic discovery.
        """
        from maggie.extensions.registry import ExtensionRegistry
        
        extensions_config = self.config.get("extensions", {})
        
        # Create extension registry
        registry = ExtensionRegistry()
        
        # Discover available extensions
        available_extensions = registry.discover_extensions()
        logger.info(f"Discovered {len(available_extensions)} extensions: {', '.join(available_extensions.keys())}")
        
        # Initialize enabled extensions from configuration
        for extension_name, extension_config in extensions_config.items():
            # Skip disabled extensions
            if extension_config.get("enabled", True) is False:
                logger.info(f"extension {extension_name} is disabled in configuration")
                continue
            
            # Try to instantiate the extension
            extension = registry.instantiate_extension(extension_name, self.event_bus, extension_config)
            
            if extension is not None:
                self.extensions[extension_name] = extension
                logger.info(f"Initialized extension: {extension_name}")
            else:
                logger.warning(f"Failed to initialize extension: {extension_name}")
        
        # Log number of initialized extensions
        logger.info(f"Initialized {len(self.extensions)} extension modules")
            
    def start(self) -> bool:
        """
        Start the Maggie AI Assistant.
        
        Returns
        -------
        bool
            True if started successfully
        """
        try:
            logger.info("Starting Maggie AI Assistant")
            
            # Initialize components
            if not self.initialize_components():
                logger.error("Failed to initialize components")
                return False
                
            # Transition to IDLE state
            self._transition_to(State.IDLE, "startup")
            
            # Start hardware monitoring
            self.hardware_manager.start_monitoring()
            
            logger.info("Maggie AI Assistant started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Maggie AI: {e}")
            return False
            
    def stop(self) -> bool:
        """
        Stop the Maggie AI Assistant.
        
        Returns
        -------
        bool
            True if stopped successfully
        """
        try:
            logger.info("Stopping Maggie AI Assistant")
            
            # Transition to SHUTDOWN state
            self._transition_to(State.SHUTDOWN, "stop_requested")
            
            logger.info("Maggie AI Assistant stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Maggie AI: {e}")
            return False
            
    def _transition_to(self, new_state: State, trigger: str) -> None:
        """
        Transition to a new state.
        
        Parameters
        ----------
        new_state : State
            State to transition to
        trigger : str
            Event that triggered the transition
        """
        if new_state == self.state:
            return
            
        old_state = self.state
        self.state = new_state
        
        # Log the transition
        logger.info(f"State transition: {old_state.name} -> {new_state.name} (trigger: {trigger})")
        
        # Create transition event
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            trigger=trigger,
            timestamp=time.time()
        )
        
        # Publish state transition event
        self.event_bus.publish("state_changed", transition)
        
        # Call state entry handler
        if new_state in self.transition_handlers:
            self.transition_handlers[new_state](transition)
            
    def _on_enter_idle(self, transition: StateTransition) -> None:
        """
        Handle entering IDLE state.
        
        Parameters
        ----------
        transition : StateTransition
            State transition information
        """
        # Cancel inactivity timer if active
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
            self.inactivity_timer = None
            
        # Stop speech processor
        if self.stt_processor:
            self.stt_processor.stop_listening()
            
        # Unload LLM model to save memory
        if self.llm_processor:
            self.llm_processor.unload_model()
            
        # Start wake word detector
        if self.wake_word_detector:
            self.wake_word_detector.start()
            
        # Free GPU resources when idle
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU memory cache cleared in IDLE state")
        except ImportError:
            pass
            
        logger.info("Entered IDLE state - waiting for wake word")
            
    def _on_enter_ready(self, transition: StateTransition) -> None:
        """
        Handle entering READY state.
        
        Parameters
        ----------
        transition : StateTransition
            State transition information
        """
        # Stop wake word detector
        if self.wake_word_detector:
            self.wake_word_detector.stop()
            
        # Start speech processor
        if self.stt_processor:
            self.stt_processor.start_listening()
            self.stt_processor.speak("Ready for your command")
            
        # Start inactivity timer
        self._start_inactivity_timer()
        
        # Start listening for commands
        self.thread_pool.submit(self._listen_for_commands)
        
        logger.info("Entered READY state - listening for commands")
            
    def _on_enter_active(self, transition: StateTransition) -> None:
        """
        Handle entering ACTIVE state.
        
        Parameters
        ----------
        transition : StateTransition
            State transition information
        """
        # Reset inactivity timer
        self._start_inactivity_timer()
        
        logger.info("Entered ACTIVE state - executing command or extension")
            
    def _on_enter_cleanup(self, transition: StateTransition) -> None:
        """
        Handle entering CLEANUP state.
        
        Parameters
        ----------
        transition : StateTransition
            State transition information
        """
        # Cancel inactivity timer
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
            self.inactivity_timer = None
            
        # Stop all extensions
        self._stop_all_extensions()
            
        # Stop speech processor
        if self.stt_processor:
            self.stt_processor.stop_listening()
            
        # Unload LLM model
        if self.llm_processor:
            self.llm_processor.unload_model()
            
        # Free GPU resources
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU memory cache cleared in CLEANUP state")
        except ImportError:
            pass
            
        # Determine next state based on trigger
        if transition.trigger == "shutdown_requested":
            self._transition_to(State.SHUTDOWN, "cleanup_completed")
        else:
            self._transition_to(State.IDLE, "cleanup_completed")
            
        logger.info("Entered CLEANUP state - releasing resources")
            
    def _stop_all_extensions(self) -> None:
        """
        Stop all running extensions.
        
        Ensures all extensions are properly shut down during cleanup.
        """
        for extension_name, extension in self.extensions.items():
            try:
                if hasattr(extension, 'stop') and callable(extension.stop):
                    extension.stop()
                    logger.debug(f"Stopped extension: {extension_name}")
            except Exception as e:
                logger.error(f"Error stopping extension {extension_name}: {e}")
            
    def _on_enter_shutdown(self, transition: StateTransition) -> None:
        """
        Handle entering SHUTDOWN state.
        
        Parameters
        ----------
        transition : StateTransition
            State transition information
        """
        # Stop hardware monitoring
        if self.hardware_manager:
            self.hardware_manager.stop_monitoring()
            
        # Stop event bus
        self.event_bus.stop()
        
        # Clean up GPU resources
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU memory freed during shutdown")
        except ImportError:
            pass
        
        # Shutdown thread pool with timeout for clean termination
        self.thread_pool.shutdown(wait=True, timeout=5)
        
        logger.info("Entered SHUTDOWN state - application will exit")
            
    def _handle_wake_word(self, _: Any) -> None:
        """
        Handle wake word detection event.
        
        Parameters
        ----------
        _ : Any
            Event data (unused)
        """
        if self.state == State.IDLE:
            logger.info("Wake word detected")
            self._transition_to(State.READY, "wake_word_detected")
            
    def _handle_command(self, command: str) -> None:
        """
        Handle detected command.
        
        Parameters
        ----------
        command : str
            Detected command
        """
        # Only process commands in READY state
        if self.state != State.READY:
            return
            
        command = command.lower().strip()
        logger.info(f"Command detected: {command}")
        
        # Handle system commands
        if command in ["sleep", "go to sleep"]:
            self.stt_processor.speak("Going to sleep")
            self._transition_to(State.CLEANUP, "sleep_command")
            return
            
        if command in ["shutdown", "turn off"]:
            self.stt_processor.speak("Shutting down")
            self._transition_to(State.CLEANUP, "shutdown_requested")
            return
            
        # Check for extension commands
        extension_triggered = self._check_extension_commands(command)
        if extension_triggered:
            return
                
        # Handle unknown command
        self.stt_processor.speak("I didn't understand that command")
        logger.warning(f"Unknown command: {command}")
        
    def _check_extension_commands(self, command: str) -> bool:
        """
        Check if command matches any extension triggers.
        
        Parameters
        ----------
        command : str
            Command to check
            
        Returns
        -------
        bool
            True if a extension was triggered, False otherwise
        """
        for extension_name, extension in self.extensions.items():
            extension_trigger = extension.get_trigger()
            if extension_trigger and extension_trigger in command:
                logger.info(f"Triggered extension: {extension_name}")
                self._transition_to(State.ACTIVE, f"extension_{extension_name}")
                self.thread_pool.submit(self._run_extension, extension_name)
                return True
        return False
        
    def _handle_timeout(self, _: Any) -> None:
        """
        Handle inactivity timeout event.
        
        Parameters
        ----------
        _ : Any
            Event data (unused)
        """
        if self.state == State.READY:
            logger.info("Inactivity timeout reached")
            self.stt_processor.speak("Going to sleep due to inactivity")
            self._transition_to(State.CLEANUP, "inactivity_timeout")
            
    def _handle_extension_completed(self, extension_name: str) -> None:
        """
        Handle extension completion event.
        
        Parameters
        ----------
        extension_name : str
            Name of the completed extension
        """
        if self.state == State.ACTIVE:
            logger.info(f"extension completed: {extension_name}")
            self._transition_to(State.READY, f"extension_{extension_name}_completed")
        
    def _start_inactivity_timer(self) -> None:
        """
        Start or reset the inactivity timer.
        
        Creates a new timer that will trigger an inactivity timeout
        event after the configured timeout period.
        """
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
            
        self.inactivity_timer = threading.Timer(
            self.inactivity_timeout,
            lambda: self.event_bus.publish("inactivity_timeout")
        )
        self.inactivity_timer.daemon = True
        self.inactivity_timer.start()
        
        logger.debug(f"Started inactivity timer: {self.inactivity_timeout}s")
        
    def _listen_for_commands(self) -> None:
        """
        Listen for commands in the READY state.
        
        Uses speech recognition to listen for commands and processes them.
        """
        if self.state != State.READY:
            return
            
        try:
            logger.debug("Listening for commands...")
            success, text = self.stt_processor.recognize_speech(timeout=10.0)
            
            if success and text:
                logger.info(f"Recognized: {text}")
                self.event_bus.publish("command_detected", text)
            else:
                # Continue listening in a new thread to avoid stack overflow
                self.thread_pool.submit(self._listen_for_commands)
                
        except Exception as e:
            logger.error(f"Error listening for commands: {e}")
            # Continue listening in a new thread
            self.thread_pool.submit(self._listen_for_commands)
            
    def _run_extension(self, extension_name: str) -> None:
        """
        Run an extension.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension to run
        """
        if extension_name not in self.extensions:
            logger.error(f"Unknown extension: {extension_name}")
            self._transition_to(State.READY, "unknown_extension")
            return
            
        extension = self.extensions[extension_name]
        
        try:
            logger.info(f"Starting extension: {extension_name}")
            success = extension.start()
            
            if not success:
                logger.error(f"Failed to start extension: {extension_name}")
                # Add event publication for extension errors
                self.event_bus.publish("extension_error", extension_name)
                self._transition_to(State.READY, f"extension_{extension_name}_failed")
                
        except Exception as e:
            logger.error(f"Error running extension {extension_name}: {e}")
            # Add event publication for extension errors
            self.event_bus.publish("extension_error", extension_name)
            self._transition_to(State.READY, f"extension_{extension_name}_error")
            
    def shutdown(self) -> bool:
        """
        Shut down the Maggie AI Assistant.
        
        Returns
        -------
        bool
            True if shutdown initiated successfully
        """
        logger.info("Shutdown initiated")
        
        # Transition to shutdown state through cleanup
        if self.state != State.SHUTDOWN:
            self._transition_to(State.CLEANUP, "shutdown_requested")
            
        # If GUI exists, properly handle shutdown
        if hasattr(self, 'gui') and self.gui:
            # GUI will close naturally through the shutdown process
            # as implemented in the improved closeEvent handler
            pass
            
        return True
        
    def timeout(self) -> None:
        """
        Handle manual timeout/sleep request.
        
        Puts the assistant to sleep on manual request.
        """
        if self.state == State.READY or self.state == State.ACTIVE:
            logger.info("Manual sleep requested")
            self.stt_processor.speak("Going to sleep")
            self._transition_to(State.CLEANUP, "manual_timeout")
            
    def process_command(self, extension: Any = None) -> bool:
        """
        Process a command or activate a extension directly.
        
        Parameters
        ----------
        extension : Any, optional
            extension to activate directly, by default None
        
        Returns
        -------
        bool
            True if command processed successfully
        """
        if extension:
            extension_name = None
            for name, ext in self.extensions.items():
                if ext == extension:
                    extension_name = name
                    break
                    
            if extension_name:
                logger.info(f"Direct activation of extension: {extension_name}")
                self._transition_to(State.ACTIVE, f"extension_{extension_name}")
                self.thread_pool.submit(self._run_extension, extension_name)
                return True
                
        return False
    
    def set_gui(self, gui) -> None:
        """
        Set the GUI instance for this MaggieAI instance.
        
        Parameters
        ----------
        gui : MainWindow
            The GUI instance for updating visual elements
            
        Returns
        -------
        None
            This method doesn't return anything
        
        Notes
        -----
        This establishes bidirectional communication between
        the core and GUI components
        """
        self.gui = gui

    def update_gui(self, event_type: str, data: Any = None) -> None:
        """
        Update GUI with event data in a thread-safe manner.
        
        Parameters
        ----------
        event_type : str
            Type of event for GUI update
        data : Any, optional
            Event data for GUI update, by default None
            
        Returns
        -------
        None
        """
        if self.gui and hasattr(self.gui, "safe_update_gui"):
            if event_type == "state_change":
                self.gui.safe_update_gui(self.gui.update_state, data)
            elif event_type == "chat_message":
                is_user = data.get("is_user", False)
                message = data.get("message", "")
                self.gui.safe_update_gui(self.gui.log_chat, message, is_user)
            elif event_type == "event_log":
                self.gui.safe_update_gui(self.gui.log_event, data)
            elif event_type == "error_log":
                self.gui.safe_update_gui(self.gui.log_error, data)