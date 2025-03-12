"""
Maggie AI Assistant - Core Implementation
=======================================
Core FSM implementation of the Maggie AI Assistant.

This module implements a simplified Finite State Machine (FSM) architecture
with event-driven state transitions and optimized resource management.
Specifically tuned for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.
"""

from enum import Enum, auto
import threading
import queue
import time
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

class State(Enum):
    """
    Simplified state enumeration for Maggie AI Assistant.
    
    Reduces the previous state machine to five essential states
    for improved clarity and manageability.
    """
    IDLE = auto()      # Waiting for wake word, minimal resource usage
    READY = auto()     # Listening for commands, resources initialized
    ACTIVE = auto()    # Processing commands and running utilities
    CLEANUP = auto()   # Cleaning up resources
    SHUTDOWN = auto()  # Final state before application exit

@dataclass
class StateTransition:
    """
    Data structure for state transition events.
    
    Parameters
    ----------
    from_state : State
        Previous state
    to_state : State
        New state
    trigger : str
        Event that triggered the transition
    timestamp : float
        Unix timestamp of the transition
    """
    from_state: State
    to_state: State
    trigger: str
    timestamp: float

class EventBus:
    """
    Centralized event management system.
    
    Handles event publication, subscription, and dispatching with
    thread-safety and prioritization.
    
    Attributes
    ----------
    subscribers : Dict[str, List[Callable]]
        Event subscribers mapped by event type
    queue : queue.PriorityQueue
        Priority queue for event processing
    running : bool
        Whether the event bus is currently running
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
            
        return True
        
    def _process_events(self) -> None:
        """
        Process events from the queue and dispatch to subscribers.
        """
        while self.running:
            try:
                priority, event = self.queue.get(timeout=0.1)
                
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
                continue
            except Exception as e:
                logger.error(f"Error processing events: {e}")

class MaggieAI:
    """
    Core implementation of the Maggie AI Assistant.
    
    Implements a simplified Finite State Machine architecture with
    event-driven state transitions and resource management optimized
    for AMD Ryzen 9 5900X and NVIDIA RTX 3080.
    
    Attributes
    ----------
    state : State
        Current state of the assistant
    event_bus : EventBus
        Central event bus for component communication
    config : Dict[str, Any]
        Configuration dictionary
    utilities : Dict[str, Any]
        Loaded utility modules
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Maggie AI Core.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        """
        self.config = config
        self.state = State.IDLE
        self.event_bus = EventBus()
        self.utilities = {}
        
        # Component references - will be initialized during startup
        self.hardware_manager = None
        self.wake_word_detector = None
        self.speech_processor = None
        self.llm_processor = None
        self.gui = None
        
        # Worker thread pool - optimized for Ryzen 9 5900X
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get("threading", {}).get("max_workers", 8),
            thread_name_prefix="maggie_worker"
        )
        
        # Thread management
        self.inactivity_timer = None
        self.inactivity_timeout = config.get("inactivity_timeout", 300)  # 5 minutes
        
        # Setup state transition handlers
        self.transition_handlers = {
            State.IDLE: self._on_enter_idle,
            State.READY: self._on_enter_ready,
            State.ACTIVE: self._on_enter_active,
            State.CLEANUP: self._on_enter_cleanup,
            State.SHUTDOWN: self._on_enter_shutdown
        }
        
        # Register event handlers
        self.event_bus.subscribe("wake_word_detected", self._handle_wake_word)
        self.event_bus.subscribe("command_detected", self._handle_command)
        self.event_bus.subscribe("inactivity_timeout", self._handle_timeout)
        self.event_bus.subscribe("utility_completed", self._handle_utility_completed)
        
    def initialize_components(self):
        """
        Initialize all required components.
        
        Returns
        -------
        bool
            True if all components initialized successfully
        """
        try:
            # Initialize hardware manager
            from hardware_manager import HardwareManager
            self.hardware_manager = HardwareManager(self.config)
            
            # Apply hardware optimizations to config
            self.config = self.hardware_manager.optimize_config(self.config)
            
            # Initialize wake word detector
            from wake_word import WakeWordDetector
            self.wake_word_detector = WakeWordDetector(self.config.get("wake_word", {}))
            self.wake_word_detector.on_detected = lambda: self.event_bus.publish("wake_word_detected")
            
            # Initialize speech processor
            from speech_processor import SpeechProcessor
            self.speech_processor = SpeechProcessor(self.config.get("speech", {}))
            
            # Initialize LLM processor
            from llm_processor import LLMProcessor
            self.llm_processor = LLMProcessor(self.config.get("llm", {}))
            
            # Initialize utilities
            self._initialize_utilities()
            
            # Start event bus
            self.event_bus.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
            
    def _initialize_utilities(self):
        """
        Initialize utility modules based on configuration.
        """
        utilities_config = self.config.get("utilities", {})
        
        # Load recipe creator if configured
        if "recipe_creator" in utilities_config:
            from utils.recipe_creator import RecipeCreator
            self.utilities["recipe_creator"] = RecipeCreator(
                self.event_bus, 
                utilities_config["recipe_creator"]
            )
            
        # Load additional utilities as needed
        # ...
            
    def start(self):
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
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting Maggie AI: {e}")
            return False
            
    def stop(self):
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
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Maggie AI: {e}")
            return False
            
    def _transition_to(self, new_state: State, trigger: str):
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
            
    def _on_enter_idle(self, transition: StateTransition):
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
        if self.speech_processor:
            self.speech_processor.stop_listening()
            
        # Unload LLM model to save memory
        if self.llm_processor:
            self.llm_processor.unload_model()
            
        # Start wake word detector
        if self.wake_word_detector:
            self.wake_word_detector.start()
            
    def _on_enter_ready(self, transition: StateTransition):
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
        if self.speech_processor:
            self.speech_processor.start_listening()
            self.speech_processor.speak("Ready for your command")
            
        # Start inactivity timer
        self._start_inactivity_timer()
        
        # Start listening for commands
        self.thread_pool.submit(self._listen_for_commands)
            
    def _on_enter_active(self, transition: StateTransition):
        """
        Handle entering ACTIVE state.
        
        Parameters
        ----------
        transition : StateTransition
            State transition information
        """
        # Reset inactivity timer
        self._start_inactivity_timer()
            
    def _on_enter_cleanup(self, transition: StateTransition):
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
            
        # Stop all utilities
        for utility in self.utilities.values():
            utility.stop()
            
        # Stop speech processor
        if self.speech_processor:
            self.speech_processor.stop_listening()
            
        # Unload LLM model
        if self.llm_processor:
            self.llm_processor.unload_model()
            
        # Determine next state based on trigger
        if transition.trigger == "shutdown_requested":
            self._transition_to(State.SHUTDOWN, "cleanup_completed")
        else:
            self._transition_to(State.IDLE, "cleanup_completed")
            
    def _on_enter_shutdown(self, transition: StateTransition):
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
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
            
    def _handle_wake_word(self, _):
        """
        Handle wake word detection event.
        """
        if self.state == State.IDLE:
            self._transition_to(State.READY, "wake_word_detected")
            
    def _handle_command(self, command: str):
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
        
        # Handle system commands
        if command in ["sleep", "go to sleep"]:
            self.speech_processor.speak("Going to sleep")
            self._transition_to(State.CLEANUP, "sleep_command")
            return
            
        if command in ["shutdown", "turn off"]:
            self.speech_processor.speak("Shutting down")
            self._transition_to(State.CLEANUP, "shutdown_requested")
            return
            
        # Check for utility commands
        for utility_name, utility in self.utilities.items():
            utility_trigger = utility.get_trigger()
            if utility_trigger and utility_trigger in command:
                self._transition_to(State.ACTIVE, f"utility_{utility_name}")
                self.thread_pool.submit(self._run_utility, utility_name)
                return
                
        # Handle unknown command
        self.speech_processor.speak("I didn't understand that command")
        
    def _handle_timeout(self, _):
        """
        Handle inactivity timeout event.
        """
        if self.state == State.READY:
            self.speech_processor.speak("Going to sleep due to inactivity")
            self._transition_to(State.CLEANUP, "inactivity_timeout")
            
    def _handle_utility_completed(self, utility_name: str):
        """
        Handle utility completion event.
        
        Parameters
        ----------
        utility_name : str
            Name of the completed utility
        """
        if self.state == State.ACTIVE:
            self._transition_to(State.READY, f"utility_{utility_name}_completed")
        
    def _start_inactivity_timer(self):
        """
        Start or reset the inactivity timer.
        """
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
            
        self.inactivity_timer = threading.Timer(
            self.inactivity_timeout,
            lambda: self.event_bus.publish("inactivity_timeout")
        )
        self.inactivity_timer.daemon = True
        self.inactivity_timer.start()
        
    def _listen_for_commands(self):
        """
        Listen for commands in the READY state.
        """
        if self.state != State.READY:
            return
            
        try:
            success, text = self.speech_processor.recognize_speech(timeout=10.0)
            
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
            
    def _run_utility(self, utility_name: str):
        """
        Run a utility.
        
        Parameters
        ----------
        utility_name : str
            Name of the utility to run
        """
        if utility_name not in self.utilities:
            logger.error(f"Unknown utility: {utility_name}")
            self._transition_to(State.READY, "unknown_utility")
            return
            
        utility = self.utilities[utility_name]
        
        try:
            success = utility.start()
            
            if not success:
                logger.error(f"Failed to start utility: {utility_name}")
                self._transition_to(State.READY, f"utility_{utility_name}_failed")
                
        except Exception as e:
            logger.error(f"Error running utility {utility_name}: {e}")
            self._transition_to(State.READY, f"utility_{utility_name}_error")
            
    def shutdown(self):
        """
        Shut down the Maggie AI Assistant.
        """
        return self.stop()
        
    def timeout(self):
        """
        Handle manual timeout/sleep request.
        """
        if self.state == State.READY or self.state == State.ACTIVE:
            self.speech_processor.speak("Going to sleep")
            self._transition_to(State.CLEANUP, "manual_timeout")
            
    def process_command(self, utility=None):
        """
        Process a command or activate a utility directly.
        
        Parameters
        ----------
        utility : UtilityBase, optional
            Utility to activate directly, by default None
        
        Returns
        -------
        bool
            True if command processed successfully
        """
        if utility:
            utility_name = None
            for name, util in self.utilities.items():
                if util == utility:
                    utility_name = name
                    break
                    
            if utility_name:
                self._transition_to(State.ACTIVE, f"utility_{utility_name}")
                self.thread_pool.submit(self._run_utility, utility_name)
                return True
                
        return False