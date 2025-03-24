"""
Maggie AI Assistant - Core Application Module
=============================================

This module defines the core application class for Maggie AI Assistant,
implementing a Finite State Machine with event-driven state transitions.

The system is optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware,
with specific optimizations for resource management and performance.
"""

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

# Import standardized error handling and logging
from maggie.utils.error_handling import (
    safe_execute, ErrorCategory, ErrorSeverity, 
    with_error_handling, record_error
)
from maggie.utils.logging import ComponentLogger, log_operation, logging_context
from maggie.utils.resource.manager import ResourceManager
from maggie.utils.config.manager import ConfigManager

# Import enhanced state and event management
from maggie.core.state import State, StateTransition, StateManager
from maggie.core.event import EventBus, EventEmitter, EventListener, EventPriority

__all__ = ['MaggieAI']

class MaggieAI(EventEmitter, EventListener):
    """
    Core application class implementing the Maggie AI Assistant.
    
    This class manages the application lifecycle, state transitions,
    component initialization, and event handling. It implements a
    finite state machine with event-driven transitions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Maggie AI Assistant.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary for all components
        """
        # Initialize event bus before other components
        self.event_bus = EventBus()
        
        # Initialize base classes
        EventEmitter.__init__(self, self.event_bus)
        EventListener.__init__(self, self.event_bus)
        
        # Setup component logger
        self.logger = ComponentLogger('MaggieAI')
        
        # Store configuration
        self.config = config
        
        # Initialize state manager
        self.state_manager = StateManager(State.IDLE, self.event_bus)
        
        # Initialize resource manager
        self.resource_manager = ResourceManager(config)
        
        # Initialize component references (set during initialization)
        self.extensions = {}
        self.inactivity_timer = None
        self.inactivity_timeout = self.config.get('inactivity_timeout', 300)
        self.wake_word_detector = None
        self.stt_processor = None
        self.llm_processor = None
        self.tts_processor = None
        self.gui = None
        
        # Initialize thread pool
        cpu_config = self.config.get('cpu', {})
        max_threads = cpu_config.get('max_threads', 10)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_threads,
            thread_name_prefix='maggie_thread_'
        )
        
        # Register state handlers
        self._register_state_handlers()
        
        # Setup resource management
        self._setup_resource_management()
        
        self.logger.info('MaggieAI instance created')
    
    def _register_state_handlers(self) -> None:
        """Register state entry and exit handlers with the state manager."""
        # Register entry handlers
        self.state_manager.register_state_handler(State.IDLE, self._on_enter_idle, True)
        self.state_manager.register_state_handler(State.READY, self._on_enter_ready, True)
        self.state_manager.register_state_handler(State.ACTIVE, self._on_enter_active, True)
        self.state_manager.register_state_handler(State.CLEANUP, self._on_enter_cleanup, True)
        self.state_manager.register_state_handler(State.SHUTDOWN, self._on_enter_shutdown, True)
        
        # Register transition handlers for specific state pairs
        self.state_manager.register_transition_handler(
            State.ACTIVE, State.READY, self._on_transition_active_to_ready
        )
        
        self.logger.debug("State handlers registered")
    
    def _register_event_handlers(self) -> None:
        """Register event handlers with the event bus."""
        event_handlers = [
            ('wake_word_detected', self._handle_wake_word, EventPriority.HIGH),
            ('error_logged', self._handle_error, EventPriority.HIGH),
            ('command_detected', self._handle_command, EventPriority.NORMAL),
            ('inactivity_timeout', self._handle_timeout, EventPriority.NORMAL),
            ('extension_completed', self._handle_extension_completed, EventPriority.NORMAL),
            ('extension_error', self._handle_extension_error, EventPriority.NORMAL),
            ('low_memory_warning', self._handle_low_memory, EventPriority.LOW),
            ('gpu_memory_warning', self._handle_gpu_memory_warning, EventPriority.LOW),
        ]
        
        for event_type, handler, priority in event_handlers:
            self.listen(event_type, handler, priority=priority)
        
        self.logger.debug(f"Registered {len(event_handlers)} event handlers")
    
    def _setup_resource_management(self) -> None:
        """Set up resource management for efficient hardware utilization."""
        self.resource_manager.setup_gpu()
        self.logger.debug("Resource management setup complete")
    
    @log_operation(component='MaggieAI')
    def initialize_components(self) -> bool:
        """
        Initialize all system components.
        
        Returns
        -------
        bool
            True if all components initialized successfully
        """
        with logging_context(component='MaggieAI', operation='initialize_components') as ctx:
            try:
                # Register core services with the ServiceLocator
                self._register_core_services()
                
                # Initialize components in dependency order
                init_success = (
                    self._initialize_wake_word_detector() and
                    self._initialize_tts_processor() and
                    self._initialize_stt_processor() and
                    self._initialize_llm_processor()
                )
                
                if not init_success:
                    self.logger.error("Failed to initialize core components")
                    return False
                
                # Initialize extensions after core components
                self._initialize_extensions()
                
                # Start the event bus
                self.event_bus.start()
                
                self.logger.info('All components initialized successfully')
                return True
                
            except ImportError as import_error:
                self.logger.error(f"Failed to import required module: {import_error}")
                return False
            except Exception as e:
                self.logger.error(f"Error initializing components: {e}")
                return False
    
    def _register_core_services(self) -> bool:
        """
        Register core services with the ServiceLocator.
        
        Returns
        -------
        bool
            True if registration successful
        """
        try:
            from maggie.service.locator import ServiceLocator
            
            # Register core services
            ServiceLocator.register('event_bus', self.event_bus)
            ServiceLocator.register('resource_manager', self.resource_manager)
            ServiceLocator.register('maggie_ai', self)
            
            self.logger.debug("Core services registered with ServiceLocator")
            return True
            
        except ImportError as e:
            self.logger.error(f"Failed to import ServiceLocator: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error registering core services: {e}")
            return False
    
    def _initialize_wake_word_detector(self) -> bool:
        """
        Initialize the wake word detector component.
        
        Returns
        -------
        bool
            True if initialization successful
        """
        try:
            from maggie.service.stt.wake_word import WakeWordDetector
            from maggie.service.locator import ServiceLocator
            
            wake_word_config = self.config.get('stt', {}).get('wake_word', {})
            self.wake_word_detector = WakeWordDetector(wake_word_config)
            self.wake_word_detector.on_detected = lambda: self.event_bus.publish('wake_word_detected')
            
            ServiceLocator.register('wake_word_detector', self.wake_word_detector)
            self.logger.debug("Wake word detector initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing wake word detector: {e}")
            return False
    
    def _initialize_tts_processor(self) -> bool:
        """
        Initialize the TTS processor component.
        
        Returns
        -------
        bool
            True if initialization successful
        """
        try:
            from maggie.service.tts.processor import TTSProcessor
            from maggie.service.locator import ServiceLocator
            
            tts_config = self.config.get('tts', {})
            self.tts_processor = TTSProcessor(tts_config)
            
            ServiceLocator.register('tts_processor', self.tts_processor)
            self.logger.debug("TTS processor initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing TTS processor: {e}")
            return False
    
    def _initialize_stt_processor(self) -> bool:
        """
        Initialize the STT processor component.
        
        Returns
        -------
        bool
            True if initialization successful
        """
        try:
            from maggie.service.stt.processor import STTProcessor
            from maggie.service.locator import ServiceLocator
            
            stt_config = self.config.get('stt', {})
            self.stt_processor = STTProcessor(stt_config)
            
            # Link to TTS processor
            if self.tts_processor:
                self.stt_processor.tts_processor = self.tts_processor
            
            ServiceLocator.register('stt_processor', self.stt_processor)
            self.logger.debug("STT processor initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing STT processor: {e}")
            return False
    
    def _initialize_llm_processor(self) -> bool:
        """
        Initialize the LLM processor component.
        
        Returns
        -------
        bool
            True if initialization successful
        """
        try:
            from maggie.service.llm.processor import LLMProcessor
            from maggie.service.locator import ServiceLocator
            
            llm_config = self.config.get('llm', {})
            self.llm_processor = LLMProcessor(llm_config)
            
            ServiceLocator.register('llm_processor', self.llm_processor)
            self.logger.debug("LLM processor initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM processor: {e}")
            return False
    
    @log_operation(component='MaggieAI')
    def _initialize_extensions(self) -> None:
        """Initialize extension modules from configuration."""
        try:
            from maggie.extensions.registry import ExtensionRegistry
            
            extensions_config = self.config.get('extensions', {})
            registry = ExtensionRegistry()
            available_extensions = registry.discover_extensions()
            
            self.logger.info(
                f"Discovered {len(available_extensions)} extensions: "
                f"{', '.join(available_extensions.keys())}"
            )
            
            for extension_name, extension_config in extensions_config.items():
                if extension_config.get('enabled', True) is False:
                    self.logger.info(f"Extension {extension_name} is disabled in configuration")
                    continue
                    
                extension = registry.instantiate_extension(
                    extension_name, 
                    self.event_bus, 
                    extension_config
                )
                
                if extension is not None:
                    self.extensions[extension_name] = extension
                    self.logger.info(f"Initialized extension: {extension_name}")
                else:
                    self.logger.warning(f"Failed to initialize extension: {extension_name}")
            
            self.logger.info(f"Initialized {len(self.extensions)} extension modules")
            
        except Exception as e:
            self.logger.error(f"Error initializing extensions: {e}")
    
    @with_error_handling(error_category=ErrorCategory.SYSTEM, error_severity=ErrorSeverity.CRITICAL)
    def start(self) -> bool:
        """
        Start the Maggie AI Assistant.
        
        Returns
        -------
        bool
            True if started successfully
        """
        self.logger.info('Starting Maggie AI Assistant')
        
        if not self.initialize_components():
            self.logger.error('Failed to initialize components')
            return False
        
        # Register event handlers after components are initialized
        self._register_event_handlers()
        
        # Transition to initial state
        self.state_manager.transition_to(State.IDLE, 'startup')
        
        # Start resource monitoring
        self.resource_manager.start_monitoring()
        
        self.logger.info('Maggie AI Assistant started successfully')
        return True
    
    @with_error_handling(error_category=ErrorCategory.SYSTEM, error_severity=ErrorSeverity.ERROR)
    def stop(self) -> bool:
        """
        Stop the Maggie AI Assistant.
        
        Returns
        -------
        bool
            True if stopped successfully
        """
        self.logger.info('Stopping Maggie AI Assistant')
        self.state_manager.transition_to(State.SHUTDOWN, 'stop_requested')
        self.logger.info('Maggie AI Assistant stopped successfully')
        return True
    
    def _on_enter_idle(self, transition: StateTransition) -> None:
        """
        Handle transition to IDLE state.
        
        Parameters
        ----------
        transition : StateTransition
            The state transition that triggered this handler
        """
        # Cancel inactivity timer if active
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
            self.inactivity_timer = None
        
        # Stop speech components
        if self.stt_processor:
            self.stt_processor.stop_streaming()
            self.stt_processor.stop_listening()
        
        # Unload LLM to free resources
        if self.llm_processor:
            self.llm_processor.unload_model()
        
        # Start wake word detection
        if self.wake_word_detector:
            self.wake_word_detector.start()
        
        # Clear GPU memory
        self.resource_manager.clear_gpu_memory()
        
        self.logger.info('Entered IDLE state - waiting for wake word')
    
    def _on_enter_ready(self, transition: StateTransition) -> None:
        """
        Handle transition to READY state.
        
        Parameters
        ----------
        transition : StateTransition
            The state transition that triggered this handler
        """
        # Stop wake word detection
        if self.wake_word_detector:
            self.wake_word_detector.stop()
        
        # Start speech recognition
        if self.stt_processor:
            self.stt_processor.start_listening()
            self.stt_processor.start_streaming(
                on_intermediate=lambda text: self.event_bus.publish('intermediate_transcription', text),
                on_final=lambda text: self.event_bus.publish('final_transcription', text)
            )
            
            # Welcome prompt
            if self.tts_processor:
                self.tts_processor.speak('Ready for your command')
        
        # Start inactivity timer
        self._start_inactivity_timer()
        
        # Start listening for commands
        self.thread_pool.submit(self._listen_for_commands)
        
        self.logger.info('Entered READY state - listening for commands with real-time transcription')
    
    def _on_enter_active(self, transition: StateTransition) -> None:
        """
        Handle transition to ACTIVE state.
        
        Parameters
        ----------
        transition : StateTransition
            The state transition that triggered this handler
        """
        # Refresh inactivity timer
        self._start_inactivity_timer()
        
        self.logger.info('Entered ACTIVE state - executing command or extension')
    
    def _on_transition_active_to_ready(self, transition: StateTransition) -> None:
        """
        Handle specific transition from ACTIVE to READY state.
        
        Parameters
        ----------
        transition : StateTransition
            The state transition being handled
        """
        # Additional cleanup or setup needed when returning to READY from ACTIVE
        # Reset any active extension state if needed
        self.logger.debug('Transitioning from ACTIVE to READY state')
    
    def _on_enter_cleanup(self, transition: StateTransition) -> None:
        """
        Handle transition to CLEANUP state.
        
        Parameters
        ----------
        transition : StateTransition
            The state transition that triggered this handler
        """
        # Cancel inactivity timer
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
            self.inactivity_timer = None
        
        # Stop all extensions
        self._stop_all_extensions()
        
        # Stop speech recognition
        if self.stt_processor:
            self.stt_processor.stop_listening()
        
        # Unload models to free memory
        if self.llm_processor:
            self.llm_processor.unload_model()
        
        # Clear GPU memory
        self.resource_manager.clear_gpu_memory()
        
        # Determine next state based on trigger
        if transition.trigger == 'shutdown_requested':
            self.state_manager.transition_to(State.SHUTDOWN, 'cleanup_completed')
        else:
            self.state_manager.transition_to(State.IDLE, 'cleanup_completed')
        
        self.logger.info('Entered CLEANUP state - releasing resources')
    
    def _stop_all_extensions(self) -> None:
        """Stop all active extensions."""
        for extension_name, extension in self.extensions.items():
            try:
                if hasattr(extension, 'stop') and callable(extension.stop):
                    extension.stop()
                    self.logger.debug(f"Stopped extension: {extension_name}")
            except Exception as e:
                self.logger.error(f"Error stopping extension {extension_name}: {e}")
    
    def _on_enter_shutdown(self, transition: StateTransition) -> None:
        """
        Handle transition to SHUTDOWN state.
        
        Parameters
        ----------
        transition : StateTransition
            The state transition that triggered this handler
        """
        # Stop resource monitoring
        self.resource_manager.stop_monitoring()
        
        # Stop event bus
        self.event_bus.stop()
        
        # Clear GPU memory
        self.resource_manager.clear_gpu_memory()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True, timeout=5)
        
        self.logger.info('Entered SHUTDOWN state - application will exit')
    
    def _handle_wake_word(self, _: Any) -> None:
        """
        Handle wake word detection event.
        
        Parameters
        ----------
        _ : Any
            Event data (unused)
        """
        if self.state_manager.is_in_state(State.IDLE):
            self.logger.info('Wake word detected')
            self.state_manager.transition_to(State.READY, 'wake_word_detected')
    
    @log_operation(component='MaggieAI')
    def _handle_command(self, command: str) -> None:
        """
        Handle command detection event.
        
        Parameters
        ----------
        command : str
            Detected command text
        """
        if not self.state_manager.is_in_state(State.READY):
            return
        
        command = command.lower().strip()
        self.logger.info(f"Command detected: {command}")
        
        # Check for system commands
        if self._process_system_command(command):
            return
            
        # Check for extension commands
        if self._check_extension_commands(command):
            return
            
        # Handle unknown command
        if self.tts_processor:
            self.tts_processor.speak("I didn't understand that command")
        else:
            self.logger.warning('No TTS processor available for speech output')
            
        self.logger.warning(f"Unknown command: {command}")
    
    def _process_system_command(self, command: str) -> bool:
        """
        Process system-level commands.
        
        Parameters
        ----------
        command : str
            Command to process
            
        Returns
        -------
        bool
            True if command was processed
        """
        tts_processor = self.tts_processor
        
        # Sleep command
        if command in ['sleep', 'go to sleep']:
            if tts_processor:
                tts_processor.speak('Going to sleep')
            else:
                self.logger.warning('No TTS processor available for speech output')
                
            self.state_manager.transition_to(State.CLEANUP, 'sleep_command')
            return True
            
        # Shutdown command
        if command in ['shutdown', 'turn off']:
            if tts_processor:
                tts_processor.speak('Shutting down')
            else:
                self.logger.warning('No TTS processor available for speech output')
                
            self.state_manager.transition_to(State.CLEANUP, 'shutdown_requested')
            return True
            
        return False
    
    def _check_extension_commands(self, command: str) -> bool:
        """
        Check if command triggers an extension.
        
        Parameters
        ----------
        command : str
            Command to check
            
        Returns
        -------
        bool
            True if an extension was triggered
        """
        for extension_name, extension in self.extensions.items():
            extension_trigger = extension.get_trigger()
            
            if extension_trigger and extension_trigger in command:
                self.logger.info(f"Triggered extension: {extension_name}")
                self.state_manager.transition_to(
                    State.ACTIVE, 
                    f"extension_{extension_name}"
                )
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
        if self.state_manager.is_in_state(State.READY):
            self.logger.info('Inactivity timeout reached')
            
            if self.tts_processor:
                self.tts_processor.speak('Going to sleep due to inactivity')
                
            self.state_manager.transition_to(State.CLEANUP, 'inactivity_timeout')
    
    def _handle_extension_completed(self, extension_name: str) -> None:
        """
        Handle extension completion event.
        
        Parameters
        ----------
        extension_name : str
            Name of the completed extension
        """
        if self.state_manager.is_in_state(State.ACTIVE):
            self.logger.info(f"Extension completed: {extension_name}")
            self.state_manager.transition_to(
                State.READY, 
                f"extension_{extension_name}_completed"
            )
    
    def _handle_extension_error(self, extension_name: str) -> None:
        """
        Handle extension error event.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension with error
        """
        self.logger.error(f"Extension error: {extension_name}")
        
        if self.tts_processor:
            self.tts_processor.speak(
                f"There was a problem with the {extension_name} extension"
            )
            
        if self.gui:
            self.update_gui('error_log', {
                'message': f"Extension error: {extension_name}",
                'source': 'extension',
                'extension': extension_name
            })
            
        # Return to READY state if currently ACTIVE
        if self.state_manager.is_in_state(State.ACTIVE):
            self.state_manager.transition_to(
                State.READY, 
                f"extension_{extension_name}_error"
            )
    
    def _start_inactivity_timer(self) -> None:
        """Start or restart the inactivity timer."""
        if self.inactivity_timer:
            self.inactivity_timer.cancel()
            
        self.inactivity_timer = threading.Timer(
            self.inactivity_timeout,
            lambda: self.event_bus.publish('inactivity_timeout')
        )
        self.inactivity_timer.daemon = True
        self.inactivity_timer.start()
        
        self.logger.debug(f"Started inactivity timer: {self.inactivity_timeout}s")
    
    @with_error_handling(error_category=ErrorCategory.PROCESSING)
    def _listen_for_commands(self) -> None:
        """Listen for voice commands in a background thread."""
        if not self.state_manager.is_in_state(State.READY):
            return
            
        self.logger.debug('Listening for commands...')
        
        try:
            success, text = self.stt_processor.recognize_speech(timeout=10.0)
            
            if success and text:
                self.logger.info(f"Recognized: {text}")
                self.event_bus.publish('command_detected', text)
            else:
                # Resubmit if no command was recognized
                self.thread_pool.submit(self._listen_for_commands)
                
        except Exception as e:
            self.logger.error(f"Error listening for commands: {e}")
            # Resubmit even after error
            self.thread_pool.submit(self._listen_for_commands)
    
    @with_error_handling(error_category=ErrorCategory.EXTENSION)
    def _run_extension(self, extension_name: str) -> None:
        """
        Run an extension in a background thread.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension to run
        """
        if extension_name not in self.extensions:
            self.logger.error(f"Unknown extension: {extension_name}")
            self.state_manager.transition_to(State.READY, 'unknown_extension')
            return
            
        extension = self.extensions[extension_name]
        
        try:
            self.logger.info(f"Starting extension: {extension_name}")
            success = extension.start()
            
            if not success:
                self.logger.error(f"Failed to start extension: {extension_name}")
                self.event_bus.publish('extension_error', extension_name)
                self.state_manager.transition_to(
                    State.READY, 
                    f"extension_{extension_name}_failed"
                )
                
        except Exception as e:
            self.logger.error(f"Error running extension {extension_name}: {e}")
            self.event_bus.publish('extension_error', extension_name)
            self.state_manager.transition_to(
                State.READY, 
                f"extension_{extension_name}_error"
            )
    
    def _handle_error(self, error_data: Any) -> None:
        """
        Handle error event.
        
        Parameters
        ----------
        error_data : Any
            Error information
        """
        if self.gui:
            self.update_gui('error_log', error_data)
    
    def _handle_low_memory(self, _: Any) -> None:
        """
        Handle low memory warning event.
        
        Parameters
        ----------
        _ : Any
            Event data (unused)
        """
        # Unload models to free memory
        if self.llm_processor:
            self.llm_processor.unload_model()
            
        self.resource_manager.reduce_memory_usage()
        self.logger.warning('Low memory condition detected - unloading models')
    
    def _handle_gpu_memory_warning(self, _: Any) -> None:
        """
        Handle GPU memory warning event.
        
        Parameters
        ----------
        _ : Any
            Event data (unused)
        """
        # Clear GPU memory
        self.resource_manager.clear_gpu_memory()
        
        # Reduce GPU layers if possible
        if self.llm_processor and hasattr(self.llm_processor, 'reduce_gpu_layers'):
            self.llm_processor.reduce_gpu_layers()
            
        self.logger.warning('High GPU memory usage detected - freeing CUDA memory')
    
    def shutdown(self) -> bool:
        """
        Initiate a graceful shutdown.
        
        Returns
        -------
        bool
            True if shutdown initiated successfully
        """
        self.logger.info('Shutdown initiated')
        
        if not self.state_manager.is_in_state(State.SHUTDOWN):
            self.state_manager.transition_to(State.CLEANUP, 'shutdown_requested')
            
        return True
    
    def timeout(self) -> None:
        """Initiate a manual timeout/sleep."""
        if self.state_manager.is_in_state(State.READY) or self.state_manager.is_in_state(State.ACTIVE):
            self.logger.info('Manual sleep requested')
            
            if self.tts_processor:
                self.tts_processor.speak('Going to sleep')
                
            self.state_manager.transition_to(State.CLEANUP, 'manual_timeout')
    
    def process_command(self, extension: Any = None) -> bool:
        """
        Process a direct command or extension activation.
        
        Parameters
        ----------
        extension : Any, optional
            Extension to activate directly
            
        Returns
        -------
        bool
            True if command was processed
        """
        if extension:
            extension_name = None
            
            # Find extension name from instance
            for name, ext in self.extensions.items():
                if ext == extension:
                    extension_name = name
                    break
                    
            if extension_name:
                self.logger.info(f"Direct activation of extension: {extension_name}")
                self.state_manager.transition_to(
                    State.ACTIVE, 
                    f"extension_{extension_name}"
                )
                self.thread_pool.submit(self._run_extension, extension_name)
                return True
                
        return False
    
    def set_gui(self, gui) -> None:
        """
        Set the GUI reference.
        
        Parameters
        ----------
        gui : Any
            GUI instance
        """
        self.gui = gui
    
    def update_gui(self, event_type: str, data: Any = None) -> None:
        """
        Update the GUI with new information.
        
        Parameters
        ----------
        event_type : str
            Type of GUI update
        data : Any, optional
            Data for the update
        """
        if self.gui and hasattr(self.gui, 'safe_update_gui'):
            if event_type == 'state_change':
                self.gui.safe_update_gui(self.gui.update_state, data)
            elif event_type == 'chat_message':
                is_user = data.get('is_user', False)
                message = data.get('message', '')
                self.gui.safe_update_gui(self.gui.log_chat, message, is_user)
            elif event_type == 'event_log':
                self.gui.safe_update_gui(self.gui.log_event, data)
            elif event_type == 'error_log':
                self.gui.safe_update_gui(self.gui.log_error, data)