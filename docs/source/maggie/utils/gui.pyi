"""
Maggie AI Assistant - Graphical User Interface Module
====================================================

This module provides the graphical user interface for the Maggie AI Assistant, 
implemented using PySide6 (Qt for Python). It serves as the visual front-end 
for interacting with the assistant's state machine architecture, speech processing,
and event management systems.

The GUI displays system state, provides input capabilities, and visualizes the 
activity of the core AI components. It implements reactive design patterns to
respond to state changes from the underlying finite state machine architecture.

Classes
-------
InputField : PySide6.QtWidgets.QLineEdit
    Custom text input field that integrates with speech-to-text capabilities
MainWindow : PySide6.QtWidgets.QMainWindow, StateAwareComponent, EventListener
    Main application window containing all GUI elements and event handlers

Notes
-----
This module follows the Model-View-Controller (MVC) pattern combined with 
state-driven reactive UI updating. The interface reacts to state changes in 
the underlying Maggie AI system, adapting the display and available controls
based on the current operational state of the AI assistant.

References
----------
.. [1] PySide6 Documentation: https://doc.qt.io/qtforpython-6/
.. [2] Finite State Machine pattern: https://en.wikipedia.org/wiki/Finite-state_machine
.. [3] Event-driven programming: https://en.wikipedia.org/wiki/Event-driven_programming
"""

import sys
import time
from typing import Dict, Any, Optional, List, Callable, Union, Type, cast

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QLabel, QSplitter, QTabWidget, QSizePolicy, 
    QListWidget, QListWidgetItem, QGroupBox, QFrame, QStatusBar, 
    QApplication, QLineEdit
)
from PySide6.QtCore import (
    Qt, Signal, QTimer, QMetaObject, QPropertyAnimation, 
    QEasingCurve, Q_ARG, QRect, QPoint, QObject, Slot, Property
)
from PySide6.QtGui import (
    QFont, QColor, QIcon, QKeySequence, QShortcut, QFocusEvent
)

from maggie.core.state import State, StateTransition, StateAwareComponent
from maggie.core.event import EventListener, EventPriority, EventEmitter
from maggie.utils.error_handling import safe_execute, ErrorCategory, with_error_handling, record_error
from maggie.utils.logging import ComponentLogger, log_operation
from maggie.service.locator import ServiceLocator

class InputField(QLineEdit):
    """
    Custom input field that integrates with Maggie's speech-to-text capabilities.
    
    This specialized QLineEdit extends Qt's standard input field with capabilities
    to automatically update based on speech recognition results, handle state changes,
    and manage the focus transitions between voice and text input modes.
    
    The InputField serves as the primary interface for user input, supporting both
    keyboard typing and automatic transcription from the speech-to-text processor.
    It displays real-time transcription updates and can submit completed inputs for
    processing.
    
    Attributes
    ----------
    state_change_requested : Signal
        Signal emitted when an input field state change is requested,
        carrying the requested State as its parameter
    stt_mode : bool
        Flag indicating if speech-to-text mode is active
    submit_callback : Optional[Callable[[str], None]]
        Callback function to handle submitted text
    intermediate_text : str
        Holds the current partial transcription text
    animation : QPropertyAnimation
        Animation for visual feedback effects
        
    See Also
    --------
    QLineEdit : The Qt class this inherits from
    MainWindow : The parent window containing this input field
    
    Notes
    -----
    This component implements a reactive UI pattern, where the appearance and
    behavior change based on system state (e.g., IDLE, READY, ACTIVE) and 
    user interaction (focus events).
    """
    
    state_change_requested: Signal = Signal(State)
    
    def __init__(self, parent: Optional[QWidget] = None, 
                 submit_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Initialize the custom input field.
        
        Creates an input field with specialized behavior for the Maggie AI 
        Assistant, integrating with the speech-to-text subsystem and handling
        state transitions.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, typically the MainWindow
        submit_callback : callable, optional
            Function to call when text is submitted, taking the submitted 
            text as its parameter
            
        Notes
        -----
        Initializes internal state tracking variables and sets up the
        visual appearance of the input field.
        """
        ...
    
    def focusInEvent(self, event: QFocusEvent) -> None:
        """
        Handle focus-in events to transition from voice to text input mode.
        
        When the input field gains focus, it switches from STT mode to manual
        typing mode, updates its visual appearance, and pauses the speech
        transcription process.
        
        Parameters
        ----------
        event : QFocusEvent
            The Qt focus event object
            
        Notes
        -----
        This method overrides QLineEdit.focusInEvent to add custom behavior
        for the Maggie AI system.
        """
        ...
    
    def focusOutEvent(self, event: QFocusEvent) -> None:
        """
        Handle focus-out events to return to voice input mode.
        
        When the input field loses focus, it reverts to STT mode if no text
        is present, updates its visual appearance, and may resume speech
        transcription.
        
        Parameters
        ----------
        event : QFocusEvent
            The Qt focus event object
            
        Notes
        -----
        This method overrides QLineEdit.focusOutEvent to add custom behavior
        for the Maggie AI system.
        """
        ...
    
    def on_return_pressed(self) -> None:
        """
        Handle the Enter/Return key press to submit text input.
        
        Validates that the input contains non-whitespace text before
        forwarding it to the submit callback function. Clears the input
        field after submission.
        
        Notes
        -----
        Connected to the returnPressed signal of QLineEdit during initialization.
        """
        ...
    
    def update_appearance_for_state(self, state: State) -> None:
        """
        Update the visual appearance of the input field based on system state.
        
        Adjusts the style (colors, read-only status) of the input field to
        reflect the current operational state of the Maggie AI system.
        
        Parameters
        ----------
        state : State
            The current system state from the state machine
            
        Notes
        -----
        Uses the style information from the State enum to maintain consistent
        visual theming across the application.
        """
        ...
    
    def update_intermediate_text(self, text: str) -> None:
        """
        Update the field with intermediate (in-progress) transcription text.
        
        Displays partial speech recognition results in the input field when
        in STT mode and not currently focused for typing.
        
        Parameters
        ----------
        text : str
            The partial transcription text from the speech recognition system
            
        Notes
        -----
        Used for real-time feedback during speech recognition. The text is
        styled differently to indicate its provisional nature.
        """
        ...
    
    def set_final_text(self, text: str) -> None:
        """
        Set confirmed transcription text from speech recognition.
        
        Updates the input field with final transcription results and may
        automatically submit the text based on configuration settings.
        
        Parameters
        ----------
        text : str
            The final transcription text from the speech recognition system
            
        Notes
        -----
        When auto_submit is enabled in the STT configuration, this method will
        also trigger submission of the recognized text.
        """
        ...
    
    def _pause_transcription(self) -> None:
        """
        Pause the speech transcription process.
        
        Signals the STT processor to temporarily stop processing audio input,
        typically when the user is manually typing.
        
        Notes
        -----
        Uses the ServiceLocator to find and control the STT processor component.
        Logs any errors that occur during this process.
        """
        ...
    
    def _resume_transcription(self) -> None:
        """
        Resume the speech transcription process.
        
        Signals the STT processor to continue processing audio input after
        being paused, typically when returning to voice input mode.
        
        Notes
        -----
        Uses the ServiceLocator to find and control the STT processor component.
        Logs any errors that occur during this process.
        """
        ...


class MainWindow(QMainWindow, StateAwareComponent, EventListener):
    """
    Main application window for the Maggie AI Assistant.
    
    Serves as the primary visual interface for the Maggie AI system, integrating with
    the state machine architecture, event system, and other core components. The window
    displays the current system state, transcription results, event logs, error logs,
    and provides controls for interacting with the AI assistant and its extensions.
    
    The MainWindow implements the StateAwareComponent interface to respond to state
    changes and the EventListener interface to handle system events. It maintains a
    reactive UI that adapts based on the current operational state of the system.
    
    Attributes
    ----------
    maggie_ai : MaggieAI
        Reference to the main Maggie AI system instance
    state_manager : StateManager
        Reference to the system state manager
    logger : ComponentLogger
        Logger for the MainWindow component
    input_field : InputField
        Custom input field for text and voice input
    chat_log : QTextEdit
        Text area displaying conversation history
    event_log : QTextEdit
        Text area displaying system events
    error_log : QTextEdit
        Text area displaying error messages
    state_display : QLabel
        Label showing the current system state
    extension_buttons : Dict[str, QPushButton]
        Buttons for activating system extensions
    is_shutting_down : bool
        Flag indicating if the system is in shutdown process
        
    See Also
    --------
    InputField : Custom input field for text and voice input
    StateAwareComponent : Interface for state-aware components
    EventListener : Interface for components that listen to events
    
    Notes
    -----
    This class implements a complex UI with multiple panels, reactive components,
    and integration points with the core AI system. It follows both the 
    Model-View-Controller pattern and reactive programming principles.
    """
    
    def __init__(self, maggie_ai: Any) -> None:
        """
        Initialize the main application window.
        
        Creates the main window UI and connects it to the provided Maggie AI
        instance, establishing the necessary communication channels with the
        state machine, event system, and other core components.
        
        Parameters
        ----------
        maggie_ai : MaggieAI
            Reference to the main Maggie AI system instance
            
        Notes
        -----
        Sets up the complete UI layout including panels for chat, events, errors,
        system state display, extension controls, and registers event handlers.
        """
        ...
    
    def _register_event_handlers(self) -> None:
        """
        Register event handlers with the event system.
        
        Connects various system events to appropriate handler methods within
        the MainWindow. This enables the UI to react to state changes, errors,
        extension activities, and transcription updates.
        
        Notes
        -----
        Uses the EventListener.listen method to register callbacks with
        appropriate priority levels for different event types.
        """
        ...
    
    def _create_main_layout(self) -> None:
        """
        Create the main window layout structure.
        
        Sets up the primary split view with left and right panels, configuring
        the basic structure of the UI without creating specific controls.
        
        Notes
        -----
        This is called during initialization to create the basic window layout
        before populating it with specific UI components.
        """
        ...
    
    def _create_log_sections(self) -> None:
        """
        Create log display sections in the UI.
        
        Sets up the chat log, event log, and error log sections with appropriate
        container widgets and scrollable text areas.
        
        Notes
        -----
        Called by _create_main_layout to populate the left panel of the UI with
        the primary logging components.
        """
        ...
    
    def _create_chat_section(self) -> None:
        """
        Create the chat interface section.
        
        Sets up the chat history display and input field for user interaction
        with the AI assistant.
        
        Notes
        -----
        Creates the InputField instance and connects it to the appropriate
        signal handlers.
        """
        ...
    
    def _create_right_panel(self) -> None:
        """
        Create the right panel of the main UI.
        
        Sets up the state display and extensions section in the right panel
        of the main window.
        
        Notes
        -----
        Called by _create_main_layout to populate the right panel of the UI with
        state information and extension controls.
        """
        ...
    
    def _create_control_panel(self) -> None:
        """
        Create the bottom control panel with system buttons.
        
        Sets up the main control buttons (shutdown, sleep) at the bottom of
        the window.
        
        Notes
        -----
        Creates buttons with appropriate event handlers for controlling the
        overall system state.
        """
        ...
    
    def _on_input_state_change(self, requested_state: State) -> None:
        """
        Handle state change requests from the input field.
        
        Processes state transition requests that originate from user interaction
        with the input field, such as activating the input field when the system
        is in IDLE state.
        
        Parameters
        ----------
        requested_state : State
            The state that the input field is requesting the system transition to
            
        Notes
        -----
        This method initiates appropriate state transitions based on the input
        field's requests, reflecting user interaction patterns.
        """
        ...
    
    def _on_error_logged(self, error_data: Any) -> None:
        """
        Handle error events from the system.
        
        Processes error information received through the event system, formatting
        and displaying it in the error log with appropriate styling.
        
        Parameters
        ----------
        error_data : Dict[str, Any] or str
            Error information, either as a structured dictionary or simple string
            
        Notes
        -----
        Formats error messages with source, state information, and timestamp before
        displaying them in the error log panel.
        """
        ...
    
    def _apply_state_specific_error_styling(self, error_data: Dict[str, Any]) -> None:
        """
        Apply state-specific styling to errors in the UI.
        
        Applies visual highlighting to the error log panel based on the state
        information associated with an error, providing visual cues about the
        context of errors.
        
        Parameters
        ----------
        error_data : Dict[str, Any]
            Error information dictionary containing state and style information
            
        Notes
        -----
        Temporarily changes the style of the error log panel to highlight errors,
        then reverts to normal styling after a delay.
        """
        ...
    
    def setup_shortcuts(self) -> None:
        """
        Set up keyboard shortcuts for the application.
        
        Configures keyboard shortcuts for common actions like sleep, shutdown,
        and focusing the input field, enhancing usability without requiring
        mouse interaction.
        
        Notes
        -----
        Creates QShortcut instances and connects them to appropriate handler methods.
        Uses Alt+S for sleep, Alt+Q for shutdown, and Alt+I to focus the input field.
        """
        ...
    
    def _create_extension_buttons(self) -> None:
        """
        Create buttons for available extensions.
        
        Dynamically generates buttons for each extension registered with the
        Maggie AI system, allowing users to activate extensions with a single click.
        
        Notes
        -----
        Cleans up any existing extension buttons before creating new ones.
        Also sets up keyboard shortcuts for certain extensions (like Alt+R for recipe_creator).
        """
        ...
    
    def _on_state_changed(self, transition: Any) -> None:
        """
        Handle system state change events.
        
        Processes state transition information received through the event system,
        updating the UI to reflect the new state and logging the transition details.
        
        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
            
        Notes
        -----
        Updates the state display, logs the transition, updates the input field's
        appearance, and animates the transition visually.
        """
        ...
    
    def _animate_transition(self, transition: StateTransition) -> None:
        """
        Animate state transitions in the UI.
        
        Creates visual animations for state transitions, with different animation
        styles based on the nature of the transition (fade, bounce, slide).
        
        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
            
        Notes
        -----
        Uses Qt property animations to create smooth visual transitions between
        states, enhancing the user experience of state changes.
        """
        ...
    
    def _cleanup_extension_buttons(self) -> None:
        """
        Clean up extension buttons when refreshing or closing.
        
        Removes existing extension buttons from the UI and cleans up associated
        resources to prevent memory leaks.
        
        Notes
        -----
        Called before creating new extension buttons and during shutdown to
        ensure proper resource management.
        """
        ...
    
    def update_ui_for_state(self, state: State) -> None:
        """
        Update the UI appearance based on system state.
        
        Adjusts the visual styling of UI components to reflect the current
        operational state of the Maggie AI system, providing visual feedback
        about the system's status.
        
        Parameters
        ----------
        state : State
            The current system state
            
        Notes
        -----
        Updates colors, labels, and component states based on the provided
        state information.
        """
        ...
    
    def refresh_extensions(self) -> None:
        """
        Refresh the extensions display.
        
        Recreates the extension buttons to reflect any changes in available
        extensions, ensuring the UI stays in sync with the system's capabilities.
        
        Notes
        -----
        Calls _create_extension_buttons to rebuild the extension controls and
        logs the update action.
        """
        ...
    
    def log_chat(self, message: str, is_user: bool = False) -> None:
        """
        Log a chat message in the chat history.
        
        Adds a formatted message to the chat log, distinguishing between user
        inputs and assistant responses with different styling.
        
        Parameters
        ----------
        message : str
            The message text to log
        is_user : bool, optional
            Flag indicating if this is a user message (True) or assistant message (False)
            
        Notes
        -----
        Formats messages with timestamps and color-coding to distinguish message sources.
        """
        ...
    
    def log_event(self, event: str) -> None:
        """
        Log a system event in the event log.
        
        Adds a formatted event notification to the event log with a timestamp.
        
        Parameters
        ----------
        event : str
            The event description to log
            
        Notes
        -----
        Used for system-level events that aren't errors or chat messages.
        """
        ...
    
    def show_download_progress(self, progress_data: Dict[str, Any]) -> None:
        """
        Display download progress in the status bar.
        
        Shows progress information for ongoing downloads in the status bar,
        providing feedback for operations like model downloads.
        
        Parameters
        ----------
        progress_data : Dict[str, Any]
            Dictionary containing progress information including:
            - item: What's being downloaded
            - percent: Download completion percentage
            
        Notes
        -----
        Temporarily updates the status bar with progress information, then
        reverts to normal status display after completion.
        """
        ...
    
    def log_error(self, error: str) -> None:
        """
        Log an error message in the error log.
        
        Adds a formatted error message to the error log panel with a timestamp
        and error styling.
        
        Parameters
        ----------
        error : str
            The error message to log
            
        Notes
        -----
        Automatically expands the error log panel if it's minimized to ensure
        errors are visible to the user.
        """
        ...
    
    def on_shutdown_clicked(self) -> None:
        """
        Handle shutdown button clicks.
        
        Initiates the shutdown sequence for the Maggie AI system when the
        shutdown button is clicked.
        
        Notes
        -----
        Logs the shutdown action and calls the maggie_ai.shutdown() method
        to begin the system shutdown process.
        """
        ...
    
    def on_sleep_clicked(self) -> None:
        """
        Handle sleep button clicks.
        
        Initiates the timeout/sleep sequence for the Maggie AI system when the
        sleep button is clicked.
        
        Notes
        -----
        Logs the sleep action and calls the maggie_ai.timeout() method
        to transition the system to the IDLE state.
        """
        ...
    
    def on_extension_clicked(self, extension_name: str) -> None:
        """
        Handle extension button clicks.
        
        Activates the specified extension when its button is clicked in the UI.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension to activate
            
        Notes
        -----
        Logs the extension activation and processes the extension command
        through the Maggie AI system.
        """
        ...
    
    def _on_input_submitted(self, text: str) -> None:
        """
        Handle text submitted from the input field.
        
        Processes text input from the user, either typed or from speech recognition,
        sending it to the appropriate processing pipeline in the Maggie AI system.
        
        Parameters
        ----------
        text : str
            The text submitted by the user
            
        Notes
        -----
        Validates non-empty input, logs it to the chat history, handles any needed
        state transitions, and publishes a command event for processing.
        """
        ...
    
    def closeEvent(self, event: Any) -> None:
        """
        Handle window close events.
        
        Processes the window close action, initiating a clean shutdown of the
        Maggie AI system before allowing the window to close.
        
        Parameters
        ----------
        event : QCloseEvent
            The Qt close event object
            
        Notes
        -----
        This overrides the QMainWindow.closeEvent method to add custom shutdown
        behavior, ensuring clean resource release.
        """
        ...
    
    @Slot(str)
    def update_intermediate_text(self, text: str) -> None:
        """
        Update intermediate transcription text.
        
        Qt slot that forwards intermediate (in-progress) transcription text
        to the input field for display.
        
        Parameters
        ----------
        text : str
            The intermediate transcription text
            
        Notes
        -----
        This is a Qt slot that can be invoked via the Qt signal-slot mechanism,
        enabling thread-safe UI updates.
        """
        ...
    
    @Slot(str)
    def update_final_text(self, text: str) -> None:
        """
        Update final transcription text.
        
        Qt slot that forwards final (confirmed) transcription text to the
        input field for display and possible submission.
        
        Parameters
        ----------
        text : str
            The final transcription text
            
        Notes
        -----
        This is a Qt slot that can be invoked via the Qt signal-slot mechanism,
        enabling thread-safe UI updates.
        """
        ...
    
    def _on_intermediate_transcription(self, text: str) -> None:
        """
        Handle intermediate transcription events.
        
        Processes intermediate transcription results from the speech recognition
        system, ensuring they're properly dispatched to the UI thread.
        
        Parameters
        ----------
        text : str
            The intermediate transcription text
            
        Notes
        -----
        Checks the current thread and uses Qt's invoke method pattern to ensure
        UI updates happen on the main thread, preventing threading issues.
        """
        ...
    
    def _on_final_transcription(self, text: str) -> None:
        """
        Handle final transcription events.
        
        Processes final transcription results from the speech recognition system,
        ensuring they're properly dispatched to the UI thread.
        
        Parameters
        ----------
        text : str
            The final transcription text
            
        Notes
        -----
        Checks the current thread and uses Qt's invoke method pattern to ensure
        UI updates happen on the main thread, preventing threading issues.
        """
        ...
    
    def _on_extension_completed(self, extension_name: str) -> None:
        """
        Handle extension completion events.
        
        Processes notifications that an extension has completed its task,
        updating the UI state accordingly.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension that completed its task
            
        Notes
        -----
        Logs the completion event and updates the UI to the READY state,
        indicating the system is available for new interactions.
        """
        ...
    
    def _on_extension_error(self, extension_name: str) -> None:
        """
        Handle extension error events.
        
        Processes error notifications from extensions, updating the UI
        state and error log accordingly.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension that encountered an error
            
        Notes
        -----
        Logs the error event and updates the UI to the READY state,
        allowing new interactions despite the error.
        """
        ...
    
    def on_enter_idle(self, transition: StateTransition) -> None:
        """
        Handle transitions to IDLE state.
        
        Updates the UI when the system enters the IDLE state, typically
        indicating the system is inactive but ready for wake word activation.
        
        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
            
        Notes
        -----
        This method is registered with the state manager and called automatically
        when the system enters the IDLE state.
        """
        ...
    
    def on_enter_ready(self, transition: StateTransition) -> None:
        """
        Handle transitions to READY state.
        
        Updates the UI when the system enters the READY state, indicating
        the system is ready for user input.
        
        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
            
        Notes
        -----
        This method is registered with the state manager and called automatically
        when the system enters the READY state.
        """
        ...
    
    def on_enter_active(self, transition: StateTransition) -> None:
        """
        Handle transitions to ACTIVE state.
        
        Updates the UI when the system enters the ACTIVE state, indicating
        the system is actively processing user interaction.
        
        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
            
        Notes
        -----
        This method is registered with the state manager and called automatically
        when the system enters the ACTIVE state.
        """
        ...
    
    def on_enter_busy(self, transition: StateTransition) -> None:
        """
        Handle transitions to BUSY state.
        
        Updates the UI when the system enters the BUSY state, indicating
        the system is performing intensive processing operations.
        
        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
            
        Notes
        -----
        This method is registered with the state manager and called automatically
        when the system enters the BUSY state.
        """
        ...