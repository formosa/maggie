"""
Maggie AI Assistant - GUI Module
===============================
GUI implementation for the Maggie AI Assistant using PySide6.

This module provides a graphical user interface for the Maggie AI Assistant,
offering visual state indication, command controls, text input capabilities,
and organized log displays for chat, events, and errors.

Examples
--------
>>> from maggie.utils.gui import MainWindow
>>> from maggie import MaggieAI
>>> from PySide6.QtWidgets import QApplication
>>> import sys
>>> config = {"threading": {"max_workers": 8}}
>>> maggie = MaggieAI(config)
>>> app = QApplication(sys.argv)
>>> window = MainWindow(maggie)
>>> window.show()
>>> sys.exit(app.exec())
"""

# Standard library imports
import sys
import time
from typing import Dict, Any, Optional, List, Callable

# Temporarily adjust PATH to allow PySide6 import
sys.path.append("C:\\AI\\claude\\fresh\\maggie\\venv\\Lib\\site-packages\\PySide6")
sys.path.append("C:\\AI\\claude\\fresh\\maggie\\venv\\Lib\\site-packages\\PySide6\\Qt6")
sys.path.append("C:\\AI\\claude\\fresh\\maggie\\venv\\Lib\\site-packages\\PySide6\\Qt6\\bin")

# Third-party imports
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QLabel, QSplitter, QTabWidget, QSizePolicy,
    QListWidget, QListWidgetItem, QGroupBox, QFrame, QStatusBar,
    QApplication, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QTimer, QMetaObject, Q_ARG, QThread
from PySide6.QtGui import QFont, QColor, QIcon, QKeySequence, QShortcut, QFocusEvent
from loguru import logger

__all__ = ['MainWindow']

class QVariant:
    """
    Simple variant class for Qt method invocation compatibility.
    
    Parameters
    ----------
    value : Any, optional
        The value to store
        
    Attributes
    ----------
    value : Any
        The stored value
    """
    def __init__(self, value=None):
        self.value = value
    def value(self):
        return self.value

class InputField(QLineEdit):
    """
    Custom input field with speech-to-text awareness.
    
    Provides an input field that can operate in two modes:
    manual typing or speech-to-text input display.
    
    Parameters
    ----------
    parent : QWidget
        Parent widget
    submit_callback : Callable
        Function to call when input is submitted
    
    Attributes
    ----------
    stt_mode : bool
        Whether in speech-to-text mode
    submit_callback : Callable
        Function to call when input is submitted
    """
    
    # Add a signal
    state_change_requested = Signal(str)

    def __init__(self, parent=None, submit_callback=None):
        """
        Initialize the input field.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None
        submit_callback : Callable, optional
            Function to call when input is submitted, by default None
        """
        super().__init__(parent)
        self.stt_mode = True
        self.submit_callback = submit_callback
        self.setPlaceholderText("Speak or type your message here...")
        
        # Connect events
        self.returnPressed.connect(self.on_return_pressed)
    
    def focusInEvent(self, event: QFocusEvent) -> None:
        """
        Handle focus in event.
        
        When input field gains focus, switch to manual input mode and
        request state transition if needed.
        
        Parameters
        ----------
        event : QFocusEvent
            Focus event
        """
        super().focusInEvent(event)
        self.stt_mode = False
        self.setStyleSheet("background-color: white;")
        
        # Signal the MainWindow to check for state transition
        self.state_change_requested.emit("ACTIVE")
    
    def focusOutEvent(self, event: QFocusEvent) -> None:
        """
        Handle focus out event.
        
        When input field loses focus, switch back to STT mode.
        
        Parameters
        ----------
        event : QFocusEvent
            Focus event
        """
        super().focusOutEvent(event)
        self.stt_mode = True
        self.update_appearance_for_state("IDLE")  # Will be overridden if state changes
    
    def on_return_pressed(self) -> None:
        """
        Handle return key press.
        
        Submits the input text and clears the field.
        """
        if self.submit_callback and self.text().strip():
            self.submit_callback(self.text())
            self.clear()
    
    def update_appearance_for_state(self, state: str) -> None:
        """
        Update appearance based on system state.
        
        Parameters
        ----------
        state : str
            Current system state
        """
        if state == "IDLE" and self.stt_mode:
            self.setStyleSheet("background-color: lightgray;")
            self.setReadOnly(True)
        else:
            self.setReadOnly(False)
            if not self.hasFocus():  # Don't change style if user is typing
                self.setStyleSheet("background-color: white;")

class MainWindow(QMainWindow):
    """
    Main window for the Maggie AI Assistant GUI.
    
    Provides a graphical interface for interacting with Maggie AI, including
    status indicators, log displays, text input, and extension controls.
    
    Parameters
    ----------
    maggie_ai : MaggieAI
        Reference to the main Maggie AI object
        
    Attributes
    ----------
    maggie_ai : MaggieAI
        Reference to the main Maggie AI object
    status_label : QLabel
        Label showing current system status
    chat_log : QTextEdit
        Text display for conversation history
    event_log : QTextEdit
        Text display for system events
    error_log : QTextEdit
        Text display for error messages
    input_field : InputField
        Text input field for user commands
    state_display : QLabel
        Visual indicator of current system state
    extension_buttons : Dict[str, QPushButton]
        Dictionary of extension activation buttons
    """
    
    def __init__(self, maggie_ai):
        """
        Initialize the main window.
        
        Parameters
        ----------
        maggie_ai : MaggieAI
            Reference to the main Maggie AI object
        """
        super().__init__()
        
        self.maggie_ai = maggie_ai
        self.setWindowTitle("Maggie AI Assistant")
        self.setMinimumSize(900, 700)
        self.is_shutting_down = False
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create status label
        self.status_label = QLabel("Status: IDLE")
        self.status_label.setStyleSheet("font-weight: bold;")
        self.status_bar.addPermanentWidget(self.status_label)
        
        # Create main content area with proper layout
        self._create_main_layout()
        
        # Initialize UI
        self.update_state("IDLE")
        self.log_event("Maggie AI Assistant started")

        # Subscribe to events from MaggieAI
        self.maggie_ai.event_bus.subscribe("state_changed", self._on_state_changed)
        self.maggie_ai.event_bus.subscribe("extension_completed", self._on_extension_completed)
        self.maggie_ai.event_bus.subscribe("extension_error", self._on_extension_error)
        self.maggie_ai.event_bus.subscribe("error_logged", self._on_error_logged)

        # Set keyboard shortcuts
        self.setup_shortcuts()
        
    def _create_main_layout(self) -> None:
        """
        Create the main window layout.
        
        Sets up a horizontal split pane layout with logs panel on the left and
        controls on the right. The logs panel contains Chat, Event Log, and
        Error Log sections arranged vertically.
        """
        # Create main content area
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.content_splitter)
        
        # Create left panel (chat and logs)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.content_splitter.addWidget(self.left_panel)
        
        # Create right panel (controls and extensions)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.content_splitter.addWidget(self.right_panel)
        
        # Set initial splitter sizes
        self.content_splitter.setSizes([700, 200])
        
        # Create log sections
        self._create_log_sections()
        
        # Create right panel contents
        self._create_right_panel()
        
        # Create bottom control panel
        self._create_control_panel()
        
    def _create_log_sections(self) -> None:
        """
        Create the log sections for chat, events, and errors.
        
        Sets up three vertical panels instead of tabs, with the chat
        panel taking more space than the others.
        """
        # Create a vertical splitter for all logs
        self.logs_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_layout.addWidget(self.logs_splitter)
        
        # Chat section (with input field below)
        self.chat_section = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_section)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat log with title
        self.chat_group = QGroupBox("Chat")
        self.chat_group_layout = QVBoxLayout(self.chat_group)
        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_group_layout.addWidget(self.chat_log)
        self.chat_layout.addWidget(self.chat_group)
        
        # Input field
        self.input_field = InputField(submit_callback=self._on_input_submitted)
        self.input_field.setFixedHeight(30)
        self.input_field.update_appearance_for_state("IDLE")
        self.input_field.state_change_requested.connect(self._on_input_state_change)
        self.chat_layout.addWidget(self.input_field)
    
        # Event log section with title
        self.event_group = QGroupBox("Event Log")
        self.event_group_layout = QVBoxLayout(self.event_group)
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_group_layout.addWidget(self.event_log)
        
        # Error log section with title
        self.error_group = QGroupBox("Error Log")
        self.error_group_layout = QVBoxLayout(self.error_group)
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_group_layout.addWidget(self.error_log)
        
        # Add all sections to the splitter
        self.logs_splitter.addWidget(self.chat_section)
        self.logs_splitter.addWidget(self.event_group)
        self.logs_splitter.addWidget(self.error_group)
        
        # Set initial sizes with chat taking more space
        self.logs_splitter.setSizes([400, 150, 150])
        
    def _create_right_panel(self) -> None:
        """
        Create the contents of the right panel.
        
        Sets up the state display and extension buttons in the right panel.
        """
        # Current state display
        self.state_group = QGroupBox("Current State")
        self.state_layout = QVBoxLayout(self.state_group)
        self.state_display = QLabel("IDLE")
        self.state_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_display.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.state_layout.addWidget(self.state_display)
        self.right_layout.addWidget(self.state_group)
        
        # Utilities group
        self.extensions_group = QGroupBox("Utilities")
        self.extensions_layout = QVBoxLayout(self.extensions_group)
        
        # Add extension buttons based on loaded extensions
        self._create_extension_buttons()
        
        self.right_layout.addWidget(self.extensions_group)
        
        # Add spacer
        self.right_layout.addStretch()
        
    def _create_control_panel(self) -> None:
        """
        Create the bottom control panel.
        
        Sets up buttons for shutdown and sleep commands.
        """
        self.control_panel = QWidget()
        self.control_layout = QHBoxLayout(self.control_panel)
        self.main_layout.addWidget(self.control_panel)
        
        # Create control buttons
        self.shutdown_button = QPushButton("Shutdown")
        self.shutdown_button.clicked.connect(self.on_shutdown_clicked)
        self.control_layout.addWidget(self.shutdown_button)
        
        self.sleep_button = QPushButton("Sleep")
        self.sleep_button.clicked.connect(self.on_sleep_clicked)
        self.control_layout.addWidget(self.sleep_button)
        
    def _on_input_state_change(self, requested_state: str) -> None:
        """
        Handle state change requests from the input field.
        
        Parameters
        ----------
        requested_state : str
            The requested state to transition to
            
        Returns
        -------
        None
        """
        if self.maggie_ai.state.name == "IDLE" and requested_state == "ACTIVE":
            # Transition from IDLE to READY when input field is activated
            self.maggie_ai._transition_to(self.maggie_ai.state.READY, "input_field_activated")
            self.log_event("State transition requested by input field")

    def _on_error_logged(self, error_data):
        """
        Handle error logging events from the system.
        
        Parameters
        ----------
        error_data : str or dict
            Error message or dictionary with error details
            
        Returns
        -------
        None
        """
        if isinstance(error_data, dict):
            message = error_data.get('message', 'Unknown error')
            source = error_data.get('source', 'system')
            self.log_error(f"[{source}] {message}")
        else:
            self.log_error(str(error_data))

    def setup_shortcuts(self) -> None:
        """
        Set up keyboard shortcuts for the GUI.
        
        Configures keyboard shortcuts for common actions to improve usability.
        
        Returns
        -------
        None
        """
        try:
            # Define shortcut configuration (could be loaded from settings)
            shortcut_config = {
                "sleep": "Alt+S",
                "shutdown": "Alt+Q",
                "focus_input": "Alt+I",
            }
            
            # Alt+S: Sleep
            sleep_shortcut = QShortcut(QKeySequence(shortcut_config["sleep"]), self)
            sleep_shortcut.activated.connect(self.on_sleep_clicked)
            
            # Alt+Q: Shutdown
            shutdown_shortcut = QShortcut(QKeySequence(shortcut_config["shutdown"]), self)
            shutdown_shortcut.activated.connect(self.on_shutdown_clicked)
            
            # Alt+I: Focus input field
            input_shortcut = QShortcut(QKeySequence(shortcut_config["focus_input"]), self)
            input_shortcut.activated.connect(lambda: self.input_field.setFocus())
            
            logger.debug("Keyboard shortcuts configured")
            
        except Exception as e:
            logger.error(f"Error setting up shortcuts: {e}")
    
    def _create_extension_buttons(self) -> None:
        """
        Create buttons for each available extension.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            This method doesn't return anything
        
        Notes
        -----
        Creates a button for each extension in the Maggie AI system,
        with appropriate handlers and shortcuts
        """
        try:
            # Clean up existing buttons first
            self._cleanup_extension_buttons()
            
            self.extension_buttons = {}
            for extension_name in self.maggie_ai.extensions:
                display_name = extension_name.replace("_", " ").title()
                extension_button = QPushButton(display_name)
                extension_button.clicked.connect(
                    lambda checked=False, name=extension_name: self.on_extension_clicked(name)
                )
                self.extensions_layout.addWidget(extension_button)
                self.extension_buttons[extension_name] = extension_button
                
                # Set shortcut if it's the recipe creator
                if extension_name == "recipe_creator":
                    try:
                        recipe_shortcut = QShortcut(QKeySequence("Alt+R"), self)
                        recipe_shortcut.activated.connect(
                            lambda: self.on_extension_clicked("recipe_creator")
                        )
                    except Exception as e:
                        logger.error(f"Error setting up recipe shortcut: {e}")
        except Exception as e:
            logger.error(f"Error creating extension buttons: {e}")
        
    def _on_state_changed(self, transition) -> None:
        """
        Handle state transition events.
        
        Parameters
        ----------
        transition : StateTransition
            Data object containing transition information
            
        Returns
        -------
        None
            This method doesn't return anything
        
        Notes
        -----
        Updates the GUI to reflect the new system state
        and logs the transition for user awareness
        """
        try:
            # Validate transition object
            if not transition or not hasattr(transition, 'to_state') or not hasattr(transition, 'from_state'):
                logger.error("Invalid state transition object received")
                return
                
            # Extract state information with defensive coding
            to_state = getattr(transition, 'to_state', None)
            from_state = getattr(transition, 'from_state', None)
            
            if to_state is None or from_state is None:
                logger.error("Invalid state transition object: to_state or from_state is None")
                return
                
            to_state_name = getattr(to_state, 'name', 'UNKNOWN')
            from_state_name = getattr(from_state, 'name', 'UNKNOWN')
            trigger = getattr(transition, 'trigger', 'UNKNOWN')
            
            # Update the state display
            self.update_state(to_state_name)
            
            # Update input field appearance
            self.input_field.update_appearance_for_state(to_state_name)
            
            # Log the transition
            self.log_event(f"State changed: {from_state_name} -> "
                        f"{to_state_name} (trigger: {trigger})")
        except Exception as e:
            logger.error(f"Error handling state transition: {e}")

    def _cleanup_extension_buttons(self) -> None:
        """
        Clean up extension buttons before recreating them.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            This method doesn't return anything
        
        Notes
        -----
        Removes existing buttons from the layout and properly
        disposes of them to prevent memory leaks
        """
        try:
            if hasattr(self, 'extension_buttons'):
                for button in self.extension_buttons.values():
                    self.extensions_layout.removeWidget(button)
                    button.deleteLater()
                self.extension_buttons.clear()
        except Exception as e:
            logger.error(f"Error cleaning up extension buttons: {e}")

    def update_state(self, state: str) -> None:
        """
        Update the displayed state.
        
        Updates the GUI to reflect the current system state with appropriate
        color coding for visual feedback.
        
        Parameters
        ----------
        state : str
            New state name
            
        Returns
        -------
        None
        """
        # Define valid states
        valid_states = ["IDLE", "STARTUP", "READY", "ACTIVE", "BUSY", "CLEANUP", "SHUTDOWN"]
        
        # Validate state
        if state not in valid_states:
            logger.warning(f"Invalid state: {state}. Defaulting to IDLE.")
            state = "IDLE"
        
        self.state_display.setText(state)
        self.status_label.setText(f"Status: {state}")
        
        # Update state display color based on state
        color_map = {
            "IDLE": "lightgray",
            "STARTUP": "lightblue",
            "READY": "lightgreen",
            "ACTIVE": "yellow",
            "BUSY": "orange",
            "CLEANUP": "pink",
            "SHUTDOWN": "red"
        }
        
        color = color_map.get(state, "white")
        self.state_display.setStyleSheet(f"font-size: 18px; font-weight: bold; background-color: {color}; padding: 5px;")
        
        # Update input field based on state
        self.input_field.update_appearance_for_state(state)
        
        logger.debug(f"GUI state updated to: {state}")
        
    def refresh_extensions(self) -> None:
        """
        Refresh the extension buttons to reflect current available extensions.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            This method doesn't return anything
        
        Notes
        -----
        Updates the GUI when extensions are added or removed
        """
        self._create_extension_buttons()
        self.log_event("Extension list updated")

    def log_chat(self, message: str, is_user: bool = False) -> None:
        """
        Log a chat message.
        
        Adds a message to the chat log with appropriate formatting and timestamps.
        
        Parameters
        ----------
        message : str
            Message to log
        is_user : bool, optional
            True if message is from user, False if from Maggie, by default False
        """
        timestamp = time.strftime("%H:%M:%S")
        prefix = "user" if is_user else "Maggie"
        color = "blue" if is_user else "green"
        
        self.chat_log.append(f'<span style="color:gray">[{timestamp}]</span> <span style="color:{color}"><b>&lt; {prefix} &gt;</b></span> {message}')
        
    def log_event(self, event: str) -> None:
        """
        Log an event message.
        
        Adds an event message to the event log with timestamp.
        
        Parameters
        ----------
        event : str
            Event to log
        """
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.append(f'<span style="color:gray">[{timestamp}]</span> {event}')
        
        logger.debug(f"Event logged: {event}")

    # Add to MainWindow class
    def show_download_progress(self, progress_data):
        """
        Display download progress in the GUI.
        
        Parameters
        ----------
        progress_data : dict
            Dictionary containing progress information
            
        Returns
        -------
        None
        """
        item = progress_data.get('item', 'file')
        percent = progress_data.get('percent', 0)
        status = f"Downloading {item}: {percent}% complete"
        self.status_label.setText(status)
        
        if percent >= 100:
            # Reset status after 3 seconds
            QTimer.singleShot(3000, lambda: self.status_label.setText(f"Status: {self.state_display.text()}"))   

    def log_error(self, error: str) -> None:
        """
        Log an error message.
        
        Adds an error message to the error log with timestamp.
        
        Parameters
        ----------
        error : str
            Error to log
        """
        timestamp = time.strftime("%H:%M:%S")
        
        # Format the error with HTML for better styling
        formatted_error = f'<span style="color:gray">[{timestamp}]</span> <span style="color:red"><b>ERROR:</b></span> {error}'
        
        # Add the error to the log
        self.error_log.append(formatted_error)
        
        # Make the Error Log section visible by updating splitter sizes
        current_sizes = self.logs_splitter.sizes()
        if current_sizes[2] < 100:  # If Error Log section is too small
            self.logs_splitter.setSizes([current_sizes[0], current_sizes[1], 200])
        
        logger.error(f"Error logged in GUI: {error}")
        
    def on_shutdown_clicked(self) -> None:
        """
        Handle shutdown button click.
        
        Initiates the shutdown process for the Maggie AI Assistant.
        """
        self.log_event("Shutdown requested")
        self.maggie_ai.shutdown()
        
        logger.info("Shutdown initiated from GUI")
        
    def on_sleep_clicked(self) -> None:
        """
        Handle sleep button click.
        
        Puts the Maggie AI Assistant into sleep mode.
        """
        self.log_event("Sleep requested")
        self.maggie_ai.timeout()
        
        logger.info("Sleep initiated from GUI")
        
    def on_extension_clicked(self, extension_name: str) -> None:
        """
        Handle extension button click.
        
        Activates the specified extension.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension to activate
        """
        self.log_event(f"Extension requested: {extension_name}")
        if extension_name in self.maggie_ai.extensions:
            extension = self.maggie_ai.extensions[extension_name]
            self.maggie_ai.process_command(extension=extension)
            
            logger.info(f"Extension '{extension_name}' activated from GUI")
    
    def _on_input_submitted(self, text: str) -> None:
        """
        Handle input submission from the input field.
        
        Parameters
        ----------
        text : str
            The input text to process
            
        Returns
        -------
        None
        """
        if not text.strip():
            return
            
        # Log the user input to the chat display
        self.log_chat(text, is_user=True)
        
        # Process the command through the Maggie AI core
        if self.maggie_ai.state.name == "IDLE":
            # If in IDLE state, transition to READY first
            self.maggie_ai._transition_to(self.maggie_ai.state.READY, "user_input")
            
        # Send the command to be processed
        self.maggie_ai.event_bus.publish("command_detected", text)
        
        logger.debug(f"User input submitted: {text}")
            
    def closeEvent(self, event) -> None:
        """
        Handle window close event.
        
        Ensures proper shutdown when the window is closed.
        
        Parameters
        ----------
        event : QCloseEvent
            The close event from PyQt
        
        Returns
        -------
        None
        """
        self.log_event("Window close requested, shutting down")
        
        # Set a flag to track shutdown progress
        self.is_shutting_down = True
        
        # Start shutdown
        self.maggie_ai.shutdown()
        
        # Add a short delay to allow shutdown to progress before closing
        QTimer.singleShot(2000, lambda: event.accept())
        
        logger.info("GUI window closed, shutdown initiated")

    def safe_update_gui(self, func: Callable, *args, **kwargs) -> None:
        """
        Safely update the GUI from another thread.
        
        Parameters
        ----------
        func : Callable
            Function to call in the GUI thread
        *args
            Arguments to pass to the function
        **kwargs
            Keyword arguments to pass to the function
                
        Returns
        -------
        None
            This method doesn't return anything
        
        Notes
        -----
        Uses Qt's invokeMethod mechanism for thread-safe GUI updates
        with fallback to direct calls when appropriate
        """
        # Check if already in GUI thread first for efficiency
        if QThread.currentThread() == self.thread():
            try:
                func(*args, **kwargs)
                return
            except Exception as e:
                logger.error(f"Error calling GUI method directly: {e}")
                return
        
        # Thread-safe call for non-GUI threads
        try:
            # Convert args to Q_ARG objects
            q_args = []
            for arg in args:
                try:
                    # Handle primitive types directly
                    if isinstance(arg, (int, float, bool, str)):
                        q_args.append(Q_ARG(type(arg), arg))
                    else:
                        # For complex objects, use QVariant
                        q_args.append(Q_ARG(QVariant, QVariant(arg)))
                except Exception as e:
                    logger.debug(f"Error converting argument to Q_ARG: {e}")
                    # Fall back to string conversion
                    q_args.append(Q_ARG(str, str(arg)))
            
            # Invoke the method in the GUI thread
            QMetaObject.invokeMethod(
                self, 
                func.__name__, 
                Qt.ConnectionType.QueuedConnection, 
                *q_args
            )
        except Exception as e:
            logger.error(f"Error invoking GUI method {func.__name__}: {e}")

    def _on_extension_completed(self, extension_name: str) -> None:
        """
        Handle extension completion events.
        
        Parameters
        ----------
        extension_name : str
            Name of the completed extension
            
        Returns
        -------
        None
            This method doesn't return anything
        """
        self.log_event(f"Extension completed: {extension_name}")
        self.update_state("READY")  # Update to match core state

    def _on_extension_error(self, extension_name: str) -> None:
        """
        Handle extension error events.
        
        Parameters
        ----------
        extension_name : str
            Name of the extension that encountered an error
            
        Returns
        -------
        None
            This method doesn't return anything
        """
        self.log_error(f"Error in extension: {extension_name}")
        self.update_state("READY")  # Update to match core state
        
    def update_stt_text(self, text: str) -> None:
        """
        Update the input field with speech-to-text results.
        
        Parameters
        ----------
        text : str
            The recognized text from speech recognition
            
        Returns
        -------
        None
        """
        if self.input_field.stt_mode:
            self.input_field.setText(text)