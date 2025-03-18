"""
Maggie AI Assistant - GUI Module
===============================
GUI implementation for the Maggie AI Assistant using PySide6.

This module provides a graphical user interface for the Maggie AI Assistant,
offering visual state indication, command controls, and log displays.

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
# import sys
# import contextlib

# @contextlib.contextmanager
# def add_to_path(path):
#     original_sys_path = sys.path[:]
#     sys.path.append(path)
#     try:
#         yield
#     finally:
#         sys.path = original_sys_path

# Use the context manager to temporarily modify sys.path
# with add_to_path('C:\AI\claude\fresh\maggie\venv\Lib\site-packages\PySide6'):
#     # Third-party imports
#     from PySide6.QtWidgets import (
#         QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
#         QPushButton, QLabel, QSplitter, QTabWidget, QSizePolicy,
#         QListWidget, QListWidgetItem, QGroupBox, QFrame, QStatusBar,
#         QApplication
#     )
#     from PySide6.QtCore import Qt, Signal, QTimer, QMetaObject, Q_ARG, QVariant, QThread
#     from PySide6.QtGui import QFont, QColor, QIcon, QKeySequence, QShortcut

# sys.path is automatically reverted after exiting the 'with' block

# Third-party imports
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QLabel, QSplitter, QTabWidget, QSizePolicy,
    QListWidget, QListWidgetItem, QGroupBox, QFrame, QStatusBar,
    QApplication
)
from PySide6.Qt6Core import Qt, Signal, QTimer, QMetaObject, Q_ARG, QVariant, QThread
from PySide6.QtGui import QFont, QColor, QIcon, QKeySequence, QShortcut
from loguru import logger

__all__ = ['MainWindow']

class MainWindow(QMainWindow):
    """
    Main window for the Maggie AI Assistant GUI.
    
    Provides a graphical interface for interacting with Maggie AI, including
    status indicators, log displays, and extension controls.
    
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
        self.setMinimumSize(800, 600)
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
        
        # Set keyboard shortcuts
        self.setup_shortcuts()
        
    def _create_main_layout(self) -> None:
        """
        Create the main window layout.
        
        Sets up the split pane layout with logs on the left and
        controls on the right.
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
        self.content_splitter.setSizes([600, 200])
        
        # Create tab widget for logs
        self._create_log_tabs()
        
        # Create right panel contents
        self._create_right_panel()
        
        # Create bottom control panel
        self._create_control_panel()
        
    def _create_log_tabs(self) -> None:
        """
        Create the tabbed log display.
        
        Sets up tabs for chat, events, and errors.
        """
        self.log_tabs = QTabWidget()
        self.left_layout.addWidget(self.log_tabs)
        
        # Chat log
        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.log_tabs.addTab(self.chat_log, "Chat")
        
        # Event log
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.log_tabs.addTab(self.event_log, "Events")
        
        # Error log
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.log_tabs.addTab(self.error_log, "Errors")
        
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
                "chat_tab": "Alt+1",
                "event_tab": "Alt+2",
                "error_tab": "Alt+3",
            }
            
            # Alt+S: Sleep
            sleep_shortcut = QShortcut(QKeySequence(shortcut_config["sleep"]), self)
            sleep_shortcut.activated.connect(self.on_sleep_clicked)
            
            # Alt+Q: Shutdown
            shutdown_shortcut = QShortcut(QKeySequence(shortcut_config["shutdown"]), self)
            shutdown_shortcut.activated.connect(self.on_shutdown_clicked)
            
            # Alt+1, Alt+2, Alt+3: Switch tabs
            chat_tab_shortcut = QShortcut(QKeySequence(shortcut_config["chat_tab"]), self)
            chat_tab_shortcut.activated.connect(lambda: self.log_tabs.setCurrentIndex(0))
            
            event_tab_shortcut = QShortcut(QKeySequence(shortcut_config["event_tab"]), self)
            event_tab_shortcut.activated.connect(lambda: self.log_tabs.setCurrentIndex(1))
            
            error_tab_shortcut = QShortcut(QKeySequence(shortcut_config["error_tab"]), self)
            error_tab_shortcut.activated.connect(lambda: self.log_tabs.setCurrentIndex(2))
            
            logger.debug("Keyboard shortcuts configured")
            
        except Exception as e:
            logger.error(f"Error setting up shortcuts: {e}")
        
    
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
                    lambda checked, name=extension_name: self.on_extension_clicked(name)
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
        prefix = "User" if is_user else "Maggie"
        color = "blue" if is_user else "green"
        
        self.chat_log.append(f'<span style="color:gray">[{timestamp}]</span> <span style="color:{color}"><b>{prefix}:</b></span> {message}')
        
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
        
    def log_error(self, error: str) -> None:
        """
        Log an error message.
        
        Adds an error message to the error log with timestamp and switches
        to the error tab to ensure visibility.
        
        Parameters
        ----------
        error : str
            Error to log
        """
        timestamp = time.strftime("%H:%M:%S")
        self.error_log.append(f'<span style="color:gray">[{timestamp}]</span> <span style="color:red"><b>ERROR:</b></span> {error}')
        self.log_tabs.setCurrentIndex(2)  # Switch to error tab
        
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