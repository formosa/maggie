"""
Maggie AI Assistant - GUI Module
===============================
GUI implementation for the Maggie AI Assistant using PyQt6.

This module provides a graphical user interface for the Maggie AI Assistant,
offering visual state indication, command controls, and log displays.

Examples
--------
>>> from utils.gui import MainWindow
>>> from maggie import MaggieAI
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

# Third-party imports
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QLabel, QSplitter, QTabWidget, QSizePolicy,
    QListWidget, QListWidgetItem, QGroupBox, QFrame, QStatusBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QIcon, QKeySequence, QShortcut
from loguru import logger

__all__ = ['MainWindow']

class MainWindow(QMainWindow):
    """
    Main window for the Maggie AI Assistant GUI.
    
    Provides a graphical interface for interacting with Maggie AI, including
    status indicators, log displays, and utility controls.
    
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
    utility_buttons : Dict[str, QPushButton]
        Dictionary of utility activation buttons
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
        
        # Create right panel (controls and utilities)
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
        """
        try:
            # Alt+S: Sleep
            sleep_shortcut = QShortcut(QKeySequence("Alt+S"), self)
            sleep_shortcut.activated.connect(self.on_sleep_clicked)
            
            # Alt+Q: Shutdown
            shutdown_shortcut = QShortcut(QKeySequence("Alt+Q"), self)
            shutdown_shortcut.activated.connect(self.on_shutdown_clicked)
            
            # Alt+1, Alt+2, Alt+3: Switch tabs
            chat_tab_shortcut = QShortcut(QKeySequence("Alt+1"), self)
            chat_tab_shortcut.activated.connect(lambda: self.log_tabs.setCurrentIndex(0))
            
            event_tab_shortcut = QShortcut(QKeySequence("Alt+2"), self)
            event_tab_shortcut.activated.connect(lambda: self.log_tabs.setCurrentIndex(1))
            
            error_tab_shortcut = QShortcut(QKeySequence("Alt+3"), self)
            error_tab_shortcut.activated.connect(lambda: self.log_tabs.setCurrentIndex(2))
            
            logger.debug("Keyboard shortcuts configured")
            
        except Exception as e:
            logger.error(f"Error setting up shortcuts: {e}")
        
    def create_right_panel(self) -> None:
        """
        Create the contents of the right panel.
        
        Sets up the state display and utility buttons in the right panel.
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
        self.utilities_group = QGroupBox("Utilities")
        self.utilities_layout = QVBoxLayout(self.utilities_group)
        
        # Add utility buttons based on loaded utilities
        self._create_utility_buttons()
        
        self.right_layout.addWidget(self.utilities_group)
        
        # Add spacer
        self.right_layout.addStretch()
    
    def _create_right_panel(self) -> None:
        """
        Create the contents of the right panel.
        
        Sets up the state display and utility buttons in the right panel.
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
        self.utilities_group = QGroupBox("Utilities")
        self.utilities_layout = QVBoxLayout(self.utilities_group)
        
        # Add utility buttons based on loaded utilities
        self._create_utility_buttons()
        
        self.right_layout.addWidget(self.utilities_group)
        
        # Add spacer
        self.right_layout.addStretch()
        
    def _create_utility_buttons(self) -> None:
        """
        Create buttons for each available utility.
        
        Creates a button for each utility in the Maggie AI system,
        with appropriate handlers and shortcuts.
        """
        self.utility_buttons = {}
        for utility_name in self.maggie_ai.utilities:
            display_name = utility_name.replace("_", " ").title()
            utility_button = QPushButton(display_name)
            utility_button.clicked.connect(
                lambda checked, name=utility_name: self.on_utility_clicked(name)
            )
            self.utilities_layout.addWidget(utility_button)
            self.utility_buttons[utility_name] = utility_button
            
            # Set shortcut if it's the recipe creator
            if utility_name == "recipe_creator":
                try:
                    recipe_shortcut = QShortcut(QKeySequence("Alt+R"), self)
                    recipe_shortcut.activated.connect(
                        lambda: self.on_utility_clicked("recipe_creator")
                    )
                except Exception as e:
                    logger.error(f"Error setting up recipe shortcut: {e}")
        
    def update_state(self, state: str) -> None:
        """
        Update the displayed state.
        
        Updates the GUI to reflect the current system state with appropriate
        color coding for visual feedback.
        
        Parameters
        ----------
        state : str
            New state name
        """
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
        
    def on_utility_clicked(self, utility_name: str) -> None:
        """
        Handle utility button click.
        
        Activates the specified utility.
        
        Parameters
        ----------
        utility_name : str
            Name of the utility to activate
        """
        self.log_event(f"Utility requested: {utility_name}")
        if utility_name in self.maggie_ai.utilities:
            utility = self.maggie_ai.utilities[utility_name]
            self.maggie_ai.process_command(utility=utility)
            
            logger.info(f"Utility '{utility_name}' activated from GUI")
            
    def closeEvent(self, event) -> None:
        """
        Handle window close event.
        
        Ensures proper shutdown when the window is closed.
        
        Parameters
        ----------
        event : QCloseEvent
            The close event
        """
        self.log_event("Window close requested, shutting down")
        self.maggie_ai.shutdown()
        event.accept()
        
        logger.info("GUI window closed, shutdown initiated")