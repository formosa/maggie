"""
Maggie AI Assistant - GUI Module
===============================
GUI implementation for the Maggie AI Assistant using PyQt6.
Provides a user interface with chat log, event log, and status indicators.
"""

import sys
import time
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QLabel, QSplitter, QTabWidget, QSizePolicy,
    QListWidget, QListWidgetItem, QGroupBox, QFrame, QStatusBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QIcon

class MainWindow(QMainWindow):
    """
    Main window for the Maggie AI Assistant GUI.
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
        
        # Create right panel contents
        self.create_right_panel()
        
        # Create bottom control panel
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
        
        # Initialize UI
        self.update_state("IDLE")
        self.log_event("Maggie AI Assistant started")
        
    def create_right_panel(self):
        """Create the contents of the right panel."""
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
        for utility_name in self.maggie_ai.utilities:
            utility_button = QPushButton(utility_name.replace("_", " ").title())
            utility_button.clicked.connect(
                lambda checked, name=utility_name: self.on_utility_clicked(name)
            )
            self.utilities_layout.addWidget(utility_button)
            
        self.right_layout.addWidget(self.utilities_group)
        
        # Add spacer
        self.right_layout.addStretch()
        
    def update_state(self, state):
        """
        Update the displayed state.
        
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
        self.state_display.setStyleSheet(f"font-size: 18px; font-weight: bold; background-color: {color};")
        
    def log_chat(self, message, is_user=False):
        """
        Log a chat message.
        
        Parameters
        ----------
        message : str
            Message to log
        is_user : bool, optional
            True if message is from user, False if from Maggie
        """
        timestamp = time.strftime("%H:%M:%S")
        prefix = "User" if is_user else "Maggie"
        color = "blue" if is_user else "green"
        
        self.chat_log.append(f'<span style="color:gray">[{timestamp}]</span> <span style="color:{color}"><b>{prefix}:</b></span> {message}')
        
    def log_event(self, event):
        """
        Log an event message.
        
        Parameters
        ----------
        event : str
            Event to log
        """
        timestamp = time.strftime("%H:%M:%S")
        self.event_log.append(f'<span style="color:gray">[{timestamp}]</span> {event}')
        
    def log_error(self, error):
        """
        Log an error message.
        
        Parameters
        ----------
        error : str
            Error to log
        """
        timestamp = time.strftime("%H:%M:%S")
        self.error_log.append(f'<span style="color:gray">[{timestamp}]</span> <span style="color:red"><b>ERROR:</b></span> {error}')
        self.log_tabs.setCurrentIndex(2)  # Switch to error tab
        
    def on_shutdown_clicked(self):
        """Handle shutdown button click."""
        self.log_event("Shutdown requested")
        self.maggie_ai.shutdown()
        
    def on_sleep_clicked(self):
        """Handle sleep button click."""
        self.log_event("Sleep requested")
        self.maggie_ai.timeout()
        
    def on_utility_clicked(self, utility_name):
        """
        Handle utility button click.
        
        Parameters
        ----------
        utility_name : str
            Name of the utility to activate
        """
        self.log_event(f"Utility requested: {utility_name}")
        if utility_name in self.maggie_ai.utilities:
            utility = self.maggie_ai.utilities[utility_name]
            self.maggie_ai.process_command(utility=utility)
