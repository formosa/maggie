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

References
----------
* PySide6 Documentation: https://doc.qt.io/qtforpython-6/
* Finite State Machine pattern: https://en.wikipedia.org/wiki/Finite-state_machine
* Event-driven programming: https://en.wikipedia.org/wiki/Event-driven_programming
"""

import sys
import time
from typing import Dict, Any, Optional, List, Callable, Union

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

__all__ = ['MainWindow', 'InputField']

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
    """

    state_change_requested = Signal(State)

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
        super().__init__(parent)
        self.logger = ComponentLogger('InputField')
        self.stt_mode = True
        self.submit_callback = submit_callback
        self.setPlaceholderText('Speak or type your message here...')
        self.intermediate_text = ''
        self.returnPressed.connect(self.on_return_pressed)
        self.animation = QPropertyAnimation(self, b'styleSheet')
        self.animation.setDuration(300)

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
        """
        super().focusInEvent(event)
        self.stt_mode = False
        self.setStyleSheet('background-color: white; color: black;')
        self.state_change_requested.emit(State.ACTIVE)
        self._pause_transcription()

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
        """
        super().focusOutEvent(event)
        self.stt_mode = True
        self.update_appearance_for_state(State.IDLE)
        if not self.text().strip():
            self._resume_transcription()

    def on_return_pressed(self) -> None:
        """
        Handle the Enter/Return key press to submit text input.

        Validates that the input contains non-whitespace text before
        forwarding it to the submit callback function. Clears the input
        field after submission.
        """
        if self.submit_callback and self.text().strip():
            self.submit_callback(self.text())
            self.clear()
            self.intermediate_text = ''

    def update_appearance_for_state(self, state: State) -> None:
        """
        Update the visual appearance of the input field based on system state.

        Adjusts the style (colors, read-only status) of the input field to
        reflect the current operational state of the Maggie AI system.

        Parameters
        ----------
        state : State
            The current system state from the state machine
        """
        state_style = state.get_style()
        if state == State.IDLE and self.stt_mode:
            self.setStyleSheet(f"background-color: lightgray; color: {state_style.get('color', 'black')};")
            self.setReadOnly(True)
        else:
            self.setReadOnly(False)
            if not self.hasFocus():
                self.setStyleSheet(f"background-color: white; color: {state_style.get('color', 'black')};")

    def update_intermediate_text(self, text: str) -> None:
        """
        Update the field with intermediate (in-progress) transcription text.

        Displays partial speech recognition results in the input field when
        in STT mode and not currently focused for typing.

        Parameters
        ----------
        text : str
            The partial transcription text from the speech recognition system
        """
        if self.stt_mode and not self.hasFocus():
            self.intermediate_text = text
            self.setText(text)
            self.animation.setStartValue('background-color: white; color: gray;')
            self.animation.setEndValue('background-color: white; color: gray; border: 1px solid #CCCCCC;')
            self.animation.start()

    def set_final_text(self, text: str) -> None:
        """
        Set confirmed transcription text from speech recognition.

        Updates the input field with final transcription results and may
        automatically submit the text based on configuration settings.

        Parameters
        ----------
        text : str
            The final transcription text from the speech recognition system
        """
        if self.stt_mode and not self.hasFocus():
            self.setText(text)
            self.animation.setStartValue('background-color: white; color: gray;')
            self.animation.setEndValue('background-color: white; color: black; border: 1px solid #000000;')
            self.animation.start()
            stt_processor = ServiceLocator.get('stt_processor')
            if (stt_processor and hasattr(stt_processor, 'config') and
                stt_processor.config.get('whisper_streaming', {}).get('auto_submit', False) and
                self.submit_callback and text.strip()):
                self.submit_callback(text)
                self.clear()
                self.intermediate_text = ''

    def _pause_transcription(self) -> None:
        """
        Pause the speech transcription process.

        Signals the STT processor to temporarily stop processing audio input,
        typically when the user is manually typing.
        """
        try:
            stt_processor = ServiceLocator.get('stt_processor')
            if stt_processor and hasattr(stt_processor, 'pause_streaming'):
                stt_processor.pause_streaming()
                self.logger.debug('Transcription paused')
        except Exception as e:
            self.logger.warning(f"Failed to pause transcription: {e}")

    def _resume_transcription(self) -> None:
        """
        Resume the speech transcription process.

        Signals the STT processor to continue processing audio input after
        being paused, typically when returning to voice input mode.
        """
        try:
            stt_processor = ServiceLocator.get('stt_processor')
            if stt_processor and hasattr(stt_processor, 'resume_streaming'):
                stt_processor.resume_streaming()
                self.logger.debug('Transcription resumed')
        except Exception as e:
            self.logger.warning(f"Failed to resume transcription: {e}")


class MainWindow(QMainWindow, StateAwareComponent, EventListener):
    """
    Main application window for the Maggie AI Assistant.

    Serves as the primary visual interface for the Maggie AI system, integrating with
    the state machine architecture, event system, and other core components. The window
    displays the current system state, transcription results, event logs, error logs,
    and provides controls for interacting with the AI assistant and its extensions.

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
        """
        super(QMainWindow, self).__init__()
        self.maggie_ai = maggie_ai
        self.state_manager = maggie_ai.state_manager
        StateAwareComponent.__init__(self, self.state_manager)
        EventListener.__init__(self, maggie_ai.event_bus)
        self.logger = ComponentLogger('MainWindow')
        self.setWindowTitle('Maggie AI Assistant')
        self.setMinimumSize(900, 700)
        self.is_shutting_down = False
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel('Status: IDLE')
        self.status_label.setStyleSheet('font-weight: bold;')
        self.status_bar.addPermanentWidget(self.status_label)
        self._create_main_layout()
        self._register_event_handlers()
        self.setup_shortcuts()
        self.update_ui_for_state(State.IDLE)
        self.log_event('Maggie AI Assistant UI initialized...')

    def _register_event_handlers(self) -> None:
        """
        Register event handlers with the event system.

        Connects various system events to appropriate handler methods within
        the MainWindow.
        """
        events = [
            ('state_changed', self._on_state_changed, EventPriority.NORMAL),
            ('extension_completed', self._on_extension_completed, EventPriority.NORMAL),
            ('extension_error', self._on_extension_error, EventPriority.NORMAL),
            ('error_logged', self._on_error_logged, EventPriority.NORMAL),
            ('intermediate_transcription', self._on_intermediate_transcription, EventPriority.NORMAL),
            ('final_transcription', self._on_final_transcription, EventPriority.NORMAL),
            ('download_progress', self.show_download_progress, EventPriority.LOW)
        ]
        for event_type, handler, priority in events:
            self.listen(event_type, handler, priority=priority)
        self.logger.debug(f"Registered {len(events)} event handlers")

    def _create_main_layout(self) -> None:
        """
        Create the main window layout structure.

        Sets up the primary split view with left and right panels.
        """
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.content_splitter)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.content_splitter.addWidget(self.left_panel)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.content_splitter.addWidget(self.right_panel)
        self.content_splitter.setSizes([700, 200])
        self._create_log_sections()
        self._create_right_panel()
        self._create_control_panel()

    def _create_log_sections(self) -> None:
        """
        Create log display sections in the UI.

        Sets up the chat log, event log, and error log sections.
        """
        self.logs_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_layout.addWidget(self.logs_splitter)
        self._create_chat_section()
        self.event_group = QGroupBox('Event Log')
        self.event_group_layout = QVBoxLayout(self.event_group)
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_group_layout.addWidget(self.event_log)
        self.error_group = QGroupBox('Error Log')
        self.error_group_layout = QVBoxLayout(self.error_group)
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_group_layout.addWidget(self.error_log)
        self.logs_splitter.addWidget(self.chat_section)
        self.logs_splitter.addWidget(self.event_group)
        self.logs_splitter.addWidget(self.error_group)
        self.logs_splitter.setSizes([400, 150, 150])

    def _create_chat_section(self) -> None:
        """
        Create the chat interface section.

        Sets up the chat history display and input field.
        """
        self.chat_section = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_section)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_group = QGroupBox('Chat')
        self.chat_group_layout = QVBoxLayout(self.chat_group)
        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_group_layout.addWidget(self.chat_log)
        self.chat_layout.addWidget(self.chat_group)
        self.input_field = InputField(submit_callback=self._on_input_submitted)
        self.input_field.setFixedHeight(30)
        self.input_field.update_appearance_for_state(State.IDLE)
        self.input_field.state_change_requested.connect(self._on_input_state_change)
        self.chat_layout.addWidget(self.input_field)

    def _create_right_panel(self) -> None:
        """
        Create the right panel of the main UI.

        Sets up the state display and extensions section.
        """
        self.state_group = QGroupBox('Current State')
        self.state_layout = QVBoxLayout(self.state_group)
        self.state_display = QLabel('IDLE')
        self.state_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_display.setStyleSheet('font-size: 18px; font-weight: bold;')
        self.state_layout.addWidget(self.state_display)
        self.right_layout.addWidget(self.state_group)
        self.extensions_group = QGroupBox('Extensions')
        self.extensions_layout = QVBoxLayout(self.extensions_group)
        self._create_extension_buttons()
        self.right_layout.addWidget(self.extensions_group)
        self.right_layout.addStretch()

    def _create_control_panel(self) -> None:
        """
        Create the bottom control panel with system buttons.

        Sets up the main control buttons (shutdown, sleep).
        """
        self.control_panel = QWidget()
        self.control_layout = QHBoxLayout(self.control_panel)
        self.main_layout.addWidget(self.control_panel)
        self.shutdown_button = QPushButton('Shutdown')
        self.shutdown_button.clicked.connect(self.on_shutdown_clicked)
        self.control_layout.addWidget(self.shutdown_button)
        self.sleep_button = QPushButton('Sleep')
        self.sleep_button.clicked.connect(self.on_sleep_clicked)
        self.control_layout.addWidget(self.sleep_button)

    def _on_input_state_change(self, requested_state: State) -> None:
        """
        Handle state change requests from the input field.

        Parameters
        ----------
        requested_state : State
            The state that the input field is requesting
        """
        current_state = self.state_manager.get_current_state()
        if current_state == State.IDLE and requested_state == State.ACTIVE:
            self.state_manager.transition_to(State.READY, 'input_field_activated')
            self.log_event('State transition requested by input field')

    def _on_error_logged(self, error_data) -> None:
        """
        Handle error events from the system.

        Parameters
        ----------
        error_data : Dict[str, Any] or str
            Error information, either as a structured dictionary or simple string
        """
        if isinstance(error_data, dict):
            message = error_data.get('message', 'Unknown error')
            source = error_data.get('source', 'system')
            category = error_data.get('category', 'unknown')
            state_info = error_data.get('state', {})
            error_msg = f"[{source}] {message}"
            if state_info and isinstance(state_info, dict) and 'current_state' in state_info:
                state_name = state_info['current_state']
                error_msg += f" (State: {state_name})"
            self.log_error(error_msg)
            self._apply_state_specific_error_styling(error_data)
        else:
            self.log_error(str(error_data))

    def _apply_state_specific_error_styling(self, error_data: Dict[str, Any]) -> None:
        """
        Apply state-specific styling to errors in the UI.

        Parameters
        ----------
        error_data : Dict[str, Any]
            Error information dictionary containing state and style information
        """
        if 'state' in error_data and isinstance(error_data['state'], dict):
            state_info = error_data['state']
            if 'style' in state_info:
                style = state_info['style']
                border_color = style.get('background', '#FF0000')
                current_style = self.error_group.styleSheet()
                highlight_style = f"QGroupBox {{ border: 2px solid {border_color}; }}"
                self.error_group.setStyleSheet(highlight_style)
                QTimer.singleShot(2000, lambda: self.error_group.setStyleSheet(current_style))

    def setup_shortcuts(self) -> None:
        """
        Set up keyboard shortcuts for the application.
        """
        try:
            shortcut_config = {'sleep': 'Alt+S', 'shutdown': 'Alt+Q', 'focus_input': 'Alt+I'}
            sleep_shortcut = QShortcut(QKeySequence(shortcut_config['sleep']), self)
            sleep_shortcut.activated.connect(self.on_sleep_clicked)
            shutdown_shortcut = QShortcut(QKeySequence(shortcut_config['shutdown']), self)
            shutdown_shortcut.activated.connect(self.on_shutdown_clicked)
            input_shortcut = QShortcut(QKeySequence(shortcut_config['focus_input']), self)
            input_shortcut.activated.connect(lambda: self.input_field.setFocus())
            self.logger.debug('Keyboard shortcuts configured')
        except Exception as e:
            self.logger.error(f"Error setting up shortcuts: {e}")

    def _create_extension_buttons(self) -> None:
        """
        Create buttons for available extensions.
        """
        try:
            self._cleanup_extension_buttons()
            self.extension_buttons = {}
            for extension_name in self.maggie_ai.extensions:
                display_name = extension_name.replace('_', ' ').title()
                extension_button = QPushButton(display_name)
                extension_button.clicked.connect(lambda checked=False, name=extension_name: self.on_extension_clicked(name))
                self.extensions_layout.addWidget(extension_button)
                self.extension_buttons[extension_name] = extension_button
                if extension_name == 'recipe_creator':
                    try:
                        recipe_shortcut = QShortcut(QKeySequence('Alt+R'), self)
                        recipe_shortcut.activated.connect(lambda: self.on_extension_clicked('recipe_creator'))
                    except Exception as e:
                        self.logger.error(f"Error setting up recipe shortcut: {e}")
        except Exception as e:
            self.logger.error(f"Error creating extension buttons: {e}")

    def _on_state_changed(self, transition) -> None:
        """
        Handle system state change events.

        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
        """
        try:
            if not transition or not hasattr(transition, 'to_state') or not hasattr(transition, 'from_state'):
                self.logger.error('Invalid state transition object received')
                return
            to_state = getattr(transition, 'to_state', None)
            from_state = getattr(transition, 'from_state', None)
            if to_state is None or from_state is None:
                self.logger.error('Invalid state transition object: to_state or from_state is None')
                return
            to_state_name = to_state.name
            from_state_name = from_state.name
            trigger = getattr(transition, 'trigger', 'UNKNOWN')
            self.log_event(f"State changed: {from_state_name} -> {to_state_name} (trigger: {trigger})")
            self.update_ui_for_state(to_state)
            self.input_field.update_appearance_for_state(to_state)
            self._animate_transition(transition)
        except Exception as e:
            self.logger.error(f"Error handling state transition: {e}")

    def _animate_transition(self, transition: StateTransition) -> None:
        """
        Animate state transitions in the UI.

        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
        """
        try:
            anim_props = transition.get_animation_properties()
            animation = QPropertyAnimation(self.state_display, b'geometry')
            animation.setDuration(anim_props['duration'])
            animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            current_geometry = self.state_display.geometry()
            if anim_props['type'] == 'fade':
                self.state_display.setStyleSheet(f"{self.state_display.styleSheet()}; color: rgba(0, 0, 0, 100%);")
                animation = QPropertyAnimation(self.state_display, b'styleSheet')
                animation.setDuration(anim_props['duration'])
                animation.setStartValue(f"{self.state_display.styleSheet()}; color: rgba(0, 0, 0, 100%);")
                animation.setEndValue(f"{self.state_display.styleSheet()}; color: rgba(0, 0, 0, 0%);")
            elif anim_props['type'] == 'bounce':
                animation.setStartValue(current_geometry)
                bounce_geometry = QRect(current_geometry.x(), current_geometry.y() - 10,
                                      current_geometry.width(), current_geometry.height())
                animation.setEndValue(bounce_geometry)
                animation.setEasingCurve(QEasingCurve.Type.OutBounce)
            else:
                animation.setStartValue(QRect(current_geometry.x() + 50, current_geometry.y(),
                                            current_geometry.width(), current_geometry.height()))
                animation.setEndValue(current_geometry)
            animation.start()
        except Exception as e:
            self.logger.warning(f"Error animating transition: {e}")

    def _cleanup_extension_buttons(self) -> None:
        """
        Clean up extension buttons when refreshing or closing.
        """
        try:
            if hasattr(self, 'extension_buttons'):
                for button in self.extension_buttons.values():
                    self.extensions_layout.removeWidget(button)
                    button.deleteLater()
                self.extension_buttons.clear()
        except Exception as e:
            self.logger.error(f"Error cleaning up extension buttons: {e}")

    def update_ui_for_state(self, state: State) -> None:
        """
        Update the UI appearance based on system state.

        Parameters
        ----------
        state : State
            The current system state
        """
        style = state.get_style()
        bg_color = style.get('background', 'white')
        font_color = style.get('color', 'black')
        self.state_display.setText(state.display_name)
        self.status_label.setText(f"Status: {state.display_name}")
        self.state_display.setStyleSheet(
            f"font-size: 18px; font-weight: bold; background-color: {bg_color}; "
            f"color: {font_color}; padding: 5px; border-radius: 4px;"
        )
        self.input_field.update_appearance_for_state(state)
        self.logger.debug(f"UI updated for state: {state.name}")

    def refresh_extensions(self) -> None:
        """
        Refresh the extensions display.
        """
        self._create_extension_buttons()
        self.log_event('Extension list updated')

    def log_chat(self, message: str, is_user: bool = False) -> None:
        """
        Log a chat message in the chat history.

        Parameters
        ----------
        message : str
            The message text to log
        is_user : bool, optional
            Flag indicating if this is a user message (True) or assistant message (False)
        """
        timestamp = time.strftime('%H:%M:%S')
        prefix = 'user' if is_user else 'Maggie'
        color = 'blue' if is_user else 'green'
        self.chat_log.append(
            f'<span style="color:gray">[{timestamp}]</span> '
            f'<span style="color:{color}"><b>&lt; {prefix} &gt;</b></span> {message}'
        )

    def log_event(self, event: str) -> None:
        """
        Log a system event in the event log.

        Parameters
        ----------
        event : str
            The event description to log
        """
        timestamp = time.strftime('%H:%M:%S')
        self.event_log.append(f'<span style="color:gray">[{timestamp}]</span> {event}')
        self.logger.debug(f"Event logged: {event}")

    def show_download_progress(self, progress_data: Dict[str, Any]) -> None:
        """
        Display download progress in the status bar.

        Parameters
        ----------
        progress_data : Dict[str, Any]
            Dictionary containing progress information
        """
        item = progress_data.get('item', 'file')
        percent = progress_data.get('percent', 0)
        status = f"Downloading {item}: {percent}% complete"
        self.status_label.setText(status)
        if percent >= 100:
            QTimer.singleShot(3000, lambda: self.status_label.setText(f"Status: {self.state_display.text()}"))

    def log_error(self, error: str) -> None:
        """
        Log an error message in the error log.

        Parameters
        ----------
        error : str
            The error message to log
        """
        timestamp = time.strftime('%H:%M:%S')
        formatted_error = (
            f'<span style="color:gray">[{timestamp}]</span> '
            f'<span style="color:red"><b>ERROR:</b></span> {error}'
        )
        self.error_log.append(formatted_error)
        current_sizes = self.logs_splitter.sizes()
        if current_sizes[2] < 100:
            self.logs_splitter.setSizes([current_sizes[0], current_sizes[1], 200])
        self.logger.error(f"Error logged in GUI: {error}")

    def on_shutdown_clicked(self) -> None:
        """
        Handle shutdown button clicks.
        """
        self.log_event('Shutdown requested')
        self.maggie_ai.shutdown()
        self.logger.info('Shutdown initiated from GUI')

    def on_sleep_clicked(self) -> None:
        """
        Handle sleep button clicks.
        """
        self.log_event('Sleep requested')
        self.maggie_ai.timeout()
        self.logger.info('Sleep initiated from GUI')

    def on_extension_clicked(self, extension_name: str) -> None:
        """
        Handle extension button clicks.

        Parameters
        ----------
        extension_name : str
            Name of the extension to activate
        """
        self.log_event(f"Extension requested: {extension_name}")
        if extension_name in self.maggie_ai.extensions:
            extension = self.maggie_ai.extensions[extension_name]
            self.maggie_ai.process_command(extension=extension)
            self.logger.info(f"Extension '{extension_name}' activated from GUI")

    def _on_input_submitted(self, text: str) -> None:
        """
        Handle text submitted from the input field.

        Parameters
        ----------
        text : str
            The text submitted by the user
        """
        if not text.strip():
            return
        self.log_chat(text, is_user=True)
        current_state = self.state_manager.get_current_state()
        if current_state == State.IDLE:
            self.state_manager.transition_to(State.READY, 'user_input')
        self.maggie_ai.event_bus.publish('command_detected', text)
        self.logger.debug(f"User input submitted: {text}")

    def closeEvent(self, event) -> None:
        """
        Handle window close events.

        Parameters
        ----------
        event : QCloseEvent
            The Qt close event object
        """
        self.log_event('Window close requested, shutting down')
        self.is_shutting_down = True
        self.maggie_ai.shutdown()
        QTimer.singleShot(2000, lambda: event.accept())
        self.logger.info('GUI window closed, shutdown initiated')

    @Slot(str)
    def update_intermediate_text(self, text: str) -> None:
        """
        Update intermediate transcription text.

        Parameters
        ----------
        text : str
            The intermediate transcription text
        """
        self.input_field.update_intermediate_text(text)

    @Slot(str)
    def update_final_text(self, text: str) -> None:
        """
        Update final transcription text.

        Parameters
        ----------
        text : str
            The final transcription text
        """
        self.input_field.set_final_text(text)

    def _on_intermediate_transcription(self, text: str) -> None:
        """
        Handle intermediate transcription events.

        Parameters
        ----------
        text : str
            The intermediate transcription text
        """
        if QThread.currentThread() == self.thread():
            self.update_intermediate_text(text)
        else:
            QMetaObject.invokeMethod(self, 'update_intermediate_text',
                                   Qt.ConnectionType.QueuedConnection,
                                   Q_ARG(str, text))

    def _on_final_transcription(self, text: str) -> None:
        """
        Handle final transcription events.

        Parameters
        ----------
        text : str
            The final transcription text
        """
        if QThread.currentThread() == self.thread():
            self.update_final_text(text)
        else:
            QMetaObject.invokeMethod(self, 'update_final_text',
                                   Qt.ConnectionType.QueuedConnection,
                                   Q_ARG(str, text))

    def _on_extension_completed(self, extension_name: str) -> None:
        """
        Handle extension completion events.

        Parameters
        ----------
        extension_name : str
            Name of the extension that completed its task
        """
        self.log_event(f"Extension completed: {extension_name}")
        self.update_ui_for_state(State.READY)

    def _on_extension_error(self, extension_name: str) -> None:
        """
        Handle extension error events.

        Parameters
        ----------
        extension_name : str
            Name of the extension that encountered an error
        """
        self.log_error(f"Error in extension: {extension_name}")
        self.update_ui_for_state(State.READY)

    def on_enter_idle(self, transition: StateTransition) -> None:
        """
        Handle transitions to IDLE state.

        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
        """
        self.update_ui_for_state(State.IDLE)

    def on_enter_ready(self, transition: StateTransition) -> None:
        """
        Handle transitions to READY state.

        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
        """
        self.update_ui_for_state(State.READY)

    def on_enter_active(self, transition: StateTransition) -> None:
        """
        Handle transitions to ACTIVE state.

        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
        """
        self.update_ui_for_state(State.ACTIVE)

    def on_enter_busy(self, transition: StateTransition) -> None:
        """
        Handle transitions to BUSY state.

        Parameters
        ----------
        transition : StateTransition
            Object containing details about the state transition
        """
        self.update_ui_for_state(State.BUSY)