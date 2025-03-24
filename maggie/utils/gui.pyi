from enum import Enum
from typing import Any, Dict, Optional, List, Callable
from PySide6.QtWidgets import QMainWindow, QWidget, QLineEdit
from PySide6.QtCore import Signal, QThread

class State(Enum):
    """
    Enumeration of possible states for the Maggie AI Assistant UI.

    Attributes
    ----------
    IDLE : State
        The assistant is in an idle state, waiting for activation.
    STARTING : State
        The assistant is in the process of starting up.
    RUNNING : State
        The assistant is actively running and processing commands.
    PAUSED : State
        The assistant's operations are temporarily suspended.
    STOPPED : State
        The assistant has been stopped.
    ERROR : State
        The assistant has encountered an error state.

    Notes
    -----
    This enum represents the different operational states of the Maggie AI Assistant 
    user interface, providing a clear state management mechanism.
    """
    IDLE: State
    STARTING: State
    RUNNING: State
    PAUSED: State
    STOPPED: State
    ERROR: State

class InputField(QLineEdit):
    """
    Specialized input field for the Maggie AI Assistant with enhanced functionality.

    This custom input field supports both text and speech input modes, 
    with special handling for intermediate transcription and state management.

    Attributes
    ----------
    state_change_requested : Signal
        A Qt signal that emits when a state change is requested.
    stt_mode : bool
        Indicates whether the input field is in speech-to-text (STT) mode.
    submit_callback : Optional[Callable]
        A callback function to be called when input is submitted.
    intermediate_text : str
        Stores intermediate transcription text.

    Methods
    -------
    focusInEvent(event: QFocusEvent) -> None
        Handles focus in event, switching to active text input mode.
    focusOutEvent(event: QFocusEvent) -> None
        Handles focus out event, returning to STT mode.
    on_return_pressed() -> None
        Handles input submission when return key is pressed.
    update_appearance_for_state(state: str) -> None
        Updates the input field's appearance based on the current state.
    update_intermediate_text(text: str) -> None
        Updates the input field with intermediate transcription text.
    set_final_text(text: str) -> None
        Sets the final transcribed text in the input field.

    Notes
    -----
    This custom input field is designed to provide a seamless interface 
    between text and speech input in the Maggie AI Assistant.
    """
    state_change_requested: Signal
    stt_mode: bool
    submit_callback: Optional[Callable]
    intermediate_text: str

    def focusInEvent(self, event: Any) -> None:
        """
        Handle focus in event for the input field.

        Switches to active text input mode, changes state, and pauses transcription.

        Parameters
        ----------
        event : Any
            The focus event object from Qt.
        """
        ...

    def focusOutEvent(self, event: Any) -> None:
        """
        Handle focus out event for the input field.

        Returns to speech-to-text mode and resumes transcription if no text is present.

        Parameters
        ----------
        event : Any
            The focus out event object from Qt.
        """
        ...

    def on_return_pressed(self) -> None:
        """
        Handle return key press event.

        Submits the current text input and clears the field.
        """
        ...

    def update_appearance_for_state(self, state: str) -> None:
        """
        Update the input field's visual appearance based on the current state.

        Parameters
        ----------
        state : str
            The current state of the assistant (e.g., 'IDLE', 'ACTIVE').
        """
        ...

    def update_intermediate_text(self, text: str) -> None:
        """
        Update the input field with intermediate transcription text.

        Parameters
        ----------
        text : str
            The intermediate transcription text to display.
        """
        ...

    def set_final_text(self, text: str) -> None:
        """
        Set the final transcribed text in the input field.

        Optionally auto-submits the text based on configuration.

        Parameters
        ----------
        text : str
            The final transcribed text to set.
        """
        ...

class MainWindow(QMainWindow):
    """
    Main user interface window for the Maggie AI Assistant.

    Provides a comprehensive GUI for interacting with the AI assistant, 
    including chat logs, event logs, error logs, and control buttons.

    Attributes
    ----------
    maggie_ai : Any
        Reference to the main Maggie AI assistant instance.
    is_shutting_down : bool
        Indicates whether the application is in the process of shutting down.
    central_widget : QWidget
        The central widget of the main window.
    input_field : InputField
        Custom input field for text and speech input.
    chat_log : QTextEdit
        Text area for displaying chat messages.
    event_log : QTextEdit
        Text area for displaying system events.
    error_log : QTextEdit
        Text area for displaying error messages.

    Methods
    -------
    _create_main_layout() -> None
        Create the main layout of the window.
    _create_log_sections() -> None
        Create log sections for chat, events, and errors.
    _create_right_panel() -> None
        Create the right panel with state display and extension buttons.
    _create_control_panel() -> None
        Create control panel with shutdown and sleep buttons.
    log_chat(message: str, is_user: bool = False) -> None
        Log a chat message.
    log_event(event: str) -> None
        Log a system event.
    log_error(error: str) -> None
        Log an error message.
    safe_update_gui(func: Callable, *args, **kwargs) -> None
        Safely update the GUI from different threads.

    Notes
    -----
    This class is the primary user interface for the Maggie AI Assistant, 
    providing a comprehensive and interactive GUI for user interaction.

    See Also
    --------
    PySide6 documentation : https://doc.qt.io/qtforpython-6/
    Maggie AI Assistant documentation : (project documentation link)
    """
    maggie_ai: Any
    is_shutting_down: bool
    central_widget: QWidget
    input_field: InputField

    def _create_main_layout(self) -> None:
        """
        Create the main layout of the application window.

        Configures the overall structure of the user interface, 
        including splitters, panels, and log sections.
        """
        ...

    def _create_log_sections(self) -> None:
        """
        Create log sections for chat, events, and errors.

        Sets up text areas for displaying different types of logs 
        with appropriate grouping and styling.
        """
        ...

    def _create_right_panel(self) -> None:
        """
        Create the right panel of the user interface.

        Configures the state display and extension buttons section.
        """
        ...

    def _create_control_panel(self) -> None:
        """
        Create the control panel with shutdown and sleep buttons.

        Provides user controls for managing the assistant's overall state.
        """
        ...

    def log_chat(self, message: str, is_user: bool = False) -> None:
        """
        Log a chat message to the chat log area.

        Parameters
        ----------
        message : str
            The chat message to log.
        is_user : bool, optional
            Flag indicating whether the message is from the user (default is False).
        """
        ...

    def log_event(self, event: str) -> None:
        """
        Log a system event to the event log area.

        Parameters
        ----------
        event : str
            The system event to log.
        """
        ...

    def log_error(self, error: str) -> None:
        """
        Log an error message to the error log area.

        Parameters
        ----------
        error : str
            The error message to log.
        """
        ...

    def safe_update_gui(self, func: Callable, *args, **kwargs) -> None:
        """
        Safely update the GUI from different threads.

        Ensures thread-safe GUI updates by using Qt's invokeMethod mechanism.

        Parameters
        ----------
        func : Callable
            The GUI update function to call.
        *args : Any
            Positional arguments for the update function.
        **kwargs : Any
            Keyword arguments for the update function.

        Notes
        -----
        This method is crucial for preventing threading-related GUI update issues.
        """
        ...

    def update_state(self, state: State) -> None:
        """
        Update the current state of the user interface.

        Parameters
        ----------
        state : State
            The new state of the assistant.

        Notes
        -----
        Updates visual indicators and logs to reflect the current assistant state.
        """
        ...

    def setup_shortcuts(self) -> None:
        """
        Configure keyboard shortcuts for the application.

        Sets up quick access shortcuts for common actions like 
        sleep, shutdown, and input focus.
        """
        ...

    def refresh_extensions(self) -> None:
        """
        Refresh the list of available extensions in the UI.

        Rebuilds the extension buttons based on current available extensions.
        """
        ...