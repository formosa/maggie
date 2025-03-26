import sys
import traceback
import logging
import time
import enum
import uuid
import functools
import threading
from typing import Any, Callable, Optional, TypeVar, Dict, Union, List, Tuple, cast, Type
from functools import wraps
T = TypeVar('T')
logger = logging.getLogger(__name__)


class ErrorSeverity(enum.Enum):
	"""
	ErrorSeverity is an enumeration that represents different levels of error severity.

	Attributes:
		DEBUG (int): Represents debug-level messages, typically used for development and troubleshooting.
		INFO (int): Represents informational messages that highlight the progress of the application.
		WARNING (int): Represents potentially harmful situations that do not prevent the program from running.
		ERROR (int): Represents error events that might still allow the application to continue running.
		CRITICAL (int): Represents critical error events that typically lead to application termination.
	"""
	DEBUG = 0
	INFO = 1
	WARNING = 2
	ERROR = 3
	CRITICAL = 4


class ErrorCategory(enum.Enum):
	"""
	ErrorCategory is an enumeration that categorizes different types of errors
	that may occur in the system. Each category represents a specific type of
	error, aiding in error handling and debugging.

	Attributes:
		SYSTEM: Errors related to system-level issues.
		NETWORK: Errors caused by network-related problems.
		RESOURCE: Errors due to resource constraints or unavailability.
		PERMISSION: Errors arising from insufficient permissions.
		CONFIGURATION: Errors caused by incorrect or missing configuration.
		INPUT: Errors related to invalid or unexpected input.
		PROCESSING: Errors occurring during data or task processing.
		MODEL: Errors specific to model-related operations.
		EXTENSION: Errors caused by issues in extensions or plugins.
		STATE: Errors due to invalid or unexpected system state.
		UNKNOWN: Errors that do not fit into any specific category.
	"""
	SYSTEM = 'system'
	NETWORK = 'network'
	RESOURCE = 'resource'
	PERMISSION = 'permission'
	CONFIGURATION = 'configuration'
	INPUT = 'input'
	PROCESSING = 'processing'
	MODEL = 'model'
	EXTENSION = 'extension'
	STATE = 'state'
	UNKNOWN = 'unknown'


ERROR_EVENT_LOGGED = 'error_logged'
ERROR_EVENT_COMPONENT_FAILURE = 'component_failure'
ERROR_EVENT_RESOURCE_WARNING = 'resource_warning'
ERROR_EVENT_SYSTEM_ERROR = 'system_error'
ERROR_EVENT_STATE_TRANSITION = 'state_transition_error'
ERROR_EVENT_RESOURCE_MANAGEMENT = 'resource_management_error'
ERROR_EVENT_INPUT_PROCESSING = 'input_processing_error'


class LLMError(Exception):
	"""
	LLMError is a custom exception class used to represent errors specific to
	Language Model (LLM) operations. This exception can be raised to indicate
	issues or unexpected behavior encountered during the execution of LLM-related
	tasks.

	Attributes:
		None
	"""
	pass


class ModelLoadError(LLMError):
	"""
	Exception raised when there is an error loading a model.

	This error is a subclass of `LLMError` and is intended to be used
	to indicate issues specifically related to loading machine learning
	models, such as missing files, corrupted data, or incompatible formats.
	"""
	pass


class GenerationError(LLMError):
	"""
	Exception raised for errors that occur during the generation process.

	This exception is a subclass of `LLMError` and is used to indicate
	issues specifically related to the generation functionality.

	Attributes:
		Inherits all attributes from the `LLMError` base class.
	"""
	pass


class STTError(Exception):
	"""
	Custom exception class for handling errors related to Speech-to-Text (STT) operations.

	This exception can be raised to indicate specific issues encountered during
	the processing or handling of STT functionality.

	Usage:
		Raise this exception when an error specific to STT occurs, providing
		a clear and descriptive message about the issue.

	Example:
		raise STTError("Failed to process audio input for STT.")
	"""
	pass


class TTSError(Exception):
	"""
	Custom exception class for handling errors related to Text-to-Speech (TTS) operations.

	This exception can be raised to indicate specific issues or failures encountered
	during TTS processing.

	Usage:
		raise TTSError("An error occurred in the TTS module.")
	"""
	pass


class ExtensionError(Exception):
	"""
	Custom exception raised for errors related to invalid or unsupported extensions.

	This exception can be used to signal issues when handling file extensions
	or other extension-related operations in the application.
	"""
	pass


class StateTransitionError(Exception):
	"""
	Exception raised for errors in state transitions.

	Attributes:
		message (str): The error message describing the state transition issue.
		from_state (Any, optional): The state from which the transition was attempted. Defaults to None.
		to_state (Any, optional): The state to which the transition was attempted. Defaults to None.
		trigger (str, optional): The trigger or event that caused the state transition. Defaults to None.
		details (Dict[str, Any], optional): Additional details about the error. Defaults to an empty dictionary.

	Args:
		message (str): The error message describing the state transition issue.
		from_state (Any, optional): The state from which the transition was attempted. Defaults to None.
		to_state (Any, optional): The state to which the transition was attempted. Defaults to None.
		trigger (str, optional): The trigger or event that caused the state transition. Defaults to None.
		details (Dict[str, Any], optional): Additional details about the error. Defaults to an empty dictionary.
	"""

	def __init__(self, message: str, from_state: Any = None, to_state: Any = None, trigger: str = None,
                 details: Dict[str, Any] = None): self.from_state = from_state; self.to_state = to_state; self.trigger = trigger; self.details = details or {}; super().__init__(message)


class ResourceManagementError(Exception):
	"""
	Exception raised for errors related to resource management.

	Attributes:
		message (str): A description of the error.
		resource_type (str, optional): The type of the resource involved in the error.
		resource_name (str, optional): The name of the resource involved in the error.
		details (Dict[str, Any], optional): Additional details about the error.

	Args:
		message (str): A description of the error.
		resource_type (str, optional): The type of the resource involved in the error.
		resource_name (str, optional): The name of the resource involved in the error.
		details (Dict[str, Any], optional): Additional details about the error.
	"""

	def __init__(self, message: str, resource_type: str = None, resource_name: str = None,
                 details: Dict[str, Any] = None): self.resource_type = resource_type; self.resource_name = resource_name; self.details = details or {}; super().__init__(message)


class InputProcessingError(Exception):
	"""
	Exception raised for errors encountered during input processing.

	Attributes:
		message (str): A description of the error.
		input_type (str, optional): The type of input that caused the error. Defaults to None.
		input_source (str, optional): The source of the input that caused the error. Defaults to None.
		details (Dict[str, Any], optional): Additional details about the error. Defaults to an empty dictionary.

	Args:
		message (str): A description of the error.
		input_type (str, optional): The type of input that caused the error. Defaults to None.
		input_source (str, optional): The source of the input that caused the error. Defaults to None.
		details (Dict[str, Any], optional): Additional details about the error. Defaults to an empty dictionary.
	"""
	def __init__(self, message: str, input_type: str = None, input_source: str = None,
                 details: Dict[str, Any] = None): self.input_type = input_type; self.input_source = input_source; self.details = details or {}; super().__init__(message)


class ErrorContext:
	"""
	ErrorContext is a utility class for capturing and managing error-related information in a structured way. 
	It provides methods to log errors, convert error details to a dictionary, and add contextual information 
	such as state, transitions, and resource details.
	Attributes:
		message (str): A descriptive error message.
		exception (Optional[Exception]): The exception instance associated with the error, if any.
		category (ErrorCategory): The category of the error (e.g., UNKNOWN, VALIDATION, etc.).
		severity (ErrorSeverity): The severity level of the error (e.g., ERROR, WARNING, CRITICAL).
		source (str): The source or origin of the error.
		details (Dict[str, Any]): Additional details or metadata about the error.
		correlation_id (Optional[str]): A unique identifier for correlating related errors.
		timestamp (float): The time when the error context was created.
		state_info (Optional[Dict[str, Any]]): Information about the current state or context.
		exception_type (str): The type of the exception (if an exception is provided).
		exception_msg (str): The message of the exception (if an exception is provided).
		filename (str): The name of the file where the error occurred (if traceback is available).
		line (int): The line number in the file where the error occurred (if traceback is available).
		function (str): The function name where the error occurred (if traceback is available).
		code (str): The code snippet where the error occurred (if traceback is available).
	Methods:
		__init__(message, exception, category, severity, source, details, correlation_id, state_info):
			Initializes an instance of ErrorContext with the provided attributes.
		to_dict() -> Dict[str, Any]:
			Converts the error context into a dictionary representation.
		log(publish: bool = True) -> None:
			Logs the error based on its severity and optionally publishes it to an event bus.
		add_state_info(state_object: Any) -> None:
			Adds information about the current state to the error context.
		add_transition_info(from_state: Any, to_state: Any, trigger: str) -> None:
			Adds information about a state transition to the error context.
		add_resource_info(resource_info: Dict[str, Any]) -> None:
			Adds resource-related information to the error context.
	"""
	def __init__(self, message: str, exception: Optional[Exception] = None, category: ErrorCategory = ErrorCategory.UNKNOWN, severity: ErrorSeverity = ErrorSeverity.ERROR, source: str = '', details: Dict[str, Any] = None, correlation_id: Optional[str] = None, state_info: Optional[Dict[str, Any]] = None):
		self.message = message
		self.exception = exception
		self.category = category
		self.severity = severity
		self.source = source
		self.details = details or {}
		self.correlation_id = correlation_id or str(uuid.uuid4())
		self.timestamp = time.time()
		self.state_info = state_info or {}
		if exception:
			self.exception_type = type(exception).__name__
			self.exception_msg = str(exception)
			exc_type, exc_value, exc_traceback = sys.exc_info()
			if exc_traceback:
				tb = traceback.extract_tb(exc_traceback)
				if tb:
					frame = tb[-1]
					self.filename = frame.filename
					self.line = frame.lineno
					self.function = frame.name
					self.code = frame.line

	def to_dict(self) -> Dict[str, Any]:
		"""
		Converts the object into a dictionary representation.

		Returns:
			Dict[str, Any]: A dictionary containing the object's attributes:
				- 'message' (str): The message associated with the object.
				- 'category' (str): The category of the object, represented by its value.
				- 'severity' (str): The severity level of the object, represented by its value.
				- 'source' (str): The source of the object.
				- 'timestamp' (Any): The timestamp associated with the object.
				- 'correlation_id' (Any): The correlation ID for tracking purposes.
				- 'exception' (dict, optional): If the object has an `exception_type` attribute, includes:
					- 'type' (str): The type of the exception.
					- 'message' (str): The message of the exception.
				- 'location' (dict, optional): If the object has a `filename` attribute, includes:
					- 'file' (str): The filename where the error occurred.
					- 'line' (int): The line number where the error occurred.
					- 'function' (str): The function name where the error occurred.
					- 'code' (str): The code snippet related to the error.
				- 'details' (Any, optional): Additional details if available.
				- 'state' (Any, optional): State information if available.
		"""
		result = {'message': self.message, 'category': self.category.value, 'severity': self.severity.value,
					'source': self.source, 'timestamp': self.timestamp, 'correlation_id': self.correlation_id}
		if hasattr(self, 'exception_type'):
			result['exception'] = {
				'type': self.exception_type, 'message': self.exception_msg}
		if hasattr(self, 'filename'):
			result['location'] = {
				'file': self.filename, 'line': self.line, 'function': self.function, 'code': self.code}
		if self.details:
			result['details'] = self.details
		if self.state_info:
			result['state'] = self.state_info
		return result

	def log(self, publish: bool = True) -> None:
		"""
		Logs the error message based on its severity and optionally publishes the error event.

		The method logs the error message using the appropriate logging level based on the
		severity of the error. If the `publish` parameter is set to `True`, it also attempts
		to publish the error event to an event bus.

		Args:
			publish (bool): Indicates whether to publish the error event to the event bus.
							Defaults to True.

		Logging Behavior:
			- Logs with `logger.critical` if the severity is `ErrorSeverity.CRITICAL`.
			- Logs with `logger.error` if the severity is `ErrorSeverity.ERROR`.
			- Logs with `logger.warning` if the severity is `ErrorSeverity.WARNING`.
			- Logs with `logger.debug` for all other severities.

		Publishing Behavior:
			- If `publish` is True, attempts to publish the error event to the event bus.
			- If publishing fails, logs an error indicating the failure.

		Raises:
			None: Any exceptions during publishing are caught and logged.
		"""
		if self.severity == ErrorSeverity.CRITICAL:
			logger.critical(self.message, exc_info=bool(self.exception))
		elif self.severity == ErrorSeverity.ERROR:
			logger.error(self.message, exc_info=bool(self.exception))
		elif self.severity == ErrorSeverity.WARNING:
			logger.warning(self.message)
		else:
			logger.debug(self.message)
		if publish:
			try:
				event_bus = get_event_bus()
				if event_bus:
					event_bus.publish('error_logged', self.to_dict())
			except Exception as e:
				logger.error(f"Failed to publish error event: {e}")

	def add_state_info(self, state_object: Any) -> None:
		"""
		Adds state information to the `state_info` dictionary.

		This method extracts the name and style (if available) from the given
		state object and updates the `state_info` dictionary with the current
		state's name and style.

		Args:
			state_object (Any): The state object to extract information from.
				- If the object has a `name` attribute, it will be used as the
				  state's name. Otherwise, the string representation of the object
				  will be used.
				- If the object has a `get_style` method, its return value will
				  be used as the state's style.

		Returns:
			None
		"""
		if state_object is not None:
			state_name = state_object.name if hasattr(
				state_object, 'name')else str(state_object)
			self.state_info['current_state'] = state_name
			if hasattr(state_object, 'get_style'):
				self.state_info['style'] = state_object.get_style()

	def add_transition_info(self, from_state: Any, to_state: Any, trigger: str) -> None:
		"""
		Adds transition information to the state_info dictionary.

		This method records details about a state transition, including the 
		originating state, the destination state, and the trigger that caused 
		the transition.

		Args:
			from_state (Any): The state from which the transition originates. 
							  It should have a 'name' attribute, or its string 
							  representation will be used.
			to_state (Any): The state to which the transition leads. It should 
							have a 'name' attribute, or its string representation 
							will be used.
			trigger (str): The event or action that triggered the state transition.

		Returns:
			None
		"""
		if from_state is not None and to_state is not None:
			from_name = from_state.name if hasattr(
				from_state, 'name')else str(from_state)
			to_name = to_state.name if hasattr(
				to_state, 'name')else str(to_state)
			self.state_info['transition'] = {
				'from': from_name, 'to': to_name, 'trigger': trigger}

	def add_resource_info(self, resource_info: Dict[str, Any]) -> None:
		"""
		Adds resource information to the details dictionary.

		Args:
			resource_info (Dict[str, Any]): A dictionary containing resource information
				to be added. If the dictionary is not empty, it will be stored under
				the 'resources' key in the details attribute.
		"""
		if resource_info:
			self.details['resources'] = resource_info


class ErrorRegistry:
	"""
	ErrorRegistry is a singleton class that manages the registration and creation of error contexts
	for a system. It provides a centralized mechanism to define, register, and retrieve error details
	based on error codes.
	Attributes:
		_instance (ErrorRegistry): The singleton instance of the ErrorRegistry.
		_lock (threading.RLock): A reentrant lock to ensure thread-safe operations.
		errors (dict): A dictionary mapping error codes to their definitions, including message templates,
			categories, and severities.
	Methods:
		get_instance() -> ErrorRegistry:
			Retrieves the singleton instance of the ErrorRegistry, creating it if necessary.
		__init__():
			Initializes the ErrorRegistry and registers a predefined set of error codes.
		register_error(code: str, message_template: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
					   severity: ErrorSeverity = ErrorSeverity.ERROR) -> None:
			Registers a new error code with its associated message template, category, and severity.
		create_error(code: str, details: Dict[str, Any] = None, exception: Optional[Exception] = None,
					 source: str = '', severity_override: Optional[ErrorSeverity] = None,
					 state_object: Any = None, from_state: Any = None, to_state: Any = None,
					 trigger: str = None) -> ErrorContext:
			Creates an ErrorContext object for a given error code, populating it with details,
			exception information, and optional state transition data.
	"""
	_instance = None
	_lock = threading.RLock()

	@classmethod
	def get_instance(cls) -> 'ErrorRegistry':
		"""
		Retrieve the singleton instance of the ErrorRegistry class.

		This method ensures that only one instance of the ErrorRegistry class
		is created (singleton pattern). If the instance does not already exist,
		it initializes it in a thread-safe manner.

		Returns:
			ErrorRegistry: The singleton instance of the ErrorRegistry class.
		"""
		if cls._instance is None:
			with cls._lock:
				if cls._instance is None:
					cls._instance = ErrorRegistry()
		return cls._instance

	def __init__(self): self.errors = {}; self.register_error('CONFIG_LOAD_ERROR', 'Failed to load configuration: {details}', ErrorCategory.CONFIGURATION, ErrorSeverity.ERROR); self.register_error('RESOURCE_UNAVAILABLE', 'Required resource unavailable: {resource}', ErrorCategory.RESOURCE, ErrorSeverity.ERROR); self.register_error('MODEL_LOAD_ERROR', 'Failed to load model: {model_name}', ErrorCategory.MODEL, ErrorSeverity.ERROR); self.register_error('EXTENSION_INIT_ERROR', 'Failed to initialize extension: {extension_name}', ErrorCategory.EXTENSION, ErrorSeverity.ERROR); self.register_error('NETWORK_ERROR', 'Network error occurred: {details}', ErrorCategory.NETWORK, ErrorSeverity.ERROR); self.register_error('STATE_TRANSITION_ERROR', 'Invalid state transition: {from_state} -> {to_state}', ErrorCategory.STATE, ErrorSeverity.ERROR); self.register_error('STATE_HANDLER_ERROR', 'Error in state handler for {state}: {details}', ErrorCategory.STATE, ErrorSeverity.ERROR); self.register_error('STATE_EVENT_ERROR', 'Failed to process state event: {event_type}', ErrorCategory.STATE, ErrorSeverity.ERROR); self.register_error(
		"""
		Initializes the error handling system and registers predefined error types.

		This method sets up a dictionary to store error definitions and registers
		a variety of common error types with their corresponding messages, categories,
		and severity levels. Each error type is associated with a unique identifier
		and a formatted message template.

		Predefined error types include:
		- CONFIG_LOAD_ERROR: Errors related to configuration loading.
		- RESOURCE_UNAVAILABLE: Errors when required resources are unavailable.
		- MODEL_LOAD_ERROR: Errors during model loading.
		- EXTENSION_INIT_ERROR: Errors initializing extensions.
		- NETWORK_ERROR: Errors related to network issues.
		- STATE_TRANSITION_ERROR: Errors in state transitions.
		- STATE_HANDLER_ERROR: Errors in state handlers.
		- STATE_EVENT_ERROR: Errors processing state events.
		- RESOURCE_ALLOCATION_ERROR: Errors allocating resources.
		- RESOURCE_DEALLOCATION_ERROR: Errors deallocating resources.
		- RESOURCE_LIMIT_EXCEEDED: Warnings for exceeding resource limits.
		- INPUT_VALIDATION_ERROR: Errors in input validation.
		- INPUT_PROCESSING_ERROR: Errors processing input.
		- INPUT_FIELD_ERROR: Errors in specific input fields.
		- SPEECH_RECOGNITION_ERROR: Errors in speech recognition.
		- TEXT_GENERATION_ERROR: Errors in text generation.

		The error definitions include placeholders for dynamic details that can be
		filled in when raising or logging the errors.
		"""
		'RESOURCE_ALLOCATION_ERROR', 'Failed to allocate {resource_type} resource: {details}', ErrorCategory.RESOURCE, ErrorSeverity.ERROR); self.register_error('RESOURCE_DEALLOCATION_ERROR', 'Failed to deallocate {resource_type} resource: {details}', ErrorCategory.RESOURCE, ErrorSeverity.ERROR); self.register_error('RESOURCE_LIMIT_EXCEEDED', '{resource_type} limit exceeded: {details}', ErrorCategory.RESOURCE, ErrorSeverity.WARNING); self.register_error('INPUT_VALIDATION_ERROR', 'Input validation failed: {details}', ErrorCategory.INPUT, ErrorSeverity.ERROR); self.register_error('INPUT_PROCESSING_ERROR', 'Failed to process {input_type} input: {details}', ErrorCategory.INPUT, ErrorSeverity.ERROR); self.register_error('INPUT_FIELD_ERROR', 'Error in input field: {field_name}', ErrorCategory.INPUT, ErrorSeverity.ERROR); self.register_error('SPEECH_RECOGNITION_ERROR', 'Speech recognition failed: {details}', ErrorCategory.PROCESSING, ErrorSeverity.ERROR); self.register_error('TEXT_GENERATION_ERROR', 'Text generation failed: {details}', ErrorCategory.PROCESSING, ErrorSeverity.ERROR)

	def register_error(self, code: str, message_template: str, category: ErrorCategory = ErrorCategory.UNKNOWN, severity: ErrorSeverity = ErrorSeverity.ERROR) -> None:
		"""
		Registers a new error with the specified code, message template, category, and severity.

		Args:
			code (str): A unique identifier for the error.
			message_template (str): A template string for the error message.
			category (ErrorCategory, optional): The category of the error. Defaults to ErrorCategory.UNKNOWN.
			severity (ErrorSeverity, optional): The severity level of the error. Defaults to ErrorSeverity.ERROR.

		Raises:
			None

		Notes:
			If an error with the same code is already registered, it will be overwritten, and a warning will be logged.
		"""
		with self._lock:
			if code in self.errors:
				logger.warning(
					f"Error code '{code}' already registered, overwriting")
			self.errors[code] = {
				'message_template': message_template, 'category': category, 'severity': severity}

	def create_error(self, code: str, details: Dict[str, Any] = None, exception: Optional[Exception] = None, source: str = '', severity_override: Optional[ErrorSeverity] = None, state_object: Any = None, from_state: Any = None, to_state: Any = None, trigger: str = None) -> ErrorContext:
		"""
		Creates an error context object based on the provided error code and additional details.

		Args:
			code (str): The error code identifying the type of error.
			details (Dict[str, Any], optional): Additional details to format the error message. Defaults to None.
			exception (Optional[Exception], optional): An exception instance associated with the error. Defaults to None.
			source (str, optional): The source or origin of the error. Defaults to an empty string.
			severity_override (Optional[ErrorSeverity], optional): Overrides the default severity of the error. Defaults to None.
			state_object (Any, optional): An object representing the current state. Defaults to None.
			from_state (Any, optional): The initial state in a state transition. Defaults to None.
			to_state (Any, optional): The target state in a state transition. Defaults to None.
			trigger (str, optional): The trigger causing the state transition. Defaults to None.

		Returns:
			ErrorContext: An object encapsulating the error details, including message, category, severity, and additional context.

		Notes:
			- If the provided error code is not found in the `errors` dictionary, a generic error context is returned.
			- If `details` are provided, they are used to format the error message. Missing keys in `details` will be logged as warnings.
			- State transition information is added to the context if both `from_state` and `to_state` are provided.
		"""
		with self._lock:
			if code not in self.errors:
				logger.warning(
					f"Unknown error code: {code}, using generic error")
				return ErrorContext(message=f"Unknown error: {code}", exception=exception, source=source, details=details)
			error_def = self.errors[code]
			message = error_def['message_template']
			if details:
				try:
					message = message.format(**details)
				except KeyError as e:
					logger.warning(f"Missing key in error details: {e}")
			context = ErrorContext(message=message, exception=exception,
									category=error_def['category'], severity=severity_override or error_def['severity'], source=source, details=details)
			if state_object is not None:
				context.add_state_info(state_object)
			if from_state is not None and to_state is not None:
				context.add_transition_info(from_state, to_state, trigger)
			return context


def get_event_bus() -> Optional[Any]:
	"""
	Retrieves the event bus instance using the ServiceLocator.

	This function attempts to fetch the 'event_bus' service from the ServiceLocator.
	If the ServiceLocator module is not available or an error occurs during retrieval,
	it logs a warning or error message and returns None.

	Returns:
		Optional[Any]: The event bus instance if successfully retrieved, otherwise None.

	Raises:
		Logs an ImportError warning if the ServiceLocator module is unavailable.
		Logs a generic error message for any other exceptions encountered.
	"""
	try:
		from maggie.service.locator import ServiceLocator
		return ServiceLocator.get('event_bus')
	except ImportError:
		logger.warning("ServiceLocator not available, can't get event_bus")
		return None
	except Exception as e:
		logger.error(f"Error getting event_bus: {e}")
		return None


def get_state_manager() -> Optional[Any]:
	"""
	Retrieves the state manager instance using the ServiceLocator.

	This function attempts to fetch the 'state_manager' service from the 
	ServiceLocator. If the ServiceLocator is not available or an error occurs 
	during retrieval, it logs a warning or error message and returns None.

	Returns:
		Optional[Any]: The state manager instance if successfully retrieved, 
		otherwise None.

	Logs:
		- A warning if the ServiceLocator module cannot be imported.
		- An error if any other exception occurs during the retrieval process.
	"""
	try:
		from maggie.service.locator import ServiceLocator
		return ServiceLocator.get('state_manager')
	except ImportError:
		logger.warning("ServiceLocator not available, can't get state_manager")
		return None
	except Exception as e:
		logger.error(f"Error getting state_manager: {e}")
		return None


def get_resource_manager() -> Optional[Any]:
	"""
	Retrieves the resource manager instance using the ServiceLocator.

	This function attempts to fetch the 'resource_manager' service from the
	ServiceLocator. If the ServiceLocator is unavailable or an error occurs
	during the retrieval, it logs a warning or error message and returns None.

	Returns:
		Optional[Any]: The resource manager instance if successfully retrieved,
		otherwise None.

	Raises:
		ImportError: If the ServiceLocator module cannot be imported.
		Exception: For any other unexpected errors during the retrieval process.
	"""
	try:
		from maggie.service.locator import ServiceLocator
		return ServiceLocator.get('resource_manager')
	except ImportError:
		logger.warning(
			"ServiceLocator not available, can't get resource_manager")
		return None
	except Exception as e:
		logger.error(f"Error getting resource_manager: {e}")
		return None


def get_current_state() -> Optional[Any]:
	"""
	Retrieves the current state from the state manager.

	This function attempts to fetch the current state using the state manager. 
	If the state manager is unavailable or an error occurs during the retrieval, 
	it logs the error and returns `None`.

	Returns:
		Optional[Any]: The current state if successfully retrieved, otherwise `None`.
	"""
	state_manager = get_state_manager()
	if state_manager:
		try:
			return state_manager.get_current_state()
		except Exception as e:
			logger.error(f"Error getting current state: {e}")
	return None


def safe_execute(func: Callable[..., T], *args: Any, error_code: Optional[str] = None, default_return: Optional[T] = None, error_details: Dict[str, Any] = None, error_category: ErrorCategory = ErrorCategory.UNKNOWN, error_severity: ErrorSeverity = ErrorSeverity.ERROR, publish_error: bool = True, include_state_info: bool = True, **kwargs: Any) -> T:
	"""
	Safely executes a given function, handling exceptions and logging errors with detailed context.

	Args:
		func (Callable[..., T]): The function to execute.
		*args (Any): Positional arguments to pass to the function.
		error_code (Optional[str], optional): A specific error code to associate with the error. Defaults to None.
		default_return (Optional[T], optional): The value to return in case of an error. Defaults to None.
		error_details (Dict[str, Any], optional): Additional details to include in the error context. Defaults to None.
		error_category (ErrorCategory, optional): The category of the error. Defaults to ErrorCategory.UNKNOWN.
		error_severity (ErrorSeverity, optional): The severity level of the error. Defaults to ErrorSeverity.ERROR.
		publish_error (bool, optional): Whether to publish the error to an event bus. Defaults to True.
		include_state_info (bool, optional): Whether to include current state information in the error context. Defaults to True.
		**kwargs (Any): Additional keyword arguments to pass to the function.

	Returns:
		T: The result of the function execution, or the default_return value if an error occurs.

	Raises:
		Exception: Any exception raised by the function is caught and logged, but not re-raised.

	Notes:
		- Logs the error with appropriate severity based on the provided error_severity.
		- Publishes the error context to an event bus if `publish_error` is True.
		- Includes current state information in the error context if `include_state_info` is True.
		- If `error_code` is provided and exists in the ErrorRegistry, it is used to create a detailed error context.
	"""
	try:
		return func(*args, **kwargs)
	except Exception as e:
		details = error_details or {}
		if not details:
			details = {'args': str(args), 'kwargs': str(kwargs)}
		source = f"{func.__module__}.{func.__name__}"
		current_state = get_current_state()if include_state_info else None
		if error_code and ErrorRegistry.get_instance().errors.get(error_code):
			context = ErrorRegistry.get_instance().create_error(code=error_code, details=details,
																exception=e, source=source, state_object=current_state)
		else:
			message = error_code if error_code else f"Error executing {func.__name__}: {e}"
			context = ErrorContext(message=message, exception=e, category=error_category,
									severity=error_severity, source=source, details=details)
			if current_state is not None:
				context.add_state_info(current_state)
		if context.severity == ErrorSeverity.CRITICAL:
			logger.critical(context.message, exc_info=True)
		elif context.severity == ErrorSeverity.ERROR:
			logger.error(context.message, exc_info=True)
		elif context.severity == ErrorSeverity.WARNING:
			logger.warning(context.message)
		else:
			logger.debug(context.message)
		if publish_error:
			try:
				event_bus = get_event_bus()
				if event_bus:
					event_bus.publish('error_logged', context.to_dict())
			except Exception as event_error:
				logger.error(f"Failed to publish error event: {event_error}")
		return default_return if default_return is not None else cast(T, None)


def retry_operation(max_attempts: int = 3, retry_delay: float = 1., exponential_backoff: bool = True, jitter: bool = True, allowed_exceptions: Tuple[Type[Exception], ...] = (Exception,), on_retry_callback: Optional[Callable[[Exception, int], None]] = None, error_category: ErrorCategory = ErrorCategory.UNKNOWN) -> Callable:
	"""
	A decorator to retry a function or method upon encountering specified exceptions.

	This decorator allows you to specify the number of retry attempts, delay between retries,
	exponential backoff, jitter, and a callback function to execute on each retry. It is useful
	for handling transient errors in operations like network requests or database queries.

	Args:
		max_attempts (int): The maximum number of retry attempts. Defaults to 3.
		retry_delay (float): The initial delay (in seconds) between retries. Defaults to 1.0.
		exponential_backoff (bool): Whether to use exponential backoff for retry delays. Defaults to True.
		jitter (bool): Whether to add random jitter to the retry delay. Defaults to True.
		allowed_exceptions (Tuple[Type[Exception], ...]): A tuple of exception types that should trigger a retry.
			Defaults to (Exception,).
		on_retry_callback (Optional[Callable[[Exception, int], None]]): An optional callback function that is called
			on each retry. The callback receives the exception and the current attempt number as arguments.
			Defaults to None.
		error_category (ErrorCategory): An optional error category to classify the error. Defaults to ErrorCategory.UNKNOWN.

	Returns:
		Callable: A decorator that wraps the target function with retry logic.

	Raises:
		Exception: The last exception encountered if all retry attempts fail.

	Example:
		@retry_operation(max_attempts=5, retry_delay=2, allowed_exceptions=(ValueError,))
		def unreliable_function():
			# Function logic that may raise a ValueError
			pass
	"""
	def decorator(func: Callable) -> Callable:
		@wraps(func)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			import random
			last_exception = None
			for attempt in range(1, max_attempts+1):
				try:
					return func(*args, **kwargs)
				except allowed_exceptions as e:
					last_exception = e
					if attempt == max_attempts:
						logger.error(
							f"All {max_attempts} attempts failed for {func.__name__}: {e}")
						raise
					delay = retry_delay
					if exponential_backoff:
						delay = retry_delay*2**(attempt-1)
					if jitter:
						delay = delay*(.5+random.random())
					if on_retry_callback:
						try:
							on_retry_callback(e, attempt)
						except Exception as callback_error:
							logger.warning(
								f"Error in retry callback: {callback_error}")
					logger.warning(
						f"Attempt {attempt}/{max_attempts} for {func.__name__} failed: {e}. Retrying in {delay:.2f}s")
					time.sleep(delay)
			if last_exception:
				raise last_exception
			return None
		return wrapper
	return decorator


def record_error(
	message: str, 
	exception: Optional[Exception] = None, 
	category: ErrorCategory = ErrorCategory.UNKNOWN, 
	severity: ErrorSeverity = ErrorSeverity.ERROR, 
	source: str = '', 
	details: Dict[str, Any] = None, 
	publish: bool = True, 
	state_object: Any = None, 
	from_state: Any = None, 
	to_state: Any = None, 
	trigger: str = None
) -> ErrorContext:
	"""
	Records an error event, logs it, and optionally publishes it to an event bus.

	Args:
		message (str): A descriptive error message.
		exception (Optional[Exception]): The exception instance associated with the error, if any.
		category (ErrorCategory): The category of the error (e.g., UNKNOWN, STATE, RESOURCE, INPUT).
		severity (ErrorSeverity): The severity level of the error (e.g., CRITICAL, ERROR, WARNING, DEBUG).
		source (str): The source or origin of the error (e.g., module or component name).
		details (Dict[str, Any]): Additional details or metadata about the error.
		publish (bool): Whether to publish the error event to the event bus. Defaults to True.
		state_object (Any): The current state object, if applicable.
		from_state (Any): The state before a transition, if applicable.
		to_state (Any): The state after a transition, if applicable.
		trigger (str): The trigger or event that caused the state transition, if applicable.

	Returns:
		ErrorContext: An object encapsulating the error context and associated metadata.

	Raises:
		None: This function does not raise exceptions directly, but logs and handles them internally.

	Notes:
		- Logs the error message with the appropriate severity level.
		- If `publish` is True, attempts to publish the error event to the event bus.
		- Handles specific error categories (e.g., STATE, RESOURCE, INPUT) for targeted event publishing.
		- Adds state transition information if `from_state` and `to_state` are provided.
	"""
	context = ErrorContext(message=message, exception=exception,
							category=category, severity=severity, source=source, details=details)
	if state_object is not None:
		context.add_state_info(state_object)
	if from_state is not None and to_state is not None:
		context.add_transition_info(from_state, to_state, trigger)
	if severity == ErrorSeverity.CRITICAL:
		logger.critical(message, exc_info=bool(exception))
	elif severity == ErrorSeverity.ERROR:
		logger.error(message, exc_info=bool(exception))
	elif severity == ErrorSeverity.WARNING:
		logger.warning(message)
	else:
		logger.debug(message)
	if publish:
		try:
			event_bus = get_event_bus()
			if event_bus:
				event_data = context.to_dict()
				if category == ErrorCategory.STATE and from_state and to_state:
					event_bus.publish(ERROR_EVENT_STATE_TRANSITION, event_data)
				elif category == ErrorCategory.RESOURCE:
					event_bus.publish(
						ERROR_EVENT_RESOURCE_MANAGEMENT, event_data)
				elif category == ErrorCategory.INPUT:
					event_bus.publish(ERROR_EVENT_INPUT_PROCESSING, event_data)
				event_bus.publish(ERROR_EVENT_LOGGED, event_data)
		except Exception as e:
			logger.error(f"Failed to publish error event: {e}")
	return context


def with_error_handling(error_code: Optional[str] = None, error_category: ErrorCategory = ErrorCategory.UNKNOWN, error_severity: ErrorSeverity = ErrorSeverity.ERROR, publish_error: bool = True, include_state_info: bool = True):
	"""
	A decorator to wrap a function with error handling logic. This ensures that any exceptions
	raised during the execution of the wrapped function are handled gracefully and optionally
	logged or published.

	Args:
		error_code (Optional[str]): A custom error code to associate with the error. Defaults to None.
		error_category (ErrorCategory): The category of the error. Defaults to ErrorCategory.UNKNOWN.
		error_severity (ErrorSeverity): The severity level of the error. Defaults to ErrorSeverity.ERROR.
		publish_error (bool): Whether to publish the error to an external system. Defaults to True.
		include_state_info (bool): Whether to include state information in the error handling process. Defaults to True.

	Returns:
		Callable: A decorator that wraps the target function with error handling logic.
	"""
	def decorator(func):
		"""
		A decorator that wraps a function to execute it safely within a controlled environment.

		The wrapped function is executed using the `safe_execute` utility, which handles errors
		and provides additional context such as error codes, categories, severity levels, and
		optional state information.

		Args:
			func (Callable): The function to be wrapped and executed safely.

		Returns:
			Callable: A wrapped version of the input function that includes error handling.

		Notes:
			- The `safe_execute` function is expected to handle the actual execution and error
			  management logic.
			- The following parameters are passed to `safe_execute`:
				- `error_code`: A code representing the type of error.
				- `error_category`: A category to classify the error.
				- `error_severity`: The severity level of the error.
				- `publish_error`: A flag indicating whether the error should be published.
				- `include_state_info`: A flag indicating whether to include state information.
		"""
		@functools.wraps(func)
		def wrapper(*args, **kwargs): return safe_execute(func, *args, error_code=error_code, error_category=error_category,
															error_severity=error_severity, publish_error=publish_error, include_state_info=include_state_info, **kwargs)
		return wrapper
	return decorator


def create_state_transition_error(
		from_state: Any, 
		to_state: Any, 
		trigger: str, 
		details: Dict[str, Any] = None
) -> StateTransitionError: 
	"""
	Creates and logs a StateTransitionError for invalid state transitions.

	Args:
		from_state (Any): The current state before the transition. Can be an object
			with a `name` attribute or any other representation of the state.
		to_state (Any): The target state for the transition. Can be an object
			with a `name` attribute or any other representation of the state.
		trigger (str): The event or action that triggered the state transition.
		details (Dict[str, Any], optional): Additional details about the error
			or context for debugging. Defaults to None.

	Returns:
		StateTransitionError: An exception object representing the invalid state
		transition, including the error message and relevant details.

	Logs:
		Records the error with the following details:
		- Message describing the invalid state transition.
		- Error category as `STATE`.
		- Severity level as `ERROR`.
		- Source as 'StateManager.transition_to'.
		- Additional details, if provided.
	"""
	from_name = from_state.name if hasattr(from_state, 'name')else str(from_state)
	to_name = to_state.name if hasattr(to_state, 'name')else str(to_state)
	message = f"Invalid state transition: {from_name} -> {to_name} (trigger: {trigger})"
	record_error(
		message=message, 
		category=ErrorCategory.STATE,
		severity=ErrorSeverity.ERROR, 
		source='StateManager.transition_to', 
		details=details or {}, 
		from_state=from_state, 
		to_state=to_state, 
		trigger=trigger
	)
	return StateTransitionError(
			message=message,
			from_state=from_state,
			to_state=to_state,
			trigger=trigger,
			details=details
		)
