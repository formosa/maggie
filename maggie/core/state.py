"""
Maggie AI Assistant - Finite State Machine Module
================================================

This module defines a state management system for handling state transitions, 
state-specific behaviors, and associated metadata. It includes the following components:

Classes:
	- State: An enumeration of possible states with associated properties for styling and display.
	- StateTransition: Represents a transition between two states, including metadata and animation properties.
	- StateManager: Manages the current state, valid transitions, and handlers for state-specific and transition-specific logic.
	- StateAwareComponent: A base class for components that need to respond to state changes.

State Enum:
	- INIT: Initial state (frozen).
	- STARTUP: Startup state (frozen).
	- IDLE: Idle state.
	- LOADING: Loading state (frozen).
	- READY: Ready state.
	- ACTIVE: Active state.
	- BUSY: Busy state (frozen).
	- CLEANUP: Cleanup state (frozen).
	- SHUTDOWN: Shutdown state (frozen).

StateTransition:
	- Attributes:
		- from_state: The state from which the transition originates.
		- to_state: The state to which the transition leads.
		- trigger: The event or action that triggered the transition.
		- timestamp: The time at which the transition occurred.
		- metadata: Additional metadata associated with the transition.
	- Methods:
		- animation_type: Returns the type of animation for the transition.
		- animation_duration: Returns the duration of the animation.
		- get_animation_properties: Returns a dictionary of animation properties.
		- to_dict: Converts the transition to a dictionary representation.

StateManager:
	- Attributes:
		- current_state: The current state of the system.
		- event_bus: An optional event bus for publishing state change events.
		- state_handlers: Handlers for entry and exit actions for each state.
		- transition_handlers: Handlers for specific state transitions.
		- valid_transitions: A dictionary defining valid state transitions.
		- transition_history: A history of state transitions.
		- max_history_size: The maximum size of the transition history.
	- Methods:
		- transition_to: Transitions to a new state if valid.
		- register_state_handler: Registers a handler for a specific state.
		- register_transition_handler: Registers a handler for a specific state transition.
		- get_current_state: Returns the current state.
		- get_style_for_current_state: Returns the style properties for the current state.
		- is_in_state: Checks if the system is in a specific state.
		- is_valid_transition: Checks if a transition between two states is valid.
		- get_valid_transitions: Returns a list of valid transitions from a given state.
		- get_transition_history: Returns the history of state transitions.
		- reset: Resets the state manager to a new state.

StateAwareComponent:
	- Attributes:
		- state_manager: The associated StateManager instance.
		- logger: A logger for the component.
		- _registered_handlers: A list of registered state handlers.
	- Methods:
		- _register_state_handlers: Registers entry and exit handlers for states.
		- handle_state_change: Handles state change events (to be overridden by subclasses).
		- on_state_entry: Handles state entry events (to be overridden by subclasses).
		- on_state_exit: Handles state exit events (to be overridden by subclasses).
		- get_component_state: Returns the current state of the component.
		- update_ui: Returns UI-related properties for the current state.
"""
import threading,time
from enum import Enum,auto
from typing import Dict,Any,Optional,List,Callable,Tuple,Set,Union,Type
from dataclasses import dataclass,field
from maggie.utils.error_handling import safe_execute,ErrorCategory,ErrorSeverity,with_error_handling,record_error,StateTransitionError
from maggie.utils.logging import ComponentLogger,log_operation,logging_context
class State(Enum):
	"""
	State(Enum):
		An enumeration representing various states with associated properties and methods for styling.

		Enum Members:
			- INIT: Represents the initial state.
			- STARTUP: Represents the startup state.
			- IDLE: Represents the idle state.
			- LOADING: Represents the loading state.
			- READY: Represents the ready state.
			- ACTIVE: Represents the active state.
			- BUSY: Represents the busy state.
			- CLEANUP: Represents the cleanup state.
			- SHUTDOWN: Represents the shutdown state.

		Properties:
			- bg_color (str): Returns the background color associated with the state.
			- font_color (str): Returns the font color associated with the state. Dark text is used for certain states.
			- display_name (str): Returns a human-readable, capitalized name of the state.

		Methods:
			- get_style() -> Dict[str, str]: Returns a dictionary containing CSS style properties for the state, 
			  including background color, font color, border, font weight, padding, and border radius.
	"""
	INIT=auto();STARTUP=auto();IDLE=auto();LOADING=auto();READY=auto();ACTIVE=auto();BUSY=auto();CLEANUP=auto();SHUTDOWN=auto()
	@property
	def bg_color(self)->str:colors={State.INIT:'#E0E0E0',State.STARTUP:'#B3E5FC',State.IDLE:'#C8E6C9',State.LOADING:'#FFE0B2',State.READY:'#A5D6A7',State.ACTIVE:'#FFCC80',State.BUSY:'#FFAB91',State.CLEANUP:'#E1BEE7',State.SHUTDOWN:'#EF9A9A'};return colors.get(self,'#FFFFFF')
	@property
	def font_color(self)->str:dark_text_states={State.INIT,State.STARTUP,State.IDLE,State.LOADING,State.READY};return'#212121'if self in dark_text_states else'#FFFFFF'
	@property
	def display_name(self)->str:return self.name.capitalize()
	def get_style(self)->Dict[str,str]:return{'background':self.bg_color,'color':self.font_color,'border':'1px solid #424242','font-weight':'bold','padding':'4px 8px','border-radius':'4px'}
@dataclass
class StateTransition:
	"""
	Represents a transition between two states in a state machine.

	Attributes:
		from_state (State): The initial state before the transition.
		to_state (State): The target state after the transition.
		trigger (str): The event or action that triggered the transition.
		timestamp (float): The time at which the transition occurred.
		metadata (Dict[str, Any]): Additional metadata associated with the transition.

	Methods:
		__lt__(other: 'StateTransition') -> bool:
			Compares two StateTransition objects based on their timestamps.

		animation_type -> str:
			Returns the type of animation associated with the transition based on the target state.

		animation_duration -> int:
			Returns the duration of the animation in milliseconds based on the target state.

		get_animation_properties() -> Dict[str, Any]:
			Returns a dictionary containing animation properties such as type, duration, and easing.

		to_dict() -> Dict[str, Any]:
			Converts the StateTransition object into a dictionary representation.
	"""
	from_state:State;to_state:State;trigger:str;timestamp:float;metadata:Dict[str,Any]=field(default_factory=dict)
	def __lt__(self,other:'StateTransition')->bool:return self.timestamp<other.timestamp
	@property
	def animation_type(self)->str:
		if self.to_state==State.SHUTDOWN:return'fade'
		elif self.to_state==State.BUSY:return'bounce'
		else:return'slide'
	@property
	def animation_duration(self)->int:
		if self.to_state in{State.SHUTDOWN,State.CLEANUP}:return 800
		elif self.to_state in{State.BUSY,State.LOADING}:return 400
		else:return 300
	def get_animation_properties(self)->Dict[str,Any]:return{'type':self.animation_type,'duration':self.animation_duration,'easing':'ease-in-out'}
	def to_dict(self)->Dict[str,Any]:return{'from_state':self.from_state.name,'to_state':self.to_state.name,'trigger':self.trigger,'timestamp':self.timestamp,'metadata':self.metadata}
class StateManager:
	"""
	StateManager is a class responsible for managing state transitions in a system. It provides mechanisms to handle state-specific logic, validate transitions, and maintain a history of state changes.

	Attributes:
		current_state (State): The current state of the system.
		event_bus (Any): An optional event bus for publishing state change events.
		state_handlers (Dict[State, List[Tuple[Callable, bool]]]): Handlers for entry and exit actions for each state.
		transition_handlers (Dict[Tuple[State, State], List[Callable]]): Handlers for specific state transitions.
		logger (ComponentLogger): Logger for the StateManager.
		_lock (threading.RLock): A reentrant lock for thread-safe operations.
		valid_transitions (Dict[State, List[State]]): A mapping of valid transitions for each state.
		transition_history (List[StateTransition]): A history of state transitions.
		max_history_size (int): The maximum size of the transition history.

	Methods:
		transition_to(new_state: State, trigger: str, metadata: Dict[str, Any] = None) -> bool:
			Transitions the system to a new state if the transition is valid. Executes state and transition handlers.

		register_state_handler(state: State, handler: Callable, is_entry: bool = True) -> None:
			Registers a handler for a specific state. Handlers can be for entry or exit actions.

		register_transition_handler(from_state: State, to_state: State, handler: Callable) -> None:
			Registers a handler for a specific state transition.

		get_current_state() -> State:
			Returns the current state of the system.

		get_style_for_current_state() -> Dict[str, str]:
			Returns the style associated with the current state.

		is_in_state(state: State) -> bool:
			Checks if the system is currently in the specified state.

		is_valid_transition(from_state: State, to_state: State) -> bool:
			Checks if a transition from one state to another is valid.

		get_valid_transitions(from_state: Optional[State] = None) -> List[State]:
			Returns a list of valid transitions from the specified state or the current state.

		get_transition_history(limit: int = 10) -> List[StateTransition]:
			Returns the most recent state transitions, limited to the specified number.

		reset(new_state: State = State.INIT) -> None:
			Resets the state manager to a new state and clears the transition history.

		_execute_state_handlers(state: State, handler_type: str, transition: StateTransition) -> None:
			Executes entry or exit handlers for a specific state.

		_execute_transition_handlers(from_state: State, to_state: State, transition: StateTransition) -> None:
			Executes handlers for a specific state transition.
	"""
	def __init__(self,initial_state:State=State.INIT,event_bus:Any=None):self.current_state=initial_state;self.event_bus=event_bus;(self.state_handlers):Dict[State,List[Tuple[Callable,bool]]]={state:[]for state in State};(self.transition_handlers):Dict[Tuple[State,State],List[Callable]]={};self.logger=ComponentLogger('StateManager');self._lock=threading.RLock();self.valid_transitions={State.INIT:[State.STARTUP,State.IDLE,State.SHUTDOWN],State.STARTUP:[State.IDLE,State.READY,State.CLEANUP,State.SHUTDOWN],State.IDLE:[State.STARTUP,State.READY,State.CLEANUP,State.SHUTDOWN],State.LOADING:[State.ACTIVE,State.READY,State.CLEANUP,State.SHUTDOWN],State.READY:[State.LOADING,State.ACTIVE,State.BUSY,State.CLEANUP,State.SHUTDOWN],State.ACTIVE:[State.READY,State.BUSY,State.CLEANUP,State.SHUTDOWN],State.BUSY:[State.READY,State.ACTIVE,State.CLEANUP,State.SHUTDOWN],State.CLEANUP:[State.IDLE,State.SHUTDOWN],State.SHUTDOWN:[]};(self.transition_history):List[StateTransition]=[];self.max_history_size=100;self.logger.info(f"StateManager initialized with state: {initial_state.name}")
	@log_operation(component='StateManager')
	@with_error_handling(error_category=ErrorCategory.SYSTEM,error_severity=ErrorSeverity.ERROR)
	def transition_to(self,new_state:State,trigger:str,metadata:Dict[str,Any]=None)->bool:
		with self._lock:
			if new_state==self.current_state:self.logger.debug(f"Already in state {new_state.name}, ignoring transition");return True
			if not self.is_valid_transition(self.current_state,new_state):error_message=f"Invalid transition from {self.current_state.name} to {new_state.name} (trigger: {trigger})";self.logger.warning(error_message);record_error(message=error_message,category=ErrorCategory.SYSTEM,severity=ErrorSeverity.ERROR,source='StateManager.transition_to');return False
			old_state=self.current_state;self.current_state=new_state;self.logger.info(f"State transition: {old_state.name} -> {new_state.name} (trigger: {trigger})");transition=StateTransition(from_state=old_state,to_state=new_state,trigger=trigger,timestamp=time.time(),metadata=metadata or{});self.transition_history.append(transition)
			if len(self.transition_history)>self.max_history_size:self.transition_history=self.transition_history[-self.max_history_size:]
			try:
				self._execute_state_handlers(old_state,'exit',transition);self._execute_transition_handlers(old_state,new_state,transition);self._execute_state_handlers(new_state,'entry',transition)
				if self.event_bus:self.event_bus.publish('state_changed',transition)
				return True
			except Exception as e:self.logger.error(f"Error during state transition: {e}");return False
	def register_state_handler(self,state:State,handler:Callable,is_entry:bool=True)->None:
		with self._lock:
			if state not in self.state_handlers:self.state_handlers[state]=[]
			self.state_handlers[state].append((handler,is_entry));self.logger.debug(f"Registered {'entry'if is_entry else'exit'} handler for state: {state.name}")
	def register_transition_handler(self,from_state:State,to_state:State,handler:Callable)->None:
		with self._lock:
			transition_key=from_state,to_state
			if transition_key not in self.transition_handlers:self.transition_handlers[transition_key]=[]
			self.transition_handlers[transition_key].append(handler);self.logger.debug(f"Registered transition handler for {from_state.name} -> {to_state.name}")
	def get_current_state(self)->State:
		with self._lock:return self.current_state
	def get_style_for_current_state(self)->Dict[str,str]:
		with self._lock:return self.current_state.get_style()
	def is_in_state(self,state:State)->bool:
		with self._lock:return self.current_state==state
	def is_valid_transition(self,from_state:State,to_state:State)->bool:return to_state in self.valid_transitions.get(from_state,[])
	def get_valid_transitions(self,from_state:Optional[State]=None)->List[State]:source_state=from_state if from_state is not None else self.current_state;return self.valid_transitions.get(source_state,[])
	def get_transition_history(self,limit:int=10)->List[StateTransition]:
		with self._lock:return self.transition_history[-limit:]
	def _execute_state_handlers(self,state:State,handler_type:str,transition:StateTransition)->None:
		is_entry=handler_type=='entry'
		for(handler,handler_is_entry)in self.state_handlers.get(state,[]):
			if handler_is_entry==is_entry:
				try:handler(transition)
				except Exception as e:self.logger.error(f"Error in state {handler_type} handler for {state.name}: {e}")
	def _execute_transition_handlers(self,from_state:State,to_state:State,transition:StateTransition)->None:
		transition_key=from_state,to_state
		for handler in self.transition_handlers.get(transition_key,[]):
			try:handler(transition)
			except Exception as e:self.logger.error(f"Error in transition handler for {from_state.name} -> {to_state.name}: {e}")
	def reset(self,new_state:State=State.INIT)->None:
		with self._lock:
			old_state=self.current_state;self.current_state=new_state;self.logger.info(f"State manager reset: {old_state.name} -> {new_state.name}");self.transition_history=[]
			if self.event_bus:self.event_bus.publish('state_manager_reset',new_state)
class StateAwareComponent:
	"""
	A base class for components that are aware of and respond to state changes 
	managed by a StateManager. This class provides mechanisms for registering 
	state entry and exit handlers, handling state transitions, and retrieving 
	component-specific state information.

	Attributes:
		state_manager (StateManager): The state manager responsible for managing 
			the states and transitions.
		logger (ComponentLogger): Logger instance for the component.
		_registered_handlers (list): A list of registered state handlers, 
			including their associated states and entry/exit flags.

	Methods:
		_register_state_handlers():
			Registers state entry and exit handlers for all states defined in 
			the State enumeration. Handlers are methods named `on_enter_<state>` 
			or `on_exit_<state>` in the subclass.

		handle_state_change(transition: StateTransition) -> None:
			Handles a state change. This method can be overridden in subclasses 
			to provide custom behavior during state transitions.

		on_state_entry(state: State, transition: StateTransition) -> None:
			Called when entering a state. Can be overridden in subclasses to 
			define behavior upon state entry.

		on_state_exit(state: State, transition: StateTransition) -> None:
			Called when exiting a state. Can be overridden in subclasses to 
			define behavior upon state exit.

		get_component_state() -> Dict[str, Any]:
			Retrieves the current state of the component, including the system 
			state and the component type.

		update_ui(state: State) -> Dict[str, Any]:
			Updates the UI representation of the component based on the given 
			state. Returns a dictionary containing state details such as name, 
			display name, and style.

	Example:
		Here's an example of creating a new component that extends the 
		StateAwareComponent class:

		```python
		class ExampleComponent(StateAwareComponent):
			def __init__(self, state_manager: StateManager):
				super().__init__(state_manager)

			def on_enter_ready(self, transition: StateTransition):
				print(f"Entering READY state. Trigger: {transition.trigger}")

			def on_exit_ready(self, transition: StateTransition):
				print(f"Exiting READY state. Trigger: {transition.trigger}")

			def handle_state_change(self, transition: StateTransition):
				print(f"State changed from {transition.from_state.name} to {transition.to_state.name}")

		# Usage
		state_manager = StateManager(initial_state=State.INIT)
		component = ExampleComponent(state_manager)

		# Transition to READY state
		state_manager.transition_to(State.READY, trigger="example_trigger")
		```
	"""
	def __init__(self,state_manager:StateManager):self.state_manager=state_manager;self.logger=ComponentLogger(self.__class__.__name__);self._registered_handlers=[];self._register_state_handlers()
	def _register_state_handlers(self)->None:
		for state in State:
			method_name=f"on_enter_{state.name.lower()}"
			if hasattr(self,method_name)and callable(getattr(self,method_name)):handler=getattr(self,method_name);self.state_manager.register_state_handler(state,handler,True);self._registered_handlers.append((state,handler,True))
			method_name=f"on_exit_{state.name.lower()}"
			if hasattr(self,method_name)and callable(getattr(self,method_name)):handler=getattr(self,method_name);self.state_manager.register_state_handler(state,handler,False);self._registered_handlers.append((state,handler,False))
	def handle_state_change(self,transition:StateTransition)->None:pass
	def on_state_entry(self,state:State,transition:StateTransition)->None:pass
	def on_state_exit(self,state:State,transition:StateTransition)->None:pass
	def get_component_state(self)->Dict[str,Any]:return{'system_state':self.state_manager.get_current_state().name,'component_type':self.__class__.__name__}
	def update_ui(self,state:State)->Dict[str,Any]:return{'state':state.name,'display_name':state.display_name,'style':state.get_style()}