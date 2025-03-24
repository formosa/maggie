import time,threading
from enum import Enum,auto
from typing import Dict,Any,Optional,List,Callable,Tuple
from dataclasses import dataclass
from loguru import logger
from maggie.utils.error_handling import safe_execute,ErrorCategory,ErrorSeverity
from maggie.utils.logging import log_operation,ComponentLogger
class State(Enum):IDLE=auto();READY=auto();ACTIVE=auto();CLEANUP=auto();SHUTDOWN=auto()
@dataclass
class StateTransition:
	from_state:State;to_state:State;trigger:str;timestamp:float
	def __lt__(self,other):return self.timestamp<other.timestamp
class StateManager:
	def __init__(self,initial_state:State=State.IDLE,event_bus:Any=None):self.current_state=initial_state;self.event_bus=event_bus;(self.state_handlers):Dict[State,List[Callable]]={state:[]for state in State};(self.transition_handlers):Dict[Tuple[State,State],List[Callable]]={};self.logger=ComponentLogger('StateManager');self._lock=threading.RLock();self.valid_transitions={State.IDLE:[State.READY,State.CLEANUP,State.SHUTDOWN],State.READY:[State.ACTIVE,State.CLEANUP,State.SHUTDOWN],State.ACTIVE:[State.READY,State.CLEANUP,State.SHUTDOWN],State.CLEANUP:[State.IDLE,State.SHUTDOWN],State.SHUTDOWN:[]};self.logger.info(f"StateManager initialized with state: {initial_state.name}")
	@log_operation(component='StateManager')
	def transition_to(self,new_state:State,trigger:str)->bool:
		with self._lock:
			if new_state==self.current_state:self.logger.debug(f"Already in state {new_state.name}, ignoring transition");return True
			if new_state not in self.valid_transitions.get(self.current_state,[]):self.logger.warning(f"Invalid transition from {self.current_state.name} to {new_state.name} (trigger: {trigger})");return False
			old_state=self.current_state;self.current_state=new_state;self.logger.info(f"State transition: {old_state.name} -> {new_state.name} (trigger: {trigger})");transition=StateTransition(from_state=old_state,to_state=new_state,trigger=trigger,timestamp=time.time());self._execute_state_handlers(old_state,'exit',transition);self._execute_transition_handlers(old_state,new_state,transition);self._execute_state_handlers(new_state,'entry',transition)
			if self.event_bus:self.event_bus.publish('state_changed',transition)
			return True
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
	def is_in_state(self,state:State)->bool:
		with self._lock:return self.current_state==state
	def is_valid_transition(self,from_state:State,to_state:State)->bool:return to_state in self.valid_transitions.get(from_state,[])
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
class StateAwareComponent:
	def __init__(self,state_manager:StateManager):self.state_manager=state_manager;self.logger=ComponentLogger(self.__class__.__name__);self._register_state_handlers()
	def _register_state_handlers(self)->None:pass
	def handle_state_change(self,transition:StateTransition)->None:pass
	def on_state_entry(self,state:State,transition:StateTransition)->None:pass
	def on_state_exit(self,state:State,transition:StateTransition)->None:pass