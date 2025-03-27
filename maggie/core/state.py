import threading,time,sys,logging
from enum import Enum,auto
from typing import Dict,Any,Optional,List,Callable,Tuple,Set,Union,Type
from dataclasses import dataclass,field
from maggie.utils.abstractions import IStateProvider
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',handlers=[logging.StreamHandler(sys.stdout)])
logger=logging.getLogger('maggie.core.state')
class State(Enum):
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
class StateManager(IStateProvider):
	def __init__(self,initial_state:State=State.INIT,event_bus:Any=None):self.current_state=initial_state;self.event_bus=event_bus;(self.state_handlers):Dict[State,List[Tuple[Callable,bool]]]={state:[]for state in State};(self.transition_handlers):Dict[Tuple[State,State],List[Callable]]={};self.logger=logging.getLogger('maggie.core.state.StateManager');self._lock=threading.RLock();self.valid_transitions={State.INIT:[State.STARTUP,State.IDLE,State.SHUTDOWN],State.STARTUP:[State.IDLE,State.READY,State.CLEANUP,State.SHUTDOWN],State.IDLE:[State.STARTUP,State.READY,State.CLEANUP,State.SHUTDOWN],State.LOADING:[State.ACTIVE,State.READY,State.CLEANUP,State.SHUTDOWN],State.READY:[State.LOADING,State.ACTIVE,State.BUSY,State.CLEANUP,State.SHUTDOWN],State.ACTIVE:[State.READY,State.BUSY,State.CLEANUP,State.SHUTDOWN],State.BUSY:[State.READY,State.ACTIVE,State.CLEANUP,State.SHUTDOWN],State.CLEANUP:[State.IDLE,State.SHUTDOWN],State.SHUTDOWN:[]};(self.transition_history):List[StateTransition]=[];self.max_history_size=100;self.logger.info(f"StateManager initialized with state: {initial_state.name}")
	def transition_to(self,new_state:State,trigger:str,metadata:Dict[str,Any]=None)->bool:
		with self._lock:
			if new_state==self.current_state:self.logger.debug(f"Already in state {new_state.name}, ignoring transition");return True
			if not self.is_valid_transition(self.current_state,new_state):
				error_message=f"Invalid transition from {self.current_state.name} to {new_state.name} (trigger: {trigger})";self.logger.warning(error_message)
				try:
					from maggie.utils.abstractions import get_error_handler;error_handler=get_error_handler()
					if error_handler:error_handler.record_error(message=error_message,category='STATE',severity='ERROR',source='StateManager.transition_to')
				except ImportError:pass
				except Exception:pass
				return False
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
	def __init__(self,state_manager:StateManager):self.state_manager=state_manager;self.logger=logging.getLogger(self.__class__.__name__);self._registered_handlers=[];self._register_state_handlers()
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