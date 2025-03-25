from typing import Dict,Any,Optional,List,Callable,Type,TypeVar,Set,Tuple,cast
from maggie.core.state import State
from maggie.utils.logging import ComponentLogger
from maggie.utils.error_handling import safe_execute,ErrorCategory,with_error_handling
T=TypeVar('T')
class ServiceLocator:
	_services:Dict[str,Any]={};_state_constraints:Dict[str,Set[State]]={};_transition_constraints:Dict[str,List[Tuple[State,State]]]={};_current_state:Optional[State]=None;_last_transition:Optional[Tuple[State,State]]=None;_logger=ComponentLogger('ServiceLocator')
	@classmethod
	def register(cls,name:str,service:Any,available_states:Optional[List[State]]=None)->None:
		cls._services[name]=service
		if available_states:cls._state_constraints[name]=set(available_states)
		cls._logger.info(f"Registered service: {name}")
	@classmethod
	def register_for_transition(cls,name:str,service:Any,transitions:List[Tuple[State,State]])->None:cls._services[name]=service;cls._transition_constraints[name]=transitions;cls._logger.info(f"Registered transition-specific service: {name}")
	@classmethod
	def update_state(cls,new_state:State)->None:
		if cls._current_state!=new_state:cls._last_transition=(cls._current_state,new_state)if cls._current_state else None;cls._current_state=new_state;cls._logger.debug(f"Updated service locator state to {new_state.name}")
	@classmethod
	@with_error_handling(error_category=ErrorCategory.SYSTEM)
	def get(cls,name:str)->Optional[Any]:
		service=cls._services.get(name)
		if service is None:cls._logger.warning(f"Service not found: {name}");return None
		if cls._current_state and name in cls._state_constraints:
			allowed_states=cls._state_constraints.get(name,set())
			if cls._current_state not in allowed_states:cls._logger.warning(f"Service {name} not available in current state {cls._current_state.name}");return None
		if cls._last_transition and name in cls._transition_constraints:
			allowed_transitions=cls._transition_constraints.get(name,[])
			if cls._last_transition not in allowed_transitions:from_state_name=cls._last_transition[0].name if cls._last_transition[0]else'None';to_state_name=cls._last_transition[1].name;cls._logger.warning(f"Service {name} not available for transition {from_state_name} -> {to_state_name}");return None
		return service
	@classmethod
	def get_typed(cls,name:str,service_type:Type[T])->Optional[T]:
		service=cls.get(name)
		if service is None:return None
		if not isinstance(service,service_type):cls._logger.error(f"Service type mismatch: {name} is {type(service).__name__}, expected {service_type.__name__}");return None
		return cast(T,service)
	@classmethod
	def has_service(cls,name:str)->bool:return name in cls._services
	@classmethod
	def get_or_create(cls,name:str,factory:Callable[[],T],available_states:Optional[List[State]]=None)->T:
		service=cls.get(name)
		if service is None:service=factory();cls.register(name,service,available_states)
		return service
	@classmethod
	def get_available_services(cls,state:Optional[State]=None)->List[str]:
		check_state=state if state is not None else cls._current_state
		if check_state is None:return list(cls._services.keys())
		available_services=[]
		for name in cls._services:
			if name not in cls._state_constraints:available_services.append(name)
			elif check_state in cls._state_constraints.get(name,set()):available_services.append(name)
		return available_services
	@classmethod
	def clear(cls)->None:cls._services.clear();cls._state_constraints.clear();cls._transition_constraints.clear();cls._current_state=None;cls._last_transition=None;cls._logger.info('Cleared all services')
	@classmethod
	def list_services(cls)->List[str]:return list(cls._services.keys())