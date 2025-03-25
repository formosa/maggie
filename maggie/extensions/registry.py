import os,sys,importlib,pkgutil
from typing import Dict,Type,Any,Optional,List,Tuple,cast
from maggie.extensions.base import ExtensionBase
from maggie.core.state import State,StateTransition,StateAwareComponent
from maggie.core.event import EventListener,EventPriority
from maggie.utils.error_handling import safe_execute,with_error_handling,ErrorCategory,ErrorSeverity,record_error
from maggie.utils.logging import ComponentLogger,log_operation,logging_context
from maggie.service.locator import ServiceLocator
class ExtensionRegistry(StateAwareComponent,EventListener):
	def __init__(self,extensions_path:Optional[str]=None):
		self.state_manager=None;self.event_bus=None
		try:
			self.state_manager=ServiceLocator.get('state_manager')
			if self.state_manager:StateAwareComponent.__init__(self,self.state_manager);self._register_state_handlers()
		except Exception as e:self.logger=ComponentLogger('ExtensionRegistry');self.logger.warning(f"State manager not available: {e}")
		try:
			self.event_bus=ServiceLocator.get('event_bus')
			if self.event_bus:EventListener.__init__(self,self.event_bus);self._register_event_handlers()
		except Exception as e:
			if not hasattr(self,'logger'):self.logger=ComponentLogger('ExtensionRegistry')
			self.logger.warning(f"Event bus not found in ServiceLocator: {e}")
		if not hasattr(self,'logger'):self.logger=ComponentLogger('ExtensionRegistry')
		(self._registry):Dict[str,Type[ExtensionBase]]={};(self._lazy_load_registry):Dict[str,Dict[str,Any]]={};(self._active_extensions):Dict[str,ExtensionBase]={}
		if extensions_path is None:package_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));self._extensions_path=os.path.join(package_dir,'extensions')
		else:self._extensions_path=extensions_path
		os.makedirs(self._extensions_path,exist_ok=True)
	def _register_state_handlers(self)->None:self.state_manager.register_state_handler(State.INIT,self._on_enter_init,True);self.state_manager.register_state_handler(State.STARTUP,self._on_enter_startup,True);self.state_manager.register_state_handler(State.IDLE,self._on_enter_idle,True);self.state_manager.register_state_handler(State.LOADING,self._on_enter_loading,True);self.state_manager.register_state_handler(State.READY,self._on_enter_ready,True);self.state_manager.register_state_handler(State.ACTIVE,self._on_enter_active,True);self.state_manager.register_state_handler(State.BUSY,self._on_enter_busy,True);self.state_manager.register_state_handler(State.CLEANUP,self._on_enter_cleanup,True);self.state_manager.register_state_handler(State.SHUTDOWN,self._on_enter_shutdown,True);self.state_manager.register_transition_handler(State.IDLE,State.READY,self._on_transition_idle_to_ready);self.state_manager.register_transition_handler(State.READY,State.ACTIVE,self._on_transition_ready_to_active);self.state_manager.register_transition_handler(State.ACTIVE,State.BUSY,self._on_transition_active_to_busy);self.state_manager.register_transition_handler(State.BUSY,State.READY,self._on_transition_busy_to_ready)
	def _register_event_handlers(self)->None:
		event_handlers=[('error_logged',self._handle_error,EventPriority.HIGH),('extension_completed',self._handle_extension_completed,EventPriority.NORMAL),('extension_error',self._handle_extension_error,EventPriority.HIGH)]
		for(event_type,handler,priority)in event_handlers:self.listen(event_type,handler,priority=priority)
	def _handle_error(self,error_data:Dict[str,Any])->None:
		if'extension'in error_data:extension_name=error_data.get('extension');self.logger.warning(f"Error in extension {extension_name}: {error_data.get('message','Unknown error')}")
	def _handle_extension_completed(self,extension_name:str)->None:
		self.logger.info(f"Extension {extension_name} completed successfully")
		if self.state_manager:
			current_state=self.state_manager.get_current_state()
			if current_state in[State.IDLE,State.CLEANUP]:self._unload_extension(extension_name)
	def _handle_extension_error(self,extension_name:str)->None:self.logger.error(f"Extension {extension_name} encountered an error");self._unload_extension(extension_name)
	def _on_enter_init(self,transition:StateTransition)->None:self._registry.clear();self._active_extensions.clear();self._lazy_load_registry.clear();self.logger.info('Extension registry reset in INIT state')
	def _on_enter_startup(self,transition:StateTransition)->None:self.discover_extensions()
	def _on_enter_idle(self,transition:StateTransition)->None:self._unload_all_extensions()
	def _on_enter_loading(self,transition:StateTransition)->None:self._preload_common_extensions()
	def _on_enter_ready(self,transition:StateTransition)->None:pass
	def _on_enter_active(self,transition:StateTransition)->None:pass
	def _on_enter_busy(self,transition:StateTransition)->None:self._pause_inactive_extensions()
	def _on_enter_cleanup(self,transition:StateTransition)->None:self._unload_all_extensions()
	def _on_enter_shutdown(self,transition:StateTransition)->None:self._registry.clear();self._active_extensions.clear();self._lazy_load_registry.clear();self.logger.info('Extension registry cleared in SHUTDOWN state')
	def _on_transition_idle_to_ready(self,transition:StateTransition)->None:pass
	def _on_transition_ready_to_active(self,transition:StateTransition)->None:pass
	def _on_transition_active_to_busy(self,transition:StateTransition)->None:self._optimize_extensions_for_busy_state()
	def _on_transition_busy_to_ready(self,transition:StateTransition)->None:self._resume_paused_extensions()
	def _preload_common_extensions(self)->None:self.logger.debug('Preloading common extensions')
	def _unload_extension(self,extension_name:str)->None:
		if extension_name in self._active_extensions:
			extension=self._active_extensions[extension_name]
			if hasattr(extension,'running')and extension.running:
				try:extension.stop()
				except Exception as e:self.logger.warning(f"Error stopping extension {extension_name}: {e}")
			del self._active_extensions[extension_name];self.logger.debug(f"Unloaded extension: {extension_name}")
	def _unload_all_extensions(self)->None:
		extension_names=list(self._active_extensions.keys())
		for extension_name in extension_names:self._unload_extension(extension_name)
	def _pause_inactive_extensions(self)->None:
		for(extension_name,extension)in self._active_extensions.items():
			if hasattr(extension,'running')and extension.running:
				if hasattr(extension,'pause'):
					try:extension.pause();self.logger.debug(f"Paused extension: {extension_name}")
					except Exception as e:self.logger.warning(f"Error pausing extension {extension_name}: {e}")
	def _resume_paused_extensions(self)->None:
		for(extension_name,extension)in self._active_extensions.items():
			if hasattr(extension,'resume'):
				try:extension.resume();self.logger.debug(f"Resumed extension: {extension_name}")
				except Exception as e:self.logger.warning(f"Error resuming extension {extension_name}: {e}")
	def _optimize_extensions_for_busy_state(self)->None:
		resource_manager=None
		try:resource_manager=ServiceLocator.get('resource_manager')
		except Exception:pass
		if resource_manager:self.logger.debug('Optimizing extensions for busy state')
	@log_operation(component='ExtensionRegistry')
	@with_error_handling(error_category=ErrorCategory.EXTENSION)
	def discover_extensions(self)->Dict[str,Type[ExtensionBase]]:
		self._registry={};self._lazy_load_registry={}
		if self._extensions_path not in sys.path:sys.path.append(self._extensions_path)
		with logging_context(component='ExtensionRegistry',operation='discover_extensions'):
			extension_modules=[]
			for(_,module_name,is_pkg)in pkgutil.iter_modules([self._extensions_path]):
				if is_pkg and module_name!='__pycache__'and not module_name.startswith('_'):extension_modules.append(module_name)
			for module_name in extension_modules:self._process_extension_package(module_name)
			extension_count=len(self._registry);self.logger.info(f"Discovered {extension_count} extensions: {', '.join(self._registry.keys())}");return self._registry
	@with_error_handling(error_category=ErrorCategory.EXTENSION,error_severity=ErrorSeverity.WARNING)
	def _process_extension_package(self,module_name:str)->None:
		current_state=None
		if self.state_manager:current_state=self.state_manager.get_current_state()
		try:
			extension_package=importlib.import_module(module_name);main_module_name=f"{module_name}.{module_name}"
			try:extension_module=importlib.import_module(main_module_name);self._extract_extension_classes(extension_module,module_name)
			except ImportError:self.logger.warning(f"Could not import main module for extension: {module_name}");self._publish_extension_error(module_name,'Failed to import main module',current_state)
		except ImportError:self.logger.warning(f"Failed to import extension package {module_name}");self._publish_extension_error(module_name,'Failed to import extension package',current_state)
		except Exception as e:self.logger.error(f"Unexpected error processing extension package {module_name}: {e}");self._publish_extension_error(module_name,str(e),current_state)
	def _extract_extension_classes(self,module:Any,module_name:str)->None:
		extension_found=False
		for attr_name in dir(module):
			attr=getattr(module,attr_name)
			if isinstance(attr,type)and issubclass(attr,ExtensionBase)and attr is not ExtensionBase:self._registry[module_name]=attr;self._lazy_load_registry[module_name]={'module_path':module.__name__,'class_name':attr.__name__};self.logger.info(f"Discovered extension: {module_name}");extension_found=True;break
		if not extension_found:self.logger.debug(f"No valid extension class found in module: {module_name}")
	def _publish_extension_error(self,extension_name:str,error_message:str,state:Optional[State]=None,severity:ErrorSeverity=ErrorSeverity.WARNING)->None:
		if self.event_bus:
			state_info={}
			if state:state_info={'current_state':state.name}
			event_data={'source':'ExtensionRegistry','extension':extension_name,'message':f"Extension error: {error_message}",'category':ErrorCategory.EXTENSION.value,'severity':severity.value,'state_info':state_info};self.event_bus.publish('error_logged',event_data)
	@log_operation(component='ExtensionRegistry')
	def get_extension_class(self,extension_name:str)->Optional[Type[ExtensionBase]]:
		if extension_name in self._registry:return self._registry[extension_name]
		if extension_name in self._lazy_load_registry:
			try:info=self._lazy_load_registry[extension_name];module=importlib.import_module(info['module_path']);ext_class=getattr(module,info['class_name']);self._registry[extension_name]=ext_class;return ext_class
			except Exception as e:self.logger.error(f"Error lazy loading extension {extension_name}: {e}")
		return None
	@log_operation(component='ExtensionRegistry')
	@with_error_handling(error_category=ErrorCategory.EXTENSION)
	def instantiate_extension(self,extension_name:str,event_bus:Any,config:Dict[str,Any])->Optional[ExtensionBase]:
		current_state=None
		if self.state_manager:current_state=self.state_manager.get_current_state()
		if extension_name in self._active_extensions:return self._active_extensions[extension_name]
		extension_class=self.get_extension_class(extension_name)
		if extension_class is None:self.logger.error(f"Extension not found: {extension_name}");self._publish_extension_error(extension_name,'Extension not found',current_state,ErrorSeverity.ERROR);return None
		try:
			extension_instance=extension_class(event_bus,config)
			if hasattr(extension_instance,'initialize')and callable(extension_instance.initialize):extension_instance.initialize()
			self._active_extensions[extension_name]=extension_instance;self.logger.info(f"Instantiated extension: {extension_name}");return extension_instance
		except Exception as e:self.logger.error(f"Error instantiating extension {extension_name}: {e}");self._publish_extension_error(extension_name,str(e),current_state,ErrorSeverity.ERROR);return None
	def get_available_extensions(self)->List[str]:return list(set(list(self._registry.keys())+list(self._lazy_load_registry.keys())))
	@log_operation(component='ExtensionRegistry')
	@with_error_handling(error_category=ErrorCategory.EXTENSION,error_severity=ErrorSeverity.WARNING)
	def reload_extension(self,extension_name:str)->bool:
		current_state=None
		if self.state_manager:current_state=self.state_manager.get_current_state()
		if extension_name not in self._registry and extension_name not in self._lazy_load_registry:self.logger.warning(f"Cannot reload unknown extension: {extension_name}");return False
		self._unload_extension(extension_name)
		try:
			module_name=f"{extension_name}.{extension_name}"
			if module_name in sys.modules:module=sys.modules[module_name];importlib.reload(module);self._extract_extension_classes(module,extension_name);self.logger.info(f"Reloaded extension: {extension_name}");return True
			else:
				try:extension_module=importlib.import_module(module_name);self._extract_extension_classes(extension_module,extension_name);self.logger.info(f"Loaded extension: {extension_name}");return True
				except ImportError:self.logger.warning(f"Extension module {module_name} not loaded, cannot reload");return False
		except Exception as e:self.logger.error(f"Error reloading extension {extension_name}: {e}");self._publish_extension_error(extension_name,f"Failed to reload: {e}",current_state,ErrorSeverity.ERROR);return False
	def get_extension_for_state(self,extension_name:str,state:Optional[State]=None)->Optional[ExtensionBase]:
		if state is None and self.state_manager:state=self.state_manager.get_current_state()
		if state in[State.ACTIVE,State.READY]:
			if self.event_bus:config=self._get_extension_config_for_state(extension_name,state);return self.instantiate_extension(extension_name,self.event_bus,config)
		elif state==State.BUSY:
			if extension_name in self._active_extensions:return self._active_extensions[extension_name]
		return None
	def _get_extension_config_for_state(self,extension_name:str,state:State)->Dict[str,Any]:
		base_config={}
		try:
			config_manager=ServiceLocator.get('config_manager')
			if config_manager:
				app_config=config_manager.config
				if'extensions'in app_config and extension_name in app_config['extensions']:base_config=app_config['extensions'][extension_name].copy()
		except Exception:pass
		if state==State.ACTIVE:base_config['optimize_for_active']=True
		elif state==State.BUSY:base_config['optimize_for_busy']=True;base_config['reduce_resource_usage']=True
		elif state==State.READY:base_config['optimize_for_ready']=True
		return base_config