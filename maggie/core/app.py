import os,queue,threading,time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum,auto
from typing import Any,Callable,Dict,List,Optional,Set,Tuple,Union,cast
from maggie.utils.error_handling import safe_execute,ErrorCategory,ErrorSeverity,with_error_handling,record_error,StateTransitionError
from maggie.utils.logging import ComponentLogger,log_operation,logging_context
from maggie.utils import get_resource_manager
from maggie.utils.config.manager import ConfigManager
from maggie.core.state import State,StateTransition,StateManager
from maggie.core.event import EventBus,EventEmitter,EventListener,EventPriority,INPUT_ACTIVATION_EVENT,INPUT_DEACTIVATION_EVENT
from maggie.service.locator import ServiceLocator
__all__=['MaggieAI']
class MaggieAI(EventEmitter,EventListener):
	def __init__(self,config_path:str='config.yaml'):self.config_manager=ConfigManager(config_path);self.config=self.config_manager.load();self.event_bus=EventBus();EventEmitter.__init__(self,self.event_bus);EventListener.__init__(self,self.event_bus);self.logger=ComponentLogger('MaggieAI');self.state_manager=StateManager(State.INIT,self.event_bus);self._register_core_services();self.extensions={};self.inactivity_timer=None;self.inactivity_timeout=self.config.get('inactivity_timeout',300);self.wake_word_detector=None;self.stt_processor=None;self.llm_processor=None;self.tts_processor=None;self.gui=None;cpu_config=self.config.get('cpu',{});max_threads=cpu_config.get('max_threads',10);self.thread_pool=ThreadPoolExecutor(max_workers=max_threads,thread_name_prefix='maggie_thread_');self._register_state_handlers();self._setup_resource_management();self.logger.info('MaggieAI instance created')
	@property
	def state(self)->State:return self.state_manager.get_current_state()
	def _register_state_handlers(self)->None:self.state_manager.register_state_handler(State.INIT,self._on_enter_init,True);self.state_manager.register_state_handler(State.STARTUP,self._on_enter_startup,True);self.state_manager.register_state_handler(State.IDLE,self._on_enter_idle,True);self.state_manager.register_state_handler(State.LOADING,self._on_enter_loading,True);self.state_manager.register_state_handler(State.READY,self._on_enter_ready,True);self.state_manager.register_state_handler(State.ACTIVE,self._on_enter_active,True);self.state_manager.register_state_handler(State.BUSY,self._on_enter_busy,True);self.state_manager.register_state_handler(State.CLEANUP,self._on_enter_cleanup,True);self.state_manager.register_state_handler(State.SHUTDOWN,self._on_enter_shutdown,True);self.state_manager.register_state_handler(State.ACTIVE,self._on_exit_active,False);self.state_manager.register_state_handler(State.BUSY,self._on_exit_busy,False);self.state_manager.register_transition_handler(State.INIT,State.STARTUP,self._on_transition_init_to_startup);self.state_manager.register_transition_handler(State.STARTUP,State.IDLE,self._on_transition_startup_to_idle);self.state_manager.register_transition_handler(State.IDLE,State.READY,self._on_transition_idle_to_ready);self.state_manager.register_transition_handler(State.READY,State.LOADING,self._on_transition_ready_to_loading);self.state_manager.register_transition_handler(State.LOADING,State.ACTIVE,self._on_transition_loading_to_active);self.state_manager.register_transition_handler(State.ACTIVE,State.READY,self._on_transition_active_to_ready);self.state_manager.register_transition_handler(State.ACTIVE,State.BUSY,self._on_transition_active_to_busy);self.state_manager.register_transition_handler(State.BUSY,State.READY,self._on_transition_busy_to_ready);self.logger.debug('State handlers registered')
	def _register_event_handlers(self)->None:
		event_handlers=[('wake_word_detected',self._handle_wake_word,EventPriority.HIGH),('error_logged',self._handle_error,EventPriority.HIGH),('command_detected',self._handle_command,EventPriority.NORMAL),('inactivity_timeout',self._handle_timeout,EventPriority.NORMAL),('extension_completed',self._handle_extension_completed,EventPriority.NORMAL),('extension_error',self._handle_extension_error,EventPriority.NORMAL),('low_memory_warning',self._handle_low_memory,EventPriority.LOW),('gpu_memory_warning',self._handle_gpu_memory_warning,EventPriority.LOW),(INPUT_ACTIVATION_EVENT,self._handle_input_activation,EventPriority.NORMAL),(INPUT_DEACTIVATION_EVENT,self._handle_input_deactivation,EventPriority.NORMAL),('intermediate_transcription',self._handle_intermediate_transcription,EventPriority.LOW),('final_transcription',self._handle_final_transcription,EventPriority.NORMAL)]
		for(event_type,handler,priority)in event_handlers:self.listen(event_type,handler,priority=priority)
		self.logger.debug(f"Registered {len(event_handlers)} event handlers")
	def _register_core_services(self)->bool:
		try:from maggie.service.locator import ServiceLocator;ServiceLocator.register('event_bus',self.event_bus);ServiceLocator.register('state_manager',self.state_manager);ServiceLocator.register('maggie_ai',self);ServiceLocator.register('config_manager',self.config_manager);self.logger.debug('Core services registered with ServiceLocator');return True
		except ImportError as e:self.logger.error(f"Failed to import ServiceLocator: {e}");return False
		except Exception as e:self.logger.error(f"Error registering core services: {e}");return False
	def _setup_resource_management(self)->None:
		ResourceManager=get_resource_manager()
		if ResourceManager is not None:
			self.resource_manager=ResourceManager(self.config)
			if ServiceLocator.has_service('resource_manager'):self.logger.debug('Resource manager already registered')
			else:ServiceLocator.register('resource_manager',self.resource_manager)
			self.resource_manager.setup_gpu();self.resource_manager.apply_hardware_specific_optimizations();self.logger.debug('Resource management setup complete')
		else:self.logger.error('Failed to get ResourceManager class');self.resource_manager=None
	def _on_enter_init(self,transition:StateTransition)->None:self.config_manager.apply_state_specific_config(State.INIT)
	def _on_enter_startup(self,transition:StateTransition)->None:self.config_manager.apply_state_specific_config(State.STARTUP)
	def _on_enter_idle(self,transition:StateTransition)->None:self.config_manager.apply_state_specific_config(State.IDLE)
	def _on_enter_loading(self,transition:StateTransition)->None:self.config_manager.apply_state_specific_config(State.LOADING)
	def _on_enter_ready(self,transition:StateTransition)->None:self.config_manager.apply_state_specific_config(State.READY)
	def _on_enter_active(self,transition:StateTransition)->None:self.config_manager.apply_state_specific_config(State.ACTIVE)
	def _on_enter_busy(self,transition:StateTransition)->None:self.config_manager.apply_state_specific_config(State.BUSY)
	def _on_enter_cleanup(self,transition:StateTransition)->None:self.config_manager.apply_state_specific_config(State.CLEANUP)
	def _on_enter_shutdown(self,transition:StateTransition)->None:self.config_manager.apply_state_specific_config(State.SHUTDOWN)
	def _on_exit_active(self,transition:StateTransition)->None:
		if transition.to_state==State.BUSY and self.resource_manager:self.resource_manager.optimizer.optimize_for_busy_state()
	def _on_exit_busy(self,transition:StateTransition)->None:
		if transition.to_state==State.READY and self.resource_manager:self.resource_manager.reduce_memory_usage()
	def _on_transition_init_to_startup(self,transition:StateTransition)->None:
		if self.resource_manager:self.resource_manager.preallocate_for_state(State.STARTUP)
	def _on_transition_startup_to_idle(self,transition:StateTransition)->None:
		if self.resource_manager:self.resource_manager.preallocate_for_state(State.IDLE)
	def _on_transition_idle_to_ready(self,transition:StateTransition)->None:
		if self.resource_manager:self.resource_manager.preallocate_for_state(State.READY)
	def _on_transition_ready_to_loading(self,transition:StateTransition)->None:
		if self.resource_manager:self.resource_manager.preallocate_for_state(State.LOADING)
	def _on_transition_loading_to_active(self,transition:StateTransition)->None:
		if self.resource_manager:self.resource_manager.preallocate_for_state(State.ACTIVE)
	def _on_transition_active_to_ready(self,transition:StateTransition)->None:
		if self.resource_manager:self.resource_manager.preallocate_for_state(State.READY)
	def _on_transition_active_to_busy(self,transition:StateTransition)->None:
		if self.resource_manager:self.resource_manager.preallocate_for_state(State.BUSY)
	def _on_transition_busy_to_ready(self,transition:StateTransition)->None:
		if self.resource_manager:self.resource_manager.preallocate_for_state(State.READY)
	@log_operation(component='MaggieAI')
	def _initialize_extensions(self)->None:
		try:
			from maggie.extensions.registry import ExtensionRegistry;extensions_config=self.config.get('extensions',{});registry=ExtensionRegistry();available_extensions=registry.discover_extensions();self.logger.info(f"Discovered {len(available_extensions)} extensions: {', '.join(available_extensions.keys())}")
			for(extension_name,extension_config)in extensions_config.items():
				if extension_config.get('enabled',True)is False:self.logger.info(f"Extension {extension_name} is disabled in configuration");continue
				extension=registry.instantiate_extension(extension_name,self.event_bus,extension_config)
				if extension is not None:self.extensions[extension_name]=extension;self.logger.info(f"Initialized extension: {extension_name}")
				else:self.logger.warning(f"Failed to initialize extension: {extension_name}")
			self.logger.info(f"Initialized {len(self.extensions)} extension modules")
		except Exception as e:self.logger.error(f"Error initializing extensions: {e}")
	@log_operation(component='MaggieAI')
	def initialize_components(self)->bool:
		with logging_context(component='MaggieAI',operation='initialize_components')as ctx:
			try:
				if not self._register_core_services():return False
				init_success=self._initialize_wake_word_detector()and self._initialize_tts_processor()and self._initialize_stt_processor()and self._initialize_llm_processor()
				if not init_success:self.logger.error('Failed to initialize core components');return False
				self._initialize_extensions();self.event_bus.start()
				if self.resource_manager:self.resource_manager.apply_hardware_specific_optimizations()
				self.logger.info('All components initialized successfully');return True
			except ImportError as import_error:self.logger.error(f"Failed to import required module: {import_error}");return False
			except Exception as e:self.logger.error(f"Error initializing components: {e}");return False
	def start(self)->bool:
		self.logger.info('Starting MaggieAI');self._register_event_handlers();success=self.initialize_components()
		if not success:self.logger.error('Failed to initialize components');return False
		if self.state_manager.get_current_state()==State.INIT:self.state_manager.transition_to(State.STARTUP,'system_start')
		if self.resource_manager and hasattr(self.resource_manager,'start_monitoring'):self.resource_manager.start_monitoring()
		self.logger.info('MaggieAI started successfully');return True
	def _initialize_wake_word_detector(self)->bool:
		try:from maggie.service.stt.wake_word import WakeWordDetector;from maggie.service.locator import ServiceLocator;wake_word_config=self.config.get('stt',{}).get('wake_word',{});self.wake_word_detector=WakeWordDetector(wake_word_config);self.wake_word_detector.on_detected=lambda:self.event_bus.publish('wake_word_detected');ServiceLocator.register('wake_word_detector',self.wake_word_detector);self.logger.debug('Wake word detector initialized');return True
		except Exception as e:self.logger.error(f"Error initializing wake word detector: {e}");return False
	def _initialize_tts_processor(self)->bool:
		try:from maggie.service.tts.processor import TTSProcessor;from maggie.service.locator import ServiceLocator;tts_config=self.config.get('tts',{});self.tts_processor=TTSProcessor(tts_config);ServiceLocator.register('tts_processor',self.tts_processor);self.logger.debug('TTS processor initialized');return True
		except Exception as e:self.logger.error(f"Error initializing TTS processor: {e}");return False
	def _initialize_stt_processor(self)->bool:
		try:
			from maggie.service.stt.processor import STTProcessor;from maggie.service.locator import ServiceLocator;stt_config=self.config.get('stt',{});self.stt_processor=STTProcessor(stt_config)
			if self.tts_processor:self.stt_processor.tts_processor=self.tts_processor
			ServiceLocator.register('stt_processor',self.stt_processor);self.logger.debug('STT processor initialized');return True
		except Exception as e:self.logger.error(f"Error initializing STT processor: {e}");return False
	def _initialize_llm_processor(self)->bool:
		try:from maggie.service.llm.processor import LLMProcessor;from maggie.service.locator import ServiceLocator;llm_config=self.config.get('llm',{});self.llm_processor=LLMProcessor(llm_config);ServiceLocator.register('llm_processor',self.llm_processor);self.logger.debug('LLM processor initialized');return True
		except Exception as e:self.logger.error(f"Error initializing LLM processor: {e}");return False
	def set_gui(self,gui:Any)->None:self.gui=gui;self.logger.debug('GUI reference set')
	def shutdown(self)->None:
		self.logger.info('Shutting down MaggieAI')
		if self.resource_manager and hasattr(self.resource_manager,'stop_monitoring'):self.resource_manager.stop_monitoring()
		if self.state_manager.get_current_state()!=State.SHUTDOWN:self.state_manager.transition_to(State.SHUTDOWN,'system_shutdown')
		if self.resource_manager and hasattr(self.resource_manager,'release_resources'):self.resource_manager.release_resources()
		self.thread_pool.shutdown(wait=False);self.logger.info('MaggieAI shutdown complete')
	def timeout(self)->None:
		self.logger.info('Inactivity timeout reached')
		if self.state_manager.get_current_state()!=State.IDLE:self.state_manager.transition_to(State.IDLE,'inactivity_timeout')
	def _handle_wake_word(self,data:Any=None)->None:pass
	def _handle_error(self,error_data:Dict[str,Any])->None:pass
	def _handle_command(self,command:str)->None:pass
	def _handle_timeout(self,data:Any=None)->None:pass
	def _handle_extension_completed(self,extension_name:str)->None:pass
	def _handle_extension_error(self,error_data:Dict[str,Any])->None:pass
	def _handle_low_memory(self,event_data:Dict[str,Any])->None:pass
	def _handle_gpu_memory_warning(self,event_data:Dict[str,Any])->None:pass
	def _handle_input_activation(self,data:Any=None)->None:pass
	def _handle_input_deactivation(self,data:Any=None)->None:pass
	def _handle_intermediate_transcription(self,text:str)->None:pass
	def _handle_final_transcription(self,text:str)->None:pass
	def process_command(self,command:str=None,extension:Any=None)->None:pass