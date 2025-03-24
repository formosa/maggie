import os,queue,threading,time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum,auto
from typing import Any,Callable,Dict,List,Optional,Set,Tuple,Union,cast
from loguru import logger
from maggie.utils.error_handling import safe_execute,ErrorCategory,ErrorSeverity
from maggie.utils.resource.manager import ResourceManager
__all__=['State','StateTransition','EventBus','MaggieAI']
class State(Enum):IDLE=auto();READY=auto();ACTIVE=auto();CLEANUP=auto();SHUTDOWN=auto()
@dataclass
class StateTransition:
	from_state:State;to_state:State;trigger:str;timestamp:float
	def __lt__(self,other):return self.timestamp<other.timestamp
class EventBus:
	def __init__(self):self.subscribers={};self.queue=queue.PriorityQueue();self.running=False;self._worker_thread=None;self._lock=threading.RLock()
	def subscribe(self,event_type:str,callback:Callable,priority:int=0)->None:
		with self._lock:
			if event_type not in self.subscribers:self.subscribers[event_type]=[]
			self.subscribers[event_type].append((priority,callback));self.subscribers[event_type].sort(key=lambda x:x[0])
	def unsubscribe(self,event_type:str,callback:Callable)->bool:
		with self._lock:
			if event_type not in self.subscribers:return False
			for(i,(_,cb))in enumerate(self.subscribers[event_type]):
				if cb==callback:self.subscribers[event_type].pop(i);return True
			return False
	def publish(self,event_type:str,data:Any=None,priority:int=0)->None:self.queue.put((priority,(event_type,data)))
	def start(self)->bool:
		with self._lock:
			if self.running:return False
			self.running=True;self._worker_thread=threading.Thread(target=self._process_events,name='EventBusThread',daemon=True);self._worker_thread.start();logger.info('Event bus started');return True
	def stop(self)->bool:
		with self._lock:
			if not self.running:return False
			self.running=False;self.queue.put((0,None))
			if self._worker_thread:self._worker_thread.join(timeout=2.)
			logger.info('Event bus stopped');return True
	def _process_events(self)->None:
		while self.running:
			try:
				priority,event=self.queue.get(timeout=.05)
				if event is None:break
				event_type,data=event
				with self._lock:
					if event_type in self.subscribers:
						for(_,callback)in self.subscribers[event_type]:
							try:callback(data)
							except Exception as e:error_msg=f"Error in event handler for {event_type}: {e}";logger.error(error_msg);self.publish('error_logged',{'message':error_msg,'event_type':event_type,'source':'event_bus'})
				self.queue.task_done()
			except queue.Empty:time.sleep(.001)
			except Exception as e:logger.error(f"Error processing events: {e}")
class MaggieAI:
	def __init__(self,config:Dict[str,Any]):self.config=config;self.state=State.IDLE;self.event_bus=EventBus();self.extensions={};self.resource_manager=ResourceManager(config);self.inactivity_timer=None;self.inactivity_timeout=self.config.get('inactivity_timeout',300);self.wake_word_detector=None;self.stt_processor=None;self.llm_processor=None;self.tts_processor=None;self.gui=None;cpu_config=self.config.get('cpu',{});max_threads=cpu_config.get('max_threads',10);self.thread_pool=ThreadPoolExecutor(max_workers=max_threads,thread_name_prefix='maggie_thread_');self.transition_handlers={State.IDLE:self._on_enter_idle,State.READY:self._on_enter_ready,State.ACTIVE:self._on_enter_active,State.CLEANUP:self._on_enter_cleanup,State.SHUTDOWN:self._on_enter_shutdown};self._register_event_handlers();self._setup_resource_management()
	def _handle_state_transition(self,new_state:State,trigger:str)->None:
		if new_state==self.state:logger.debug(f"Already in state {new_state.name}, ignoring transition");return
		valid_transitions={State.IDLE:[State.READY,State.CLEANUP,State.SHUTDOWN],State.READY:[State.ACTIVE,State.CLEANUP,State.SHUTDOWN],State.ACTIVE:[State.READY,State.CLEANUP,State.SHUTDOWN],State.CLEANUP:[State.IDLE,State.SHUTDOWN],State.SHUTDOWN:[]}
		if new_state not in valid_transitions.get(self.state,[]):logger.warning(f"Invalid transition from {self.state.name} to {new_state.name} (trigger: {trigger})");return
		old_state=self.state;self.state=new_state;logger.info(f"State transition: {old_state.name} -> {new_state.name} (trigger: {trigger})")
		if new_state in[State.IDLE,State.CLEANUP,State.SHUTDOWN]:self.resource_manager.release_resources()
		transition=StateTransition(from_state=old_state,to_state=new_state,trigger=trigger,timestamp=time.time());self.event_bus.publish('state_changed',transition)
		if new_state in self.transition_handlers:self.transition_handlers[new_state](transition)
	def _setup_resource_management(self)->None:self.resource_manager.setup_gpu()
	def _register_event_handlers(self)->None:
		event_handlers=[('wake_word_detected',self._handle_wake_word,0),('error_logged',self._handle_error,0),('command_detected',self._handle_command,10),('inactivity_timeout',self._handle_timeout,20),('extension_completed',self._handle_extension_completed,30),('extension_error',self._handle_extension_error,30),('low_memory_warning',self._handle_low_memory,40),('gpu_memory_warning',self._handle_gpu_memory_warning,40)]
		for(event_type,handler,priority)in event_handlers:self.event_bus.subscribe(event_type,handler,priority=priority)
		logger.debug(f"Registered {len(event_handlers)} event handlers")
	def initialize_components(self) -> bool:
		try:
			from maggie.service.locator import ServiceLocator
			ServiceLocator.register('event_bus', self.event_bus)
			ServiceLocator.register('resource_manager', self.resource_manager)
			
			from maggie.service.stt.wake_word import WakeWordDetector
			self.wake_word_detector = WakeWordDetector(self.config.get('stt', {}).get('wake_word', {}))
			self.wake_word_detector.on_detected = lambda: self.event_bus.publish('wake_word_detected')
			ServiceLocator.register('wake_word_detector', self.wake_word_detector)
			
			from maggie.service.tts.processor import TTSProcessor
			self.tts_processor = TTSProcessor(self.config.get('tts', {}))
			ServiceLocator.register('tts_processor', self.tts_processor)
			
			from maggie.service.stt.processor import STTProcessor
			self.stt_processor = STTProcessor(self.config.get('stt', {}))
			self.stt_processor.tts_processor = self.tts_processor
			ServiceLocator.register('stt_processor', self.stt_processor)
			
			from maggie.service.llm.processor import LLMProcessor
			self.llm_processor = LLMProcessor(self.config.get('llm', {}))
			ServiceLocator.register('llm_processor', self.llm_processor)
			
			ServiceLocator.register('maggie_ai', self)
			
			self._initialize_extensions()
			self.event_bus.start()
			
			logger.info('All components initialized successfully')
			return True
		except ImportError as import_error:
			logger.error(f"Failed to import required module: {import_error}")
			return False
		except Exception as e:
			logger.error(f"Error initializing components: {e}")
			return False
	def _initialize_extensions(self)->None:
		from maggie.extensions.registry import ExtensionRegistry;extensions_config=self.config.get('extensions',{});registry=ExtensionRegistry();available_extensions=registry.discover_extensions();logger.info(f"Discovered {len(available_extensions)} extensions: {', '.join(available_extensions.keys())}")
		for(extension_name,extension_config)in extensions_config.items():
			if extension_config.get('enabled',True)is False:logger.info(f"Extension {extension_name} is disabled in configuration");continue
			extension=registry.instantiate_extension(extension_name,self.event_bus,extension_config)
			if extension is not None:self.extensions[extension_name]=extension;logger.info(f"Initialized extension: {extension_name}")
			else:logger.warning(f"Failed to initialize extension: {extension_name}")
		logger.info(f"Initialized {len(self.extensions)} extension modules")
	def start(self)->bool:return safe_execute(self._start_impl,error_message='Error starting Maggie AI',error_category=ErrorCategory.SYSTEM,error_severity=ErrorSeverity.CRITICAL,default_return=False)
	def _start_impl(self)->bool:
		logger.info('Starting Maggie AI Assistant')
		if not self.initialize_components():logger.error('Failed to initialize components');return False
		self._transition_to(State.IDLE,'startup');self.resource_manager.start_monitoring();logger.info('Maggie AI Assistant started successfully');return True
	def stop(self)->bool:return safe_execute(self._stop_impl,error_message='Error stopping Maggie AI',error_category=ErrorCategory.SYSTEM,error_severity=ErrorSeverity.ERROR,default_return=False)
	def _stop_impl(self)->bool:logger.info('Stopping Maggie AI Assistant');self._transition_to(State.SHUTDOWN,'stop_requested');logger.info('Maggie AI Assistant stopped successfully');return True
	def _transition_to(self,new_state:State,trigger:str)->None:self._handle_state_transition(new_state,trigger)
	def _on_enter_idle(self,transition:StateTransition)->None:
		if self.inactivity_timer:self.inactivity_timer.cancel();self.inactivity_timer=None
		if self.stt_processor:self.stt_processor.stop_streaming();self.stt_processor.stop_listening()
		if self.llm_processor:self.llm_processor.unload_model()
		if self.wake_word_detector:self.wake_word_detector.start()
		self.resource_manager.clear_gpu_memory();logger.info('Entered IDLE state - waiting for wake word')
	def _on_enter_ready(self,transition:StateTransition)->None:
		if self.wake_word_detector:self.wake_word_detector.stop()
		if self.stt_processor:self.stt_processor.start_listening();self.stt_processor.start_streaming(on_intermediate=lambda text:self.event_bus.publish('intermediate_transcription',text),on_final=lambda text:self.event_bus.publish('final_transcription',text));self.tts_processor.speak('Ready for your command')
		self._start_inactivity_timer();self.thread_pool.submit(self._listen_for_commands);logger.info('Entered READY state - listening for commands with real-time transcription')
	def _on_enter_active(self,transition:StateTransition)->None:self._start_inactivity_timer();logger.info('Entered ACTIVE state - executing command or extension')
	def _on_enter_cleanup(self,transition:StateTransition)->None:
		if self.inactivity_timer:self.inactivity_timer.cancel();self.inactivity_timer=None
		self._stop_all_extensions()
		if self.stt_processor:self.stt_processor.stop_listening()
		if self.llm_processor:self.llm_processor.unload_model()
		self.resource_manager.clear_gpu_memory()
		if transition.trigger=='shutdown_requested':self._transition_to(State.SHUTDOWN,'cleanup_completed')
		else:self._transition_to(State.IDLE,'cleanup_completed')
		logger.info('Entered CLEANUP state - releasing resources')
	def _stop_all_extensions(self)->None:
		for(extension_name,extension)in self.extensions.items():
			try:
				if hasattr(extension,'stop')and callable(extension.stop):extension.stop();logger.debug(f"Stopped extension: {extension_name}")
			except Exception as e:logger.error(f"Error stopping extension {extension_name}: {e}")
	def _on_enter_shutdown(self,transition:StateTransition)->None:self.resource_manager.stop_monitoring();self.event_bus.stop();self.resource_manager.clear_gpu_memory();self.thread_pool.shutdown(wait=True,timeout=5);logger.info('Entered SHUTDOWN state - application will exit')
	def _handle_wake_word(self,_:Any)->None:
		if self.state==State.IDLE:logger.info('Wake word detected');self._transition_to(State.READY,'wake_word_detected')
	def _handle_command(self,command:str)->None:
		if self.state!=State.READY:return
		command=command.lower().strip();logger.info(f"Command detected: {command}");tts_processor=self.tts_processor
		if command in['sleep','go to sleep']:
			if tts_processor:tts_processor.speak('Going to sleep')
			else:logger.warning('No TTS processor available for speech output')
			self._transition_to(State.CLEANUP,'sleep_command');return
		if command in['shutdown','turn off']:
			if tts_processor:tts_processor.speak('Shutting down')
			else:logger.warning('No TTS processor available for speech output')
			self._transition_to(State.CLEANUP,'shutdown_requested');return
		extension_triggered=self._check_extension_commands(command)
		if extension_triggered:return
		if tts_processor:tts_processor.speak("I didn't understand that command")
		else:logger.warning('No TTS processor available for speech output')
		logger.warning(f"Unknown command: {command}")
	def _check_extension_commands(self,command:str)->bool:
		for(extension_name,extension)in self.extensions.items():
			extension_trigger=extension.get_trigger()
			if extension_trigger and extension_trigger in command:logger.info(f"Triggered extension: {extension_name}");self._transition_to(State.ACTIVE,f"extension_{extension_name}");self.thread_pool.submit(self._run_extension,extension_name);return True
		return False
	def _handle_timeout(self,_:Any)->None:
		if self.state==State.READY:logger.info('Inactivity timeout reached');self.tts_processor.speak('Going to sleep due to inactivity');self._transition_to(State.CLEANUP,'inactivity_timeout')
	def _handle_extension_completed(self,extension_name:str)->None:
		if self.state==State.ACTIVE:logger.info(f"Extension completed: {extension_name}");self._transition_to(State.READY,f"extension_{extension_name}_completed")
	def _start_inactivity_timer(self)->None:
		if self.inactivity_timer:self.inactivity_timer.cancel()
		self.inactivity_timer=threading.Timer(self.inactivity_timeout,lambda:self.event_bus.publish('inactivity_timeout'));self.inactivity_timer.daemon=True;self.inactivity_timer.start();logger.debug(f"Started inactivity timer: {self.inactivity_timeout}s")
	def _listen_for_commands(self)->None:
		if self.state!=State.READY:return
		try:
			logger.debug('Listening for commands...');success,text=self.stt_processor.recognize_speech(timeout=1e1)
			if success and text:logger.info(f"Recognized: {text}");self.event_bus.publish('command_detected',text)
			else:self.thread_pool.submit(self._listen_for_commands)
		except Exception as e:logger.error(f"Error listening for commands: {e}");self.thread_pool.submit(self._listen_for_commands)
	def _run_extension(self,extension_name:str)->None:
		if extension_name not in self.extensions:logger.error(f"Unknown extension: {extension_name}");self._transition_to(State.READY,'unknown_extension');return
		extension=self.extensions[extension_name]
		try:
			logger.info(f"Starting extension: {extension_name}");success=extension.start()
			if not success:logger.error(f"Failed to start extension: {extension_name}");self.event_bus.publish('extension_error',extension_name);self._transition_to(State.READY,f"extension_{extension_name}_failed")
		except Exception as e:logger.error(f"Error running extension {extension_name}: {e}");self.event_bus.publish('extension_error',extension_name);self._transition_to(State.READY,f"extension_{extension_name}_error")
	def shutdown(self)->bool:
		logger.info('Shutdown initiated')
		if self.state!=State.SHUTDOWN:self._transition_to(State.CLEANUP,'shutdown_requested')
		return True
	def timeout(self)->None:
		if self.state in[State.READY,State.ACTIVE]:logger.info('Manual sleep requested');self.tts_processor.speak('Going to sleep');self._transition_to(State.CLEANUP,'manual_timeout')
	def process_command(self,extension:Any=None)->bool:
		if extension:
			extension_name=None
			for(name,ext)in self.extensions.items():
				if ext==extension:extension_name=name;break
			if extension_name:logger.info(f"Direct activation of extension: {extension_name}");self._transition_to(State.ACTIVE,f"extension_{extension_name}");self.thread_pool.submit(self._run_extension,extension_name);return True
		return False
	def set_gui(self,gui)->None:self.gui=gui
	def update_gui(self,event_type:str,data:Any=None)->None:
		if self.gui and hasattr(self.gui,'safe_update_gui'):
			if event_type=='state_change':self.gui.safe_update_gui(self.gui.update_state,data)
			elif event_type=='chat_message':is_user=data.get('is_user',False);message=data.get('message','');self.gui.safe_update_gui(self.gui.log_chat,message,is_user)
			elif event_type=='event_log':self.gui.safe_update_gui(self.gui.log_event,data)
			elif event_type=='error_log':self.gui.safe_update_gui(self.gui.log_error,data)
	def _handle_error(self,error_data:Any)->None:
		if self.gui:self.update_gui('error_log',error_data)
	def _handle_low_memory(self,_:Any)->None:
		if self.llm_processor:self.llm_processor.unload_model()
		self.resource_manager.reduce_memory_usage();logger.warning('Low memory condition detected - unloading models')
	def _handle_gpu_memory_warning(self,_:Any)->None:
		self.resource_manager.clear_gpu_memory()
		if self.llm_processor and hasattr(self.llm_processor,'reduce_gpu_layers'):self.llm_processor.reduce_gpu_layers()
		logger.warning('High GPU memory usage detected - freeing CUDA memory')
	def _handle_extension_error(self,extension_name:str)->None:
		logger.error(f"Extension error: {extension_name}")
		if self.tts_processor:self.tts_processor.speak(f"There was a problem with the {extension_name} extension")
		if self.gui:self.update_gui('error_log',{'message':f"Extension error: {extension_name}",'source':'extension','extension':extension_name})