import os,threading,time,queue
from typing import Dict,Any,Optional,Callable,List,Union
import pvporcupine,pyaudio,numpy as np
from maggie.core.state import State,StateTransition,StateAwareComponent
from maggie.core.event import EventEmitter,EventListener,EventPriority
from maggie.utils.logging import ComponentLogger,log_operation
from maggie.utils.error_handling import safe_execute,with_error_handling,ErrorCategory,ErrorSeverity,record_error
from maggie.service.locator import ServiceLocator
__all__=['WakeWordDetector']
class WakeWordDetector(StateAwareComponent,EventListener):
	def __init__(self,config:Dict[str,Any]):
		self.state_manager=None;self.event_bus=None
		try:
			self.state_manager=ServiceLocator.get('state_manager')
			if self.state_manager:StateAwareComponent.__init__(self,self.state_manager);self._register_state_handlers()
		except Exception as e:self.logger=ComponentLogger('WakeWordDetector');self.logger.warning(f"State manager not available: {e}")
		try:
			self.event_bus=ServiceLocator.get('event_bus')
			if self.event_bus:EventListener.__init__(self,self.event_bus);self._register_event_handlers()
		except Exception as e:
			if not hasattr(self,'logger'):self.logger=ComponentLogger('WakeWordDetector')
			self.logger.warning(f"Event bus not found in ServiceLocator, falling back to callback mode: {e}")
		if not hasattr(self,'logger'):self.logger=ComponentLogger('WakeWordDetector')
		self.logger.info('Initializing Wake Word Detection...');self.config=config;self.access_key=config.get('access_key',None);self.sensitivity=config.get('sensitivity',.5);self.keyword=config.get('keyword','Maggie');self.keyword_path=config.get('keyword_path',None);self.cpu_threshold=config.get('cpu_threshold',5.);self.dedicated_core_enabled=config.get('dedicated_core_enabled',True);self.dedicated_core=config.get('dedicated_core',0);self.real_time_priority=config.get('real_time_priority',True);self.minimal_processing=config.get('minimal_processing',True);self.on_detected=None;self.running=False;self._active_state=True;self._porcupine=None;self._pyaudio=None;self._audio_stream=None;self._detection_thread=None;self._stop_event=threading.Event();self._audio_queue=queue.Queue(maxsize=3);self._lock=threading.RLock();self.logger.info(f"Wake word detector initialized with sensitivity: {self.sensitivity}")
	def _register_state_handlers(self)->None:self.state_manager.register_state_handler(State.INIT,self._on_enter_init,True);self.state_manager.register_state_handler(State.STARTUP,self._on_enter_startup,True);self.state_manager.register_state_handler(State.IDLE,self._on_enter_idle,True);self.state_manager.register_state_handler(State.READY,self._on_enter_ready,True);self.state_manager.register_state_handler(State.ACTIVE,self._on_enter_active,True);self.state_manager.register_state_handler(State.BUSY,self._on_enter_busy,True);self.state_manager.register_state_handler(State.CLEANUP,self._on_enter_cleanup,True);self.state_manager.register_state_handler(State.SHUTDOWN,self._on_enter_shutdown,True);self.state_manager.register_state_handler(State.IDLE,self._on_exit_idle,False);self.state_manager.register_state_handler(State.ACTIVE,self._on_exit_active,False);self.logger.debug('State handlers registered')
	def _register_event_handlers(self)->None:
		event_handlers=[('resource_warning',self._handle_resource_warning,EventPriority.HIGH),('low_memory_warning',self._handle_low_memory,EventPriority.HIGH),('input_activation',self._handle_input_activation,EventPriority.NORMAL),('input_deactivation',self._handle_input_deactivation,EventPriority.NORMAL)]
		for(event_type,handler,priority)in event_handlers:self.listen(event_type,handler,priority=priority)
		self.logger.debug(f"Registered {len(event_handlers)} event handlers")
	def _on_enter_init(self,transition:StateTransition)->None:self._deactivate_detection();self.logger.debug('Inactive in INIT state')
	def _on_enter_startup(self,transition:StateTransition)->None:self._deactivate_detection();self.logger.debug('Inactive in STARTUP state')
	def _on_enter_idle(self,transition:StateTransition)->None:self._activate_detection();self.logger.debug('Active in IDLE state')
	def _on_enter_ready(self,transition:StateTransition)->None:self._activate_detection();self.logger.debug('Active in READY state')
	def _on_enter_active(self,transition:StateTransition)->None:self._deactivate_detection();self.logger.debug('Inactive in ACTIVE state')
	def _on_enter_busy(self,transition:StateTransition)->None:self._deactivate_detection();self.logger.debug('Inactive in BUSY state')
	def _on_enter_cleanup(self,transition:StateTransition)->None:self._deactivate_detection();self.logger.debug('Inactive in CLEANUP state')
	def _on_enter_shutdown(self,transition:StateTransition)->None:self._deactivate_detection();self.logger.debug('Inactive in SHUTDOWN state')
	def _on_exit_idle(self,transition:StateTransition)->None:
		if hasattr(transition,'to_state')and transition.to_state==State.READY:self.logger.debug('Transitioning from IDLE to READY')
	def _on_exit_active(self,transition:StateTransition)->None:
		if hasattr(transition,'to_state'):
			if transition.to_state==State.READY:self._activate_detection()
	def _handle_resource_warning(self,event_data:Dict[str,Any])->None:
		resource_type=event_data.get('resource_type','')
		if resource_type=='cpu'and self.running:self.cpu_threshold=max(2.,self.cpu_threshold*.8);self.logger.info(f"Reduced CPU threshold to {self.cpu_threshold} due to resource warning")
	def _handle_low_memory(self,event_data:Dict[str,Any])->None:
		if self.running and self.state_manager:
			current_state=self.state_manager.get_current_state()
			if current_state in[State.IDLE,State.READY]:self.logger.info('Adjusting wake word behavior due to low memory')
	def _handle_input_activation(self,event_data:Any)->None:self._deactivate_detection()
	def _handle_input_deactivation(self,event_data:Any)->None:
		if self.state_manager:
			current_state=self.state_manager.get_current_state()
			if current_state in[State.IDLE,State.READY]:self._activate_detection()
	def _activate_detection(self)->None:
		self._active_state=True
		if not self.running:self.start()
	def _deactivate_detection(self)->None:
		self._active_state=False
		if self.running:self.stop()
	@log_operation(component='WakeWordDetector')
	@with_error_handling(error_category=ErrorCategory.SYSTEM,error_severity=ErrorSeverity.ERROR)
	def start(self)->bool:
		with self._lock:
			if self.running:self.logger.warning('Wake word detector already running');return True
			if not self._init_porcupine():return False
			if not self._init_audio_stream():self._cleanup_resources();return False
			self._stop_event.clear();self._detection_thread=threading.Thread(target=self._detection_loop,name='WakeWordThread',daemon=True);self._detection_thread.start();self.running=True;self.logger.info('Wake word detection started');return True
	def _init_porcupine(self)->bool:
		try:
			keywords=[];sensitivities=[]
			if self.keyword_path and os.path.exists(self.keyword_path):keywords=[self.keyword_path];sensitivities=[self.sensitivity];self.logger.info(f"Using custom keyword model: {self.keyword_path}")
			else:default_keyword='computer';keywords=[default_keyword];sensitivities=[self.sensitivity];self.logger.warning(f"Custom keyword not found, using default: '{default_keyword}'");self.logger.warning(f"To use '{self.keyword}', create a custom keyword at console.picovoice.ai")
			try:self._porcupine=pvporcupine.create(access_key=self.access_key,keywords=keywords,sensitivities=sensitivities);return True
			except ValueError as e:self.logger.error(f"Keyword error: {e}");self.logger.info("Falling back to 'computer' keyword");self._porcupine=pvporcupine.create(access_key=self.access_key,keywords=['computer'],sensitivities=[self.sensitivity]);return True
		except pvporcupine.PorcupineError as e:record_error(message=f"Porcupine error: {e}",exception=e,category=ErrorCategory.SYSTEM,severity=ErrorSeverity.ERROR,source='WakeWordDetector._init_porcupine');return False
		except Exception as e:record_error(message=f"Error initializing Porcupine: {e}",exception=e,category=ErrorCategory.SYSTEM,severity=ErrorSeverity.ERROR,source='WakeWordDetector._init_porcupine');return False
	def _init_audio_stream(self)->bool:
		try:self._pyaudio=pyaudio.PyAudio();self._audio_stream=self._pyaudio.open(rate=self._porcupine.sample_rate,channels=1,format=pyaudio.paInt16,input=True,frames_per_buffer=self._porcupine.frame_length,stream_callback=self._audio_callback);return True
		except pyaudio.PyAudioError as e:record_error(message=f"PyAudio error: {e}",exception=e,category=ErrorCategory.SYSTEM,severity=ErrorSeverity.ERROR,source='WakeWordDetector._init_audio_stream');return False
		except Exception as e:record_error(message=f"Error initializing audio stream: {e}",exception=e,category=ErrorCategory.SYSTEM,severity=ErrorSeverity.ERROR,source='WakeWordDetector._init_audio_stream');return False
	def _audio_callback(self,in_data,frame_count,time_info,status):
		if self._stop_event.is_set():return in_data,pyaudio.paComplete
		try:self._audio_queue.put(in_data,block=False)
		except queue.Full:pass
		return in_data,pyaudio.paContinue
	def _detection_loop(self)->None:
		self.logger.debug('Wake word detection thread started')
		try:
			import psutil;process=psutil.Process()
			if self.dedicated_core_enabled and hasattr(process,'cpu_affinity'):
				try:process.cpu_affinity([self.dedicated_core]);self.logger.debug(f"Set wake word detection to use dedicated core: {self.dedicated_core}")
				except Exception as e:self.logger.warning(f"Failed to set CPU affinity: {e}")
			if self.real_time_priority and hasattr(process,'nice'):
				try:import psutil;process.nice(psutil.REALTIME_PRIORITY_CLASS if hasattr(psutil,'REALTIME_PRIORITY_CLASS')else psutil.HIGH_PRIORITY_CLASS);self.logger.debug('Set wake word detection thread to high priority')
				except Exception as e:self.logger.warning(f"Failed to set process priority: {e}")
		except ImportError:process=None;self.logger.warning('psutil not available, CPU monitoring disabled')
		cpu_threshold=self.cpu_threshold;current_state=None
		if self.state_manager:
			try:
				current_state=self.state_manager.get_current_state()
				if current_state==State.IDLE:cpu_threshold=max(1.,cpu_threshold*.5)
			except Exception:pass
		base_sleep_time=.005 if self.minimal_processing else .01
		while not self._stop_event.is_set():
			try:
				audio_data=self._audio_queue.get(timeout=.1);pcm=np.frombuffer(audio_data,dtype=np.int16);keyword_index=self._porcupine.process(pcm)
				if keyword_index>=0:self._handle_wake_word_detected()
				if process:
					current_cpu=process.cpu_percent(interval=None)
					if current_cpu>cpu_threshold:time.sleep(base_sleep_time)
			except queue.Empty:continue
			except Exception as e:self.logger.error(f"Error in wake word detection loop: {e}");time.sleep(.1)
		self.logger.debug('Wake word detection thread stopped')
	def _handle_wake_word_detected(self)->None:
		self.logger.info('Wake word detected!')
		if self.state_manager:
			try:
				current_state=self.state_manager.get_current_state()
				if current_state==State.IDLE:self.state_manager.transition_to(State.READY,trigger='wake_word_detected')
			except Exception as e:self.logger.error(f"Error handling state transition: {e}")
		if self.event_bus:self.logger.debug('Publishing wake_word_detected event');self.event_bus.publish('wake_word_detected')
		elif self.on_detected:self.logger.debug('Using callback function for wake word detection');threading.Thread(target=self.on_detected).start()
	@log_operation(component='WakeWordDetector')
	@with_error_handling(error_category=ErrorCategory.SYSTEM,error_severity=ErrorSeverity.WARNING)
	def stop(self)->bool:
		with self._lock:
			if not self.running:self.logger.debug('Wake word detector already stopped');return True
			self._stop_event.set()
			if self._detection_thread and self._detection_thread.is_alive():self._detection_thread.join(timeout=2.)
			self._cleanup_resources();self.running=False;self.logger.info('Wake word detection stopped');return True
	def _cleanup_resources(self)->None:
		if self._audio_stream:
			try:self._audio_stream.stop_stream();self._audio_stream.close()
			except Exception as e:self.logger.debug(f"Error closing audio stream: {e}")
			self._audio_stream=None
		if self._pyaudio:
			try:self._pyaudio.terminate()
			except Exception as e:self.logger.debug(f"Error terminating PyAudio: {e}")
			self._pyaudio=None
		if self._porcupine:
			try:self._porcupine.delete()
			except Exception as e:self.logger.debug(f"Error deleting Porcupine: {e}")
			self._porcupine=None
		self._clear_audio_queue()
	def _clear_audio_queue(self)->None:
		try:
			while not self._audio_queue.empty():self._audio_queue.get_nowait()
		except Exception as e:self.logger.debug(f"Error clearing audio queue: {e}")