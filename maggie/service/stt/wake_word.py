import os,threading,time,queue
from typing import Dict,Any,Optional,Callable,List,Union
import pvporcupine,pyaudio,numpy as np
from maggie.utils.logging import ComponentLogger
from maggie.utils.error_handling import safe_execute,with_error_handling,ErrorCategory,ErrorSeverity,record_error
from maggie.service.locator import ServiceLocator
__all__=['WakeWordDetector']
class WakeWordDetector:
	def __init__(self,config:Dict[str,Any]):
		self.logger=ComponentLogger('WakeWordDetector');self.logger.info('Initializing Wake Word Detection...');self.config=config;self.access_key=config.get('access_key',None);self.sensitivity=config.get('sensitivity',.5);self.keyword=config.get('keyword','Maggie');self.keyword_path=config.get('keyword_path',None);self.cpu_threshold=config.get('cpu_threshold',5.);self.on_detected=None;self.running=False
		try:
			self.event_bus=ServiceLocator.get('event_bus')
			if not self.event_bus:self.logger.warning('Event bus not found in ServiceLocator, falling back to callback mode')
		except Exception as e:self.logger.warning(f"Error getting event bus: {e}");self.event_bus=None
		self._validate_config();self._porcupine=None;self._pyaudio=None;self._audio_stream=None;self._detection_thread=None;self._stop_event=threading.Event();self._audio_queue=queue.Queue(maxsize=3);self._lock=threading.RLock();self.logger.info(f"Wake word detector initialized with sensitivity: {self.sensitivity}")
	def _validate_config(self)->None:
		if not self.access_key:self.logger.error('Missing Porcupine access key in configuration');raise ValueError('Porcupine access key is required')
		if not .0<=self.sensitivity<=1.:self.logger.warning(f"Invalid sensitivity value: {self.sensitivity}, using default: 0.5");self.sensitivity=.5
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
		try:import psutil;process=psutil.Process()
		except ImportError:process=None;self.logger.warning('psutil not available, CPU monitoring disabled')
		while not self._stop_event.is_set():
			try:
				audio_data=self._audio_queue.get(timeout=.1);pcm=np.frombuffer(audio_data,dtype=np.int16);keyword_index=self._porcupine.process(pcm)
				if keyword_index>=0:self._handle_wake_word_detected()
				if process:
					current_cpu=process.cpu_percent(interval=None)
					if current_cpu>self.cpu_threshold:time.sleep(.01)
			except queue.Empty:continue
			except Exception as e:self.logger.error(f"Error in wake word detection loop: {e}");time.sleep(.1)
		self.logger.debug('Wake word detection thread stopped')
	def _handle_wake_word_detected(self)->None:
		self.logger.info('Wake word detected!')
		if self.event_bus:self.logger.debug('Publishing wake_word_detected event');self.event_bus.publish('wake_word_detected')
		elif self.on_detected:self.logger.debug('Using callback function for wake word detection');threading.Thread(target=self.on_detected).start()
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