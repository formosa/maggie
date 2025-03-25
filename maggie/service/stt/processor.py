import os,io,threading,time
from typing import Dict,Any,Optional,Tuple,List,Union,Callable
import numpy as np,pyaudio
from maggie.core.state import State,StateTransition,StateAwareComponent
from maggie.core.event import EventListener,EventEmitter,EventPriority
from maggie.utils.error_handling import safe_execute,retry_operation,ErrorCategory,ErrorSeverity,with_error_handling,record_error,STTError
from maggie.utils.logging import ComponentLogger,log_operation,logging_context
from maggie.service.locator import ServiceLocator
class STTProcessor(StateAwareComponent,EventListener):
	def __init__(self,config:Dict[str,Any]):
		self.state_manager=ServiceLocator.get('state_manager')
		if self.state_manager:StateAwareComponent.__init__(self,self.state_manager)
		self.event_bus=ServiceLocator.get('event_bus')
		if self.event_bus:EventListener.__init__(self,self.event_bus)
		self.config=config;self.whisper_config=config.get('whisper',{});self.model_size=self.whisper_config.get('model_size','base');self.compute_type=self.whisper_config.get('compute_type','float16');self.streaming_config=config.get('whisper_streaming',{});self.use_streaming=self.streaming_config.get('enabled',True);self.sample_rate=self.whisper_config.get('sample_rate',16000);self.chunk_size=self.config.get('chunk_size',1024);self.channels=1;self.whisper_model=None;self.tts_processor=None;self.audio_stream=None;self.pyaudio_instance=None;self.streaming_server=None;self.streaming_client=None;self.streaming_active=False;self.streaming_paused=False;self.listening=False;self.on_intermediate_result=None;self.on_final_result=None;self._streaming_thread=None;self._streaming_stop_event=threading.Event();self._stop_event=threading.Event();self.lock=threading.RLock();self.buffer_lock=threading.RLock();self.audio_buffer=[];self.use_gpu=not config.get('cpu_only',False);self.vad_enabled=config.get('vad_enabled',True);self.vad_threshold=config.get('vad_threshold',.5)
		try:self.resource_manager=ServiceLocator.get('resource_manager')
		except Exception:self.resource_manager=None
		self._model_info={'size':None,'compute_type':None};self.logger=ComponentLogger('STTProcessor')
		if self.state_manager:self._register_state_handlers()
		if self.event_bus:self._register_event_handlers()
		self.logger.info(f"Speech processor initialized with model: {self.model_size}, compute type: {self.compute_type}");self.logger.info(f"Streaming mode: {'enabled'if self.use_streaming else'disabled'}")
	def _register_state_handlers(self)->None:self.state_manager.register_state_handler(State.INIT,self.on_enter_init,True);self.state_manager.register_state_handler(State.STARTUP,self.on_enter_startup,True);self.state_manager.register_state_handler(State.IDLE,self.on_enter_idle,True);self.state_manager.register_state_handler(State.LOADING,self.on_enter_loading,True);self.state_manager.register_state_handler(State.READY,self.on_enter_ready,True);self.state_manager.register_state_handler(State.ACTIVE,self.on_enter_active,True);self.state_manager.register_state_handler(State.BUSY,self.on_enter_busy,True);self.state_manager.register_state_handler(State.CLEANUP,self.on_enter_cleanup,True);self.state_manager.register_state_handler(State.SHUTDOWN,self.on_enter_shutdown,True);self.state_manager.register_state_handler(State.ACTIVE,self.on_exit_active,False);self.state_manager.register_state_handler(State.BUSY,self.on_exit_busy,False);self.logger.debug('STT state handlers registered')
	def _register_event_handlers(self)->None:
		event_handlers=[('input_activation',self._handle_input_activation,EventPriority.NORMAL),('input_deactivation',self._handle_input_deactivation,EventPriority.NORMAL),('pause_transcription',self._handle_pause_transcription,EventPriority.NORMAL),('resume_transcription',self._handle_resume_transcription,EventPriority.NORMAL),('wake_word_detected',self._handle_wake_word_detected,EventPriority.HIGH)]
		for(event_type,handler,priority)in event_handlers:self.listen(event_type,handler,priority=priority)
		self.logger.debug(f"Registered {len(event_handlers)} STT event handlers")
	def on_enter_init(self,transition:StateTransition)->None:self._cleanup_audio_resources();self.logger.debug('STT processor reset in INIT state')
	def on_enter_startup(self,transition:StateTransition)->None:
		if self.resource_manager:self.resource_manager.preallocate_for_state(State.STARTUP)
		if self.config.get('preload_in_startup',False):self._load_whisper_model()
	def on_enter_idle(self,transition:StateTransition)->None:
		self.stop_listening();self.stop_streaming();self.whisper_model=None
		if self.resource_manager:self.resource_manager.clear_gpu_memory()
	def on_enter_loading(self,transition:StateTransition)->None:
		if self.resource_manager:self.resource_manager.preallocate_for_state(State.LOADING)
		if self.resource_manager:self.resource_manager.clear_gpu_memory()
	def on_enter_ready(self,transition:StateTransition)->None:
		self.start_listening()
		if self.use_streaming and self.config.get('auto_start_streaming',True):self.start_streaming(on_intermediate=self._on_intermediate_transcription,on_final=self._on_final_transcription)
	def on_enter_active(self,transition:StateTransition)->None:
		if not self.listening:self.start_listening()
		if self.use_streaming and not self.streaming_active:self.start_streaming(on_intermediate=self._on_intermediate_transcription,on_final=self._on_final_transcription)
		elif self.streaming_paused:self.resume_streaming()
	def on_enter_busy(self,transition:StateTransition)->None:
		if self.streaming_active and not self.streaming_paused:self.pause_streaming()
	def on_enter_cleanup(self,transition:StateTransition)->None:
		self.stop_streaming();self.stop_listening();self.whisper_model=None
		if self.resource_manager:self.resource_manager.reduce_memory_usage()
	def on_enter_shutdown(self,transition:StateTransition)->None:self._cleanup_audio_resources();self.stop_streaming();self.whisper_model=None
	def on_exit_active(self,transition:StateTransition)->None:
		if transition.to_state==State.BUSY:self.pause_streaming()
	def on_exit_busy(self,transition:StateTransition)->None:
		if transition.to_state==State.READY:
			if self.streaming_paused:self.resume_streaming()
	def _handle_input_activation(self,data:Any)->None:self.pause_streaming()
	def _handle_input_deactivation(self,data:Any)->None:self.resume_streaming()
	def _handle_pause_transcription(self,data:Any=None)->None:self.pause_streaming()
	def _handle_resume_transcription(self,data:Any=None)->None:self.resume_streaming()
	def _handle_wake_word_detected(self,data:Any=None)->None:
		current_state=self.state_manager.get_current_state()if self.state_manager else None
		if current_state==State.IDLE:self.state_manager.transition_to(State.READY,'wake_word_detected')
		elif current_state==State.READY:self.state_manager.transition_to(State.ACTIVE,'wake_word_detected')
	def _audio_callback(self,in_data,frame_count,time_info,status):
		if self._stop_event.is_set():return in_data,pyaudio.paComplete
		try:
			with self.buffer_lock:self.audio_buffer.append(in_data)
		except Exception as e:self.logger.error(f"Error in audio callback: {e}")
		return in_data,pyaudio.paContinue
	def _get_audio_data_with_timeout(self,timeout:float)->Optional[bytes]:
		start_time=time.time();audio_data=None
		while time.time()-start_time<timeout:
			with self.buffer_lock:
				if self.audio_buffer:audio_data=b''.join(self.audio_buffer);self.audio_buffer=[];break
			time.sleep(.05)
		return audio_data
	def _convert_audio_to_numpy(self,audio_data:bytes)->np.ndarray:return np.frombuffer(audio_data,dtype=np.int16).astype(np.float32)/32768.
	def _extract_text_from_result(self,result:Any)->str:
		if isinstance(result,dict)and'text'in result:return result['text'].strip()
		elif hasattr(result,'text'):return result.text.strip()
		else:self.logger.warning('Unexpected result format from Whisper model');return''
	def _on_intermediate_transcription(self,text:str)->None:
		if self.event_bus:self.event_bus.publish('intermediate_transcription',text)
	def _on_final_transcription(self,text:str)->None:
		if self.event_bus:self.event_bus.publish('final_transcription',text)
	@log_operation(component='STTProcessor')
	@with_error_handling(error_category=ErrorCategory.SYSTEM)
	def start_listening(self)->bool:
		with self.lock:
			if self.listening:self.logger.debug('Already listening');return True
			try:
				if self.pyaudio_instance is None:self.pyaudio_instance=pyaudio.PyAudio()
				self.audio_stream=self.pyaudio_instance.open(format=pyaudio.paInt16,channels=self.channels,rate=self.sample_rate,input=True,frames_per_buffer=self.chunk_size,stream_callback=self._audio_callback);self._stop_event.clear();self.listening=True
				with self.buffer_lock:self.audio_buffer=[]
				self.logger.info('Audio listening started');return True
			except Exception as e:self.logger.error(f"Error starting audio listening: {e}");self._cleanup_audio_resources();raise STTError(f"Could not start listening: {e}")
	@log_operation(component='STTProcessor')
	@with_error_handling(error_category=ErrorCategory.SYSTEM)
	def stop_listening(self)->bool:
		with self.lock:
			if not self.listening:self.logger.debug('Not listening');return True
			try:self._stop_event.set();self._cleanup_audio_resources();self.listening=False;self.logger.info('Audio listening stopped');return True
			except Exception as e:self.logger.error(f"Error stopping audio listening: {e}");raise STTError(f"Could not stop listening: {e}")
	def _cleanup_audio_resources(self)->None:
		if self.audio_stream is not None:
			try:
				if self.audio_stream.is_active():self.audio_stream.stop_stream()
				self.audio_stream.close()
			except:pass
			self.audio_stream=None
		if self.pyaudio_instance is not None:
			try:self.pyaudio_instance.terminate()
			except:pass
			self.pyaudio_instance=None
	@retry_operation(max_attempts=2,retry_delay=.5)
	@log_operation(component='STTProcessor')
	@with_error_handling(error_category=ErrorCategory.PROCESSING,error_severity=ErrorSeverity.ERROR)
	def recognize_speech(self,timeout:float=1e1)->Tuple[bool,str]:
		if not self.listening:self.logger.error('Cannot recognize speech - not listening');return False,''
		try:
			audio_data=self._get_audio_data_with_timeout(timeout)
			if audio_data is None:self.logger.warning('No audio data received within timeout');return False,''
			audio_np=self._convert_audio_to_numpy(audio_data)
			if self.whisper_model is None:self._load_whisper_model()
			if self.whisper_model is None:self.logger.error('Whisper model not initialized');return False,''
			start_time=time.time();result=self.whisper_model.transcribe(audio_np);elapsed=time.time()-start_time;text=self._extract_text_from_result(result);self.logger.log_performance('recognize_speech',elapsed,{'text_length':len(text)if text else 0});return True,text
		except Exception as e:self.logger.error(f"Error recognizing speech: {e}");raise STTError(f"Speech recognition failed: {e}")
	@retry_operation(max_attempts=2,allowed_exceptions=(ImportError,RuntimeError))
	@with_error_handling(error_category=ErrorCategory.MODEL,error_severity=ErrorSeverity.ERROR)
	def _load_whisper_model(self)->bool:
		try:
			from faster_whisper import WhisperModel
			with logging_context(component='STTProcessor',operation='load_model')as ctx:model_size=self.model_size;compute_type=self.compute_type;device='cuda'if self.use_gpu else'cpu';self.logger.info(f"Loading Whisper model: {model_size} on {device} with {compute_type}");self._apply_hardware_optimizations();start_time=time.time();self.whisper_model=WhisperModel(model_size,device=device,compute_type=compute_type);elapsed=time.time()-start_time;self._model_info={'size':model_size,'compute_type':compute_type};self.logger.log_performance('load_model',elapsed,{'model_size':model_size,'device':device,'compute_type':compute_type});return True
		except ImportError as e:self.logger.error(f"Error importing WhisperModel: {e}");self.logger.error('Please install faster-whisper with: pip install faster-whisper');return False
		except Exception as e:self.logger.error(f"Error loading Whisper model: {e}");return False
	def _apply_hardware_optimizations(self)->None:
		if not self.resource_manager:return
		hardware_info=self.resource_manager.detector.hardware_info
		if hardware_info['gpu'].get('is_rtx_3080',False):
			try:
				import torch
				if torch.cuda.is_available():
					self.compute_type='float16'
					if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.allow_tf32=True
					torch.backends.cudnn.benchmark=True;self.logger.info('Applied RTX 3080 optimizations for Whisper')
			except ImportError:pass
		if hardware_info['cpu'].get('is_ryzen_9_5900x',False):
			self.chunk_size=512
			try:
				import psutil
				if os.name=='nt':process=psutil.Process();process.cpu_affinity([0,1,2,3,4,5,6,7])
			except ImportError:pass
	@log_operation(component='STTProcessor')
	@with_error_handling(error_category=ErrorCategory.SYSTEM,error_severity=ErrorSeverity.ERROR)
	def start_streaming(self,on_intermediate:Optional[Callable[[str],None]]=None,on_final:Optional[Callable[[str],None]]=None)->bool:
		with self.lock:
			if self.streaming_active:self.logger.warning('Already streaming');return True
			if not self.listening:self.logger.error('Must start listening before streaming can begin');return False
			self.on_intermediate_result=on_intermediate;self.on_final_result=on_final
			if not self._init_streaming():self.logger.error('Failed to initialize streaming components');return False
			self._streaming_stop_event.clear();self._streaming_thread=threading.Thread(target=self._streaming_process_loop,name='StreamingTranscriptionThread',daemon=True);self._streaming_thread.start();self.streaming_active=True;self.streaming_paused=False;self.logger.info('Real-time transcription streaming started');return True
	@log_operation(component='STTProcessor')
	@with_error_handling(error_category=ErrorCategory.SYSTEM,error_severity=ErrorSeverity.ERROR)
	def stop_streaming(self)->bool:
		with self.lock:
			if not self.streaming_active:self.logger.debug('Not streaming, nothing to stop');return True
			try:
				self._streaming_stop_event.set()
				if self._streaming_thread and self._streaming_thread.is_alive():self._streaming_thread.join(timeout=2.)
				self.streaming_active=False;self.streaming_paused=False;self.logger.info('Streaming transcription stopped');return True
			except Exception as e:self.logger.error(f"Error stopping streaming: {e}");raise STTError(f"Failed to stop streaming: {e}")
	@log_operation(component='STTProcessor')
	def pause_streaming(self)->bool:
		with self.lock:
			if not self.streaming_active:self.logger.debug('Not streaming, nothing to pause');return False
			if self.streaming_paused:return True
			self.streaming_paused=True;self.logger.info('Streaming transcription paused');return True
	@log_operation(component='STTProcessor')
	def resume_streaming(self)->bool:
		with self.lock:
			if not self.streaming_active:self.logger.debug('Not streaming, nothing to resume');return False
			if not self.streaming_paused:return True
			self.streaming_paused=False;self.logger.info('Streaming transcription resumed');return True
	@with_error_handling(error_category=ErrorCategory.SYSTEM,error_severity=ErrorSeverity.ERROR)
	def _init_streaming(self)->bool:
		if self.streaming_server is not None and self.streaming_client is not None:return True
		try:from maggie.service.stt.whisper_streaming import FasterWhisperASR,WhisperTimestampedASR,MLXWhisper,OnlineASRProcessor,VACOnlineASRProcessor,asr_factory;model_name=self.streaming_config.get('model_name',self.model_size);language=self.streaming_config.get('language','en');compute_type=self.streaming_config.get('compute_type',self.compute_type);import argparse;args=argparse.Namespace();args.backend=self.streaming_config.get('backend','faster-whisper');args.model=model_name;args.lan=language;args.model_cache_dir=None;args.model_dir=None;args.vad=self.streaming_config.get('vad_enabled',True);args.vac=self.streaming_config.get('vac_enabled',False);args.vac_chunk_size=self.streaming_config.get('vac_chunk_size',.04);args.min_chunk_size=self.streaming_config.get('min_chunk_size',1.);args.buffer_trimming=self.streaming_config.get('buffer_trimming','segment');args.buffer_trimming_sec=self.streaming_config.get('buffer_trimming_sec',15);args.task=self.streaming_config.get('task','transcribe');args.log_level='INFO';self.streaming_asr,self.streaming_processor=asr_factory(args);self.streaming_server=self.streaming_asr;self.streaming_client=self.streaming_processor;self.logger.info(f"Whisper streaming initialized with model: {model_name}, backend: {args.backend}");return True
		except ImportError as e:self.logger.error(f"Failed to import whisper_streaming: {e}");self.logger.error('Please install whisper_streaming package: pip install git+https://github.com/ufal/whisper_streaming.git');return False
		except Exception as e:self.logger.error(f"Error initializing whisper_streaming: {e}");return False
	def _streaming_process_loop(self)->None:
		self.logger.debug('Streaming transcription processing thread started');last_result='';intermediate_timeout=self.streaming_config.get('result_timeout',.5);commit_timeout=self.streaming_config.get('commit_timeout',2.);last_commit_time=time.time()
		try:
			while not self._streaming_stop_event.is_set():
				if self.streaming_paused:time.sleep(.1);continue
				audio_data=None
				with self.buffer_lock:
					if self.audio_buffer:audio_data=b''.join(self.audio_buffer);self.audio_buffer=[]
				if audio_data is not None:
					audio_float32=np.frombuffer(audio_data,dtype=np.int16).astype(np.float32)/32768.;self.streaming_processor.insert_audio_chunk(audio_float32);result=self.streaming_processor.process_iter();current_time=time.time()
					if result and result[2]:
						current_result=result[2]
						if current_result and current_result!=last_result:
							last_result=current_result
							if self.on_intermediate_result:cleaned_text=current_result.strip();self.on_intermediate_result(cleaned_text)
							last_commit_time=current_time
						if current_result and current_time-last_commit_time>commit_timeout:
							if self.on_final_result:final_text=current_result.strip();self.on_final_result(final_text)
							last_result='';last_commit_time=current_time
				time.sleep(.05)
		except Exception as e:self.logger.error(f"Error in streaming transcription loop: {e}")
		finally:self.logger.debug('Streaming transcription processing thread stopped')