import io,os,time,threading,wave,hashlib,concurrent.futures
from typing import Dict,Any,Optional,Union,Tuple,List
import numpy as np,soundfile as sf
from maggie.core.state import State,StateTransition,StateAwareComponent
from maggie.core.event import EventListener,EventEmitter,EventPriority,INPUT_ACTIVATION_EVENT,INPUT_DEACTIVATION_EVENT
from maggie.utils.error_handling import safe_execute,ErrorCategory,ErrorSeverity,with_error_handling,record_error,TTSError
from maggie.utils.logging import ComponentLogger,log_operation,logging_context
from maggie.service.locator import ServiceLocator
__all__=['TTSProcessor']
class TTSProcessor(StateAwareComponent,EventListener):
	def __init__(self,config:Dict[str,Any]):
		self.state_manager=ServiceLocator.get('state_manager')
		if self.state_manager:StateAwareComponent.__init__(self,self.state_manager)
		self.event_bus=ServiceLocator.get('event_bus')
		if self.event_bus:EventListener.__init__(self,self.event_bus);EventEmitter.__init__(self,self.event_bus)
		self.config=config;self.voice_model=config.get('voice_model','af_heart.pt');self.model_path=config.get('model_path','');self.sample_rate=config.get('sample_rate',22050);self.use_cache=config.get('use_cache',True);self.cache_dir=config.get('cache_dir','cache/tts');self.cache_size=config.get('cache_size',100);self.gpu_device=config.get('gpu_device',0);self.gpu_acceleration=config.get('gpu_acceleration',True);self.gpu_precision=config.get('gpu_precision','mixed_float16');self.max_workers=config.get('max_workers',2);self.voice_preprocessing=config.get('voice_preprocessing',True);self.logger=ComponentLogger('TTSProcessor');self.kokoro_instance=None;self.cache={};self.lock=threading.RLock();self.thread_pool=concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers,thread_name_prefix='maggie_tts_thread_');self.is_paused=False
		try:
			self.resource_manager=ServiceLocator.get('resource_manager')
			if not self.resource_manager:self.logger.warning('Resource manager not found in ServiceLocator')
		except Exception as e:self.logger.warning(f"Failed to get resource manager: {e}");self.resource_manager=None
		self._log_initialization_params()
		if self.use_cache and not os.path.exists(self.cache_dir):
			try:os.makedirs(self.cache_dir,exist_ok=True);self.logger.info(f"Created TTS cache directory: {self.cache_dir}")
			except Exception as e:self.logger.error(f"Failed to create TTS cache directory: {e}");self.use_cache=False
		if self.state_manager:self._register_state_handlers()
		if self.event_bus:self._register_event_handlers()
	def _register_state_handlers(self)->None:
		for state in State:self.state_manager.register_state_handler(state,self._handle_state_change,True)
		self.state_manager.register_state_handler(State.ACTIVE,self._handle_exit_active,False);self.state_manager.register_state_handler(State.BUSY,self._handle_exit_busy,False);self.logger.debug('TTS state handlers registered')
	def _register_event_handlers(self)->None:
		event_handlers=[(INPUT_ACTIVATION_EVENT,self._handle_input_activation,EventPriority.NORMAL),(INPUT_DEACTIVATION_EVENT,self._handle_input_deactivation,EventPriority.NORMAL),('pause_tts',self._handle_pause_tts,EventPriority.HIGH),('resume_tts',self._handle_resume_tts,EventPriority.HIGH),('low_memory_warning',self._handle_low_memory,EventPriority.HIGH),('gpu_memory_warning',self._handle_gpu_memory_warning,EventPriority.HIGH)]
		for(event_type,handler,priority)in event_handlers:self.listen(event_type,handler,priority=priority)
		self.logger.debug(f"Registered {len(event_handlers)} event handlers")
	def _log_initialization_params(self)->None:self.logger.info(f"TTS voice model: {self.voice_model}");self.logger.info(f"TTS model path: {self.model_path}");self.logger.info(f"TTS sample rate: {self.sample_rate} Hz");self.logger.info(f"TTS caching: {'enabled'if self.use_cache else'disabled'}");self.logger.info(f"TTS cache directory: {self.cache_dir}");self.logger.info(f"TTS cache size: {self.cache_size}");self.logger.info(f"TTS GPU device: {self.gpu_device}");self.logger.info(f"TTS GPU acceleration: {self.gpu_acceleration}");self.logger.info(f"TTS GPU precision: {self.gpu_precision}");self.logger.info(f"TTS max workers: {self.max_workers}");self.logger.info(f"TTS voice preprocessing: {self.voice_preprocessing}")
	def _handle_state_change(self,transition:StateTransition)->None:
		if not hasattr(transition,'to_state'):return
		to_state=transition.to_state;self.logger.debug(f"TTS processor handling state change to {to_state.name}")
		if to_state==State.INIT:self._optimize_for_init_state()
		elif to_state==State.STARTUP:self._optimize_for_startup_state()
		elif to_state==State.IDLE:self._optimize_for_idle_state()
		elif to_state==State.LOADING:self._optimize_for_loading_state()
		elif to_state==State.READY:self._optimize_for_ready_state()
		elif to_state==State.ACTIVE:self._optimize_for_active_state()
		elif to_state==State.BUSY:self._optimize_for_busy_state()
		elif to_state==State.CLEANUP:self._optimize_for_cleanup_state()
		elif to_state==State.SHUTDOWN:self._optimize_for_shutdown_state()
	def _handle_exit_active(self,transition:StateTransition)->None:
		if transition.to_state==State.BUSY:self.pause()
	def _handle_exit_busy(self,transition:StateTransition)->None:
		if transition.to_state==State.READY:self.resume()
	def _handle_input_activation(self,data:Any)->None:self.pause()
	def _handle_input_deactivation(self,data:Any)->None:self.resume()
	def _handle_pause_tts(self,data:Any=None)->None:self.pause()
	def _handle_resume_tts(self,data:Any=None)->None:self.resume()
	def _handle_low_memory(self,event_data:Dict[str,Any])->None:
		memory_percent=event_data.get('percent',0);available_gb=event_data.get('available_gb',0);self.logger.warning(f"Low memory warning: {memory_percent:.1f}% used, {available_gb:.1f} GB available")
		if memory_percent>80:self.clear_cache()
	def _handle_gpu_memory_warning(self,event_data:Dict[str,Any])->None:
		gpu_percent=event_data.get('percent',0);allocated_gb=event_data.get('allocated_gb',0);self.logger.warning(f"GPU memory warning: {gpu_percent:.1f}% used, {allocated_gb:.1f} GB allocated")
		if gpu_percent>90:
			with self.lock:
				self.kokoro_instance=None
				if self.resource_manager:self.resource_manager.clear_gpu_memory()
	def _optimize_for_init_state(self)->None:
		with self.lock:self.kokoro_instance=None;self.clear_cache()
	def _optimize_for_startup_state(self)->None:pass
	def _optimize_for_idle_state(self)->None:
		with self.lock:
			self.kokoro_instance=None
			if self.resource_manager:self.resource_manager.clear_gpu_memory()
	def _optimize_for_loading_state(self)->None:
		if self.resource_manager:self.resource_manager.clear_gpu_memory()
	def _optimize_for_ready_state(self)->None:
		if self.config.get('preload_in_ready',False):self._init_kokoro()
	def _optimize_for_active_state(self)->None:
		self._init_kokoro();self.resume()
		if self.resource_manager and self.hardware_info.get('gpu',{}).get('is_rtx_3080',False):self._apply_rtx_3080_optimizations()
	def _optimize_for_busy_state(self)->None:self.pause()
	def _optimize_for_cleanup_state(self)->None:self.clear_cache()
	def _optimize_for_shutdown_state(self)->None:self.cleanup()
	@property
	def hardware_info(self)->Dict[str,Any]:
		if self.resource_manager and hasattr(self.resource_manager,'detector'):return self.resource_manager.detector.hardware_info
		return{}
	@with_error_handling(error_category=ErrorCategory.MODEL,error_severity=ErrorSeverity.ERROR)
	def _init_kokoro(self)->bool:
		if self.kokoro_instance is not None:return True
		with self.lock:
			if self.kokoro_instance is not None:return True
			with logging_context(component='TTSProcessor',operation='init_kokoro')as ctx:
				try:
					import kokoro;model_path=os.path.normpath(self.model_path)
					if not os.path.isabs(model_path):model_path=os.path.abspath(model_path)
					voice_path=os.path.join(model_path,self.voice_model);voice_path=os.path.normpath(voice_path);self.logger.info(f"Loading TTS voice model: {voice_path}")
					if not os.path.exists(voice_path):
						self.logger.error(f"TTS voice model not found: {voice_path}")
						try:
							if self.event_bus:self.event_bus.publish('error_logged',{'source':'tts','message':f"Voice model not found: {voice_path}",'path':voice_path})
						except Exception:pass
						self.logger.info(f"Attempting to download missing voice model...")
						if self._download_voice_model():
							self.logger.info(f"Successfully downloaded voice model")
							if not os.path.exists(voice_path):return False
						else:self.logger.error('Voice model download failed');return False
					gpu_options={}
					if self.gpu_acceleration:gpu_options={'precision':self.gpu_precision,'cuda_graphs':self.config.get('tts',{}).get('cuda_graphs',False),'max_batch_size':self.config.get('tts',{}).get('max_batch_size',16),'mixed_precision':self.config.get('tts',{}).get('mixed_precision',False),'tensor_cores':self.config.get('tts',{}).get('tensor_cores',False)}
					start_time=time.time()
					if hasattr(kokoro,'load'):self.kokoro_instance=kokoro.load(voice_path,use_cuda=self.gpu_acceleration,sample_rate=self.sample_rate,**gpu_options)
					elif hasattr(kokoro,'TTS'):self.kokoro_instance=kokoro.TTS(voice_path,use_cuda=self.gpu_acceleration,sample_rate=self.sample_rate,**gpu_options)
					elif hasattr(kokoro,'create_tts'):self.kokoro_instance=kokoro.create_tts(voice_path,use_cuda=self.gpu_acceleration,sample_rate=self.sample_rate,**gpu_options)
					elif hasattr(kokoro,'create_model'):self.kokoro_instance=kokoro.create_model(voice_path,use_cuda=self.gpu_acceleration,sample_rate=self.sample_rate,**gpu_options)
					else:from kokoro import TTSModel;self.kokoro_instance=TTSModel(voice_path,use_cuda=self.gpu_acceleration,sample_rate=self.sample_rate,**gpu_options)
					load_time=time.time()-start_time;self.logger.log_performance('load_model',load_time,{'model':self.voice_model,'gpu_enabled':self.gpu_acceleration});self.logger.info(f"Initialized Kokoro TTS with voice {self.voice_model} in {load_time:.2f}s")
					if self.gpu_acceleration:self._warm_up_model()
					return True
				except ImportError as e:self.logger.error(f"Failed to import kokoro: {e}");self.logger.error('Please install with: pip install git+https://github.com/hexgrad/kokoro.git');return False
				except Exception as e:self.logger.error(f"Failed to initialize Kokoro TTS engine: {e}");return False
	def _warm_up_model(self)->None:
		try:self.logger.debug('Warming up TTS model...');_=self.kokoro_instance.synthesize('Warming up the model.');self.logger.debug('TTS model warm-up complete')
		except Exception as e:self.logger.warning(f"Failed to warm up TTS model: {e}")
	@with_error_handling(error_category=ErrorCategory.NETWORK,error_severity=ErrorSeverity.WARNING)
	def _download_voice_model(self)->bool:
		try:
			model_dir=os.path.normpath(self.model_path)
			if not os.path.isabs(model_dir):model_dir=os.path.abspath(model_dir)
			os.makedirs(model_dir,exist_ok=True);model_filename=self.voice_model;target_path=os.path.join(model_dir,model_filename);model_urls=self.config.get('tts',{}).get('model_urls',['https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt','https://github.com/hexgrad/kokoro/releases/download/v0.1/af_heart.pt','https://huggingface.co/hexgrad/kokoro-voices/resolve/main/af_heart.pt'])
			for url in model_urls:
				self.logger.info(f"Downloading voice model from {url}")
				try:
					import requests;response=requests.get(url,stream=True,timeout=30)
					if response.status_code!=200:self.logger.warning(f"URL inaccessible (status: {response.status_code})");continue
					content_length=response.headers.get('Content-Length')
					if content_length and int(content_length)<1024*1024:self.logger.warning(f"URL returns undersized file");continue
					total_size=int(content_length or 0);downloaded=0
					with open(target_path,'wb')as f:
						for chunk in response.iter_content(chunk_size=8192):
							if chunk:
								f.write(chunk);downloaded+=len(chunk)
								if total_size>0 and downloaded%(1024*1024)==0:
									progress=downloaded/total_size*100;self.logger.debug(f"Download progress: {progress:.1f}%")
									if self.event_bus:self.event_bus.publish('download_progress',{'item':'voice model','percent':progress,'current_bytes':downloaded,'total_bytes':total_size})
					if os.path.exists(target_path)and os.path.getsize(target_path)>0:self.logger.info(f"Voice model successfully downloaded to {target_path}");return True
					else:self.logger.error('Downloaded file is empty or does not exist');os.remove(target_path)
				except Exception as e:self.logger.warning(f"Error downloading from {url}: {e}");continue
			self.logger.error('Failed to download voice model from any source');return False
		except Exception as e:self.logger.error(f"Error downloading voice model: {e}");return False
	def _get_cache_key(self,text:str)->str:cache_text=f"{self.voice_model}:{text}";return hashlib.md5(cache_text.encode('utf-8')).hexdigest()
	def _get_cached_audio(self,text:str)->Optional[np.ndarray]:
		if not self.use_cache:return None
		cache_key=self._get_cache_key(text)
		if cache_key in self.cache:self.logger.debug(f"TTS cache hit (memory): {text[:30]}...");return self.cache[cache_key]
		cache_path=os.path.join(self.cache_dir,f"{cache_key}.npy")
		if os.path.exists(cache_path):
			try:audio_data=np.load(cache_path);self.cache[cache_key]=audio_data;self.logger.debug(f"TTS cache hit (disk): {text[:30]}...");return audio_data
			except Exception as e:self.logger.warning(f"Failed to load cached audio: {e}")
		return None
	def _save_audio_to_cache(self,text:str,audio_data:np.ndarray)->None:
		if not self.use_cache:return
		try:
			cache_key=self._get_cache_key(text);self.cache[cache_key]=audio_data
			if len(self.cache)>self.cache_size:
				to_remove=list(self.cache.keys())[0:len(self.cache)-self.cache_size]
				for key in to_remove:self.cache.pop(key,None)
			self.thread_pool.submit(self._save_to_disk_cache,cache_key,audio_data)
		except Exception as e:self.logger.warning(f"Failed to save audio to cache: {e}")
	def _save_to_disk_cache(self,cache_key:str,audio_data:np.ndarray)->None:
		try:cache_path=os.path.join(self.cache_dir,f"{cache_key}.npy");np.save(cache_path,audio_data);self.logger.debug(f"Saved TTS output to disk cache: {cache_key}")
		except Exception as e:self.logger.warning(f"Failed to save audio to disk cache: {e}")
	def _preprocess_text(self,text:str)->str:
		if not self.voice_preprocessing:return text
		abbreviations=self.config.get('tts',{}).get('abbreviations',{'Dr.':'Doctor','Mr.':'Mister','Mrs.':'Misses','Ms.':'Miss','Prof.':'Professor','e.g.':'for example','i.e.':'that is','vs.':'versus'})
		for(abbr,expansion)in abbreviations.items():text=text.replace(abbr,expansion)
		text=text.replace(' - ',', ')
		for punct in['.','!','?']:text=text.replace(f"{punct}",f"{punct} ");text=text.replace(f"{punct}  ",f"{punct} ")
		return text
	@log_operation(component='TTSProcessor')
	@with_error_handling(error_category=ErrorCategory.PROCESSING,error_severity=ErrorSeverity.WARNING)
	def speak(self,text:str)->bool:
		if not text or self.is_paused:return False
		current_state=None
		if self.state_manager:current_state=self.state_manager.get_current_state()
		if current_state in[State.INIT,State.STARTUP,State.CLEANUP,State.SHUTDOWN]:self.logger.debug(f"Speech not allowed in {current_state.name} state");return False
		with self.lock:
			try:
				cached_audio=self._get_cached_audio(text)
				if cached_audio is not None:self.logger.debug(f"Using cached audio for: {text[:30]}...");self._play_audio(cached_audio);return True
				if not self._init_kokoro():return False
				if self.voice_preprocessing:text=self._preprocess_text(text)
				start_time=time.time()
				if self.resource_manager and self.gpu_acceleration:self.resource_manager.clear_gpu_memory()
				audio_data=None
				try:audio_data=self.kokoro_instance.synthesize(text);audio_data=np.array(audio_data)
				except Exception as e:
					self.logger.error(f"Error synthesizing speech: {e}")
					if self.event_bus:self.event_bus.publish('error_logged',{'source':'tts','message':f"Error synthesizing speech: {e}",'text':text[:50]+('...'if len(text)>50 else'')})
					return False
				synth_time=time.time()-start_time
				if audio_data is None:self.logger.error('Failed to synthesize speech');return False
				chars_per_second=len(text)/synth_time if synth_time>0 else 0;self.logger.debug(f"Synthesized {len(text)} chars in {synth_time:.2f}s ({chars_per_second:.1f} chars/s)");self._save_audio_to_cache(text,audio_data);self._play_audio(audio_data);return True
			except Exception as e:
				self.logger.error(f"Error in TTS: {e}");state_info={}
				if current_state:state_info={'state':current_state.name}
				record_error(message=f"TTS error: {e}",exception=e,category=ErrorCategory.PROCESSING,severity=ErrorSeverity.ERROR,source='TTSProcessor.speak',details={'text_length':len(text)},state_object=current_state);return False
	@with_error_handling(error_category=ErrorCategory.PROCESSING)
	def _play_audio(self,audio_data:np.ndarray)->None:
		try:
			import pyaudio;audio_int16=np.clip(audio_data*32767,-32768,32767).astype(np.int16);p=pyaudio.PyAudio();stream=p.open(format=pyaudio.paInt16,channels=1,rate=self.sample_rate,output=True,frames_per_buffer=512);chunk_size=512
			for i in range(0,len(audio_int16),chunk_size):
				if self.is_paused:break
				chunk=audio_int16[i:i+chunk_size].tobytes();stream.write(chunk)
			stream.stop_stream();stream.close();p.terminate()
		except ImportError as import_error:self.logger.error(f"Failed to import PyAudio: {import_error}");self.logger.error('Please install PyAudio with: pip install PyAudio==0.2.13')
		except Exception as e:self.logger.error(f"Error playing audio: {e}")
	@log_operation(component='TTSProcessor')
	@with_error_handling(error_category=ErrorCategory.PROCESSING)
	def save_to_file(self,text:str,output_path:str)->bool:
		if not text:return False
		with self.lock:
			try:
				cached_audio=self._get_cached_audio(text)
				if cached_audio is not None:audio_data=cached_audio
				else:
					if not self._init_kokoro():return False
					if self.voice_preprocessing:text=self._preprocess_text(text)
					if self.resource_manager and self.gpu_acceleration:self.resource_manager.clear_gpu_memory()
					try:audio_data=self.kokoro_instance.synthesize(text);audio_data=np.array(audio_data)
					except Exception as e:self.logger.error(f"Error synthesizing speech: {e}");return False
					if audio_data is None:return False
					self._save_audio_to_cache(text,audio_data)
				output_dir=os.path.dirname(os.path.abspath(output_path))
				if not os.path.exists(output_dir):os.makedirs(output_dir,exist_ok=True)
				try:sf.write(output_path,audio_data,self.sample_rate,subtype='PCM_24')
				except Exception as e:
					self.logger.error(f"Error saving audio with soundfile: {e}")
					with wave.open(output_path,'wb')as wf:wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(self.sample_rate);audio_int16=(audio_data*32767).astype(np.int16);wf.writeframes(audio_int16.tobytes())
				self.logger.info(f"Saved TTS audio to {output_path}");return True
			except Exception as e:self.logger.error(f"Error saving TTS to file: {e}");return False
	def pause(self)->None:self.is_paused=True;self.logger.debug('TTS playback paused')
	def resume(self)->None:self.is_paused=False;self.logger.debug('TTS playback resumed')
	def clear_cache(self)->None:
		with self.lock:self.cache.clear();self.logger.info('TTS cache cleared')
	def _apply_rtx_3080_optimizations(self)->None:
		try:
			import torch
			if torch.cuda.is_available():
				if hasattr(torch.backends.cuda,'matmul'):torch.backends.cuda.matmul.allow_tf32=True
				if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.allow_tf32=True
				if self.config.get('tts',{}).get('tensor_cores',False):torch.set_float32_matmul_precision('high')
				self.logger.info('Applied RTX 3080 optimizations for TTS')
		except ImportError:self.logger.debug('PyTorch not available for RTX 3080 optimizations')
		except Exception as e:self.logger.warning(f"Failed to apply RTX 3080 optimizations: {e}")
	@log_operation(component='TTSProcessor')
	def cleanup(self)->None:
		with self.lock:
			self.kokoro_instance=None;self.cache.clear();self.thread_pool.shutdown(wait=False)
			if self.resource_manager:self.resource_manager.clear_gpu_memory()
			self.logger.info('TTS resources cleaned up')