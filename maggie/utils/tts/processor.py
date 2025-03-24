import io,os,time,threading,wave,hashlib,concurrent.futures
from typing import Dict,Any,Optional,Union,Tuple
import numpy as np,soundfile as sf
from loguru import logger
__all__=['TTSProcessor']
class TTSProcessor:
	def __init__(self,config:Dict[str,Any]):
		self.voice_model=config.get('voice_model',None);logger.info(f"TTS voice model: {self.voice_model}");self.model_path=config.get('model_path','');logger.info(f"TTS model path: {self.model_path}");self.sample_rate=config.get('sample_rate',22050);logger.info(f"TTS sample rate: {self.sample_rate} Hz");self.use_cache=config.get('use_cache',True);self.cache={};logger.info(f"TTS caching: {'enabled'if self.use_cache else'disabled'}");self.cache_dir=config.get('cache_dir','cache/tts');logger.info(f"TTS cache directory: {self.cache_dir}");self.cache_size=config.get('cache_size',100);logger.info(f"TTS cache size: {self.cache_size}");self.gpu_device=config.get('gpu_device',0);logger.info(f"TTS GPU device: {self.gpu_device}");self.gpu_acceleration=config.get('gpu_acceleration',100);logger.info(f"TTS GPU acceloration: {self.gpu_acceleration}");self.gpu_precision=config.get('gpu_precision','float16');logger.info(f"TTS GPU precision: {self.gpu_precision}");self.max_workers=config.get('max_workers',2);logger.info(f"TTS max workers: {self.max_workers}");self.voice_preprocessing=config.get('voice_preprocessing',True);logger.info(f"TTS voice preprocessing: {self.voice_preprocessing}");self.kokoro_instance=None;self.lock=threading.Lock();self.thread_pool=concurrent.futures.ThreadPoolExecutor(max_workers=2,thread_name_prefix='maggie[tts]_thread_')
		if self.use_cache and not os.path.exists(self.cache_dir):
			try:os.makedirs(self.cache_dir,exist_ok=True);logger.info(f"Created TTS cache directory: {self.cache_dir}")
			except Exception as e:logger.error(f"Failed to create TTS cache directory: {e}");self.use_cache=False
	def _init_kokoro(self)->bool:
		if self.kokoro_instance is not None:return True
		try:
			model_dir=os.path.normpath(self.model_path)
			if not os.path.isabs(model_dir):model_dir=os.path.abspath(model_dir)
			if not self.voice_model:logger.error('Voice model not specified in configuration');return False
			if not model_dir:logger.error('Model path not specified in configuration');return False
			voice_path=os.path.join(model_dir,self.voice_model);voice_path=os.path.normpath(voice_path);logger.info(f"Loading TTS voice model: {voice_path}")
			if not os.path.exists(voice_path):
				error_msg=f"TTS voice model not found: {voice_path}";logger.error(error_msg)
				try:
					from maggie.utils.service_locator import ServiceLocator;event_bus=ServiceLocator.get('event_bus')
					if event_bus:event_bus.publish('error_logged',{'source':'tts','message':f"Voice model not found: {voice_path}",'path':voice_path})
				except ImportError:pass
				logger.info(f"Attempting to download missing voice model...")
				if self._download_voice_model():
					logger.info(f"Successfully downloaded voice model")
					if os.path.exists(voice_path)and os.path.getsize(voice_path)>0:logger.info(f"Downloaded voice model verified at: {voice_path}");return self._initialize_kokoro_engine(voice_path)
					else:logger.error(f"Downloaded file verification failed at: {voice_path}")
				else:logger.error('Voice model download failed')
				return False
			import kokoro;voice_path=os.path.join(self.model_path,self.voice_model);logger.error(f"_____________self.model_path = {self.model_path}");logger.error(f"_____________self.voice_model = {self.voice_model}");logger.error(f"_____________voice_path = {voice_path}")
			if not self.voice_model:logger.error('Voice model not specified in configuration');return False
			if not self.model_path:logger.error('Model path not specified in configuration');return False
			voice_path=os.path.join(self.model_path,self.voice_model);logger.info(f"Loading TTS voice model: {voice_path}")
			if not os.path.exists(voice_path):
				error_msg=f"TTS voice model not found: {voice_path}";logger.error(error_msg)
				try:
					from maggie.utils.service_locator import ServiceLocator;event_bus=ServiceLocator.get('event_bus')
					if event_bus:event_bus.publish('error_logged',{'source':'tts','message':f"Voice model not found: {voice_path}",'path':voice_path})
				except ImportError:pass
				logger.info(f"Attempting to download missing voice model...")
				if self._download_voice_model():
					logger.info(f"Successfully downloaded voice model")
					if os.path.exists(voice_path):return self._initialize_kokoro_engine(voice_path)
				return False
			logger.debug(f"Available kokoro methods: {dir(kokoro)}")
			if hasattr(kokoro,'TTS'):self.kokoro_instance=kokoro.TTS(voice_path)
			elif hasattr(kokoro,'Model'):self.kokoro_instance=kokoro.Model.load(voice_path)
			else:self.kokoro_instance=kokoro.TTSModel(voice_path,use_cuda=self.gpu_acceleration)
			return self._initialize_kokoro_engine(voice_path)
		except Exception as e:logger.error(f"Failed to initialize Kokoro TTS engine: {e}");return False
	def _initialize_kokoro_engine(self,voice_path:str)->bool:
		try:
			import kokoro;gpu_options={}
			if self.gpu_acceleration:
				try:
					import torch
					if torch.cuda.is_available():
						gpu_name=torch.cuda.get_device_name(0)
						if'3080'in gpu_name:gpu_options={'precision':self.gpu_precision,'cuda_graphs':True,'max_batch_size':64,'mixed_precision':True,'tensor_cores':True,'stream_buffer_size':8}
				except ImportError:pass
			start_time=time.time();self.kokoro_instance=kokoro.load_tts_model(voice_path,use_cuda=self.gpu_acceleration,sample_rate=self.sample_rate,**gpu_options);load_time=time.time()-start_time;logger.info(f"Initialized Kokoro TTS with voice {self.voice_model} in {load_time:.2f}s");self._log_cuda_status()
			if self.gpu_acceleration:self._warm_up_model()
			return True
		except Exception as e:logger.error(f"Failed to initialize Kokoro TTS engine: {e}");return False
	def _warm_up_model(self)->None:
		try:logger.debug('Warming up TTS model...');_=self.kokoro_instance.synthesize('Warming up the model.');logger.debug('TTS model warm-up complete')
		except Exception as e:logger.warning(f"Failed to warm up TTS model: {e}")
	def _log_cuda_status(self)->None:
		try:
			import torch
			if torch.cuda.is_available():
				gpu_name=torch.cuda.get_device_name(0);vram_total=torch.cuda.get_device_properties(0).total_memory/1024**3;vram_allocated=torch.cuda.memory_allocated(0)/1024**3;logger.info(f"TTS using GPU acceleration on {gpu_name} with {vram_total:.2f}GB VRAM");logger.debug(f"Current VRAM usage: {vram_allocated:.2f}GB")
				if'3080'in gpu_name:logger.info('Applied RTX 3080 specific optimizations for TTS')
			else:logger.info('TTS using CPU (CUDA not available)')
		except ImportError:logger.debug('PyTorch not available for GPU detection')
	def _download_voice_model(self)->bool:
		try:
			model_dir=os.path.normpath(self.model_path)
			if not os.path.isabs(model_dir):model_dir=os.path.abspath(model_dir)
			os.makedirs(model_dir,exist_ok=True);logger.info(f"Ensuring model directory exists: {model_dir}");model_filename=self.voice_model;target_path=os.path.join(model_dir,model_filename);logger.info(f"Target path for downloaded model: {target_path}");model_url='https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt';logger.info(f"Downloading voice model from {model_url}");import requests;response=requests.get(model_url,stream=True,timeout=30);response.raise_for_status()
			if response.status_code!=200:logger.error(f"Download failed with status code: {response.status_code}");return False
			total_size=int(response.headers.get('content-length',0));downloaded=0
			with open(target_path,'wb')as f:
				for chunk in response.iter_content(chunk_size=8192):
					if chunk:
						f.write(chunk);downloaded+=len(chunk)
						if total_size>0:
							progress=downloaded/total_size*100
							if downloaded%(1024*1024)==0:logger.debug(f"Download progress: {progress:.1f}% ({downloaded/(1024*1024):.1f}MB)")
			if os.path.exists(target_path)and os.path.getsize(target_path)>0:logger.info(f"Voice model successfully downloaded to {target_path}");return True
			else:logger.error('Downloaded file is empty or does not exist');return False
		except requests.exceptions.RequestException as e:logger.error(f"Network error downloading voice model: {e}");return False
		except(IOError,OSError)as e:logger.error(f"File system error saving voice model: {e}");return False
		except Exception as e:logger.error(f"Unexpected error downloading voice model: {e}");return False
	def _get_cache_key(self,text:str)->str:cache_text=f"{self.voice_model}:{text}";return hashlib.md5(cache_text.encode('utf-8')).hexdigest()
	def _get_cached_audio(self,text:str)->Optional[np.ndarray]:
		if not self.use_cache:return None
		cache_key=self._get_cache_key(text)
		if cache_key in self.cache:logger.debug(f"TTS cache hit (memory): {text[:30]}...");return self.cache[cache_key]
		cache_path=os.path.join(self.cache_dir,f"{cache_key}.npy")
		if os.path.exists(cache_path):
			try:audio_data=np.load(cache_path);self.cache[cache_key]=audio_data;logger.debug(f"TTS cache hit (disk): {text[:30]}...");return audio_data
			except Exception as e:logger.warning(f"Failed to load cached audio: {e}")
		return None
	def _save_audio_to_cache(self,text:str,audio_data:np.ndarray)->None:
		if not self.use_cache:return
		try:
			cache_key=self._get_cache_key(text);self.cache[cache_key]=audio_data
			if len(self.cache)>self.cache_size:
				to_remove=list(self.cache.keys())[0:len(self.cache)-self.cache_size]
				for key in to_remove:self.cache.pop(key,None)
			self.thread_pool.submit(self._save_to_disk_cache,cache_key,audio_data)
		except Exception as e:logger.warning(f"Failed to save audio to cache: {e}")
	def _save_to_disk_cache(self,cache_key:str,audio_data:np.ndarray)->None:
		try:cache_path=os.path.join(self.cache_dir,f"{cache_key}.npy");np.save(cache_path,audio_data);logger.debug(f"Saved TTS output to disk cache: {cache_key}")
		except Exception as e:logger.warning(f"Failed to save audio to disk cache: {e}")
	def speak(self,text:str)->bool:
		if not text:return False
		with self.lock:
			try:
				cached_audio=self._get_cached_audio(text)
				if cached_audio is not None:logger.debug(f"Using cached audio for: {text[:30]}...");self._play_audio(cached_audio);return True
				if not self._init_kokoro():return False
				if self.voice_preprocessing:text=self._preprocess_text(text)
				start_time=time.time();audio_data=self._synthesize(text);synth_time=time.time()-start_time
				if audio_data is None:logger.error('Failed to synthesize speech');return False
				chars_per_second=len(text)/synth_time if synth_time>0 else 0;logger.debug(f"Synthesized {len(text)} chars in {synth_time:.2f}s ({chars_per_second:.1f} chars/s)");self._save_audio_to_cache(text,audio_data);self._play_audio(audio_data);return True
			except Exception as e:logger.error(f"Error in TTS: {e}");return False
	def _preprocess_text(self,text:str)->str:
		abbreviations={'Dr.':'Doctor','Mr.':'Mister','Mrs.':'Misses','Ms.':'Miss','Prof.':'Professor','e.g.':'for example','i.e.':'that is','vs.':'versus'}
		for(abbr,expansion)in abbreviations.items():text=text.replace(abbr,expansion)
		text=text.replace(' - ',', ')
		for punct in['.','!','?']:text=text.replace(f"{punct}",f"{punct} ");text=text.replace(f"{punct}  ",f"{punct} ")
		return text
	def _synthesize(self,text:str)->Optional[np.ndarray]:
		try:
			if self.gpu_acceleration:
				try:
					import torch
					if torch.cuda.is_available():torch.cuda.empty_cache()
				except ImportError:pass
			audio_data=self.kokoro_instance.synthesize(text);return np.array(audio_data)
		except Exception as e:logger.error(f"Error synthesizing speech: {e}");return None
	def _play_audio(self,audio_data:np.ndarray)->None:
		try:import pyaudio;audio_int16=np.clip(audio_data*32767,-32768,32767).astype(np.int16);p=pyaudio.PyAudio();stream=p.open(format=pyaudio.paInt16,channels=1,rate=self.sample_rate,output=True,frames_per_buffer=512,output_device_index=None);self._play_audio_chunks(stream,audio_int16);stream.stop_stream();stream.close();p.terminate()
		except ImportError as import_error:logger.error(f"Failed to import PyAudio: {import_error}");logger.error('Please install PyAudio with: pip install PyAudio==0.2.13')
		except Exception as e:logger.error(f"Error playing audio: {e}")
	def _play_audio_chunks(self,stream,audio_int16:np.ndarray,chunk_size:int=512)->None:
		for i in range(0,len(audio_int16),chunk_size):chunk=audio_int16[i:i+chunk_size].tobytes();stream.write(chunk)
	def save_to_file(self,text:str,output_path:str)->bool:
		if not text:return False
		with self.lock:
			try:
				if not self._init_kokoro():return False
				cached_audio=self._get_cached_audio(text)
				if cached_audio is not None:audio_data=cached_audio
				else:
					if self.voice_preprocessing:text=self._preprocess_text(text)
					audio_data=self._synthesize(text)
					if audio_data is None:return False
					self._save_audio_to_cache(text,audio_data)
				output_dir=os.path.dirname(os.path.abspath(output_path))
				if not os.path.exists(output_dir):os.makedirs(output_dir,exist_ok=True)
				self._save_audio_to_wav(audio_data,output_path);logger.info(f"Saved TTS audio to {output_path}");return True
			except Exception as e:logger.error(f"Error saving TTS to file: {e}");return False
	def _save_audio_to_wav(self,audio_data:np.ndarray,output_path:str)->None:
		try:sf.write(output_path,audio_data,self.sample_rate,subtype='PCM_24')
		except Exception as e:
			logger.error(f"Error saving audio with soundfile: {e}")
			with wave.open(output_path,'wb')as wf:wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(self.sample_rate);audio_int16=(audio_data*32767).astype(np.int16);wf.writeframes(audio_int16.tobytes())
	def cleanup(self)->None:
		with self.lock:
			self.kokoro_instance=None;self.cache.clear();self.thread_pool.shutdown(wait=False)
			if self.gpu_acceleration:
				try:
					import torch
					if torch.cuda.is_available():torch.cuda.empty_cache()
				except ImportError:pass
			logger.info('TTS resources cleaned up')