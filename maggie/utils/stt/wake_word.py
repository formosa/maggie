import os,threading,time,queue
from typing import Dict,Any,Optional,Callable,List,Union
import pvporcupine,pyaudio,numpy as np
from loguru import logger
__all__=['WakeWordDetector']
class WakeWordDetector:
	def __init__(self,config:Dict[str,Any]):
		logger.info('Initializing Wake Word Detection...');self.on_detected=None;self.running=False;self.access_key=config.get('access_key',None);self.sensitivity=config.get('sensitivity',.5);self.keyword=config.get('keyword','Maggie');self.keyword_path=config.get('keyword_path',None);self.cpu_threshold=config.get('cpu_threshold',5.)
		if not self.access_key:logger.error('Missing Porcupine access key in configuration');raise ValueError('Porcupine access key is required')
		if not .0<=self.sensitivity<=1.:logger.warning(f"Invalid sensitivity value: {self.sensitivity}, using default: 0.5");self.sensitivity=.5
		self._porcupine=None;self._pyaudio=None;self._audio_stream=None;self._detection_thread=None;self._stop_event=threading.Event();self._audio_queue=queue.Queue(maxsize=3);self._lock=threading.Lock();logger.info(f"Wake word detector initialized with sensitivity: {self.sensitivity}")
	def start(self)->bool:
		with self._lock:
			if self.running:logger.warning('Wake word detector already running');return True
			try:
				keywords=[];sensitivities=[]
				if self.keyword_path and os.path.exists(self.keyword_path):keywords=[self.keyword_path];sensitivities=[self.sensitivity];logger.info(f"Using custom keyword model: {self.keyword_path}")
				else:default_keyword='computer';keywords=[default_keyword];sensitivities=[self.sensitivity];logger.warning(f"Custom keyword not found, using default: '{default_keyword}'");logger.warning(f"To use 'maggie', create a custom keyword at console.picovoice.ai")
				try:self._porcupine=pvporcupine.create(access_key=self.access_key,keywords=keywords,sensitivities=sensitivities)
				except ValueError as e:logger.error(f"Keyword error: {e}");logger.info("Falling back to 'computer' keyword");self._porcupine=pvporcupine.create(access_key=self.access_key,keywords=['computer'],sensitivities=[self.sensitivity])
				self._pyaudio=pyaudio.PyAudio();self._audio_stream=self._pyaudio.open(rate=self._porcupine.sample_rate,channels=1,format=pyaudio.paInt16,input=True,frames_per_buffer=self._porcupine.frame_length,stream_callback=self._audio_callback);self._stop_event.clear();self._detection_thread=threading.Thread(target=self._detection_loop,name='WakeWordThread',daemon=True);self._detection_thread.start();self.running=True;logger.info('Wake word detection started');return True
			except pvporcupine.PorcupineError as e:logger.error(f"Porcupine error: {e}");self._cleanup_resources();return False
			except pyaudio.PyAudioError as e:logger.error(f"PyAudio error: {e}");self._cleanup_resources();return False
			except Exception as e:logger.error(f"Error starting wake word detection: {e}");self._cleanup_resources();return False
	def stop(self)->bool:
		with self._lock:
			if not self.running:logger.debug('Wake word detector already stopped');return True
			try:
				self._stop_event.set()
				if self._detection_thread and self._detection_thread.is_alive():self._detection_thread.join(timeout=2.)
				self._cleanup_resources();self.running=False;logger.info('Wake word detection stopped');return True
			except Exception as e:logger.error(f"Error stopping wake word detection: {e}");return False
	def _audio_callback(self,in_data,frame_count,time_info,status):
		if self._stop_event.is_set():return in_data,pyaudio.paComplete
		try:self._audio_queue.put(in_data,block=False)
		except queue.Full:pass
		return in_data,pyaudio.paContinue
	def _detection_loop(self)->None:
		logger.debug('Wake word detection thread started');import psutil;process=psutil.Process()
		if hasattr(process,'nice'):
			try:process.nice(10)
			except:pass
		while not self._stop_event.is_set():
			try:
				audio_data=self._audio_queue.get(timeout=.1);pcm=np.frombuffer(audio_data,dtype=np.int16);keyword_index=self._porcupine.process(pcm)
				if keyword_index>=0:
					logger.info('Wake word detected!')
					if self.on_detected:threading.Thread(target=self.on_detected).start()
				current_cpu=process.cpu_percent(interval=None)
				if current_cpu>self.cpu_threshold:time.sleep(.01)
			except queue.Empty:continue
			except Exception as e:logger.error(f"Error in wake word detection loop: {e}");time.sleep(.1)
		logger.debug('Wake word detection thread stopped')
	def _cleanup_resources(self)->None:
		if self._audio_stream:
			try:self._audio_stream.stop_stream();self._audio_stream.close()
			except:pass
			self._audio_stream=None
		if self._pyaudio:
			try:self._pyaudio.terminate()
			except:pass
			self._pyaudio=None
		if self._porcupine:
			try:self._porcupine.delete()
			except:pass
			self._porcupine=None
		while not self._audio_queue.empty():
			try:self._audio_queue.get_nowait()
			except:pass