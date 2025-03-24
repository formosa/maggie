import os, io, tempfile, threading, time, wave, argparse, numpy as np, threading, time
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import numpy as np, pyaudio
from loguru import logger
from maggie.service.tts.processor import TTSProcessor

class STTProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.whisper_config = config.get('whisper', {})
        self.model_size = self.whisper_config.get('model_size', 'base')
        self.compute_type = self.whisper_config.get('compute_type', 'float16')
        self.whisper_model = None
        self.audio_stream = None
        self.pyaudio_instance = None
        self.streaming_config = config.get('whisper_streaming', {})
        self.use_streaming = self.streaming_config.get('enabled', False)
        self.streaming_server = None
        self.streaming_client = None
        self.tts_processor = None
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        self.listening = False
        self.streaming_active = False
        self.streaming_paused = False
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.on_intermediate_result = None
        self.on_final_result = None
        self._streaming_thread = None
        self._streaming_stop_event = threading.Event()
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.use_gpu = not config.get('cpu_only', False)
        self.vad_enabled = config.get('vad_enabled', True)
        self.vad_threshold = config.get('vad_threshold', .5)
        self._model_info = {'size': None, 'compute_type': None}
        logger.info(f"Enhanced speech processor initialized with model: {self.model_size}, compute type: {self.compute_type}")
        logger.info(f"Streaming mode: {'enabled' if self.use_streaming else 'disabled'}")
    
    def speak(self, text: str) -> bool:
        try:
            if self.tts_processor is None:
                from maggie.service.locator import ServiceLocator
                self.tts_processor = ServiceLocator.get('tts_processor')
                if self.tts_processor is None:
                    logger.error('TTS processor not found in ServiceLocator')
                    return False
            return self.tts_processor.speak(text)
        except Exception as e:
            logger.error(f"Error in STTProcessor.speak(): {e}")
            return False
    
    def start_listening(self) -> bool:
        with self.lock:
            if self.listening:
                logger.debug('Already listening')
                return True
                
            try:
                if self.pyaudio_instance is None:
                    self.pyaudio_instance = pyaudio.PyAudio()
                
                self.audio_stream = self.pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback
                )
                self._stop_event.clear()
                self.listening = True
                
                with self.buffer_lock:
                    self.audio_buffer = []
                
                logger.info('Audio listening started')
                return True
                
            except Exception as e:
                logger.error(f"Error starting audio listening: {e}")
                self._cleanup_audio_resources()
                return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self._stop_event.is_set():
            return in_data, pyaudio.paComplete
            
        try:
            with self.buffer_lock:
                self.audio_buffer.append(in_data)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            
        return in_data, pyaudio.paContinue
    
    def stop_listening(self) -> bool:
        with self.lock:
            if not self.listening:
                logger.debug('Not listening')
                return True
                
            try:
                self._stop_event.set()
                self._cleanup_audio_resources()
                self.listening = False
                logger.info('Audio listening stopped')
                return True
            except Exception as e:
                logger.error(f"Error stopping audio listening: {e}")
                return False
    
    def _cleanup_audio_resources(self) -> None:
        if self.audio_stream is not None:
            try:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
            
        if self.pyaudio_instance is not None:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass
            self.pyaudio_instance = None
    
    def recognize_speech(self, timeout: float = 1e1) -> Tuple[bool, str]:
        if not self.listening:
            logger.error('Cannot recognize speech - not listening')
            return False, ''
        
        try:
            start_time = time.time()
            audio_data = None
            
            while time.time() - start_time < timeout:
                with self.buffer_lock:
                    if self.audio_buffer:
                        audio_data = b''.join(self.audio_buffer)
                        self.audio_buffer = []
                        break
                time.sleep(.05)
                
            if audio_data is None:
                logger.warning('No audio data received within timeout')
                return False, ''
                
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.
            
            if self.whisper_model is None:
                self._load_whisper_model()
                
            if self.whisper_model is not None:
                result = self.whisper_model.transcribe(audio_np)
                if isinstance(result, dict) and 'text' in result:
                    return True, result['text'].strip()
                elif hasattr(result, 'text'):
                    return True, result.text.strip()
                else:
                    logger.warning('Unexpected result format from Whisper model')
                    return False, ''
            else:
                logger.error('Whisper model not initialized')
                return False, ''
                
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            return False, ''
    
    def _load_whisper_model(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            model_size = self.model_size
            compute_type = self.compute_type
            device = 'cuda' if self.use_gpu else 'cpu'
            logger.info(f"Loading Whisper model: {model_size} on {device} with {compute_type}")
            self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            return True
        except ImportError as e:
            logger.error(f"Error importing WhisperModel: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            return False
    
    def start_streaming(self, on_intermediate: Optional[Callable[[str], None]] = None, on_final: Optional[Callable[[str], None]] = None) -> bool:
        with self.lock:
            if self.streaming_active:
                logger.warning('Already streaming')
                return True
                
            if not self.listening:
                logger.error('Must start listening before streaming can begin')
                return False
                
            self.on_intermediate_result = on_intermediate
            self.on_final_result = on_final
            
            if not self._init_streaming():
                logger.error('Failed to initialize streaming components')
                return False
                
            self._streaming_stop_event.clear()
            self._streaming_thread = threading.Thread(
                target=self._streaming_process_loop,
                name='StreamingTranscriptionThread',
                daemon=True
            )
            self._streaming_thread.start()
            self.streaming_active = True
            self.streaming_paused = False
            logger.info('Real-time transcription streaming started')
            return True
    
    def stop_streaming(self) -> bool:
        with self.lock:
            if not self.streaming_active:
                logger.debug('Not streaming, nothing to stop')
                return True
                
            try:
                self._streaming_stop_event.set()
                if self._streaming_thread and self._streaming_thread.is_alive():
                    self._streaming_thread.join(timeout=2.)
                self.streaming_active = False
                self.streaming_paused = False
                logger.info('Streaming transcription stopped')
                return True
            except Exception as e:
                logger.error(f"Error stopping streaming: {e}")
                return False
    
    def pause_streaming(self) -> bool:
        with self.lock:
            if not self.streaming_active:
                logger.debug('Not streaming, nothing to pause')
                return False
                
            if self.streaming_paused:
                return True
                
            self.streaming_paused = True
            logger.info('Streaming transcription paused')
            return True
    
    def resume_streaming(self) -> bool:
        with self.lock:
            if not self.streaming_active:
                logger.debug('Not streaming, nothing to resume')
                return False
                
            if not self.streaming_paused:
                return True
                
            self.streaming_paused = False
            logger.info('Streaming transcription resumed')
            return True
    
    def _init_streaming(self) -> bool:
        if self.streaming_server is not None and self.streaming_client is not None:
            return True
            
        try:
            from maggie.service.stt.whisper_streaming import FasterWhisperASR, WhisperTimestampedASR, MLXWhisper, OnlineASRProcessor, VACOnlineASRProcessor, asr_factory
            
            model_name = self.streaming_config.get('model_name', self.model_size)
            language = self.streaming_config.get('language', 'en')
            compute_type = self.streaming_config.get('compute_type', self.compute_type)
            
            args = argparse.Namespace()
            args.backend = self.streaming_config.get('backend', 'faster-whisper')
            args.model = model_name
            args.lan = language
            args.model_cache_dir = None
            args.model_dir = None
            args.vad = self.streaming_config.get('vad_enabled', True)
            args.vac = self.streaming_config.get('vac_enabled', False)
            args.vac_chunk_size = self.streaming_config.get('vac_chunk_size', .04)
            args.min_chunk_size = self.streaming_config.get('min_chunk_size', 1.)
            args.buffer_trimming = self.streaming_config.get('buffer_trimming', 'segment')
            args.buffer_trimming_sec = self.streaming_config.get('buffer_trimming_sec', 15)
            args.task = self.streaming_config.get('task', 'transcribe')
            args.log_level = 'INFO'
            
            if self.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        if '3080' in torch.cuda.get_device_name(0):
                            logger.info('Applying RTX 3080 optimizations for whisper_streaming')
                            args.buffer_trimming_sec = 3e1
                except:
                    pass
                    
            self.streaming_asr, self.streaming_processor = asr_factory(args)
            self.streaming_server = self.streaming_asr
            self.streaming_client = self.streaming_processor
            
            logger.info(f"Whisper streaming initialized with model: {model_name}, backend: {args.backend}")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import whisper_streaming: {e}")
            logger.error('Please install whisper_streaming package: pip install git+https://github.com/ufal/whisper_streaming.git')
            return False
        except Exception as e:
            logger.error(f"Error initializing whisper_streaming: {e}")
            return False
    
    def _streaming_process_loop(self) -> None:
        logger.debug('Streaming transcription processing thread started')
        last_result = ''
        intermediate_timeout = self.streaming_config.get('result_timeout', .5)
        commit_timeout = self.streaming_config.get('commit_timeout', 2.)
        last_commit_time = time.time()
        
        try:
            while not self._streaming_stop_event.is_set():
                if self.streaming_paused:
                    time.sleep(.1)
                    continue
                    
                audio_data = None
                with self.buffer_lock:
                    if self.audio_buffer:
                        audio_data = b''.join(self.audio_buffer)
                        self.audio_buffer = []
                        
                if audio_data is not None:
                    audio_float32 = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.
                    self.streaming_processor.insert_audio_chunk(audio_float32)
                    result = self.streaming_processor.process_iter()
                    current_time = time.time()
                    
                    if result and result[2]:
                        current_result = result[2]
                        if current_result and current_result != last_result:
                            last_result = current_result
                            if self.on_intermediate_result:
                                cleaned_text = current_result.strip()
                                self.on_intermediate_result(cleaned_text)
                            last_commit_time = current_time
                            
                        if current_result and current_time - last_commit_time > commit_timeout:
                            if self.on_final_result:
                                final_text = current_result.strip()
                                self.on_final_result(final_text)
                            last_result = ''
                            last_commit_time = current_time
                            
                time.sleep(.05)
                
        except Exception as e:
            logger.error(f"Error in streaming transcription loop: {e}")
            
        finally:
            logger.debug('Streaming transcription processing thread stopped')