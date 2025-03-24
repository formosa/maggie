import os,yaml,json,shutil,time
from typing import Dict,Any,Optional,List,Tuple,Set
from loguru import logger
__all__=['ConfigManager']
class ConfigManager:
	def __init__(self,config_path:str='config.yaml',backup_dir:str='config_backups'):self.config_path=config_path;self.backup_dir=backup_dir;self.config={};self.validation_errors=[];self.validation_warnings=[];self.default_config=self._create_default_config();os.makedirs(self.backup_dir,exist_ok=True)
	def _create_default_config(self)->Dict[str,Any]:return{'inactivity_timeout':60,'stt':{'whisper':{'model_size':'base','compute_type':'float16','model_path':'\\maggie\\models\\stt\\whisper','sample_rate':16000},'whisper_streaming':{'enabled':True,'model_name':'base','language':'en','compute_type':'float16','result_timeout':.5,'commit_timeout':2.,'auto_submit':False,'buffer_size_seconds':3e1},'wake_word':{'engine':'porcupine','access_key':'','sensitivity':.5,'keyword':'maggie','keyword_path':'\\maggie\\models\\stt\\porcupine\\Hey-Maggie_en_windows_v3_0_0.ppn','cpu_threshold':5.}},'tts':{'voice_model':'af_heart.pt','model_path':'\\maggie\\models\\tts','sample_rate':22050,'use_cache':True,'cache_dir':'\\maggie\\cache\\tts','cache_size':200,'gpu_device':0,'gpu_acceleration':True,'gpu_precision':'mixed_float16','max_workers':4,'voice_preprocessing':True},'llm':{'model_path':'maggie\\models\\llm\\mistral-7b-instruct-v0.3-GPTQ-4bit','model_type':'mistral','gpu_layers':32,'gpu_layer_auto_adjust':True},'logging':{'path':'logs','console_level':'INFO','file_level':'DEBUG'},'extensions':{'recipe_creator':{'enabled':True,'template_path':'templates\\recipe_template.docx','output_dir':'recipes'}},'cpu':{'max_threads':8,'thread_timeout':30},'memory':{'max_percent':75,'model_unload_threshold':85},'gpu':{'max_percent':90,'model_unload_threshold':95}}
	def load(self)->Dict[str,Any]:
		if os.path.exists(self.config_path):
			try:
				with open(self.config_path,'r')as file:self.config=yaml.safe_load(file)or{}
				logger.info(f"Configuration loaded from {self.config_path}");self._create_backup('loaded')
			except yaml.YAMLError as yaml_error:logger.error(f"YAML error in configuration: {yaml_error}");self._attempt_config_recovery(yaml_error)
			except IOError as io_error:logger.error(f"IO error reading configuration: {io_error}");self._attempt_config_recovery(io_error)
		else:logger.info(f"Configuration file {self.config_path} not found, creating with defaults");self.config=self.default_config;self.save()
		self._merge_with_defaults();self.validate();return self.config
	def _attempt_config_recovery(self,error:Exception)->None:
		backup_path=self._find_latest_backup()
		if backup_path:
			logger.info(f"Attempting to recover from backup: {backup_path}")
			try:
				with open(backup_path,'r')as file:self.config=yaml.safe_load(file)or{}
				logger.info(f"Configuration recovered from backup: {backup_path}")
			except Exception as recover_error:logger.error(f"Failed to recover from backup: {recover_error}");self.config=self.default_config;logger.info('Using default configuration')
		else:self.config=self.default_config;logger.info('Using default configuration')
	def _merge_with_defaults(self)->None:self.config=self._deep_merge(self.default_config.copy(),self.config)
	def _deep_merge(self,default_dict:Dict,user_dict:Dict)->Dict:
		result=default_dict.copy()
		for(key,value)in user_dict.items():
			if key in result and isinstance(result[key],dict)and isinstance(value,dict):result[key]=self._deep_merge(result[key],value)
			else:result[key]=value
		return result
	def save(self)->bool:
		try:
			os.makedirs(os.path.dirname(os.path.abspath(self.config_path)),exist_ok=True);self._create_backup('before_save')
			with open(self.config_path,'w')as file:file.write('# Maggie AI Assistant Configuration\n');file.write('# Optimized for AMD Ryzen 9 5900X and NVIDIA GeForce RTX 3080\n');file.write('# System configuration for performance and resource management\n\n');yaml.dump(self.config,file,default_flow_style=False,sort_keys=False)
			logger.info(f"Configuration saved to {self.config_path}");self._create_backup('after_save');return True
		except IOError as io_error:logger.error(f"Error saving configuration: {io_error}");return False
		except yaml.YAMLError as yaml_error:logger.error(f"YAML error in configuration: {yaml_error}");return False
		except Exception as general_error:logger.error(f"Unexpected error saving configuration: {general_error}");return False
	def _create_backup(self,reason:str)->Optional[str]:
		try:
			timestamp=time.strftime('%Y%m%d_%H%M%S');backup_path=os.path.join(self.backup_dir,f"config_{timestamp}_{reason}.yaml");os.makedirs(self.backup_dir,exist_ok=True)
			with open(backup_path,'w')as file:yaml.dump(self.config,file,default_flow_style=False)
			logger.debug(f"Configuration backup created: {backup_path}");self._cleanup_old_backups();return backup_path
		except IOError as io_error:logger.error(f"IO error creating configuration backup: {io_error}");return None
		except Exception as general_error:logger.error(f"Error creating configuration backup: {general_error}");return None
	def _find_latest_backup(self)->Optional[str]:
		try:
			if not os.path.exists(self.backup_dir):return None
			backup_files=[os.path.join(self.backup_dir,f)for f in os.listdir(self.backup_dir)if f.startswith('config_')and f.endswith('.yaml')]
			if not backup_files:return None
			backup_files.sort(key=os.path.getmtime,reverse=True);return backup_files[0]
		except IOError as io_error:logger.error(f"IO error finding latest backup: {io_error}");return None
		except Exception as general_error:logger.error(f"Error finding latest backup: {general_error}");return None
	def _cleanup_old_backups(self,keep:int=10)->None:
		try:
			if not os.path.exists(self.backup_dir):return
			backup_files=[os.path.join(self.backup_dir,f)for f in os.listdir(self.backup_dir)if f.startswith('config_')and f.endswith('.yaml')]
			if len(backup_files)<=keep:return
			backup_files.sort(key=os.path.getmtime,reverse=True)
			for old_backup in backup_files[keep:]:os.remove(old_backup);logger.debug(f"Removed old backup: {old_backup}")
		except IOError as io_error:logger.error(f"IO error cleaning up old backups: {io_error}")
		except Exception as general_error:logger.error(f"Error cleaning up old backups: {general_error}")
	def validate(self)->bool:
		self.validation_errors=[];self.validation_warnings=[];self._validate_required_params();self._validate_paths();self._validate_value_ranges()
		for error in self.validation_errors:logger.error(f"Configuration error: {error}")
		for warning in self.validation_warnings:logger.warning(f"Configuration warning: {warning}")
		return len(self.validation_errors)==0
	def _validate_required_params(self)->None:
		required_params=[('stt.wake_word.access_key','Picovoice access key'),('llm.model_path','LLM model path'),('tts.voice_model','TTS voice model')]
		for(param_path,param_name)in required_params:
			value=self._get_nested_value(self.config,param_path)
			if value is None:self.validation_errors.append(f"Missing required configuration: {param_name} ({param_path})")
			elif isinstance(value,str)and not value:self.validation_errors.append(f"Empty required configuration: {param_name} ({param_path})")
	def _get_nested_value(self,config:Dict[str,Any],path:str)->Any:
		parts=path.split('.');current=config
		for part in parts:
			if not isinstance(current,dict)or part not in current:return None
			current=current[part]
		return current
	def _validate_paths(self)->None:
		model_paths=[(self._get_nested_value(self.config,'llm.model_path'),'LLM model directory',True),(os.path.join(self._get_nested_value(self.config,'tts.model_path')or'\\maggie\\models\\tts',self._get_nested_value(self.config,'tts.voice_model')or''),'TTS voice model file',False),(self._get_nested_value(self.config,'stt.whisper.model_path'),'Whisper model directory',True),(self._get_nested_value(self.config,'stt.wake_word.keyword_path'),'Wake word model file',False)]
		for(path,name,is_dir)in model_paths:
			if not path:continue
			norm_path=os.path.normpath(path)
			if not os.path.exists(norm_path):
				self.validation_warnings.append(f"{name} path does not exist: {norm_path}")
				if is_dir:
					try:os.makedirs(norm_path,exist_ok=True);logger.info(f"Created directory for {name}: {norm_path}")
					except IOError as io_error:self.validation_warnings.append(f"Could not create directory for {name}: {io_error}")
					except Exception as general_error:self.validation_warnings.append(f"Could not create directory for {name}: {general_error}")
			elif is_dir and not os.path.isdir(norm_path):self.validation_errors.append(f"{name} path is not a directory: {norm_path}")
			elif not is_dir and os.path.isdir(norm_path):self.validation_errors.append(f"{name} path is a directory, not a file: {norm_path}")
		other_paths=[(self._get_nested_value(self.config,'logging.path'),'Logging directory',True),(self._get_nested_value(self.config,'tts.cache_dir'),'TTS cache directory',True)]
		for(path,name,is_dir)in other_paths:
			if not path:continue
			norm_path=os.path.normpath(path)
			if not os.path.exists(norm_path):
				self.validation_warnings.append(f"{name} path does not exist: {norm_path}")
				try:os.makedirs(norm_path,exist_ok=True);logger.info(f"Created {name}: {norm_path}")
				except IOError as io_error:self.validation_warnings.append(f"Could not create {name}: {io_error}")
				except Exception as general_error:self.validation_warnings.append(f"Could not create {name}: {general_error}")
		self._validate_extension_paths()
	def _validate_extension_paths(self)->None:
		extensions=self.config.get('extensions',{})
		for(extension_name,extension_config)in extensions.items():
			if'output_dir'in extension_config:
				output_dir=extension_config['output_dir'];norm_output_dir=os.path.normpath(output_dir)
				if not os.path.exists(norm_output_dir):
					self.validation_warnings.append(f"{extension_name} output directory does not exist: {norm_output_dir}")
					try:os.makedirs(norm_output_dir,exist_ok=True);logger.info(f"Created {extension_name} output directory: {norm_output_dir}")
					except IOError as io_error:self.validation_errors.append(f"Failed to create {extension_name} output directory: {io_error}")
					except Exception as general_error:self.validation_errors.append(f"Failed to create {extension_name} output directory: {general_error}")
			if'template_path'in extension_config:
				template_path=extension_config['template_path'];norm_template_path=os.path.normpath(template_path);template_dir=os.path.dirname(norm_template_path)
				if not os.path.exists(template_dir):
					self.validation_warnings.append(f"{extension_name} template directory does not exist: {template_dir}")
					try:os.makedirs(template_dir,exist_ok=True);logger.info(f"Created {extension_name} template directory: {template_dir}")
					except IOError as io_error:self.validation_errors.append(f"Failed to create {extension_name} template directory: {io_error}")
					except Exception as general_error:self.validation_errors.append(f"Failed to create {extension_name} template directory: {general_error}")
	def _validate_value_ranges(self)->None:self._validate_wake_word_settings();self._validate_speech_settings();self._validate_llm_settings();self._validate_threading_settings();self._validate_memory_settings();self._validate_gpu_settings();self._validate_system_settings()
	def _validate_wake_word_settings(self)->None:
		sensitivity=self._get_nested_value(self.config,'stt.wake_word.sensitivity')
		if sensitivity is not None:
			if not isinstance(sensitivity,(int,float)):self.validation_errors.append(f"Wake word sensitivity must be a number, got {type(sensitivity).__name__}")
			elif sensitivity<.0 or sensitivity>1.:self.validation_errors.append(f"Wake word sensitivity must be between 0.0 and 1.0, got {sensitivity}")
		engine=self._get_nested_value(self.config,'stt.wake_word.engine')
		if engine is not None:
			valid_engines=['porcupine','snowboy']
			if engine not in valid_engines:self.validation_errors.append(f"Invalid wake word engine: {engine}. Valid values: {', '.join(valid_engines)}")
	def _validate_speech_settings(self)->None:
		model_size=self._get_nested_value(self.config,'stt.whisper.model_size')
		if model_size is not None:
			valid_sizes=['tiny','base','small','medium','large']
			if model_size not in valid_sizes:self.validation_errors.append(f"Invalid whisper model size: {model_size}. Valid values: {', '.join(valid_sizes)}")
		compute_type=self._get_nested_value(self.config,'stt.whisper.compute_type')
		if compute_type is not None:
			valid_types=['int8','float16','float32']
			if compute_type not in valid_types:self.validation_errors.append(f"Invalid compute type: {compute_type}. Valid values: {', '.join(valid_types)}")
		streaming_compute_type=self._get_nested_value(self.config,'stt.whisper_streaming.compute_type')
		if streaming_compute_type is not None:
			valid_types=['int8','float16','float32']
			if streaming_compute_type not in valid_types:self.validation_errors.append(f"Invalid streaming compute type: {streaming_compute_type}. Valid values: {', '.join(valid_types)}")
		tts_sample_rate=self._get_nested_value(self.config,'tts.sample_rate')
		if tts_sample_rate is not None:
			if not isinstance(tts_sample_rate,int):self.validation_errors.append(f"TTS sample rate must be an integer, got {type(tts_sample_rate).__name__}")
			elif tts_sample_rate not in[16000,22050,24000,44100,48000]:self.validation_warnings.append(f"Unusual TTS sample rate: {tts_sample_rate}. Common values: 16000, 22050, 44100, 48000")
	def _validate_llm_settings(self)->None:
		model_type=self._get_nested_value(self.config,'llm.model_type')
		if model_type is not None:
			valid_types=['mistral','llama','phi']
			if model_type not in valid_types:self.validation_warnings.append(f"Unusual LLM model type: {model_type}. Common values: {', '.join(valid_types)}")
		gpu_layers=self._get_nested_value(self.config,'llm.gpu_layers')
		if gpu_layers is not None:
			if not isinstance(gpu_layers,int):self.validation_errors.append(f"LLM GPU layers must be an integer, got {type(gpu_layers).__name__}")
			elif gpu_layers<0:self.validation_errors.append(f"LLM GPU layers must be non-negative, got {gpu_layers}")
		gpu_auto_adjust=self._get_nested_value(self.config,'llm.gpu_layer_auto_adjust')
		if gpu_auto_adjust is not None and not isinstance(gpu_auto_adjust,bool):self.validation_errors.append(f"LLM GPU layer auto-adjust must be a boolean, got {type(gpu_auto_adjust).__name__}")
	def _validate_threading_settings(self)->None:
		max_workers=self._get_nested_value(self.config,'cpu.max_threads')
		if max_workers is not None:
			import os;cpu_count=os.cpu_count()or 4
			if not isinstance(max_workers,int):self.validation_errors.append(f"cpu.max_threads must be an integer, got {type(max_workers).__name__}")
			elif max_workers<1:self.validation_errors.append(f"cpu.max_threads must be at least 1, got {max_workers}")
			elif max_workers>cpu_count*2:self.validation_warnings.append(f"max_workers ({max_workers}) exceeds twice the number of CPU cores ({cpu_count})")
		thread_timeout=self._get_nested_value(self.config,'cpu.thread_timeout')
		if thread_timeout is not None:
			if not isinstance(thread_timeout,(int,float)):self.validation_errors.append(f"cpu.thread_timeout must be a number, got {type(thread_timeout).__name__}")
			elif thread_timeout<0:self.validation_errors.append(f"cpu.thread_timeout must be non-negative, got {thread_timeout}")
	def _validate_memory_settings(self)->None:
		max_percent=self._get_nested_value(self.config,'memory.max_percent')
		if max_percent is not None:
			if not isinstance(max_percent,(int,float)):self.validation_errors.append(f"memory.max_percent must be a number, got {type(max_percent).__name__}")
			elif max_percent<10:self.validation_errors.append(f"memory.max_percent must be at least 10, got {max_percent}")
			elif max_percent>95:self.validation_errors.append(f"memory.max_percent must be at most 95, got {max_percent}")
		unload_threshold=self._get_nested_value(self.config,'memory.model_unload_threshold')
		if unload_threshold is not None and max_percent is not None:
			if not isinstance(unload_threshold,(int,float)):self.validation_errors.append(f"memory.model_unload_threshold must be a number, got {type(unload_threshold).__name__}")
			elif unload_threshold<=max_percent:self.validation_errors.append(f"memory.model_unload_threshold ({unload_threshold}) must be greater than memory.max_percent ({max_percent})")
	def _validate_gpu_settings(self)->None:
		max_percent=self._get_nested_value(self.config,'gpu.max_percent')
		if max_percent is not None:
			if not isinstance(max_percent,(int,float)):self.validation_errors.append(f"gpu.max_percent must be a number, got {type(max_percent).__name__}")
			elif max_percent<10:self.validation_errors.append(f"gpu.max_percent must be at least 10, got {max_percent}")
			elif max_percent>95:self.validation_errors.append(f"gpu.max_percent must be at most 95, got {max_percent}")
		unload_threshold=self._get_nested_value(self.config,'gpu.model_unload_threshold')
		if unload_threshold is not None and max_percent is not None:
			if not isinstance(unload_threshold,(int,float)):self.validation_errors.append(f"gpu.model_unload_threshold must be a number, got {type(unload_threshold).__name__}")
			elif unload_threshold<=max_percent:self.validation_errors.append(f"gpu.model_unload_threshold ({unload_threshold}) must be greater than gpu.max_percent ({max_percent})")
	def _validate_system_settings(self)->None:
		inactivity_timeout=self._get_nested_value(self.config,'inactivity_timeout')
		if inactivity_timeout is not None:
			if not isinstance(inactivity_timeout,(int,float)):self.validation_errors.append(f"inactivity_timeout must be a number, got {type(inactivity_timeout).__name__}")
			elif inactivity_timeout<10:self.validation_warnings.append(f"Very short inactivity_timeout ({inactivity_timeout}s), system may sleep unexpectedly")
			elif inactivity_timeout>3600:self.validation_warnings.append(f"Very long inactivity_timeout ({inactivity_timeout}s), system may not sleep when expected")
		console_level=self._get_nested_value(self.config,'logging.console_level')
		if console_level is not None:
			valid_levels=['DEBUG','INFO','WARNING','ERROR','CRITICAL']
			if console_level not in valid_levels:self.validation_errors.append(f"Invalid logging.console_level: {console_level}. Valid values: {', '.join(valid_levels)}")
		file_level=self._get_nested_value(self.config,'logging.file_level')
		if file_level is not None:
			valid_levels=['DEBUG','INFO','WARNING','ERROR','CRITICAL']
			if file_level not in valid_levels:self.validation_errors.append(f"Invalid logging.file_level: {file_level}. Valid values: {', '.join(valid_levels)}")
	def apply_hardware_optimizations(self,hardware_info:Dict[str,Any])->None:self._apply_cpu_optimizations(hardware_info.get('cpu',{}));self._apply_memory_optimizations(hardware_info.get('memory',{}));self._apply_gpu_optimizations(hardware_info.get('gpu',{}))
	def _apply_cpu_optimizations(self,cpu_info:Dict[str,Any])->None:
		if cpu_info.get('is_ryzen_9_5900x',False):logger.info('Applying Ryzen 9 5900X optimizations');self.config.setdefault('cpu',{});self.config['cpu']['max_threads']=8;self.config['cpu']['thread_timeout']=30;self.config.setdefault('process',{});self.config['process']['priority']='above_normal';self.config['process']['affinity']='performance_cores'
	def _apply_memory_optimizations(self,memory_info:Dict[str,Any])->None:
		if memory_info.get('is_32gb',False):logger.info('Applying 32GB RAM optimizations');self.config.setdefault('memory',{});self.config['memory']['max_percent']=75;self.config['memory']['model_unload_threshold']=85;self.config.setdefault('tts',{});self.config['tts']['cache_size']=200;self.config.setdefault('stt',{}).setdefault('whisper_streaming',{});self.config['stt']['whisper_streaming']['buffer_size_seconds']=3e1
	def _apply_gpu_optimizations(self,gpu_info:Dict[str,Any])->None:
		if gpu_info.get('is_rtx_3080',False):logger.info('Applying RTX 3080 optimizations');self.config.setdefault('llm',{});self.config['llm']['gpu_layers']=32;self.config['llm']['gpu_layer_auto_adjust']=True;self.config.setdefault('stt',{});self.config['stt'].setdefault('whisper',{});self.config['stt']['whisper']['compute_type']='float16';self.config['stt'].setdefault('whisper_streaming',{});self.config['stt']['whisper_streaming']['compute_type']='float16';self.config.setdefault('tts',{});self.config['tts']['gpu_acceleration']=True;self.config['tts']['gpu_precision']='mixed_float16';self.config.setdefault('gpu',{});self.config['gpu']['max_percent']=90;self.config['gpu']['model_unload_threshold']=95