import os,yaml,time
from typing import Dict,Any,Optional,List,Tuple,Union
from maggie.utils.error_handling import with_error_handling,ErrorCategory,ErrorSeverity,record_error,safe_execute
from maggie.utils.logging import ComponentLogger,log_operation,logging_context
from maggie.utils.resource.detector import HardwareDetector
from maggie.service.locator import ServiceLocator
__all__=['ConfigManager']
class ConfigManager:
	def __init__(self,config_path:str='config.yaml',backup_dir:str='config_backups'):self.config_path=config_path;self.backup_dir=backup_dir;self.config={};self.validation_errors=[];self.validation_warnings=[];self.logger=ComponentLogger('ConfigManager');self.default_config=self._create_default_config();self.hardware_detector=HardwareDetector();self.hardware_optimizer=None;self.hardware_info=None;os.makedirs(self.backup_dir,exist_ok=True)
	def _create_default_config(self)->Dict[str,Any]:return{'inactivity_timeout':60,'fsm':{'state_styles':{'INIT':{'bg_color':'#E0E0E0','font_color':'#212121'},'STARTUP':{'bg_color':'#B3E5FC','font_color':'#212121'},'IDLE':{'bg_color':'#C8E6C9','font_color':'#212121'},'LOADING':{'bg_color':'#FFE0B2','font_color':'#212121'},'READY':{'bg_color':'#A5D6A7','font_color':'#212121'},'ACTIVE':{'bg_color':'#FFCC80','font_color':'#FFFFFF'},'BUSY':{'bg_color':'#FFAB91','font_color':'#FFFFFF'},'CLEANUP':{'bg_color':'#E1BEE7','font_color':'#FFFFFF'},'SHUTDOWN':{'bg_color':'#EF9A9A','font_color':'#FFFFFF'}},'transition_animations':{'default':{'type':'slide','duration':300},'to_shutdown':{'type':'fade','duration':800},'to_busy':{'type':'bounce','duration':400}},'valid_transitions':{'INIT':['STARTUP','IDLE','SHUTDOWN'],'STARTUP':['IDLE','READY','CLEANUP','SHUTDOWN'],'IDLE':['STARTUP','READY','CLEANUP','SHUTDOWN'],'LOADING':['ACTIVE','READY','CLEANUP','SHUTDOWN'],'READY':['LOADING','ACTIVE','BUSY','CLEANUP','SHUTDOWN'],'ACTIVE':['READY','BUSY','CLEANUP','SHUTDOWN'],'BUSY':['READY','ACTIVE','CLEANUP','SHUTDOWN'],'CLEANUP':['IDLE','SHUTDOWN'],'SHUTDOWN':[]},'input_field_states':{'IDLE':{'enabled':False,'style':'background-color: lightgray;'},'READY':{'enabled':True,'style':'background-color: white;'},'ACTIVE':{'enabled':True,'style':'background-color: white;'},'BUSY':{'enabled':False,'style':'background-color: #FFAB91;'}}},'stt':{'whisper':{'model_size':'base','compute_type':'float16','model_path':'\\maggie\\models\\stt\\whisper','sample_rate':16000,'tensor_cores_enabled':True,'flash_attention_enabled':True,'max_batch_size':16,'memory_efficient':True,'parallel_processing':True,'chunk_size':512,'simd_optimization':True,'cache_models':True},'whisper_streaming':{'enabled':True,'model_name':'base','language':'en','compute_type':'float16','cuda_streams':2,'batch_processing':True,'low_latency_mode':True,'tensor_cores_enabled':True,'dedicated_threads':2,'thread_affinity_enabled':True},'wake_word':{'engine':'porcupine','access_key':'','sensitivity':.5,'keyword':'maggie','keyword_path':'\\maggie\\models\\stt\\porcupine\\Hey-Maggie_en_windows_v3_0_0.ppn','cpu_threshold':5.,'dedicated_core_enabled':True,'dedicated_core':0,'real_time_priority':True,'minimal_processing':True}},'tts':{'voice_model':'af_heart.pt','model_path':'\\maggie\\models\\tts','sample_rate':22050,'use_cache':True,'cache_dir':'\\maggie\\cache\\tts','cache_size':200,'gpu_device':0,'gpu_acceleration':True,'gpu_precision':'mixed_float16','max_workers':4,'voice_preprocessing':True,'tensor_cores_enabled':True,'cuda_graphs_enabled':True,'amp_optimization_level':'O2','max_batch_size':64,'dynamic_memory_management':True,'dedicated_threads':2,'thread_affinity_enabled':True,'thread_affinity_cores':[8,9],'realtime_priority':True,'simd_optimization':True,'buffer_size':4096,'spectral_processing_on_gpu':True},'llm':{'model_path':'maggie\\models\\llm\\mistral-7b-instruct-v0.3-GPTQ-4bit','model_type':'mistral','gpu_layers':32,'gpu_layer_auto_adjust':True,'tensor_cores_enabled':True,'mixed_precision_enabled':True,'precision_type':'float16','kv_cache_optimization':True,'attention_optimization':True,'context_length':8192,'batch_size':16,'offload_strategy':'auto','vram_efficient_loading':True,'rtx_3080_optimized':True},'logging':{'path':'logs','console_level':'INFO','file_level':'DEBUG'},'extensions':{'recipe_creator':{'enabled':True,'template_path':'templates\\recipe_template.docx','output_dir':'recipes'}},'cpu':{'max_threads':8,'thread_timeout':30,'ryzen_9_5900x_optimized':True,'thread_affinity_enabled':True,'performance_cores':[0,1,2,3,4,5,6,7],'background_cores':[8,9,10,11],'high_performance_plan':True,'disable_core_parking':True,'precision_boost_overdrive':True,'simultaneous_multithreading':True},'memory':{'max_percent':75,'model_unload_threshold':85,'xpg_d10_memory':True,'large_pages_enabled':True,'numa_aware':True,'preload_models':True,'cache_size_mb':6144,'min_free_gb':4,'defragmentation_threshold':70},'gpu':{'max_percent':90,'model_unload_threshold':95,'rtx_3080_optimized':True,'tensor_cores_enabled':True,'tensor_precision':'tf32','cuda_compute_type':'float16','cuda_streams':3,'cuda_memory_pool':True,'cuda_graphs':True,'max_batch_size':16,'reserved_memory_mb':256,'dynamic_memory':True,'fragmentation_threshold':15,'pre_allocation':True}}
	@log_operation(component='ConfigManager')
	@with_error_handling(error_category=ErrorCategory.CONFIGURATION)
	def optimize_config_for_hardware(self)->Dict[str,Any]:
		with logging_context(component='ConfigManager',operation='optimize_for_hardware'):
			optimizations={'cpu':{},'gpu':{},'memory':{},'llm':{},'stt':{},'tts':{}}
			if not self.hardware_optimizer and self.hardware_info:self.hardware_optimizer=HardwareOptimizer(self.hardware_info,self.config)
			if not self.hardware_info or not self.hardware_optimizer:self.logger.warning('Cannot optimize configuration: hardware information not available');return optimizations
			cpu_info=self.hardware_info.get('cpu',{})
			if cpu_info.get('is_ryzen_9_5900x',False):
				cpu_opts=self.hardware_optimizer.optimize_for_ryzen_9_5900x()
				if cpu_opts.get('applied',False):
					optimizations['cpu']=cpu_opts.get('settings',{});self.logger.info('Applied Ryzen 9 5900X-specific optimizations')
					if'cpu'not in self.config:self.config['cpu']={}
					for(key,value)in optimizations['cpu'].items():self.config['cpu'][key]=value
					if'stt'in self.config:
						stt_config=self.config['stt']
						if'whisper'in stt_config:stt_config['whisper']['chunk_size']=512;stt_config['whisper']['simd_optimization']=True;optimizations['stt']['chunk_size']=512;optimizations['stt']['simd_optimization']=True
			gpu_info=self.hardware_info.get('gpu',{})
			if gpu_info.get('is_rtx_3080',False):
				gpu_opts=self.hardware_optimizer.optimize_for_rtx_3080()
				if gpu_opts.get('applied',False):
					optimizations['gpu']=gpu_opts.get('settings',{});self.logger.info('Applied RTX 3080-specific optimizations')
					if'gpu'not in self.config:self.config['gpu']={}
					for(key,value)in optimizations['gpu'].items():self.config['gpu'][key]=value
					if'llm'in self.config:self.config['llm']['gpu_layers']=32;self.config['llm']['tensor_cores_enabled']=True;self.config['llm']['mixed_precision_enabled']=True;self.config['llm']['precision_type']='float16';optimizations['llm']['gpu_layers']=32;optimizations['llm']['tensor_cores_enabled']=True;optimizations['llm']['mixed_precision_enabled']=True;optimizations['llm']['precision_type']='float16'
					if'stt'in self.config:
						if'whisper'in self.config['stt']:self.config['stt']['whisper']['compute_type']='float16';self.config['stt']['whisper']['tensor_cores_enabled']=True;self.config['stt']['whisper']['flash_attention_enabled']=True;optimizations['stt']['compute_type']='float16';optimizations['stt']['tensor_cores_enabled']=True
						if'whisper_streaming'in self.config['stt']:self.config['stt']['whisper_streaming']['compute_type']='float16';self.config['stt']['whisper_streaming']['tensor_cores_enabled']=True;optimizations['stt']['streaming_compute_type']='float16'
					if'tts'in self.config:self.config['tts']['gpu_acceleration']=True;self.config['tts']['gpu_precision']='mixed_float16';self.config['tts']['tensor_cores_enabled']=True;optimizations['tts']['gpu_acceleration']=True;optimizations['tts']['gpu_precision']='mixed_float16';optimizations['tts']['tensor_cores_enabled']=True
			memory_info=self.hardware_info.get('memory',{})
			if memory_info.get('is_xpg_d10',False)and memory_info.get('is_32gb',False):
				if'memory'not in self.config:self.config['memory']={}
				self.config['memory']['large_pages_enabled']=True;self.config['memory']['numa_aware']=True;self.config['memory']['cache_size_mb']=6144;optimizations['memory']['large_pages_enabled']=True;optimizations['memory']['numa_aware']=True;optimizations['memory']['cache_size_mb']=6144;self.logger.info('Applied XPG D10 memory-specific optimizations')
			if any(settings for settings in optimizations.values()):self.save()
			return optimizations
	def _validate_required_params(self)->None:
		required_params=[('stt.wake_word.access_key','Picovoice access key'),('llm.model_path','LLM model path'),('tts.voice_model','TTS voice model')]
		for(param_path,param_name)in required_params:
			value=self._get_nested_value(self.config,param_path)
			if value is None:self.validation_errors.append(f"Missing required configuration: {param_name} ({param_path})")
			elif isinstance(value,str)and not value:self.validation_errors.append(f"Empty required configuration: {param_name} ({param_path})")
	def _validate_fsm_config(self)->None:
		fsm_config=self.config.get('fsm',{});state_styles=fsm_config.get('state_styles',{});required_states=['INIT','STARTUP','IDLE','LOADING','READY','ACTIVE','BUSY','CLEANUP','SHUTDOWN']
		for state in required_states:
			if state not in state_styles:self.validation_warnings.append(f"Missing style configuration for state: {state}")
			else:
				style=state_styles[state]
				if'bg_color'not in style or'font_color'not in style:self.validation_warnings.append(f"Incomplete style configuration for state: {state}")
		valid_transitions=fsm_config.get('valid_transitions',{})
		for state in required_states:
			if state not in valid_transitions:self.validation_warnings.append(f"Missing transition configuration for state: {state}")
		input_field_states=fsm_config.get('input_field_states',{})
		for state in['IDLE','READY','ACTIVE','BUSY']:
			if state not in input_field_states:self.validation_warnings.append(f"Missing input field configuration for state: {state}")
	def _validate_hardware_specific_settings(self)->None:
		cpu_config=self.config.get('cpu',{})
		if cpu_config.get('ryzen_9_5900x_optimized',False):
			if'max_threads'not in cpu_config:self.validation_warnings.append("Missing 'max_threads' setting for Ryzen 9 5900X optimization")
			elif cpu_config['max_threads']>10:self.validation_warnings.append("'max_threads' value too high for optimal Ryzen 9 5900X performance")
			if'thread_affinity_enabled'not in cpu_config:self.validation_warnings.append("Missing 'thread_affinity_enabled' setting for Ryzen 9 5900X optimization")
			if'performance_cores'not in cpu_config or not cpu_config.get('performance_cores'):self.validation_warnings.append("Missing 'performance_cores' configuration for Ryzen 9 5900X")
		gpu_config=self.config.get('gpu',{})
		if gpu_config.get('rtx_3080_optimized',False):
			if'tensor_cores_enabled'not in gpu_config:self.validation_warnings.append("Missing 'tensor_cores_enabled' setting for RTX 3080 optimization")
			if'cuda_streams'not in gpu_config:self.validation_warnings.append("Missing 'cuda_streams' setting for RTX 3080 optimization")
			if'max_batch_size'not in gpu_config:self.validation_warnings.append("Missing 'max_batch_size' setting for RTX 3080 optimization")
		llm_config=self.config.get('llm',{})
		if gpu_config.get('rtx_3080_optimized',False)and llm_config:
			if'gpu_layers'not in llm_config:self.validation_warnings.append("Missing 'gpu_layers' setting for LLM with RTX 3080")
			elif llm_config.get('gpu_layers',0)!=32 and not llm_config.get('gpu_layer_auto_adjust',False):self.validation_warnings.append("Non-optimal 'gpu_layers' setting (should be 32) for RTX 3080 without auto-adjust")
		memory_config=self.config.get('memory',{})
		if memory_config.get('xpg_d10_memory',False):
			if'large_pages_enabled'not in memory_config:self.validation_warnings.append("Missing 'large_pages_enabled' setting for XPG D10 memory optimization")
			if'numa_aware'not in memory_config:self.validation_warnings.append("Missing 'numa_aware' setting for XPG D10 memory optimization")
	@log_operation(component='ConfigManager')
	@with_error_handling(error_category=ErrorCategory.CONFIGURATION)
	def load(self)->Dict[str,Any]:
		with logging_context(component='ConfigManager',operation='load'):
			self.hardware_info=self._detect_hardware()
			if os.path.exists(self.config_path):
				try:
					with open(self.config_path,'r')as file:self.config=yaml.safe_load(file)or{}
					self.logger.info(f"Configuration loaded from {self.config_path}");self._create_backup('loaded')
				except yaml.YAMLError as yaml_error:self.logger.error(f"YAML error in configuration: {yaml_error}");self._attempt_config_recovery(yaml_error)
				except IOError as io_error:self.logger.error(f"IO error reading configuration: {io_error}");self._attempt_config_recovery(io_error)
			else:self.logger.info(f"Configuration file {self.config_path} not found, creating with defaults");self.config=self.default_config;self.save()
			self._merge_with_defaults();self.validate()
			if self.hardware_info:self.optimize_config_for_hardware()
			return self.config
	def _detect_hardware(self)->Dict[str,Any]:
		try:
			hardware_info=self.hardware_detector.detect_system();cpu_info=hardware_info.get('cpu',{});gpu_info=hardware_info.get('gpu',{});memory_info=hardware_info.get('memory',{})
			if cpu_info.get('is_ryzen_9_5900x',False):self.logger.info('Detected AMD Ryzen 9 5900X CPU - applying optimized settings')
			else:self.logger.info(f"Detected CPU: {cpu_info.get('model','Unknown')}")
			if gpu_info.get('is_rtx_3080',False):self.logger.info('Detected NVIDIA RTX 3080 GPU - applying optimized settings')
			elif gpu_info.get('available',False):self.logger.info(f"Detected GPU: {gpu_info.get('name','Unknown')}")
			else:self.logger.warning('No compatible GPU detected - some features may be limited')
			if memory_info.get('is_xpg_d10',False):self.logger.info('Detected ADATA XPG D10 memory - applying optimized settings')
			from maggie.utils.resource.optimizer import HardwareOptimizer;self.hardware_optimizer=HardwareOptimizer(hardware_info,self.default_config);return hardware_info
		except Exception as e:self.logger.error(f"Error detecting hardware: {e}");return{}
	def _attempt_config_recovery(self,error:Exception)->None:
		backup_path=self._find_latest_backup()
		if backup_path:
			self.logger.info(f"Attempting to recover from backup: {backup_path}")
			try:
				with open(backup_path,'r')as file:self.config=yaml.safe_load(file)or{}
				self.logger.info(f"Configuration recovered from backup: {backup_path}")
			except Exception as recover_error:self.logger.error(f"Failed to recover from backup: {recover_error}");self.config=self.default_config;self.logger.info('Using default configuration')
		else:self.config=self.default_config;self.logger.info('Using default configuration')
	def _merge_with_defaults(self)->None:hardware_specific_settings=self._extract_hardware_specific_settings(self.config);self.config=self._deep_merge(self.default_config.copy(),self.config);self._restore_hardware_specific_settings(hardware_specific_settings)
	def apply_state_specific_config(self,state)->None:self.logger.debug(f"Applying state-specific configuration for state: {state.name}")
	def _get_nested_value(self,config_dict,path_string):
		parts=path_string.split('.');current=config_dict
		for part in parts:
			if part not in current:return None
			current=current[part]
		return current
	def _extract_hardware_specific_settings(self,config):hardware_settings={};return hardware_settings
	def _restore_hardware_specific_settings(self,settings):pass
	def _deep_merge(self,default_dict,user_dict):return default_dict
	def _find_latest_backup(self):return None
	def _create_backup(self,reason):pass
	def validate(self):pass
	def save(self):pass
	def _get_hardware_optimizer(self,hardware_info):from maggie.utils.resource.optimizer import HardwareOptimizer;return HardwareOptimizer(hardware_info,self.config)