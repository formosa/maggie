import os,yaml,time
from typing import Dict,Any,Optional,List
from maggie.utils.error_handling import with_error_handling,ErrorCategory
from maggie.utils.logging import ComponentLogger
__all__=['ConfigManager']
class ConfigManager:
	def __init__(self,config_path:str='config.yaml',backup_dir:str='config_backups'):self.config_path=config_path;self.backup_dir=backup_dir;self.config={};self.validation_errors=[];self.validation_warnings=[];self.logger=ComponentLogger('ConfigManager');self.default_config=self._create_default_config();os.makedirs(self.backup_dir,exist_ok=True)
	def _create_default_config(self)->Dict[str,Any]:return{'inactivity_timeout':60,'fsm':{'state_styles':{'INIT':{'bg_color':'#E0E0E0','font_color':'#212121'},'STARTUP':{'bg_color':'#B3E5FC','font_color':'#212121'},'IDLE':{'bg_color':'#C8E6C9','font_color':'#212121'},'LOADING':{'bg_color':'#FFE0B2','font_color':'#212121'},'READY':{'bg_color':'#A5D6A7','font_color':'#212121'},'ACTIVE':{'bg_color':'#FFCC80','font_color':'#FFFFFF'},'BUSY':{'bg_color':'#FFAB91','font_color':'#FFFFFF'},'CLEANUP':{'bg_color':'#E1BEE7','font_color':'#FFFFFF'},'SHUTDOWN':{'bg_color':'#EF9A9A','font_color':'#FFFFFF'}},'transition_animations':{'default':{'type':'slide','duration':300},'to_shutdown':{'type':'fade','duration':800},'to_busy':{'type':'bounce','duration':400}},'valid_transitions':{'INIT':['STARTUP','IDLE','SHUTDOWN'],'STARTUP':['IDLE','READY','CLEANUP','SHUTDOWN'],'IDLE':['STARTUP','READY','CLEANUP','SHUTDOWN'],'LOADING':['ACTIVE','READY','CLEANUP','SHUTDOWN'],'READY':['LOADING','ACTIVE','BUSY','CLEANUP','SHUTDOWN'],'ACTIVE':['READY','BUSY','CLEANUP','SHUTDOWN'],'BUSY':['READY','ACTIVE','CLEANUP','SHUTDOWN'],'CLEANUP':['IDLE','SHUTDOWN'],'SHUTDOWN':[]},'input_field_states':{'IDLE':{'enabled':False,'style':'background-color: lightgray;'},'READY':{'enabled':True,'style':'background-color: white;'},'ACTIVE':{'enabled':True,'style':'background-color: white;'},'BUSY':{'enabled':False,'style':'background-color: #FFAB91;'}}},'stt':{'whisper':{'model_size':'base','compute_type':'float16','model_path':'\\maggie\\models\\stt\\whisper','sample_rate':16000},'whisper_streaming':{'enabled':True,'model_name':'base','language':'en','compute_type':'float16'},'wake_word':{'engine':'porcupine','access_key':'','sensitivity':.5,'keyword':'maggie','keyword_path':'\\maggie\\models\\stt\\porcupine\\Hey-Maggie_en_windows_v3_0_0.ppn'}},'tts':{'voice_model':'af_heart.pt','model_path':'\\maggie\\models\\tts','sample_rate':22050,'use_cache':True,'cache_dir':'\\maggie\\cache\\tts'},'llm':{'model_path':'maggie\\models\\llm\\mistral-7b-instruct-v0.3-GPTQ-4bit','model_type':'mistral'},'logging':{'path':'logs','console_level':'INFO','file_level':'DEBUG'},'extensions':{'recipe_creator':{'enabled':True,'template_path':'templates\\recipe_template.docx','output_dir':'recipes'}}}
	@with_error_handling(error_category=ErrorCategory.CONFIGURATION)
	def load(self)->Dict[str,Any]:
		if os.path.exists(self.config_path):
			try:
				with open(self.config_path,'r')as file:self.config=yaml.safe_load(file)or{}
				self.logger.info(f"Configuration loaded from {self.config_path}");self._create_backup('loaded')
			except yaml.YAMLError as yaml_error:self.logger.error(f"YAML error in configuration: {yaml_error}");self._attempt_config_recovery(yaml_error)
			except IOError as io_error:self.logger.error(f"IO error reading configuration: {io_error}");self._attempt_config_recovery(io_error)
		else:self.logger.info(f"Configuration file {self.config_path} not found, creating with defaults");self.config=self.default_config;self.save()
		self._merge_with_defaults();self.validate();return self.config
	def _attempt_config_recovery(self,error:Exception)->None:
		backup_path=self._find_latest_backup()
		if backup_path:
			self.logger.info(f"Attempting to recover from backup: {backup_path}")
			try:
				with open(backup_path,'r')as file:self.config=yaml.safe_load(file)or{}
				self.logger.info(f"Configuration recovered from backup: {backup_path}")
			except Exception as recover_error:self.logger.error(f"Failed to recover from backup: {recover_error}");self.config=self.default_config;self.logger.info('Using default configuration')
		else:self.config=self.default_config;self.logger.info('Using default configuration')
	def _merge_with_defaults(self)->None:self.config=self._deep_merge(self.default_config.copy(),self.config)
	def _deep_merge(self,default_dict:Dict,user_dict:Dict)->Dict:
		result=default_dict.copy()
		for(key,value)in user_dict.items():
			if key in result and isinstance(result[key],dict)and isinstance(value,dict):result[key]=self._deep_merge(result[key],value)
			else:result[key]=value
		return result
	@with_error_handling(error_category=ErrorCategory.CONFIGURATION)
	def save(self)->bool:
		try:
			os.makedirs(os.path.dirname(os.path.abspath(self.config_path)),exist_ok=True);self._create_backup('before_save')
			with open(self.config_path,'w')as file:file.write('# Maggie AI Assistant Configuration\n');file.write('# Optimized for AMD Ryzen 9 5900X and NVIDIA GeForce RTX 3080\n');file.write('# System configuration for performance and resource management\n\n');yaml.dump(self.config,file,default_flow_style=False,sort_keys=False)
			self.logger.info(f"Configuration saved to {self.config_path}");self._create_backup('after_save');return True
		except Exception as general_error:self.logger.error(f"Error saving configuration: {general_error}");return False
	def _create_backup(self,reason:str)->Optional[str]:
		try:
			timestamp=time.strftime('%Y%m%d_%H%M%S');backup_path=os.path.join(self.backup_dir,f"config_{timestamp}_{reason}.yaml");os.makedirs(self.backup_dir,exist_ok=True)
			with open(backup_path,'w')as file:yaml.dump(self.config,file,default_flow_style=False)
			self.logger.debug(f"Configuration backup created: {backup_path}");self._cleanup_old_backups();return backup_path
		except Exception as general_error:self.logger.error(f"Error creating configuration backup: {general_error}");return None
	def _find_latest_backup(self)->Optional[str]:
		try:
			if not os.path.exists(self.backup_dir):return None
			backup_files=[os.path.join(self.backup_dir,f)for f in os.listdir(self.backup_dir)if f.startswith('config_')and f.endswith('.yaml')]
			if not backup_files:return None
			backup_files.sort(key=os.path.getmtime,reverse=True);return backup_files[0]
		except Exception as general_error:self.logger.error(f"Error finding latest backup: {general_error}");return None
	def _cleanup_old_backups(self,keep:int=10)->None:
		try:
			if not os.path.exists(self.backup_dir):return
			backup_files=[os.path.join(self.backup_dir,f)for f in os.listdir(self.backup_dir)if f.startswith('config_')and f.endswith('.yaml')]
			if len(backup_files)<=keep:return
			backup_files.sort(key=os.path.getmtime,reverse=True)
			for old_backup in backup_files[keep:]:os.remove(old_backup);self.logger.debug(f"Removed old backup: {old_backup}")
		except Exception as general_error:self.logger.error(f"Error cleaning up old backups: {general_error}")
	@with_error_handling(error_category=ErrorCategory.CONFIGURATION)
	def validate(self)->bool:
		self.validation_errors=[];self.validation_warnings=[];self._validate_required_params();self._validate_fsm_config()
		for error in self.validation_errors:self.logger.error(f"Configuration error: {error}")
		for warning in self.validation_warnings:self.logger.warning(f"Configuration warning: {warning}")
		return len(self.validation_errors)==0
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
	def _get_nested_value(self,config:Dict[str,Any],path:str)->Any:
		parts=path.split('.');current=config
		for part in parts:
			if not isinstance(current,dict)or part not in current:return None
			current=current[part]
		return current
	def get_state_visualization_config(self,state_name:str)->Dict[str,Any]:
		fsm_config=self.config.get('fsm',{});state_styles=fsm_config.get('state_styles',{})
		if state_name in state_styles:return state_styles[state_name]
		else:return{'bg_color':'#E0E0E0','font_color':'#212121'}
	def get_transition_animation(self,from_state:str,to_state:str)->Dict[str,Any]:
		fsm_config=self.config.get('fsm',{});transition_animations=fsm_config.get('transition_animations',{});transition_key=f"{from_state}_to_{to_state}"
		if transition_key in transition_animations:return transition_animations[transition_key]
		to_key=f"to_{to_state}"
		if to_key in transition_animations:return transition_animations[to_key]
		if'default'in transition_animations:return transition_animations['default']
		return{'type':'slide','duration':300,'easing':'ease-in-out'}
	def get_valid_transitions(self,from_state:str)->List[str]:
		fsm_config=self.config.get('fsm',{});valid_transitions=fsm_config.get('valid_transitions',{})
		if from_state in valid_transitions:return valid_transitions[from_state]
		else:return[]
	def get_input_field_config(self,state_name:str)->Dict[str,Any]:
		fsm_config=self.config.get('fsm',{});input_field_states=fsm_config.get('input_field_states',{})
		if state_name in input_field_states:return input_field_states[state_name]
		else:return{'enabled':False,'style':'background-color: lightgray;'}
	def is_valid_transition(self,from_state:str,to_state:str)->bool:valid_transitions=self.get_valid_transitions(from_state);return to_state in valid_transitions