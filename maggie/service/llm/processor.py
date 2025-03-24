import os,time,threading
from typing import Dict,Any,Optional,List,Tuple,Union,cast
from maggie.utils.error_handling import safe_execute,retry_operation,ErrorCategory,ErrorSeverity,with_error_handling,record_error,LLMError,ModelLoadError,GenerationError
from maggie.utils.logging import ComponentLogger,log_operation,logging_context
from maggie.service.locator import ServiceLocator
class LLMProcessor:
	def __init__(self,config:Dict[str,Any]):
		self.config=config;self.model_path=config.get('model_path','models/mistral-7b-instruct-v0.3-GPTQ-4bit');self.model_type=config.get('model_type','mistral');self.gpu_layers=config.get('gpu_layers',0);self.gpu_layer_auto_adjust=config.get('gpu_layer_auto_adjust',False);self.model=None;self._load_lock=threading.RLock();self.logger=ComponentLogger('LLMProcessor')
		try:
			self.resource_manager=ServiceLocator.get('resource_manager')
			if not self.resource_manager:self.logger.warning('Resource manager not found in ServiceLocator')
		except Exception as e:self.logger.warning(f"Failed to get resource manager: {e}");self.resource_manager=None
		self.logger.info(f"Initialized with model: {self.model_type}, path: {self.model_path}");self.logger.info(f"GPU layers set to {self.gpu_layers} from configuration")
	def _get_model_type_mapping(self)->str:model_type_mapping={'mistral':'mistral','llama2':'llama','phi':'phi'};return model_type_mapping.get(self.model_type,'mistral')
	def _get_model_loading_kwargs(self)->Dict[str,Any]:
		threads=self.config.get('cpu',{}).get('max_threads',8);kwargs={'threads':threads}
		if self.config.get('llm',{}).get('use_cache',True):kwargs['use_cache']=True
		batch_size=self.config.get('llm',{}).get('batch_size',16)
		if batch_size>0:kwargs['batch_size']=batch_size
		return kwargs
	@retry_operation(max_attempts=2,allowed_exceptions=(OSError,RuntimeError))
	@with_error_handling(error_category=ErrorCategory.MODEL,error_severity=ErrorSeverity.ERROR)
	def _load_model(self)->bool:
		with self._load_lock:
			if self.model is not None:return True
			with logging_context(component='LLMProcessor',operation='load_model')as ctx:
				try:from ctransformers import AutoModelForCausalLM;model_type=self._get_model_type_mapping();self.logger.info(f"Loading {self.model_type} model with {self.gpu_layers} GPU layers");start_time=time.time();model_kwargs=self._get_model_loading_kwargs();self.model=AutoModelForCausalLM.from_pretrained(self.model_path,model_type=model_type,gpu_layers=self.gpu_layers,**model_kwargs);load_time=time.time()-start_time;self.logger.log_performance('load_model',load_time,{'model_type':self.model_type,'gpu_layers':self.gpu_layers});self.logger.info(f"Model {self.model_type} loaded successfully in {load_time:.2f}s");return True
				except ImportError as e:self.logger.error('Failed to import ctransformers',exception=e);self.logger.error('Please install with: pip install ctransformers');raise ModelLoadError(f"Failed to import ctransformers: {e}")from e
				except Exception as e:self.logger.error(f"Error loading LLM model",exception=e);raise ModelLoadError(f"Error loading model {self.model_type}: {e}")from e
	def reduce_gpu_layers(self)->bool:
		with self._load_lock:
			if self.gpu_layers<=1:self.logger.warning('Cannot reduce GPU layers below 1');return False
			if self.resource_manager:self.resource_manager.clear_gpu_memory()
			new_layers=max(1,int(self.gpu_layers*.75));self.logger.info(f"Reducing GPU layers from {self.gpu_layers} to {new_layers}");self.gpu_layers=new_layers
			if self.model is not None:self.unload_model()
			return True
	@log_operation(component='LLMProcessor',log_args=True)
	def generate_text(self,prompt:str,max_tokens:int=512,temperature:float=.7,top_p:float=.95,repetition_penalty:float=1.1)->str:
		try:
			if not self._load_model():self.logger.error('Failed to load model for text generation');return''
			with logging_context(component='LLMProcessor',operation='generate')as ctx:
				start_time=time.time();generation_config={'max_new_tokens':max_tokens,'temperature':temperature,'top_p':top_p,'repetition_penalty':repetition_penalty};model_specific_config=self.config.get('llm',{}).get('generation_params',{})
				if model_specific_config:generation_config.update(model_specific_config)
				output=self.model(prompt,**generation_config);generation_time=time.time()-start_time;tokens_generated=len(output.split())-len(prompt.split());tokens_per_second=tokens_generated/generation_time if generation_time>0 else 0;self.logger.log_performance('generate',generation_time,{'tokens':tokens_generated,'tokens_per_second':tokens_per_second,'model_type':self.model_type});self.logger.debug(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)");return output
		except Exception as e:
			self.logger.error('Error generating text',exception=e)
			try:
				event_bus=ServiceLocator.get('event_bus')
				if event_bus:event_bus.publish('error_logged',{'component':'LLMProcessor','operation':'generate_text','error':str(e),'severity':'ERROR'})
			except ImportError:pass
			return''
	@log_operation(component='LLMProcessor')
	def unload_model(self)->bool:return safe_execute(self._unload_model_impl,error_message='Error unloading model',error_category=ErrorCategory.RESOURCE,default_return=False)
	def _unload_model_impl(self)->bool:
		with self._load_lock:
			try:
				self.model=None
				if self.resource_manager:self.resource_manager.clear_gpu_memory()
				self.logger.info('LLM model unloaded');return True
			except Exception as e:self.logger.error(f"Error unloading model",exception=e);return False