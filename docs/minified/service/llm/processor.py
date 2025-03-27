import os,time,threading
from typing import Dict,Any,Optional,List,Tuple,Union,cast
from maggie.core.state import State,StateTransition,StateAwareComponent
from maggie.core.event import EventListener,EventPriority
from maggie.utils.error_handling import safe_execute,retry_operation,ErrorCategory,ErrorSeverity,with_error_handling,record_error,LLMError,ModelLoadError,GenerationError
from maggie.utils.logging import ComponentLogger,log_operation,logging_context
from maggie.service.locator import ServiceLocator
class LLMProcessor(StateAwareComponent,EventListener):
	def __init__(self,config:Dict[str,Any]):
		self.state_manager=ServiceLocator.get('state_manager')
		if self.state_manager:StateAwareComponent.__init__(self,self.state_manager)
		self.event_bus=ServiceLocator.get('event_bus')
		if self.event_bus:EventListener.__init__(self,self.event_bus)
		self.config=config;self.model_path=config.get('model_path','models/mistral-7b-instruct-v0.3-GPTQ-4bit');self.model_type=config.get('model_type','mistral');self.gpu_layers=config.get('gpu_layers',0);self.gpu_layer_auto_adjust=config.get('gpu_layer_auto_adjust',False);self.model=None;self._load_lock=threading.RLock();self.logger=ComponentLogger('LLMProcessor')
		try:
			self.resource_manager=ServiceLocator.get('resource_manager')
			if not self.resource_manager:self.logger.warning('Resource manager not found in ServiceLocator')
		except Exception as e:self.logger.warning(f"Failed to get resource manager: {e}");self.resource_manager=None
		if self.state_manager:self._register_state_handlers()
		if self.event_bus:self._register_event_handlers()
		self.logger.info(f"Initialized with model: {self.model_type}, path: {self.model_path}");self.logger.info(f"GPU layers set to {self.gpu_layers} from configuration")
	def _register_state_handlers(self)->None:self.state_manager.register_state_handler(State.INIT,self.on_enter_init,True);self.state_manager.register_state_handler(State.STARTUP,self.on_enter_startup,True);self.state_manager.register_state_handler(State.IDLE,self.on_enter_idle,True);self.state_manager.register_state_handler(State.LOADING,self.on_enter_loading,True);self.state_manager.register_state_handler(State.READY,self.on_enter_ready,True);self.state_manager.register_state_handler(State.ACTIVE,self.on_enter_active,True);self.state_manager.register_state_handler(State.BUSY,self.on_enter_busy,True);self.state_manager.register_state_handler(State.CLEANUP,self.on_enter_cleanup,True);self.state_manager.register_state_handler(State.SHUTDOWN,self.on_enter_shutdown,True);self.state_manager.register_state_handler(State.ACTIVE,self.on_exit_active,False);self.state_manager.register_state_handler(State.BUSY,self.on_exit_busy,False);self.logger.debug('LLM state handlers registered')
	def _register_event_handlers(self)->None:
		event_handlers=[('low_memory_warning',self._handle_low_memory,EventPriority.HIGH),('gpu_memory_warning',self._handle_gpu_memory_warning,EventPriority.HIGH),('model_unload_request',self._handle_model_unload_request,EventPriority.NORMAL)]
		for(event_type,handler,priority)in event_handlers:self.listen(event_type,handler,priority=priority)
		self.logger.debug(f"Registered {len(event_handlers)} event handlers")
	def on_enter_init(self,transition:StateTransition)->None:self.unload_model();self.logger.debug('LLM processor reset in INIT state')
	def on_enter_startup(self,transition:StateTransition)->None:
		if self.gpu_layer_auto_adjust:self._adjust_gpu_layers_for_hardware()
	def on_enter_idle(self,transition:StateTransition)->None:self.unload_model()
	def on_enter_loading(self,transition:StateTransition)->None:
		if self.resource_manager:self.resource_manager.clear_gpu_memory()
		self._optimize_for_rtx_3080()
	def on_enter_ready(self,transition:StateTransition)->None:
		if self.config.get('llm',{}).get('preload_in_ready',False):self._load_model()
	def on_enter_active(self,transition:StateTransition)->None:self._prepare_for_active_state()
	def on_enter_busy(self,transition:StateTransition)->None:self._optimize_for_busy_state()
	def on_enter_cleanup(self,transition:StateTransition)->None:self.unload_model()
	def on_enter_shutdown(self,transition:StateTransition)->None:self.unload_model()
	def on_exit_active(self,transition:StateTransition)->None:
		if transition.to_state==State.BUSY:self._prepare_for_busy_state()
	def on_exit_busy(self,transition:StateTransition)->None:
		if transition.to_state==State.READY:
			if self.resource_manager:self.resource_manager.clear_gpu_memory()
	def _handle_low_memory(self,event_data:Dict[str,Any])->None:
		memory_percent=event_data.get('percent',0);available_gb=event_data.get('available_gb',0);self.logger.warning(f"Low memory warning: {memory_percent:.1f}% used, {available_gb:.1f} GB available");current_state=self.state_manager.get_current_state()if self.state_manager else None
		if current_state in[State.IDLE,State.READY,State.CLEANUP]:self.unload_model()
		elif current_state in[State.LOADING,State.ACTIVE,State.BUSY]:
			self.logger.warning('Low memory during active processing, consider completing operations')
			if self.gpu_layer_auto_adjust:self.reduce_gpu_layers()
	def _handle_gpu_memory_warning(self,event_data:Dict[str,Any])->None:
		gpu_percent=event_data.get('percent',0);allocated_gb=event_data.get('allocated_gb',0);self.logger.warning(f"GPU memory warning: {gpu_percent:.1f}% used, {allocated_gb:.1f} GB allocated");current_state=self.state_manager.get_current_state()if self.state_manager else None
		if current_state in[State.IDLE,State.READY,State.CLEANUP]:self.unload_model()
		elif current_state in[State.LOADING,State.ACTIVE,State.BUSY]:
			if self.gpu_layer_auto_adjust:self.reduce_gpu_layers()
	def _handle_model_unload_request(self,_:Any)->None:self.unload_model()
	def _prepare_for_active_state(self)->None:
		if self.model is None:self._optimize_for_rtx_3080();self._load_model()
	def _prepare_for_busy_state(self)->None:
		if self.resource_manager:self.resource_manager.clear_gpu_memory()
		if self.config.get('gpu',{}).get('rtx_3080_optimized',False):
			try:
				import torch
				if torch.cuda.is_available():
					torch.backends.cudnn.benchmark=True
					if hasattr(torch.backends.cuda,'matmul'):torch.backends.cuda.matmul.allow_tf32=True
					if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.allow_tf32=True
			except ImportError:pass
	def _optimize_for_busy_state(self)->None:
		try:
			if os.name=='nt':import psutil;psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS);self.logger.debug('Set process priority to HIGH for BUSY state')
		except Exception as e:self.logger.debug(f"Could not set process priority: {e}")
	def _adjust_gpu_layers_for_hardware(self)->None:
		if self.config.get('gpu',{}).get('rtx_3080_optimized',False):self.gpu_layers=self.config.get('llm',{}).get('gpu_layers',32);self.logger.info(f"Using {self.gpu_layers} GPU layers from configuration")
	def _optimize_for_rtx_3080(self)->None:
		if self.config.get('gpu',{}).get('rtx_3080_optimized',False):self.logger.info('Applying RTX 3080 optimizations from configuration')
	def _get_model_type_mapping(self)->str:model_type_mapping={'mistral':'mistral','llama2':'llama','phi':'phi'};return model_type_mapping.get(self.model_type,'mistral')
	def _get_model_loading_kwargs(self)->Dict[str,Any]:
		threads=self.config.get('cpu',{}).get('max_threads',8);kwargs={'threads':threads};llm_config=self.config.get('llm',{})
		if llm_config.get('use_cache',True):kwargs['use_cache']=True
		batch_size=llm_config.get('batch_size',16)
		if batch_size>0:kwargs['batch_size']=batch_size
		return kwargs
	@retry_operation(max_attempts=2,allowed_exceptions=(OSError,RuntimeError))
	@with_error_handling(error_category=ErrorCategory.MODEL,error_severity=ErrorSeverity.ERROR)
	def _load_model(self)->bool:
		with self._load_lock:
			if self.model is not None:return True
			with logging_context(component='LLMProcessor',operation='load_model')as ctx:
				try:
					from ctransformers import AutoModelForCausalLM;model_type=self._get_model_type_mapping();self.logger.info(f"Loading {self.model_type} model with {self.gpu_layers} GPU layers")
					if self.state_manager:current_state=self.state_manager.get_current_state();ctx['state']=current_state.name if current_state else'unknown'
					start_time=time.time();model_kwargs=self._get_model_loading_kwargs();self.model=AutoModelForCausalLM.from_pretrained(self.model_path,model_type=model_type,gpu_layers=self.gpu_layers,**model_kwargs);load_time=time.time()-start_time;self.logger.log_performance('load_model',load_time,{'model_type':self.model_type,'gpu_layers':self.gpu_layers,'state':ctx.get('state','unknown')});self.logger.info(f"Model {self.model_type} loaded successfully in {load_time:.2f}s");return True
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
		current_state=None
		if self.state_manager:current_state=self.state_manager.get_current_state()
		if current_state not in[State.ACTIVE,State.BUSY,State.READY]:
			if current_state:
				self.logger.warning(f"Text generation not allowed in {current_state.name} state")
				if self.event_bus:self.event_bus.publish('error_logged',{'component':'LLMProcessor','operation':'generate_text','error':f"Text generation not allowed in {current_state.name} state",'severity':'WARNING','state':current_state.name})
				return''
		try:
			if not self._load_model():self.logger.error('Failed to load model for text generation');return''
			with logging_context(component='LLMProcessor',operation='generate')as ctx:
				start_time=time.time()
				if current_state:ctx['state']=current_state.name
				generation_config={'max_new_tokens':max_tokens,'temperature':temperature,'top_p':top_p,'repetition_penalty':repetition_penalty};model_specific_config=self.config.get('llm',{}).get('generation_params',{})
				if model_specific_config:generation_config.update(model_specific_config)
				if current_state==State.BUSY:generation_config['max_new_tokens']=min(generation_config['max_new_tokens'],self.config.get('llm',{}).get('busy_max_tokens',256))
				output=self.model(prompt,**generation_config);generation_time=time.time()-start_time;tokens_generated=len(output.split())-len(prompt.split());tokens_per_second=tokens_generated/generation_time if generation_time>0 else 0;self.logger.log_performance('generate',generation_time,{'tokens':tokens_generated,'tokens_per_second':tokens_per_second,'model_type':self.model_type,'state':current_state.name if current_state else'unknown'});self.logger.debug(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)");return output
		except Exception as e:
			self.logger.error('Error generating text',exception=e);error_data={'component':'LLMProcessor','operation':'generate_text','error':str(e),'severity':'ERROR'}
			if current_state:error_data['state']=current_state.name
			try:
				if self.event_bus:self.event_bus.publish('error_logged',error_data)
			except Exception:pass
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