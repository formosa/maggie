import os,platform
from typing import Dict,Any,Optional,Tuple,List,Union
from maggie.utils.error_handling import safe_execute,ErrorCategory,ErrorSeverity,with_error_handling,record_error
from maggie.utils.logging import ComponentLogger,log_operation,logging_context
from maggie.service.locator import ServiceLocator
class HardwareOptimizer:
	def __init__(self,hardware_info:Dict[str,Any],config:Dict[str,Any]):
		self.hardware_info=hardware_info;self.config=config;self.logger=ComponentLogger('HardwareOptimizer');self.state_manager=None;self._state_specific_profiles={};self._transition_profiles={}
		try:
			self.state_manager=ServiceLocator.get('state_manager')
			if self.state_manager:self.logger.info('State manager detected, enabling FSM-aware optimizations');self._initialize_state_profiles()
		except Exception as e:self.logger.debug(f"State manager not available: {e}")
	def _initialize_state_profiles(self)->None:from maggie.core.state import State;self._state_specific_profiles={State.INIT:{'cpu':{'priority':'normal','affinity':'minimal'},'gpu':{'enabled':False},'memory':{'preload':False,'clear_cache':True}},State.STARTUP:{'cpu':{'priority':'above_normal','affinity':'balanced'},'gpu':{'enabled':True,'memory_percent':30},'memory':{'preload':False,'clear_cache':True}},State.IDLE:{'cpu':{'priority':'below_normal','affinity':'efficient'},'gpu':{'enabled':False},'memory':{'preload':False,'clear_cache':True}},State.LOADING:{'cpu':{'priority':'high','affinity':'performance'},'gpu':{'enabled':True,'memory_percent':90,'tensor_cores':True},'memory':{'preload':True,'clear_cache':False}},State.READY:{'cpu':{'priority':'normal','affinity':'balanced'},'gpu':{'enabled':True,'memory_percent':60},'memory':{'preload':True,'clear_cache':False}},State.ACTIVE:{'cpu':{'priority':'high','affinity':'performance'},'gpu':{'enabled':True,'memory_percent':80,'tensor_cores':True},'memory':{'preload':True,'clear_cache':False}},State.BUSY:{'cpu':{'priority':'high','affinity':'performance'},'gpu':{'enabled':True,'memory_percent':95,'tensor_cores':True},'memory':{'preload':True,'clear_cache':False}},State.CLEANUP:{'cpu':{'priority':'normal','affinity':'balanced'},'gpu':{'enabled':True,'memory_percent':30},'memory':{'preload':False,'clear_cache':True}},State.SHUTDOWN:{'cpu':{'priority':'normal','affinity':'minimal'},'gpu':{'enabled':False},'memory':{'preload':False,'clear_cache':True}}};from maggie.core.state import StateTransition;self._transition_profiles={(State.ACTIVE,State.BUSY):{'clear_gpu_cache':True,'optimize_tensor_cores':True,'increase_priority':True},(State.LOADING,State.ACTIVE):{'clear_gpu_cache':True,'preload_models':True},(State.READY,State.LOADING):{'clear_gpu_cache':True,'clear_memory_cache':True},(State.BUSY,State.READY):{'reduce_memory_usage':True,'clear_gpu_cache':True,'decrease_priority':True}}
	@log_operation(component='HardwareOptimizer')
	def create_optimization_profile(self)->Dict[str,Any]:profile={'threading':self._optimize_threading(),'memory':self._optimize_memory(),'gpu':self._optimize_gpu(),'llm':self._optimize_llm(),'audio':self._optimize_audio(),'input_processing':self._optimize_input_processing()};return profile
	def get_state_specific_profile(self,state=None)->Dict[str,Any]:
		if state is None and self.state_manager:
			try:state=self.state_manager.get_current_state()
			except Exception as e:self.logger.error(f"Error getting current state: {e}");return{}
		if state in self._state_specific_profiles:return self._state_specific_profiles[state]
		return{}
	def get_transition_profile(self,from_state,to_state)->Dict[str,Any]:return self._transition_profiles.get((from_state,to_state),{})
	def _optimize_threading(self)->Dict[str,Any]:
		cpu_info=self.hardware_info['cpu']
		if cpu_info.get('is_ryzen_9_5900x',False):return{'max_workers':10,'thread_timeout':30,'worker_affinity':list(range(10)),'priority_boost':True,'use_dedicated_threads':True,'numa_aware':True,'power_plan':'High performance','core_parking':False,'smt_enabled':True,'background_thread_priority':'below_normal','thread_scheduler':'dynamic','performance_cores':[0,1,2,3,4,5,6,7],'background_cores':[8,9,10,11],'io_threads':[8,9],'compute_threads':[0,1,2,3,4,5,6,7],'thread_affinity_strategy':'performance_first','thread_affinity_enabled':True,'thread_priority_scheme':'adaptive'}
		else:physical_cores=cpu_info.get('physical_cores',4);max_workers=max(2,min(physical_cores-2,physical_cores*4//5));return{'max_workers':max_workers,'thread_timeout':30}
	def _optimize_memory(self)->Dict[str,Any]:
		memory_info=self.hardware_info['memory']
		if memory_info.get('is_32gb',False)and memory_info.get('is_xpg_d10',False):return{'max_percent':80,'unload_threshold':85,'cache_size_mb':6144,'preloading':True,'large_pages':True,'numa_aware':True,'memory_pool':True,'prefetch_enabled':True,'cache_strategy':'aggressive','defrag_threshold':70,'min_free_gb':4,'defragmentation_enabled':True,'defragmentation_interval':300,'large_page_size_kb':2048,'memory_allocator':'jemalloc','model_cache_policy':'lru','memory_growth_factor':1.5,'preallocated_buffers':True,'memory_limit_gb':24}
		elif memory_info.get('is_32gb',False):return{'max_percent':75,'unload_threshold':85,'cache_size_mb':4096,'preloading':True,'large_pages':True}
		else:total_gb=memory_info.get('total_gb',8);return{'max_percent':min(70,max(50,int(60+(total_gb-8)*1.25))),'unload_threshold':min(80,max(60,int(70+(total_gb-8)*1.25))),'cache_size_mb':min(2048,max(512,int(total_gb*64)))}
	def _optimize_gpu(self)->Dict[str,Any]:
		gpu_info=self.hardware_info['gpu']
		if not gpu_info.get('available',False):return{'enabled':False}
		if gpu_info.get('is_rtx_3080',False):return{'enabled':True,'compute_type':'float16','tensor_cores':True,'cuda_streams':3,'reserved_memory_mb':256,'max_batch_size':16,'memory_fraction':.95,'cudnn_benchmark':True,'trt_optimization':True,'bfloat16_supported':False,'dynamic_mem_mgmt':True,'vram_gb':gpu_info.get('memory_gb',10),'vram_efficient_loading':True,'amp_optimization_level':'O2','cuda_graphs':True,'tensor_precision':'tf32','tf32_allowed':True,'multi_stream_inference':True,'stream_priority':{'high':['inference','training'],'normal':['dataloading'],'low':['garbage_collection']},'texture_cache':True,'asynchronous_execution':True,'cuda_malloc_async':True,'fused_attention':True,'memory_defrag_interval':100,'precision_combo':'mixed_float16','shared_memory_optimization':True,'kernel_autotuning':True,'workspace_memory_limit_mb':1024,'fragmentation_threshold':15,'pre_allocation':True}
		else:memory_gb=gpu_info.get('memory_gb',0);return{'enabled':True,'compute_type':'float16'if memory_gb>=6 else'int8','tensor_cores':'tensor_cores'in gpu_info.get('name','').lower(),'cuda_streams':1,'reserved_memory_mb':min(512,max(128,int(memory_gb*64))),'max_batch_size':min(16,max(4,int(memory_gb/2)))}
	def _optimize_llm(self)->Dict[str,Any]:
		gpu_info=self.hardware_info['gpu']
		if gpu_info.get('is_rtx_3080',False):return{'gpu_layers':32,'precision':'float16','kv_cache_optimization':True,'context_length':8192,'attention_sinks':True,'auto_adjust':True,'tensor_parallel':1,'draft_model':'small','speculative_decoding':True,'llm_rope_scaling':'dynamic','batch_inference':True,'offload_layers':{'enabled':True,'threshold_gb':9.,'cpu_layers':[0,1]},'flash_attention':True,'multi_query_attention':True,'rotary_embedding':True,'gradient_checkpointing':False,'use_kernel_optimizations':True,'bf16_support':False,'streaming_llm':True,'xformers_optimization':True,'vram_efficient_loading':True,'rtx_3080_optimized':True,'tensor_cores_enabled':True,'mixed_precision_enabled':True,'precision_type':'float16','attention_optimization':True}
		elif gpu_info.get('available',False):memory_gb=gpu_info.get('memory_gb',0);return{'gpu_layers':min(40,max(1,int(memory_gb*3.2))),'precision':'float16'if memory_gb>=6 else'int8','context_length':min(8192,max(2048,int(memory_gb*819.2))),'auto_adjust':True}
		else:return{'gpu_layers':0,'precision':'int8','context_length':2048,'auto_adjust':False}
	def _optimize_audio(self)->Dict[str,Any]:
		cpu_info=self.hardware_info['cpu'];gpu_info=self.hardware_info['gpu'];audio_opt={'sample_rate':22050,'chunk_size':1024,'use_gpu':gpu_info.get('available',False),'buffer_size':4096}
		if cpu_info.get('is_ryzen_9_5900x',False):audio_opt.update({'chunk_size':512,'audio_threads':2,'realtime_priority':True,'cpu_affinity':[10,11],'advanced_dsp':True,'simd_optimization':True})
		if gpu_info.get('is_rtx_3080',False):audio_opt.update({'whisper_model':'small','whisper_compute':'float16','cache_models':True,'vad_sensitivity':.6,'cuda_audio_processing':True,'parallel_processing':True,'precision':'mixed','use_tensor_cores':True,'audio_cache_size_mb':512,'tts_batch_size':64,'stt_batch_size':16,'spectral_processing':'gpu'})
		return audio_opt
	def _optimize_input_processing(self)->Dict[str,Any]:
		input_opt={'voice_activation':True,'text_preprocessing':True,'auto_punctuation':True,'noise_reduction':True}
		if self.hardware_info['gpu'].get('is_rtx_3080',False):input_opt.update({'real_time_transcription':True,'parallel_processing':True,'streaming_mode':True,'gpu_accelerated_vad':True,'confidence_threshold':.6,'enhance_partial_results':True,'gpu_accelerated_nlp':True})
		if self.hardware_info['cpu'].get('is_ryzen_9_5900x',False):input_opt.update({'dedicated_audio_thread':True,'wake_word_sensitivity':.7,'input_buffering':'adaptive','voice_activity_min_duration_ms':300})
		return input_opt
	@log_operation(component='HardwareOptimizer')
	@with_error_handling(error_category=ErrorCategory.RESOURCE)
	def optimize_for_current_state(self)->Dict[str,Any]:
		if not self.state_manager:return{}
		try:
			current_state=self.state_manager.get_current_state();profile=self.get_state_specific_profile(current_state)
			if'cpu'in profile:self._apply_cpu_optimizations(profile['cpu'])
			if'gpu'in profile:self._apply_gpu_optimizations(profile['gpu'])
			if'memory'in profile:self._apply_memory_optimizations(profile['memory'])
			self.logger.info(f"Applied optimizations for state: {current_state.name}");return profile
		except Exception as e:self.logger.error(f"Failed to optimize for current state: {e}");return{}
	def _apply_cpu_optimizations(self,cpu_profile:Dict[str,Any])->None:
		try:
			if platform.system()=='Windows':
				import psutil;process=psutil.Process();priority=cpu_profile.get('priority','normal');priority_map={'low':psutil.IDLE_PRIORITY_CLASS,'below_normal':psutil.BELOW_NORMAL_PRIORITY_CLASS,'normal':psutil.NORMAL_PRIORITY_CLASS,'above_normal':psutil.ABOVE_NORMAL_PRIORITY_CLASS,'high':psutil.HIGH_PRIORITY_CLASS,'realtime':psutil.REALTIME_PRIORITY_CLASS}
				if priority in priority_map:process.nice(priority_map[priority])
				affinity_type=cpu_profile.get('affinity')
				if affinity_type=='performance'and self.hardware_info['cpu'].get('is_ryzen_9_5900x',False):process.cpu_affinity(list(range(10)))
				elif affinity_type=='minimal':process.cpu_affinity([0,1,2,3])
		except Exception as e:self.logger.warning(f"Failed to apply CPU optimizations: {e}")
	def _apply_gpu_optimizations(self,gpu_profile:Dict[str,Any])->None:
		try:
			if not gpu_profile.get('enabled',True):return
			import torch
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				if gpu_profile.get('tensor_cores',False):
					if hasattr(torch.backends,'cuda')and hasattr(torch.backends.cuda,'matmul'):torch.backends.cuda.matmul.allow_tf32=True
					if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.allow_tf32=True
				if gpu_profile.get('cudnn_benchmark',False):
					if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.benchmark=True
		except Exception as e:self.logger.warning(f"Failed to apply GPU optimizations: {e}")
	def _apply_memory_optimizations(self,memory_profile:Dict[str,Any])->None:
		try:
			if memory_profile.get('clear_cache',False):
				import gc;gc.collect()
				if platform.system()=='Windows':import ctypes;ctypes.windll.kernel32.SetProcessWorkingSetSize(ctypes.windll.kernel32.GetCurrentProcess(),-1,-1)
		except Exception as e:self.logger.warning(f"Failed to apply memory optimizations: {e}")
	@log_operation(component='HardwareOptimizer')
	@with_error_handling(error_category=ErrorCategory.RESOURCE)
	def setup_gpu(self)->None:
		gpu_info=self.hardware_info['gpu']
		if not gpu_info.get('available',False):self.logger.info('No GPU available for setup');return
		try:
			import torch
			if torch.cuda.is_available():
				torch.cuda.set_device(0);torch.cuda.empty_cache()
				if gpu_info.get('is_rtx_3080',False):
					self.logger.info(f"Configuring GPU optimizations for {gpu_info['name']}")
					if hasattr(torch.backends,'cuda')and hasattr(torch.backends.cuda,'matmul'):torch.backends.cuda.matmul.allow_tf32=True;self.logger.info('Enabled TF32 precision for matrix multiplications')
					if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=True;self.logger.info('Enabled cuDNN benchmark mode and TF32 precision')
					if hasattr(torch.cuda,'memory_stats')and torch.cuda.is_available():os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:512';self.logger.info('Set CUDA memory allocator config')
					if hasattr(torch._C,'_jit_set_profiling_executor'):torch._C._jit_set_profiling_executor(True)
					if hasattr(torch._C,'_jit_set_profiling_mode'):torch._C._jit_set_profiling_mode(True)
					self.logger.info('Configured PyTorch JIT profiling for kernel fusion')
				else:
					device_name=torch.cuda.get_device_name(0);memory_gb=torch.cuda.get_device_properties(0).total_memory/1024**3;self.logger.info(f"GPU configured: {device_name} ({memory_gb:.2f} GB VRAM)")
					if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.benchmark=True;self.logger.info('Enabled cuDNN benchmark mode')
		except ImportError:self.logger.debug('PyTorch not available for GPU setup')
		except Exception as e:self.logger.error(f"Error setting up GPU: {e}")
	@log_operation(component='HardwareOptimizer')
	@with_error_handling(error_category=ErrorCategory.RESOURCE)
	def optimize_for_rtx_3080(self)->Dict[str,Any]:
		optimizations={'applied':False,'settings':{}}
		try:
			import torch
			if torch.cuda.is_available()and'3080'in torch.cuda.get_device_name(0):
				optimizations['settings']['amp_enabled']=True
				if hasattr(torch.cuda,'memory_stats'):optimizations['settings']['memory_stats_enabled']=True
				torch.backends.cudnn.benchmark=True;optimizations['settings']['cudnn_benchmark']=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;optimizations['settings']['tf32_enabled']=True;os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:512';optimizations['settings']['max_split_size_mb']=512;optimizations['settings']['cuda_graphs_enabled']=True;optimizations['settings']['optimization_level']='maximum';optimizations['settings']['memory_efficient_optimizer']=True;optimizations['settings']['channels_last_memory_format']=True
				try:import xformers;optimizations['settings']['xformers_available']=True;optimizations['settings']['flash_attention_enabled']=True
				except ImportError:optimizations['settings']['xformers_available']=False
				optimizations['applied']=True;self.logger.info('Applied RTX 3080-specific optimizations')
			return optimizations
		except ImportError:self.logger.debug('PyTorch not available for GPU optimizations');return optimizations
		except Exception as e:self.logger.error(f"Error applying GPU optimizations: {e}");return optimizations
	@log_operation(component='HardwareOptimizer')
	@with_error_handling(error_category=ErrorCategory.RESOURCE)
	def optimize_for_ryzen_9_5900x(self)->Dict[str,Any]:
		optimizations={'applied':False,'settings':{}}
		try:
			cpu_info=self.hardware_info.get('cpu',{})
			if not cpu_info.get('is_ryzen_9_5900x',False):return optimizations
			if platform.system()=='Windows':
				try:import psutil;process=psutil.Process();process.nice(psutil.HIGH_PRIORITY_CLASS);optimizations['settings']['process_priority']='high';affinity=list(range(12));process.cpu_affinity(affinity);optimizations['settings']['cpu_affinity']=affinity;self.logger.info('Set process priority and CPU affinity for Ryzen 9 5900X')
				except ImportError:self.logger.debug('psutil not available for process optimization')
				except Exception as e:self.logger.warning(f"Error setting process priority/affinity: {e}")
				try:import subprocess;subprocess.run(['powercfg','/s','8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);optimizations['settings']['power_plan']='high_performance';self.logger.info('Set high performance power plan')
				except Exception as e:self.logger.warning(f"Error setting power plan: {e}")
			else:
				try:
					os.nice(-10);optimizations['settings']['nice_level']=-10;self.logger.info('Set process nice level to -10')
					try:
						if os.geteuid()==0:os.system('echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null');optimizations['settings']['cpu_governor']='performance';self.logger.info('Set CPU governor to performance mode')
					except:pass
				except Exception as e:self.logger.warning(f"Error setting process priority: {e}")
			optimizations['settings']['num_threads']=10;optimizations['settings']['thread_priority']='high';optimizations['settings']['memory_first_policy']=True;optimizations['settings']['smt_enabled']=True;optimizations['settings']['numa_aware']=True;optimizations['applied']=True;self.logger.info('Applied Ryzen 9 5900X-specific optimizations');return optimizations
		except Exception as e:self.logger.error(f"Error applying CPU optimizations: {e}");return optimizations