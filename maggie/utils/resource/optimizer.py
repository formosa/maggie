import os,platform
from typing import Dict,Any,Optional,Tuple,List
from loguru import logger
class HardwareOptimizer:
	def __init__(self,hardware_info:Dict[str,Any],config:Dict[str,Any]):self.hardware_info=hardware_info;self.config=config
	def create_optimization_profile(self)->Dict[str,Any]:profile={'threading':self._optimize_threading(),'memory':self._optimize_memory(),'gpu':self._optimize_gpu(),'llm':self._optimize_llm(),'audio':self._optimize_audio()};return profile
	def _optimize_threading(self)->Dict[str,Any]:
		cpu_info=self.hardware_info['cpu']
		if cpu_info.get('is_ryzen_9_5900x',False):return{'max_workers':10,'thread_timeout':30,'worker_affinity':[0,1,2,3,4,5,6,7,8,9],'priority_boost':True,'use_dedicated_threads':True,'numa_aware':True,'power_plan':'High performance','core_parking':False}
		else:physical_cores=cpu_info.get('physical_cores',4);max_workers=max(2,min(physical_cores-2,physical_cores*4//5));return{'max_workers':max_workers,'thread_timeout':30}
	def _optimize_memory(self)->Dict[str,Any]:
		memory_info=self.hardware_info['memory']
		if memory_info.get('is_32gb',False)and memory_info.get('type')=='DDR4'and memory_info.get('speed')=='3200MHz':return{'max_percent':80,'unload_threshold':85,'cache_size_mb':6144,'preloading':True,'large_pages':True,'numa_aware':True}
		elif memory_info.get('is_32gb',False):return{'max_percent':75,'unload_threshold':85,'cache_size_mb':4096,'preloading':True}
		else:total_gb=memory_info.get('total_gb',8);return{'max_percent':min(70,max(50,int(60+(total_gb-8)*1.25))),'unload_threshold':min(80,max(60,int(70+(total_gb-8)*1.25))),'cache_size_mb':min(2048,max(512,int(total_gb*64)))}
	def _optimize_gpu(self)->Dict[str,Any]:
		gpu_info=self.hardware_info['gpu']
		if not gpu_info.get('available',False):return{'enabled':False}
		if gpu_info.get('is_rtx_3080',False):return{'enabled':True,'compute_type':'float16','tensor_cores':True,'cuda_streams':3,'reserved_memory_mb':256,'max_batch_size':16,'memory_fraction':.95,'cudnn_benchmark':True,'trt_optimization':True,'bfloat16_supported':False,'dynamic_mem_mgmt':True,'vram_gb':gpu_info.get('memory_gb',10),'vram_efficient_loading':True,'amp_optimization_level':'O2','cuda_graphs':True,'tensor_precision':'mixed','tf32_allowed':True,'multi_stream_inference':True,'stream_priority':{'high':['inference','training'],'normal':['dataloading'],'low':['garbage_collection']},'texture_cache':True,'asynchronous_execution':True,'cuda_malloc_async':True}
		else:memory_gb=gpu_info.get('memory_gb',0);return{'enabled':True,'compute_type':'float16'if memory_gb>=6 else'int8','tensor_cores':'tensor_cores'in gpu_info.get('name','').lower(),'cuda_streams':1,'reserved_memory_mb':min(512,max(128,int(memory_gb*64))),'max_batch_size':min(16,max(4,int(memory_gb/2)))}
	def _optimize_llm(self)->Dict[str,Any]:
		gpu_info=self.hardware_info['gpu']
		if gpu_info.get('is_rtx_3080',False):return{'gpu_layers':32,'precision':'float16','kv_cache_optimization':True,'context_length':8192,'attention_sinks':True,'auto_adjust':True,'tensor_parallel':1,'draft_model':'small','speculative_decoding':True,'llm_rope_scaling':'dynamic','batch_inference':True,'offload_layers':{'enabled':True,'threshold_gb':9.,'cpu_layers':[0,1]},'flash_attention':True,'multi_query_attention':True,'rotary_embedding':True,'gradient_checkpointing':False,'use_kernel_optimizations':True,'bf16_support':False,'streaming_llm':True,'xformers_optimization':True}
		elif gpu_info.get('available',False):memory_gb=gpu_info.get('memory_gb',0);return{'gpu_layers':min(40,max(1,int(memory_gb*3.2))),'precision':'float16'if memory_gb>=6 else'int8','context_length':min(8192,max(2048,int(memory_gb*819.2))),'auto_adjust':True}
		else:return{'gpu_layers':0,'precision':'int8','context_length':2048,'auto_adjust':False}
	def _optimize_audio(self)->Dict[str,Any]:
		cpu_info=self.hardware_info['cpu'];gpu_info=self.hardware_info['gpu'];audio_opt={'sample_rate':22050,'chunk_size':1024,'use_gpu':gpu_info.get('available',False),'buffer_size':4096}
		if cpu_info.get('is_ryzen_9_5900x',False):audio_opt['chunk_size']=512;audio_opt['audio_threads']=2;audio_opt['realtime_priority']=True;audio_opt['cpu_affinity']=[10,11]
		if gpu_info.get('is_rtx_3080',False):audio_opt['whisper_model']='small';audio_opt['whisper_compute']='float16';audio_opt['cache_models']=True;audio_opt['vad_sensitivity']=.6;audio_opt['cuda_audio_processing']=True;audio_opt['parallel_processing']=True;audio_opt['precision']='mixed';audio_opt['use_tensor_cores']=True;audio_opt['audio_cache_size_mb']=512;audio_opt['tts_batch_size']=64;audio_opt['stt_batch_size']=16
		elif gpu_info.get('available',False):audio_opt['whisper_model']='base';audio_opt['whisper_compute']='float16'if gpu_info.get('memory_gb',0)>=6 else'int8'
		else:audio_opt['whisper_model']='tiny';audio_opt['whisper_compute']='int8'
		return audio_opt
	def setup_gpu(self)->None:
		gpu_info=self.hardware_info['gpu']
		if not gpu_info.get('available',False):logger.info('No GPU available for setup');return
		try:
			import torch
			if torch.cuda.is_available():
				torch.cuda.set_device(0);torch.cuda.empty_cache()
				if gpu_info.get('is_rtx_3080',False):
					logger.info(f"Configuring GPU optimizations for {gpu_info['name']}")
					if hasattr(torch.backends,'cuda')and hasattr(torch.backends.cuda,'matmul'):torch.backends.cuda.matmul.allow_tf32=True;logger.info('Enabled TF32 precision for matrix multiplications')
					if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=True;logger.info('Enabled cuDNN benchmark mode and TF32 precision')
					if hasattr(torch.cuda,'memory_stats')and torch.cuda.is_available():os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:512';logger.info('Set CUDA memory allocator config')
					if hasattr(torch._C,'_jit_set_profiling_executor'):torch._C._jit_set_profiling_executor(True)
					if hasattr(torch._C,'_jit_set_profiling_mode'):torch._C._jit_set_profiling_mode(True)
					logger.info('Configured PyTorch JIT profiling for kernel fusion')
				else:
					device_name=torch.cuda.get_device_name(0);memory_gb=torch.cuda.get_device_properties(0).total_memory/1024**3;logger.info(f"GPU configured: {device_name} ({memory_gb:.2f} GB VRAM)")
					if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.benchmark=True;logger.info('Enabled cuDNN benchmark mode')
		except ImportError:logger.debug('PyTorch not available for GPU setup')
		except Exception as e:logger.error(f"Error setting up GPU: {e}")
	def test_gpu_compatibility(self)->Dict[str,Any]:
		gpu_info=self.hardware_info['gpu'];result={'success':False,'warnings':[],'errors':[],'tests_passed':[]}
		if not gpu_info.get('available',False):result['warnings'].append('No GPU detected, running in CPU-only mode');return result
		try:
			import torch
			if not torch.cuda.is_available():result['warnings'].append('PyTorch reports CUDA not available');return result
			try:
				test_sizes=[(1000,1000),(2000,2000),(4000,4000)]
				for size in test_sizes:test_tensor=torch.ones(size,device='cuda');test_result=torch.matmul(test_tensor,test_tensor);del test_tensor,test_result;torch.cuda.empty_cache()
				result['tests_passed'].append(f"cuda_operations_{size[0]}x{size[0]}");test_tensor=torch.ones(1000,1000,device='cuda',dtype=torch.float16);test_result=torch.matmul(test_tensor,test_tensor);del test_tensor,test_result;torch.cuda.empty_cache();result['tests_passed'].append('half_precision_operations')
			except Exception as e:result['errors'].append(f"Basic CUDA operations failed: {e}");return result
			memory_gb=gpu_info.get('memory_gb',0)
			if memory_gb<8:result['warnings'].append(f"GPU memory ({memory_gb:.1f}GB) is less than recommended 8GB")
			if gpu_info.get('is_rtx_3080',False):
				result['tests_passed'].append('rtx_3080_detected')
				if memory_gb<9.5:result['warnings'].append(f"RTX 3080 VRAM ({memory_gb:.1f}GB) is less than expected 10GB")
				compute_capability=gpu_info.get('compute_capability')
				if compute_capability!=(8,6)and compute_capability!='8.6':result['warnings'].append(f"RTX 3080 compute capability ({compute_capability}) is not 8.6")
				cuda_version=torch.version.cuda
				if not cuda_version.startswith('11.'):result['warnings'].append(f"CUDA version {cuda_version} - version 11.x recommended for RTX 3080")
				try:
					a=torch.randn(4096,4096,device='cuda',dtype=torch.float16);b=torch.randn(4096,4096,device='cuda',dtype=torch.float16)
					if hasattr(torch.backends,'cuda')and hasattr(torch.backends.cuda,'matmul'):torch.backends.cuda.matmul.allow_tf32=True
					if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.allow_tf32=True
					c=torch.matmul(a,b);del a,b,c;torch.cuda.empty_cache();result['tests_passed'].append('tensor_core_operations')
				except Exception as e:result['warnings'].append(f"Tensor core operations test failed: {e}")
				try:import xformers,xformers.ops;result['tests_passed'].append('xformers_available')
				except ImportError:result['warnings'].append('xformers not installed, some optimizations will not be available')
			result['success']=True;return result
		except ImportError as e:result['errors'].append(f"PyTorch not available: {e}");result['warnings'].append('Install PyTorch with: pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118');return result
		except Exception as e:result['errors'].append(f"Error testing GPU compatibility: {e}");return result
	def optimize_for_rtx_3080(self)->Dict[str,Any]:
		optimizations={'applied':False,'settings':{}}
		try:
			import torch
			if torch.cuda.is_available()and'3080'in torch.cuda.get_device_name(0):
				optimizations['settings']['amp_enabled']=True
				if hasattr(torch.cuda,'memory_stats'):optimizations['settings']['memory_stats_enabled']=True
				torch.backends.cudnn.benchmark=True;optimizations['settings']['cudnn_benchmark']=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;optimizations['settings']['tf32_enabled']=True;os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:512';optimizations['settings']['max_split_size_mb']=512;optimizations['settings']['cuda_graphs_enabled']=True;optimizations['settings']['optimization_level']='maximum';optimizations['settings']['memory_efficient_optimizer']=True;optimizations['settings']['channels_last_memory_format']=True
				try:import xformers,xformers.ops;optimizations['settings']['xformers_available']=True;optimizations['settings']['flash_attention_enabled']=True
				except ImportError:optimizations['settings']['xformers_available']=False
				optimizations['applied']=True;logger.info('Applied RTX 3080-specific optimizations')
			return optimizations
		except ImportError:logger.debug('PyTorch not available for GPU optimizations');return optimizations
		except Exception as e:logger.error(f"Error applying GPU optimizations: {e}");return optimizations
	def optimize_for_ryzen_9_5900x(self)->Dict[str,Any]:
		optimizations={'applied':False,'settings':{}}
		try:
			cpu_info=self.hardware_info.get('cpu',{})
			if not cpu_info.get('is_ryzen_9_5900x',False):return optimizations
			if platform.system()=='Windows':
				try:import psutil;process=psutil.Process();process.nice(psutil.HIGH_PRIORITY_CLASS);optimizations['settings']['process_priority']='high';affinity=list(range(12));process.cpu_affinity(affinity);optimizations['settings']['cpu_affinity']=affinity;logger.info('Set process priority and CPU affinity for Ryzen 9 5900X')
				except ImportError:logger.debug('psutil not available for process optimization')
				except Exception as e:logger.warning(f"Error setting process priority/affinity: {e}")
				try:import subprocess;subprocess.run(['powercfg','/s','8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);optimizations['settings']['power_plan']='high_performance';logger.info('Set high performance power plan')
				except Exception as e:logger.warning(f"Error setting power plan: {e}")
			else:
				try:
					os.nice(-10);optimizations['settings']['nice_level']=-10;logger.info('Set process nice level to -10')
					try:
						if os.geteuid()==0:os.system('echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null');optimizations['settings']['cpu_governor']='performance';logger.info('Set CPU governor to performance mode')
					except:pass
				except Exception as e:logger.warning(f"Error setting process priority: {e}")
			optimizations['settings']['num_threads']=10;optimizations['settings']['thread_priority']='high';optimizations['settings']['memory_first_policy']=True;optimizations['applied']=True;logger.info('Applied Ryzen 9 5900X-specific optimizations');return optimizations
		except Exception as e:logger.error(f"Error applying CPU optimizations: {e}");return optimizations