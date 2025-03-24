import os,platform,threading,time
from typing import Dict,Any,Optional,List,Tuple,Set
import psutil
from loguru import logger
class ResourceManager:
	def __init__(self,config:Dict[str,Any]):self.config=config;self._lock=threading.RLock();self.hardware_info=self._detect_system();self.optimization_profile=self._create_optimization_profile();self._monitoring_enabled=False;self._monitoring_thread=None;self._resource_history_max_samples=60;self._cpu_history=[];self._memory_history=[];self._gpu_memory_history=[];self.cpu_config=config.get('cpu',{});self.memory_config=config.get('memory',{});self.gpu_config=config.get('gpu',{});self.memory_max_percent=self.memory_config.get('max_percent',75);self.memory_unload_threshold=self.memory_config.get('model_unload_threshold',85);self.gpu_max_percent=self.gpu_config.get('max_percent',90);self.gpu_unload_threshold=self.gpu_config.get('model_unload_threshold',95)
	def _detect_system(self)->Dict[str,Any]:
		system_info={'os':{'system':platform.system(),'release':platform.release(),'version':platform.version()},'cpu':self._detect_cpu(),'memory':self._detect_memory(),'gpu':self._detect_gpu(),'disk':self._detect_disk()};logger.info(f"Detected OS: {system_info['os']['system']} {system_info['os']['release']}")
		if system_info['cpu']['is_ryzen_9_5900x']:logger.info('Detected AMD Ryzen 9 5900X CPU - applying optimized settings')
		else:logger.info(f"Detected CPU: {system_info['cpu']['model']} with {system_info['cpu']['physical_cores']} cores")
		if system_info['gpu']['is_rtx_3080']:logger.info('Detected NVIDIA RTX 3080 GPU - applying optimized settings')
		elif system_info['gpu']['available']:logger.info(f"Detected GPU: {system_info['gpu']['name']} with {system_info['gpu']['memory_gb']:.2f}GB VRAM")
		else:logger.warning('No compatible GPU detected - some features may be limited')
		return system_info
	def _detect_cpu(self)->Dict[str,Any]:
		try:
			cpu_info={'physical_cores':psutil.cpu_count(logical=False),'logical_cores':psutil.cpu_count(logical=True),'model':platform.processor(),'is_ryzen_9_5900x':False,'frequency_mhz':{'current':0,'min':0,'max':0}};model_lower=cpu_info['model'].lower()
			if'ryzen 9'in model_lower and'5900x'in model_lower:cpu_info['is_ryzen_9_5900x']=True
			try:
				cpu_freq=psutil.cpu_freq()
				if cpu_freq:cpu_info['frequency_mhz']={'current':cpu_freq.current,'min':cpu_freq.min,'max':cpu_freq.max}
			except Exception as e:logger.debug(f"Error getting CPU frequency: {e}")
			if platform.system()=='Windows':
				try:
					import wmi;c=wmi.WMI()
					for processor in c.Win32_Processor():
						if'Ryzen 9 5900X'in processor.Name:cpu_info['is_ryzen_9_5900x']=True;cpu_info['model']=processor.Name;cpu_info['frequency_mhz']['max']=processor.MaxClockSpeed;break
				except ImportError:logger.debug('WMI module not available for detailed CPU detection')
			return cpu_info
		except Exception as e:logger.error(f"Error detecting CPU: {e}");return{'physical_cores':0,'logical_cores':0,'model':'Unknown','is_ryzen_9_5900x':False,'error':str(e)}
	def _detect_memory(self)->Dict[str,Any]:
		try:
			memory=psutil.virtual_memory();memory_info={'total_bytes':memory.total,'total_gb':memory.total/1024**3,'available_bytes':memory.available,'available_gb':memory.available/1024**3,'percent_used':memory.percent,'is_32gb':30<=memory.total/1024**3<=34}
			if platform.system()=='Windows':
				try:
					import wmi;c=wmi.WMI()
					for physical_memory in c.Win32_PhysicalMemory():
						if hasattr(physical_memory,'PartNumber')and physical_memory.PartNumber:
							if'DDR4'in physical_memory.PartNumber:
								memory_info['type']='DDR4'
								if'3200'in physical_memory.PartNumber:memory_info['speed']='3200MHz'
								break
				except ImportError:logger.debug('WMI module not available for detailed memory detection')
			return memory_info
		except Exception as e:logger.error(f"Error detecting memory: {e}");return{'total_bytes':0,'total_gb':0,'available_bytes':0,'available_gb':0,'percent_used':0,'is_32gb':False,'error':str(e)}
	def _detect_gpu(self)->Dict[str,Any]:
		gpu_info={'available':False,'name':None,'memory_gb':None,'cuda_version':None,'is_rtx_3080':False,'driver_version':None,'architectures':[]}
		try:
			import torch
			if torch.cuda.is_available():
				gpu_info['available']=True;gpu_info['name']=torch.cuda.get_device_name(0);gpu_info['memory_gb']=torch.cuda.get_device_properties(0).total_memory/1024**3;gpu_info['cuda_version']=torch.version.cuda
				if'3080'in gpu_info['name']:
					gpu_info['is_rtx_3080']=True;gpu_info['compute_capability']=torch.cuda.get_device_capability(0);gpu_info['tensor_cores']=True;gpu_info['optimal_precision']='float16';gpu_info['architectures']=['Ampere']
					if platform.system()=='Windows':
						try:import pynvml;pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0);info=pynvml.nvmlDeviceGetMemoryInfo(handle);gpu_info['memory_gb']=info.total/1024**3;gpu_info['driver_version']=pynvml.nvmlSystemGetDriverVersion();pynvml.nvmlShutdown()
						except ImportError:logger.debug('PYNVML module not available for detailed GPU detection')
				gpu_info['memory_free_gb']=(torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_allocated(0)-torch.cuda.memory_reserved(0))/1024**3
		except ImportError:logger.debug('PyTorch not available for GPU detection')
		except Exception as e:logger.error(f"Error detecting GPU: {e}")
		return gpu_info
	def _detect_disk(self)->Dict[str,Any]:
		disk_info={'root_free_gb':0,'models_free_gb':0,'is_ssd':False}
		try:
			root_usage=psutil.disk_usage('/');disk_info['root_free_gb']=root_usage.free/1024**3;models_dir=os.path.abspath('models')
			if os.path.exists(models_dir):models_usage=psutil.disk_usage(models_dir);disk_info['models_free_gb']=models_usage.free/1024**3
		except Exception as e:logger.error(f"Error detecting disk space: {e}")
		if platform.system()=='Windows':
			try:
				import wmi;c=wmi.WMI()
				for disk in c.Win32_DiskDrive():
					model=disk.Model.lower()if disk.Model else''
					if any(x in model for x in['ssd','nvme','m.2']):disk_info['is_ssd']=True;disk_info['model']=disk.Model;break
			except ImportError:logger.debug('WMI module not available for SSD detection')
		return disk_info
	def _create_optimization_profile(self)->Dict[str,Any]:profile={'threading':self._optimize_threading(),'memory':self._optimize_memory(),'gpu':self._optimize_gpu(),'llm':self._optimize_llm(),'audio':self._optimize_audio()};return profile
	def _optimize_threading(self)->Dict[str,Any]:
		cpu_info=self.hardware_info['cpu']
		if cpu_info['is_ryzen_9_5900x']:return{'max_workers':10,'thread_timeout':30,'worker_affinity':[0,1,2,3,4,5,6,7,8,9],'priority_boost':True}
		else:physical_cores=cpu_info.get('physical_cores',4);max_workers=max(2,min(physical_cores-2,physical_cores*4//5));return{'max_workers':max_workers,'thread_timeout':30}
	def _optimize_memory(self)->Dict[str,Any]:
		memory_info=self.hardware_info['memory']
		if memory_info.get('is_32gb',False)and memory_info.get('type')=='DDR4'and memory_info.get('speed')=='3200MHz':return{'max_percent':80,'unload_threshold':85,'cache_size_mb':6144,'preloading':True,'large_pages':True,'numa_aware':True}
		elif memory_info.get('is_32gb',False):return{'max_percent':75,'unload_threshold':85,'cache_size_mb':4096,'preloading':True}
		else:total_gb=memory_info.get('total_gb',8);return{'max_percent':min(70,max(50,int(60+(total_gb-8)*1.25))),'unload_threshold':min(80,max(60,int(70+(total_gb-8)*1.25))),'cache_size_mb':min(2048,max(512,int(total_gb*64)))}
	def _optimize_gpu(self)->Dict[str,Any]:
		gpu_info=self.hardware_info['gpu']
		if not gpu_info.get('available',False):return{'enabled':False}
		if gpu_info.get('is_rtx_3080',False):return{'enabled':True,'compute_type':'float16','tensor_cores':True,'cuda_streams':3,'reserved_memory_mb':256,'max_batch_size':16,'memory_fraction':.95,'cudnn_benchmark':True,'trt_optimization':True,'bfloat16_supported':False,'dynamic_mem_mgmt':True,'vram_gb':gpu_info.get('memory_gb',10),'vram_efficient_loading':True,'amp_optimization_level':'O2'}
		else:memory_gb=gpu_info.get('memory_gb',0);return{'enabled':True,'compute_type':'float16'if memory_gb>=6 else'int8','tensor_cores':'tensor_cores'in gpu_info.get('name','').lower(),'cuda_streams':1,'reserved_memory_mb':min(512,max(128,int(memory_gb*64))),'max_batch_size':min(16,max(4,int(memory_gb/2)))}
	def _optimize_llm(self)->Dict[str,Any]:
		gpu_info=self.hardware_info['gpu']
		if gpu_info.get('is_rtx_3080',False):return{'gpu_layers':32,'precision':'float16','kv_cache_optimization':True,'context_length':8192,'attention_sinks':True,'auto_adjust':True,'tensor_parallel':1,'draft_model':'small','speculative_decoding':True,'llm_rope_scaling':'dynamic','batch_inference':True,'offload_layers':{'enabled':True,'threshold_gb':9.,'cpu_layers':[0,1]}}
		elif gpu_info.get('available',False):memory_gb=gpu_info.get('memory_gb',0);return{'gpu_layers':min(40,max(1,int(memory_gb*3.2))),'precision':'float16'if memory_gb>=6 else'int8','context_length':min(8192,max(2048,int(memory_gb*819.2))),'auto_adjust':True}
		else:return{'gpu_layers':0,'precision':'int8','context_length':2048,'auto_adjust':False}
	def _optimize_audio(self)->Dict[str,Any]:
		cpu_info=self.hardware_info['cpu'];gpu_info=self.hardware_info['gpu'];audio_opt={'sample_rate':22050,'chunk_size':1024,'use_gpu':gpu_info.get('available',False),'buffer_size':4096}
		if cpu_info.get('is_ryzen_9_5900x',False):audio_opt['chunk_size']=512;audio_opt['audio_threads']=2
		if gpu_info.get('is_rtx_3080',False):audio_opt['whisper_model']='small';audio_opt['whisper_compute']='float16';audio_opt['cache_models']=True;audio_opt['vad_sensitivity']=.6;audio_opt['cuda_audio_processing']=True
		elif gpu_info.get('available',False):audio_opt['whisper_model']='base';audio_opt['whisper_compute']='float16'if gpu_info.get('memory_gb',0)>=6 else'int8'
		else:audio_opt['whisper_model']='tiny';audio_opt['whisper_compute']='int8'
		return audio_opt
	def setup_gpu(self)->None:
		gpu_info=self.hardware_info['gpu']
		if not gpu_info.get('available',False):return
		try:
			import torch
			if torch.cuda.is_available():
				torch.cuda.set_device(0);torch.cuda.empty_cache()
				if gpu_info.get('is_rtx_3080',False):
					if hasattr(torch.cuda,'amp')and hasattr(torch.cuda.amp,'autocast'):logger.info('Enabling automatic mixed precision for CUDA')
					if hasattr(torch.backends,'cuda')and hasattr(torch.backends.cuda,'matmul'):torch.backends.cuda.matmul.allow_tf32=True;logger.info('Enabled TF32 precision for matrix multiplications')
					if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.benchmark=True;torch.backends.cudnn.allow_tf32=True;logger.info('Enabled cuDNN benchmark mode and TF32 precision')
				device_name=torch.cuda.get_device_name(0);memory_gb=torch.cuda.get_device_properties(0).total_memory/1024**3;logger.info(f"GPU configured: {device_name} ({memory_gb:.2f} GB VRAM)")
		except ImportError:logger.debug('PyTorch not available for GPU setup')
		except Exception as e:logger.error(f"Error setting up GPU: {e}")
	def start_monitoring(self,interval:float=5.)->bool:
		with self._lock:
			if self._monitoring_enabled:return True
			self._monitoring_enabled=True;self._monitoring_thread=threading.Thread(target=self._monitor_resources,args=(interval,),daemon=True,name='ResourceMonitorThread');self._monitoring_thread.start();logger.info(f"Resource monitoring started with {interval}s interval");return True
	def stop_monitoring(self)->bool:
		with self._lock:
			if not self._monitoring_enabled:return False
			self._monitoring_enabled=False
			if self._monitoring_thread:self._monitoring_thread.join(timeout=2.)
			logger.info('Resource monitoring stopped');return True
	def _monitor_resources(self,interval:float)->None:
		try:from maggie.utils.service_locator import ServiceLocator;event_bus=ServiceLocator.get('event_bus')
		except ImportError:logger.warning('ServiceLocator not available - event publishing disabled');event_bus=None
		while self._monitoring_enabled:
			try:
				cpu_percent=psutil.cpu_percent(percpu=True);cpu_avg=sum(cpu_percent)/len(cpu_percent)if cpu_percent else 0;memory=psutil.virtual_memory()
				with self._lock:
					self._cpu_history.append(cpu_avg);self._memory_history.append(memory.percent)
					if len(self._cpu_history)>self._resource_history_max_samples:self._cpu_history.pop(0)
					if len(self._memory_history)>self._resource_history_max_samples:self._memory_history.pop(0)
				gpu_util=self._get_gpu_utilization()
				if gpu_util and'memory_percent'in gpu_util:
					with self._lock:
						self._gpu_memory_history.append(gpu_util['memory_percent'])
						if len(self._gpu_memory_history)>self._resource_history_max_samples:self._gpu_memory_history.pop(0)
				self._check_resource_thresholds(cpu_avg,memory,gpu_util,event_bus);self._check_resource_trends(event_bus);time.sleep(interval)
			except Exception as e:logger.error(f"Error monitoring resources: {e}");time.sleep(interval)
	def _get_gpu_utilization(self)->Optional[Dict[str,float]]:
		if not self.hardware_info['gpu'].get('available',False):return None
		try:
			import torch
			if torch.cuda.is_available():
				allocated=torch.cuda.memory_allocated(0);reserved=torch.cuda.memory_reserved(0);total=torch.cuda.get_device_properties(0).total_memory;memory_percent=allocated/total*100;reserved_percent=reserved/total*100;active_allocations=0
				if hasattr(torch.cuda,'memory_stats'):stats=torch.cuda.memory_stats(0);active_allocations=stats.get('num_alloc_retries',0)
				return{'memory_allocated':allocated,'memory_reserved':reserved,'memory_total':total,'memory_percent':memory_percent,'reserved_percent':reserved_percent,'active_allocations':active_allocations,'fragmentation':(reserved-allocated)/total*100 if reserved>allocated else 0}
		except ImportError:pass
		except Exception as e:logger.error(f"Error getting GPU utilization: {e}")
		return None
	def _check_resource_thresholds(self,cpu_percent:float,memory:psutil._psplatform.svmem,gpu_util:Optional[Dict[str,float]],event_bus:Any)->None:
		if memory.percent>self.memory_unload_threshold:
			logger.warning(f"High memory usage: {memory.percent:.1f}% "+f"(threshold: {self.memory_unload_threshold}%) - "+f"available: {memory.available/1024**3:.1f} GB")
			if event_bus:event_bus.publish('low_memory_warning',{'percent':memory.percent,'available_gb':memory.available/1024**3})
		if self.hardware_info['cpu'].get('is_ryzen_9_5900x',False):
			per_core=psutil.cpu_percent(percpu=True);max_core=max(per_core)if per_core else 0;cores_above_95=sum(1 for core in per_core if core>95)
			if cores_above_95>=4:logger.warning(f"High load on {cores_above_95} CPU cores (>95% usage)")
			if cpu_percent>85:logger.warning(f"High overall CPU usage: {cpu_percent:.1f}%")
		elif cpu_percent>90:logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
		if gpu_util:
			if self.hardware_info['gpu'].get('is_rtx_3080',False):
				if gpu_util['memory_percent']>self.gpu_unload_threshold:
					logger.warning(f"High GPU memory usage: {gpu_util['memory_percent']:.1f}% - "+f"allocated: {gpu_util['memory_allocated']/1024**3:.1f} GB")
					if event_bus:event_bus.publish('gpu_memory_warning',{'percent':gpu_util['memory_percent'],'allocated_gb':gpu_util['memory_allocated']/1024**3})
	def _check_resource_trends(self,event_bus:Any)->None:
		with self._lock:
			if len(self._memory_history)<10:return
			if all(self._memory_history[i]<=self._memory_history[i+1]for i in range(len(self._memory_history)-10,len(self._memory_history)-1)):
				if self._memory_history[-1]-self._memory_history[-10]>10:
					logger.warning('Memory usage steadily increasing - possible memory leak')
					if event_bus:event_bus.publish('memory_leak_warning',{'increase':self._memory_history[-1]-self._memory_history[-10],'current':self._memory_history[-1]})
			if len(self._gpu_memory_history)>=10:
				if all(self._gpu_memory_history[i]<=self._gpu_memory_history[i+1]for i in range(len(self._gpu_memory_history)-10,len(self._gpu_memory_history)-1)):
					if self._gpu_memory_history[-1]-self._gpu_memory_history[-10]>15:
						logger.warning('GPU memory usage steadily increasing - possible CUDA memory leak')
						if event_bus:event_bus.publish('gpu_memory_leak_warning',{'increase':self._gpu_memory_history[-1]-self._gpu_memory_history[-10],'current':self._gpu_memory_history[-1]})
	def clear_gpu_memory(self)->bool:
		try:
			import torch
			if torch.cuda.is_available():torch.cuda.empty_cache();logger.debug('GPU memory cache cleared');return True
			return False
		except ImportError:logger.debug('PyTorch not available for clearing GPU cache');return False
		except Exception as e:logger.error(f"Error clearing GPU cache: {e}");return False
	def reduce_memory_usage(self)->bool:
		success=True
		try:import gc;gc.collect();logger.debug('Python garbage collection performed')
		except Exception as e:logger.error(f"Error performing garbage collection: {e}");success=False
		if not self.clear_gpu_memory():success=False
		return success
	def release_resources(self)->bool:
		success=True
		if not self.reduce_memory_usage():success=False
		try:
			if platform.system()=='Windows':import psutil;psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS);logger.debug('Process priority reset to normal')
		except Exception as e:logger.error(f"Error releasing system resources: {e}");success=False
		return success
	def get_resource_status(self)->Dict[str,Any]:
		status={}
		try:cpu_percent=psutil.cpu_percent(percpu=True);status['cpu']={'total_percent':sum(cpu_percent)/len(cpu_percent)if cpu_percent else 0,'per_core':cpu_percent,'cores':len(cpu_percent)}
		except Exception as e:logger.error(f"Error getting CPU status: {e}");status['cpu']={'error':str(e)}
		try:memory=psutil.virtual_memory();status['memory']={'total_gb':memory.total/1024**3,'available_gb':memory.available/1024**3,'used_percent':memory.percent}
		except Exception as e:logger.error(f"Error getting memory status: {e}");status['memory']={'error':str(e)}
		gpu_util=self._get_gpu_utilization()
		if gpu_util:status['gpu']=gpu_util
		else:status['gpu']={'available':False}
		return status
	def optimize_for_rtx_3080(self)->Dict[str,Any]:
		optimizations={'applied':False,'settings':{}}
		if not self.hardware_info['gpu'].get('is_rtx_3080',False):return optimizations
		try:
			import torch
			if torch.cuda.is_available()and'3080'in torch.cuda.get_device_name(0):
				optimizations['settings']['amp_enabled']=True
				if hasattr(torch.cuda,'memory_stats'):optimizations['settings']['memory_stats_enabled']=True
				torch.backends.cudnn.benchmark=True;optimizations['settings']['cudnn_benchmark']=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;optimizations['settings']['tf32_enabled']=True;optimizations['settings']['max_split_size_mb']=512;optimizations['settings']['cuda_graphs_enabled']=True;optimizations['settings']['cuda_streams']=4;optimizations['applied']=True;logger.info('Applied RTX 3080-specific optimizations')
			return optimizations
		except ImportError:logger.debug('PyTorch not available for GPU optimizations');return optimizations
		except Exception as e:logger.error(f"Error applying GPU optimizations: {e}");return optimizations