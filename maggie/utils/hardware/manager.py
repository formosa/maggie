import os,platform,subprocess,json,threading,time
from typing import Dict,Any,Optional,List,Tuple
import psutil
from loguru import logger
__all__=['HardwareManager']
class HardwareManager:
	def __init__(self,config_path:str='config.yaml'):self.config_path=config_path;self.hardware_info=self._detect_system();self.optimization_profile=self._create_optimization_profile();self._monitoring_enabled=False;self._monitoring_thread=None;self._cpu_history=[];self._memory_history=[];self._gpu_memory_history=[];self._resource_history_max_samples=60
	def _detect_system(self)->Dict[str,Any]:
		system_info={'os':{'system':platform.system(),'release':platform.release(),'version':platform.version()},'cpu':self._detect_cpu(),'memory':self._detect_memory(),'gpu':self._detect_gpu(),'disk':self._detect_disk()};logger.info(f"Detected OS: {system_info['os']['system']} {system_info['os']['release']}")
		if system_info['cpu']['is_ryzen_9_5900x']:logger.info('Detected AMD Ryzen 9 5900X CPU - applying optimized settings')
		else:logger.info(f"Detected CPU: {system_info['cpu']['model']} with {system_info['cpu']['physical_cores']} cores")
		if system_info['gpu']['is_rtx_3080']:logger.info('Detected NVIDIA RTX 3080 GPU - applying optimized settings')
		elif system_info['gpu']['available']:logger.info(f"Detected GPU: {system_info['gpu']['name']} with {system_info['gpu']['memory_gb']:.2f}GB VRAM")
		else:logger.warning('No compatible GPU detected - some features may be limited')
		return system_info
	def _detect_cpu(self)->Dict[str,Any]:
		cpu_info={'physical_cores':psutil.cpu_count(logical=False),'logical_cores':psutil.cpu_count(logical=True),'model':platform.processor(),'is_ryzen_9_5900x':False,'frequency_mhz':{'current':0,'min':0,'max':0}};model_lower=cpu_info['model'].lower()
		if'ryzen 9'in model_lower and'5900x'in model_lower:cpu_info['is_ryzen_9_5900x']=True
		try:
			cpu_freq=psutil.cpu_freq()
			if cpu_freq:cpu_info['frequency_mhz']={'current':cpu_freq.current,'min':cpu_freq.min,'max':cpu_freq.max}
		except:pass
		if platform.system()=='Windows':
			try:
				import wmi;c=wmi.WMI()
				for processor in c.Win32_Processor():
					if'Ryzen 9 5900X'in processor.Name:cpu_info['is_ryzen_9_5900x']=True;cpu_info['model']=processor.Name;cpu_info['frequency_mhz']['max']=processor.MaxClockSpeed;break
			except ImportError:logger.debug('WMI module not available for detailed CPU detection')
		return cpu_info
	def _detect_memory(self)->Dict[str,Any]:
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
						except ImportError:pass
				gpu_info['memory_free_gb']=(torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_allocated(0)-torch.cuda.memory_reserved(0))/1024**3
		except ImportError:logger.debug('PyTorch not available for GPU detection')
		return gpu_info
	def _detect_disk(self)->Dict[str,Any]:
		disk_info={'root_free_gb':0,'models_free_gb':0,'is_ssd':False}
		try:root_usage=psutil.disk_usage('/');disk_info['root_free_gb']=root_usage.free/1024**3
		except:pass
		try:
			models_dir=os.path.abspath('models')
			if os.path.exists(models_dir):models_usage=psutil.disk_usage(models_dir);disk_info['models_free_gb']=models_usage.free/1024**3
		except:pass
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
		else:physical_cores=cpu_info['physical_cores'];max_workers=max(2,min(physical_cores-2,physical_cores*4//5));return{'max_workers':max_workers,'thread_timeout':30}
	def _optimize_memory(self)->Dict[str,Any]:
		memory_info=self.hardware_info['memory']
		if memory_info['is_32gb']and memory_info.get('type')=='DDR4'and memory_info.get('speed')=='3200MHz':return{'max_percent':80,'unload_threshold':85,'cache_size_mb':6144,'preloading':True,'large_pages':True,'numa_aware':True}
		elif memory_info['is_32gb']:return{'max_percent':75,'unload_threshold':85,'cache_size_mb':4096,'preloading':True}
		else:total_gb=memory_info['total_gb'];return{'max_percent':min(70,max(50,int(60+(total_gb-8)*1.25))),'unload_threshold':min(80,max(60,int(70+(total_gb-8)*1.25))),'cache_size_mb':min(2048,max(512,int(total_gb*64)))}
	def _optimize_gpu(self)->Dict[str,Any]:
		gpu_info=self.hardware_info['gpu']
		if not gpu_info['available']:return{'enabled':False}
		if gpu_info['is_rtx_3080']:return{'enabled':True,'compute_type':'float16','tensor_cores':True,'cuda_streams':3,'reserved_memory_mb':256,'max_batch_size':16,'memory_fraction':.95,'cudnn_benchmark':True,'trt_optimization':True,'bfloat16_supported':False,'dynamic_mem_mgmt':True,'vram_gb':gpu_info.get('memory_gb',10),'vram_efficient_loading':True,'amp_optimization_level':'O2'}
		else:memory_gb=gpu_info.get('memory_gb',0);return{'enabled':True,'compute_type':'float16'if memory_gb>=6 else'int8','tensor_cores':'tensor_cores'in gpu_info.get('name','').lower(),'cuda_streams':1,'reserved_memory_mb':min(512,max(128,int(memory_gb*64))),'max_batch_size':min(16,max(4,int(memory_gb/2)))}
	def _optimize_llm(self)->Dict[str,Any]:
		gpu_info=self.hardware_info['gpu']
		if gpu_info['is_rtx_3080']:return{'gpu_layers':32,'precision':'float16','kv_cache_optimization':True,'context_length':8192,'attention_sinks':True,'auto_adjust':True,'tensor_parallel':1,'draft_model':'small','speculative_decoding':True,'llm_rope_scaling':'dynamic','batch_inference':True,'offload_layers':{'enabled':True,'threshold_gb':9.,'cpu_layers':[0,1]}}
		elif gpu_info['available']:memory_gb=gpu_info.get('memory_gb',0);return{'gpu_layers':min(40,max(1,int(memory_gb*3.2))),'precision':'float16'if memory_gb>=6 else'int8','context_length':min(8192,max(2048,int(memory_gb*819.2))),'auto_adjust':True}
		else:return{'gpu_layers':0,'precision':'int8','context_length':2048,'auto_adjust':False}
	def _optimize_audio(self)->Dict[str,Any]:
		cpu_info=self.hardware_info['cpu'];gpu_info=self.hardware_info['gpu'];audio_opt={'sample_rate':22050,'chunk_size':1024,'use_gpu':gpu_info['available'],'buffer_size':4096}
		if cpu_info['is_ryzen_9_5900x']:audio_opt['chunk_size']=512;audio_opt['audio_threads']=2
		if gpu_info['is_rtx_3080']:audio_opt['whisper_model']='small';audio_opt['whisper_compute']='float16';audio_opt['cache_models']=True;audio_opt['vad_sensitivity']=.6;audio_opt['cuda_audio_processing']=True
		elif gpu_info['available']:audio_opt['whisper_model']='base';audio_opt['whisper_compute']='float16'if gpu_info.get('memory_gb',0)>=6 else'int8'
		else:audio_opt['whisper_model']='tiny';audio_opt['whisper_compute']='int8'
		return audio_opt
	def apply_hardware_optimizations(self,hardware_info:Dict[str,Any])->None:self._apply_cpu_optimizations(hardware_info.get('cpu',{}));self._apply_memory_optimizations(hardware_info.get('memory',{}));self._apply_gpu_optimizations(hardware_info.get('gpu',{}))
	def start_monitoring(self,interval:float=5.)->bool:
		if self._monitoring_enabled:return True
		self._monitoring_enabled=True;self._monitoring_thread=threading.Thread(target=self._monitor_resources,args=(interval,),daemon=True,name='ResourceMonitorThread');self._monitoring_thread.start();logger.info(f"Resource monitoring started with {interval}s interval");return True
	def stop_monitoring(self)->bool:
		if not self._monitoring_enabled:return False
		self._monitoring_enabled=False
		if self._monitoring_thread:self._monitoring_thread.join(timeout=2.)
		logger.info('Resource monitoring stopped');return True
	def _monitor_resources(self,interval:float)->None:
		while self._monitoring_enabled:
			try:
				cpu_percent=psutil.cpu_percent(percpu=True);cpu_avg=sum(cpu_percent)/len(cpu_percent)if cpu_percent else 0;memory=psutil.virtual_memory();self._cpu_history.append(cpu_avg);self._memory_history.append(memory.percent)
				if len(self._cpu_history)>self._resource_history_max_samples:self._cpu_history.pop(0)
				if len(self._memory_history)>self._resource_history_max_samples:self._memory_history.pop(0)
				gpu_util=self._get_gpu_utilization()
				if gpu_util and'memory_percent'in gpu_util:
					self._gpu_memory_history.append(gpu_util['memory_percent'])
					if len(self._gpu_memory_history)>self._resource_history_max_samples:self._gpu_memory_history.pop(0)
				self._check_resource_thresholds(cpu_avg,memory,gpu_util);self._check_resource_trends();time.sleep(interval)
			except Exception as e:logger.error(f"Error monitoring resources: {e}");time.sleep(interval)
	def _get_gpu_utilization(self)->Optional[Dict[str,float]]:
		if not self.hardware_info['gpu']['available']:return None
		try:
			import torch
			if torch.cuda.is_available():
				allocated=torch.cuda.memory_allocated(0);reserved=torch.cuda.memory_reserved(0);total=torch.cuda.get_device_properties(0).total_memory;memory_percent=allocated/total*100;reserved_percent=reserved/total*100;active_allocations=0
				if hasattr(torch.cuda,'memory_stats'):stats=torch.cuda.memory_stats(0);active_allocations=stats.get('num_alloc_retries',0)
				return{'memory_allocated':allocated,'memory_reserved':reserved,'memory_total':total,'memory_percent':memory_percent,'reserved_percent':reserved_percent,'active_allocations':active_allocations,'fragmentation':(reserved-allocated)/total*100 if reserved>allocated else 0}
		except ImportError:pass
		except Exception as e:logger.error(f"Error getting GPU utilization: {e}")
		return None
	def _check_resource_thresholds(self,cpu_percent:float,memory:psutil._psplatform.svmem,gpu_util:Optional[Dict[str,float]])->None:
		memory_threshold=self.optimization_profile['memory']['unload_threshold']
		if self.hardware_info['cpu'].get('is_ryzen_9_5900x',False):
			per_core=psutil.cpu_percent(percpu=True);max_core=max(per_core)if per_core else 0;cores_above_95=sum(1 for core in per_core if core>95)
			if cores_above_95>=4:logger.warning(f"High load on {cores_above_95} CPU cores (>95% usage)")
			if cpu_percent>85:logger.warning(f"High overall CPU usage: {cpu_percent:.1f}%")
		elif cpu_percent>90:logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
		if memory.percent>memory_threshold:logger.warning(f"High memory usage: {memory.percent:.1f}% "+f"(threshold: {memory_threshold}%) - "+f"available: {memory.available/1024**3:.1f} GB")
		if gpu_util:
			if self.hardware_info['gpu'].get('is_rtx_3080',False):
				if gpu_util['memory_percent']>90:logger.warning(f"High GPU memory usage: {gpu_util['memory_percent']:.1f}% - "+f"allocated: {gpu_util['memory_allocated']/1024**3:.1f} GB")
				if gpu_util.get('fragmentation',0)>15:logger.warning(f"High GPU memory fragmentation: {gpu_util['fragmentation']:.1f}% - "+f"consider clearing cache")
			elif gpu_util['memory_percent']>90:logger.warning(f"High GPU memory usage: {gpu_util['memory_percent']:.1f}% - "+f"allocated: {gpu_util['memory_allocated']/1024**3:.1f} GB")
	def _check_resource_trends(self)->None:
		if len(self._memory_history)<10:return
		if all(self._memory_history[i]<=self._memory_history[i+1]for i in range(len(self._memory_history)-10,len(self._memory_history)-1)):
			if self._memory_history[-1]-self._memory_history[-10]>10:logger.warning('Memory usage steadily increasing - possible memory leak')
		if len(self._gpu_memory_history)>=10:
			if all(self._gpu_memory_history[i]<=self._gpu_memory_history[i+1]for i in range(len(self._gpu_memory_history)-10,len(self._gpu_memory_history)-1)):
				if self._gpu_memory_history[-1]-self._gpu_memory_history[-10]>15:logger.warning('GPU memory usage steadily increasing - possible CUDA memory leak')