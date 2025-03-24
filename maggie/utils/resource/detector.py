import os,platform
from typing import Dict,Any,Optional
import psutil
from loguru import logger
class HardwareDetector:
	def detect_system(self)->Dict[str,Any]:
		system_info={'os':{'system':platform.system(),'release':platform.release(),'version':platform.version()},'cpu':self.detect_cpu(),'memory':self.detect_memory(),'gpu':self.detect_gpu(),'disk':self.detect_disk()};logger.info(f"Detected OS: {system_info['os']['system']} {system_info['os']['release']}")
		if system_info['cpu']['is_ryzen_9_5900x']:logger.info('Detected AMD Ryzen 9 5900X CPU - applying optimized settings')
		else:logger.info(f"Detected CPU: {system_info['cpu']['model']} with {system_info['cpu']['physical_cores']} cores")
		if system_info['gpu']['is_rtx_3080']:logger.info('Detected NVIDIA RTX 3080 GPU - applying optimized settings')
		elif system_info['gpu']['available']:logger.info(f"Detected GPU: {system_info['gpu']['name']} with {system_info['gpu']['memory_gb']:.2f}GB VRAM")
		else:logger.warning('No compatible GPU detected - some features may be limited')
		return system_info
	def detect_cpu(self)->Dict[str,Any]:
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
	def detect_memory(self)->Dict[str,Any]:
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
	def detect_gpu(self)->Dict[str,Any]:
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
	def detect_disk(self)->Dict[str,Any]:
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