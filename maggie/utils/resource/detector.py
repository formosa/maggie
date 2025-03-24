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
			cpu_info={'physical_cores':psutil.cpu_count(logical=False)or 0,'logical_cores':psutil.cpu_count(logical=True)or 0,'model':platform.processor()or'Unknown','is_ryzen_9_5900x':False,'frequency_mhz':{'current':0,'min':0,'max':0}};model_lower=cpu_info['model'].lower()
			if'ryzen 9'in model_lower and'5900x'in model_lower:cpu_info['is_ryzen_9_5900x']=True
			try:
				cpu_freq=psutil.cpu_freq()
				if cpu_freq:cpu_info['frequency_mhz']={'current':cpu_freq.current,'min':cpu_freq.min,'max':cpu_freq.max}
			except Exception as e:logger.debug(f"Error getting CPU frequency: {e}")
			if platform.system()=='Windows':
				try:
					import wmi;c=wmi.WMI()
					for processor in c.Win32_Processor():
						if'Ryzen 9 5900X'in processor.Name:cpu_info['is_ryzen_9_5900x']=True;cpu_info['model']=processor.Name;cpu_info['frequency_mhz']['max']=processor.MaxClockSpeed;cpu_info['architecture']='Zen 3';cpu_info['cache_size_mb']=70;cpu_info['tdp_watts']=105;cpu_info['supports_pbo']=True;break
				except ImportError:logger.debug('WMI module not available for detailed CPU detection')
			return cpu_info
		except Exception as e:logger.error(f"Error detecting CPU: {e}");return{'physical_cores':0,'logical_cores':0,'model':'Unknown','is_ryzen_9_5900x':False,'error':str(e)}
	def detect_memory(self)->Dict[str,Any]:
		try:
			memory=psutil.virtual_memory();memory_info={'total_bytes':memory.total,'total_gb':memory.total/1024**3,'available_bytes':memory.available,'available_gb':memory.available/1024**3,'percent_used':memory.percent,'is_32gb':30<=memory.total/1024**3<=34}
			if platform.system()=='Windows':
				try:
					import wmi;c=wmi.WMI();memory_info['modules']=[];total_capacity=0;is_ddr4_3200=False;modules_count=0
					for physical_memory in c.Win32_PhysicalMemory():
						module_info={}
						if hasattr(physical_memory,'Capacity')and physical_memory.Capacity:capacity=int(physical_memory.Capacity)/1024**3;module_info['capacity_gb']=capacity;total_capacity+=capacity;modules_count+=1
						if hasattr(physical_memory,'PartNumber')and physical_memory.PartNumber:
							module_info['part_number']=physical_memory.PartNumber.strip()
							if'DDR4'in physical_memory.PartNumber:
								memory_info['type']='DDR4';module_info['type']='DDR4'
								if'3200'in physical_memory.PartNumber:memory_info['speed']='3200MHz';module_info['speed']='3200MHz';is_ddr4_3200=True
						if hasattr(physical_memory,'Manufacturer')and physical_memory.Manufacturer:module_info['manufacturer']=physical_memory.Manufacturer.strip()
						memory_info['modules'].append(module_info)
					memory_info['is_xpg_d10']=False
					for module in memory_info['modules']:
						if'part_number'in module and'XPG'in module.get('part_number','')and'D10'in module.get('part_number',''):memory_info['is_xpg_d10']=True;break
					memory_info['is_dual_channel']=modules_count>=2;memory_info['is_ddr4_3200']=is_ddr4_3200
				except ImportError:logger.debug('WMI module not available for detailed memory detection')
				except Exception as e:logger.debug(f"Error getting detailed memory information: {e}")
			return memory_info
		except Exception as e:logger.error(f"Error detecting memory: {e}");return{'total_bytes':0,'total_gb':0,'available_bytes':0,'available_gb':0,'percent_used':0,'is_32gb':False,'error':str(e)}
	def detect_gpu(self)->Dict[str,Any]:
		gpu_info={'available':False,'name':None,'memory_gb':None,'cuda_version':None,'is_rtx_3080':False,'driver_version':None,'architectures':[]}
		try:
			import torch
			if torch.cuda.is_available():
				gpu_info['available']=True;gpu_info['name']=torch.cuda.get_device_name(0);gpu_info['memory_gb']=torch.cuda.get_device_properties(0).total_memory/1024**3;gpu_info['cuda_version']=torch.version.cuda
				if'3080'in gpu_info['name']:
					gpu_info['is_rtx_3080']=True;gpu_info['compute_capability']=torch.cuda.get_device_capability(0);gpu_info['tensor_cores']=True;gpu_info['optimal_precision']='float16';gpu_info['architectures']=['Ampere'];gpu_info['sm_count']=68;gpu_info['cuda_cores']=8704;gpu_info['tensor_cores_count']=272;gpu_info['rt_cores_count']=68;gpu_info['memory_type']='GDDR6X';gpu_info['memory_bus_width']=320;gpu_info['max_power_draw_watts']=320
					if platform.system()=='Windows':
						try:
							import pynvml;pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0);info=pynvml.nvmlDeviceGetMemoryInfo(handle);gpu_info['memory_gb']=info.total/1024**3;gpu_info['driver_version']=pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
							try:gpu_clock=pynvml.nvmlDeviceGetClockInfo(handle,pynvml.NVML_CLOCK_GRAPHICS);mem_clock=pynvml.nvmlDeviceGetClockInfo(handle,pynvml.NVML_CLOCK_MEM);max_gpu_clock=pynvml.nvmlDeviceGetMaxClockInfo(handle,pynvml.NVML_CLOCK_GRAPHICS);max_mem_clock=pynvml.nvmlDeviceGetMaxClockInfo(handle,pynvml.NVML_CLOCK_MEM);gpu_info['clock_speeds']={'gpu_current_mhz':gpu_clock,'memory_current_mhz':mem_clock,'gpu_max_mhz':max_gpu_clock,'memory_max_mhz':max_mem_clock}
							except Exception as e:logger.debug(f"Error getting clock speeds: {e}")
							pynvml.nvmlShutdown()
						except ImportError:logger.debug('PYNVML module not available for detailed GPU detection')
						except Exception as e:logger.debug(f"Error getting detailed GPU information: {e}")
				try:test_tensor=torch.ones(100,100,device='cuda');test_result=torch.matmul(test_tensor,test_tensor);del test_tensor,test_result;torch.cuda.empty_cache();gpu_info['cuda_operations_test']='passed'
				except Exception as e:logger.warning(f"CUDA operations test failed: {e}");gpu_info['cuda_operations_test']='failed';gpu_info['cuda_operations_error']=str(e)
				gpu_info['memory_free_gb']=(torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_allocated(0)-torch.cuda.memory_reserved(0))/1024**3
		except ImportError:logger.debug('PyTorch not available for GPU detection')
		except Exception as e:logger.error(f"Error detecting GPU: {e}")
		return gpu_info
	def detect_disk(self)->Dict[str,Any]:
		disk_info={'root_free_gb':0,'models_free_gb':0,'is_ssd':False}
		try:
			root_usage=psutil.disk_usage('/');disk_info['root_free_gb']=root_usage.free/1024**3;disk_info['root_total_gb']=root_usage.total/1024**3;disk_info['root_used_gb']=root_usage.used/1024**3;disk_info['root_percent']=root_usage.percent
			try:
				models_dir=os.path.abspath('models')
				if os.path.exists(models_dir):models_usage=psutil.disk_usage(models_dir);disk_info['models_free_gb']=models_usage.free/1024**3;disk_info['models_total_gb']=models_usage.total/1024**3;disk_info['models_used_gb']=models_usage.used/1024**3;disk_info['models_percent']=models_usage.percent
			except Exception as e:logger.debug(f"Error getting models directory disk usage: {e}")
			if platform.system()=='Windows':
				try:
					import wmi;c=wmi.WMI();disk_info['disks']=[]
					for disk in c.Win32_DiskDrive():
						disk_data={};model=disk.Model.lower()if disk.Model else'';disk_data['model']=disk.Model;disk_data['size_gb']=int(disk.Size)/1024**3 if hasattr(disk,'Size')and disk.Size else 0;disk_data['interface_type']=disk.InterfaceType if hasattr(disk,'InterfaceType')else'Unknown';is_ssd=any(x in model for x in['ssd','nvme','m.2','solid']);disk_data['is_ssd']=is_ssd
						if is_ssd:disk_info['is_ssd']=True;disk_info['ssd_model']=disk.Model
						disk_info['disks'].append(disk_data)
				except ImportError:logger.debug('WMI module not available for SSD detection')
				except Exception as e:logger.debug(f"Error detecting disk information: {e}")
		except Exception as e:logger.error(f"Error detecting disk space: {e}")
		return disk_info
	def get_detailed_hardware_report(self)->Dict[str,Any]:
		hardware_info=self.detect_system();report={'hardware':hardware_info,'optimizations':{},'recommendations':[]};cpu_info=hardware_info['cpu']
		if cpu_info.get('is_ryzen_9_5900x',False):report['optimizations']['cpu']={'use_cores_for_processing':min(8,cpu_info.get('physical_cores',4)),'use_pbo':True,'affinity_strategy':'performance_cores','priority_boost':True}
		else:
			cores=cpu_info.get('physical_cores',0);report['optimizations']['cpu']={'use_cores_for_processing':max(1,min(cores-2,int(cores*.75)))if cores>0 else 2}
			if cores<8:report['recommendations'].append('CPU has fewer than 8 cores, which may impact performance')
		memory_info=hardware_info['memory']
		if memory_info.get('is_32gb',False):
			if memory_info.get('is_ddr4_3200',False):report['optimizations']['memory']={'max_percent':80,'cache_size_mb':6144,'preloading':True}
			else:report['optimizations']['memory']={'max_percent':75,'cache_size_mb':4096,'preloading':True}
		else:
			total_gb=memory_info.get('total_gb',8)
			if total_gb<16:report['recommendations'].append(f"System has only {total_gb:.1f}GB RAM, which may not be sufficient for optimal performance");report['optimizations']['memory']={'max_percent':60,'cache_size_mb':min(2048,int(total_gb*100))}
			else:report['optimizations']['memory']={'max_percent':70,'cache_size_mb':min(4096,int(total_gb*150))}
		gpu_info=hardware_info['gpu']
		if gpu_info.get('available',False):
			if gpu_info.get('is_rtx_3080',False):
				report['optimizations']['gpu']={'compute_type':'float16','tensor_cores':True,'cuda_streams':3,'reserved_memory_mb':256,'max_batch_size':16,'memory_fraction':.95,'cudnn_benchmark':True,'trt_optimization':True,'dynamic_mem_mgmt':True,'amp_optimization_level':'O2'}
				if gpu_info.get('cuda_operations_test')=='failed':report['recommendations'].append(f"CUDA operations test failed: {gpu_info.get('cuda_operations_error','Unknown error')}");report['recommendations'].append('GPU acceleration may not be fully functional')
			else:
				memory_gb=gpu_info.get('memory_gb',0);report['optimizations']['gpu']={'compute_type':'float16'if memory_gb>=6 else'int8','cuda_streams':1,'reserved_memory_mb':min(512,max(128,int(memory_gb*64))),'max_batch_size':min(16,max(4,int(memory_gb/2)))}
				if memory_gb<8:report['recommendations'].append(f"GPU has only {memory_gb:.1f}GB VRAM, which may limit model size and performance")
		else:report['optimizations']['gpu']={'enabled':False};report['recommendations'].append('No CUDA-capable GPU detected, falling back to CPU-only mode');report['recommendations'].append('Performance may be significantly reduced without GPU acceleration')
		return report