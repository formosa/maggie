import os,time,threading
from typing import Dict,Any,Optional,List,Tuple,Set
import psutil
from loguru import logger
from maggie.utils.resource.detector import HardwareDetector
from maggie.utils.resource.optimizer import HardwareOptimizer
from maggie.utils.resource.monitor import ResourceMonitor
class ResourceManager:
	def __init__(self,config:Dict[str,Any]):self.config=config;self.detector=HardwareDetector();self.hardware_info=self.detector.detect_system();self.cpu_config=config.get('cpu',{});self.memory_config=config.get('memory',{});self.gpu_config=config.get('gpu',{});self.memory_max_percent=self.memory_config.get('max_percent',75);self.memory_unload_threshold=self.memory_config.get('model_unload_threshold',85);self.gpu_max_percent=self.gpu_config.get('max_percent',90);self.gpu_unload_threshold=self.gpu_config.get('model_unload_threshold',95);self.optimizer=HardwareOptimizer(self.hardware_info,self.config);self.monitor=ResourceMonitor(self.config,self.hardware_info,memory_threshold=self.memory_unload_threshold,gpu_threshold=self.gpu_unload_threshold);self._optimization_profile=self.optimizer.create_optimization_profile();logger.info('Resource Manager initialized')
	def setup_gpu(self)->None:self.optimizer.setup_gpu()
	def start_monitoring(self,interval:float=5.)->bool:return self.monitor.start(interval)
	def stop_monitoring(self)->bool:return self.monitor.stop()
	def clear_gpu_memory(self)->bool:
		try:
			import torch
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				if self.hardware_info['gpu'].get('is_rtx_3080',False):
					if hasattr(torch.cuda,'reset_peak_memory_stats'):torch.cuda.reset_peak_memory_stats()
					if hasattr(torch.cuda,'reset_accumulated_memory_stats'):torch.cuda.reset_accumulated_memory_stats()
				logger.debug('GPU memory cache cleared');return True
			return False
		except ImportError:logger.debug('PyTorch not available for clearing GPU cache');return False
		except Exception as e:logger.error(f"Error clearing GPU cache: {e}");return False
	def reduce_memory_usage(self)->bool:
		success=True
		try:
			import gc;gc.collect();logger.debug('Python garbage collection performed')
			if self.hardware_info['memory'].get('is_32gb',False):
				for _ in range(3):gc.collect()
			if not self.clear_gpu_memory():success=False
			if os.name=='nt':
				try:import ctypes;ctypes.windll.kernel32.SetProcessWorkingSetSize(ctypes.windll.kernel32.GetCurrentProcess(),-1,-1);logger.debug('Windows working set reduced')
				except Exception as e:logger.debug(f"Error reducing Windows working set: {e}")
			elif os.name=='posix'and os.geteuid()==0:
				try:
					with open('/proc/sys/vm/drop_caches','w')as f:f.write('1\n')
					logger.debug('Linux page cache cleared')
				except Exception as e:logger.debug(f"Error clearing Linux caches: {e}")
		except Exception as e:logger.error(f"Error reducing memory usage: {e}");success=False
		return success
	def release_resources(self)->bool:
		success=True
		if not self.reduce_memory_usage():success=False
		try:
			if os.name=='nt':import psutil;psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS);logger.debug('Process priority reset to normal')
		except Exception as e:logger.error(f"Error releasing system resources: {e}");success=False
		return success
	def get_resource_status(self)->Dict[str,Any]:return self.monitor.get_current_status()
	def get_optimization_profile(self)->Dict[str,Any]:return self._optimization_profile
	def apply_hardware_specific_optimizations(self)->Dict[str,Any]:
		optimizations={'cpu':{},'gpu':{},'memory':{}}
		if self.hardware_info['cpu'].get('is_ryzen_9_5900x',False):
			cpu_opts=self.optimizer.optimize_for_ryzen_9_5900x()
			if cpu_opts['applied']:optimizations['cpu']=cpu_opts['settings'];logger.info('Applied Ryzen 9 5900X-specific optimizations')
		if self.hardware_info['gpu'].get('is_rtx_3080',False):
			gpu_opts=self.optimizer.optimize_for_rtx_3080()
			if gpu_opts['applied']:optimizations['gpu']=gpu_opts['settings'];logger.info('Applied RTX 3080-specific optimizations')
		return optimizations
	def get_hardware_report(self)->Dict[str,Any]:return self.detector.get_detailed_hardware_report()
	def test_gpu_compatibility(self)->Tuple[bool,List[str]]:result=self.optimizer.test_gpu_compatibility();return result['success'],result.get('warnings',[])
	def recommend_configuration(self)->Dict[str,Any]:
		recommendations=self._optimization_profile.copy();hardware_report=self.detector.get_detailed_hardware_report();hardware_specific={'cpu':{},'memory':{},'gpu':{},'llm':{},'stt':{},'tts':{}};cpu_info=self.hardware_info['cpu']
		if cpu_info.get('is_ryzen_9_5900x',False):hardware_specific['cpu']={'max_threads':8,'thread_timeout':30,'priority_boost':True};hardware_specific['stt']={'chunk_size':512,'buffer_size':4096,'vad_threshold':.5}
		memory_info=self.hardware_info['memory']
		if memory_info.get('is_32gb',False):hardware_specific['memory']={'max_percent':75,'model_unload_threshold':85};hardware_specific['tts']={'cache_size':200,'max_workers':4};hardware_specific['stt']['whisper_streaming']={'buffer_size_seconds':3e1}
		gpu_info=self.hardware_info['gpu']
		if gpu_info.get('is_rtx_3080',False):hardware_specific['gpu']={'max_percent':90,'model_unload_threshold':95};hardware_specific['llm']={'gpu_layers':32,'gpu_layer_auto_adjust':True};hardware_specific['stt']['whisper']={'compute_type':'float16'};hardware_specific['stt']['whisper_streaming']={'compute_type':'float16'};hardware_specific['tts'].update({'gpu_acceleration':True,'gpu_precision':'mixed_float16'})
		return{'recommended_config':hardware_specific,'optimization_profile':recommendations,'recommendations':hardware_report.get('recommendations',[])}