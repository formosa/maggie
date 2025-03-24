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
			if os.name=='nt':import psutil;psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS);logger.debug('Process priority reset to normal')
		except Exception as e:logger.error(f"Error releasing system resources: {e}");success=False
		return success
	def get_resource_status(self)->Dict[str,Any]:return self.monitor.get_current_status()
	def get_optimization_profile(self)->Dict[str,Any]:return self._optimization_profile