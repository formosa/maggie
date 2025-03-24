import logging
from typing import Dict,Any,Optional,Tuple,List,Union,Callable
import os
from maggie.utils.error_handling import safe_execute
logger=logging.getLogger(__name__)
def clear_gpu_cache()->bool:
	def _clear_cache():
		import torch
		if torch.cuda.is_available():torch.cuda.empty_cache();logger.debug('GPU memory cache cleared');return True
		return False
	return safe_execute(_clear_cache,default_return=False,error_message='Error clearing GPU cache')
def get_gpu_info()->Dict[str,Any]:
	info={'available':False,'name':None,'memory_total_gb':0,'memory_allocated_gb':0,'memory_reserved_gb':0,'memory_free_gb':0,'utilization':0,'compute_capability':None,'is_rtx_3080':False}
	def _gather_gpu_info():
		import torch
		if torch.cuda.is_available():
			info['available']=True;info['name']=torch.cuda.get_device_name(0);info['is_rtx_3080']='3080'in info['name'];cc=torch.cuda.get_device_capability(0);info['compute_capability']=f"{cc[0]}.{cc[1]}";props=torch.cuda.get_device_properties(0);info['memory_total_gb']=props.total_memory/1024**3;info['memory_allocated_gb']=torch.cuda.memory_allocated(0)/1024**3;info['memory_reserved_gb']=torch.cuda.memory_reserved(0)/1024**3;info['memory_free_gb']=(props.total_memory-torch.cuda.memory_allocated(0)-torch.cuda.memory_reserved(0))/1024**3;info['cuda_version']=torch.version.cuda
			try:import pynvml;pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0);info['utilization']=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu;pynvml.nvmlShutdown()
			except ImportError:pass
		return info
	return safe_execute(_gather_gpu_info,default_return=info,error_message='Error gathering GPU information')
def optimize_for_rtx_3080()->Dict[str,Any]:
	optimizations={'applied':False,'settings':{}}
	def _apply_optimizations():
		import torch
		if torch.cuda.is_available()and'3080'in torch.cuda.get_device_name(0):
			optimizations['settings']['amp_enabled']=True
			if hasattr(torch.cuda,'memory_stats'):optimizations['settings']['memory_stats_enabled']=True
			torch.backends.cudnn.benchmark=True;optimizations['settings']['cudnn_benchmark']=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;optimizations['settings']['tf32_enabled']=True;optimizations['settings']['max_split_size_mb']=512;optimizations['settings']['cuda_graphs_enabled']=True;optimizations['applied']=True;logger.info('Applied RTX 3080-specific optimizations')
		return optimizations
	return safe_execute(_apply_optimizations,default_return=optimizations,error_message='Error applying GPU optimizations')
def test_gpu_compatibility()->Tuple[bool,List[str]]:
	warnings=[]
	def _test_gpu():
		import torch
		if not torch.cuda.is_available():warnings.append('CUDA not available, running in CPU-only mode');return False,warnings
		try:test_tensor=torch.ones(1000,1000,device='cuda');result=torch.matmul(test_tensor,test_tensor);del test_tensor,result;torch.cuda.empty_cache()
		except Exception as e:warnings.append(f"Basic CUDA operations failed: {e}");return False,warnings
		gpu_info=get_gpu_info()
		if gpu_info['memory_total_gb']<8:warnings.append(f"GPU memory ({gpu_info['memory_total_gb']:.1f}GB) is less than recommended 8GB")
		if'3080'in gpu_info['name']:
			if gpu_info['compute_capability']!='8.6':warnings.append(f"RTX 3080 compute capability ({gpu_info['compute_capability']}) is not 8.6")
			if not gpu_info['cuda_version'].startswith('11.'):warnings.append(f"CUDA version {gpu_info['cuda_version']} - version 11.x recommended for RTX 3080")
		return True,warnings
	result=safe_execute(_test_gpu,default_return=(False,['Error testing GPU compatibility']),error_message='Error testing GPU compatibility');return result