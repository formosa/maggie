import threading,time
from typing import Dict,Any,Optional,List,Tuple,Callable
import psutil
from loguru import logger
class ResourceMonitor:
	def __init__(self,config:Dict[str,Any],hardware_info:Dict[str,Any],memory_threshold:float=85.,gpu_threshold:float=95.,event_callback:Optional[Callable]=None):self.config=config;self.hardware_info=hardware_info;self.memory_threshold=memory_threshold;self.gpu_threshold=gpu_threshold;self.event_callback=event_callback;self._monitoring_enabled=False;self._monitoring_thread=None;self._lock=threading.RLock();self._resource_history_max_samples=60;self._cpu_history=[];self._memory_history=[];self._gpu_memory_history=[]
	def start(self,interval:float=5.)->bool:
		with self._lock:
			if self._monitoring_enabled:return True
			self._monitoring_enabled=True;self._monitoring_thread=threading.Thread(target=self._monitor_resources,args=(interval,),daemon=True,name='ResourceMonitorThread');self._monitoring_thread.start();logger.info(f"Resource monitoring started with {interval}s interval");return True
	def stop(self)->bool:
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
		if memory.percent>self.memory_threshold:
			logger.warning(f"High memory usage: {memory.percent:.1f}% "+f"(threshold: {self.memory_threshold}%) - "+f"available: {memory.available/1024**3:.1f} GB")
			if event_bus:event_bus.publish('low_memory_warning',{'percent':memory.percent,'available_gb':memory.available/1024**3})
		if self.hardware_info['cpu'].get('is_ryzen_9_5900x',False):
			per_core=psutil.cpu_percent(percpu=True);max_core=max(per_core)if per_core else 0;cores_above_95=sum(1 for core in per_core if core>95)
			if cores_above_95>=4:logger.warning(f"High load on {cores_above_95} CPU cores (>95% usage)")
			if cpu_percent>85:logger.warning(f"High overall CPU usage: {cpu_percent:.1f}%")
		elif cpu_percent>90:logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
		if gpu_util:
			if self.hardware_info['gpu'].get('is_rtx_3080',False):
				if gpu_util['memory_percent']>self.gpu_threshold:
					logger.warning(f"High GPU memory usage: {gpu_util['memory_percent']:.1f}% - "+f"allocated: {gpu_util['memory_allocated']/1024**3:.1f} GB")
					if event_bus:event_bus.publish('gpu_memory_warning',{'percent':gpu_util['memory_percent'],'allocated_gb':gpu_util['memory_allocated']/1024**3})
				if gpu_util.get('fragmentation',0)>15:logger.warning(f"High GPU memory fragmentation: {gpu_util['fragmentation']:.1f}% - "+f"consider clearing cache")
			elif gpu_util['memory_percent']>self.gpu_threshold:
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
	def get_current_status(self)->Dict[str,Any]:
		status={}
		try:cpu_percent=psutil.cpu_percent(percpu=True);status['cpu']={'total_percent':sum(cpu_percent)/len(cpu_percent)if cpu_percent else 0,'per_core':cpu_percent,'cores':len(cpu_percent)}
		except Exception as e:logger.error(f"Error getting CPU status: {e}");status['cpu']={'error':str(e)}
		try:memory=psutil.virtual_memory();status['memory']={'total_gb':memory.total/1024**3,'available_gb':memory.available/1024**3,'used_percent':memory.percent}
		except Exception as e:logger.error(f"Error getting memory status: {e}");status['memory']={'error':str(e)}
		gpu_util=self._get_gpu_utilization()
		if gpu_util:status['gpu']=gpu_util
		else:status['gpu']={'available':False}
		return status