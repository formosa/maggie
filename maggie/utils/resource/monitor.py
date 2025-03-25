import threading,time
from typing import Dict,Any,Optional,List,Tuple,Callable
import psutil
from maggie.utils.error_handling import safe_execute,ErrorCategory,with_error_handling,record_error
from maggie.utils.logging import ComponentLogger,log_operation
from maggie.core.state import State,StateTransition,StateAwareComponent
from maggie.core.event import EventListener,EventPriority
class ResourceMonitor(StateAwareComponent,EventListener):
	def __init__(self,config:Dict[str,Any],hardware_info:Dict[str,Any],memory_threshold:float=85.,gpu_threshold:float=95.,event_callback:Optional[Callable]=None):
		self.state_manager=None
		try:
			from maggie.service.locator import ServiceLocator;self.state_manager=ServiceLocator.get('state_manager')
			if self.state_manager:StateAwareComponent.__init__(self,self.state_manager)
		except Exception:pass
		self.event_bus=None
		try:
			from maggie.service.locator import ServiceLocator;self.event_bus=ServiceLocator.get('event_bus')
			if self.event_bus:EventListener.__init__(self,self.event_bus)
		except Exception:pass
		self.config=config;self.hardware_info=hardware_info;self.base_memory_threshold=memory_threshold;self.base_gpu_threshold=gpu_threshold;self.event_callback=event_callback;self._monitoring_enabled=False;self._monitoring_thread=None;self._lock=threading.RLock();self._resource_history_max_samples=60;self._cpu_history=[];self._memory_history=[];self._gpu_memory_history=[];self._setup_hardware_thresholds();self._state_thresholds=self._init_state_thresholds();self._state_sampling_intervals=self._init_state_sampling_intervals();self.logger=ComponentLogger('ResourceMonitor');self._register_state_handlers()
	def _register_state_handlers(self)->None:
		if self.state_manager:
			for state in State:self.state_manager.register_state_handler(state,self._on_state_change,True)
			self.logger.info('Resource monitor registered for state change events')
		if self.event_bus:self.listen('state_changed',self._on_state_event,priority=EventPriority.NORMAL);self.logger.info('Resource monitor registered for state change events via event bus')
	def _on_state_change(self,transition:StateTransition)->None:
		if hasattr(transition,'to_state'):new_state=transition.to_state;self.logger.info(f"Adjusting resource monitoring for state: {new_state.name}");self._adjust_monitoring_for_state(new_state)
	def _on_state_event(self,event_data:Any)->None:
		if hasattr(event_data,'to_state'):new_state=event_data.to_state;self._adjust_monitoring_for_state(new_state)
	def _adjust_monitoring_for_state(self,state:State)->None:
		if self._monitoring_thread and self._monitoring_enabled:self.logger.debug(f"Adjusted monitoring for state {state.name}")
	def _setup_hardware_thresholds(self)->None:
		cpu_info=self.hardware_info.get('cpu',{});gpu_info=self.hardware_info.get('gpu',{});memory_info=self.hardware_info.get('memory',{})
		if cpu_info.get('is_ryzen_9_5900x',False):self.cpu_high_threshold=85.;self.cpu_per_core_threshold=95.;self.cpu_core_count_high_threshold=4
		else:self.cpu_high_threshold=9e1;self.cpu_per_core_threshold=98.;self.cpu_core_count_high_threshold=min(2,cpu_info.get('physical_cores',1))
		if gpu_info.get('is_rtx_3080',False):self.gpu_threshold=92.;self.gpu_fragmentation_threshold=15.;self.gpu_temperature_threshold=8e1
		elif gpu_info.get('available',False):self.gpu_fragmentation_threshold=2e1;memory_gb=gpu_info.get('memory_gb',8);self.gpu_temperature_threshold=85. if memory_gb>=8 else 75.
		if memory_info.get('is_32gb',False):self.memory_threshold=self.base_memory_threshold;self.memory_growth_threshold=1e1
		else:
			total_gb=memory_info.get('total_gb',8)
			if total_gb<16:self.memory_threshold=min(75.,self.base_memory_threshold);self.memory_growth_threshold=8.
			else:self.memory_growth_threshold=12.
	def _init_state_thresholds(self)->Dict[State,Dict[str,float]]:default_thresholds={'memory':self.memory_threshold,'gpu':self.gpu_threshold,'cpu':self.cpu_high_threshold,'cpu_core':self.cpu_per_core_threshold};state_thresholds={State.INIT:{'memory':default_thresholds['memory']*.8,'gpu':default_thresholds['gpu']*.8,'cpu':default_thresholds['cpu']*.8,'cpu_core':default_thresholds['cpu_core']},State.STARTUP:{'memory':default_thresholds['memory']*.9,'gpu':default_thresholds['gpu']*.9,'cpu':default_thresholds['cpu']*1.1,'cpu_core':default_thresholds['cpu_core']},State.IDLE:{'memory':default_thresholds['memory']*.7,'gpu':default_thresholds['gpu']*.7,'cpu':5e1,'cpu_core':default_thresholds['cpu_core']*.8},State.LOADING:{'memory':default_thresholds['memory']*1.1,'gpu':default_thresholds['gpu']*1.1,'cpu':default_thresholds['cpu']*1.2,'cpu_core':default_thresholds['cpu_core']*1.1},State.READY:{'memory':default_thresholds['memory'],'gpu':default_thresholds['gpu'],'cpu':default_thresholds['cpu'],'cpu_core':default_thresholds['cpu_core']},State.ACTIVE:{'memory':default_thresholds['memory']*1.05,'gpu':default_thresholds['gpu']*1.05,'cpu':default_thresholds['cpu']*1.1,'cpu_core':default_thresholds['cpu_core']},State.BUSY:{'memory':default_thresholds['memory']*1.1,'gpu':default_thresholds['gpu']*1.1,'cpu':default_thresholds['cpu']*1.2,'cpu_core':default_thresholds['cpu_core']*1.1},State.CLEANUP:{'memory':default_thresholds['memory']*.9,'gpu':default_thresholds['gpu']*.8,'cpu':default_thresholds['cpu'],'cpu_core':default_thresholds['cpu_core']},State.SHUTDOWN:{'memory':default_thresholds['memory']*.8,'gpu':default_thresholds['gpu']*.7,'cpu':default_thresholds['cpu']*.9,'cpu_core':default_thresholds['cpu_core']*.9}};return state_thresholds
	def _init_state_sampling_intervals(self)->Dict[State,float]:default_interval=5.;return{State.INIT:default_interval,State.STARTUP:3.,State.IDLE:1e1,State.LOADING:2.,State.READY:default_interval,State.ACTIVE:3.,State.BUSY:2.,State.CLEANUP:default_interval,State.SHUTDOWN:1e1}
	def _get_current_state_thresholds(self)->Dict[str,float]:
		current_state=None
		if self.state_manager:
			try:current_state=self.state_manager.get_current_state()
			except Exception as e:self.logger.warning(f"Error getting current state: {e}")
		if current_state is None:return{'memory':self.memory_threshold,'gpu':self.gpu_threshold,'cpu':self.cpu_high_threshold,'cpu_core':self.cpu_per_core_threshold}
		return self._state_thresholds.get(current_state,self._state_thresholds[State.READY])
	def _get_current_sampling_interval(self)->float:
		current_state=None
		if self.state_manager:
			try:current_state=self.state_manager.get_current_state()
			except Exception as e:self.logger.warning(f"Error getting current state: {e}")
		if current_state is None:return 5.
		return self._state_sampling_intervals.get(current_state,5.)
	@log_operation(component='ResourceMonitor')
	def start(self,interval:float=5.)->bool:
		with self._lock:
			if self._monitoring_enabled:return True
			self._monitoring_enabled=True;self._monitoring_thread=threading.Thread(target=self._monitor_resources,name='ResourceMonitorThread',daemon=True);self._monitoring_thread.start();self.logger.info(f"Resource monitoring started with dynamic intervals");return True
	@log_operation(component='ResourceMonitor')
	def stop(self)->bool:
		with self._lock:
			if not self._monitoring_enabled:return False
			self._monitoring_enabled=False
			if self._monitoring_thread:self._monitoring_thread.join(timeout=2.)
			self.logger.info('Resource monitoring stopped');return True
	def _monitor_resources(self)->None:
		try:from maggie.service.locator import ServiceLocator;event_bus=ServiceLocator.get('event_bus')
		except ImportError:self.logger.warning('ServiceLocator not available - event publishing disabled');event_bus=None
		while self._monitoring_enabled:
			try:
				interval=self._get_current_sampling_interval();cpu_percent=psutil.cpu_percent(percpu=True);cpu_avg=sum(cpu_percent)/len(cpu_percent)if cpu_percent else 0;memory=psutil.virtual_memory()
				with self._lock:
					self._cpu_history.append(cpu_avg);self._memory_history.append(memory.percent)
					if len(self._cpu_history)>self._resource_history_max_samples:self._cpu_history.pop(0)
					if len(self._memory_history)>self._resource_history_max_samples:self._memory_history.pop(0)
				gpu_util=self._get_gpu_utilization()
				if gpu_util and'memory_percent'in gpu_util:
					with self._lock:
						self._gpu_memory_history.append(gpu_util['memory_percent'])
						if len(self._gpu_memory_history)>self._resource_history_max_samples:self._gpu_memory_history.pop(0)
				self._check_cpu_thresholds(cpu_percent,cpu_avg,event_bus);self._check_memory_thresholds(memory,event_bus);self._check_gpu_thresholds(gpu_util,event_bus);self._check_resource_trends(event_bus);time.sleep(interval)
			except Exception as e:self.logger.error(f"Error monitoring resources: {e}");time.sleep(5.)
	@with_error_handling(error_category=ErrorCategory.RESOURCE)
	def _get_gpu_utilization(self)->Optional[Dict[str,float]]:
		if not self.hardware_info['gpu'].get('available',False):return None
		try:
			import torch
			if torch.cuda.is_available():
				allocated=torch.cuda.memory_allocated(0);reserved=torch.cuda.memory_reserved(0);total=torch.cuda.get_device_properties(0).total_memory;memory_percent=allocated/total*100;reserved_percent=reserved/total*100;active_allocations=0;detailed_metrics={}
				if hasattr(torch.cuda,'memory_stats'):stats=torch.cuda.memory_stats(0);active_allocations=stats.get('num_alloc_retries',0);detailed_metrics={'num_alloc_retries':stats.get('num_alloc_retries',0),'num_ooms':stats.get('num_ooms',0),'max_split_size':stats.get('max_split_size',0)/1024**2,'allocated_bytes.all.current':stats.get('allocated_bytes.all.current',0)/1024**3}
				gpu_util={'memory_allocated':allocated,'memory_reserved':reserved,'memory_total':total,'memory_percent':memory_percent,'reserved_percent':reserved_percent,'active_allocations':active_allocations,'fragmentation':(reserved-allocated)/total*100 if reserved>allocated else 0};gpu_util.update(detailed_metrics)
				if self.hardware_info['gpu'].get('is_rtx_3080',False):self._add_nvidia_specific_metrics(gpu_util)
				return gpu_util
		except ImportError:self.logger.debug('PyTorch not available for GPU monitoring')
		except Exception as e:self.logger.error(f"Error getting GPU utilization: {e}")
		return None
	def _add_nvidia_specific_metrics(self,gpu_util:Dict[str,Any])->None:
		try:
			import pynvml;pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0);temp=pynvml.nvmlDeviceGetTemperature(handle,pynvml.NVML_TEMPERATURE_GPU);gpu_util['temperature']=temp
			try:fan=pynvml.nvmlDeviceGetFanSpeed(handle);gpu_util['fan_percent']=fan
			except:pass
			try:power=pynvml.nvmlDeviceGetPowerUsage(handle)/1000;power_limit=pynvml.nvmlDeviceGetPowerManagementLimit(handle)/1000;gpu_util['power_watts']=power;gpu_util['power_limit_watts']=power_limit;gpu_util['power_percent']=power/power_limit*100 if power_limit>0 else 0
			except:pass
			try:util_rates=pynvml.nvmlDeviceGetUtilizationRates(handle);gpu_util['gpu_utilization']=util_rates.gpu;gpu_util['memory_utilization']=util_rates.memory
			except:pass
			pynvml.nvmlShutdown()
		except ImportError:self.logger.debug('PYNVML module not available for detailed GPU monitoring')
		except Exception as e:self.logger.debug(f"Error getting detailed GPU information: {e}")
	def _check_cpu_thresholds(self,cpu_percent:List[float],cpu_avg:float,event_bus:Any)->None:
		current_thresholds=self._get_current_state_thresholds();cpu_threshold=current_thresholds['cpu'];cpu_core_threshold=current_thresholds['cpu_core'];cpu_info=self.hardware_info.get('cpu',{})
		if cpu_info.get('is_ryzen_9_5900x',False):
			per_core=cpu_percent;cores_above_threshold=sum(1 for core in per_core if core>cpu_core_threshold)
			if cores_above_threshold>=self.cpu_core_count_high_threshold:
				message=f"High load on {cores_above_threshold} CPU cores (>{cpu_core_threshold}% usage)";self.logger.warning(message)
				if event_bus:event_data={'cores_above_threshold':cores_above_threshold,'threshold':cpu_core_threshold,'message':message};event_bus.publish('high_cpu_core_usage',event_data)
		if cpu_avg>cpu_threshold:
			message=f"High overall CPU usage: {cpu_avg:.1f}%";self.logger.warning(message)
			if event_bus:event_data={'percent':cpu_avg,'threshold':cpu_threshold,'message':message};event_bus.publish('high_cpu_usage',event_data)
	def _check_memory_thresholds(self,memory:psutil._psplatform.svmem,event_bus:Any)->None:
		current_thresholds=self._get_current_state_thresholds();memory_threshold=current_thresholds['memory']
		if memory.percent>memory_threshold:
			message=f"High memory usage: {memory.percent:.1f}% "+f"(threshold: {memory_threshold}%) - "+f"available: {memory.available/1024**3:.1f} GB";self.logger.warning(message)
			if event_bus:
				event_data={'percent':memory.percent,'available_gb':memory.available/1024**3,'threshold':memory_threshold,'total_gb':memory.total/1024**3,'message':message};event_bus.publish('low_memory_warning',event_data)
				if memory.percent>memory_threshold*1.2:event_data['critical']=True;event_bus.publish('critical_memory_warning',event_data)
	def _check_gpu_thresholds(self,gpu_util:Optional[Dict[str,float]],event_bus:Any)->None:
		if not gpu_util:return
		current_thresholds=self._get_current_state_thresholds();gpu_threshold=current_thresholds['gpu'];gpu_info=self.hardware_info.get('gpu',{})
		if gpu_info.get('is_rtx_3080',False):
			if gpu_util['memory_percent']>gpu_threshold:
				message=f"High GPU memory usage: {gpu_util['memory_percent']:.1f}% - "+f"allocated: {gpu_util['memory_allocated']/1024**3:.1f} GB";self.logger.warning(message)
				if event_bus:
					event_data={'percent':gpu_util['memory_percent'],'allocated_gb':gpu_util['memory_allocated']/1024**3,'threshold':gpu_threshold,'message':message};event_bus.publish('gpu_memory_warning',event_data)
					if gpu_util['memory_percent']>gpu_threshold*1.1:event_data['critical']=True;event_bus.publish('critical_gpu_memory_warning',event_data)
			if gpu_util.get('fragmentation',0)>self.gpu_fragmentation_threshold:
				message=f"High GPU memory fragmentation: {gpu_util['fragmentation']:.1f}% - "+f"consider clearing cache";self.logger.warning(message)
				if event_bus:event_data={'percent':gpu_util['fragmentation'],'threshold':self.gpu_fragmentation_threshold,'message':message};event_bus.publish('gpu_fragmentation_warning',event_data)
			if'temperature'in gpu_util and gpu_util['temperature']>self.gpu_temperature_threshold:
				message=f"High GPU temperature: {gpu_util['temperature']}°C";self.logger.warning(message)
				if event_bus:event_data={'temperature':gpu_util['temperature'],'threshold':self.gpu_temperature_threshold,'message':message};event_bus.publish('gpu_temperature_warning',event_data)
			if'power_percent'in gpu_util and gpu_util['power_percent']>95:
				message=f"High GPU power usage: {gpu_util['power_watts']:.1f}W "+f"({gpu_util['power_percent']:.1f}% of limit)";self.logger.warning(message)
				if event_bus:event_data={'power_watts':gpu_util['power_watts'],'power_percent':gpu_util['power_percent'],'message':message};event_bus.publish('gpu_power_warning',event_data)
		elif gpu_util['memory_percent']>gpu_threshold:
			message=f"High GPU memory usage: {gpu_util['memory_percent']:.1f}% - "+f"allocated: {gpu_util['memory_allocated']/1024**3:.1f} GB";self.logger.warning(message)
			if event_bus:event_data={'percent':gpu_util['memory_percent'],'allocated_gb':gpu_util['memory_allocated']/1024**3,'threshold':gpu_threshold,'message':message};event_bus.publish('gpu_memory_warning',event_data)
	def _check_resource_trends(self,event_bus:Any)->None:
		with self._lock:
			if len(self._memory_history)<10:return
			memory_increasing=all(self._memory_history[i]<=self._memory_history[i+1]for i in range(len(self._memory_history)-10,len(self._memory_history)-1))
			if memory_increasing:
				memory_increase=self._memory_history[-1]-self._memory_history[-10]
				if memory_increase>self.memory_growth_threshold:
					message=f"Memory usage steadily increasing by {memory_increase:.1f}% over recent samples - possible memory leak";self.logger.warning(message)
					if event_bus:event_data={'increase':memory_increase,'current':self._memory_history[-1],'threshold':self.memory_growth_threshold,'message':message};event_bus.publish('memory_leak_warning',event_data)
			if len(self._gpu_memory_history)>=10:
				gpu_increasing=all(self._gpu_memory_history[i]<=self._gpu_memory_history[i+1]for i in range(len(self._gpu_memory_history)-10,len(self._gpu_memory_history)-1))
				if gpu_increasing:
					gpu_increase=self._gpu_memory_history[-1]-self._gpu_memory_history[-10]
					if gpu_increase>15:
						message=f"GPU memory usage steadily increasing by {gpu_increase:.1f}% - possible CUDA memory leak";self.logger.warning(message)
						if event_bus:event_data={'increase':gpu_increase,'current':self._gpu_memory_history[-1],'message':message};event_bus.publish('gpu_memory_leak_warning',event_data)
			if len(self._cpu_history)>=20:
				recent_avg=sum(self._cpu_history[-5:])/5;previous_avg=sum(self._cpu_history[-20:-5])/15
				if recent_avg>previous_avg*2 and recent_avg>80:
					message=f"Sudden CPU usage spike detected: {recent_avg:.1f}% (was {previous_avg:.1f}%)";self.logger.warning(message)
					if event_bus:event_data={'recent_avg':recent_avg,'previous_avg':previous_avg,'message':message};event_bus.publish('cpu_spike_warning',event_data)
	def get_current_status(self)->Dict[str,Any]:
		status={}
		try:cpu_percent=psutil.cpu_percent(percpu=True);status['cpu']={'total_percent':sum(cpu_percent)/len(cpu_percent)if cpu_percent else 0,'per_core':cpu_percent,'cores':len(cpu_percent)}
		except Exception as e:self.logger.error(f"Error getting CPU status: {e}");status['cpu']={'error':str(e)}
		try:memory=psutil.virtual_memory();status['memory']={'total_gb':memory.total/1024**3,'available_gb':memory.available/1024**3,'used_percent':memory.percent}
		except Exception as e:self.logger.error(f"Error getting memory status: {e}");status['memory']={'error':str(e)}
		gpu_util=self._get_gpu_utilization()
		if gpu_util:status['gpu']=gpu_util
		else:status['gpu']={'available':False}
		with self._lock:
			if self._cpu_history:status['cpu']['history']=self._cpu_history[-30:]
			if self._memory_history:status['memory']['history']=self._memory_history[-30:]
			if self._gpu_memory_history:status['gpu']['memory_history']=self._gpu_memory_history[-30:]
		if self.state_manager:
			try:current_state=self.state_manager.get_current_state();status['state']=current_state.name;status['thresholds']=self._get_current_state_thresholds();status['sampling_interval']=self._get_current_sampling_interval()
			except Exception as e:self.logger.error(f"Error getting state info: {e}")
		return status
	def get_resource_trends(self)->Dict[str,Any]:
		trends={'cpu':{'trend':'stable','value':.0},'memory':{'trend':'stable','value':.0},'gpu':{'trend':'stable','value':.0}}
		with self._lock:
			if len(self._cpu_history)>=10:
				cpu_change=self._cpu_history[-1]-self._cpu_history[-10]
				if cpu_change>10:trends['cpu']={'trend':'increasing','value':cpu_change}
				elif cpu_change<-10:trends['cpu']={'trend':'decreasing','value':cpu_change}
				else:trends['cpu']={'trend':'stable','value':cpu_change}
			if len(self._memory_history)>=10:
				memory_change=self._memory_history[-1]-self._memory_history[-10]
				if memory_change>5:trends['memory']={'trend':'increasing','value':memory_change}
				elif memory_change<-5:trends['memory']={'trend':'decreasing','value':memory_change}
				else:trends['memory']={'trend':'stable','value':memory_change}
			if len(self._gpu_memory_history)>=10:
				gpu_change=self._gpu_memory_history[-1]-self._gpu_memory_history[-10]
				if gpu_change>5:trends['gpu']={'trend':'increasing','value':gpu_change}
				elif gpu_change<-5:trends['gpu']={'trend':'decreasing','value':gpu_change}
				else:trends['gpu']={'trend':'stable','value':gpu_change}
		return trends
	def update_thresholds_for_state(self,state:State,thresholds:Dict[str,float])->None:
		if state in self._state_thresholds:self._state_thresholds[state].update(thresholds);self.logger.info(f"Updated thresholds for state {state.name}: {thresholds}")
	def update_sampling_interval(self,state:State,interval:float)->None:
		if state in self._state_sampling_intervals:self._state_sampling_intervals[state]=interval;self.logger.info(f"Updated sampling interval for state {state.name}: {interval} seconds")