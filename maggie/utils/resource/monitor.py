import threading,time
from typing import Dict,Any,Optional,List,Tuple,Callable
import psutil
from maggie.utils.error_handling import safe_execute,ErrorCategory,with_error_handling,record_error
from maggie.utils.logging import ComponentLogger,log_operation
class ResourceMonitor:
	def __init__(self,config:Dict[str,Any],hardware_info:Dict[str,Any],memory_threshold:float=85.,gpu_threshold:float=95.,event_callback:Optional[Callable]=None):
		self.state_manager=None
		try:
			from maggie.service.locator import ServiceLocator;self.state_manager=ServiceLocator.get('state_manager')
			if self.state_manager:from maggie.core.state import StateAwareComponent;StateAwareComponent.__init__(self,self.state_manager)
		except Exception:pass
		self.event_bus=None
		try:
			from maggie.service.locator import ServiceLocator;self.event_bus=ServiceLocator.get('event_bus')
			if self.event_bus:from maggie.core.event import EventListener,EventPriority;EventListener.__init__(self,self.event_bus);self._event_priority=EventPriority
		except Exception:pass
		self.config=config;self.hardware_info=hardware_info;self.base_memory_threshold=memory_threshold;self.base_gpu_threshold=gpu_threshold;self.event_callback=event_callback;self._monitoring_enabled=False;self._monitoring_thread=None;self._lock=threading.RLock();self._resource_history_max_samples=60;self._cpu_history=[];self._memory_history=[];self._gpu_memory_history=[];self._setup_hardware_thresholds();self._state_thresholds=self._init_state_thresholds();self._state_sampling_intervals=self._init_state_sampling_intervals();self.logger=ComponentLogger('ResourceMonitor');self._register_state_handlers()
	def _register_state_handlers(self)->None:
		if self.state_manager:
			from maggie.core.state import State
			for state in State:self.state_manager.register_state_handler(state,self._on_state_change,True)
			self.logger.info('Resource monitor registered for state change events')
		if self.event_bus:self.listen('state_changed',self._on_state_event,priority=self._event_priority.NORMAL if hasattr(self,'_event_priority')else None);self.logger.info('Resource monitor registered for state change events via event bus')
	def _on_state_change(self,transition)->None:
		if hasattr(transition,'to_state'):new_state=transition.to_state;self.logger.info(f"Adjusting resource monitoring for state: {new_state.name}");self._adjust_monitoring_for_state(new_state)
	def _on_state_event(self,event_data:Any)->None:
		if hasattr(event_data,'to_state'):new_state=event_data.to_state;self._adjust_monitoring_for_state(new_state)
	def _adjust_monitoring_for_state(self,state)->None:
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
	def _init_state_thresholds(self)->Dict[Any,Dict[str,float]]:from maggie.core.state import State;default_thresholds={'memory':self.memory_threshold,'gpu':self.gpu_threshold,'cpu':self.cpu_high_threshold,'cpu_core':self.cpu_per_core_threshold};state_thresholds={State.INIT:{'memory':default_thresholds['memory']*.8,'gpu':default_thresholds['gpu']*.8,'cpu':default_thresholds['cpu']*.8,'cpu_core':default_thresholds['cpu_core']},State.STARTUP:{'memory':default_thresholds['memory']*.9,'gpu':default_thresholds['gpu']*.9,'cpu':default_thresholds['cpu']*1.1,'cpu_core':default_thresholds['cpu_core']},State.IDLE:{'memory':default_thresholds['memory']*.7,'gpu':default_thresholds['gpu']*.7,'cpu':5e1,'cpu_core':default_thresholds['cpu_core']*.8},State.LOADING:{'memory':default_thresholds['memory']*1.1,'gpu':default_thresholds['gpu']*1.1,'cpu':default_thresholds['cpu']*1.2,'cpu_core':default_thresholds['cpu_core']*1.1},State.READY:{'memory':default_thresholds['memory'],'gpu':default_thresholds['gpu'],'cpu':default_thresholds['cpu'],'cpu_core':default_thresholds['cpu_core']},State.ACTIVE:{'memory':default_thresholds['memory']*1.05,'gpu':default_thresholds['gpu']*1.05,'cpu':default_thresholds['cpu']*1.1,'cpu_core':default_thresholds['cpu_core']},State.BUSY:{'memory':default_thresholds['memory']*1.1,'gpu':default_thresholds['gpu']*1.1,'cpu':default_thresholds['cpu']*1.2,'cpu_core':default_thresholds['cpu_core']*1.1},State.CLEANUP:{'memory':default_thresholds['memory']*.9,'gpu':default_thresholds['gpu']*.8,'cpu':default_thresholds['cpu'],'cpu_core':default_thresholds['cpu_core']},State.SHUTDOWN:{'memory':default_thresholds['memory']*.8,'gpu':default_thresholds['gpu']*.7,'cpu':default_thresholds['cpu']*.9,'cpu_core':default_thresholds['cpu_core']*.9}};return state_thresholds
	def _init_state_sampling_intervals(self)->Dict[Any,float]:from maggie.core.state import State;default_interval=5.;return{State.INIT:default_interval,State.STARTUP:3.,State.IDLE:1e1,State.LOADING:2.,State.READY:default_interval,State.ACTIVE:3.,State.BUSY:2.,State.CLEANUP:default_interval,State.SHUTDOWN:1e1}
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
	def _add_nvidia_specific_metrics(self,gpu_util:Dict[str,Any])->None:pass
	def _check_cpu_thresholds(self,cpu_percent:List[float],cpu_avg:float,event_bus:Any)->None:pass
	def _check_memory_thresholds(self,memory,event_bus:Any)->None:pass
	def _check_gpu_thresholds(self,gpu_util:Optional[Dict[str,float]],event_bus:Any)->None:pass
	def _check_resource_trends(self,event_bus:Any)->None:pass
	def get_current_status(self)->Dict[str,Any]:pass
	def get_resource_trends(self)->Dict[str,Any]:pass
	def update_thresholds_for_state(self,state,thresholds:Dict[str,float])->None:pass
	def update_sampling_interval(self,state,interval:float)->None:pass