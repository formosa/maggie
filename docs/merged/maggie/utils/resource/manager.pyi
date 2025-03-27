import os, time, threading
from typing import Dict, Any, Optional, List, Tuple, Set, Callable, Union
import psutil
from maggie.core.state import State, StateTransition, StateAwareComponent
from maggie.core.event import EventListener, EventPriority
from maggie.utils.resource.detector import HardwareDetector
from maggie.utils.resource.optimizer import HardwareOptimizer
from maggie.utils.resource.monitor import ResourceMonitor
from maggie.service.locator import ServiceLocator

class ResourceManager(StateAwareComponent, EventListener):
    def __init__(self, config: Dict[str, Any]):
        if config is None:
            raise ValueError('Configuration cannot be None when initializing ResourceManager')
            
        self.config_manager = None
        try:
            self.config_manager = ServiceLocator.get('config_manager')
        except Exception:
            pass
            
        self.state_manager = None
        try:
            self.state_manager = ServiceLocator.get('state_manager')
        except Exception:
            pass
            
        if self.state_manager:
            StateAwareComponent.__init__(self, self.state_manager)
            
        self.event_bus = None
        try:
            self.event_bus = ServiceLocator.get('event_bus')
        except Exception:
            pass
            
        if self.event_bus:
            EventListener.__init__(self, self.event_bus)
            
        self.config = config
        self.detector = HardwareDetector()
        self.hardware_info = self.detector.detect_system()
        
        self.cpu_config = config.get('cpu', {})
        self.memory_config = config.get('memory', {})
        self.gpu_config = config.get('gpu', {})
        
        self.memory_max_percent = self.memory_config.get('max_percent', 75)
        self.memory_unload_threshold = self.memory_config.get('model_unload_threshold', 85)
        self.gpu_max_percent = self.gpu_config.get('max_percent', 90)
        self.gpu_unload_threshold = self.gpu_config.get('model_unload_threshold', 95)
        
        self.optimizer = HardwareOptimizer(self.hardware_info, self.config)
        self.monitor = ResourceMonitor(
            self.config, 
            self.hardware_info,
            memory_threshold=self.memory_unload_threshold,
            gpu_threshold=self.gpu_unload_threshold,
            event_callback=self._handle_resource_event
        )
        
        self._optimization_profile = self.optimizer.create_optimization_profile()
        self._resource_event_listeners = set()
        self._state_resource_allocations = {}
        
        self._init_state_resource_policies()
        
        if self.config_manager:
            try:
                self.config_manager.optimize_config_for_hardware()
            except Exception:
                pass
                
    def setup_state_management(self) -> bool:
        if self.state_manager is None:
            try:
                self.state_manager = ServiceLocator.get('state_manager')
                if self.state_manager:
                    StateAwareComponent.__init__(self, self.state_manager)
                    self._register_state_handlers()
                    return True
                else:
                    return False
            except Exception:
                return False
        return True
        
    def setup_event_system(self) -> bool:
        if self.event_bus is None:
            try:
                self.event_bus = ServiceLocator.get('event_bus')
                if self.event_bus:
                    EventListener.__init__(self, self.event_bus)
                    self._register_event_handlers()
                    return True
                else:
                    return False
            except Exception:
                return False
        return True
        
    def _register_state_handlers(self) -> None:
        if not self.state_manager:
            return
            
        self.state_manager.register_state_handler(State.INIT, self.on_enter_init, True)
        self.state_manager.register_state_handler(State.STARTUP, self.on_enter_startup, True)
        self.state_manager.register_state_handler(State.IDLE, self.on_enter_idle, True)
        self.state_manager.register_state_handler(State.LOADING, self.on_enter_loading, True)
        self.state_manager.register_state_handler(State.READY, self.on_enter_ready, True)
        self.state_manager.register_state_handler(State.ACTIVE, self.on_enter_active, True)
        self.state_manager.register_state_handler(State.BUSY, self.on_enter_busy, True)
        self.state_manager.register_state_handler(State.CLEANUP, self.on_enter_cleanup, True)
        self.state_manager.register_state_handler(State.SHUTDOWN, self.on_enter_shutdown, True)
        self.state_manager.register_state_handler(State.ACTIVE, self.on_exit_active, False)
        self.state_manager.register_state_handler(State.BUSY, self.on_exit_busy, False)
        
    def _register_event_handlers(self) -> None:
        if not self.event_bus:
            return
            
        event_handlers = [
            ('low_memory_warning', self._handle_low_memory_warning, EventPriority.HIGH),
            ('gpu_memory_warning', self._handle_gpu_memory_warning, EventPriority.HIGH),
            ('state_changed', self._handle_state_changed, EventPriority.NORMAL),
            ('transition_completed', self._handle_transition_completed, EventPriority.NORMAL)
        ]
        
        for (event_type, handler, priority) in event_handlers:
            self.listen(event_type, handler, priority=priority)
            
    def _init_state_resource_policies(self) -> None:
        self._state_resource_policies = {
            State.INIT: {
                'memory_allocation': 'minimal',
                'gpu_memory_allocation': 'none',
                'cpu_priority': 'normal',
                'preload_models': False,
                'clear_cache': True
            },
            State.STARTUP: {
                'memory_allocation': 'low',
                'gpu_memory_allocation': 'minimal',
                'cpu_priority': 'high',
                'preload_models': False,
                'clear_cache': True
            },
            State.IDLE: {
                'memory_allocation': 'low',
                'gpu_memory_allocation': 'minimal',
                'cpu_priority': 'below_normal',
                'preload_models': False,
                'clear_cache': True
            },
            State.LOADING: {
                'memory_allocation': 'high',
                'gpu_memory_allocation': 'high',
                'cpu_priority': 'high',
                'preload_models': True,
                'clear_cache': False
            },
            State.READY: {
                'memory_allocation': 'medium',
                'gpu_memory_allocation': 'medium',
                'cpu_priority': 'normal',
                'preload_models': True,
                'clear_cache': False
            },
            State.ACTIVE: {
                'memory_allocation': 'high',
                'gpu_memory_allocation': 'high',
                'cpu_priority': 'above_normal',
                'preload_models': True,
                'clear_cache': False
            },
            State.BUSY: {
                'memory_allocation': 'maximum',
                'gpu_memory_allocation': 'maximum',
                'cpu_priority': 'high',
                'preload_models': True,
                'clear_cache': False
            },
            State.CLEANUP: {
                'memory_allocation': 'low',
                'gpu_memory_allocation': 'minimal',
                'cpu_priority': 'normal',
                'preload_models': False,
                'clear_cache': True
            },
            State.SHUTDOWN: {
                'memory_allocation': 'minimal',
                'gpu_memory_allocation': 'none',
                'cpu_priority': 'normal',
                'preload_models': False,
                'clear_cache': True
            }
        }
        
    def on_enter_init(self, transition: StateTransition) -> None:
        self._apply_state_resource_policy(State.INIT)
        self.reduce_memory_usage()
        
        if self.config_manager:
            self.config_manager.apply_state_specific_config(State.INIT)
            
    def on_enter_startup(self, transition: StateTransition) -> None:
        self._apply_state_resource_policy(State.STARTUP)
        
        if self.config_manager:
            self.config_manager.apply_state_specific_config(State.STARTUP)
            
    def on_enter_idle(self, transition: StateTransition) -> None:
        self._apply_state_resource_policy(State.IDLE)
        self.reduce_memory_usage()
        
        if self.config_manager:
            self.config_manager.apply_state_specific_config(State.IDLE)
            
    def on_enter_loading(self, transition: StateTransition) -> None:
        self._apply_state_resource_policy(State.LOADING)
        self.clear_gpu_memory()
        
        if self.config_manager:
            self.config_manager.apply_state_specific_config(State.LOADING)
            
    def on_enter_ready(self, transition: StateTransition) -> None:
        self._apply_state_resource_policy(State.READY)
        
        if self.config_manager:
            self.config_manager.apply_state_specific_config(State.READY)
            
    def on_enter_active(self, transition: StateTransition) -> None:
        self._apply_state_resource_policy(State.ACTIVE)
        
        if self.config_manager:
            self.config_manager.apply_state_specific_config(State.ACTIVE)
            
    def on_enter_busy(self, transition: StateTransition) -> None:
        self._apply_state_resource_policy(State.BUSY)
        
        if self.config_manager:
            self.config_manager.apply_state_specific_config(State.BUSY)
            
    def on_enter_cleanup(self, transition: StateTransition) -> None:
        self._apply_state_resource_policy(State.CLEANUP)
        self.reduce_memory_usage()
        
        if self.config_manager:
            self.config_manager.apply_state_specific_config(State.CLEANUP)
            
    def on_enter_shutdown(self, transition: StateTransition) -> None:
        self._apply_state_resource_policy(State.SHUTDOWN)
        self.release_resources()
        self.stop_monitoring()
        
        if self.config_manager:
            self.config_manager.apply_state_specific_config(State.SHUTDOWN)
            
    def on_exit_active(self, transition: StateTransition) -> None:
        if transition.to_state == State.BUSY:
            self._prepare_for_busy_state()
            
    def on_exit_busy(self, transition: StateTransition) -> None:
        if transition.to_state == State.READY:
            self.reduce_memory_usage()
            
    def _handle_state_changed(self, transition: Any) -> None:
        if not hasattr(transition, 'to_state') or not hasattr(transition, 'from_state'):
            return
            
        self._apply_state_resource_policy(transition.to_state)
        
    def _handle_transition_completed(self, transition: Any) -> None:
        if not hasattr(transition, 'to_state') or not hasattr(transition, 'from_state'):
            return
            
        from_state = transition.from_state
        to_state = transition.to_state
        
        if from_state == State.ACTIVE and to_state == State.BUSY:
            self.optimizer.optimize_for_busy_state()
        elif from_state == State.BUSY and to_state == State.READY:
            self.reduce_memory_usage()
        elif from_state == State.READY and to_state == State.LOADING:
            self.clear_gpu_memory()
        elif from_state == State.LOADING and to_state == State.ACTIVE:
            self.optimizer.optimize_for_current_state()
            
    def _prepare_for_busy_state(self) -> None:
        self.clear_gpu_memory()
        
        if self.hardware_info['gpu'].get('is_rtx_3080', False):
            self.optimizer.optimize_for_rtx_3080()
            
    def _apply_state_resource_policy(self, state: State) -> None:
        policy = self._state_resource_policies.get(state, {})
        self._apply_cpu_priority(policy.get('cpu_priority', 'normal'))
        
        memory_allocation = policy.get('memory_allocation', 'normal')
        if memory_allocation in ['minimal', 'low']:
            self.reduce_memory_usage()
            
        gpu_allocation = policy.get('gpu_memory_allocation', 'normal')
        if gpu_allocation in ['none', 'minimal', 'low']:
            self.clear_gpu_memory()
            
        if policy.get('clear_cache', False):
            self.clear_caches()
            
        self._state_resource_allocations[state] = {
            'applied_at': time.time(),
            'policy': policy
        }
        
    def _apply_cpu_priority(self, priority_level: str) -> None:
        try:
            if os.name == 'nt':
                import psutil
                priority_map = {
                    'low': psutil.IDLE_PRIORITY_CLASS,
                    'below_normal': psutil.BELOW_NORMAL_PRIORITY_CLASS,
                    'normal': psutil.NORMAL_PRIORITY_CLASS,
                    'above_normal': psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                    'high': psutil.HIGH_PRIORITY_CLASS,
                    'realtime': psutil.REALTIME_PRIORITY_CLASS
                }
                priority_class = priority_map.get(priority_level, psutil.NORMAL_PRIORITY_CLASS)
                psutil.Process().nice(priority_class)
            elif os.name == 'posix':
                nice_map = {
                    'low': 19,
                    'below_normal': 10,
                    'normal': 0,
                    'above_normal': -5,
                    'high': -10,
                    'realtime': -20
                }
                nice_level = nice_map.get(priority_level, 0)
                
                if nice_level < 0 and os.geteuid() != 0:
                    nice_level = 0
                    
                os.nice(nice_level)
        except Exception:
            pass
            
    def _handle_resource_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        for listener in self._resource_event_listeners:
            try:
                listener(event_type, event_data)
            except Exception:
                pass
                
        if self.event_bus:
            self.event_bus.publish(event_type, event_data)
            
        if event_type == 'low_memory_warning':
            self._handle_low_memory_warning(event_data)
        elif event_type == 'gpu_memory_warning':
            self._handle_gpu_memory_warning(event_data)
            
    def _handle_low_memory_warning(self, event_data: Dict[str, Any]) -> None:
        current_state = None
        if self.state_manager:
            current_state = self.state_manager.get_current_state()
            
        if current_state in [State.IDLE, State.READY, State.CLEANUP]:
            self.reduce_memory_usage()
        elif current_state in [State.LOADING, State.ACTIVE, State.BUSY]:
            if self.event_bus:
                self.event_bus.publish('system_notification', {
                    'type': 'warning',
                    'message': 'System is low on memory. Performance may be affected.',
                    'source': 'ResourceManager'
                })
                
    def _handle_gpu_memory_warning(self, event_data: Dict[str, Any]) -> None:
        current_state = None
        if self.state_manager:
            current_state = self.state_manager.get_current_state()
            
        if current_state in [State.IDLE, State.READY, State.CLEANUP]:
            self.clear_gpu_memory()
        elif current_state in [State.LOADING, State.ACTIVE, State.BUSY]:
            if self.event_bus:
                self.event_bus.publish('system_notification', {
                    'type': 'warning',
                    'message': 'GPU memory is running low. Performance may be affected.',
                    'source': 'ResourceManager'
                })
                
    def setup_gpu(self) -> None:
        self.optimizer.setup_gpu()
        
        if self.hardware_info['gpu'].get('is_rtx_3080', False):
            gpu_opts = self.optimizer.optimize_for_rtx_3080()
                
    def start_monitoring(self, interval: float = 5.) -> bool:
        return self.monitor.start(interval)
        
    def stop_monitoring(self) -> bool:
        return self.monitor.stop()
        
    def clear_gpu_memory(self) -> bool:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                if self.hardware_info['gpu'].get('is_rtx_3080', False):
                    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                        torch.cuda.reset_peak_memory_stats()
                    if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                        torch.cuda.reset_accumulated_memory_stats()
                        
                return True
            return False
        except ImportError:
            return False
            
    def clear_caches(self) -> bool:
        success = True
        
        try:
            import gc
            gc.collect()
        except Exception:
            success = False
            
        if not self.clear_gpu_memory():
            success = False
            
        if os.name == 'nt':
            try:
                import ctypes
                ctypes.windll.kernel32.SetProcessWorkingSetSize(
                    ctypes.windll.kernel32.GetCurrentProcess(), -1, -1
                )
            except Exception:
                success = False
        elif os.name == 'posix' and os.geteuid() == 0:
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1\n')
            except Exception:
                success = False
                
        return success
        
    def reduce_memory_usage(self) -> bool:
        success = True
        
        try:
            import gc
            gc.collect()
            
            if self.hardware_info['memory'].get('is_32gb', False):
                for _ in range(3):
                    gc.collect()
        except Exception:
            success = False
            
        if not self.clear_gpu_memory():
            success = False
            
        if os.name == 'nt':
            try:
                import ctypes
                ctypes.windll.kernel32.SetProcessWorkingSetSize(
                    ctypes.windll.kernel32.GetCurrentProcess(), -1, -1
                )
            except Exception:
                success = False
        elif os.name == 'posix' and os.geteuid() == 0:
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1\n')
            except Exception:
                success = False
                
        return success
        
    def release_resources(self) -> bool:
        success = True
        
        if not self.reduce_memory_usage():
            success = False
            
        try:
            if os.name == 'nt':
                import psutil
                psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS)
        except Exception:
            success = False
            
        return success
        
    def get_resource_status(self) -> Dict[str, Any]:
        return self.monitor.get_current_status()
        
    def get_optimization_profile(self) -> Dict[str, Any]:
        return self._optimization_profile
        
    def apply_hardware_specific_optimizations(self) -> Dict[str, Any]:
        optimizations = {'cpu': {}, 'gpu': {}, 'memory': {}}
        
        if self.hardware_info['cpu'].get('is_ryzen_9_5900x', False):
            cpu_opts = self.optimizer.optimize_for_ryzen_9_5900x()
            if cpu_opts['applied']:
                optimizations['cpu'] = cpu_opts['settings']
                
                if self.config_manager and hasattr(self.config_manager, 'config'):
                    self.config_manager.config.setdefault('cpu', {}).update(cpu_opts['settings'])
                    
        if self.hardware_info['gpu'].get('is_rtx_3080', False):
            gpu_opts = self.optimizer.optimize_for_rtx_3080()
            if gpu_opts['applied']:
                optimizations['gpu'] = gpu_opts['settings']
                
                if self.config_manager and hasattr(self.config_manager, 'config'):
                    self.config_manager.config.setdefault('gpu', {}).update(gpu_opts['settings'])
                    
        if self.hardware_info['memory'].get('is_xpg_d10', False) and self.hardware_info['memory'].get('is_32gb', False):
            memory_settings = {
                'large_pages_enabled': True,
                'numa_aware': True,
                'cache_size_mb': 6144,
                'min_free_gb': 4
            }
            optimizations['memory'] = memory_settings
            
            if self.config_manager and hasattr(self.config_manager, 'config'):
                self.config_manager.config.setdefault('memory', {}).update(memory_settings)
                
        if self.config_manager and hasattr(self.config_manager, 'save') and (
            optimizations['cpu'] or optimizations['gpu'] or optimizations['memory']
        ):
            try:
                self.config_manager.save()
            except Exception:
                pass
                
        return optimizations
        
    def get_hardware_report(self) -> Dict[str, Any]:
        return self.detector.get_detailed_hardware_report()
        
    def recommend_configuration(self) -> Dict[str, Any]:
        recommendations = self._optimization_profile.copy()
        hardware_report = self.detector.get_detailed_hardware_report()
        hardware_specific = {'cpu': {}, 'memory': {}, 'gpu': {}, 'llm': {}, 'stt': {}, 'tts': {}}
        
        cpu_info = self.hardware_info['cpu']
        if cpu_info.get('is_ryzen_9_5900x', False):
            hardware_specific['cpu'] = {
                'max_threads': 8,
                'thread_timeout': 30,
                'priority_boost': True
            }
            hardware_specific['stt'] = {
                'chunk_size': 512,
                'buffer_size': 4096,
                'vad_threshold': .5
            }
            
        memory_info = self.hardware_info['memory']
        if memory_info.get('is_32gb', False):
            hardware_specific['memory'] = {
                'max_percent': 75,
                'model_unload_threshold': 85
            }
            hardware_specific['tts'] = {
                'cache_size': 200,
                'max_workers': 4
            }
            hardware_specific['stt']['whisper_streaming'] = {
                'buffer_size_seconds': 3e1
            }
            
        gpu_info = self.hardware_info['gpu']
        if gpu_info.get('is_rtx_3080', False):
            hardware_specific['gpu'] = {
                'max_percent': 90,
                'model_unload_threshold': 95
            }
            hardware_specific['llm'] = {
                'gpu_layers': 32,
                'gpu_layer_auto_adjust': True
            }
            hardware_specific['stt']['whisper'] = {
                'compute_type': 'float16'
            }
            hardware_specific['stt']['whisper_streaming'].update({
                'compute_type': 'float16'
            })
            hardware_specific['tts'].update({
                'gpu_acceleration': True,
                'gpu_precision': 'mixed_float16'
            })
            
        return {
            'recommended_config': hardware_specific,
            'optimization_profile': recommendations,
            'recommendations': hardware_report.get('recommendations', [])
        }
        
    def register_resource_event_listener(self, listener: Callable[[str, Dict[str, Any]], None]) -> None:
        self._resource_event_listeners.add(listener)
        
    def unregister_resource_event_listener(self, listener: Callable[[str, Dict[str, Any]], None]) -> None:
        if listener in self._resource_event_listeners:
            self._resource_event_listeners.remove(listener)
            
    def preallocate_for_state(self, state: State) -> bool:
        if state not in [State.LOADING, State.ACTIVE, State.BUSY]:
            return True
            
        success = True
        requirements = self.detector.get_resource_requirements_for_state(state)
        
        if state in [State.LOADING, State.ACTIVE, State.BUSY]:
            if requirements.get('gpu_memory_mb', 0) > 0:
                if not self.clear_gpu_memory():
                    success = False
                    
                status = self.get_resource_status()
                if 'gpu' in status and status['gpu'].get('available', False):
                    available_memory_mb = status['gpu'].get('memory_free_gb', 0) * 1024
                    if available_memory_mb < requirements.get('gpu_memory_mb', 0):
                        success = False
                        
        self._apply_cpu_priority(requirements.get('priority', 'normal'))
        
        return success