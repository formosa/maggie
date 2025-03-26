import os, platform, time
from typing import Dict, Any, Optional, List, Union, Tuple
import psutil

class HardwareDetector:
    def __init__(self):
        self._cached_system_info = None
        
    def detect_system(self) -> Dict[str, Any]:
        if self._cached_system_info is not None:
            return self._cached_system_info
            
        system_info = {
            'os': self._detect_os(),
            'cpu': self.detect_cpu(),
            'memory': self.detect_memory(),
            'gpu': self.detect_gpu(),
            'disk': self.detect_disk()
        }
        
        self._cached_system_info = system_info
        return system_info
        
    def _detect_os(self) -> Dict[str, str]:
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version()
        }
        
    def detect_cpu(self) -> Dict[str, Any]:
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False) or 0,
            'logical_cores': psutil.cpu_count(logical=True) or 0,
            'model': platform.processor() or 'Unknown',
            'is_ryzen_9_5900x': False,
            'frequency_mhz': {'current': 0, 'min': 0, 'max': 0}
        }
        
        model_lower = cpu_info['model'].lower()
        if 'ryzen 9' in model_lower and '5900x' in model_lower:
            cpu_info['is_ryzen_9_5900x'] = True
            
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info['frequency_mhz'] = {
                    'current': cpu_freq.current,
                    'min': cpu_freq.min,
                    'max': cpu_freq.max
                }
        except Exception:
            pass
            
        if platform.system() == 'Windows':
            self._enhance_windows_cpu_detection(cpu_info)
            
        return cpu_info
        
    def _enhance_windows_cpu_detection(self, cpu_info: Dict[str, Any]) -> None:
        try:
            import wmi
            c = wmi.WMI()
            for processor in c.Win32_Processor():
                if 'Ryzen 9 5900X' in processor.Name:
                    cpu_info['is_ryzen_9_5900x'] = True
                    cpu_info['model'] = processor.Name
                    cpu_info['frequency_mhz']['max'] = processor.MaxClockSpeed
                    cpu_info['architecture'] = 'Zen 3'
                    cpu_info['cache_size_mb'] = 70
                    cpu_info['tdp_watts'] = 105
                    cpu_info['supports_pbo'] = True
                    cpu_info['recommended_settings'] = {
                        'max_threads': 10,
                        'affinity_strategy': 'performance_cores',
                        'power_plan': 'high_performance'
                    }
                    break
        except ImportError:
            pass
        except Exception:
            pass
            
    def detect_memory(self) -> Dict[str, Any]:
        try:
            memory = psutil.virtual_memory()
            memory_info = {
                'total_bytes': memory.total,
                'total_gb': memory.total / 1024**3,
                'available_bytes': memory.available,
                'available_gb': memory.available / 1024**3,
                'percent_used': memory.percent,
                'is_32gb': 30 <= memory.total / 1024**3 <= 34,
                'is_xpg_d10': False
            }
            
            if platform.system() == 'Windows':
                self._enhance_windows_memory_detection(memory_info)
                
            return memory_info
        except Exception as e:
            return {
                'total_bytes': 0,
                'total_gb': 0,
                'available_bytes': 0,
                'available_gb': 0,
                'percent_used': 0,
                'is_32gb': False,
                'error': str(e)
            }
            
    def _enhance_windows_memory_detection(self, memory_info: Dict[str, Any]) -> None:
        try:
            import wmi
            c = wmi.WMI()
            memory_info['modules'] = []
            total_capacity = 0
            is_ddr4_3200 = False
            modules_count = 0
            
            for physical_memory in c.Win32_PhysicalMemory():
                module_info = {}
                
                if hasattr(physical_memory, 'Capacity') and physical_memory.Capacity:
                    capacity = int(physical_memory.Capacity) / 1024**3
                    module_info['capacity_gb'] = capacity
                    total_capacity += capacity
                    modules_count += 1
                    
                if hasattr(physical_memory, 'PartNumber') and physical_memory.PartNumber:
                    module_info['part_number'] = physical_memory.PartNumber.strip()
                    
                    if 'DDR4' in physical_memory.PartNumber:
                        memory_info['type'] = 'DDR4'
                        module_info['type'] = 'DDR4'
                        
                        if '3200' in physical_memory.PartNumber:
                            memory_info['speed'] = '3200MHz'
                            module_info['speed'] = '3200MHz'
                            is_ddr4_3200 = True
                            
                    if 'XPG' in physical_memory.PartNumber and 'D10' in physical_memory.PartNumber:
                        memory_info['is_xpg_d10'] = True
                        
                if hasattr(physical_memory, 'Manufacturer') and physical_memory.Manufacturer:
                    module_info['manufacturer'] = physical_memory.Manufacturer.strip()
                    
                    if 'ADATA' in physical_memory.Manufacturer:
                        module_info['is_adata'] = True
                        
                        if hasattr(physical_memory, 'PartNumber') and \
                           ('D10' in physical_memory.PartNumber or 'AX4U' in physical_memory.PartNumber):
                            memory_info['is_xpg_d10'] = True
                            
                memory_info['modules'].append(module_info)
                
            memory_info['is_dual_channel'] = modules_count >= 2
            memory_info['is_ddr4_3200'] = is_ddr4_3200
            
            if memory_info['is_xpg_d10'] and memory_info['is_32gb']:
                memory_info['recommended_settings'] = {
                    'max_percent': 80,
                    'cache_size_mb': 6144,
                    'preloading': True,
                    'large_pages': True,
                    'numa_aware': True
                }
        except ImportError:
            pass
        except Exception:
            pass
            
    def detect_gpu(self) -> Dict[str, Any]:
        gpu_info = {
            'available': False,
            'name': None,
            'memory_gb': None,
            'cuda_version': None,
            'is_rtx_3080': False,
            'driver_version': None,
            'architectures': []
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['name'] = torch.cuda.get_device_name(0)
                gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_info['cuda_version'] = torch.version.cuda
                
                if '3080' in gpu_info['name']:
                    gpu_info['is_rtx_3080'] = True
                    gpu_info['compute_capability'] = torch.cuda.get_device_capability(0)
                    gpu_info['tensor_cores'] = True
                    gpu_info['optimal_precision'] = 'float16'
                    gpu_info['architectures'] = ['Ampere']
                    gpu_info['sm_count'] = 68
                    gpu_info['cuda_cores'] = 8704
                    gpu_info['tensor_cores_count'] = 272
                    gpu_info['rt_cores_count'] = 68
                    gpu_info['memory_type'] = 'GDDR6X'
                    gpu_info['memory_bus_width'] = 320
                    gpu_info['max_power_draw_watts'] = 320
                    gpu_info['recommended_settings'] = {
                        'compute_type': 'float16',
                        'tensor_cores': True,
                        'cuda_streams': 3,
                        'reserved_memory_mb': 256,
                        'max_batch_size': 16,
                        'memory_fraction': .95,
                        'cudnn_benchmark': True,
                        'trt_optimization': True,
                        'dynamic_mem_mgmt': True,
                        'amp_optimization_level': 'O2'
                    }
                    
                    if platform.system() == 'Windows':
                        self._enhance_windows_gpu_detection(gpu_info)
                        
                try:
                    test_tensor = torch.ones(100, 100, device='cuda')
                    test_result = torch.matmul(test_tensor, test_tensor)
                    del test_tensor, test_result
                    torch.cuda.empty_cache()
                    gpu_info['cuda_operations_test'] = 'passed'
                except Exception as e:
                    gpu_info['cuda_operations_test'] = 'failed'
                    gpu_info['cuda_operations_error'] = str(e)
                    
                gpu_info['memory_free_gb'] = (torch.cuda.get_device_properties(0).total_memory - 
                                             torch.cuda.memory_allocated(0) - 
                                             torch.cuda.memory_reserved(0)) / 1024**3
        except ImportError:
            pass
        except Exception:
            pass
            
        return gpu_info
        
    def _enhance_windows_gpu_detection(self, gpu_info: Dict[str, Any]) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info['memory_gb'] = info.total / 1024**3
            gpu_info['driver_version'] = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            
            try:
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                max_gpu_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                max_mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                
                gpu_info['clock_speeds'] = {
                    'gpu_current_mhz': gpu_clock,
                    'memory_current_mhz': mem_clock,
                    'gpu_max_mhz': max_gpu_clock,
                    'memory_max_mhz': max_mem_clock
                }
            except Exception:
                pass
                
            pynvml.nvmlShutdown()
        except ImportError:
            pass
        except Exception:
            pass
            
    def detect_disk(self) -> Dict[str, Any]:
        disk_info = {
            'root_free_gb': 0,
            'models_free_gb': 0,
            'is_ssd': False
        }
        
        try:
            root_usage = psutil.disk_usage('/')
            disk_info['root_free_gb'] = root_usage.free / 1024**3
            disk_info['root_total_gb'] = root_usage.total / 1024**3
            disk_info['root_used_gb'] = root_usage.used / 1024**3
            disk_info['root_percent'] = root_usage.percent
            
            try:
                models_dir = os.path.abspath('models')
                if os.path.exists(models_dir):
                    models_usage = psutil.disk_usage(models_dir)
                    disk_info['models_free_gb'] = models_usage.free / 1024**3
                    disk_info['models_total_gb'] = models_usage.total / 1024**3
                    disk_info['models_used_gb'] = models_usage.used / 1024**3
                    disk_info['models_percent'] = models_usage.percent
            except Exception:
                pass
                
            if platform.system() == 'Windows':
                self._enhance_windows_disk_detection(disk_info)
        except Exception:
            pass
            
        return disk_info
        
    def _enhance_windows_disk_detection(self, disk_info: Dict[str, Any]) -> None:
        try:
            import wmi
            c = wmi.WMI()
            disk_info['disks'] = []
            
            for disk in c.Win32_DiskDrive():
                disk_data = {}
                model = disk.Model.lower() if disk.Model else ''
                disk_data['model'] = disk.Model
                disk_data['size_gb'] = int(disk.Size) / 1024**3 if hasattr(disk, 'Size') and disk.Size else 0
                disk_data['interface_type'] = disk.InterfaceType if hasattr(disk, 'InterfaceType') else 'Unknown'
                is_ssd = any(x in model for x in ['ssd', 'nvme', 'm.2', 'solid'])
                disk_data['is_ssd'] = is_ssd
                
                if is_ssd:
                    disk_info['is_ssd'] = True
                    disk_info['ssd_model'] = disk.Model
                    
                disk_info['disks'].append(disk_data)
        except ImportError:
            pass
        except Exception:
            pass
            
    def get_resource_requirements_for_state(self, state) -> Dict[str, Any]:
        from maggie.core.state import State
        
        base_requirements = {
            'memory_mb': 512,
            'cpu_cores': 1,
            'gpu_memory_mb': 0,
            'disk_mb': 100,
            'priority': 'normal'
        }
        
        state_requirements = {
            State.INIT: {'memory_mb': 1024, 'cpu_cores': 2, 'gpu_memory_mb': 0, 'priority': 'normal'},
            State.STARTUP: {'memory_mb': 2048, 'cpu_cores': 4, 'gpu_memory_mb': 1024, 'priority': 'high'},
            State.IDLE: {'memory_mb': 1024, 'cpu_cores': 1, 'gpu_memory_mb': 256, 'priority': 'low'},
            State.LOADING: {'memory_mb': 4096, 'cpu_cores': 6, 'gpu_memory_mb': 4096, 'priority': 'high'},
            State.READY: {'memory_mb': 2048, 'cpu_cores': 2, 'gpu_memory_mb': 2048, 'priority': 'normal'},
            State.ACTIVE: {'memory_mb': 4096, 'cpu_cores': 4, 'gpu_memory_mb': 6144, 'priority': 'high'},
            State.BUSY: {'memory_mb': 8192, 'cpu_cores': 8, 'gpu_memory_mb': 8192, 'priority': 'high'},
            State.CLEANUP: {'memory_mb': 2048, 'cpu_cores': 4, 'gpu_memory_mb': 1024, 'priority': 'normal'},
            State.SHUTDOWN: {'memory_mb': 1024, 'cpu_cores': 2, 'gpu_memory_mb': 0, 'priority': 'normal'}
        }
        
        requirements = state_requirements.get(state, base_requirements).copy()
        system_info = self.detect_system()
        
        if system_info['cpu'].get('is_ryzen_9_5900x', False):
            if state in [State.LOADING, State.ACTIVE, State.BUSY]:
                requirements['cpu_cores'] = min(requirements['cpu_cores'] + 2, 10)
            requirements['cpu_affinity'] = list(range(requirements['cpu_cores']))
            
        if system_info['gpu'].get('is_rtx_3080', False):
            if state in [State.ACTIVE, State.BUSY]:
                requirements['gpu_memory_mb'] = min(requirements['gpu_memory_mb'] * 1.2, 8192)
            requirements['gpu_precision'] = 'float16'
            requirements['use_tensor_cores'] = True
            
        if system_info['memory'].get('is_32gb', False) and system_info['memory'].get('is_xpg_d10', False):
            requirements['memory_mb'] = min(requirements['memory_mb'] * 1.5, 16384)
            requirements['use_large_pages'] = True
            
        return requirements
        
    def get_resource_availability_report(self) -> Dict[str, Any]:
        system_info = self.detect_system()
        memory = psutil.virtual_memory()
        memory_available_mb = memory.available / 1024**2
        cpu_count = psutil.cpu_count(logical=False) or 0
        cpu_percent = psutil.cpu_percent(interval=.1) / 100
        cpu_available = max(0, cpu_count - cpu_count * cpu_percent)
        gpu_memory_available_mb = 0
        gpu_memory_total_mb = 0
        
        if system_info['gpu']['available']:
            try:
                import torch
                if torch.cuda.is_available():
                    total = torch.cuda.get_device_properties(0).total_memory
                    allocated = torch.cuda.memory_allocated(0)
                    reserved = torch.cuda.memory_reserved(0)
                    gpu_memory_total_mb = total / 1024**2
                    gpu_memory_available_mb = (total - allocated - reserved) / 1024**2
            except:
                pass
                
        report = {
            'timestamp': time.time(),
            'memory': {
                'total_mb': memory.total / 1024**2,
                'available_mb': memory_available_mb,
                'percent_available': 100 - memory.percent
            },
            'cpu': {
                'total_cores': cpu_count,
                'available_cores': cpu_available,
                'percent_available': 100 - psutil.cpu_percent(interval=None)
            },
            'gpu': {
                'available': system_info['gpu']['available'],
                'total_memory_mb': gpu_memory_total_mb,
                'available_memory_mb': gpu_memory_available_mb,
                'percent_available': gpu_memory_available_mb / gpu_memory_total_mb * 100 if gpu_memory_total_mb > 0 else 0
            }
        }
        
        from maggie.core.state import State
        report['status'] = {
            'can_support_idle': True,
            'can_support_ready': report['memory']['available_mb'] >= 2048 and report['cpu']['available_cores'] >= 2,
            'can_support_active': report['memory']['available_mb'] >= 4096 and report['cpu']['available_cores'] >= 4 and
                                (not system_info['gpu']['available'] or report['gpu']['available_memory_mb'] >= 4096),
            'can_support_busy': report['memory']['available_mb'] >= 8192 and report['cpu']['available_cores'] >= 6 and
                               (not system_info['gpu']['available'] or report['gpu']['available_memory_mb'] >= 8192)
        }
        
        return report
        
    def optimize_detection_for_state(self, state) -> Dict[str, Any]:
        from maggie.core.state import State
        
        optimizations = {
            'full_detection': False,
            'cpu_detection': True,
            'memory_detection': True,
            'gpu_detection': True,
            'disk_detection': True,
            'detailed_detection': False,
            'fast_mode': False
        }
        
        if state == State.INIT or state == State.STARTUP:
            optimizations['full_detection'] = True
            optimizations['detailed_detection'] = True
        elif state == State.IDLE:
            optimizations['gpu_detection'] = False
            optimizations['disk_detection'] = False
            optimizations['fast_mode'] = True
        elif state == State.LOADING:
            optimizations['cpu_detection'] = False
            optimizations['disk_detection'] = False
            optimizations['detailed_detection'] = True
        elif state == State.ACTIVE or state == State.BUSY:
            optimizations['detailed_detection'] = False
            optimizations['fast_mode'] = True
        elif state == State.CLEANUP or state == State.SHUTDOWN:
            optimizations['detailed_detection'] = False
            optimizations['gpu_detection'] = False
            optimizations['fast_mode'] = True
            
        return optimizations
        
    def get_detailed_hardware_report(self) -> Dict[str, Any]:
        hardware_info = self.detect_system()
        report = {
            'hardware': hardware_info,
            'optimizations': {},
            'recommendations': []
        }
        
        cpu_info = hardware_info['cpu']
        if cpu_info.get('is_ryzen_9_5900x', False):
            report['optimizations']['cpu'] = {
                'use_cores_for_processing': min(8, cpu_info.get('physical_cores', 4)),
                'use_pbo': True,
                'affinity_strategy': 'performance_cores',
                'priority_boost': True
            }
        else:
            cores = cpu_info.get('physical_cores', 0)
            report['optimizations']['cpu'] = {
                'use_cores_for_processing': max(1, min(cores - 2, int(cores * .75))) if cores > 0 else 2
            }
            
            if cores < 8:
                report['recommendations'].append('CPU has fewer than 8 cores, which may impact performance')
                
        memory_info = hardware_info['memory']
        if memory_info.get('is_32gb', False):
            if memory_info.get('is_ddr4_3200', False) or memory_info.get('is_xpg_d10', False):
                report['optimizations']['memory'] = {
                    'max_percent': 80,
                    'cache_size_mb': 6144,
                    'preloading': True
                }
            else:
                report['optimizations']['memory'] = {
                    'max_percent': 75,
                    'cache_size_mb': 4096,
                    'preloading': True
                }
        else:
            total_gb = memory_info.get('total_gb', 8)
            if total_gb < 16:
                report['recommendations'].append(
                    f"System has only {total_gb:.1f}GB RAM, which may not be sufficient for optimal performance"
                )
                report['optimizations']['memory'] = {
                    'max_percent': 60,
                    'cache_size_mb': min(2048, int(total_gb * 100))
                }
            else:
                report['optimizations']['memory'] = {
                    'max_percent': 70,
                    'cache_size_mb': min(4096, int(total_gb * 150))
                }
                
        gpu_info = hardware_info['gpu']
        if gpu_info.get('available', False):
            if gpu_info.get('is_rtx_3080', False):
                report['optimizations']['gpu'] = {
                    'compute_type': 'float16',
                    'tensor_cores': True,
                    'cuda_streams': 3,
                    'reserved_memory_mb': 256,
                    'max_batch_size': 16,
                    'memory_fraction': .95,
                    'cudnn_benchmark': True,
                    'trt_optimization': True,
                    'dynamic_mem_mgmt': True,
                    'amp_optimization_level': 'O2'
                }
                
                if gpu_info.get('cuda_operations_test') == 'failed':
                    report['recommendations'].append(
                        f"CUDA operations test failed: {gpu_info.get('cuda_operations_error', 'Unknown error')}"
                    )
                    report['recommendations'].append('GPU acceleration may not be fully functional')
            else:
                memory_gb = gpu_info.get('memory_gb', 0)
                report['optimizations']['gpu'] = {
                    'compute_type': 'float16' if memory_gb >= 6 else 'int8',
                    'cuda_streams': 1,
                    'reserved_memory_mb': min(512, max(128, int(memory_gb * 64))),
                    'max_batch_size': min(16, max(4, int(memory_gb / 2)))
                }
                
                if memory_gb < 8:
                    report['recommendations'].append(
                        f"GPU has only {memory_gb:.1f}GB VRAM, which may limit model size and performance"
                    )
        else:
            report['optimizations']['gpu'] = {'enabled': False}
            report['recommendations'].append('No CUDA-capable GPU detected, falling back to CPU-only mode')
            report['recommendations'].append('Performance may be significantly reduced without GPU acceleration')
            
        return report