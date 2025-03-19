"""
Maggie AI Assistant - Hardware Management Module
===============================================
Comprehensive hardware detection and optimization for Maggie AI Assistant.

This module provides centralized hardware management, including detection,
optimization, and performance monitoring specifically tailored for
AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.

Examples
--------
>>> from hardware_manager import HardwareManager
>>> hw_manager = HardwareManager()
>>> hw_info = hw_manager._detect_system()
>>> print(f"CPU: {hw_info['cpu']['model']}")
>>> print(f"GPU: {hw_info['gpu']['name']}")
>>> # Start monitoring system resources
>>> hw_manager.start_monitoring()
>>> # Later, stop monitoring
>>> hw_manager.stop_monitoring()
"""

# Standard library imports
import os
import platform
import subprocess
import json
import threading
import time
from typing import Dict, Any, Optional, List, Tuple

# Third-party imports
import psutil
from loguru import logger

__all__ = ['HardwareManager']

class HardwareManager:
    """
    Unified hardware detection and optimization manager.
    
    Provides comprehensive hardware management capabilities including
    system detection, configuration optimization, and performance monitoring
    with specific enhancements for AMD Ryzen 9 5900X and NVIDIA RTX 3080.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file, by default "config.yaml"
    
    Attributes
    ----------
    hardware_info : Dict[str, Any]
        Detected hardware information
    optimization_profile : Dict[str, Any]
        Hardware-specific optimization profile
    _monitoring_enabled : bool
        Whether resource monitoring is currently enabled
    _monitoring_thread : Optional[threading.Thread]
        Thread for resource monitoring
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the hardware manager.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration file, by default "config.yaml"
        """
        self.config_path = config_path
        self.hardware_info = self._detect_system()
        self.optimization_profile = self._create_optimization_profile()
        
        # Performance monitoring
        self._monitoring_enabled = False
        self._monitoring_thread = None
        
        # For tracking resource trends
        self._cpu_history = []
        self._memory_history = []
        self._gpu_memory_history = []
        self._resource_history_max_samples = 60  # 5 minutes at 5-second intervals

    def _detect_system(self) -> Dict[str, Any]:
        """
        Detect comprehensive system hardware information.
        
        Returns
        -------
        Dict[str, Any]
            Detailed hardware information including CPU, RAM, and GPU
            
        Notes
        -----
        Identifies system components and capabilities, with special detection
        for AMD Ryzen 9 5900X CPU and NVIDIA RTX 3080 GPU.
        """
        system_info = {
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version()
            },
            "cpu": self._detect_cpu(),
            "memory": self._detect_memory(),
            "gpu": self._detect_gpu(),
            "disk": self._detect_disk()  # Added disk detection
        }
        
        # Log detected hardware
        logger.info(f"Detected OS: {system_info['os']['system']} {system_info['os']['release']}")
        
        if system_info["cpu"]["is_ryzen_9_5900x"]:
            logger.info("Detected AMD Ryzen 9 5900X CPU - applying optimized settings")
        else:
            logger.info(f"Detected CPU: {system_info['cpu']['model']} with {system_info['cpu']['physical_cores']} cores")
            
        if system_info["gpu"]["is_rtx_3080"]:
            logger.info("Detected NVIDIA RTX 3080 GPU - applying optimized settings")
        elif system_info["gpu"]["available"]:
            logger.info(f"Detected GPU: {system_info['gpu']['name']} with {system_info['gpu']['memory_gb']:.2f}GB VRAM")
        else:
            logger.warning("No compatible GPU detected - some features may be limited")
            
        return system_info
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """
        Detect CPU information with detailed Ryzen 9 5900X detection.
        
        Returns
        -------
        Dict[str, Any]
            CPU information including model, cores, and Ryzen 9 5900X detection
        """
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "model": platform.processor(),
            "is_ryzen_9_5900x": False,
            "frequency_mhz": {
                "current": 0,
                "min": 0,
                "max": 0
            }
        }
        
        # Check if CPU is Ryzen 9 5900X
        model_lower = cpu_info["model"].lower()
        if "ryzen 9" in model_lower and "5900x" in model_lower:
            cpu_info["is_ryzen_9_5900x"] = True
        
        # Get CPU frequency if available
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info["frequency_mhz"] = {
                    "current": cpu_freq.current,
                    "min": cpu_freq.min,
                    "max": cpu_freq.max
                }
        except:
            pass
            
        # Additional Ryzen 9 5900X identification approaches
        if platform.system() == "Windows":
            try:
                import wmi
                c = wmi.WMI()
                for processor in c.Win32_Processor():
                    if "Ryzen 9 5900X" in processor.Name:
                        cpu_info["is_ryzen_9_5900x"] = True
                        cpu_info["model"] = processor.Name
                        cpu_info["frequency_mhz"]["max"] = processor.MaxClockSpeed
                        break
            except ImportError:
                logger.debug("WMI module not available for detailed CPU detection")
        
        return cpu_info
    
    def _detect_memory(self) -> Dict[str, Any]:
        """
        Detect system memory information.
        
        Returns
        -------
        Dict[str, Any]
            Memory information including total, available, and DDR type if possible
        """
        memory = psutil.virtual_memory()
        memory_info = {
            "total_bytes": memory.total,
            "total_gb": memory.total / (1024**3),
            "available_bytes": memory.available,
            "available_gb": memory.available / (1024**3),
            "percent_used": memory.percent,
            "is_32gb": 30 <= memory.total / (1024**3) <= 34  # 32GB with tolerance
        }
        
        # Try to detect memory type on Windows
        if platform.system() == "Windows":
            try:
                import wmi
                c = wmi.WMI()
                for physical_memory in c.Win32_PhysicalMemory():
                    if hasattr(physical_memory, 'PartNumber') and physical_memory.PartNumber:
                        if "DDR4" in physical_memory.PartNumber:
                            memory_info["type"] = "DDR4"
                            # If 3200 MHz is in the part number, tag it
                            if "3200" in physical_memory.PartNumber:
                                memory_info["speed"] = "3200MHz"
                            break
            except ImportError:
                logger.debug("WMI module not available for detailed memory detection")
                
        return memory_info
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """
        Detect GPU information with detailed RTX 3080 detection.
        
        Returns
        -------
        Dict[str, Any]
            GPU information including model, VRAM, and RTX 3080 detection
        """
        gpu_info = {
            "available": False,
            "name": None,
            "memory_gb": None,
            "cuda_version": None,
            "is_rtx_3080": False,
            "driver_version": None,
            "architectures": []
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["name"] = torch.cuda.get_device_name(0)
                gpu_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_info["cuda_version"] = torch.version.cuda
                
                # Check if GPU is RTX 3080
                if "3080" in gpu_info["name"]:
                    gpu_info["is_rtx_3080"] = True
                    gpu_info["compute_capability"] = torch.cuda.get_device_capability(0)
                    gpu_info["tensor_cores"] = True
                    gpu_info["optimal_precision"] = "float16"
                    gpu_info["architectures"] = ["Ampere"]
                    
                    # Get more detailed information on Windows
                    if platform.system() == "Windows":
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_info["memory_gb"] = info.total / 1024**3
                            gpu_info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
                            pynvml.nvmlShutdown()
                        except ImportError:
                            pass
                            
                # Get VRAM usage
                gpu_info["memory_free_gb"] = (torch.cuda.get_device_properties(0).total_memory - 
                                              torch.cuda.memory_allocated(0) - 
                                              torch.cuda.memory_reserved(0)) / (1024**3)
                
        except ImportError:
            logger.debug("PyTorch not available for GPU detection")
            
        return gpu_info
        
    def _detect_disk(self) -> Dict[str, Any]:
        """
        Detect disk information.
        
        Returns
        -------
        Dict[str, Any]
            Disk information including available space for model storage
        """
        disk_info = {
            "root_free_gb": 0,
            "models_free_gb": 0,
            "is_ssd": False
        }
        
        # Check root directory space
        try:
            root_usage = psutil.disk_usage('/')
            disk_info["root_free_gb"] = root_usage.free / (1024**3)
        except:
            pass
            
        # Check models directory space
        try:
            models_dir = os.path.abspath("models")
            if os.path.exists(models_dir):
                models_usage = psutil.disk_usage(models_dir)
                disk_info["models_free_gb"] = models_usage.free / (1024**3)
        except:
            pass
            
        # Try to detect if SSD on Windows
        if platform.system() == "Windows":
            try:
                import wmi
                c = wmi.WMI()
                for disk in c.Win32_DiskDrive():
                    # Most SSDs will report this in the model name
                    model = disk.Model.lower() if disk.Model else ""
                    if any(x in model for x in ["ssd", "nvme", "m.2"]):
                        disk_info["is_ssd"] = True
                        disk_info["model"] = disk.Model
                        break
            except ImportError:
                logger.debug("WMI module not available for SSD detection")
                
        return disk_info
    
    def _create_optimization_profile(self) -> Dict[str, Any]:
        """
        Create hardware-specific optimization profile.
        
        Returns
        -------
        Dict[str, Any]
            Optimization parameters tuned for the detected hardware
        """
        profile = {
            "threading": self._optimize_threading(),
            "memory": self._optimize_memory(),
            "gpu": self._optimize_gpu(),
            "llm": self._optimize_llm(),
            "audio": self._optimize_audio()
        }
        
        return profile
    
    def _optimize_threading(self) -> Dict[str, Any]:
        """
        Create threading optimization parameters.
        
        Returns
        -------
        Dict[str, Any]
            Threading optimization parameters based on detected CPU
        """
        cpu_info = self.hardware_info["cpu"]
        
        # Ryzen 9 5900X optimized threading
        if cpu_info["is_ryzen_9_5900x"]:
            return {
                "max_workers": 10,  # Increased from 8 to 10 for better Ryzen 9 5900X utilization
                "thread_timeout": 30,
                # Improved thread affinity for Ryzen 9 5900X's CCX architecture
                # Use performance cores in both CCXs (Ryzen 9 5900X has 2x6 cores)
                "worker_affinity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "priority_boost": True  # Enable priority boost for workloads
            }
        # General threading optimization based on core count
        else:
            physical_cores = cpu_info["physical_cores"]
            # Leave at least 2 cores for system processes
            max_workers = max(2, min(physical_cores - 2, physical_cores * 4 // 5))
            return {
                "max_workers": max_workers,
                "thread_timeout": 30
            }
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """
        Create memory optimization parameters.
        
        Returns
        -------
        Dict[str, Any]
            Memory optimization parameters based on detected RAM
        """
        memory_info = self.hardware_info["memory"]
        
        # 32GB RAM optimization (DDR4-3200) for Ryzen 9 5900X
        if memory_info["is_32gb"] and memory_info.get("type") == "DDR4" and memory_info.get("speed") == "3200MHz":
            return {
                "max_percent": 80,  # Increased from 75% to 80% for 32GB systems
                "unload_threshold": 85,
                "cache_size_mb": 6144,  # Increased from 4GB to 6GB cache for 32GB systems
                "preloading": True,  # Enable model preloading
                "large_pages": True,  # Try to use large pages for better performance (if available)
                "numa_aware": True    # Optimize for NUMA architecture in Ryzen
            }
        # 32GB RAM optimization (generic)
        elif memory_info["is_32gb"]:
            return {
                "max_percent": 75,
                "unload_threshold": 85,
                "cache_size_mb": 4096,
                "preloading": True
            }
        # Optimization for lower memory systems
        else:
            total_gb = memory_info["total_gb"]
            return {
                "max_percent": min(70, max(50, int(60 + (total_gb - 8) * 1.25))),
                "unload_threshold": min(80, max(60, int(70 + (total_gb - 8) * 1.25))),
                "cache_size_mb": min(2048, max(512, int(total_gb * 64)))
            }
    
    def _optimize_gpu(self) -> Dict[str, Any]:
        """
        Create GPU optimization parameters.
        
        Returns
        -------
        Dict[str, Any]
            GPU optimization parameters based on detected GPU
        """
        gpu_info = self.hardware_info["gpu"]
        
        if not gpu_info["available"]:
            return {
                "enabled": False
            }
        
        # RTX 3080 specific optimizations (enhanced for 10GB VRAM)
        if gpu_info["is_rtx_3080"]:
            return {
                "enabled": True,
                "compute_type": "float16",  # Use mixed precision for RTX 3080
                "tensor_cores": True,
                "cuda_streams": 3,        # Increased from 2 to 3 for better parallelism
                "reserved_memory_mb": 256, # Reduced from 512MB to 256MB to leave more for models
                "max_batch_size": 16,
                "memory_fraction": 0.95,  # Use 95% of VRAM for ML tasks
                "cudnn_benchmark": True,  # Enable cuDNN benchmarking for optimal kernels
                "trt_optimization": True, # Enable TensorRT optimizations if available
                "bfloat16_supported": False, # RTX 3080 doesn't support BF16 natively
                "dynamic_mem_mgmt": True,  # Enable dynamic memory management
                "vram_gb": gpu_info.get("memory_gb", 10), # Typically 10GB for RTX 3080
                "vram_efficient_loading": True,
                "amp_optimization_level": "O2"  # Automatic Mixed Precision optimization level
            }
        # General GPU optimizations
        else:
            memory_gb = gpu_info.get("memory_gb", 0)
            return {
                "enabled": True,
                "compute_type": "float16" if memory_gb >= 6 else "int8",
                "tensor_cores": "tensor_cores" in gpu_info.get("name", "").lower(),
                "cuda_streams": 1,
                "reserved_memory_mb": min(512, max(128, int(memory_gb * 64))),
                "max_batch_size": min(16, max(4, int(memory_gb / 2)))
            }
    
    def _optimize_llm(self) -> Dict[str, Any]:
        """
        Create LLM optimization parameters.
        
        Returns
        -------
        Dict[str, Any]
            LLM optimization parameters based on detected GPU
        """
        gpu_info = self.hardware_info["gpu"]
        
        # RTX 3080 optimizations for LLM (enhanced for 10GB VRAM)
        if gpu_info["is_rtx_3080"]:
            return {
                "gpu_layers": 32,  # Optimal for 10GB VRAM
                "precision": "float16",
                "kv_cache_optimization": True,  # Enable KV cache optimization
                "context_length": 8192,
                "attention_sinks": True,      # Enable attention sinks for longer contexts
                "auto_adjust": True,
                "tensor_parallel": 1,         # Single GPU
                "draft_model": "small",       # Use small draft model for speculative decoding
                "speculative_decoding": True, # Enable speculative decoding for faster generations
                "llm_rope_scaling": "dynamic", # Use dynamic rope scaling for long contexts
                "batch_inference": True,      # Enable batch inference
                "offload_layers": {
                    "enabled": True,
                    "threshold_gb": 9.0,      # Offload when less than 9GB VRAM available
                    "cpu_layers": [0, 1]      # Offload first two layers if needed
                }
            }
        # General GPU optimizations
        elif gpu_info["available"]:
            memory_gb = gpu_info.get("memory_gb", 0)
            return {
                "gpu_layers": min(40, max(1, int(memory_gb * 3.2))),
                "precision": "float16" if memory_gb >= 6 else "int8",
                "context_length": min(8192, max(2048, int(memory_gb * 819.2))),
                "auto_adjust": True
            }
        # CPU fallback
        else:
            return {
                "gpu_layers": 0,
                "precision": "int8",
                "context_length": 2048,
                "auto_adjust": False
            }
    
    def _optimize_audio(self) -> Dict[str, Any]:
        """
        Create audio processing optimization parameters.
        
        Returns
        -------
        Dict[str, Any]
            Audio optimization parameters based on detected hardware
        """
        cpu_info = self.hardware_info["cpu"]
        gpu_info = self.hardware_info["gpu"]
        
        # Audio processing optimizations
        audio_opt = {
            "sample_rate": 22050,
            "chunk_size": 1024,
            "use_gpu": gpu_info["available"],
            "buffer_size": 4096
        }
        
        # Adjust settings based on hardware
        if cpu_info["is_ryzen_9_5900x"]:
            audio_opt["chunk_size"] = 512  # Lower latency for high-performance CPU
            audio_opt["audio_threads"] = 2  # Dedicate 2 threads for audio processing
            
        if gpu_info["is_rtx_3080"]:
            audio_opt["whisper_model"] = "small"  # Better model for RTX 3080
            audio_opt["whisper_compute"] = "float16"
            audio_opt["cache_models"] = True
            audio_opt["vad_sensitivity"] = 0.6  # Optimized VAD sensitivity
            audio_opt["cuda_audio_processing"] = True  # Use CUDA for audio when possible
        elif gpu_info["available"]:
            audio_opt["whisper_model"] = "base"
            audio_opt["whisper_compute"] = "float16" if gpu_info.get("memory_gb", 0) >= 6 else "int8"
        else:
            audio_opt["whisper_model"] = "tiny"
            audio_opt["whisper_compute"] = "int8"
            
        return audio_opt
    
    def apply_hardware_optimizations(self, hardware_info: Dict[str, Any]) -> None:
        """
        Apply hardware-specific optimizations to the configuration.
        
        Parameters
        ----------
        hardware_info : Dict[str, Any]
            Hardware information dictionary containing CPU, GPU, and memory details
            
        Notes
        -----
        Optimizes configuration based on detected hardware capabilities,
        with specific optimizations for Ryzen 9 5900X and RTX 3080.
        """
        # CPU optimizations for Ryzen 9 5900X
        self._apply_cpu_optimizations(hardware_info.get("cpu", {}))
            
        # Memory optimizations for 32GB RAM
        self._apply_memory_optimizations(hardware_info.get("memory", {}))
            
        # GPU optimizations for RTX 3080
        self._apply_gpu_optimizations(hardware_info.get("gpu", {}))

    # def optimize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Apply hardware-specific optimizations to configuration.
        
    #     Parameters
    #     ----------
    #     config : Dict[str, Any]
    #         Current configuration dictionary
            
    #     Returns
    #     -------
    #     Dict[str, Any]
    #         Optimized configuration dictionary with hardware-specific settings
    #     """
    #     # Create a copy of the config to avoid modifying the original
    #     optimized_config = config.copy()
        
    #     # Apply threading optimizations
    #     threading_opt = self.optimization_profile["threading"]
    #     optimized_config.setdefault("threading", {})
    #     optimized_config["threading"].update(threading_opt)
        
    #     # Apply memory optimizations
    #     memory_opt = self.optimization_profile["memory"]
    #     optimized_config.setdefault("memory", {})
    #     optimized_config["memory"].update(memory_opt)
        
    #     # Apply LLM optimizations
    #     llm_opt = self.optimization_profile["llm"]
    #     optimized_config.setdefault("llm", {})
    #     optimized_config["llm"].update(llm_opt)
        
    #     # Apply audio optimizations
    #     audio_opt = self.optimization_profile["audio"]
    #     optimized_config.setdefault("speech", {})
    #     optimized_config["speech"].setdefault("whisper", {})
    #     optimized_config["speech"]["whisper"]["model_size"] = audio_opt["whisper_model"]
    #     optimized_config["speech"]["whisper"]["compute_type"] = audio_opt["whisper_compute"]
        
    #     # Apply GPU optimizations if available
    #     if self.hardware_info["gpu"]["available"]:
    #         gpu_opt = self.optimization_profile["gpu"]
    #         optimized_config.setdefault("gpu", {})
    #         optimized_config["gpu"].update(gpu_opt)
        
    #     return optimized_config
    
    def start_monitoring(self, interval: float = 5.0) -> bool:
        """
        Start monitoring system resource usage.
        
        Parameters
        ----------
        interval : float, optional
            Monitoring interval in seconds, by default 5.0
            
        Returns
        -------
        bool
            True if monitoring started successfully, False otherwise
        """
        if self._monitoring_enabled:
            return True
            
        self._monitoring_enabled = True
        self._monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True,
            name="ResourceMonitorThread"
        )
        self._monitoring_thread.start()
        
        logger.info(f"Resource monitoring started with {interval}s interval")
        return True
    
    def stop_monitoring(self) -> bool:
        """
        Stop monitoring system resource usage.
        
        Returns
        -------
        bool
            True if monitoring stopped successfully, False otherwise
        """
        if not self._monitoring_enabled:
            return False
            
        self._monitoring_enabled = False
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
            
        logger.info("Resource monitoring stopped")
        return True
    
    def _monitor_resources(self, interval: float) -> None:
        """
        Monitor system resource usage and log warnings when thresholds are exceeded.
        
        Parameters
        ----------
        interval : float
            Monitoring interval in seconds
        """
        while self._monitoring_enabled:
            try:
                cpu_percent = psutil.cpu_percent(percpu=True)
                cpu_avg = sum(cpu_percent) / len(cpu_percent) if cpu_percent else 00
                memory = psutil.virtual_memory()
                
                # Store historical data for trend analysis
                self._cpu_history.append(cpu_avg)
                self._memory_history.append(memory.percent)
                
                # Limit history size
                if len(self._cpu_history) > self._resource_history_max_samples:
                    self._cpu_history.pop(0)
                if len(self._memory_history) > self._resource_history_max_samples:
                    self._memory_history.pop(0)
                
                # Get GPU utilization if available
                gpu_util = self._get_gpu_utilization()
                
                # If GPU is available, store history
                if gpu_util and "memory_percent" in gpu_util:
                    self._gpu_memory_history.append(gpu_util["memory_percent"])
                    if len(self._gpu_memory_history) > self._resource_history_max_samples:
                        self._gpu_memory_history.pop(0)
                
                # Log resource usage if exceeding thresholds
                self._check_resource_thresholds(cpu_avg, memory, gpu_util)
                
                # Check for trends that might indicate issues
                self._check_resource_trends()
                    
                # Sleep for specified interval
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                time.sleep(interval)
    
    def _get_gpu_utilization(self) -> Optional[Dict[str, float]]:
        """
        Get GPU utilization metrics if GPU is available.
        
        Returns
        -------
        Optional[Dict[str, float]]
            GPU utilization metrics or None if not available
        """
        if not self.hardware_info["gpu"]["available"]:
            return None
            
        try:
            import torch
            
            if torch.cuda.is_available():
                # Calculate memory usage percentage
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                total = torch.cuda.get_device_properties(0).total_memory
                
                memory_percent = allocated / total * 100
                reserved_percent = reserved / total * 100
                
                # Get active CUDA memory allocations
                active_allocations = 0
                if hasattr(torch.cuda, 'memory_stats'):
                    stats = torch.cuda.memory_stats(0)
                    active_allocations = stats.get("num_alloc_retries", 0)
                
                return {
                    "memory_allocated": allocated,
                    "memory_reserved": reserved,
                    "memory_total": total,
                    "memory_percent": memory_percent,
                    "reserved_percent": reserved_percent,
                    "active_allocations": active_allocations,
                    "fragmentation": (reserved - allocated) / total * 100 if reserved > allocated else 0
                }
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error getting GPU utilization: {e}")
            
        return None
    
    def _check_resource_thresholds(self, 
                                  cpu_percent: float, 
                                  memory: psutil._psplatform.svmem, 
                                  gpu_util: Optional[Dict[str, float]]) -> None:
        """
        Check resource utilization against thresholds and log warnings when exceeded.
        
        Parameters
        ----------
        cpu_percent : float
            CPU utilization percentage
        memory : psutil._psplatform.svmem
            Memory utilization information
        gpu_util : Optional[Dict[str, float]]
            GPU utilization information or None if not available
        """
        memory_threshold = self.optimization_profile["memory"]["unload_threshold"]
        
        # Check CPU core utilization for Ryzen 9 5900X optimization
        if self.hardware_info["cpu"].get("is_ryzen_9_5900x", False):
            # Check if any core is maxing out 
            per_core = psutil.cpu_percent(percpu=True)
            max_core = max(per_core) if per_core else 0
            cores_above_95 = sum(1 for core in per_core if core > 95)
            
            if cores_above_95 >= 4:  # 4+ cores maxed out indicates high load
                logger.warning(f"High load on {cores_above_95} CPU cores (>95% usage)")
            
            if cpu_percent > 85:  # Overall CPU usage high
                logger.warning(f"High overall CPU usage: {cpu_percent:.1f}%")
        else:
            # Generic CPU monitoring
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
        if memory.percent > memory_threshold:
            logger.warning(f"High memory usage: {memory.percent:.1f}% " +
                         f"(threshold: {memory_threshold}%) - " +
                         f"available: {memory.available / (1024**3):.1f} GB")
            
        if gpu_util:
            # More detailed GPU monitoring for RTX 3080
            if self.hardware_info["gpu"].get("is_rtx_3080", False):
                # Check memory usage
                if gpu_util["memory_percent"] > 90:
                    logger.warning(f"High GPU memory usage: {gpu_util['memory_percent']:.1f}% - " +
                                 f"allocated: {gpu_util['memory_allocated'] / (1024**3):.1f} GB")
                    
                # Check for memory fragmentation issues
                if gpu_util.get("fragmentation", 0) > 15:  # More than 15% fragmentation
                    logger.warning(f"High GPU memory fragmentation: {gpu_util['fragmentation']:.1f}% - " +
                                 f"consider clearing cache")
            else:
                # Basic GPU monitoring for other GPUs
                if gpu_util["memory_percent"] > 90:
                    logger.warning(f"High GPU memory usage: {gpu_util['memory_percent']:.1f}% - " +
                                 f"allocated: {gpu_util['memory_allocated'] / (1024**3):.1f} GB")
    
    def _check_resource_trends(self) -> None:
        """
        Analyze resource usage trends to detect potential issues.
        
        Detects steadily increasing resource usage that might indicate 
        memory leaks or other resource management issues.
        """
        # Need at least 10 samples for trend analysis
        if len(self._memory_history) < 10:
            return
            
        # Check for steadily increasing memory usage
        if all(self._memory_history[i] <= self._memory_history[i+1] for i in range(len(self._memory_history)-10, len(self._memory_history)-1)):
            # Memory usage has been steadily increasing for 10 samples
            if self._memory_history[-1] - self._memory_history[-10] > 10:  # >10% increase
                logger.warning("Memory usage steadily increasing - possible memory leak")
        
        # Check for GPU memory growth if available
        if len(self._gpu_memory_history) >= 10:
            if all(self._gpu_memory_history[i] <= self._gpu_memory_history[i+1] for i in range(len(self._gpu_memory_history)-10, len(self._gpu_memory_history)-1)):
                # GPU memory has been steadily increasing for 10 samples
                if self._gpu_memory_history[-1] - self._gpu_memory_history[-10] > 15:  # >15% increase
                    logger.warning("GPU memory usage steadily increasing - possible CUDA memory leak")