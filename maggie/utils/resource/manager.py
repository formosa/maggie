"""
Maggie AI Assistant - Resource Management Facade
===============================================

Primary interface for hardware detection, optimization, and resource monitoring.
Implements a facade pattern that delegates to specialized components.

This module provides a unified interface for:
1. Hardware detection (CPU, GPU, memory, disk)
2. Hardware-specific optimization
3. Runtime resource monitoring
4. Resource threshold management
5. GPU memory management

Optimized for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.
"""

import os
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Set

import psutil
from loguru import logger

from maggie.utils.resource.detector import HardwareDetector
from maggie.utils.resource.optimizer import HardwareOptimizer
from maggie.utils.resource.monitor import ResourceMonitor


class ResourceManager:
    """
    Primary facade for resource management functionality.
    
    This class serves as the main entry point for hardware detection,
    optimization, and resource monitoring functionality. It delegates
    specific tasks to specialized components.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary with resource management settings
        
    Attributes
    ----------
    hardware_info : Dict[str, Any]
        Detected hardware information
    config : Dict[str, Any]
        Application configuration
    _optimization_profile : Dict[str, Any]
        Generated optimization profile based on hardware
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the resource manager.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary with resource management settings
        """
        self.config = config
        
        # Initialize components
        self.detector = HardwareDetector()
        self.hardware_info = self.detector.detect_system()
        
        # Parse configuration
        self.cpu_config = config.get('cpu', {})
        self.memory_config = config.get('memory', {})
        self.gpu_config = config.get('gpu', {})
        
        # Set threshold values
        self.memory_max_percent = self.memory_config.get('max_percent', 75)
        self.memory_unload_threshold = self.memory_config.get('model_unload_threshold', 85)
        self.gpu_max_percent = self.gpu_config.get('max_percent', 90)
        self.gpu_unload_threshold = self.gpu_config.get('model_unload_threshold', 95)
        
        # Initialize optimizer and monitor
        self.optimizer = HardwareOptimizer(self.hardware_info, self.config)
        self.monitor = ResourceMonitor(
            self.config,
            self.hardware_info,
            memory_threshold=self.memory_unload_threshold,
            gpu_threshold=self.gpu_unload_threshold
        )
        
        # Create optimization profile
        self._optimization_profile = self.optimizer.create_optimization_profile()
        
        logger.info("Resource Manager initialized")
    
    def setup_gpu(self) -> None:
        """
        Set up GPU with optimized settings if available.
        
        Configures PyTorch and CUDA settings for optimal performance
        on the detected GPU hardware, particularly RTX 3080.
        """
        self.optimizer.setup_gpu()
    
    def start_monitoring(self, interval: float = 5.0) -> bool:
        """
        Start resource usage monitoring.
        
        Parameters
        ----------
        interval : float, optional
            Monitoring interval in seconds, by default 5.0
            
        Returns
        -------
        bool
            True if monitoring started successfully
        """
        return self.monitor.start(interval)
    
    def stop_monitoring(self) -> bool:
        """
        Stop resource usage monitoring.
        
        Returns
        -------
        bool
            True if monitoring stopped successfully
        """
        return self.monitor.stop()
    
    def clear_gpu_memory(self) -> bool:
        """
        Clear GPU memory cache to free up VRAM.
        
        Returns
        -------
        bool
            True if GPU memory was cleared successfully
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug('GPU memory cache cleared')
                return True
            return False
        except ImportError:
            logger.debug('PyTorch not available for clearing GPU cache')
            return False
        except Exception as e:
            logger.error(f"Error clearing GPU cache: {e}")
            return False
    
    def reduce_memory_usage(self) -> bool:
        """
        Reduce memory usage by performing garbage collection.
        
        Returns
        -------
        bool
            True if memory reduction was successful
        """
        success = True
        try:
            import gc
            gc.collect()
            logger.debug('Python garbage collection performed')
        except Exception as e:
            logger.error(f"Error performing garbage collection: {e}")
            success = False
        
        if not self.clear_gpu_memory():
            success = False
            
        return success
    
    def release_resources(self) -> bool:
        """
        Release system resources when transitioning to idle state.
        
        Returns
        -------
        bool
            True if resource release was successful
        """
        success = True
        if not self.reduce_memory_usage():
            success = False
            
        try:
            if os.name == 'nt':  # Windows
                import psutil
                psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS)
                logger.debug('Process priority reset to normal')
        except Exception as e:
            logger.error(f"Error releasing system resources: {e}")
            success = False
            
        return success
    
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get current status of system resources.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with CPU, memory, and GPU status
        """
        return self.monitor.get_current_status()
    
    def get_optimization_profile(self) -> Dict[str, Any]:
        """
        Get the hardware optimization profile.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with optimization settings
        """
        return self._optimization_profile