"""
Maggie AI Hardware Optimization Utility

Provides advanced system configuration and performance optimization 
for Maggie AI Assistant, with specific tuning for high-performance 
AMD Ryzen and NVIDIA GPU environments.
"""

import os
import platform
import psutil
import torch
import multiprocessing
from typing import Dict, Any, List, Optional

class HardwareOptimizer:
    """
    Comprehensive hardware configuration and optimization utility.

    Analyzes system capabilities, provides performance recommendations, 
    and dynamically adapts AI model configurations.

    Attributes
    ----------
    _system_info : Dict[str, Any]
        Comprehensive system configuration details
    """

    def __init__(self):
        """
        Initialize hardware optimization analysis.
        
        Captures detailed system configuration and capabilities.
        """
        self._system_info = self._detect_system_configuration()

    def _detect_system_configuration(self) -> Dict[str, Any]:
        """
        Perform comprehensive system configuration detection.

        Returns
        -------
        Dict[str, Any]
            Detailed system hardware and software configuration
        """
        system_config = {
            "os": {
                "name": platform.system(),
                "version": platform.release(),
                "architecture": platform.machine()
            },
            "cpu": {
                "name": platform.processor(),
                "physical_cores": os.cpu_count(),
                "logical_cores": multiprocessing.cpu_count(),
                "frequency": psutil.cpu_freq().current
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
            },
            "gpu": self._detect_gpu_configuration()
        }
        return system_config

    def _detect_gpu_configuration(self) -> Dict[str, Any]:
        """
        Detect and analyze GPU configuration with advanced details.

        Returns
        -------
        Dict[str, Any]
            Comprehensive GPU configuration details
        """
        gpu_info = {
            "cuda_available": False,
            "device_count": 0,
            "devices": []
        }

        try:
            if torch.cuda.is_available():
                gpu_info["cuda_available"] = True
                gpu_info["device_count"] = torch.cuda.device_count()
                
                for i in range(gpu_info["device_count"]):
                    device_details = {
                        "name": torch.cuda.get_device_name(i),
                        "compute_capability": torch.cuda.get_device_capability(i),
                        "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                    }
                    gpu_info["devices"].append(device_details)
        
        except ImportError:
            pass

        return gpu_info

    def optimize_pytorch_configuration(self) -> Dict[str, Any]:
        """
        Generate optimized PyTorch configuration for current hardware.

        Provides recommended settings for:
        - Tensor computation
        - Memory management
        - Parallel processing

        Returns
        -------
        Dict[str, Any]
            Recommended PyTorch optimization parameters
        """
        torch_optimizations = {
            "performance_hints": [],
            "recommended_settings": {}
        }

        # GPU Optimization
        if self._system_info['gpu']['cuda_available']:
            torch_optimizations['performance_hints'].append(
                "Use CUDA for accelerated tensor computations"
            )
            torch_optimizations['recommended_settings'] = {
                "backend": "cudnn",
                "precision": "float16",  # Mixed precision training
                "multi_gpu": self._system_info['gpu']['device_count'] > 1
            }

        # CPU Optimization
        torch_optimizations['recommended_settings'].update({
            "num_threads": max(1, os.cpu_count() - 2),  # Reserve some cores
            "use_openmp": True
        })

        return torch_optimizations

    def generate_performance_report(self) -> str:
        """
        Generate a human-readable performance and compatibility report.

        Returns
        -------
        str
            Formatted performance and compatibility report
        """
        report_lines = [
            "🚀 Maggie AI - System Performance Report",
            "=" * 50,
            f"Operating System: {self._system_info['os']['name']} {self._system_info['os']['version']}",
            f"CPU: {self._system_info['cpu']['name']}",
            f"Cores: {self._system_info['cpu']['physical_cores']} Physical / {self._system_info['cpu']['logical_cores']} Logical",
            f"Memory: {self._system_info['memory']['total_gb']} GB Total",
            "GPU Configuration:"
        ]

        if self._system_info['gpu']['cuda_available']:
            for idx, device in enumerate(self._system_info['gpu']['devices'], 1):
                report_lines.extend([
                    f"  Device {idx}: {device['name']}",
                    f"    Compute Capability: {device['compute_capability']}",
                    f"    Total Memory: {device['total_memory_gb']} GB"
                ])
        else:
            report_lines.append("  No CUDA-capable GPU detected")

        return "\n".join(report_lines)

def main():
    """
    Demonstrate hardware optimization capabilities.
    """
    optimizer = HardwareOptimizer()
    
    print(optimizer.generate_performance_report())
    print("\nPyTorch Optimizations:")
    print(json.dumps(optimizer.optimize_pytorch_configuration(), indent=2))

if __name__ == '__main__':
    main()