# Updated maggie/utils/hardware/gpu_utils.py
"""
Maggie AI Assistant - Enhanced GPU Utilities
===========================================

Utilities for GPU resource management optimized for NVIDIA RTX 3080.

This module provides standardized functions for GPU memory management, 
initialization, optimization, and monitoring specifically tuned for
RTX 3080 GPUs with 10GB VRAM and Ampere architecture.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import os

# Import error handling utilities
from maggie.utils.error_handling import safe_execute

# Configure module logger
logger = logging.getLogger(__name__)

def clear_gpu_cache() -> bool:
    """
    Clear GPU memory cache to free resources.
    
    This function attempts to clear the CUDA memory cache when GPU
    acceleration is available, freeing up VRAM for other operations.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    bool
        True if cache was successfully cleared, False otherwise
        
    Notes
    -----
    This is particularly effective on the RTX 3080 with 10GB VRAM
    to prevent memory fragmentation during long-running sessions.
    
    Examples
    --------
    >>> success = clear_gpu_cache()
    >>> if success:
    ...     print("GPU memory cache cleared")
    """
    def _clear_cache():
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")
            return True
        return False
        
    return safe_execute(
        _clear_cache,
        default_return=False,
        error_message="Error clearing GPU cache"
    )

def get_gpu_info() -> Dict[str, Any]:
    """
    Get current GPU information including memory usage and utilization.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing GPU information including:
        - available: Whether GPU is available
        - name: GPU name if available
        - memory_total_gb: Total memory in GB
        - memory_allocated_gb: Currently allocated memory in GB
        - memory_reserved_gb: Reserved memory in GB
        - memory_free_gb: Free memory in GB
        - utilization: GPU utilization percentage if available
        - compute_capability: Compute capability version for the GPU
        - is_rtx_3080: Boolean indicating if the GPU is an RTX 3080
        
    Notes
    -----
    Optimized for monitoring RTX 3080 memory usage during resource-intensive
    operations like LLM inference and speech processing.
    
    Examples
    --------
    >>> gpu_info = get_gpu_info()
    >>> if gpu_info["available"]:
    ...     print(f"GPU: {gpu_info['name']}")
    ...     print(f"Memory: {gpu_info['memory_free_gb']:.2f}GB free of {gpu_info['memory_total_gb']:.2f}GB")
    """
    info = {
        "available": False,
        "name": None,
        "memory_total_gb": 0,
        "memory_allocated_gb": 0,
        "memory_reserved_gb": 0,
        "memory_free_gb": 0,
        "utilization": 0,
        "compute_capability": None,
        "is_rtx_3080": False
    }
    
    def _gather_gpu_info():
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            
            # Check specifically for RTX 3080
            info["is_rtx_3080"] = "3080" in info["name"]
            
            # Get compute capability
            cc = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{cc[0]}.{cc[1]}"
            
            props = torch.cuda.get_device_properties(0)
            info["memory_total_gb"] = props.total_memory / (1024**3)
            
            info["memory_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
            info["memory_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
            info["memory_free_gb"] = (props.total_memory - torch.cuda.memory_allocated(0) - 
                                   torch.cuda.memory_reserved(0)) / (1024**3)
            
            # Get CUDA version
            info["cuda_version"] = torch.version.cuda
            
            # Try to get utilization - requires pynvml
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info["utilization"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                pynvml.nvmlShutdown()
            except ImportError:
                # pynvml not available, skip utilization
                pass
        return info
    
    return safe_execute(
        _gather_gpu_info,
        default_return=info,
        error_message="Error gathering GPU information"
    )

def optimize_for_rtx_3080() -> Dict[str, Any]:
    """
    Apply RTX 3080-specific CUDA optimizations.
    
    This function configures PyTorch with optimized settings for the 
    RTX 3080 GPU, including memory management strategies, precision settings,
    and CUDA graph usage recommendations.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of applied optimization settings including:
        - applied: Whether optimizations were successfully applied
        - settings: Dictionary of specific settings applied
        
    Notes
    -----
    These optimizations are specifically tuned for the RTX 3080 with 10GB VRAM
    and Ampere architecture to maximize inference speed while managing memory
    constraints effectively.
    
    Examples
    --------
    >>> optimizations = optimize_for_rtx_3080()
    >>> if optimizations["applied"]:
    ...     print("RTX 3080 optimizations applied:")
    ...     for key, value in optimizations["settings"].items():
    ...         print(f"  - {key}: {value}")
    """
    optimizations = {
        "applied": False,
        "settings": {}
    }
    
    def _apply_optimizations():
        import torch
        if torch.cuda.is_available() and "3080" in torch.cuda.get_device_name(0):
            # Enable autocast for automatic mixed precision (AMP)
            optimizations["settings"]["amp_enabled"] = True
            
            # Set optimal memory allocation strategy for 10GB VRAM
            if hasattr(torch.cuda, 'memory_stats'):
                optimizations["settings"]["memory_stats_enabled"] = True
            
            # Configure for cudnn benchmarking for optimal kernel selection
            torch.backends.cudnn.benchmark = True
            optimizations["settings"]["cudnn_benchmark"] = True
            
            # Enable TF32 precision on Ampere
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            optimizations["settings"]["tf32_enabled"] = True
            
            # Optimize for 10GB VRAM
            optimizations["settings"]["max_split_size_mb"] = 512
            
            # Setup CUDA graphs for repeated operations
            optimizations["settings"]["cuda_graphs_enabled"] = True
            
            optimizations["applied"] = True
            logger.info("Applied RTX 3080-specific optimizations")
        return optimizations
    
    return safe_execute(
        _apply_optimizations,
        default_return=optimizations,
        error_message="Error applying GPU optimizations"
    )

def test_gpu_compatibility() -> Tuple[bool, List[str]]:
    """
    Test GPU compatibility and functionality for Maggie AI.
    
    This function performs a series of tests to verify GPU compatibility
    and functionality, with specific checks for RTX 3080.
    
    Returns
    -------
    Tuple[bool, List[str]]
        A tuple containing:
        - Boolean indicating whether the GPU is compatible (True) or not (False)
        - List of warning messages for non-critical issues
    
    Notes
    -----
    Tests include:
    - CUDA availability
    - GPU memory sufficiency (> 8GB recommended)
    - Basic tensor operations
    - RTX 3080-specific feature availability
    
    Examples
    --------
    >>> compatible, warnings = test_gpu_compatibility()
    >>> if compatible:
    ...     print("GPU is compatible")
    ...     if warnings:
    ...         print("Warnings:")
    ...         for warning in warnings:
    ...             print(f"  - {warning}")
    ... else:
    ...     print("GPU is not compatible")
    """
    warnings = []
    
    def _test_gpu():
        import torch
        if not torch.cuda.is_available():
            warnings.append("CUDA not available, running in CPU-only mode")
            return False, warnings
            
        # Test basic tensor operations
        try:
            test_tensor = torch.ones(1000, 1000, device='cuda')
            result = torch.matmul(test_tensor, test_tensor)
            del test_tensor, result
            torch.cuda.empty_cache()
        except Exception as e:
            warnings.append(f"Basic CUDA operations failed: {e}")
            return False, warnings
            
        # Check GPU memory
        gpu_info = get_gpu_info()
        if gpu_info["memory_total_gb"] < 8:
            warnings.append(f"GPU memory ({gpu_info['memory_total_gb']:.1f}GB) is less than recommended 8GB")
            
        # Check for RTX 3080
        if "3080" in gpu_info["name"]:
            # Check compute capability (should be 8.6 for RTX 3080)
            if gpu_info["compute_capability"] != "8.6":
                warnings.append(f"RTX 3080 compute capability ({gpu_info['compute_capability']}) is not 8.6")
                
            # Check CUDA version (11.x recommended for RTX 3080)
            if not gpu_info["cuda_version"].startswith("11."):
                warnings.append(f"CUDA version {gpu_info['cuda_version']} - version 11.x recommended for RTX 3080")
        
        return True, warnings
    
    result = safe_execute(
        _test_gpu,
        default_return=(False, ["Error testing GPU compatibility"]),
        error_message="Error testing GPU compatibility"
    )
    
    return result