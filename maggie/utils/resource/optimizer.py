"""
Maggie AI Assistant - Hardware Optimization Module
===============================================

Hardware-specific optimization functionality for the Maggie AI Assistant.

This module provides optimization profiles and settings based on detected
hardware, with particular focus on AMD Ryzen 9 5900X and NVIDIA RTX 3080
hardware configurations.
"""

import os
import platform
from typing import Dict, Any, Optional

from loguru import logger


class HardwareOptimizer:
    """
    Hardware optimization based on detected specifications.
    
    This class creates optimization profiles for CPU, memory, and GPU
    configurations, with specialized optimizations for AMD Ryzen 9 5900X
    and NVIDIA RTX 3080 hardware.
    
    Parameters
    ----------
    hardware_info : Dict[str, Any]
        Detected hardware information
    config : Dict[str, Any]
        Configuration dictionary with optimization settings
        
    Attributes
    ----------
    hardware_info : Dict[str, Any]
        Detected hardware information
    config : Dict[str, Any]
        Application configuration
    """
    
    def __init__(self, hardware_info: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the hardware optimizer.
        
        Parameters
        ----------
        hardware_info : Dict[str, Any]
            Detected hardware information
        config : Dict[str, Any]
            Configuration dictionary with optimization settings
        """
        self.hardware_info = hardware_info
        self.config = config
    
    def create_optimization_profile(self) -> Dict[str, Any]:
        """
        Create comprehensive optimization profile for detected hardware.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with optimization settings for threading, memory, GPU, etc.
        """
        profile = {
            'threading': self._optimize_threading(),
            'memory': self._optimize_memory(),
            'gpu': self._optimize_gpu(),
            'llm': self._optimize_llm(),
            'audio': self._optimize_audio()
        }
        return profile
    
    def _optimize_threading(self) -> Dict[str, Any]:
        """
        Optimize threading settings based on CPU capabilities.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with threading optimization settings
        """
        cpu_info = self.hardware_info['cpu']
        if cpu_info.get('is_ryzen_9_5900x', False):
            return {
                'max_workers': 10,
                'thread_timeout': 30,
                'worker_affinity': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                'priority_boost': True
            }
        else:
            physical_cores = cpu_info.get('physical_cores', 4)
            max_workers = max(2, min(physical_cores - 2, physical_cores * 4 // 5))
            return {
                'max_workers': max_workers,
                'thread_timeout': 30
            }
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """
        Optimize memory settings based on available RAM.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with memory optimization settings
        """
        memory_info = self.hardware_info['memory']
        
        # Ideal case: 32GB DDR4-3200 RAM
        if memory_info.get('is_32gb', False) and memory_info.get('type') == 'DDR4' and memory_info.get('speed') == '3200MHz':
            return {
                'max_percent': 80,
                'unload_threshold': 85,
                'cache_size_mb': 6144,
                'preloading': True,
                'large_pages': True,
                'numa_aware': True
            }
        # Good case: 32GB RAM of unknown type
        elif memory_info.get('is_32gb', False):
            return {
                'max_percent': 75,
                'unload_threshold': 85,
                'cache_size_mb': 4096,
                'preloading': True
            }
        # General case: Scale based on available RAM
        else:
            total_gb = memory_info.get('total_gb', 8)
            return {
                'max_percent': min(70, max(50, int(60 + (total_gb - 8) * 1.25))),
                'unload_threshold': min(80, max(60, int(70 + (total_gb - 8) * 1.25))),
                'cache_size_mb': min(2048, max(512, int(total_gb * 64)))
            }
    
    def _optimize_gpu(self) -> Dict[str, Any]:
        """
        Optimize GPU settings based on detected graphics card.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with GPU optimization settings
        """
        gpu_info = self.hardware_info['gpu']
        
        # No GPU available
        if not gpu_info.get('available', False):
            return {'enabled': False}
        
        # RTX 3080 specific optimizations
        if gpu_info.get('is_rtx_3080', False):
            return {
                'enabled': True,
                'compute_type': 'float16',
                'tensor_cores': True,
                'cuda_streams': 3,
                'reserved_memory_mb': 256,
                'max_batch_size': 16,
                'memory_fraction': 0.95,
                'cudnn_benchmark': True,
                'trt_optimization': True,
                'bfloat16_supported': False,
                'dynamic_mem_mgmt': True,
                'vram_gb': gpu_info.get('memory_gb', 10),
                'vram_efficient_loading': True,
                'amp_optimization_level': 'O2'
            }
        # Other GPU optimizations scaled by VRAM size
        else:
            memory_gb = gpu_info.get('memory_gb', 0)
            return {
                'enabled': True,
                'compute_type': 'float16' if memory_gb >= 6 else 'int8',
                'tensor_cores': 'tensor_cores' in gpu_info.get('name', '').lower(),
                'cuda_streams': 1,
                'reserved_memory_mb': min(512, max(128, int(memory_gb * 64))),
                'max_batch_size': min(16, max(4, int(memory_gb / 2)))
            }
    
    def _optimize_llm(self) -> Dict[str, Any]:
        """
        Optimize language model settings based on available hardware.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with LLM optimization settings
        """
        gpu_info = self.hardware_info['gpu']
        
        # RTX 3080 optimizations
        if gpu_info.get('is_rtx_3080', False):
            return {
                'gpu_layers': 32,
                'precision': 'float16',
                'kv_cache_optimization': True,
                'context_length': 8192,
                'attention_sinks': True,
                'auto_adjust': True,
                'tensor_parallel': 1,
                'draft_model': 'small',
                'speculative_decoding': True,
                'llm_rope_scaling': 'dynamic',
                'batch_inference': True,
                'offload_layers': {
                    'enabled': True,
                    'threshold_gb': 9.0,
                    'cpu_layers': [0, 1]
                }
            }
        # Other GPU optimizations scaled by VRAM
        elif gpu_info.get('available', False):
            memory_gb = gpu_info.get('memory_gb', 0)
            return {
                'gpu_layers': min(40, max(1, int(memory_gb * 3.2))),
                'precision': 'float16' if memory_gb >= 6 else 'int8',
                'context_length': min(8192, max(2048, int(memory_gb * 819.2))),
                'auto_adjust': True
            }
        # CPU-only settings
        else:
            return {
                'gpu_layers': 0,
                'precision': 'int8',
                'context_length': 2048,
                'auto_adjust': False
            }
    
    def _optimize_audio(self) -> Dict[str, Any]:
        """
        Optimize audio processing settings based on hardware.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with audio optimization settings
        """
        cpu_info = self.hardware_info['cpu']
        gpu_info = self.hardware_info['gpu']
        
        # Base audio settings
        audio_opt = {
            'sample_rate': 22050,
            'chunk_size': 1024,
            'use_gpu': gpu_info.get('available', False),
            'buffer_size': 4096
        }
        
        # CPU-specific optimizations
        if cpu_info.get('is_ryzen_9_5900x', False):
            audio_opt['chunk_size'] = 512
            audio_opt['audio_threads'] = 2
        
        # GPU-specific optimizations
        if gpu_info.get('is_rtx_3080', False):
            audio_opt['whisper_model'] = 'small'
            audio_opt['whisper_compute'] = 'float16'
            audio_opt['cache_models'] = True
            audio_opt['vad_sensitivity'] = 0.6
            audio_opt['cuda_audio_processing'] = True
        elif gpu_info.get('available', False):
            audio_opt['whisper_model'] = 'base'
            audio_opt['whisper_compute'] = 'float16' if gpu_info.get('memory_gb', 0) >= 6 else 'int8'
        else:
            audio_opt['whisper_model'] = 'tiny'
            audio_opt['whisper_compute'] = 'int8'
        
        return audio_opt
    
    def setup_gpu(self) -> None:
        """
        Set up GPU with optimized settings for detected hardware.
        
        Configures PyTorch and CUDA settings for optimal performance,
        particularly for RTX 3080 GPUs.
        """
        gpu_info = self.hardware_info['gpu']
        
        if not gpu_info.get('available', False):
            logger.info("No GPU available for setup")
            return
        
        try:
            import torch
            if torch.cuda.is_available():
                # Set primary device
                torch.cuda.set_device(0)
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # RTX 3080 specific optimizations
                if gpu_info.get('is_rtx_3080', False):
                    # Enable TF32 precision for matrix multiplications
                    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                        torch.backends.cuda.matmul.allow_tf32 = True
                        logger.info('Enabled TF32 precision for matrix multiplications')
                    
                    # Enable cuDNN benchmark and TF32 precision
                    if hasattr(torch.backends, 'cudnn'):
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cudnn.allow_tf32 = True
                        logger.info('Enabled cuDNN benchmark mode and TF32 precision')
                    
                    # Log GPU setup
                    logger.info(f"Configured GPU optimizations for {gpu_info['name']}")
                else:
                    # Log GPU setup
                    device_name = torch.cuda.get_device_name(0)
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"GPU configured: {device_name} ({memory_gb:.2f} GB VRAM)")
                    
                    # Set appropriate GPU optimizations
                    if hasattr(torch.backends, 'cudnn'):
                        torch.backends.cudnn.benchmark = True
                        logger.info('Enabled cuDNN benchmark mode')
                    
        except ImportError:
            logger.debug('PyTorch not available for GPU setup')
        except Exception as e:
            logger.error(f"Error setting up GPU: {e}")
    
    def test_gpu_compatibility(self) -> Dict[str, Any]:
        """
        Test GPU compatibility with required operations.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with test results and warnings
        """
        gpu_info = self.hardware_info['gpu']
        result = {
            'success': False,
            'warnings': [],
            'errors': [],
            'tests_passed': []
        }
        
        if not gpu_info.get('available', False):
            result['warnings'].append('No GPU detected, running in CPU-only mode')
            return result
        
        try:
            import torch
            if not torch.cuda.is_available():
                result['warnings'].append('PyTorch reports CUDA not available')
                return result
            
            # Test basic CUDA operations
            try:
                test_tensor = torch.ones(1000, 1000, device='cuda')
                test_result = torch.matmul(test_tensor, test_tensor)
                del test_tensor, test_result
                torch.cuda.empty_cache()
                result['tests_passed'].append('basic_cuda_operations')
            except Exception as e:
                result['errors'].append(f"Basic CUDA operations failed: {e}")
                return result
            
            # Check GPU specifications meet requirements
            memory_gb = gpu_info.get('memory_gb', 0)
            
            if memory_gb < 8:
                result['warnings'].append(f"GPU memory ({memory_gb:.1f}GB) is less than recommended 8GB")
            
            # Check RTX 3080 specific requirements
            if gpu_info.get('is_rtx_3080', False):
                result['tests_passed'].append('rtx_3080_detected')
                
                if memory_gb < 9.5:
                    result['warnings'].append(f"RTX 3080 VRAM ({memory_gb:.1f}GB) is less than expected 10GB")
                
                if gpu_info.get('compute_capability') != '8.6':
                    result['warnings'].append(f"RTX 3080 compute capability ({gpu_info.get('compute_capability')}) is not 8.6")
                
                cuda_version = torch.version.cuda
                if not cuda_version.startswith('11.'):
                    result['warnings'].append(f"CUDA version {cuda_version} - version 11.x recommended for RTX 3080")
            
            result['success'] = True
            return result
            
        except ImportError as e:
            result['errors'].append(f"PyTorch not available: {e}")
            result['warnings'].append('Install PyTorch with: pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118')
            return result
        except Exception as e:
            result['errors'].append(f"Error testing GPU compatibility: {e}")
            return result