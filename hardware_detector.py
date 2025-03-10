<<<<<<< HEAD
"""
Maggie AI Assistant - Hardware Detection and Optimization
=====================================================
Detects system hardware capabilities and optimizes configuration.
"""

import os
import platform
import subprocess
import psutil
from typing import Dict, Any, Tuple, List, Optional
import yaml
from loguru import logger


class HardwareDetector:
    """
    Detect and report system hardware capabilities.
    """
    
    def __init__(self):
        """Initialize the hardware detector."""
        self.cpu_info = self._detect_cpu()
        self.memory_info = self._detect_memory()
        self.gpu_info = self._detect_gpu()
        
    def _detect_cpu(self) -> Dict[str, Any]:
        """
        Detect CPU information.
        
        Returns
        -------
        Dict[str, Any]
            CPU information including cores, threads, and model
        """
        info = {
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "model": platform.processor()
        }
        
        # Try to get more detailed CPU info on Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            info["model"] = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass
                
        return info
        
    def _detect_memory(self) -> Dict[str, Any]:
        """
        Detect system memory information.
        
        Returns
        -------
        Dict[str, Any]
            Memory information including total, available, and percent used
        """
        vm = psutil.virtual_memory()
        return {
            "total_bytes": vm.total,
            "total_gb": vm.total / (1024**3),
            "available_bytes": vm.available,
            "available_gb": vm.available / (1024**3),
            "percent_used": vm.percent
        }
        
    def _detect_gpu(self) -> Dict[str, Any]:
        """
        Detect GPU information.
        
        Returns
        -------
        Dict[str, Any]
            GPU information including name, memory, and CUDA version
        """
        info = {
            "available": False,
            "name": None,
            "memory_gb": None,
            "cuda_version": None,
            "count": 0
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                info["available"] = True
                info["count"] = torch.cuda.device_count()
                info["name"] = torch.cuda.get_device_name(0)
                info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info["cuda_version"] = torch.version.cuda
        except ImportError:
            pass
            
        return info
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of system hardware information.
        
        Returns
        -------
        Dict[str, Any]
            Summary of system hardware information
        """
        return {
            "system": platform.system(),
            "release": platform.release(),
            "cpu": self.cpu_info,
            "memory": self.memory_info,
            "gpu": self.gpu_info
        }
        
    def print_summary(self) -> None:
        """Print a summary of system hardware information."""
        summary = self.get_summary()
        
        print("\n=== Hardware Information ===")
        print(f"System: {summary['system']} {summary['release']}")
        
        cpu = summary["cpu"]
        print(f"\nCPU: {cpu.get('model', 'Unknown')}")
        print(f"  Physical cores: {cpu.get('cores_physical', 'Unknown')}")
        print(f"  Logical cores: {cpu.get('cores_logical', 'Unknown')}")
        if cpu.get('frequency'):
            print(f"  Frequency: {cpu.get('frequency') / 1000:.2f} GHz")
        
        mem = summary["memory"]
        print(f"\nMemory:")
        print(f"  Total: {mem.get('total_gb', 0):.2f} GB")
        print(f"  Available: {mem.get('available_gb', 0):.2f} GB")
        print(f"  Used: {mem.get('percent_used', 0):.1f}%")
        
        gpu = summary["gpu"]
        print(f"\nGPU:")
        if gpu.get("available"):
            print(f"  Name: {gpu.get('name', 'Unknown')}")
            print(f"  Memory: {gpu.get('memory_gb', 0):.2f} GB")
            print(f"  CUDA Version: {gpu.get('cuda_version', 'Unknown')}")
            print(f"  Device Count: {gpu.get('count', 0)}")
        else:
            print("  No CUDA-capable GPU detected")


class ConfigOptimizer:
    """
    Optimize configuration based on detected hardware.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration optimizer.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        """
        self.config_path = config_path
        self.hardware = HardwareDetector()
        self.original_config = self._load_config()
        self.optimized_config = self._optimize_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns
        -------
        Dict[str, Any]
            Loaded configuration or empty dict if file not found
        """
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found: {self.config_path}")
            return {}
            
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
            
    def _optimize_config(self) -> Dict[str, Any]:
        """
        Optimize configuration based on hardware.
        
        Returns
        -------
        Dict[str, Any]
            Optimized configuration
        """
        config = self.original_config.copy()
        hardware = self.hardware.get_summary()
        
        # Optimize threading configuration
        cpu_cores = hardware["cpu"].get("cores_physical", 4)
        if cpu_cores > 0:
            if "threading" not in config:
                config["threading"] = {}
            # Use 75% of physical cores, at least 2
            config["threading"]["max_workers"] = max(2, int(cpu_cores * 0.75))
        
        # Optimize memory configuration
        mem_gb = hardware["memory"].get("total_gb", 16)
        if mem_gb > 0:
            if "memory" not in config:
                config["memory"] = {}
            if mem_gb >= 32:
                # For 32GB or more, use 75%
                config["memory"]["max_percent"] = 75
            elif mem_gb >= 16:
                # For 16-32GB, use 70%
                config["memory"]["max_percent"] = 70
            else:
                # For less than 16GB, use 60%
                config["memory"]["max_percent"] = 60
                
            # Add model unload threshold
            config["memory"]["model_unload_threshold"] = min(95, config["memory"]["max_percent"] + 10)
        
        # Optimize LLM configuration
        gpu = hardware["gpu"]
        if gpu.get("available"):
            if "llm" not in config:
                config["llm"] = {}
            
            gpu_mem_gb = gpu.get("memory_gb", 0)
            
            if "model_type" not in config["llm"]:
                config["llm"]["model_type"] = "mistral"
                
            # Optimize GPU layers based on available memory
            if gpu_mem_gb >= 10:  # RTX 3080 or better
                config["llm"]["gpu_layers"] = 32
                if gpu_mem_gb >= 24:  # RTX 4090, etc.
                    config["llm"]["gpu_layers"] = 40
            elif gpu_mem_gb >= 8:  # RTX 3070, etc.
                config["llm"]["gpu_layers"] = 24
            elif gpu_mem_gb >= 6:  # RTX 2060, etc.
                config["llm"]["gpu_layers"] = 16
            else:  # Older GPUs
                config["llm"]["gpu_layers"] = 8
                
            # Enable auto-adjustment
            config["llm"]["gpu_layer_auto_adjust"] = True
            
            # Set compute type for whisper
            if "speech" not in config:
                config["speech"] = {}
            if "whisper" not in config["speech"]:
                config["speech"]["whisper"] = {}
                
            config["speech"]["whisper"]["compute_type"] = "float16"
        else:
            # CPU-only configuration
            if "llm" not in config:
                config["llm"] = {}
            config["llm"]["gpu_layers"] = 0
            
            if "speech" not in config:
                config["speech"] = {}
            if "whisper" not in config["speech"]:
                config["speech"]["whisper"] = {}
                
            config["speech"]["whisper"]["compute_type"] = "int8"
            config["speech"]["whisper"]["model_size"] = "tiny"  # Use smallest model for CPU
        
        return config
        
    def save_optimized_config(self, output_path: Optional[str] = None) -> bool:
        """
        Save optimized configuration to file.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save optimized configuration, defaults to original path
            
        Returns
        -------
        bool
            True if saved successfully, False otherwise
        """
        if output_path is None:
            output_path = self.config_path
            
        try:
            with open(output_path, "w") as f:
                yaml.dump(self.optimized_config, f, default_flow_style=False)
            logger.info(f"Saved optimized configuration to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving optimized configuration: {e}")
            return False
            
    def print_optimization_summary(self) -> None:
        """Print a summary of configuration optimizations."""
        print("\n=== Configuration Optimization Summary ===")
        
        # Check threading settings
        orig_workers = self.original_config.get("threading", {}).get("max_workers", "Not set")
        opt_workers = self.optimized_config.get("threading", {}).get("max_workers", "Not set")
        print(f"Threading max_workers: {orig_workers} -> {opt_workers}")
        
        # Check memory settings
        orig_mem = self.original_config.get("memory", {}).get("max_percent", "Not set")
        opt_mem = self.optimized_config.get("memory", {}).get("max_percent", "Not set")
        print(f"Memory max_percent: {orig_mem} -> {opt_mem}")
        
        # Check LLM settings
        orig_layers = self.original_config.get("llm", {}).get("gpu_layers", "Not set")
        opt_layers = self.optimized_config.get("llm", {}).get("gpu_layers", "Not set")
        print(f"LLM gpu_layers: {orig_layers} -> {opt_layers}")
        
        # Check Whisper settings
        orig_compute = self.original_config.get("speech", {}).get("whisper", {}).get("compute_type", "Not set")
        opt_compute = self.optimized_config.get("speech", {}).get("whisper", {}).get("compute_type", "Not set")
        print(f"Whisper compute_type: {orig_compute} -> {opt_compute}")
        
        # Check auto-adjust setting
        orig_adjust = self.original_config.get("llm", {}).get("gpu_layer_auto_adjust", "Not set")
        opt_adjust = self.optimized_config.get("llm", {}).get("gpu_layer_auto_adjust", "Not set")
        print(f"LLM gpu_layer_auto_adjust: {orig_adjust} -> {opt_adjust}")
        
        print("\nUse this optimized configuration? (y/n): ", end="")


if __name__ == "__main__":
    detector = HardwareDetector()
    detector.print_summary()
    
    optimizer = ConfigOptimizer()
    optimizer.print_optimization_summary()
    
    response = input().strip().lower()
    if response == "y":
        optimizer.save_optimized_config()
        print("Configuration optimized and saved.")
    else:
=======
"""
Maggie AI Assistant - Hardware Detection and Optimization
=====================================================
Detects system hardware capabilities and optimizes configuration.
"""

import os
import platform
import subprocess
import psutil
from typing import Dict, Any, Tuple, List, Optional
import yaml
from loguru import logger


class HardwareDetector:
    """
    Detect and report system hardware capabilities.
    """
    
    def __init__(self):
        """Initialize the hardware detector."""
        self.cpu_info = self._detect_cpu()
        self.memory_info = self._detect_memory()
        self.gpu_info = self._detect_gpu()
        
    def _detect_cpu(self) -> Dict[str, Any]:
        """
        Detect CPU information.
        
        Returns
        -------
        Dict[str, Any]
            CPU information including cores, threads, and model
        """
        info = {
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "model": platform.processor()
        }
        
        # Try to get more detailed CPU info on Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            info["model"] = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass
                
        return info
        
    def _detect_memory(self) -> Dict[str, Any]:
        """
        Detect system memory information.
        
        Returns
        -------
        Dict[str, Any]
            Memory information including total, available, and percent used
        """
        vm = psutil.virtual_memory()
        return {
            "total_bytes": vm.total,
            "total_gb": vm.total / (1024**3),
            "available_bytes": vm.available,
            "available_gb": vm.available / (1024**3),
            "percent_used": vm.percent
        }
        
    def _detect_gpu(self) -> Dict[str, Any]:
        """
        Detect GPU information.
        
        Returns
        -------
        Dict[str, Any]
            GPU information including name, memory, and CUDA version
        """
        info = {
            "available": False,
            "name": None,
            "memory_gb": None,
            "cuda_version": None,
            "count": 0
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                info["available"] = True
                info["count"] = torch.cuda.device_count()
                info["name"] = torch.cuda.get_device_name(0)
                info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info["cuda_version"] = torch.version.cuda
        except ImportError:
            pass
            
        return info
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of system hardware information.
        
        Returns
        -------
        Dict[str, Any]
            Summary of system hardware information
        """
        return {
            "system": platform.system(),
            "release": platform.release(),
            "cpu": self.cpu_info,
            "memory": self.memory_info,
            "gpu": self.gpu_info
        }
        
    def print_summary(self) -> None:
        """Print a summary of system hardware information."""
        summary = self.get_summary()
        
        print("\n=== Hardware Information ===")
        print(f"System: {summary['system']} {summary['release']}")
        
        cpu = summary["cpu"]
        print(f"\nCPU: {cpu.get('model', 'Unknown')}")
        print(f"  Physical cores: {cpu.get('cores_physical', 'Unknown')}")
        print(f"  Logical cores: {cpu.get('cores_logical', 'Unknown')}")
        if cpu.get('frequency'):
            print(f"  Frequency: {cpu.get('frequency') / 1000:.2f} GHz")
        
        mem = summary["memory"]
        print(f"\nMemory:")
        print(f"  Total: {mem.get('total_gb', 0):.2f} GB")
        print(f"  Available: {mem.get('available_gb', 0):.2f} GB")
        print(f"  Used: {mem.get('percent_used', 0):.1f}%")
        
        gpu = summary["gpu"]
        print(f"\nGPU:")
        if gpu.get("available"):
            print(f"  Name: {gpu.get('name', 'Unknown')}")
            print(f"  Memory: {gpu.get('memory_gb', 0):.2f} GB")
            print(f"  CUDA Version: {gpu.get('cuda_version', 'Unknown')}")
            print(f"  Device Count: {gpu.get('count', 0)}")
        else:
            print("  No CUDA-capable GPU detected")


class ConfigOptimizer:
    """
    Optimize configuration based on detected hardware.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration optimizer.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        """
        self.config_path = config_path
        self.hardware = HardwareDetector()
        self.original_config = self._load_config()
        self.optimized_config = self._optimize_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns
        -------
        Dict[str, Any]
            Loaded configuration or empty dict if file not found
        """
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found: {self.config_path}")
            return {}
            
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
            
    def _optimize_config(self) -> Dict[str, Any]:
        """
        Optimize configuration based on hardware.
        
        Returns
        -------
        Dict[str, Any]
            Optimized configuration
        """
        config = self.original_config.copy()
        hardware = self.hardware.get_summary()
        
        # Optimize threading configuration
        cpu_cores = hardware["cpu"].get("cores_physical", 4)
        if cpu_cores > 0:
            if "threading" not in config:
                config["threading"] = {}
            # Use 75% of physical cores, at least 2
            config["threading"]["max_workers"] = max(2, int(cpu_cores * 0.75))
        
        # Optimize memory configuration
        mem_gb = hardware["memory"].get("total_gb", 16)
        if mem_gb > 0:
            if "memory" not in config:
                config["memory"] = {}
            if mem_gb >= 32:
                # For 32GB or more, use 75%
                config["memory"]["max_percent"] = 75
            elif mem_gb >= 16:
                # For 16-32GB, use 70%
                config["memory"]["max_percent"] = 70
            else:
                # For less than 16GB, use 60%
                config["memory"]["max_percent"] = 60
                
            # Add model unload threshold
            config["memory"]["model_unload_threshold"] = min(95, config["memory"]["max_percent"] + 10)
        
        # Optimize LLM configuration
        gpu = hardware["gpu"]
        if gpu.get("available"):
            if "llm" not in config:
                config["llm"] = {}
            
            gpu_mem_gb = gpu.get("memory_gb", 0)
            
            if "model_type" not in config["llm"]:
                config["llm"]["model_type"] = "mistral"
                
            # Optimize GPU layers based on available memory
            if gpu_mem_gb >= 10:  # RTX 3080 or better
                config["llm"]["gpu_layers"] = 32
                if gpu_mem_gb >= 24:  # RTX 4090, etc.
                    config["llm"]["gpu_layers"] = 40
            elif gpu_mem_gb >= 8:  # RTX 3070, etc.
                config["llm"]["gpu_layers"] = 24
            elif gpu_mem_gb >= 6:  # RTX 2060, etc.
                config["llm"]["gpu_layers"] = 16
            else:  # Older GPUs
                config["llm"]["gpu_layers"] = 8
                
            # Enable auto-adjustment
            config["llm"]["gpu_layer_auto_adjust"] = True
            
            # Set compute type for whisper
            if "speech" not in config:
                config["speech"] = {}
            if "whisper" not in config["speech"]:
                config["speech"]["whisper"] = {}
                
            config["speech"]["whisper"]["compute_type"] = "float16"
        else:
            # CPU-only configuration
            if "llm" not in config:
                config["llm"] = {}
            config["llm"]["gpu_layers"] = 0
            
            if "speech" not in config:
                config["speech"] = {}
            if "whisper" not in config["speech"]:
                config["speech"]["whisper"] = {}
                
            config["speech"]["whisper"]["compute_type"] = "int8"
            config["speech"]["whisper"]["model_size"] = "tiny"  # Use smallest model for CPU
        
        return config
        
    def save_optimized_config(self, output_path: Optional[str] = None) -> bool:
        """
        Save optimized configuration to file.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save optimized configuration, defaults to original path
            
        Returns
        -------
        bool
            True if saved successfully, False otherwise
        """
        if output_path is None:
            output_path = self.config_path
            
        try:
            with open(output_path, "w") as f:
                yaml.dump(self.optimized_config, f, default_flow_style=False)
            logger.info(f"Saved optimized configuration to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving optimized configuration: {e}")
            return False
            
    def print_optimization_summary(self) -> None:
        """Print a summary of configuration optimizations."""
        print("\n=== Configuration Optimization Summary ===")
        
        # Check threading settings
        orig_workers = self.original_config.get("threading", {}).get("max_workers", "Not set")
        opt_workers = self.optimized_config.get("threading", {}).get("max_workers", "Not set")
        print(f"Threading max_workers: {orig_workers} -> {opt_workers}")
        
        # Check memory settings
        orig_mem = self.original_config.get("memory", {}).get("max_percent", "Not set")
        opt_mem = self.optimized_config.get("memory", {}).get("max_percent", "Not set")
        print(f"Memory max_percent: {orig_mem} -> {opt_mem}")
        
        # Check LLM settings
        orig_layers = self.original_config.get("llm", {}).get("gpu_layers", "Not set")
        opt_layers = self.optimized_config.get("llm", {}).get("gpu_layers", "Not set")
        print(f"LLM gpu_layers: {orig_layers} -> {opt_layers}")
        
        # Check Whisper settings
        orig_compute = self.original_config.get("speech", {}).get("whisper", {}).get("compute_type", "Not set")
        opt_compute = self.optimized_config.get("speech", {}).get("whisper", {}).get("compute_type", "Not set")
        print(f"Whisper compute_type: {orig_compute} -> {opt_compute}")
        
        # Check auto-adjust setting
        orig_adjust = self.original_config.get("llm", {}).get("gpu_layer_auto_adjust", "Not set")
        opt_adjust = self.optimized_config.get("llm", {}).get("gpu_layer_auto_adjust", "Not set")
        print(f"LLM gpu_layer_auto_adjust: {orig_adjust} -> {opt_adjust}")
        
        print("\nUse this optimized configuration? (y/n): ", end="")


if __name__ == "__main__":
    detector = HardwareDetector()
    detector.print_summary()
    
    optimizer = ConfigOptimizer()
    optimizer.print_optimization_summary()
    
    response = input().strip().lower()
    if response == "y":
        optimizer.save_optimized_config()
        print("Configuration optimized and saved.")
    else:
>>>>>>> 6062514b96de23fbf6dcdbfd4420d6e2f22903ff
        print("Configuration optimization skipped.")