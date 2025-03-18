"""
Maggie AI Assistant - LLM Processor Module
==========================================
Language Model processing using various backend engines.

This module provides language model inference capabilities for the Maggie AI Assistant
using different backend engines like ctransformers, llama-cpp, etc. It includes
optimizations for AMD Ryzen 9 5900X and NVIDIA RTX 3080 hardware.
"""

# Standard library imports
import os
import time
from typing import Dict, Any, Optional, List, Tuple

# Third-party imports
from loguru import logger

class LLMProcessor:
    """
    Language Model processor using various backends.
    
    This class provides a unified interface to different LLM backends,
    with optimizations for RTX 3080 GPUs when available.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary for LLM settings
        
    Attributes
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    model_path : str
        Path to the model directory
    model_type : str
        Type of the model to use (mistral, llama2, etc.)
    model : Optional[Any]
        Loaded model instance
    gpu_layers : int
        Number of layers to offload to GPU
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM processor with configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary for LLM containing:
            - model_path: Path to LLM models (default: "models/llm")
            - model_type: Type of model (default: "mistral")
            - gpu_layers: Number of layers to offload to GPU (default: 0)
        """
        self.config = config
        self.model_path = config.get("model_path", "models/mistral-7b-instruct-v0.3-GPTQ-4bit")
        self.model_type = config.get("model_type", "mistral")
        self.gpu_layers = config.get("gpu_layers", 0)
        self.model = None
        
        # Initialize model in a lazy manner
        logger.info(f"LLM processor initialized with model: {self.model_type}, path: {self.model_path}")
    
    def _load_model(self) -> bool:
        """
        Load the language model with the configured settings.
        
        Returns
        -------
        bool
            True if model loaded successfully, False otherwise
        """
        if self.model is not None:
            return True
            
        try:
            # Try to import ctransformers for model loading
            from ctransformers import AutoModelForCausalLM
            
            # Set model type based on configuration
            model_type_mapping = {
                "mistral": "mistral",
                "llama2": "llama",
                "phi": "phi"
            }
            
            model_type = model_type_mapping.get(self.model_type, "mistral")
            
            # Load model with GPU acceleration if configured
            logger.info(f"Loading {self.model_type} model with {self.gpu_layers} GPU layers")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_type=model_type,
                gpu_layers=self.gpu_layers,
                threads=8  # Optimize for Ryzen 9 5900X
            )
            
            logger.info(f"LLM model {self.model_type} loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import ctransformers: {e}")
            logger.error("Please install with: pip install ctransformers")
            return False
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            return False
    
    def generate_text(self, prompt: str, max_tokens: int = 512, 
                     temperature: float = 0.7, top_p: float = 0.95) -> str:
        """
        Generate text from a prompt.
        
        This method generates text from the given prompt using the loaded
        language model.
        
        Parameters
        ----------
        prompt : str
            Prompt to generate from
        max_tokens : int, optional
            Maximum number of tokens to generate, by default 512
        temperature : float, optional
            Sampling temperature, by default 0.7
        top_p : float, optional
            Top-p sampling parameter, by default 0.95
            
        Returns
        -------
        str
            Generated text or empty string if failed
        """
        try:
            # Load model if not already loaded
            if not self._load_model():
                return ""
                
            # Measure generation time for performance tracking
            start_time = time.time()
            
            # Generate text with the model
            output = self.model(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1
            )
            
            generation_time = time.time() - start_time
            
            logger.debug(f"Generated {len(output.split())} words in {generation_time:.2f}s")
            
            return output
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""
    
    def unload_model(self) -> bool:
        """
        Unload the model to free memory.
        
        Returns
        -------
        bool
            True if unloaded successfully, False otherwise
        """
        try:
            # Release model
            self.model = None
            
            # Clean up CUDA memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA memory cache cleared")
            except ImportError:
                pass
                
            logger.info("LLM model unloaded")
            return True
                
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False