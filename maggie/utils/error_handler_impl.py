"""
Example integration of error handling and logging systems in LLM processor component.
This shows how to refactor the LLM processor to use the standardized error handling.
"""

# Import required components
from typing import Dict, Any, Optional
from loguru import logger

from maggie.utils.error_handling import (
    safe_execute, retry_operation, ErrorCategory, ErrorSeverity, 
    with_error_handling, ModelLoadError, GenerationError
)
from maggie.utils.logging import ComponentLogger, log_operation, logging_context

class LLMProcessor:
    """
    Refactored LLM processor with standardized error handling and logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM processor with configuration."""
        self.config = config
        self.model_path = config.get('model_path', 'models/mistral-7b-instruct-v0.3-GPTQ-4bit')
        self.model_type = config.get('model_type', 'mistral')
        self.gpu_layers = config.get('gpu_layers', 0)
        self.model = None
        
        # Create component logger for consistent logging
        self.logger = ComponentLogger("LLM")
        self.logger.info(f"Initialized with model: {self.model_type}, path: {self.model_path}")
    
    @retry_operation(max_attempts=2, allowed_exceptions=(OSError, RuntimeError))
    @with_error_handling(error_code="MODEL_LOAD_ERROR", error_category=ErrorCategory.MODEL)
    def _load_model(self) -> bool:
        """
        Load the language model with standardized error handling.
        
        Returns
        -------
        bool
            True if model was loaded successfully
        
        Raises
        ------
        ModelLoadError
            If model loading fails
        """
        if self.model is not None:
            return True
            
        with logging_context(component="LLM", operation="load_model") as ctx:
            try:
                from ctransformers import AutoModelForCausalLM
                
                # Map internal model type to library model type
                model_type_mapping = {
                    'mistral': 'mistral',
                    'llama2': 'llama',
                    'phi': 'phi'
                }
                model_type = model_type_mapping.get(self.model_type, 'mistral')
                
                self.logger.info(f"Loading {self.model_type} model with {self.gpu_layers} GPU layers")
                
                start_time = time.time()
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    model_type=model_type,
                    gpu_layers=self.gpu_layers,
                    threads=8
                )
                load_time = time.time() - start_time
                
                self.logger.log_performance("load_model", load_time, {
                    "model_type": self.model_type,
                    "gpu_layers": self.gpu_layers
                })
                
                self.logger.info(f"Model {self.model_type} loaded successfully in {load_time:.2f}s")
                return True
                
            except ImportError as e:
                # Specific error for missing dependency
                self.logger.error("Failed to import ctransformers", exception=e)
                self.logger.error('Please install with: pip install ctransformers')
                raise ModelLoadError(f"Failed to import ctransformers: {e}") from e
                
            except Exception as e:
                # General model loading error
                self.logger.error(f"Error loading LLM model", exception=e)
                raise ModelLoadError(f"Error loading model {self.model_type}: {e}") from e
    
    @log_operation(component="LLM", log_args=True)
    def generate_text(self, 
                     prompt: str, 
                     max_tokens: int = 512, 
                     temperature: float = 0.7, 
                     top_p: float = 0.95) -> str:
        """
        Generate text using the language model.
        
        Parameters
        ----------
        prompt : str
            Input prompt for text generation
        max_tokens : int
            Maximum number of tokens to generate
        temperature : float
            Sampling temperature (higher = more creative)
        top_p : float
            Nucleus sampling probability threshold
            
        Returns
        -------
        str
            Generated text
            
        Notes
        -----
        Uses standardized error handling and performance logging.
        """
        try:
            # Ensure model is loaded
            if not self._load_model():
                return ''
                
            # Generate text with performance tracking
            with logging_context(component="LLM", operation="generate") as ctx:
                start_time = time.time()
                
                output = self.model(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1
                )
                
                generation_time = time.time() - start_time
                
                # Log performance metrics
                tokens_generated = len(output.split()) - len(prompt.split())
                self.logger.log_performance("generate", generation_time, {
                    "tokens": tokens_generated,
                    "tokens_per_second": tokens_generated / generation_time if generation_time > 0 else 0
                })
                
                self.logger.debug(f"Generated {tokens_generated} tokens in {generation_time:.2f}s")
                return output
                
        except Exception as e:
            # Handle generation errors
            self.logger.error("Error generating text", exception=e)
            
            # Publish specific error event for monitoring
            from maggie.utils.error_handling import get_event_bus, ERROR_EVENT_COMPONENT_FAILURE
            event_bus = get_event_bus()
            if event_bus:
                event_bus.publish(ERROR_EVENT_COMPONENT_FAILURE, {
                    "component": "LLM",
                    "operation": "generate",
                    "error": str(e)
                })
                
            # Return empty result on error
            return ''
    
    def unload_model(self) -> bool:
        """
        Unload the language model to free resources.
        
        Returns
        -------
        bool
            True if model was unloaded successfully
        """
        return safe_execute(
            self._unload_model_impl,
            error_message="Error unloading model",
            error_category=ErrorCategory.RESOURCE,
            default_return=False
        )
    
    def _unload_model_impl(self) -> bool:
        """Implementation of model unloading with resource cleanup."""
        try:
            self.model = None
            
            # Clean up CUDA memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.debug('CUDA memory cache cleared')
            except ImportError:
                pass
                
            self.logger.info('LLM model unloaded')
            return True
            
        except Exception as e:
            self.logger.error(f"Error unloading model", exception=e)
            return False
