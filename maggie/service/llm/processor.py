import os, time
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

class LLMProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get('model_path', 'models/mistral-7b-instruct-v0.3-GPTQ-4bit')
        self.model_type = config.get('model_type', 'mistral')
        self.gpu_layers = config.get('gpu_layers', 0)
        self.model = None
        logger.info(f"LLM processor initialized with model: {self.model_type}, path: {self.model_path}")
    
    def _load_model(self) -> bool:
        if self.model is not None:
            return True
        try:
            from ctransformers import AutoModelForCausalLM
            model_type_mapping = {
                'mistral': 'mistral',
                'llama2': 'llama',
                'phi': 'phi'
            }
            model_type = model_type_mapping.get(self.model_type, 'mistral')
            logger.info(f"Loading {self.model_type} model with {self.gpu_layers} GPU layers")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                model_type=model_type,
                gpu_layers=self.gpu_layers,
                threads=8
            )
            logger.info(f"LLM model {self.model_type} loaded successfully")
            return True
        except ImportError as e:
            logger.error(f"Failed to import ctransformers: {e}")
            logger.error('Please install with: pip install ctransformers')
            return False
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            return False
    
    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = .7, top_p: float = .95) -> str:
        try:
            if not self._load_model():
                return ''
            start_time = time.time()
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
            return ''
    
    def unload_model(self) -> bool:
        try:
            self.model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug('CUDA memory cache cleared')
            except ImportError:
                pass
            logger.info('LLM model unloaded')
            return True
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False