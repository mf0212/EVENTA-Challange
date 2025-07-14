"""
Qwen model implementations for Event-Enriched Image Captioning
=============================================================

This module provides wrapper classes for Qwen-VL and Qwen text models.
"""

import torch
import yaml
from typing import Optional, Dict, Any, List
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM, 
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info
from ..utils.logging_utils import get_logger, LoggerMixin

logger = get_logger(__name__)


class QwenVLModel(LoggerMixin):
    """
    Wrapper class for Qwen-VL model for visual-language tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Qwen-VL model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.processor = None
        self.device = None
        
    def load_model(self, model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Load the Qwen-VL model and processor.
        
        Args:
            model_config: Model configuration override
        """
        config = model_config or self.config.get('qwen_vl', {})
        
        model_id = config.get('model_id', 'Qwen/Qwen2.5-VL-7B-Instruct')
        torch_dtype = getattr(torch, config.get('torch_dtype', 'bfloat16'))
        device_map = config.get('device_map', 'auto')
        min_pixels = config.get('min_pixels', 200704)
        max_pixels = config.get('max_pixels', 1003520)
        
        self.logger.info(f"Loading Qwen-VL model: {model_id}")
        
        try:
            # Load model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
            
            self.device = next(self.model.parameters()).device
            self.logger.info(f"✅ Qwen-VL model loaded on device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Qwen-VL model: {str(e)}")
            raise
    
    def generate_caption(
        self, 
        image_path: str, 
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate caption for an image using Qwen-VL.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for caption generation
            generation_config: Generation parameters override
            
        Returns:
            Generated caption text
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        gen_config = generation_config or self.config.get('generation', {})
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        try:
            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_config.get('max_new_tokens', 2048),
                    temperature=gen_config.get('temperature', 0.7),
                    do_sample=gen_config.get('do_sample', True),
                    top_p=gen_config.get('top_p', 0.9),
                    top_k=gen_config.get('top_k', 50),
                    repetition_penalty=gen_config.get('repetition_penalty', 1.1)
                )
            
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0].strip()
            
        except Exception as e:
            self.logger.error(f"Error generating caption: {str(e)}")
            return f"Error: {str(e)}"


class QwenTextModel(LoggerMixin):
    """
    Wrapper class for Qwen text model for text-only tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Qwen text model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def load_model(self, model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Load the Qwen text model and tokenizer.
        
        Args:
            model_config: Model configuration override
        """
        config = model_config or self.config.get('qwen_text', {})
        
        model_id = config.get('model_id', 'Qwen/Qwen2.5-7B-Instruct')
        torch_dtype = getattr(torch, config.get('torch_dtype', 'bfloat16'))
        device_map = config.get('device_map', 'auto')
        
        self.logger.info(f"Loading Qwen text model: {model_id}")
        
        try:
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            self.device = next(self.model.parameters()).device
            self.logger.info(f"✅ Qwen text model loaded on device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Qwen text model: {str(e)}")
            raise
    
    def generate_text(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text using Qwen text model.
        
        Args:
            prompt: Input prompt
            system_message: System message (optional)
            generation_config: Generation parameters override
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        gen_config = generation_config or self.config.get('generation', {})
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=gen_config.get('max_new_tokens', 2048),
                    temperature=gen_config.get('temperature', 0.7),
                    do_sample=gen_config.get('do_sample', True),
                    top_p=gen_config.get('top_p', 0.9),
                    top_k=gen_config.get('top_k', 50),
                    repetition_penalty=gen_config.get('repetition_penalty', 1.1),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract generated part
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Decode
            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            return f"Error: {str(e)}"


def load_models_from_config(config_path: str) -> Dict[str, Any]:
    """
    Load models from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing loaded models
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    models = {}
    
    # Load Qwen-VL model if configured
    if 'qwen_vl' in config:
        qwen_vl = QwenVLModel(config)
        qwen_vl.load_model()
        models['qwen_vl'] = qwen_vl
    
    # Load Qwen text model if configured
    if 'qwen_text' in config:
        qwen_text = QwenTextModel(config)
        qwen_text.load_model()
        models['qwen_text'] = qwen_text
    
    return models