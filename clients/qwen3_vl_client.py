# -*- coding: utf-8 -*-
"""
Qwen3-VL Client - Multimodal LLM inference with vLLM
Vision-Language model for image and video understanding
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Configure environment
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    checkpoint_path: str = "Qwen/Qwen2-VL-7B-Instruct"
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = 8192  # Default to 8k
    trust_remote_code: bool = True
    dtype: str = "auto"
    enable_expert_parallel: bool = False
    mm_encoder_tp_mode: str = "data"


class Qwen3VLClient:
    """
    Production-grade Qwen3-VL client with vLLM
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct", 
        gpu_memory_utilization: float = 0.9, 
        tensor_parallel_size: int = 1, 
        max_model_len: int = 8192,  # <--- Added param
        trust_remote_code: bool = True
    ):
        """
        Initialize Qwen3-VL client
        """
        self.model_name = model_name
        
        # Store configuration in dataclass
        self.model_config = ModelConfig(
            checkpoint_path=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,  # <--- Store it here
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code
        )
        
        self.processor = None
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        """Initialize processor and vLLM model"""
        try:
            logger.info(f"Loading Qwen3-VL processor from {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.model_config.trust_remote_code
            )
            
            # Determine tensor parallel size
            tp_size = self.model_config.tensor_parallel_size or torch.cuda.device_count()
            logger.info(f"Initializing Qwen3-VL with tensor_parallel_size={tp_size}, context={self.model_config.max_model_len}")
            
            # Initialize vLLM
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=self.model_config.gpu_memory_utilization,
                trust_remote_code=self.model_config.trust_remote_code,
                dtype=self.model_config.dtype,
                max_model_len=self.model_config.max_model_len, # <--- Passed to vLLM here
                mm_encoder_tp_mode=self.model_config.mm_encoder_tp_mode,
                enable_expert_parallel=self.model_config.enable_expert_parallel,
                seed=42
            )
            
            logger.info("Qwen3-VL model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qwen3-VL: {e}")
            raise
    
    def prepare_inputs(self, messages: List[Dict]) -> Dict[str, Any]:
        """Prepare inputs for vLLM inference"""
        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process vision information (images/videos)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True
            )
            
            # Prepare multimodal data
            mm_data = {}
            if image_inputs is not None:
                mm_data['image'] = image_inputs
            
            if video_inputs is not None:
                mm_data['video'] = video_inputs
            
            return {
                'prompt': text,
                'multi_modal_data': mm_data,
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare inputs: {e}")
            raise
    
    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.05
    ) -> str:
        """Generate response for multimodal input"""
        try:
            # Prepare inputs
            inputs = self.prepare_inputs(messages)
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[]
            )
            
            logger.info(f"Generating response with Qwen3-VL")
            
            # Generate
            outputs = self.llm.generate([inputs], sampling_params=sampling_params)
            
            # Extract text
            generated_text = outputs[0].outputs[0].text
            logger.info(f"Qwen3-VL response: {len(generated_text)} chars")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in Qwen3-VL generation: {e}")
            raise
    
    def batch_generate(
        self,
        messages_batch: List[List[Dict]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.05
    ) -> List[str]:
        """Generate responses for a batch of messages"""
        try:
            inputs = [self.prepare_inputs(msgs) for msgs in messages_batch]
            
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[]
            )
            
            logger.info(f"Batch generating responses for {len(inputs)} inputs")
            outputs = self.llm.generate(inputs, sampling_params=sampling_params)
            
            results = [output.outputs[0].text for output in outputs]
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise
    
    def analyze_image(
        self,
        image_url: str,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> str:
        """Analyze an image and answer a question about it"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": question}
                ]
            }
        ]
        return self.generate(messages, temperature=temperature, max_tokens=max_tokens)

    def is_available(self) -> bool:
        return self.llm is not None