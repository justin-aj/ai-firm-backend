"""
vLLM Client - Production-grade LLM inference with Speculative Decoding
Uses vLLM direct API with speculative decoding for faster inference
"""

from typing import List, Dict, Optional
import logging
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class VLLMClient:
    """
    Client for vLLM direct API with Speculative Decoding
    
    Speculative decoding uses a small draft model (TinyLlama) to predict tokens,
    then validates with the main model (Llama-3-70B). This gives you:
    - 2-3x faster inference than standard decoding
    - Same quality as the main model (no accuracy loss)
    - Lower latency per token
    
    vLLM provides:
    - Continuous batching
    - Paged attention (PagedAttention)
    - Efficient KV cache management
    - High throughput with low latency
    - Direct GPU access (no HTTP overhead)
    
    This is what production companies use, NOT transformers.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-70B-Instruct",
        draft_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = False,
        num_speculative_tokens: int = 5,
        speculative_method: str = "eagle"
    ):
        """
        Initialize vLLM client with speculative decoding
        
        Args:
            model_name: Main model (e.g., Llama-3-70B)
            draft_model: Draft model for speculation (e.g., TinyLlama-1.1B)
            gpu_memory_utilization: GPU memory fraction (0.0-1.0)
            max_model_len: Maximum sequence length (context window)
            trust_remote_code: Allow custom model code
            num_speculative_tokens: Number of tokens to speculate (default: 5)
            speculative_method: "eagle" or "ngram" (eagle is better)
        """
        self.model_name = model_name
        self.draft_model = draft_model
        
        logger.info(f"Loading vLLM with speculative decoding:")
        logger.info(f"  Main model: {model_name}")
        logger.info(f"  Draft model: {draft_model}")
        logger.info(f"  Method: {speculative_method}")
        
        # Configure speculative decoding
        speculative_config = {
            "method": speculative_method,
            "model": draft_model,
            "num_speculative_tokens": num_speculative_tokens,
        }
        
        # Initialize vLLM engine with speculative decoding
        self.llm = LLM(
            model=model_name,
            speculative_config=speculative_config,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=1,
            enforce_eager=False  # Better performance
        )
        
        logger.info(f"vLLM model loaded successfully with speculative decoding")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> str:
        """
        Chat completion using vLLM with speculative decoding
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalize frequent tokens
            presence_penalty: Penalize tokens that appear
        
        Returns:
            Generated text
        """
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            
            logger.info(f"Generating chat completion with speculative decoding")
            
            # Generate with vLLM direct API
            outputs = self.llm.chat(messages, sampling_params=sampling_params)
            
            # Extract generated text
            generated_text = outputs[0].outputs[0].text
            
            logger.info(f"vLLM response: {len(generated_text)} chars")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in vLLM chat completion: {e}")
            raise
    
    def completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Text completion using vLLM with speculative decoding
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
        
        Returns:
            Generated text
        """
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop if stop else None
            )
            
            logger.info(f"Generating completion with speculative decoding")
            
            # Generate with vLLM direct API
            outputs = self.llm.generate([prompt], sampling_params=sampling_params)
            
            # Extract generated text
            generated_text = outputs[0].outputs[0].text
            
            logger.info(f"vLLM completion: {len(generated_text)} chars")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in vLLM completion: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if vLLM model is loaded"""
        return self.llm is not None
    
    def get_model_name(self) -> str:
        """Get the loaded model name"""
        return self.model_name
    
    def get_metrics(self) -> Optional[Dict]:
        """Get inference metrics if available"""
        try:
            return self.llm.get_metrics()
        except:
            logger.warning("Metrics not available in this vLLM version")
            return None
