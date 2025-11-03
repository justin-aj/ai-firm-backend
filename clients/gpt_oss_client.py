"""
GPT-OSS-20B Client
Specialized client for interacting with the GPT-OSS-20B-Q4_0 model via LM Studio
"""

from typing import List, Dict, Any, Optional
import logging
from clients.lm_studio_client import LMStudioClient

logger = logging.getLogger(__name__)


class GPTOSSClient:
    """
    Client for GPT-OSS-20B model running in LM Studio
    Provides easy-to-use methods for chatting with the model
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1"):
        """
        Initialize GPT-OSS client
        
        Args:
            base_url: LM Studio server URL
        """
        self.lm_client = LMStudioClient()
        self.lm_client.base_url = base_url
        # Model name as it appears in LM Studio (without Q4_0 quantization suffix)
        # From unsloth/gpt-oss-20b-GGUF repository
        self.model_name = "gpt-oss-20b"
        self.conversation_history: List[Dict[str, str]] = []
        logger.info(f"GPT-OSS client initialized for model: {self.model_name}")
    
    async def chat(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        reset_history: bool = False
    ) -> Dict[str, Any]:
        """
        Chat with the GPT-OSS model
        Uses completion endpoint due to jinja template compatibility issues
        
        Args:
            user_input: User's message/question
            system_prompt: Optional system prompt to set behavior
            temperature: Sampling temperature (0.0-2.0). Higher = more creative
            max_tokens: Maximum tokens in response
            reset_history: If True, clears conversation history before this message
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Reset conversation if requested
            if reset_history:
                self.conversation_history = []
                logger.info("Conversation history reset")
            
            # Build prompt from conversation history (using completion endpoint)
            prompt_parts = []
            
            # Add system prompt if provided
            if system_prompt:
                prompt_parts.append(f"System: {system_prompt}\n")
            
            # Add conversation history
            for msg in self.conversation_history:
                role = msg["role"].capitalize()
                content = msg["content"]
                prompt_parts.append(f"{role}: {content}\n")
            
            # Add current user input
            prompt_parts.append(f"User: {user_input}\nAssistant:")
            prompt = "\n".join(prompt_parts)
            
            # Call LM Studio (using completion endpoint to avoid jinja template errors)
            logger.info(f"Sending message to {self.model_name}")
            response = await self.lm_client.completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model=self.model_name
            )
            
            # Extract assistant's response
            assistant_message = response.get("choices", [{}])[0].get("text", "").strip()
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Build result
            result = {
                "success": True,
                "user_input": user_input,
                "assistant_response": assistant_message,
                "model": self.model_name,
                "conversation_length": len(self.conversation_history),
                "full_response": response
            }
            
            logger.info(f"Response received: {len(assistant_message)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error in GPT-OSS chat: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_input": user_input
            }
    
    async def ask(
        self,
        question: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048
    ) -> str:
        """
        Simple question-answer interface (no conversation history)
        Uses completion endpoint for better compatibility with GPT-OSS model
        
        Args:
            question: Question to ask the model
            temperature: Sampling temperature
            max_tokens: Maximum response length
            
        Returns:
            String response from the model
        """
        try:
            # GPT-OSS works better with completion endpoint due to jinja template issues
            logger.info(f"Asking GPT-OSS: {question[:50]}...")
            response = await self.lm_client.completion(
                prompt=question,
                temperature=temperature,
                max_tokens=max_tokens,
                model=self.model_name
            )
            
            logger.info(f"Completion response: {response}")
            result = response.get("choices", [{}])[0].get("text", "")
            logger.info(f"Extracted text: {len(result)} characters")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in GPT-OSS ask: {e}")
            return f"Error: {str(e)}"
    
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048
    ) -> str:
        """
        Text completion interface
        
        Args:
            prompt: Text prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Completed text
        """
        try:
            response = await self.lm_client.completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model=self.model_name
            )
            
            return response.get("choices", [{}])[0].get("text", "")
            
        except Exception as e:
            logger.error(f"Error in GPT-OSS completion: {e}")
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.conversation_history.copy()
    
    def set_history(self, history: List[Dict[str, str]]):
        """Set conversation history"""
        self.conversation_history = history
        logger.info(f"Conversation history set: {len(history)} messages")
    
    async def is_available(self) -> bool:
        """Check if LM Studio server is available"""
        try:
            models = await self.lm_client.get_models()
            return bool(models)
        except Exception as e:
            logger.error(f"LM Studio not available: {e}")
            return False
