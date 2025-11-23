import httpx
import logging
from typing import Optional, Dict, Any, List
from config import get_settings


logger = logging.getLogger(__name__)


class LMStudioClient:
    """Client for interacting with LM Studio local server"""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.lm_studio_base_url
        self.model = self.settings.lm_studio_model
    
    async def get_models(self) -> Dict[str, Any]:
        """Get available models from LM Studio"""
        async with httpx.AsyncClient() as client:
            try:
                logger.info("Requesting LM Studio models from %s/models", self.base_url)
                response = await client.get(f"{self.base_url}/models")
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError:
                logger.error("LM Studio connection refused at %s", self.base_url)
                return {
                    "error": "Cannot connect to LM Studio. Please ensure LM Studio is running on port 1234"
                }
            except Exception as e:
                logger.error("Error fetching models from LM Studio: %s", str(e))
                return {"error": f"Error fetching models: {str(e)}"}
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a chat completion request to LM Studio"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                payload = {
                    "model": model or self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": stream
                }
                if max_tokens:
                    payload["max_tokens"] = max_tokens
                
                logger.info("LM Studio chat completion request to %s", f"{self.base_url}/chat/completions")
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
                )
                
                # Log response for debugging
                if response.status_code != 200:
                    logger.error("LM Studio error %s: %s", response.status_code, response.text)
                
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError:
                logger.error("LM Studio connection refused when posting chat completion to %s", self.base_url)
                return {
                    "error": "Cannot connect to LM Studio. Please ensure LM Studio is running on port 1234"
                }
            except Exception as e:
                logger.error("Error during LM Studio chat completion: %s", str(e))
                return {"error": f"Error during chat completion: {str(e)}"}
    
    async def completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a completion request to LM Studio"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                payload = {
                    "model": model or self.model,
                    "prompt": prompt,
                    "temperature": temperature
                }
                if max_tokens:
                    payload["max_tokens"] = max_tokens
                
                logger.info("LM Studio completion request to %s", f"{self.base_url}/completions")
                response = await client.post(
                    f"{self.base_url}/completions",
                    json=payload
                )
                
                # Log response for debugging
                if response.status_code != 200:
                    logger.error("LM Studio completions error %s: %s", response.status_code, response.text)
                
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError:
                logger.error("LM Studio connection refused when posting completion to %s", self.base_url)
                return {
                    "error": "Cannot connect to LM Studio. Please ensure LM Studio is running on port 1234"
                }
            except Exception as e:
                logger.error("Error during LM Studio completion: %s", str(e))
                return {"error": f"Error during completion: {str(e)}"}
