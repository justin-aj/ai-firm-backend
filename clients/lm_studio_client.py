import httpx
from typing import Optional, Dict, Any, List
from config import get_settings


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
                response = await client.get(f"{self.base_url}/models")
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError:
                return {
                    "error": "Cannot connect to LM Studio. Please ensure LM Studio is running on port 1234"
                }
            except Exception as e:
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
                
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
                )
                
                # Log response for debugging
                if response.status_code != 200:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"LM Studio error {response.status_code}: {response.text}")
                
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError:
                return {
                    "error": "Cannot connect to LM Studio. Please ensure LM Studio is running on port 1234"
                }
            except Exception as e:
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
                
                response = await client.post(
                    f"{self.base_url}/completions",
                    json=payload
                )
                
                # Log response for debugging
                if response.status_code != 200:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"LM Studio completions error {response.status_code}: {response.text}")
                
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError:
                return {
                    "error": "Cannot connect to LM Studio. Please ensure LM Studio is running on port 1234"
                }
            except Exception as e:
                return {"error": f"Error during completion: {str(e)}"}
