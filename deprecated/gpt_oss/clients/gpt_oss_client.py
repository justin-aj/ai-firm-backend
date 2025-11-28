"""
Archived GPT-OSS-20B client

This file contains the archived GPT-OSS client implementation as it existed
prior to deprecation. Keep this here for historical reference only.
"""

from typing import List, Dict, Any, Optional
import logging
from clients.lm_studio_client import LMStudioClient

logger = logging.getLogger(__name__)


class GPTOSSClient:
    """
    Archived GPT-OSS client (for historical reference only)
    """
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1"):
        self.lm_client = LMStudioClient()
        self.lm_client.base_url = base_url
        self.model_name = "gpt-oss-20b"
        self.conversation_history: List[Dict[str, str]] = []

    async def is_available(self) -> bool:
        try:
            models = await self.lm_client.get_models()
            return bool(models)
        except Exception:
            return False

    # Original methods omitted for brevity in archival copy
