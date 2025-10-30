from pydantic import BaseModel
from typing import List, Optional


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str  # system, user, or assistant
    content: str


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion"""
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class CompletionRequest(BaseModel):
    """Request model for text completion"""
    prompt: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
