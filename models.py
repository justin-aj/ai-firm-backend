from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


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


class SearchRequest(BaseModel):
    """Request model for Google Custom Search"""
    query: str = Field(..., description="Search query string")
    num_results: int = Field(10, ge=1, le=10, description="Number of results to return (1-10)")
    start: int = Field(1, ge=1, description="The index of the first result to return")
    additional_params: Optional[Dict[str, Any]] = Field(None, description="Additional search parameters")


class ImageSearchRequest(BaseModel):
    """Request model for Google Custom Image Search"""
    query: str = Field(..., description="Image search query string")
    num_results: int = Field(10, ge=1, le=10, description="Number of results to return (1-10)")
    start: int = Field(1, ge=1, description="The index of the first result to return")
    additional_params: Optional[Dict[str, Any]] = Field(None, description="Additional search parameters")
