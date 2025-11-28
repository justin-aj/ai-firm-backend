from pydantic import BaseModel, Field, field_validator
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


class SequentialThinkingRequest(BaseModel):
    """Request model for Sequential Thinking"""
    thought: str = Field(..., min_length=1, max_length=1000, description="Current reasoning step")
    thought_number: int = Field(..., ge=1, description="Current step number")
    total_thoughts: int = Field(..., ge=1, description="Total planned steps")
    next_thought_needed: bool = Field(..., description="Whether another step is needed")


class ScrapeUrlRequest(BaseModel):
    """Request model for scraping a single URL"""
    url: str = Field(..., description="URL to scrape")
    extract_markdown: bool = Field(True, description="Extract markdown content")
    extract_html: bool = Field(False, description="Include raw HTML")
    extract_links: bool = Field(False, description="Extract all links from the page")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate that URL starts with http:// or https://"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError(f"URL must start with 'http://' or 'https://': {v}")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://example.com",
                    "extract_markdown": True,
                    "extract_html": False,
                    "extract_links": False
                }
            ]
        }
    }


class ScrapeUrlsRequest(BaseModel):
    """Request model for scraping multiple URLs"""
    urls: List[str] = Field(..., min_length=1, description="List of URLs to scrape")
    extract_markdown: bool = Field(True, description="Extract markdown content")
    extract_html: bool = Field(False, description="Include raw HTML")
    extract_links: bool = Field(False, description="Extract all links from the page")
    max_concurrent: int = Field(5, ge=1, le=20, description="Maximum concurrent requests")
    
    @field_validator('urls')
    @classmethod
    def validate_urls(cls, v: List[str]) -> List[str]:
        """Validate that all URLs start with http:// or https://"""
        for url in v:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"URL must start with 'http://' or 'https://': {url}")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "urls": [
                        "https://example.com",
                        "https://example.org"
                    ],
                    "extract_markdown": True,
                    "extract_html": False,
                    "extract_links": False,
                    "max_concurrent": 5
                }
            ]
        }
    }


class SearchAndScrapeRequest(BaseModel):
    """Request model for searching and scraping results"""
    query: str = Field(..., description="Search query string")
    num_results: int = Field(5, ge=1, le=10, description="Number of results to scrape (1-10)")
    start: int = Field(1, ge=1, description="The index of the first result")
    extract_markdown: bool = Field(True, description="Extract markdown content")
    extract_html: bool = Field(False, description="Include raw HTML")
    max_concurrent: int = Field(3, ge=1, le=10, description="Maximum concurrent scrape requests")
    additional_params: Optional[Dict[str, Any]] = Field(None, description="Additional search parameters")

