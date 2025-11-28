"""
Client modules for external services
"""

from .lm_studio_client import LMStudioClient
from .google_search_client import GoogleCustomSearchClient
from .vllm_client import VLLMClient
# WebScraperClient, EmbeddingClient, and MilvusClient are imported lazily to avoid slow dependency loading
# from .web_scraper_client import WebScraperClient
# from .embedding_client import EmbeddingClient
# from .milvus_client import MilvusClient

__all__ = [
    "LMStudioClient",
    "GoogleCustomSearchClient",
    "VLLMClient",
    # "WebScraperClient",  # Import directly when needed
    # "EmbeddingClient",  # Import directly when needed
    # "MilvusClient",  # Import directly when needed
    # GPTOSSClient removed — vLLM is used as the default LLM
]
