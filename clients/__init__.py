"""
Client modules for external services
"""

from .lm_studio_client import LMStudioClient
from .google_search_client import GoogleCustomSearchClient
# WebScraperClient, EmbeddingClient, and MilvusClient are imported lazily to avoid slow dependency loading
# from .web_scraper_client import WebScraperClient
# from .embedding_client import EmbeddingClient
# from .milvus_client import MilvusClient
from .gpt_oss_client import GPTOSSClient

__all__ = [
    "LMStudioClient",
    "GoogleCustomSearchClient",
    # "WebScraperClient",  # Import directly when needed
    # "EmbeddingClient",  # Import directly when needed
    # "MilvusClient",  # Import directly when needed
    "GPTOSSClient"
]
