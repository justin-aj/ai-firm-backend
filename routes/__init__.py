"""
Router package initialization
"""

from routes.core import router as core_router
from routes.lm_studio import router as lm_studio_router
from routes.search import router as search_router
from routes.sequential_thinking import router as sequential_thinking_router
from routes.scraper import router as scraper_router
from routes.embeddings import router as embeddings_router
from routes.gpt_oss import router as gpt_oss_router
from routes.intelligent_query import router as intelligent_query_router

__all__ = [
    "core_router",
    "lm_studio_router",
    "search_router",
    "sequential_thinking_router",
    "scraper_router",
    "embeddings_router",
    "gpt_oss_router",
    "intelligent_query_router"
]
