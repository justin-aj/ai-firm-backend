"""
Router package initialization
"""

from .core import router as core_router
from .lm_studio import router as lm_studio_router
from .search import router as search_router
from .sequential_thinking import router as sequential_thinking_router
from .scraper import router as scraper_router
from .embeddings import router as embeddings_router

__all__ = [
    "core_router",
    "lm_studio_router",
    "search_router",
    "sequential_thinking_router",
    "scraper_router",
    "embeddings_router"
]
