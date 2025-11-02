"""
Core application routes (health, debug, etc.)
"""

from fastapi import APIRouter
from typing import Dict, Any
from config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["Core"])


@router.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to AI Firm Backend API", "version": "1.0.0"}


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}


# Debug endpoint (only in development)
if settings.debug:
    @router.get("/debug/config")
    async def debug_config() -> Dict[str, Any]:
        """Debug endpoint to check configuration (dev only)"""
        return {
            "google_api_key": settings.google_api_key[:10] + "..." if settings.google_api_key else "NOT SET",
            "google_cx": settings.google_cx if settings.google_cx else "NOT SET",
            "lm_studio_base_url": settings.lm_studio_base_url,
            "debug_mode": settings.debug
        }
