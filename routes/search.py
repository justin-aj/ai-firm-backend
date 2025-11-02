"""
Google Search API Routes
"""

from fastapi import APIRouter, HTTPException
from models import SearchRequest, ImageSearchRequest
from clients.google_search_client import GoogleCustomSearchClient
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Google Search"])

# Initialize Google Search client
search_client = GoogleCustomSearchClient()


@router.post("")
async def search(request: SearchRequest):
    """Perform a Google Custom Search and return full results"""
    try:
        logger.info(f"Search request for query: {request.query}")
        params = request.additional_params or {}
        result = search_client.search(
            query=request.query,
            num_results=request.num_results,
            start=request.start,
            **params
        )
        if "error" in result:
            logger.error(f"Google Search error: {result['error']}")
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/urls")
async def search_urls(request: SearchRequest):
    """Perform a Google Custom Search and return only URLs"""
    try:
        logger.info(f"URL search request for query: {request.query}")
        params = request.additional_params or {}
        urls = search_client.search_urls(
            query=request.query,
            num_results=request.num_results,
            start=request.start,
            **params
        )
        return {"urls": urls}
    except Exception as e:
        logger.error(f"Unexpected error in search_urls: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/detailed")
async def search_detailed(request: SearchRequest):
    """Perform a Google Custom Search and return detailed results"""
    try:
        logger.info(f"Detailed search request for query: {request.query}")
        params = request.additional_params or {}
        results = search_client.search_detailed(
            query=request.query,
            num_results=request.num_results,
            start=request.start,
            **params
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Unexpected error in search_detailed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/images")
async def search_images(request: ImageSearchRequest):
    """Perform a Google Custom Image Search"""
    try:
        logger.info(f"Image search request for query: {request.query}")
        params = request.additional_params or {}
        results = search_client.search_images(
            query=request.query,
            num_results=request.num_results,
            start=request.start,
            **params
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Unexpected error in search_images: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
