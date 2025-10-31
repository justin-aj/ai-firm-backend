from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from lm_studio_client import LMStudioClient
from google_search_client import GoogleCustomSearchClient
from models import ChatCompletionRequest, CompletionRequest, SearchRequest, ImageSearchRequest
from config import get_settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="AI Firm Backend",
    description="Backend API for AI Firm application with LM Studio and Google Custom Search integration",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,  # Disable docs in production
    redoc_url="/redoc" if settings.debug else None
)

# Configure CORS
allowed_origins = ["*"] if settings.debug else [
    "http://localhost:3000",
    "http://localhost:5173",
    # Add your production frontend URLs here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add trusted host middleware (disable in development)
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
    )

# Initialize clients
lm_client = LMStudioClient()
search_client = GoogleCustomSearchClient()

@app.get("/")
async def root():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to AI Firm Backend API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Remove debug endpoint in production
if settings.debug:
    @app.get("/debug/config")
    async def debug_config():
        """Debug endpoint to check configuration (dev only)"""
        return {
            "google_api_key": settings.google_api_key[:10] + "..." if settings.google_api_key else "NOT SET",
            "google_cx": settings.google_cx if settings.google_cx else "NOT SET",
            "lm_studio_base_url": settings.lm_studio_base_url,
            "debug_mode": settings.debug
        }

@app.get("/lm-studio/models")
async def get_lm_studio_models():
    """Get available models from LM Studio"""
    try:
        logger.info("Fetching LM Studio models")
        result = await lm_client.get_models()
        if "error" in result:
            logger.error(f"LM Studio error: {result['error']}")
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Unexpected error in get_lm_studio_models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/lm-studio/chat")
async def chat_completion(request: ChatCompletionRequest):
    """Send a chat completion request to LM Studio"""
    try:
        logger.info(f"Chat completion request with {len(request.messages)} messages")
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        result = await lm_client.chat_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        if "error" in result:
            logger.error(f"LM Studio chat error: {result['error']}")
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat_completion: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/lm-studio/completion")
async def completion(request: CompletionRequest):
    """Send a text completion request to LM Studio"""
    try:
        logger.info(f"Completion request for prompt length: {len(request.prompt)}")
        result = await lm_client.completion(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        if "error" in result:
            logger.error(f"LM Studio completion error: {result['error']}")
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in completion: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/search")
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

@app.post("/search/urls")
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

@app.post("/search/detailed")
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

@app.post("/search/images")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )
