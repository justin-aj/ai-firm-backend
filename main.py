from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from config import get_settings
from routes import (
    core_router,
    lm_studio_router,
    search_router,
    sequential_thinking_router,
    scraper_router,
    embeddings_router,
    # gpt_oss_router removed
    intelligent_query_router
)
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

# Include routers
app.include_router(core_router)
app.include_router(lm_studio_router, prefix="/lm-studio", tags=["LM Studio"])
app.include_router(search_router, prefix="/search", tags=["Search"])
app.include_router(sequential_thinking_router)
app.include_router(scraper_router)
app.include_router(embeddings_router, prefix="/embeddings", tags=["Embeddings"])
    # GPT-OSS routes removed in favor of vLLM client
app.include_router(intelligent_query_router, prefix="/intelligent-query", tags=["Intelligent Query"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )
