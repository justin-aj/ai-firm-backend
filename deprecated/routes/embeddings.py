"""
Embedding and vector database routes (archived from `routes/embeddings.py`)
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from clients.embedding_client import EmbeddingClient
from clients.milvus_client import MilvusClient
from clients.web_scraper_client import WebScraperClient
from config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["Embeddings"])

# Initialize clients (lazy loaded)
embedding_client = EmbeddingClient()
milvus_client = None  # Will be initialized on first use
scraper_client = WebScraperClient()


def get_milvus_client() -> MilvusClient:
	"""Get or create Milvus client"""
	global milvus_client
	if milvus_client is None:
		milvus_client = MilvusClient(
			host=settings.milvus_host,
			port=settings.milvus_port,
			collection_name=settings.milvus_collection
		)
	return milvus_client


# Request/Response Models
class EmbeddingRequest(BaseModel):
	"""Request model for generating embeddings"""
	text: str = Field(..., description="Text to embed")
	max_length: int = Field(512, ge=1, le=2048, description="Maximum token length")


class BatchEmbeddingRequest(BaseModel):
	"""Request model for batch embedding generation"""
	texts: List[str] = Field(..., min_length=1, description="List of texts to embed")
	max_length: int = Field(512, ge=1, le=2048, description="Maximum token length")
	batch_size: int = Field(32, ge=1, le=128, description="Batch size for processing")


class SimilarityRequest(BaseModel):
	"""Request model for similarity calculation"""
	text1: str = Field(..., description="First text")
	text2: str = Field(..., description="Second text")


class VectorInsertRequest(BaseModel):
	"""Request model for inserting vectors into Milvus"""
	texts: List[str] = Field(..., min_length=1, description="List of texts to insert")
	metadata: Optional[List[Dict[str, Any]]] = Field(None, description="Optional metadata for each text")
	auto_embed: bool = Field(True, description="Automatically generate embeddings")


class VectorSearchRequest(BaseModel):
	"""Request model for vector similarity search"""
	query: str = Field(..., description="Search query text")
	top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
	metric_type: str = Field("L2", description="Distance metric (L2, IP, COSINE)")


class ScrapeAndEmbedRequest(BaseModel):
	"""Request model for scraping URLs and storing in Milvus"""
	urls: List[str] = Field(..., min_length=1, max_length=100, description="URLs to scrape and embed")
	extract_markdown: bool = Field(True, description="Extract markdown content")
	extract_html: bool = Field(False, description="Include raw HTML")
	chunk_size: int = Field(1000, ge=100, le=5000, description="Text chunk size for splitting content")
	chunk_overlap: int = Field(200, ge=0, le=1000, description="Overlap between chunks")
	max_concurrent: int = Field(5, ge=1, le=20, description="Maximum concurrent scraping requests")
	auto_init: bool = Field(True, description="Automatically initialize Milvus collection if needed")
    
	model_config = {
		"json_schema_extra": {
			"examples": [{
				"urls": ["https://example.com", "https://example.org"],
				"extract_markdown": True,
				"chunk_size": 1000,
				"chunk_overlap": 200,
				"max_concurrent": 5
			}]
		}
	}

# (remaining implementation archived)
