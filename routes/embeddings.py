"""
Embedding and vector database routes
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


# Embedding Endpoints
@router.post("/embed")
async def generate_embedding(request: EmbeddingRequest):
    """Generate embedding for a single text"""
    try:
        embedding = embedding_client.generate_embedding(
            request.text,
            max_length=request.max_length
        )
        
        return {
            "text": request.text,
            "embedding": embedding,
            "dimension": len(embedding)
        }
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")


@router.post("/embed/batch")
async def generate_batch_embeddings(request: BatchEmbeddingRequest):
    """Generate embeddings for multiple texts"""
    try:
        embeddings_tensor = embedding_client.generate_embeddings(
            request.texts,
            max_length=request.max_length,
            batch_size=request.batch_size
        )
        
        # Convert to list of lists
        embeddings = embeddings_tensor.tolist()
        
        return {
            "count": len(request.texts),
            "dimension": len(embeddings[0]) if embeddings else 0,
            "embeddings": embeddings
        }
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


@router.post("/embed/similarity")
async def calculate_similarity(request: SimilarityRequest):
    """Calculate cosine similarity between two texts"""
    try:
        similarity_score = embedding_client.similarity(
            request.text1,
            request.text2
        )
        
        return {
            "text1": request.text1,
            "text2": request.text2,
            "similarity": similarity_score
        }
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating similarity: {str(e)}")


# Milvus Vector Database Endpoints
@router.post("/vectors/init")
async def initialize_collection():
    """Initialize Milvus collection"""
    try:
        client = get_milvus_client()
        client.connect()
        
        # Get embedding dimension
        embedding_dim = embedding_client.get_embedding_dimension()
        
        # Create collection
        client.create_collection(embedding_dim=embedding_dim)
        
        # Create index
        client.create_index(index_type="IVF_FLAT", metric_type="L2")
        
        stats = client.get_collection_stats()
        
        return {
            "status": "initialized",
            "collection": stats
        }
    except Exception as e:
        logger.error(f"Error initializing collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing collection: {str(e)}")


@router.post("/vectors/insert")
async def insert_vectors(request: VectorInsertRequest):
    """Insert texts and their embeddings into Milvus"""
    try:
        client = get_milvus_client()
        
        # Generate embeddings if needed
        if request.auto_embed:
            embeddings_tensor = embedding_client.generate_embeddings(request.texts)
            embeddings = embeddings_tensor.tolist()
        else:
            raise HTTPException(status_code=400, detail="auto_embed=False not yet supported")
        
        # Insert into Milvus
        ids = client.insert(
            texts=request.texts,
            embeddings=embeddings,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "inserted_count": len(ids),
            "ids": ids
        }
    except Exception as e:
        logger.error(f"Error inserting vectors: {e}")
        raise HTTPException(status_code=500, detail=f"Error inserting vectors: {str(e)}")


@router.post("/vectors/search")
async def search_vectors(request: VectorSearchRequest):
    """Search for similar vectors in Milvus"""
    try:
        client = get_milvus_client()
        
        # Generate query embedding
        query_embedding = embedding_client.generate_embedding(request.query)
        
        # Search
        results = client.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            metric_type=request.metric_type
        )
        
        return {
            "query": request.query,
            "top_k": request.top_k,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching vectors: {str(e)}")


@router.get("/vectors/status")
async def get_vector_status():
    """Get Milvus collection status"""
    try:
        client = get_milvus_client()
        client.connect()
        
        stats = client.get_collection_stats()
        
        return {
            "status": "connected",
            "collection": stats,
            "embedding_model": embedding_client.model_name
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@router.delete("/vectors/collection")
async def delete_collection():
    """Delete the Milvus collection"""
    try:
        client = get_milvus_client()
        client.delete_collection()
        
        return {
            "status": "deleted",
            "collection": settings.milvus_collection
        }
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


# Helper function to chunk text
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If not the last chunk, try to break at sentence or word boundary
        if end < len(text):
            # Look for sentence end
            sentence_end = text.rfind('. ', start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
            else:
                # Look for word boundary
                space = text.rfind(' ', start, end)
                if space > start + chunk_size // 2:
                    end = space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else end
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


@router.post("/scrape-and-embed")
async def scrape_and_embed(request: ScrapeAndEmbedRequest):
    """
    Scrape URLs, generate embeddings, and store in Milvus
    
    This endpoint:
    1. Scrapes content from provided URLs
    2. Chunks the text into manageable pieces
    3. Generates embeddings for each chunk
    4. Stores in Milvus with metadata (URL, title, chunk_index)
    """
    try:
        client = get_milvus_client()
        
        # Auto-initialize collection if needed
        if request.auto_init:
            try:
                client.connect()
                embedding_dim = embedding_client.get_embedding_dimension()
                client.create_collection(embedding_dim=embedding_dim)
                client.create_index(index_type="IVF_FLAT", metric_type="L2")
                logger.info("Milvus collection initialized")
            except Exception as e:
                logger.warning(f"Collection may already exist: {e}")
        
        # Scrape URLs
        logger.info(f"Scraping {len(request.urls)} URLs...")
        scrape_results = await scraper_client.scrape_urls(
            urls=request.urls,
            extract_markdown=request.extract_markdown,
            extract_html=request.extract_html,
            max_concurrent=request.max_concurrent
        )
        
        all_texts = []
        all_metadata = []
        total_chunks = 0
        failed_scrapes = 0
        
        # Process each scraped result
        for result in scrape_results:
            if not result.get("success"):
                failed_scrapes += 1
                logger.warning(f"Failed to scrape {result.get('url')}: {result.get('error')}")
                continue
            
            url = result.get("url", "unknown")
            content = result.get("markdown", "") if request.extract_markdown else result.get("html", "")
            
            if not content or len(content.strip()) < 50:
                logger.warning(f"No meaningful content from {url}")
                continue
            
            # Chunk the content
            chunks = chunk_text(
                content,
                chunk_size=request.chunk_size,
                overlap=request.chunk_overlap
            )
            
            # Create metadata for each chunk
            for idx, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadata.append({
                    "url": url,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "source": "web_scraper"
                })
                total_chunks += 1
        
        if not all_texts:
            raise HTTPException(
                status_code=400,
                detail="No content was successfully scraped from the provided URLs"
            )
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_texts)} text chunks...")
        embeddings_tensor = embedding_client.generate_embeddings(all_texts)
        embeddings = embeddings_tensor.tolist()
        
        # Insert into Milvus
        logger.info(f"Inserting {len(embeddings)} vectors into Milvus...")
        ids = client.insert(
            texts=all_texts,
            embeddings=embeddings,
            metadata=all_metadata
        )
        
        return {
            "status": "success",
            "urls_scraped": len(request.urls),
            "urls_failed": failed_scrapes,
            "total_chunks": total_chunks,
            "vectors_inserted": len(ids),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "collection": settings.milvus_collection,
            "details": {
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap,
                "avg_chunk_size": sum(len(t) for t in all_texts) // len(all_texts) if all_texts else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in scrape-and-embed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
