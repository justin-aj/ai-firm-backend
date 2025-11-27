"""
Intelligent Query Routes
Multi-agent workflow for processing user queries
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from routes.intelligent_query_service import (
    process_intelligent_query,
    run_ingestion_phase,
    run_synthesis_phase,
)
from clients.embedding_client import EmbeddingClient
from clients.vllm_client import VLLMClient as LLMClient
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Intelligent Query"])


class IntelligentQueryRequest(BaseModel):
    """Request model for intelligent query processing"""
    question: str = Field(..., description="The user's question")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    # Image search/analysis options
    include_image_search: bool = Field(False, description="Whether to search for images")
    image_query: Optional[str] = Field(None, description="Optional alternate query string for image search")
    image_num_results: int = Field(5, ge=1, le=10, description="Number of image results to fetch")
    enable_image_analysis: bool = Field(False, description="Whether to analyze found images with VLM")
    image_analysis_question: Optional[str] = Field(None, description="Optional custom question to ask VLM about each image")
    store_image_analysis: bool = Field(False, description="Store image analysis embeddings to Milvus collection")


class IntelligentQueryResponse(BaseModel):
    """Response model for intelligent query"""
    success: bool
    topics: List[str]
    search_results: List[Dict[str, Any]]
    scraped_content: List[Dict[str, Any]]
    image_search_results: List[Dict[str, Any]]
    image_analysis_results: List[Dict[str, Any]]
    stored_in_milvus: bool
    milvus_ids: List[int]
    stored_image_in_milvus: bool
    image_milvus_ids: List[int]
    retrieved_context: List[Dict[str, Any]]
    llm_answer: str


@router.post("/ask/ingest", response_model=IntelligentQueryResponse)
async def intelligent_ask_run_ingestion_phase(request: IntelligentQueryRequest):
    """
    Process a user question through multi-agent workflow.
    """
    # Input validation
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Delegate orchestration to the service layer
        partial_response, existing_context, embedding_client, llm_client = await run_ingestion_phase(request)

        # Build a response containing default fields required by IntelligentQueryResponse
        response: Dict[str, Any] = {
            "success": partial_response.get("success", True),
            "topics": partial_response.get("topics", []),
            "search_results": partial_response.get("search_results", []),
            "scraped_content": partial_response.get("scraped_content", []),
            "image_search_results": [],
            "image_analysis_results": [],
            "stored_in_milvus": partial_response.get("stored_in_milvus", False),
            "milvus_ids": partial_response.get("milvus_ids", []),
            "stored_image_in_milvus": False,
            "image_milvus_ids": [],
            "retrieved_context": existing_context or [],
            "llm_answer": "",
        }
        return IntelligentQueryResponse(**response)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class IntelligentQuerySynthesisRequest(IntelligentQueryRequest):
    """Synthesis-only request schema: existing_context can be provided and image_num_results can be 0."""
    existing_context: Optional[List[Dict[str, Any]]] = None
    # Allow passing image_num_results=0 when include_image_search is False
    image_num_results: int = Field(0, ge=0, le=10)


@router.post("/ask/synth", response_model=IntelligentQueryResponse)
async def intelligent_ask_run_synthesis_phase(request: IntelligentQuerySynthesisRequest):
    """
    Process a user question through multi-agent workflow.
    """
    # Input validation
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Delegate orchestration to the service layer
        # For synthesis, reinitialize necessary clients and run the pipeline with provided context
        embedding_client = EmbeddingClient()
        llm_client = LLMClient(gpu_memory_utilization=0.6, num_speculative_tokens=3, max_model_len=8192)
        existing_context = request.existing_context or []
        synthesis_result = await run_synthesis_phase(request, existing_context, embedding_client, llm_client)

        # Build final response with default placeholders for fields not produced during synthesis
        response: Dict[str, Any] = {
            "success": True,
            "topics": [],
            "search_results": [],
            "scraped_content": [],
            "image_search_results": [],
            "image_analysis_results": [],
            "stored_in_milvus": False,
            "milvus_ids": [],
            "stored_image_in_milvus": False,
            "image_milvus_ids": [],
            "retrieved_context": synthesis_result.get("retrieved_context", []),
            "llm_answer": synthesis_result.get("llm_answer", ""),
        }
        return IntelligentQueryResponse(**response)
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))