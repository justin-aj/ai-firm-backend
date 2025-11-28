"""
Intelligent Query Routes (archived from `routes/intelligent_query.py`)
Multi-agent workflow for processing user queries
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from routes.intelligent_query_service import process_intelligent_query
from routes.intelligent_query_service import preload_image_vlm
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


@router.post("/ask", response_model=IntelligentQueryResponse)
async def intelligent_ask(request: IntelligentQueryRequest):
	"""
	Process a user question through multi-agent workflow:
	1. Question Analyzer Agent: Uses LLM to break down question into topics/concepts
	2. Google Search: Search for relevant URLs
	3. Web Scraper: Scrape content from URLs
	4. Milvus Storage: Store scraped content with embeddings
    
	Example:
	```json
	{
		"question": "What is the use of Triton in ML? How can it help me in my GPU based project?",
		"temperature": 0.7
	}
	```
	"""
	# Input validation
	if not request.question or len(request.question.strip()) == 0:
		raise HTTPException(status_code=400, detail="Question cannot be empty")

	# Delegate orchestration to the service layer
	result = await process_intelligent_query(request)

	# Build and return response
	return IntelligentQueryResponse(**result)


class PreloadVLMRequest(BaseModel):
	tensor_parallel_size: int = Field(1, ge=1, description="Number of GPUs to use in tensor parallel mode")
	force_reload: bool = Field(False, description="Force reload the model if already loaded")


@router.post("/preload-vlm")
async def preload_vlm_endpoint(request: PreloadVLMRequest):
	"""Eagerly preload the Qwen3-VL VLM into memory to avoid first-call latency.

	This endpoint will create or recreate a shared ImageAnalyzerClient and load its VLM.
	"""
	try:
		result = preload_image_vlm(tensor_parallel_size=request.tensor_parallel_size, force_reload=request.force_reload)
		return result
	except Exception as e:
		logger.error(f"Failed to preload VLM: {e}")
		raise HTTPException(status_code=500, detail=str(e))
