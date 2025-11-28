"""
LM Studio API Routes (archived from `routes/lm_studio.py`)
"""

from fastapi import APIRouter, HTTPException
from deprecated.models import ChatCompletionRequest, CompletionRequest
from clients.lm_studio_client import LMStudioClient
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lm-studio", tags=["LM Studio"])

# Initialize LM Studio client
lm_client = LMStudioClient()


@router.get("/models")
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


@router.post("/chat")
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


@router.post("/completion")
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
