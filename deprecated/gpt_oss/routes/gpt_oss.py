"""
Archived GPT-OSS API Routes

This file is a direct copy of the previous `routes/gpt_oss.py` implementation
and is kept for historical reference only. It should not be imported by the
active application. Use vLLM routes instead.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from clients.gpt_oss_client import GPTOSSClient
from config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# Global client instance
gpt_client: Optional[GPTOSSClient] = None


def get_gpt_client() -> GPTOSSClient:
    """Get or create GPT-OSS client"""
    global gpt_client
    if gpt_client is None:
        gpt_client = GPTOSSClient(base_url=settings.lm_studio_base_url)
    return gpt_client


class ChatRequest(BaseModel):
    message: str = Field(...)
    system_prompt: Optional[str] = Field(None)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2048)
    reset_history: bool = Field(False)


class AskRequest(BaseModel):
    question: str = Field(...)
    temperature: float = Field(0.7)
    max_tokens: Optional[int] = Field(2048)


class CompleteRequest(BaseModel):
    prompt: str = Field(...)
    temperature: float = Field(0.7)
    max_tokens: Optional[int] = Field(2048)


class SetHistoryRequest(BaseModel):
    history: List[Dict[str, str]] = Field(...)


@router.post("/chat")
async def chat_with_gpt_oss(request: ChatRequest):
    try:
        client = get_gpt_client()
        result = await client.chat(
            user_input=request.message,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            reset_history=request.reset_history,
        )
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        return {
            "response": result["assistant_response"],
            "conversation_length": result["conversation_length"],
            "model": result["model"],
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint (archived): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def ask_gpt_oss(request: AskRequest):
    try:
        client = get_gpt_client()
        response = await client.ask(
            question=request.question,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return {"question": request.question, "answer": response, "model": "gpt-oss-20b-Q4_0"}
    except Exception as e:
        logger.error(f"Error in ask endpoint (archived): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete")
async def complete_text(request: CompleteRequest):
    try:
        client = get_gpt_client()
        completion = await client.complete(
            prompt=request.prompt, temperature=request.temperature, max_tokens=request.max_tokens
        )
        return {"prompt": request.prompt, "completion": completion, "model": "gpt-oss-20b-Q4_0"}
    except Exception as e:
        logger.error(f"Error in complete endpoint (archived): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_conversation_history():
    try:
        client = get_gpt_client()
        history = client.get_history()
        return {"history": history, "length": len(history)}
    except Exception as e:
        logger.error(f"Error getting history (archived): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/clear")
async def clear_conversation_history():
    try:
        client = get_gpt_client()
        client.clear_history()
        return {"status": "success", "message": "Conversation history cleared"}
    except Exception as e:
        logger.error(f"Error clearing history (archived): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/set")
async def set_conversation_history(request: SetHistoryRequest):
    try:
        client = get_gpt_client()
        client.set_history(request.history)
        return {"status": "success", "message": f"History set with {len(request.history)} messages"}
    except Exception as e:
        logger.error(f"Error setting history (archived): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_gpt_oss_status():
    try:
        client = get_gpt_client()
        is_available = await client.is_available()
        return {"available": is_available, "model": "gpt-oss-20b-Q4_0", "lm_studio_url": settings.lm_studio_base_url}
    except Exception as e:
        logger.error(f"Error checking status (archived): {e}")
        return {"available": False, "error": str(e)}
"""
Archived GPT-OSS route handlers for historical reference only.
Do not import these routes in the running application.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/deprecated/gptoss/status")
async def status():
    return {"status": "deprecated", "note": "GPT-OSS routes are archived and not part of the app."}
