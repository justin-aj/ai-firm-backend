"""
GPT-OSS-20B API Routes
Endpoints for chatting with the GPT-OSS-20B model via LM Studio
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


# Request Models
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's message/question")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, description="Maximum response length")
    reset_history: bool = Field(False, description="Clear conversation history before this message")


class AskRequest(BaseModel):
    """Request model for simple question-answer"""
    question: str = Field(..., description="Question to ask the model")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, description="Maximum response length")


class CompleteRequest(BaseModel):
    """Request model for text completion"""
    prompt: str = Field(..., description="Text prompt to complete")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, description="Maximum tokens to generate")


class SetHistoryRequest(BaseModel):
    """Request model for setting conversation history"""
    history: List[Dict[str, str]] = Field(..., description="Conversation history")


# Endpoints
@router.post("/chat")
async def chat_with_gpt_oss(request: ChatRequest):
    """
    Chat with GPT-OSS-20B model
    
    Maintains conversation history across messages.
    Use reset_history=true to start a new conversation.
    
    Example:
    ```json
    {
        "message": "What is machine learning?",
        "system_prompt": "You are a helpful AI assistant",
        "temperature": 0.7,
        "max_tokens": 2048,
        "reset_history": false
    }
    ```
    """
    try:
        client = get_gpt_client()
        
        result = await client.chat(
            user_input=request.message,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            reset_history=request.reset_history
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        return {
            "response": result["assistant_response"],
            "conversation_length": result["conversation_length"],
            "model": result["model"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def ask_gpt_oss(request: AskRequest):
    """
    Simple question-answer with GPT-OSS-20B
    
    Does NOT maintain conversation history.
    Good for one-off questions.
    
    Example:
    ```json
    {
        "question": "Explain quantum computing in simple terms",
        "temperature": 0.7,
        "max_tokens": 1024
    }
    ```
    """
    try:
        client = get_gpt_client()
        
        response = await client.ask(
            question=request.question,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return {
            "question": request.question,
            "answer": response,
            "model": "gpt-oss-20b-Q4_0"
        }
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/complete")
async def complete_text(request: CompleteRequest):
    """
    Text completion with GPT-OSS-20B
    
    Completes the given prompt.
    
    Example:
    ```json
    {
        "prompt": "Once upon a time, in a distant galaxy",
        "temperature": 0.8,
        "max_tokens": 512
    }
    ```
    """
    try:
        client = get_gpt_client()
        
        completion = await client.complete(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return {
            "prompt": request.prompt,
            "completion": completion,
            "model": "gpt-oss-20b-Q4_0"
        }
        
    except Exception as e:
        logger.error(f"Error in complete endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_conversation_history():
    """
    Get current conversation history
    
    Returns the list of messages in the current conversation.
    """
    try:
        client = get_gpt_client()
        history = client.get_history()
        
        return {
            "history": history,
            "length": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/clear")
async def clear_conversation_history():
    """
    Clear conversation history
    
    Resets the conversation to start fresh.
    """
    try:
        client = get_gpt_client()
        client.clear_history()
        
        return {
            "status": "success",
            "message": "Conversation history cleared"
        }
        
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/history/set")
async def set_conversation_history(request: SetHistoryRequest):
    """
    Set conversation history
    
    Load a previous conversation or set up a specific context.
    
    Example:
    ```json
    {
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"}
        ]
    }
    ```
    """
    try:
        client = get_gpt_client()
        client.set_history(request.history)
        
        return {
            "status": "success",
            "message": f"History set with {len(request.history)} messages"
        }
        
    except Exception as e:
        logger.error(f"Error setting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_gpt_oss_status():
    """
    Check if GPT-OSS model is available
    
    Verifies LM Studio connection and model availability.
    """
    try:
        client = get_gpt_client()
        is_available = await client.is_available()
        
        return {
            "available": is_available,
            "model": "gpt-oss-20b-Q4_0",
            "lm_studio_url": settings.lm_studio_base_url
        }
        
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return {
            "available": False,
            "error": str(e)
        }
