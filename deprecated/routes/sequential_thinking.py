"""
Sequential Thinking routes (archived from `routes/sequential_thinking.py`)
"""

from fastapi import APIRouter, HTTPException
from deprecated.models import SequentialThinkingRequest
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sequential-thinking", tags=["Sequential Thinking"])

# Sequential thinking session storage (shared with MCP server concept)
thinking_sessions = {}


@router.post("")
async def sequential_thinking(request: SequentialThinkingRequest):
	"""Process a sequential thinking step"""
	try:
		logger.info(f"Sequential thinking step {request.thought_number}/{request.total_thoughts}")
        
		# Store the thinking step
		session_key = "default_session"
		if session_key not in thinking_sessions:
			thinking_sessions[session_key] = []
        
		thinking_sessions[session_key].append({
			"step": request.thought_number,
			"thought": request.thought,
			"timestamp": f"Step {request.thought_number}/{request.total_thoughts}"
		})
        
		# Create response
		result = {
			"stepResult": f"Step {request.thought_number}/{request.total_thoughts} processed: {request.thought}",
			"nextStepNeeded": request.next_thought_needed,
			"progress": f"{request.thought_number}/{request.total_thoughts}",
			"currentSession": thinking_sessions[session_key]
		}
        
		# If this is the final thought, return all thoughts and clear
		if not request.next_thought_needed or request.thought_number >= request.total_thoughts:
			result["allThoughts"] = thinking_sessions[session_key]
			result["completed"] = True
			thinking_sessions[session_key] = []
		else:
			result["completed"] = False
        
		return result
	except Exception as e:
		logger.error(f"Unexpected error in sequential_thinking: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/session")
async def get_thinking_session():
	"""Get current sequential thinking session"""
	session_key = "default_session"
	return {
		"session": thinking_sessions.get(session_key, []),
		"total_steps": len(thinking_sessions.get(session_key, []))
	}


@router.delete("/session")
async def clear_thinking_session():
	"""Clear the current sequential thinking session"""
	session_key = "default_session"
	thinking_sessions[session_key] = []
	return {"message": "Session cleared", "status": "success"}
