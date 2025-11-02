"""
Model Context Protocol (MCP) Server for AI Firm Backend

This module implements MCP tools and resources for:
- Google Custom Search integration
- LM Studio chat/completion endpoints
- Sequential thinking capabilities for multi-step reasoning
"""

from mcp.server import Server
from mcp.types import Tool, TextContent
from typing import Any, Sequence
import json
from clients.google_search_client import GoogleCustomSearchClient
from clients.lm_studio_client import LMStudioClient


# Initialize MCP server with Sequential Thinking support
app = Server("ai-firm-backend")

# Initialize clients
search_client = GoogleCustomSearchClient()
lm_client = LMStudioClient()


# Sequential Thinking Storage
thinking_sessions = {}


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="sequential_thinking",
            description="Enable step-by-step reasoning for complex tasks. Use this to break down problems into multiple thinking steps before taking action.",
            inputSchema={
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "The current step or thought in the reasoning process"
                    },
                    "thoughtNumber": {
                        "type": "integer",
                        "description": "Current step number (1-indexed)",
                        "minimum": 1
                    },
                    "totalThoughts": {
                        "type": "integer",
                        "description": "Total number of planned thinking steps",
                        "minimum": 1
                    },
                    "nextThoughtNeeded": {
                        "type": "boolean",
                        "description": "Whether another thinking step is needed after this one"
                    }
                },
                "required": ["thought", "thoughtNumber", "totalThoughts", "nextThoughtNeeded"]
            }
        ),
        Tool(
            name="google_search",
            description="Search the web using Google Custom Search API. Returns URLs, titles, and snippets for relevant web pages.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find information on the web"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="google_search_urls_only",
            description="Search the web and return only URLs without titles or snippets. Useful when you just need the links.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="google_image_search",
            description="Search for images using Google Custom Search API. Returns image URLs, thumbnails, and context links.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The image search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of image results to return (1-10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="lm_studio_chat",
            description="Send a chat message to the local LM Studio model and get a response. Use this to have conversations with your locally running AI model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "Array of message objects with role (system/user/assistant) and content",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["system", "user", "assistant"],
                                    "description": "The role of the message sender"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The message content"
                                }
                            },
                            "required": ["role", "content"]
                        }
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature for sampling (0.0-2.0). Higher values = more creative",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 2.0
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate in the response",
                        "default": None
                    }
                },
                "required": ["messages"]
            }
        ),
        Tool(
            name="lm_studio_completion",
            description="Send a prompt to LM Studio for text completion. Use this for simple text generation tasks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt text to complete"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature for sampling (0.0-2.0)",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 2.0
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                        "default": None
                    }
                },
                "required": ["prompt"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Handle MCP tool calls"""
    
    if name == "sequential_thinking":
        thought = arguments.get("thought")
        thought_number = arguments.get("thoughtNumber")
        total_thoughts = arguments.get("totalThoughts")
        next_thought_needed = arguments.get("nextThoughtNeeded")
        
        # Store the thinking step
        session_key = "current_session"
        if session_key not in thinking_sessions:
            thinking_sessions[session_key] = []
        
        thinking_sessions[session_key].append({
            "step": thought_number,
            "thought": thought,
            "timestamp": json.dumps({"step": thought_number, "total": total_thoughts})
        })
        
        # Create response
        result = {
            "stepResult": f"Step {thought_number}/{total_thoughts} processed: {thought}",
            "nextStepNeeded": next_thought_needed,
            "progress": f"{thought_number}/{total_thoughts}",
            "allThoughts": thinking_sessions[session_key] if thought_number == total_thoughts else None
        }
        
        # Clear session if this was the last thought
        if not next_thought_needed or thought_number >= total_thoughts:
            thinking_sessions[session_key] = []
        
        return [
            TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]
    
    elif name == "google_search":
        query = arguments.get("query")
        num_results = arguments.get("num_results", 10)
        
        results = search_client.search_detailed(
            query=query,
            num_results=num_results
        )
        
        if not results:
            return [TextContent(type="text", text="No results found")]
        
        return [
            TextContent(
                type="text",
                text=json.dumps({"results": results}, indent=2)
            )
        ]
    
    elif name == "google_search_urls_only":
        query = arguments.get("query")
        num_results = arguments.get("num_results", 10)
        
        urls = search_client.search_urls(
            query=query,
            num_results=num_results
        )
        
        return [
            TextContent(
                type="text",
                text=json.dumps({"urls": urls}, indent=2)
            )
        ]
    
    elif name == "google_image_search":
        query = arguments.get("query")
        num_results = arguments.get("num_results", 10)
        
        results = search_client.search_images(
            query=query,
            num_results=num_results
        )
        
        return [
            TextContent(
                type="text",
                text=json.dumps({"images": results}, indent=2)
            )
        ]
    
    elif name == "lm_studio_chat":
        messages = arguments.get("messages")
        temperature = arguments.get("temperature", 0.7)
        max_tokens = arguments.get("max_tokens")
        
        result = await lm_client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return [
            TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]
    
    elif name == "lm_studio_completion":
        prompt = arguments.get("prompt")
        temperature = arguments.get("temperature", 0.7)
        max_tokens = arguments.get("max_tokens")
        
        result = await lm_client.completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return [
            TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")
