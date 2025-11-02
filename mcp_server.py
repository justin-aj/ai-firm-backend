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
from clients.web_scraper_client import WebScraperClient
from clients.embedding_client import EmbeddingClient
from clients.milvus_client import MilvusClient
from config import get_settings


# Initialize MCP server with Sequential Thinking support
app = Server("ai-firm-backend")

# Get settings
settings = get_settings()

# Initialize clients
search_client = GoogleCustomSearchClient()
lm_client = LMStudioClient()
scraper_client = WebScraperClient()
embedding_client = EmbeddingClient()
milvus_client = None  # Lazy initialization


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
        ),
        Tool(
            name="scrape_url",
            description="Scrape content from a single URL using Crawl4AI. Returns markdown content, raw HTML, and metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape"
                    },
                    "include_links": {
                        "type": "boolean",
                        "description": "Whether to extract and include links from the page",
                        "default": True
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="scrape_urls_batch",
            description="Scrape multiple URLs in parallel using Crawl4AI with Dask for distributed processing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "description": "List of URLs to scrape",
                        "items": {
                            "type": "string"
                        }
                    },
                    "include_links": {
                        "type": "boolean",
                        "description": "Whether to extract links from pages",
                        "default": True
                    }
                },
                "required": ["urls"]
            }
        ),
        Tool(
            name="generate_embedding",
            description="Generate a text embedding vector using the BGE-M3 model (1024 dimensions). Useful for semantic search and similarity comparison.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to generate an embedding for"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="scrape_and_embed",
            description="Complete RAG pipeline: Scrape URLs, chunk content, generate embeddings, and store in Milvus vector database for semantic search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "description": "URLs to scrape and embed",
                        "items": {
                            "type": "string"
                        }
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Size of text chunks in characters",
                        "default": 1000,
                        "minimum": 100,
                        "maximum": 8000
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "description": "Character overlap between chunks",
                        "default": 200,
                        "minimum": 0,
                        "maximum": 1000
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Milvus collection name to store embeddings",
                        "default": "ai_firm_vectors"
                    }
                },
                "required": ["urls"]
            }
        ),
        Tool(
            name="semantic_search",
            description="Search the Milvus vector database for semantically similar content based on a query. Returns relevant text chunks with similarity scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find similar content"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Milvus collection to search in",
                        "default": "ai_firm_vectors"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Handle MCP tool calls"""
    global milvus_client
    
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
    
    elif name == "scrape_url":
        url = arguments.get("url")
        include_links = arguments.get("include_links", True)
        
        result = await scraper_client.scrape_url(
            url=url,
            include_links=include_links
        )
        
        return [
            TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )
        ]
    
    elif name == "scrape_urls_batch":
        urls = arguments.get("urls")
        include_links = arguments.get("include_links", True)
        
        results = await scraper_client.scrape_urls(
            urls=urls,
            include_links=include_links
        )
        
        return [
            TextContent(
                type="text",
                text=json.dumps({"results": results, "total": len(results)}, indent=2)
            )
        ]
    
    elif name == "generate_embedding":
        text = arguments.get("text")
        
        embedding = embedding_client.generate_embedding(text)
        
        return [
            TextContent(
                type="text",
                text=json.dumps({
                    "embedding": embedding,
                    "dimension": len(embedding),
                    "model": embedding_client.model_name
                }, indent=2)
            )
        ]
    
    elif name == "scrape_and_embed":
        urls = arguments.get("urls")
        chunk_size = arguments.get("chunk_size", 1000)
        chunk_overlap = arguments.get("chunk_overlap", 200)
        collection_name = arguments.get("collection_name", settings.milvus_collection)
        
        # Initialize Milvus client if needed
        if milvus_client is None:
            milvus_client = MilvusClient(
                host=settings.milvus_host,
                port=settings.milvus_port,
                collection_name=collection_name
            )
            milvus_client.connect()
            
            # Create collection if needed
            embedding_dim = embedding_client.get_embedding_dimension()
            milvus_client.create_collection(dim=embedding_dim)
            milvus_client.create_index()
        
        # Step 1: Scrape URLs
        scrape_results = await scraper_client.scrape_urls(urls=urls, include_links=False)
        
        # Step 2: Chunk and embed
        all_texts = []
        all_metadata = []
        
        for result in scrape_results:
            if result.get("success"):
                content = result.get("markdown_content", "")
                url = result.get("url", "")
                
                # Smart chunking
                chunks = chunk_text(content, chunk_size, chunk_overlap)
                
                for idx, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    all_metadata.append({
                        "url": url,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "source": "scrape_and_embed"
                    })
        
        if not all_texts:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": "No content to embed"}, indent=2)
                )
            ]
        
        # Step 3: Generate embeddings
        import torch
        embeddings_tensor = embedding_client.generate_embeddings(all_texts)
        embeddings = embeddings_tensor.tolist()
        
        # Step 4: Store in Milvus
        ids = milvus_client.insert(
            texts=all_texts,
            embeddings=embeddings,
            metadata=all_metadata
        )
        
        return [
            TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "urls_scraped": len(scrape_results),
                    "chunks_created": len(all_texts),
                    "embeddings_stored": len(ids),
                    "collection": collection_name
                }, indent=2)
            )
        ]
    
    elif name == "semantic_search":
        query = arguments.get("query")
        top_k = arguments.get("top_k", 5)
        collection_name = arguments.get("collection_name", settings.milvus_collection)
        
        # Initialize Milvus client if needed
        if milvus_client is None:
            milvus_client = MilvusClient(
                host=settings.milvus_host,
                port=settings.milvus_port,
                collection_name=collection_name
            )
            milvus_client.connect()
        
        # Generate query embedding
        query_embedding = embedding_client.generate_embedding(query)
        
        # Search Milvus
        results = milvus_client.search(
            query_embeddings=[query_embedding],
            top_k=top_k
        )
        
        return [
            TextContent(
                type="text",
                text=json.dumps({
                    "query": query,
                    "results": results,
                    "total_results": len(results)
                }, indent=2)
            )
        ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Smart text chunking with sentence/word boundary detection
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position
        end = start + chunk_size
        
        if end >= text_length:
            # Last chunk - take everything remaining
            chunks.append(text[start:].strip())
            break
        
        # Try to find a good breaking point (sentence boundary)
        chunk = text[start:end]
        
        # Look for sentence endings near the end
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        best_break = -1
        
        for ending in sentence_endings:
            pos = chunk.rfind(ending)
            if pos > chunk_size * 0.7:  # Only break if we're at least 70% through
                best_break = max(best_break, pos + len(ending))
        
        if best_break > 0:
            # Found a good sentence boundary
            chunks.append(text[start:start + best_break].strip())
            start = start + best_break - overlap
        else:
            # No sentence boundary, try word boundary
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.7:
                chunks.append(text[start:start + last_space].strip())
                start = start + last_space - overlap
            else:
                # No good boundary, just split at chunk_size
                chunks.append(chunk.strip())
                start = end - overlap
        
        # Ensure we make progress
        if start < 0:
            start = 0
    
    return [c for c in chunks if c]  # Filter out empty chunks
