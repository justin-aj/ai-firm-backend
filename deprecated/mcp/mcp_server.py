
"""
Archived MCP server (`mcp_server.py`).

This file is a historical copy of the MCP server implementation.
Do not run code from `deprecated/` — it's kept for reference only.

The contents below are a snapshot of the MCP server that used to run
in the active tree. It was archived here so maintainers can review
the implementation without it being importable from the active code.
"""

from typing import Any, Sequence
import json

# NOTE: This archived server referenced many project-specific clients
# (embedding_client, search_client, lm_client, scraper_client, MilvusClient,
# settings, etc.). Those dependencies are not included here — this file is
# intended for historical reference only.

# The original server exposed a number of MCP tools. Below are the
# implementations inlined for reference. They assume the MCP framework's
# decorators and helper types such as `TextContent` and `app.call_tool()`.

# --- Begin archived MCP server implementation (historical) ---

from modelcontextprotocol.server import MCPApp  # type: ignore
from modelcontextprotocol.types import TextContent  # type: ignore

# Project clients (archived references)
from clients.embedding_client import EmbeddingClient  # type: ignore
from clients.lm_studio_client import LMStudioClient  # type: ignore
from clients.scraper_client import ScraperClient  # type: ignore
from clients.search_client import SearchClient  # type: ignore
from clients.milvus_client import MilvusClient  # type: ignore
from config import settings  # type: ignore

app = MCPApp()

# Global clients (initialized lazily in original implementation)
embedding_client: EmbeddingClient | None = None
lm_client: LMStudioClient | None = None
scraper_client: ScraperClient | None = None
search_client: SearchClient | None = None
milvus_client: MilvusClient | None = None

# Simple in-memory sequential thinking session storage used by tools
thinking_sessions: dict[str, list[dict[str, Any]]] = {}


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
	"""Handle MCP tool calls (archived implementation).

	This function implemented many tools the MCP server supported, such
	as `sequential_thinking`, various Google search wrappers, LM Studio
	chat/completion, scraping helpers, and embedding flows that saved
	vectors into Milvus. The implementation below is preserved for
	reference but kept non-functional without the rest of the project
	environment.
	"""
	global milvus_client, embedding_client, lm_client, scraper_client, search_client

	# Lazily initialize clients when referenced (historical pattern)
	if embedding_client is None:
		embedding_client = EmbeddingClient()
	if lm_client is None:
		lm_client = LMStudioClient()
	if scraper_client is None:
		scraper_client = ScraperClient()
	if search_client is None:
		search_client = SearchClient()

	if name == "sequential_thinking":
		thought = arguments.get("thought")
		thought_number = arguments.get("thoughtNumber")
		total_thoughts = arguments.get("totalThoughts")
		next_thought_needed = arguments.get("nextThoughtNeeded")

		# Store the thinking step
		session_key = arguments.get("session", "current_session")
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
		import torch  # type: ignore
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

# --- End archived MCP server implementation ---

# Reminder: This file is for historical/reference purposes only. Do not import
# it into running systems; the active MCP entrypoints have been removed from
# the main codebase and replaced with deprecation shims.
