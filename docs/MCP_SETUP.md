# Model Context Protocol (MCP) Setup Guide

## What is MCP?

Model Context Protocol (MCP) allows AI assistants like Claude Desktop to directly use your tools (Google Search, LM Studio) without going through REST APIs. It's a standardized way for AI to interact with external systems.

**Sequential Thinking Tool:** Your MCP server now includes a dedicated sequential thinking tool that allows Claude to explicitly break down complex tasks into reasoning steps, store them, and track progress.

## Architecture Overview

```
┌─────────────────────┐
│  Claude Desktop     │
│  (MCP Client)       │
└──────────┬──────────┘
           │ MCP Protocol (stdio)
           │
┌──────────▼──────────┐
│  run_mcp_server.py  │
│  (MCP Server)       │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   mcp_server.py     │
│   (MCP Tools)       │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────┐  ┌────▼────┐
│ Google │  │   LM    │
│ Search │  │ Studio  │
└────────┘  └─────────┘
```

## Available MCP Tools

### 1. Search Tools
- **google_search** - Search the web and get detailed results (title, URL, snippet)
- **google_search_urls_only** - Get only URLs from search results
- **google_image_search** - Search for images

### 2. LLM Tools
- **lm_studio_chat** - Chat with your local LLM
- **lm_studio_completion** - Text completion with your local LLM

### 3. Web Scraping Tools
- **scrape_url** - Scrape content from a single URL using Crawl4AI
- **scrape_urls_batch** - Scrape multiple URLs in parallel with Dask

### 4. Embedding & Vector Search Tools
- **generate_embedding** - Generate BGE-M3 embeddings (1024-dim) for text
- **scrape_and_embed** - Complete RAG pipeline: scrape → chunk → embed → store in Milvus
- **semantic_search** - Search Milvus vector database for semantically similar content

### 5. Advanced Reasoning
- **sequential_thinking** - Enable step-by-step reasoning for complex tasks

## Setup Instructions

### 1. Install Dependencies

Already done! MCP is installed in your virtual environment.

### 2. Test MCP Server

Run the MCP server standalone:

```bash
python run_mcp_server.py
```

### 3. Configure Claude Desktop

**Location of config file:**
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

**Add this configuration:**

```json
{
  "mcpServers": {
    "ai-firm-backend": {
      "command": "python",
      "args": [
        "C:/Users/ajinf/2025/fall/webdev/ai-firm-backend/run_mcp_server.py"
      ],
      "env": {
        "GOOGLE_API_KEY": "YOUR_API_KEY",
        "GOOGLE_CX": "YOUR_CX_ID",
        "LM_STUDIO_BASE_URL": "http://127.0.0.1:1234/v1",
        "MILVUS_HOST": "localhost",
        "MILVUS_PORT": "19530"
      }
    }
  }
}
```

**Important:** Replace `YOUR_API_KEY` and `YOUR_CX_ID` with your actual credentials.

### 4. Restart Claude Desktop

After editing the config, restart Claude Desktop to load the MCP server.

### 5. Verify Connection

In Claude Desktop, you should see your tools available. You can ask:
- "Search the web for Python tutorials"
- "Find images of mountains"
- "Use my local LM to explain quantum computing"

## Testing with MCP Inspector

You can test your MCP server with the official inspector:

```bash
npx @modelcontextprotocol/inspector python run_mcp_server.py
```

This opens a web interface to test your MCP tools.

## Usage Examples

### Example 1: Web Search
**User:** "Search for the latest AI news"

**Claude uses:** `google_search` tool
```json
{
  "query": "latest AI news 2025",
  "num_results": 10
}
```

### Example 2: Get URLs Only
**User:** "Find me 5 URLs about FastAPI tutorials"

**Claude uses:** `google_search_urls_only` tool
```json
{
  "query": "FastAPI tutorials",
  "num_results": 5
}
```

### Example 3: Chat with Local LLM
**User:** "Ask my local model to explain recursion"

**Claude uses:** `lm_studio_chat` tool
```json
{
  "messages": [
    {"role": "user", "content": "Explain recursion in simple terms"}
  ],
  "temperature": 0.7
}
```

### Example 4: Scrape and Embed for RAG
**User:** "Scrape these URLs and make them searchable"

**Claude uses:** `scrape_and_embed` tool
```json
{
  "urls": [
    "https://example.com/article1",
    "https://example.com/article2"
  ],
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

### Example 5: Semantic Search
**User:** "Search my stored content for information about Python"

**Claude uses:** `semantic_search` tool
```json
{
  "query": "Python programming best practices",
  "top_k": 5
}
```

### Example 6: Multi-Step RAG Workflow
**User:** "Scrape these AI news sites, then search for articles about GPT-4"

**Claude performs:**
1. Uses `scrape_and_embed` to ingest content
2. Uses `semantic_search` to find relevant articles
3. Uses `lm_studio_chat` to summarize findings

## Dual Architecture Benefits

Your app now supports TWO ways to access the same tools:

### 1. MCP Protocol (for AI assistants)
- Direct tool access from Claude Desktop
- Type-safe with JSON Schema
- Standardized protocol

### 2. REST API (for web/mobile apps)
- HTTP endpoints remain available
- Can be called from any HTTP client
- Postman, curl, frontend apps

## Troubleshooting

### MCP Server won't start
- Check that your `.env` file exists with correct credentials
- Verify Python path in Claude config is correct
- Check terminal for error messages

### Tools not showing in Claude
- Restart Claude Desktop completely
- Verify config file syntax (valid JSON)
- Check Claude logs for errors

### Google Search returns errors
- Verify `GOOGLE_API_KEY` and `GOOGLE_CX` in config
- Test with the test script: `python test_google_search.py`

### LM Studio connection fails
- Make sure LM Studio is running
- Verify it's on port 1234
- Load a model in LM Studio

### Milvus connection fails
- Make sure Docker is running
- Verify Milvus container is running: `docker ps`
- Start Milvus: `docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest`
- Check Milvus logs: `docker logs milvus`

### Embedding/Scraping errors
- First run may take time to download BGE-M3 model (~2.3GB)
- Check internet connection for Crawl4AI scraping
- Verify sufficient disk space for embeddings

## Security Notes

- Your credentials in `claude_desktop_config.json` are stored locally
- MCP server runs on your machine, not in the cloud
- No data is sent to external servers (except Google API and LM Studio)
- The stdio transport is secure and local-only

## Next Steps

Once MCP is working:
1. ✅ Test each tool individually
2. ✅ Try combining tools (search + scrape + embed + semantic search)
3. ✅ Build RAG workflows (scrape → embed → search → LLM answer)
4. ✅ Keep REST API for web clients
5. ✅ Monitor Milvus storage and performance

## Support

For MCP-specific issues:
- MCP Docs: https://modelcontextprotocol.io
- GitHub: https://github.com/modelcontextprotocol

For this project:
- Check `SECURITY.md` for security practices
- See `DEPLOYMENT_CHECKLIST.md` before deploying
