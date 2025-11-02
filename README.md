# AI Firm Backend

A FastAPI backend application for AI Firm with LM Studio integration, Google Custom Search, and **Dask-powered distributed web scraping**, featuring **Model Context Protocol (MCP)** with **Sequential Thinking** support.

## Features

- **FastAPI REST API** - Traditional HTTP endpoints
- **Model Context Protocol (MCP)** - Direct tool access for AI assistants
- **Sequential Thinking Tool** - Multi-step reasoning capabilities
- **LM Studio Integration** - Local LLM chat and completions
- **Google Custom Search API** - Web and image search
- **🆕 Crawl4AI Web Scraping** - Extract content from any URL
- **🆕 Dask Distributed Scraping** - Scale to 1000s of concurrent scrapes
- **CORS enabled** - Ready for frontend integration
- **Security hardened** - Input validation, error handling, logging

## Dual Architecture

This backend supports **two ways** to access the same tools:

### 1. MCP Protocol (for AI Assistants like Claude Desktop)
- Direct tool invocation
- Type-safe with JSON Schema
- Standardized protocol
- See `MCP_SETUP.md` for configuration

### 2. REST API (for Web/Mobile Apps)
- Traditional HTTP endpoints
- Swagger docs at `/docs`
- Compatible with any HTTP client

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
```

Edit `.env` with your credentials:
- `GOOGLE_API_KEY` - Your Google Custom Search API key
- `GOOGLE_CX` - Your Custom Search Engine ID

### 3. Option A: Run FastAPI Server (REST API)

```bash
python main.py
```

Access at: http://localhost:8000

### 4. Option B: Run MCP Server (for Claude Desktop)

```bash
python run_mcp_server.py
```

Then configure Claude Desktop - see `MCP_SETUP.md`

### 5. Start LM Studio

- Open LM Studio
- Load a model
- Start the local server (port 1234)

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## MCP Tools (for AI Assistants)

When configured with Claude Desktop, the following tools are available:

1. **google_search** - Search the web with detailed results
2. **google_search_urls_only** - Get only URLs
3. **google_image_search** - Search for images
4. **lm_studio_chat** - Chat with your local LLM
5. **lm_studio_completion** - Text completion with local LLM

See `MCP_SETUP.md` for detailed setup instructions.

## Endpoints

### General
- `GET /`: Root endpoint
- `GET /health`: Health check endpoint

### LM Studio Integration
- `GET /lm-studio/models`: Get available models from LM Studio
- `POST /lm-studio/chat`: Chat completion with LM Studio
- `POST /lm-studio/completion`: Text completion with LM Studio

## Example Requests

### Get Models
```bash
curl http://localhost:8000/lm-studio/models
```

### Chat Completion
```bash
curl -X POST http://localhost:8000/lm-studio/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7
  }'
```

### Text Completion
```bash
curl -X POST http://localhost:8000/lm-studio/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## Configuration

Edit `.env` file to configure:
- `LM_STUDIO_BASE_URL`: LM Studio API base URL (default: http://127.0.0.1:1234/v1)
- `LM_STUDIO_MODEL`: Model identifier (default: local-model)
- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)

## Requirements

- Python 3.8+
- LM Studio (running locally)
- Google Custom Search API credentials
- Dependencies listed in requirements.txt

## Documentation

- **`MCP_SETUP.md`** - Complete MCP configuration guide
- **`SECURITY.md`** - Security best practices
- **`DEPLOYMENT_CHECKLIST.md`** - Pre-deployment checklist

## Project Structure

```
ai-firm-backend/
├── main.py                    # FastAPI application entry point
├── mcp_server.py             # MCP server with tools
├── run_mcp_server.py         # MCP server runner
├── config.py                 # Configuration management
├── models.py                 # Pydantic models
├── clients/                  # External service clients
│   ├── __init__.py
│   ├── lm_studio_client.py   # LM Studio integration
│   ├── google_search_client.py # Google Search integration
│   └── web_scraper_client.py # Crawl4AI web scraper
├── routes/                   # API route handlers
│   ├── __init__.py
│   ├── core.py              # Core endpoints (health, root)
│   ├── lm_studio.py         # LM Studio endpoints
│   ├── search.py            # Google Search endpoints
│   ├── scraper.py           # Web scraping endpoints
│   └── sequential_thinking.py # Sequential thinking endpoints
├── tests/                    # Test files
│   ├── __init__.py
│   └── test_dask.py         # Dask integration tests
├── docs/                     # Documentation
│   ├── MCP_SETUP.md         # MCP configuration guide
│   ├── SECURITY.md          # Security best practices
│   ├── DEPLOYMENT_CHECKLIST.md # Deployment guide
│   ├── DASK_GUIDE.md        # Dask distributed scraping guide
│   └── MODULARIZATION.md    # Code organization notes
├── .env                      # Environment variables (local, gitignored)
├── .env.example              # Environment template
└── requirements.txt          # Python dependencies
```
