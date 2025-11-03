# AI Firm Backend

An intelligent **Retrieval-Augmented Generation (RAG) system** that answers questions by searching the web, scraping relevant content, and using a local LLM (GPT-OSS-20B) to generate contextual answers. Features **smart topic-based caching** to avoid redundant scraping.

## 🎯 Project Idea

**Problem**: LLMs have knowledge cutoffs and can't access real-time information.

**Solution**: Our system creates a **dynamic knowledge base** that:
1. 🔍 **Analyzes** user questions to extract topics
2. 🧠 **Checks** if we already have content on similar topics (smart caching)
3. 🌐 **Searches** Google only when needed (saves API calls)
4. 📄 **Scrapes** web content and stores it with embeddings
5. 🎯 **Retrieves** the most relevant context from our vector database
6. 💬 **Generates** answers using GPT-OSS-20B with retrieved context

### Key Innovation: **Smart Topic Caching**
- Maintains a lightweight topics database
- Compares new questions against previously explored topics
- Skips scraping when we already have relevant content
- Dramatically reduces latency and API costs

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Question                            │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    ┌────────────────┐
                    │ Topic Analyzer │ (LLM extracts topics)
                    └────────┬───────┘
                             ↓
                    ┌────────────────┐
                    │ Topics Cache   │ (Milvus: ai_firm_topics)
                    │ Check Similar? │
                    └────┬───────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
    Similar?                          Not Similar?
        │                                 │
        ↓                                 ↓
┌───────────────┐              ┌─────────────────────┐
│ Use Cached    │              │ Google Custom Search│
│ Content       │              └──────────┬──────────┘
└───────┬───────┘                         ↓
        │                       ┌─────────────────────┐
        │                       │ Crawl4AI Scraper    │
        │                       └──────────┬──────────┘
        │                                  ↓
        │                       ┌─────────────────────┐
        │                       │ BGE-M3 Embeddings   │
        │                       └──────────┬──────────┘
        │                                  ↓
        │                       ┌─────────────────────┐
        │                       │ Store in Milvus     │
        │                       │ - Full Content      │
        │                       │ - Topics Index      │
        │                       └──────────┬──────────┘
        │                                  │
        └──────────────┬───────────────────┘
                       ↓
            ┌─────────────────────┐
            │ Retrieve Top 5 Docs │ (Vector similarity search)
            └──────────┬──────────┘
                       ↓
            ┌─────────────────────┐
            │ GPT-OSS-20B         │ (Generate answer with context)
            └──────────┬──────────┘
                       ↓
            ┌─────────────────────┐
            │   Final Answer      │
            └─────────────────────┘
```

## ✨ Features

### Core RAG Pipeline
- **🤖 LLM-Based Topic Extraction** - GPT-OSS-20B analyzes questions intelligently
- **📊 Dual Vector Storage** - Separate collections for topics (fast) and content (comprehensive)
- **🔍 Google Custom Search** - Real-time web search integration
- **🕷️ Crawl4AI Scraping** - Extract markdown content from any URL
- **🧮 BGE-M3 Embeddings** - 1024-dimensional multilingual embeddings
- **💾 Milvus Vector DB** - Efficient similarity search with L2 distance
- **⚡ Smart Caching** - Topic-based deduplication (saves ~70% of scraping)

### Additional Features
- **FastAPI REST API** - Modern async endpoints
- **Model Context Protocol (MCP)** - Direct tool access for AI assistants
- **LM Studio Integration** - Local LLM support
- **Lazy Loading** - Heavy dependencies load only when needed
- **CORS Enabled** - Ready for frontend integration
- **Security Hardened** - Input validation, error handling, logging

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- LM Studio with GPT-OSS-20B model loaded
- Google Custom Search API credentials
- Milvus vector database (standalone or docker)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
```

Edit `.env` with your credentials:
```env
# Google Custom Search
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CX=your_custom_search_engine_id

# LM Studio (GPT-OSS-20B)
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1

# Milvus Vector Database
MILVUS_HOST=localhost
MILVUS_PORT=19530

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

### 3. Start Required Services

**Start Milvus:**
```bash
# Using Docker
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

**Start LM Studio:**
1. Open LM Studio
2. Load GPT-OSS-20B model (or any compatible model)
3. Start local server (port 1234)

### 4. Run the API Server

```bash
python main.py
```

Access at: **http://localhost:8000**

### 5. Test the RAG Pipeline

```bash
curl -X POST http://localhost:8000/intelligent-query/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is CUDA and how does it help with GPU programming?",
    "temperature": 0.7,
    "max_tokens": 2048
  }'
```

**First request:** Scrapes web, stores in Milvus (~15-20 seconds)  
**Subsequent similar requests:** Uses cached content (~2-3 seconds) ⚡

## 📡 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main Endpoint: `/intelligent-query/ask`

**Request:**
```json
{
  "question": "Your question here",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Response:**
```json
{
  "success": true,
  "topics": ["topic1", "topic2", "topic3"],
  "search_results": [...],
  "scraped_content": [...],
  "stored_in_milvus": true,
  "milvus_ids": [1, 2, 3, 4, 5],
  "retrieved_context": [
    {
      "id": 1,
      "score": 0.23,
      "text": "Relevant content...",
      "metadata": {"url": "...", "topics": [...]}
    }
  ],
  "llm_answer": "Based on the context, CUDA is..."
}
```

### Status Endpoint: `/intelligent-query/status`

Check if all services are operational:
```bash
curl http://localhost:8000/intelligent-query/status
```

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
