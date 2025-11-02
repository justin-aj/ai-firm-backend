# Embedding & Vector Database Integration

This document explains how to set up and use the embedding and vector database features.

## Overview

The AI Firm Backend now includes:
- **Snowflake Arctic Embed** - High-quality text embeddings
- **Milvus** - Vector database for similarity search
- **RAG-ready** - Build semantic search and retrieval systems

## Architecture

```
┌─────────────┐
│  Text Input │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  EmbeddingClient        │
│  (Arctic Embed Model)   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Vector (768-dim)       │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  MilvusClient           │
│  (Vector Database)      │
└─────────────────────────┘
```

## Setup

### 1. Install Dependencies

```bash
pip install torch transformers pymilvus
```

Or use the updated `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Start Milvus

#### Option A: Docker (Recommended)

```bash
# Download docker-compose.yml from Milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start Milvus
docker-compose up -d
```

#### Option B: Milvus Lite (Development)

```bash
pip install milvus
python -m milvus
```

### 3. Configure Environment

Add to `.env`:

```bash
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=ai_firm_vectors
```

### 4. Start the Server

```bash
python main.py
```

## API Endpoints

### Embedding Endpoints

#### Generate Single Embedding

```bash
POST /embeddings/embed
```

**Request:**
```json
{
  "text": "What is artificial intelligence?",
  "max_length": 512
}
```

**Response:**
```json
{
  "text": "What is artificial intelligence?",
  "embedding": [0.123, -0.456, ...],  // 768-dimensional vector
  "dimension": 768
}
```

#### Generate Batch Embeddings

```bash
POST /embeddings/embed/batch
```

**Request:**
```json
{
  "texts": [
    "First document",
    "Second document",
    "Third document"
  ],
  "max_length": 512,
  "batch_size": 32
}
```

#### Calculate Similarity

```bash
POST /embeddings/embed/similarity
```

**Request:**
```json
{
  "text1": "Machine learning is awesome",
  "text2": "AI is incredible"
}
```

**Response:**
```json
{
  "text1": "Machine learning is awesome",
  "text2": "AI is incredible",
  "similarity": 0.87
}
```

### Vector Database Endpoints

#### Initialize Collection

```bash
POST /embeddings/vectors/init
```

Creates a Milvus collection with the correct schema and index.

#### Insert Vectors

```bash
POST /embeddings/vectors/insert
```

**Request:**
```json
{
  "texts": [
    "Python is a programming language",
    "JavaScript is used for web development"
  ],
  "metadata": [
    {"source": "wikipedia", "category": "programming"},
    {"source": "docs", "category": "web"}
  ],
  "auto_embed": true
}
```

#### Search Vectors

```bash
POST /embeddings/vectors/search
```

**Request:**
```json
{
  "query": "What programming languages exist?",
  "top_k": 5,
  "metric_type": "L2"
}
```

**Response:**
```json
{
  "query": "What programming languages exist?",
  "top_k": 5,
  "results": [
    {
      "id": 1,
      "score": 0.123,
      "text": "Python is a programming language",
      "metadata": {"source": "wikipedia", "category": "programming"}
    }
  ]
}
```

#### Get Status

```bash
GET /embeddings/vectors/status
```

Returns collection stats and embedding model info.

## Use Cases

### 1. Semantic Search over Scraped Content

Combine web scraping with vector search:

```python
# 1. Scrape content
POST /scrape/urls
{
  "urls": ["https://example.com/article1", "https://example.com/article2"],
  "extract_markdown": true
}

# 2. Insert into Milvus
POST /embeddings/vectors/insert
{
  "texts": ["<scraped content 1>", "<scraped content 2>"],
  "metadata": [{"url": "..."}, {"url": "..."}],
  "auto_embed": true
}

# 3. Search
POST /embeddings/vectors/search
{
  "query": "What is the main topic?",
  "top_k": 3
}
```

### 2. RAG with LM Studio

Build a RAG (Retrieval-Augmented Generation) system:

```python
# 1. Search for relevant context
POST /embeddings/vectors/search
{
  "query": "How do I use Python decorators?",
  "top_k": 3
}

# 2. Use results as context in LM Studio chat
POST /lm-studio/chat
{
  "messages": [
    {"role": "system", "content": "Use this context: <search results>"},
    {"role": "user", "content": "How do I use Python decorators?"}
  ]
}
```

### 3. Document Similarity

Find similar documents:

```python
# Insert documents
POST /embeddings/vectors/insert
{
  "texts": ["doc1", "doc2", "doc3"],
  "auto_embed": true
}

# Find similar
POST /embeddings/embed/similarity
{
  "text1": "new document",
  "text2": "doc1"
}
```

## Performance Tips

### GPU Acceleration

The embedding model automatically uses GPU if available:

```python
# Check GPU availability
import torch
print(torch.cuda.is_available())
```

### Batch Processing

For large datasets, use batch processing:

```python
POST /embeddings/embed/batch
{
  "texts": [...],  # Up to 1000s of texts
  "batch_size": 32  # Adjust based on GPU memory
}
```

### Index Types

Milvus supports different index types:

- **IVF_FLAT** - Good balance (default)
- **HNSW** - Fastest search, more memory
- **IVF_SQ8** - Compressed, less memory

## Client Usage

### Python Client Example

```python
import requests

# Initialize collection
requests.post("http://localhost:8000/embeddings/vectors/init")

# Insert data
requests.post("http://localhost:8000/embeddings/vectors/insert", json={
    "texts": ["Machine learning", "Deep learning", "AI"],
    "auto_embed": True
})

# Search
response = requests.post("http://localhost:8000/embeddings/vectors/search", json={
    "query": "What is ML?",
    "top_k": 3
})

results = response.json()["results"]
for result in results:
    print(f"Score: {result['score']}, Text: {result['text']}")
```

## Troubleshooting

### Milvus Connection Error

```bash
# Check if Milvus is running
docker ps | grep milvus

# Check logs
docker logs milvus-standalone
```

### Out of Memory (GPU)

Reduce batch size:

```python
{
  "batch_size": 16  # Reduce from 32
}
```

### Slow Embedding Generation

First call loads the model (takes 5-10 seconds). Subsequent calls are fast (~100ms per batch).

## Model Information

**Snowflake Arctic Embed M v2.0**
- **Dimension**: 768
- **Max Length**: 512 tokens
- **Performance**: State-of-the-art retrieval quality
- **License**: Apache 2.0
- **Size**: ~500MB

## Next Steps

1. **Build a RAG chatbot** - Combine scraper + embeddings + LM Studio
2. **Semantic code search** - Index your codebase for AI-powered search
3. **Content recommendation** - Find similar articles, products, etc.
4. **Long-term memory** - Store conversation history with embeddings

See the full API docs at: http://localhost:8000/docs
