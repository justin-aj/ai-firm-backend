# Scrape-and-Embed Integration

## Overview

The `/embeddings/scrape-and-embed` endpoint combines web scraping, embedding generation, and vector storage into a single powerful workflow.

## How It Works

```
URLs → Web Scraper → Text Chunking → Embeddings (BGE-M3) → Milvus Storage
```

### Step-by-Step Process:

1. **Scrape URLs** - Extract content using Crawl4AI
2. **Chunk Text** - Split into manageable pieces (with overlap)
3. **Generate Embeddings** - Create 1024-dim vectors using BGE-M3
4. **Store in Milvus** - Save with metadata (URL, chunk index, etc.)

## API Usage

### Scrape and Embed URLs

```bash
POST /embeddings/scrape-and-embed
```

**Request:**
```json
{
  "urls": [
    "https://example.com/article1",
    "https://example.com/article2"
  ],
  "extract_markdown": true,
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "max_concurrent": 5,
  "auto_init": true
}
```

**Response:**
```json
{
  "status": "success",
  "urls_scraped": 2,
  "urls_failed": 0,
  "total_chunks": 15,
  "vectors_inserted": 15,
  "embedding_dimension": 1024,
  "collection": "ai_firm_vectors",
  "details": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "avg_chunk_size": 856
  }
}
```

### Search the Scraped Content

```bash
POST /embeddings/vectors/search
```

**Request:**
```json
{
  "query": "How do I install Python?",
  "top_k": 5,
  "metric_type": "L2"
}
```

**Response:**
```json
{
  "query": "How do I install Python?",
  "top_k": 5,
  "results": [
    {
      "id": 12345,
      "score": 0.234,
      "text": "Python can be installed from python.org...",
      "metadata": {
        "url": "https://example.com/python-guide",
        "chunk_index": 2,
        "total_chunks": 8,
        "chunk_size": 943,
        "source": "web_scraper"
      }
    }
  ]
}
```

## Configuration

### Chunking Parameters

- **`chunk_size`** (100-5000): Characters per chunk
  - Smaller = More precise search, more vectors
  - Larger = More context, fewer vectors
  - Recommended: 1000

- **`chunk_overlap`** (0-1000): Overlap between chunks
  - Prevents breaking sentences/paragraphs
  - Recommended: 200 (20% of chunk_size)

### Scraping Parameters

- **`extract_markdown`**: Clean text extraction (recommended: true)
- **`extract_html`**: Raw HTML (for special cases)
- **`max_concurrent`**: Parallel requests (1-20)

## Metadata Stored

Each vector includes:

```python
{
  "url": "https://...",           # Source URL
  "chunk_index": 2,               # Position in document (0-based)
  "total_chunks": 8,              # Total chunks from this URL
  "chunk_size": 943,              # Actual chunk size
  "source": "web_scraper"         # Always "web_scraper"
}
```

## Use Cases

### 1. Build a Knowledge Base

```python
# Scrape documentation
POST /embeddings/scrape-and-embed
{
  "urls": [
    "https://docs.python.org/3/tutorial/",
    "https://docs.python.org/3/library/",
    "https://docs.python.org/3/reference/"
  ],
  "chunk_size": 1500,
  "chunk_overlap": 300
}

# Search when needed
POST /embeddings/vectors/search
{
  "query": "How to use decorators?",
  "top_k": 3
}
```

### 2. RAG Chatbot with LM Studio

```python
# 1. Build knowledge base
POST /embeddings/scrape-and-embed
{ "urls": ["..."] }

# 2. Search for context
search_result = POST /embeddings/vectors/search
{ "query": "user question" }

# 3. Use in chat
POST /lm-studio/chat
{
  "messages": [
    {
      "role": "system",
      "content": f"Context: {search_result['results']}"
    },
    {
      "role": "user", 
      "content": "user question"
    }
  ]
}
```

### 3. Content Aggregation

```python
# Scrape multiple sources
POST /embeddings/scrape-and-embed
{
  "urls": [
    "https://news-site-1.com/article",
    "https://news-site-2.com/article",
    "https://blog.com/post"
  ]
}

# Find all mentions of a topic
POST /embeddings/vectors/search
{
  "query": "climate change initiatives",
  "top_k": 20
}
```

## Performance Tips

### Optimal Chunk Size

| Content Type | Chunk Size | Overlap |
|--------------|------------|---------|
| News articles | 500-800 | 100 |
| Documentation | 1000-1500 | 200-300 |
| Books/Papers | 1500-2000 | 300-400 |
| Short posts | 300-500 | 50-100 |

### Batch Processing

For many URLs:
```python
# Process in batches of 10-20
for batch in url_batches:
    POST /embeddings/scrape-and-embed
    { "urls": batch, "max_concurrent": 10 }
```

### Memory Management

- Each chunk ~1KB text = ~1KB metadata + 4KB embedding
- 1000 chunks ≈ 6 MB in memory
- 1M chunks ≈ 6 GB in Milvus

## Error Handling

The endpoint is resilient:
- Failed scrapes are logged but don't stop the process
- Returns `urls_failed` count
- Only stores successfully scraped content

**Example:**
```json
{
  "urls_scraped": 10,
  "urls_failed": 2,  // 2 URLs failed to scrape
  "total_chunks": 87,
  "vectors_inserted": 87
}
```

## Testing

Run the test script:
```bash
python tests/test_scrape_embed.py
```

Or use curl:
```bash
curl -X POST http://localhost:8000/embeddings/scrape-and-embed \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com"],
    "chunk_size": 500,
    "auto_init": true
  }'
```

## Complete Workflow Example

```python
# 1. Initialize (auto-init does this for you)
POST /embeddings/vectors/init

# 2. Scrape and embed
POST /embeddings/scrape-and-embed
{
  "urls": ["https://example.com/docs"],
  "chunk_size": 1000,
  "auto_init": true
}

# 3. Check status
GET /embeddings/vectors/status

# 4. Search
POST /embeddings/vectors/search
{
  "query": "your question",
  "top_k": 5
}

# 5. Use results in LLM
POST /lm-studio/chat
{
  "messages": [...include search results...]
}
```

## Next Steps

- Scrape your docs: Add URLs to your knowledge base
- Integrate with LM Studio: Build RAG chatbot
- Schedule updates: Re-scrape URLs periodically
- Monitor storage: Check /vectors/status

See full API docs: http://localhost:8000/docs
