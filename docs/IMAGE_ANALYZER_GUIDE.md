# Image Analyzer Integration Guide

## Overview

The Image Analyzer Client combines Google Image Search with Qwen3-VL vision-language model to intelligently search and analyze images.

**Architecture**: Single GPU optimized, expandable to multi-GPU

## Quick Start

```python
from clients.image_analyzer_client import ImageAnalyzerClient, AnalysisConfig

# Initialize (loads Qwen3-VL on GPU)
analyzer = ImageAnalyzerClient()

# Search and describe images
results = analyzer.describe_images(
    query="neural network architecture diagram",
    num_images=5
)

for result in results:
    print(f"Image: {result.image_title}")
    print(f"Analysis: {result.analysis}\n")
```

## Core Features

### 1. Image Description
Automatically describe images found via search:

```python
results = analyzer.describe_images(
    query="machine learning visualization",
    num_images=5
)
```

### 2. OCR / Text Extraction
Extract all visible text from images:

```python
results = analyzer.extract_text_from_images(
    query="python code snippet",
    num_images=5
)
```

### 3. Visual Question Answering
Ask specific questions about images:

```python
results = analyzer.answer_visual_question(
    search_query="transformer architecture diagram",
    question="What is the attention mechanism shown in this diagram?",
    num_images=5
)
```

### 4. Custom Analysis
Full control with custom questions and configuration:

```python
config = AnalysisConfig(
    num_images=10,
    image_size="xlarge",      # Prefer high-res images
    image_type="photo",        # Photos vs clipart
    temperature=0.0,           # Deterministic
    max_tokens=512,
    batch_size=5               # GPU memory consideration
)

results = analyzer.search_and_analyze(
    query="AI research visualization",
    analysis_question="Explain the concept shown in this image",
    config=config
)
```

## GPU Configuration

### Single GPU (Current Setup)

```python
analyzer = ImageAnalyzerClient(
    load_vlm=True,
    gpu_memory_utilization=0.90,  # Use 90% of GPU memory
    tensor_parallel_size=1         # Single GPU
)
```

**Memory Usage**:
- Qwen3-VL-8B: ~16GB VRAM
- Batch processing: 5 images default
- Efficient reuse: Model loaded once

### Multi-GPU Expansion (Future)

```python
# When multiple GPUs are available
analyzer = ImageAnalyzerClient(
    load_vlm=True,
    gpu_memory_utilization=0.90,
    tensor_parallel_size=2  # Use 2 GPUs
)
```

**Benefits**:
- Larger batch sizes
- Faster inference
- Handle more concurrent requests

## Workflow

```
User Query
    ↓
Google Image Search API
    ↓
Retrieve n images
    ↓
Qwen3-VL (batch inference)
    ↓
Analysis Results
```

### Batch Processing

The client automatically batches images for efficient GPU utilization:

```python
# This will process 10 images in 2 batches of 5
config = AnalysisConfig(
    num_images=10,
    batch_size=5  # GPU memory sweet spot
)

results = analyzer.search_and_analyze(
    query="data science workflow",
    analysis_question="What step in the workflow is shown?",
    config=config
)
```

## Result Format

```python
@dataclass
class ImageAnalysisResult:
    image_url: str        # Direct link to image
    image_title: str      # Title from search
    image_source: str     # Source domain
    analysis: str         # VLM analysis
    error: Optional[str]  # Error if analysis failed
```

### Summary Statistics

```python
summary = analyzer.get_summary(results)
print(summary)

# Output:
{
    "total_images": 5,
    "successful": 5,
    "failed": 0,
    "success_rate": 1.0,
    "images": [...]
}
```

## Advanced Usage

### Lazy Loading (Memory Optimization)

```python
# Don't load VLM immediately
analyzer = ImageAnalyzerClient(load_vlm=False)

# VLM loads only when needed
results = analyzer.describe_images("AI technology", 3)
```

### Error Handling

```python
results = analyzer.search_and_analyze(
    query="machine learning",
    analysis_question="What is shown?",
    config=AnalysisConfig(num_images=5)
)

# Check for errors
for result in results:
    if result.error:
        print(f"Failed to analyze {result.image_url}: {result.error}")
    else:
        print(f"Success: {result.analysis}")
```

### Custom Image Filters

```python
from clients.google_image_search_client import GoogleImageSearchClient

# For fine-grained control, use components directly
search = GoogleImageSearchClient()

# Get specific image types
images = search.search_images(
    query="data visualization",
    num_results=10,
    image_size="xxlarge",  # Very high resolution
    image_type="lineart"   # Line art diagrams
)

# Then analyze with VLM
analyzer = ImageAnalyzerClient()
# ... process images manually
```

## Performance Considerations

### Single GPU Best Practices

1. **Batch Size**: Default 5 images balances speed/memory
2. **Image Quality**: Use "large" or "xlarge" for better analysis
3. **Temperature**: 0.0 for deterministic, 0.3 for creative
4. **Max Tokens**: 512 default, increase for detailed analysis

### Optimization Tips

```python
# For fast OCR
config = AnalysisConfig(
    batch_size=10,        # More images per batch
    max_tokens=2048,      # More tokens for text
    temperature=0.0       # Deterministic
)

# For detailed analysis
config = AnalysisConfig(
    batch_size=3,         # Fewer images, more attention
    max_tokens=1024,      # Detailed descriptions
    temperature=0.3       # Slightly creative
)
```

## Integration with RAG Pipeline

### Multimodal Query Detection

```python
def handle_user_query(query: str):
    # Detect if query needs visual information
    visual_keywords = ["image", "diagram", "visualization", "chart", "photo"]
    
    if any(kw in query.lower() for kw in visual_keywords):
        # Use Image Analyzer
        analyzer = ImageAnalyzerClient()
        results = analyzer.search_and_analyze(
            query=query,
            analysis_question="Answer based on this image",
            config=AnalysisConfig(num_images=5)
        )
        return results
    else:
        # Use standard RAG
        return standard_rag_query(query)
```

### Multimodal RAG with Retrieval

```python
from clients.image_analyzer_client import ImageAnalyzerClient

analyzer = ImageAnalyzerClient()

def multimodal_rag_query(user_question: str):
    """RAG with both text and image results"""
    
    # 1. Search image analysis collection
    image_results = analyzer.search_vectordb(
        query=user_question,
        top_k=3
    )
    
    # 2. Search text collection (existing RAG)
    # text_results = text_milvus.search(...)
    
    # 3. Combine context
    context = []
    
    for r in image_results:
        context.append(f"Image ({r['image_title']}): {r['analysis']}")
        context.append(f"Source: {r['image_url']}")
    
    # 4. Generate answer with LLM
    prompt = f"Context:\n" + "\n\n".join(context)
    prompt += f"\n\nQuestion: {user_question}"
    
    # Use vLLM or your LLM client
    answer = vllm_client.chat_completion(prompt)
    
    return {
        "answer": answer,
        "image_sources": [r['image_url'] for r in image_results]
    }
```

### Store in Vector Database

```python
# After analyzing images
results = analyzer.describe_images("AI concepts", 5)

# Store in Milvus
for result in results:
    embedding = get_embedding(result.analysis)
    
    milvus_client.insert({
        "embedding": embedding,
        "text": result.analysis,
        "image_url": result.image_url,
        "image_title": result.image_title,
        "metadata": {
            "source": "image_analysis",
            "query": "AI concepts"
        }
    })
```

## Testing

### Unit Tests

```bash
pytest tests/test_image_analyzer.py -v
```

### Integration Test

```python
# Quick test
from clients.image_analyzer_client import ImageAnalyzerClient

analyzer = ImageAnalyzerClient()
results = analyzer.describe_images("test query", 2)

for r in results:
    print(f"{r.image_title}: {r.analysis}")
```

## API Reference

### ImageAnalyzerClient

#### `__init__(load_vlm, gpu_memory_utilization, tensor_parallel_size)`
Initialize the analyzer

**Parameters**:
- `load_vlm` (bool): Load VLM immediately (default: True)
- `gpu_memory_utilization` (float): GPU memory fraction 0.0-1.0 (default: 0.90)
- `tensor_parallel_size` (int): Number of GPUs (default: 1)

#### `search_and_analyze(query, analysis_question, config)`
Core method: Search and analyze images

**Returns**: List[ImageAnalysisResult]

#### `describe_images(query, num_images)`
High-level: Get image descriptions

#### `extract_text_from_images(query, num_images)`
High-level: OCR extraction

#### `answer_visual_question(search_query, question, num_images)`
High-level: Visual Q&A

#### `get_summary(results)`
Generate summary statistics

### AnalysisConfig

```python
@dataclass
class AnalysisConfig:
    num_images: int = 5
    image_size: Optional[str] = "large"
    image_type: Optional[str] = "photo"
    temperature: float = 0.0
    max_tokens: int = 512
    batch_size: int = 5
```

## Examples

### Example 1: Research Assistant

```python
analyzer = ImageAnalyzerClient()

# Find and analyze research diagrams
results = analyzer.answer_visual_question(
    search_query="transformer attention mechanism diagram",
    question="Explain the attention mechanism shown in this diagram",
    num_images=5
)

# Generate report
report = []
for i, result in enumerate(results, 1):
    report.append(f"## Diagram {i}: {result.image_title}")
    report.append(f"Source: {result.image_source}")
    report.append(f"Analysis: {result.analysis}\n")

print("\n".join(report))
```

### Example 2: Code Screenshot Analysis

```python
results = analyzer.extract_text_from_images(
    query="python error message screenshot",
    num_images=3
)

for result in results:
    print(f"Extracted error message:")
    print(result.analysis)
    print("-" * 50)
```

### Example 3: Comparative Analysis

```python
# Analyze different architecture diagrams
architectures = ["ResNet", "BERT", "GPT", "Diffusion"]

all_results = []
for arch in architectures:
    results = analyzer.answer_visual_question(
        search_query=f"{arch} architecture diagram",
        question="What are the key components of this architecture?",
        num_images=2
    )
    all_results.extend(results)

# Compare
for result in all_results:
    print(f"{result.image_title}:")
    print(result.analysis)
    print()
```

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
config = AnalysisConfig(batch_size=2)

# Or reduce GPU memory
analyzer = ImageAnalyzerClient(gpu_memory_utilization=0.80)
```

### Slow Inference

```python
# Increase batch size (if memory allows)
config = AnalysisConfig(batch_size=10)

# Reduce max_tokens
config = AnalysisConfig(max_tokens=256)
```

### No Images Found

```python
# Try different search terms
# Check Google API quotas
# Verify .env has GOOGLE_API_KEY and GOOGLE_CSE_ID
```

## Future Enhancements

- [ ] Multi-GPU tensor parallelism
- [ ] Video analysis workflow
- [ ] Real-time streaming analysis
- [ ] Caching frequently analyzed images
- [ ] Custom VLM models (Llama-Vision, Gemma-VL)
