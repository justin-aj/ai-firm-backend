# GPT-OSS-20B Integration Guide

## Overview

New dedicated module for the **GPT-OSS-20B-Q4_0** model running in LM Studio.

## Features

✅ **Chat Interface** - Conversational with history  
✅ **Simple Q&A** - One-off questions  
✅ **Text Completion** - Code/text generation  
✅ **History Management** - Save/load/clear conversations  

## Quick Start

### 1. Load Model in LM Studio

1. Open LM Studio
2. Load `gpt-oss-20b-Q4_0.gguf`
3. Start the server (port 1234)

### 2. Use the API

```bash
# Start the backend
python main.py

# Access Swagger UI
http://localhost:8000/docs
```

## API Endpoints

### Chat (with conversation history)

**POST** `/gpt-oss/chat`

```json
{
  "message": "What is machine learning?",
  "system_prompt": "You are a helpful AI assistant",
  "temperature": 0.7,
  "max_tokens": 2048,
  "reset_history": false
}
```

**Response:**
```json
{
  "response": "Machine learning is...",
  "conversation_length": 2,
  "model": "gpt-oss-20b-Q4_0"
}
```

### Ask (simple question, no history)

**POST** `/gpt-oss/ask`

```json
{
  "question": "Explain quantum computing",
  "temperature": 0.7,
  "max_tokens": 1024
}
```

**Response:**
```json
{
  "question": "Explain quantum computing",
  "answer": "Quantum computing is...",
  "model": "gpt-oss-20b-Q4_0"
}
```

### Complete (text/code completion)

**POST** `/gpt-oss/complete`

```json
{
  "prompt": "def fibonacci(n):\n    # Calculate",
  "temperature": 0.3,
  "max_tokens": 200
}
```

**Response:**
```json
{
  "prompt": "def fibonacci(n):\n    # Calculate",
  "completion": "    if n <= 1:\n        return n\n    ...",
  "model": "gpt-oss-20b-Q4_0"
}
```

### Get Conversation History

**GET** `/gpt-oss/history`

**Response:**
```json
{
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
  ],
  "length": 2
}
```

### Clear History

**POST** `/gpt-oss/history/clear`

**Response:**
```json
{
  "status": "success",
  "message": "Conversation history cleared"
}
```

### Set History

**POST** `/gpt-oss/history/set`

```json
{
  "history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

### Check Status

**GET** `/gpt-oss/status`

**Response:**
```json
{
  "available": true,
  "model": "gpt-oss-20b-Q4_0",
  "lm_studio_url": "http://127.0.0.1:1234/v1"
}
```

## Python Client Usage

```python
from clients.gpt_oss_client import GPTOSSClient

# Initialize
client = GPTOSSClient()

# Chat with history
result = await client.chat(
    user_input="What is Python?",
    system_prompt="You are a programming tutor",
    temperature=0.7
)
print(result["assistant_response"])

# Simple question (no history)
answer = await client.ask("Explain async/await")
print(answer)

# Text completion
completion = await client.complete(
    "def hello_world():\n    ",
    temperature=0.3
)
print(completion)

# Manage history
client.clear_history()
history = client.get_history()
```

## Use Cases

### 1. Chatbot
```python
# Multi-turn conversation
await client.chat("Hello!", system_prompt="You are friendly")
await client.chat("Tell me a joke")
await client.chat("Explain that joke")
```

### 2. Code Assistant
```python
# Code completion
code = await client.complete(
    "def binary_search(arr, target):\n    ",
    temperature=0.2
)
```

### 3. Q&A System
```python
# One-off questions
answer = await client.ask("What is REST API?")
```

### 4. Document Analysis
```python
# Set context, then ask questions
await client.chat(
    "Here is a document: ...",
    system_prompt="Analyze this document"
)
answer = await client.chat("What are the key points?")
```

## Configuration

In `.env`:
```env
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
```

## Files Created

- `clients/gpt_oss_client.py` - Client implementation
- `routes/gpt_oss.py` - API endpoints
- `tests/test_gpt_oss.py` - Test script
- `docs/GPT_OSS_GUIDE.md` - This guide

## Integration Points

The GPT-OSS client integrates with:
- ✅ REST API (via routes/gpt_oss.py)
- ✅ MCP Server (can be added as tool)
- ✅ Direct Python usage

## Next Steps

1. Start LM Studio with GPT-OSS-20B model
2. Run `python main.py`
3. Test at http://localhost:8000/docs
4. Build your application!

## Support

- Model: GPT-OSS-20B-Q4_0 GGUF
- LM Studio: https://lmstudio.ai
- Backend: FastAPI with async support
