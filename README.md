# AI Firm Backend

A FastAPI backend application for AI Firm with LM Studio integration.

## Features

- FastAPI REST API
- LM Studio local LLM integration
- Chat completions endpoint
- Text completions endpoint
- CORS enabled
- Environment-based configuration

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
copy .env.example .env
```

Edit `.env` to configure LM Studio settings if needed.

4. **Start LM Studio**:
   - Open LM Studio
   - Load a model
   - Start the local server (default port: 1234)

5. Run the application:
```bash
python main.py
```

Or use uvicorn with auto-reload:
```bash
uvicorn main:app --reload
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

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
- LM Studio running locally
- Dependencies listed in requirements.txt
