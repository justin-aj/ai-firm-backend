from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lm_studio_client import LMStudioClient
from models import ChatCompletionRequest, CompletionRequest

app = FastAPI(
    title="AI Firm Backend",
    description="Backend API for AI Firm application with LM Studio integration",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LM Studio client
lm_client = LMStudioClient()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to AI Firm Backend API"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/lm-studio/models")
async def get_lm_studio_models():
    """Get available models from LM Studio"""
    return await lm_client.get_models()

@app.post("/lm-studio/chat")
async def chat_completion(request: ChatCompletionRequest):
    """Send a chat completion request to LM Studio"""
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    return await lm_client.chat_completion(
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

@app.post("/lm-studio/completion")
async def completion(request: CompletionRequest):
    """Send a text completion request to LM Studio"""
    return await lm_client.completion(
        prompt=request.prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
