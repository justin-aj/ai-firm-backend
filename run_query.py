import asyncio
import sys
import os

# Ensure Python can find the 'routes' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import directly from your file
from routes.intelligent_query import intelligent_ask, IntelligentQueryRequest, preload_vlm_endpoint, PreloadVLMRequest

async def main():
    print("--- 1. Preloading VLM (Optional) ---")
    # We wrap the request in the Pydantic model your code expects
    vlm_req = PreloadVLMRequest(tensor_parallel_size=1)
    await preload_vlm_endpoint(vlm_req)

    print("\n--- 2. Running Intelligent Query ---")
    # Construct the request object
    query_payload = IntelligentQueryRequest(
        question="What is the use of Triton in ML?",
        temperature=0.5,
        include_image_search=True
    )

    try:
        # Call the function directly. No HTTP, no uvicorn.
        result = await intelligent_ask(query_payload)
        
        # Print the result nicely
        print(f"Success: {result.success}")
        print(f"Answer: {result.llm_answer}")
        print(f"Topics: {result.topics}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Async functions must be run inside an event loop
    asyncio.run(main())