"""
Test vLLM client with TinyLlama (Direct API)
Verifies vLLM direct API is working correctly
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.vllm_client import VLLMClient


def test_initialization():
    """Test model initialization"""
    print("\n" + "=" * 80)
    print("1. Testing Model Initialization")
    print("=" * 80)
    
    try:
        print("\nLoading TinyLlama-1.1B with vLLM...")
        print("(This may take 10-20 seconds on first run)")
        
        vllm = VLLMClient(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            gpu_memory_utilization=0.9
        )
        
        print("✅ vLLM model loaded successfully")
        return vllm
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def test_chat_completion(vllm: VLLMClient):
    """Test chat completion"""
    print("\n" + "=" * 80)
    print("2. Testing Chat Completion")
    print("=" * 80)
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is artificial intelligence? Answer in 2 sentences."}
    ]
    
    print(f"\nPrompt: {messages[-1]['content']}")
    print("\nGenerating response...")
    
    try:
        response = vllm.chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"\n✅ Response:")
        print(f"   {response}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def test_text_completion(vllm: VLLMClient):
    """Test text completion"""
    print("\n" + "=" * 80)
    print("3. Testing Text Completion")
    print("=" * 80)
    
    prompt = "The three laws of robotics are:"
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating completion...")
    
    try:
        response = vllm.completion(
            prompt=prompt,
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"\n✅ Completion:")
        print(f"   {response}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def test_topic_extraction(vllm: VLLMClient):
    """Test topic extraction (RAG use case)"""
    print("\n" + "=" * 80)
    print("4. Testing Topic Extraction (RAG Use Case)")
    print("=" * 80)
    
    question = "What are the best practices for building scalable microservices with Docker and Kubernetes?"
    
    messages = [
        {"role": "system", "content": "Extract 3-5 key topics from the user's question. Return only comma-separated topics, no explanation."},
        {"role": "user", "content": question}
    ]
    
    print(f"\nQuestion: {question}")
    print("\nExtracting topics...")
    
    try:
        response = vllm.chat_completion(
            messages=messages,
            temperature=0.3,  # Lower temperature for extraction
            max_tokens=50
        )
        
        print(f"\n✅ Extracted topics:")
        print(f"   {response}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 18 + "vLLM Direct API Test Suite" + " " * 33 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Initialize model
    vllm = test_initialization()
    if not vllm:
        print("\n❌ Cannot continue without model. Exiting.")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(("Chat Completion", test_chat_completion(vllm)))
    results.append(("Text Completion", test_text_completion(vllm)))
    results.append(("Topic Extraction", test_topic_extraction(vllm)))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! vLLM is working correctly.")
    else:
        print("\n⚠️  Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

