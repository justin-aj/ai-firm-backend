"""
Test GPT-OSS-20B Client
Simple tests for the GPT-OSS client
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clients.gpt_oss_client import GPTOSSClient


async def test_gpt_oss():
    """Test GPT-OSS client functionality"""
    print("\n" + "="*60)
    print("Testing GPT-OSS-20B Client")
    print("="*60 + "\n")
    
    # Initialize client
    print("1. Initializing GPT-OSS client...")
    client = GPTOSSClient()
    print(f"✅ Client initialized for model: {client.model_name}\n")
    
    # Check availability
    print("2. Checking if LM Studio is available...")
    is_available = await client.is_available()
    if is_available:
        print("✅ LM Studio is running and accessible\n")
    else:
        print("❌ LM Studio not available. Make sure it's running on port 1234\n")
        return
    
    # Test simple question
    print("3. Testing simple question (ask method)...")
    question = "What is artificial intelligence?"
    print(f"Question: {question}")
    answer = await client.ask(question, max_tokens=150)
    print(f"Answer: {answer[:200]}...\n")
    
    # Test conversation
    print("4. Testing conversation (chat method)...")
    
    # First message
    print("User: Hello! Can you help me understand Python?")
    result1 = await client.chat(
        user_input="Hello! Can you help me understand Python?",
        system_prompt="You are a helpful programming tutor.",
        max_tokens=150
    )
    print(f"Assistant: {result1['assistant_response'][:200]}...")
    print(f"Conversation length: {result1['conversation_length']} messages\n")
    
    # Second message (continues conversation)
    print("User: What are Python's main features?")
    result2 = await client.chat(
        user_input="What are Python's main features?",
        max_tokens=200
    )
    print(f"Assistant: {result2['assistant_response'][:200]}...")
    print(f"Conversation length: {result2['conversation_length']} messages\n")
    
    # Get conversation history
    print("5. Getting conversation history...")
    history = client.get_history()
    print(f"Total messages in history: {len(history)}")
    for i, msg in enumerate(history):
        role = msg['role'].capitalize()
        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"  {i+1}. {role}: {content}")
    print()
    
    # Clear history
    print("6. Clearing conversation history...")
    client.clear_history()
    history = client.get_history()
    print(f"✅ History cleared. Current length: {len(history)}\n")
    
    # Test text completion
    print("7. Testing text completion...")
    prompt = "def fibonacci(n):\n    # Calculate fibonacci number\n"
    print(f"Prompt:\n{prompt}")
    completion = await client.complete(prompt, temperature=0.3, max_tokens=200)
    print(f"Completion:\n{completion}\n")
    
    print("="*60)
    print("✅ All tests completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\nMake sure LM Studio is running with the GPT-OSS-20B model loaded!")
    print("Press Ctrl+C to cancel or wait 3 seconds to start...\n")
    
    try:
        import time
        time.sleep(3)
        asyncio.run(test_gpt_oss())
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
