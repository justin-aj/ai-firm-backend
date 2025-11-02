"""
Quick test for embedding and Milvus integration
"""

import sys
sys.path.append('.')

from clients.embedding_client import EmbeddingClient
from clients.milvus_client import MilvusClient

def test_embeddings():
    """Test embedding generation"""
    print("=" * 60)
    print("Testing Embedding Generation")
    print("=" * 60)
    
    client = EmbeddingClient()
    
    # Single embedding
    print("\n1. Generating single embedding...")
    text = "What is artificial intelligence?"
    embedding = client.generate_embedding(text)
    print(f"   Text: {text}")
    print(f"   Dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Batch embeddings
    print("\n2. Generating batch embeddings...")
    texts = [
        "Machine learning is awesome",
        "Deep learning uses neural networks",
        "AI is transforming the world"
    ]
    embeddings = client.generate_embeddings(texts)
    print(f"   Generated {len(embeddings)} embeddings")
    print(f"   Shape: {embeddings.shape}")
    
    # Similarity
    print("\n3. Calculating similarity...")
    text1 = "Python is a programming language"
    text2 = "Java is a programming language"
    similarity = client.similarity(text1, text2)
    print(f"   Text 1: {text1}")
    print(f"   Text 2: {text2}")
    print(f"   Similarity: {similarity:.4f}")
    
    print("\n✅ Embedding tests passed!")
    return True


def test_milvus_basic():
    """Test basic Milvus operations (without actual Milvus server)"""
    print("\n" + "=" * 60)
    print("Testing Milvus Client (Offline)")
    print("=" * 60)
    
    client = MilvusClient(
        host="localhost",
        port="19530",
        collection_name="test_collection"
    )
    
    print("\n✅ Milvus client created successfully!")
    print("   Note: To test full functionality, start Milvus server:")
    print("   docker run -d -p 19530:19530 milvusdb/milvus:latest")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("AI FIRM BACKEND - EMBEDDING & MILVUS TESTS")
    print("=" * 60)
    
    try:
        # Test embeddings
        test_embeddings()
        
        # Test Milvus client creation
        test_milvus_basic()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start Milvus: docker-compose up -d")
        print("2. Run the server: python main.py")
        print("3. Visit: http://localhost:8000/docs")
        print("4. Try the /embeddings endpoints!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
