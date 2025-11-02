"""
Test scrape-and-embed functionality
"""

import sys
import requests

def test_scrape_and_embed():
    """Test scraping URLs and storing in vector database"""
    
    print("\n" + "=" * 60)
    print("Testing Scrape-and-Embed to Milvus")
    print("=" * 60)
    
    # Test URLs (use example.com for safe testing)
    urls = [
        "https://example.com",
        "https://www.iana.org/help/example-domains"
    ]
    
    # Prepare request
    data = {
        "urls": urls,
        "extract_markdown": True,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "max_concurrent": 2,
        "auto_init": True
    }
    
    print(f"\n📥 Scraping {len(urls)} URLs...")
    for url in urls:
        print(f"   - {url}")
    
    try:
        # Make request
        response = requests.post(
            "http://localhost:8000/embeddings/scrape-and-embed",
            json=data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success!")
            print(f"   URLs scraped: {result['urls_scraped']}")
            print(f"   URLs failed: {result['urls_failed']}")
            print(f"   Total chunks: {result['total_chunks']}")
            print(f"   Vectors inserted: {result['vectors_inserted']}")
            print(f"   Embedding dimension: {result['embedding_dimension']}")
            print(f"   Collection: {result['collection']}")
            print(f"\n   Details:")
            print(f"   - Chunk size: {result['details']['chunk_size']}")
            print(f"   - Avg chunk size: {result['details']['avg_chunk_size']}")
            
            # Now test search
            print("\n🔍 Testing semantic search...")
            search_response = requests.post(
                "http://localhost:8000/embeddings/vectors/search",
                json={
                    "query": "What is an example domain?",
                    "top_k": 3
                }
            )
            
            if search_response.status_code == 200:
                search_result = search_response.json()
                print(f"\n   Found {len(search_result['results'])} results:")
                for i, r in enumerate(search_result['results'][:3], 1):
                    print(f"\n   Result {i}:")
                    print(f"   - Score: {r['score']:.4f}")
                    print(f"   - URL: {r['metadata'].get('url', 'N/A')}")
                    print(f"   - Chunk: {r['metadata'].get('chunk_index', 0)}/{r['metadata'].get('total_chunks', 0)}")
                    print(f"   - Text preview: {r['text'][:100]}...")
            else:
                print(f"   Search failed: {search_response.status_code}")
                print(f"   {search_response.text}")
            
        else:
            print(f"\n❌ Failed: {response.status_code}")
            print(f"   {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n⚠️  Server not running!")
        print("   Start the server with: python main.py")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SCRAPE-AND-EMBED TEST")
    print("=" * 60)
    print("\nThis test will:")
    print("1. Scrape content from URLs")
    print("2. Chunk the text into smaller pieces")
    print("3. Generate embeddings using BGE-M3")
    print("4. Store in Milvus vector database")
    print("5. Search for similar content")
    
    success = test_scrape_and_embed()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETE!")
        print("=" * 60)
        print("\nYou can now:")
        print("- Search: POST /embeddings/vectors/search")
        print("- View status: GET /embeddings/vectors/status")
        print("- Add more URLs: POST /embeddings/scrape-and-embed")
    
    sys.exit(0 if success else 1)
