"""
Test Google Image Search Client
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.google_image_search_client import GoogleImageSearchClient


def test_basic_image_search():
    """Test basic image search"""
    print("\n" + "=" * 80)
    print("1. Testing Basic Image Search")
    print("=" * 80)
    
    client = GoogleImageSearchClient()
    
    query = "artificial intelligence technology"
    print(f"\nQuery: {query}")
    print("Searching for 5 images...")
    
    try:
        images = client.search_images(query, num_results=5)
        
        if images:
            print(f"\n✅ Found {len(images)} images:")
            for i, img in enumerate(images, 1):
                print(f"\n{i}. {img['title']}")
                print(f"   URL: {img['link']}")
                print(f"   Size: {img['width']}x{img['height']}")
                print(f"   Source: {img['displayLink']}")
            return True
        else:
            print("❌ No images found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_image_urls_only():
    """Test getting only image URLs"""
    print("\n" + "=" * 80)
    print("2. Testing Image URLs Only")
    print("=" * 80)
    
    client = GoogleImageSearchClient()
    
    query = "machine learning"
    print(f"\nQuery: {query}")
    print("Getting image URLs...")
    
    try:
        urls = client.search_image_urls(query, num_results=3)
        
        if urls:
            print(f"\n✅ Found {len(urls)} image URLs:")
            for i, url in enumerate(urls, 1):
                print(f"{i}. {url}")
            return True
        else:
            print("❌ No URLs found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_large_images():
    """Test searching for large images"""
    print("\n" + "=" * 80)
    print("3. Testing Large Image Search")
    print("=" * 80)
    
    client = GoogleImageSearchClient()
    
    query = "neural network"
    print(f"\nQuery: {query}")
    print("Searching for large images...")
    
    try:
        images = client.search_large_images(query, num_results=3)
        
        if images:
            print(f"\n✅ Found {len(images)} large images:")
            for i, img in enumerate(images, 1):
                print(f"\n{i}. {img['title']}")
                print(f"   Size: {img['width']}x{img['height']}")
                print(f"   URL: {img['link'][:60]}...")
            return True
        else:
            print("❌ No large images found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_photos_only():
    """Test searching for photos specifically"""
    print("\n" + "=" * 80)
    print("4. Testing Photo Search (no clipart)")
    print("=" * 80)
    
    client = GoogleImageSearchClient()
    
    query = "data center"
    print(f"\nQuery: {query}")
    print("Searching for photos only...")
    
    try:
        photos = client.search_photos(query, num_results=3)
        
        if photos:
            print(f"\n✅ Found {len(photos)} photos:")
            for i, photo in enumerate(photos, 1):
                print(f"{i}. {photo['title']}")
            return True
        else:
            print("❌ No photos found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_pagination():
    """Test pagination for more than 10 results"""
    print("\n" + "=" * 80)
    print("5. Testing Pagination (20 results)")
    print("=" * 80)
    
    client = GoogleImageSearchClient()
    
    query = "cloud computing"
    print(f"\nQuery: {query}")
    print("Fetching 20 results with pagination...")
    
    try:
        images = client.search_with_pagination(query, total_results=20)
        
        if images:
            print(f"\n✅ Retrieved {len(images)} images across multiple pages")
            print(f"First image: {images[0]['title']}")
            print(f"Last image: {images[-1]['title']}")
            return True
        else:
            print("❌ Pagination failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 18 + "Google Image Search Test Suite" + " " * 30 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Run tests
    results = []
    results.append(("Basic Image Search", test_basic_image_search()))
    results.append(("Image URLs Only", test_image_urls_only()))
    results.append(("Large Images", test_large_images()))
    results.append(("Photo Search", test_photos_only()))
    results.append(("Pagination", test_pagination()))
    
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
        print("\n🎉 All tests passed! Google Image Search is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check API credentials in .env")
        sys.exit(1)


if __name__ == "__main__":
    main()
