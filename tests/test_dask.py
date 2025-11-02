"""
Test Dask functionality in isolation
Run this to verify Dask works before starting the full server
"""

import asyncio
import sys


def test_dask_import():
    """Test 1: Check if Dask is installed"""
    print("Test 1: Checking Dask installation...")
    try:
        import dask
        from dask.distributed import Client
        print(f"✓ Dask version: {dask.__version__}")
        print(f"✓ Dask distributed imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import Dask: {e}")
        return False


def test_dask_client_simple():
    """Test 2: Try creating a simple Dask client (this might hang on Windows)"""
    print("\nTest 2: Creating Dask Client (simple)...")
    print("⚠ WARNING: This may hang on Windows - wait 10 seconds max")
    
    try:
        from dask.distributed import Client
        import time
        
        start = time.time()
        # Try with minimal workers
        client = Client(n_workers=2, threads_per_worker=1, timeout='5s')
        elapsed = time.time() - start
        
        print(f"✓ Client created in {elapsed:.2f}s")
        print(f"✓ Dashboard: {client.dashboard_link}")
        print(f"✓ Workers: {len(client.scheduler_info()['workers'])}")
        
        # Clean up
        client.close()
        print("✓ Client closed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        return False


def test_dask_delayed():
    """Test 3: Test Dask delayed tasks (requires working client)"""
    print("\nTest 3: Testing Dask delayed tasks...")
    
    try:
        from dask.distributed import Client
        import dask
        
        def simple_task(x):
            return x * 2
        
        client = Client(n_workers=2, threads_per_worker=1, timeout='5s')
        
        # Create delayed tasks
        tasks = [dask.delayed(simple_task)(i) for i in range(5)]
        results = dask.compute(*tasks, scheduler='distributed')
        
        print(f"✓ Computed results: {results}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


async def test_crawl4ai_basic():
    """Test 4: Test Crawl4AI without Dask"""
    print("\nTest 4: Testing Crawl4AI (AsyncIO mode)...")
    
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai import DefaultMarkdownGenerator, PruningContentFilter
        
        browser_conf = BrowserConfig(headless=True)
        run_conf = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(min_word_threshold=20)
            )
        )
        
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(url="https://example.com", config=run_conf)
            
            print(f"✓ Crawled example.com")
            print(f"✓ Success: {result.success if hasattr(result, 'success') else 'N/A'}")
            
            if hasattr(result, 'markdown') and hasattr(result.markdown, 'raw_markdown'):
                markdown_len = len(result.markdown.raw_markdown)
                print(f"✓ Markdown length: {markdown_len} chars")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_crawl4ai_with_dask():
    """Test 5: Test Crawl4AI with Dask distributed"""
    print("\nTest 5: Testing Crawl4AI with Dask...")
    
    try:
        from dask.distributed import Client
        import dask
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai import DefaultMarkdownGenerator, PruningContentFilter
        
        async def crawl_url(url):
            browser_conf = BrowserConfig(headless=True)
            run_conf = CrawlerRunConfig(
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(min_word_threshold=20)
                )
            )
            async with AsyncWebCrawler(config=browser_conf) as crawler:
                result = await crawler.arun(url=url, config=run_conf)
                markdown = result.markdown.raw_markdown if hasattr(result.markdown, 'raw_markdown') else ""
                return {"url": url, "length": len(markdown)}
        
        def delayed_crawl(url):
            return dask.delayed(lambda u: asyncio.run(crawl_url(u)))(url)
        
        # Create client
        client = Client(n_workers=2, threads_per_worker=1, timeout='5s')
        print(f"✓ Client created: {client.dashboard_link}")
        
        # Test with a few URLs
        urls = ["https://example.com", "https://example.org"]
        tasks = [delayed_crawl(u) for u in urls]
        results = dask.compute(*tasks)
        
        print(f"✓ Crawled {len(results)} URLs with Dask")
        for r in results:
            print(f"  - {r['url']}: {r['length']} chars")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("DASK TESTING SUITE")
    print("=" * 60)
    
    results = []
    
    # Test 1: Import
    results.append(("Import Test", test_dask_import()))
    
    if results[-1][1]:
        # Test 2: Client creation (may hang on Windows)
        print("\n⚠ Next test may hang on Windows. Press Ctrl+C if stuck > 10s")
        try:
            results.append(("Client Creation", test_dask_client_simple()))
        except KeyboardInterrupt:
            print("\n✗ Test interrupted (likely hung)")
            results.append(("Client Creation", False))
        
        if results[-1][1]:
            # Test 3: Delayed tasks
            try:
                results.append(("Delayed Tasks", test_dask_delayed()))
            except KeyboardInterrupt:
                print("\n✗ Test interrupted")
                results.append(("Delayed Tasks", False))
    
    # Test 4: Crawl4AI basic (always run)
    print("\n" + "=" * 60)
    results.append(("Crawl4AI Basic", asyncio.run(test_crawl4ai_basic())))
    
    # Test 5: Crawl4AI with Dask (only if client works)
    if any(name == "Client Creation" and passed for name, passed in results):
        try:
            results.append(("Crawl4AI + Dask", asyncio.run(test_crawl4ai_with_dask())))
        except KeyboardInterrupt:
            print("\n✗ Test interrupted")
            results.append(("Crawl4AI + Dask", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    print(f"\nPassed: {total_passed}/{total_tests}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if all(passed for _, passed in results):
        print("✓ All tests passed! Dask is working on your system.")
        print("  You can enable USE_DASK=True in .env")
    elif any(name == "Crawl4AI Basic" and passed for name, passed in results):
        print("⚠ Dask has issues but AsyncIO scraping works perfectly!")
        print("  Recommendation: Keep USE_DASK=False in .env")
        print("  AsyncIO handles 20 concurrent requests efficiently.")
        print("\n  To use Dask on Windows:")
        print("    1. Use WSL (Windows Subsystem for Linux)")
        print("    2. Connect to external Linux Dask scheduler")
    else:
        print("✗ Both Dask and Crawl4AI have issues")
        print("  Check your installation: pip install -r requirements.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
