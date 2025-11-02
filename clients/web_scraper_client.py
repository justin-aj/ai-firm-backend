"""
Web Scraper Client using Crawl4AI
Scrapes and extracts content from URLs with optional Dask distributed computing
"""

import asyncio
from typing import List, Dict, Any, Optional
from crawl4ai import AsyncWebCrawler
import logging

logger = logging.getLogger(__name__)


# Standalone function for Dask workers (must be at module level to be picklable)
def _scrape_single_url_sync(
    url: str,
    extract_markdown: bool = True,
    extract_html: bool = False,
    extract_links: bool = False
) -> Dict[str, Any]:
    """
    Synchronous wrapper for scraping a single URL (for Dask workers)
    This is a standalone function so it can be properly pickled
    """
    async def _scrape():
        try:
            from crawl4ai import BrowserConfig, CrawlerRunConfig, DefaultMarkdownGenerator, PruningContentFilter
            
            browser_conf = BrowserConfig(headless=True)
            run_conf = CrawlerRunConfig(
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(min_word_threshold=20)
                )
            )
            
            async with AsyncWebCrawler(config=browser_conf) as crawler:
                result = await crawler.arun(url=url, config=run_conf)
                
                response = {
                    "url": url,
                    "success": result.success if hasattr(result, 'success') else True,
                }
                
                if extract_markdown and hasattr(result, 'markdown'):
                    response["markdown"] = result.markdown.raw_markdown if hasattr(result.markdown, 'raw_markdown') else str(result.markdown)
                
                if extract_html and hasattr(result, 'html'):
                    response["html"] = result.html
                
                if extract_links and hasattr(result, 'links'):
                    response["links"] = result.links
                
                if hasattr(result, 'metadata'):
                    response["metadata"] = result.metadata
                
                return response
                
        except Exception as e:
            return {
                "url": url,
                "success": False,
                "error": str(e)
            }
    
    return asyncio.run(_scrape())


class WebScraperClient:
    """Client for scraping web pages using Crawl4AI with optional Dask distribution"""
    
    def __init__(self, use_dask: bool = False, dask_scheduler: Optional[str] = None, dask_workers: int = 4):
        """
        Initialize the web scraper client
        
        Args:
            use_dask: Whether to use Dask for distributed scraping
            dask_scheduler: Dask scheduler address (e.g., 'localhost:8786')
                          If None and use_dask=True, creates a local client with threaded workers
            dask_workers: Number of Dask workers (default: 4)
        """
        self.use_dask = use_dask
        self.dask_client = None
        
        if use_dask:
            try:
                from dask.distributed import Client
                
                if dask_scheduler:
                    # Connect to existing scheduler
                    self.dask_client = Client(dask_scheduler)
                    logger.info(f"Connected to Dask scheduler at {dask_scheduler}")
                else:
                    # Create local threaded client (Windows compatible - simpler than LocalCluster)
                    self.dask_client = Client(n_workers=dask_workers, threads_per_worker=1)
                    logger.info(f"Created Dask client with {dask_workers} threaded workers")
                    logger.info(f"Dask dashboard: {self.dask_client.dashboard_link}")
                    
            except ImportError:
                logger.warning("Dask not installed. Falling back to asyncio.")
                self.use_dask = False
            except Exception as e:
                logger.error(f"Error initializing Dask: {str(e)}. Falling back to asyncio.")
                self.use_dask = False
    
    async def scrape_url(
        self,
        url: str,
        extract_markdown: bool = True,
        extract_html: bool = False,
        extract_links: bool = False
    ) -> Dict[str, Any]:
        """
        Scrape a single URL
        
        Args:
            url: The URL to scrape
            extract_markdown: Whether to extract markdown content
            extract_html: Whether to include raw HTML
            extract_links: Whether to extract all links from the page
        
        Returns:
            Dict containing scraped content
        """
        try:
            from crawl4ai import BrowserConfig, CrawlerRunConfig, DefaultMarkdownGenerator, PruningContentFilter
            
            browser_conf = BrowserConfig(headless=True)
            run_conf = CrawlerRunConfig(
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(min_word_threshold=20)
                )
            )
            
            async with AsyncWebCrawler(config=browser_conf) as crawler:
                result = await crawler.arun(url=url, config=run_conf)
                
                response = {
                    "url": url,
                    "success": result.success if hasattr(result, 'success') else True,
                }
                
                if extract_markdown and hasattr(result, 'markdown'):
                    # Use raw_markdown from the markdown generator
                    response["markdown"] = result.markdown.raw_markdown if hasattr(result.markdown, 'raw_markdown') else str(result.markdown)
                
                if extract_html and hasattr(result, 'html'):
                    response["html"] = result.html
                
                if extract_links and hasattr(result, 'links'):
                    response["links"] = result.links
                
                # Add metadata if available
                if hasattr(result, 'metadata'):
                    response["metadata"] = result.metadata
                
                return response
                
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return {
                "url": url,
                "success": False,
                "error": str(e)
            }
    
    async def scrape_urls(
        self,
        urls: List[str],
        extract_markdown: bool = True,
        extract_html: bool = False,
        extract_links: bool = False,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs concurrently
        
        Args:
            urls: List of URLs to scrape
            extract_markdown: Whether to extract markdown content
            extract_html: Whether to include raw HTML
            extract_links: Whether to extract all links from the page
            max_concurrent: Maximum number of concurrent requests (ignored if using Dask)
        
        Returns:
            List of dicts containing scraped content from each URL
        """
        if not urls:
            return []
        
        # Use Dask if available and enabled
        if self.use_dask and self.dask_client:
            return await self._scrape_urls_dask(
                urls, extract_markdown, extract_html, extract_links
            )
        else:
            return await self._scrape_urls_async(
                urls, extract_markdown, extract_html, extract_links, max_concurrent
            )
    
    async def _scrape_urls_dask(
        self,
        urls: List[str],
        extract_markdown: bool,
        extract_html: bool,
        extract_links: bool
    ) -> List[Dict[str, Any]]:
        """
        Scrape URLs using Dask distributed computing
        Uses standalone function to avoid pickling issues with class methods
        """
        logger.info(f"Scraping {len(urls)} URLs using Dask distributed computing")
        
        try:
            futures = []
            for url in urls:
                # Use the standalone function instead of class method
                future = self.dask_client.submit(
                    _scrape_single_url_sync,
                    url, 
                    extract_markdown, 
                    extract_html, 
                    extract_links
                )
                futures.append(future)
            
            # Gather all results
            results = self.dask_client.gather(futures)
            logger.info(f"Dask scraping completed: {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error in Dask scraping: {str(e)}")
            # Fallback to async scraping
            logger.info("Falling back to async scraping")
            return await self._scrape_urls_async(
                urls, extract_markdown, extract_html, extract_links, max_concurrent=20
            )
    
    async def _scrape_urls_async(
        self,
        urls: List[str],
        extract_markdown: bool,
        extract_html: bool,
        extract_links: bool,
        max_concurrent: int
    ) -> List[Dict[str, Any]]:
        """
        Scrape URLs using asyncio (original implementation)
        """
        logger.info(f"Scraping {len(urls)} URLs using asyncio (max_concurrent={max_concurrent})")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_limit(url: str):
            async with semaphore:
                return await self.scrape_url(
                    url=url,
                    extract_markdown=extract_markdown,
                    extract_html=extract_html,
                    extract_links=extract_links
                )
        
        # Scrape all URLs concurrently (with limit)
        tasks = [scrape_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to error responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception scraping {urls[i]}: {str(result)}")
                processed_results.append({
                    "url": urls[i],
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def scrape_with_summary(
        self,
        url: str,
        max_length: Optional[int] = 1000
    ) -> Dict[str, Any]:
        """
        Scrape a URL and return a summary of the content
        
        Args:
            url: The URL to scrape
            max_length: Maximum length of the content summary
        
        Returns:
            Dict containing URL, summary, and metadata
        """
        result = await self.scrape_url(url, extract_markdown=True)
        
        if not result.get("success"):
            return result
        
        markdown = result.get("markdown", "")
        
        # Create summary
        if max_length and len(markdown) > max_length:
            summary = markdown[:max_length] + "..."
        else:
            summary = markdown
        
        return {
            "url": url,
            "success": True,
            "summary": summary,
            "full_length": len(markdown),
            "metadata": result.get("metadata", {})
        }
