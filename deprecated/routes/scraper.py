"""
Web Scraper routes (archived from `routes/scraper.py`)
"""

from fastapi import APIRouter, HTTPException
from deprecated.models import ScrapeUrlRequest, ScrapeUrlsRequest, SearchAndScrapeRequest
from clients.web_scraper_client import WebScraperClient
from clients.google_search_client import GoogleCustomSearchClient
from config import get_settings
import logging
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scrape", tags=["Web Scraper"])

# Initialize clients (scraper client uses lazy loading to avoid Windows multiprocessing issues)
settings = get_settings()
scraper_client: Optional[WebScraperClient] = None
search_client = GoogleCustomSearchClient()


def get_scraper_client() -> WebScraperClient:
	"""Get or create scraper client (lazy loading to avoid multiprocessing issues on Windows)"""
	global scraper_client
	if scraper_client is None:
		scraper_client = WebScraperClient(
			use_dask=settings.use_dask,
			dask_scheduler=settings.dask_scheduler if settings.dask_scheduler else None,
			dask_workers=settings.dask_workers
		)
	return scraper_client


@router.get("/status")
async def get_scraper_status():
	"""
	Get scraper status including Dask information
	"""
	client = get_scraper_client()
	status = {
		"dask_enabled": client.use_dask,
		"backend": "Dask Distributed" if client.use_dask else "AsyncIO"
	}
    
	if client.use_dask and client.dask_client:
		try:
			status["dask_dashboard"] = client.dask_client.dashboard_link
			status["dask_workers"] = len(client.dask_client.scheduler_info()["workers"])
			status["dask_scheduler"] = client.dask_client.scheduler.address
		except Exception as e:
			status["dask_error"] = str(e)
    
	return status


@router.post("/url")
async def scrape_url(request: ScrapeUrlRequest):
	"""
	Scrape a single URL and extract content
    
	Example:
	```json
	{
		"url": "https://www.example.com",
		"extract_markdown": true,
		"extract_html": false,
		"extract_links": false
	}
	```
	"""
	try:
		logger.info(f"Scraping URL: {request.url}")
		client = get_scraper_client()
		result = await client.scrape_url(
			url=request.url,
			extract_markdown=request.extract_markdown,
			extract_html=request.extract_html,
			extract_links=request.extract_links
		)
        
		if not result.get("success"):
			logger.error(f"Scraping failed: {result.get('error')}")
			raise HTTPException(status_code=500, detail=result.get("error", "Scraping failed"))
        
		return result
	except HTTPException:
		raise
	except Exception as e:
		logger.error(f"Unexpected error in scrape_url: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/urls")
async def scrape_urls(request: ScrapeUrlsRequest):
	"""
	Scrape multiple URLs concurrently
    
	Example:
	```json
	{
		"urls": [
			"https://www.example1.com",
			"https://www.example2.com"
		],
		"extract_markdown": true,
		"max_concurrent": 5
	}
	```
	"""
	try:
		# Validate URLs before processing
		invalid_urls = []
		for url in request.urls:
			if not url.startswith(('http://', 'https://')):
				invalid_urls.append(url)
        
		if invalid_urls:
			logger.error(f"Invalid URLs detected: {invalid_urls}")
			raise HTTPException(
				status_code=400, 
				detail=f"Invalid URLs must start with http:// or https://. Invalid: {invalid_urls}"
			)
        
		logger.info(f"Scraping {len(request.urls)} URLs")
		client = get_scraper_client()
		results = await client.scrape_urls(
			urls=request.urls,
			extract_markdown=request.extract_markdown,
			extract_html=request.extract_html,
			extract_links=request.extract_links,
			max_concurrent=request.max_concurrent
		)
        
		# Count successes and failures
		successes = sum(1 for r in results if r.get("success"))
		failures = len(results) - successes
        
		return {
			"total": len(results),
			"successful": successes,
			"failed": failures,
			"results": results
		}
	except Exception as e:
		logger.error(f"Unexpected error in scrape_urls: {str(e)}")
		raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/search-and-scrape")
async def search_and_scrape(request: SearchAndScrapeRequest):
	"""
	Search Google and scrape the resulting URLs in one request
    
	This endpoint combines Google Custom Search with web scraping:
	1. Performs a Google search
	2. Extracts URLs from search results
	3. Scrapes each URL concurrently
	4. Returns aggregated results
    
	Example:
	```json
	{
		"query": "artificial intelligence news",
		"num_results": 5,
		"extract_markdown": true,
		"max_concurrent": 3
	}
	```
	"""
	try:
		logger.info(f"Search and scrape for query: {request.query}")
        
		# Step 1: Perform Google search
		logger.info("Step 1: Performing Google search")
		params = request.additional_params or {}
		search_results = search_client.search_detailed(
			query=request.query,
			num_results=request.num_results,
			start=request.start,
			**params
		)
        
		if not search_results:
			logger.warning("No search results found")
			return {
				"query": request.query,
				"search_results_count": 0,
				"scraped_count": 0,
				"results": []
			}
        
		# Step 2: Extract URLs from search results
		urls = [result["link"] for result in search_results]
		logger.info(f"Step 2: Found {len(urls)} URLs to scrape")
        
		# Step 3: Scrape all URLs concurrently
		logger.info(f"Step 3: Scraping {len(urls)} URLs")
		client = get_scraper_client()
		scraped_results = await client.scrape_urls(
			urls=urls,
			extract_markdown=request.extract_markdown,
			extract_html=request.extract_html,
			extract_links=False,  # Don't extract links in combined mode
			max_concurrent=request.max_concurrent
		)
        
		# Step 4: Combine search metadata with scraped content
		combined_results = []
		for i, search_result in enumerate(search_results):
			scraped = scraped_results[i] if i < len(scraped_results) else None
            
			combined = {
				"search_metadata": {
					"title": search_result.get("title"),
					"snippet": search_result.get("snippet"),
					"display_link": search_result.get("displayLink")
				},
				"url": search_result.get("link"),
				"scrape_success": scraped.get("success") if scraped else False,
			}
            
			if scraped and scraped.get("success"):
				if request.extract_markdown and "markdown" in scraped:
					combined["content"] = scraped["markdown"]
				if request.extract_html and "html" in scraped:
					combined["html"] = scraped["html"]
				if "metadata" in scraped:
					combined["page_metadata"] = scraped["metadata"]
			else:
				combined["error"] = scraped.get("error") if scraped else "No scrape result"
            
			combined_results.append(combined)
        
		# Count successes
		successful_scrapes = sum(1 for r in combined_results if r.get("scrape_success"))
        
		return {
			"query": request.query,
			"search_results_count": len(search_results),
			"scraped_count": len(scraped_results),
			"successful_scrapes": successful_scrapes,
			"results": combined_results
		}
        
	except Exception as e:
		logger.error(f"Unexpected error in search_and_scrape: {str(e)}")
		raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
