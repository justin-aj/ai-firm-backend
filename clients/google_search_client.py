import requests
import logging
from typing import Dict, Any, List
from config import get_settings


class GoogleCustomSearchClient:
    """Client for interacting with Google Custom Search API"""
    
    def __init__(self):
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing GoogleCustomSearchClient")
        self.settings = get_settings()
        self.api_key = self.settings.google_api_key
        self.cx = self.settings.google_cx
        self.base_url = "https://customsearch.googleapis.com/customsearch/v1"
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        start: int = 1,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Perform a custom search query
        
        Args:
            query: Search query string
            num_results: Number of results to return (1-10)
            start: The index of the first result to return (1-indexed)
            **kwargs: Additional parameters for the API (e.g., dateRestrict, siteSearch, etc.)
        
        Returns:
            Dict containing the full API response
        """
        if not self.api_key or not self.cx:
            self.logger.warning("Google Custom Search credentials are missing - please set GOOGLE_API_KEY and GOOGLE_CX in .env")
            return {
                "error": "Google Custom Search API credentials not configured. Please set GOOGLE_API_KEY and GOOGLE_CX in .env file"
            }
        
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": min(num_results, 10),  # API maximum is 10
            "start": start
        }
        
        # Add any additional parameters
        params.update(kwargs)
        
        self.logger.info("Running Google Custom Search")
        self.logger.debug("Search params: %s", {k: (v if k != 'key' else '***') for k,v in params.items()})

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Add debug info if no items found
            if "items" not in data:
                self.logger.debug("Search returned no items", extra={"search_information": data.get("searchInformation", {})})
                return {
                    "error": "No results found",
                    "debug_info": {
                        "search_information": data.get("searchInformation", {}),
                        "full_response": data
                    }
                }

            self.logger.info("Search successful: %s results", len(data.get("items", [])))
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Full search response: %s", data)

            return data
        except requests.exceptions.RequestException as e:
            self.logger.error("Google Custom Search request failed: %s", str(e))
            response_text = getattr(e.response, 'text', 'No response') if hasattr(e, 'response') else 'No response'
            self.logger.debug("Failed response text: %s", response_text)
            return {
                "error": f"Error performing search: {str(e)}",
                "response_text": response_text
            }
    
    def search_urls(
        self,
        query: str,
        num_results: int = 10,
        start: int = 1,
        **kwargs: Any
    ) -> List[str]:
        """
        Perform a search and return only the URLs
        
        Args:
            query: Search query string
            num_results: Number of results to return (1-10)
            start: The index of the first result to return
            **kwargs: Additional parameters for the API
        
        Returns:
            List of URLs from search results
        """
        data = self.search(query, num_results, start, **kwargs)
        
        if "error" in data:
            self.logger.debug("search_urls: upstream search returned error: %s", data.get("error"))
            return []
        
        return [item["link"] for item in data.get("items", [])]
    
    def search_detailed(
        self,
        query: str,
        num_results: int = 10,
        start: int = 1,
        **kwargs: Any
    ) -> List[Dict[str, str]]:
        """
        Perform a search and return detailed results
        
        Args:
            query: Search query string
            num_results: Number of results to return (1-10)
            start: The index of the first result to return
            **kwargs: Additional parameters for the API
        
        Returns:
            List of dicts with title, link, and snippet
        """
        data = self.search(query, num_results, start, **kwargs)
        
        if "error" in data:
            self.logger.debug("search_detailed: upstream search returned error: %s", data.get("error"))
            return []
        
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "displayLink": item.get("displayLink", "")
            })

        self.logger.info("search_detailed returning %d items for query=%s", len(results), query)
        
        return results
    

