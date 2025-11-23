"""
Google Image Search Client
Dedicated client for searching images using Google Custom Search API
"""

import requests
import logging
from typing import Dict, Any, List, Optional
from config import get_settings


class GoogleImageSearchClient:
    """Client for searching images with Google Custom Search API"""
    
    def __init__(self):
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing GoogleImageSearchClient")

        self.settings = get_settings()
        self.api_key = self.settings.google_api_key
        self.cx = self.settings.google_cx
        self.base_url = "https://customsearch.googleapis.com/customsearch/v1"
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        start: int = 1,
        image_size: Optional[str] = None,
        image_type: Optional[str] = None,
        image_color_type: Optional[str] = None,
        safe_search: str = "off",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Search for images
        
        Args:
            query: Search query string
            num_results: Number of results to return (1-10)
            start: The index of the first result to return (1-indexed)
            image_size: Image size filter (icon, small, medium, large, xlarge, xxlarge, huge)
            image_type: Image type (clipart, face, lineart, stock, photo, animated)
            image_color_type: Color type (color, gray, mono, trans)
            safe_search: Safe search level (active, off)
            **kwargs: Additional parameters for the API
        
        Returns:
            Dict containing the full API response
        """
        if not self.api_key or not self.cx:
            self.logger.warning("Google Custom Search API credentials missing; set GOOGLE_API_KEY and GOOGLE_CX in .env")
            return {
                "error": "Google Custom Search API credentials not configured. Please set GOOGLE_API_KEY and GOOGLE_CX in .env file"
            }
        
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": min(num_results, 10),  # API maximum is 10
            "start": start,
            "searchType": "image",  # Image search
            "safe": safe_search
        }
        
        # Add image filters
        if image_size:
            params["imgSize"] = image_size
        
        if image_type:
            params["imgType"] = image_type
        
        if image_color_type:
            params["imgColorType"] = image_color_type
        
        # Add any additional parameters
        params.update(kwargs)
        
        try:
            self.logger.info("Executing Google Image Search: %s (num=%s start=%s)", query, params.get('num'), params.get('start'))
            self.logger.debug("Search params: %s", {k: (v if k != 'key' else '***') for k,v in params.items()})

            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Add debug info if no items found
            if "items" not in data:
                return {
                    "error": "No results found",
                    "debug_info": {
                        "search_information": data.get("searchInformation", {}),
                        "full_response": data
                    }
                }
            
            return data
        except requests.exceptions.RequestException as e:
            self.logger.error("Image search request failed: %s", str(e))
            response_text = getattr(e.response, 'text', 'No response') if hasattr(e, 'response') else 'No response'
            self.logger.debug("Failed response text: %s", response_text)
            return {
                "error": f"Error performing image search: {str(e)}",
                "response_text": response_text
            }
    
    def search_images(
        self,
        query: str,
        num_results: int = 10,
        start: int = 1,
        image_size: Optional[str] = None,
        image_type: Optional[str] = None,
        **kwargs: Any
    ) -> List[Dict[str, str]]:
        """
        Search for images and return detailed results
        
        Args:
            query: Search query string
            num_results: Number of results to return (1-10)
            start: The index of the first result to return
            image_size: Image size filter (icon, small, medium, large, xlarge, xxlarge, huge)
            image_type: Image type (clipart, face, lineart, stock, photo, animated)
            **kwargs: Additional parameters for the API
        
        Returns:
            List of dicts with image information
        """
        data = self.search(query, num_results, start, image_size, image_type, **kwargs)
        
        if "error" in data:
            self.logger.debug("search_images upstream error: %s", data.get("error"))
            return []
        
        results = []
        for item in data.get("items", []):
            image_info = item.get("image", {})
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),  # Direct image URL
                "thumbnail": image_info.get("thumbnailLink", ""),
                "thumbnailWidth": image_info.get("thumbnailWidth", 0),
                "thumbnailHeight": image_info.get("thumbnailHeight", 0),
                "contextLink": image_info.get("contextLink", ""),  # Page URL where image is hosted
                "width": image_info.get("width", 0),
                "height": image_info.get("height", 0),
                "byteSize": image_info.get("byteSize", 0),
                "displayLink": item.get("displayLink", "")
            })
        
        self.logger.info("search_images returning %d items for query=%s", len(results), query)
        return results
    
    def search_image_urls(
        self,
        query: str,
        num_results: int = 10,
        start: int = 1,
        image_size: Optional[str] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Search for images and return only the direct image URLs
        
        Args:
            query: Search query string
            num_results: Number of results to return (1-10)
            start: The index of the first result to return
            image_size: Image size filter
            **kwargs: Additional parameters for the API
        
        Returns:
            List of direct image URLs
        """
        images = self.search_images(query, num_results, start, image_size, **kwargs)
        self.logger.debug("search_image_urls found %d image URLs", len(images))
        return [img["link"] for img in images]
    
    def search_large_images(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, str]]:
        """
        Search for large/high-quality images
        
        Args:
            query: Search query string
            num_results: Number of results to return
            **kwargs: Additional parameters
        
        Returns:
            List of large image results
        """
        self.logger.debug("search_large_images: query=%s num_results=%d", query, num_results)
        return self.search_images(
            query,
            num_results,
            image_size="large",
            **kwargs
        )
    
    def search_photos(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, str]]:
        """
        Search specifically for photos (not clipart/lineart)
        
        Args:
            query: Search query string
            num_results: Number of results to return
            **kwargs: Additional parameters
        
        Returns:
            List of photo results
        """
        self.logger.debug("search_photos: query=%s num_results=%d", query, num_results)
        return self.search_images(
            query,
            num_results,
            image_type="photo",
            **kwargs
        )
    
    def search_clipart(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, str]]:
        """
        Search specifically for clipart images
        
        Args:
            query: Search query string
            num_results: Number of results to return
            **kwargs: Additional parameters
        
        Returns:
            List of clipart results
        """
        self.logger.debug("search_clipart: query=%s num_results=%d", query, num_results)
        return self.search_images(
            query,
            num_results,
            image_type="clipart",
            **kwargs
        )
    
    def search_faces(
        self,
        query: str,
        num_results: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, str]]:
        """
        Search for images containing faces
        
        Args:
            query: Search query string
            num_results: Number of results to return
            **kwargs: Additional parameters
        
        Returns:
            List of face image results
        """
        self.logger.debug("search_faces: query=%s num_results=%d", query, num_results)
        return self.search_images(
            query,
            num_results,
            image_type="face",
            **kwargs
        )
    
    def search_with_pagination(
        self,
        query: str,
        total_results: int = 50,
        image_size: Optional[str] = None,
        **kwargs: Any
    ) -> List[Dict[str, str]]:
        """
        Search for images with automatic pagination to get more than 10 results
        
        Args:
            query: Search query string
            total_results: Total number of results to fetch (will paginate)
            image_size: Image size filter
            **kwargs: Additional parameters
        
        Returns:
            List of image results (up to total_results)
        """
        all_results = []
        start = 1
        
        self.logger.info("search_with_pagination: fetching up to %d results for query=%s", total_results, query)
        while len(all_results) < total_results:
            batch_size = min(10, total_results - len(all_results))
            batch = self.search_images(
                query,
                num_results=batch_size,
                start=start,
                image_size=image_size,
                **kwargs
            )
            
            if not batch:
                break
            
            all_results.extend(batch)
            start += 10
        
        return all_results[:total_results]
