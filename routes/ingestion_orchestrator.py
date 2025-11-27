"""
Ingestion Orchestrator - Steps 1-4
Handles client initialization, query analysis, cache checking, and web ingestion.
"""

import logging
from typing import Dict, Any, Tuple, List

from clients.embedding_client import EmbeddingClient
from routes import ingestion_service, retrieval_service

try:
    from clients.vllm_client import VLLMClient as LLMClient
    _VLLM_AVAILABLE = True
except Exception:
    from clients.gpt_oss_client import GPTOSSClient as LLMClient
    _VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    """Orchestrates steps 1-4: initialization, analysis, cache check, and ingestion."""
    
    def __init__(self):
        self.embedding_client: EmbeddingClient = None
        self.llm_client: LLMClient = None
    
    def _initialize_clients(self) -> None:
        """Step 1: Initialize core clients."""
        self.embedding_client = EmbeddingClient()
        self.llm_client = LLMClient(
            gpu_memory_utilization=0.6,
            num_speculative_tokens=3
        )
    
    async def run(self, request: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Execute steps 1-4 and return partial response + retrieved context.
        
        Returns:
            Tuple of (partial_response dict, existing_context list)
        """
        # Step 1: Initialize clients
        self._initialize_clients()
        
        partial_response: Dict[str, Any] = {
            "success": True,
            "topics": [],
            "search_results": [],
            "scraped_content": [],
            "stored_in_milvus": False,
        }
        
        # Step 2: ANALYZE (Ingestion Service)
        topics, search_query = await ingestion_service.analyze_and_optimize_query(
            request, self.llm_client
        )
        partial_response["topics"] = topics
        logger.info(f"IngestionOrchestrator: Optimized Query -> '{search_query}'")
        
        # Step 3: CHECK CACHE (Retrieval Service)
        should_scrape, existing_context = retrieval_service.check_text_cache(
            self.embedding_client, topics, request
        )
        
        # Step 4: INGEST (Ingestion Service) - if needed
        if should_scrape:
            logger.info("IngestionOrchestrator: Cache miss. Starting Web Ingestion...")
            s_results, s_content, m_ids = await ingestion_service.run_web_ingestion(
                search_query, topics, self.embedding_client
            )
            partial_response["search_results"] = s_results
            partial_response["scraped_content"] = s_content
            partial_response["stored_in_milvus"] = bool(m_ids)
        else:
            logger.info("IngestionOrchestrator: Cache hit. Skipping Web Ingestion.")
        
        return partial_response, existing_context
    
    def get_clients(self) -> Tuple[EmbeddingClient, LLMClient]:
        """Return initialized clients for use in subsequent steps."""
        return self.embedding_client, self.llm_client