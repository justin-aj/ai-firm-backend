"""
Service layer for intelligent_query endpoint.
Master orchestration linking Ingestion and Synthesis orchestrators.
"""

import logging
from typing import Dict, Any

from routes.ingestion_orchestrator import IngestionOrchestrator
from routes.synthesis_orchestrator import SynthesisOrchestrator

logger = logging.getLogger(__name__)


async def run_ingestion_phase(request: Any) -> tuple:
    """
    Execute steps 1-4: initialization, analysis, cache check, and web ingestion.
    
    Returns:
        Tuple of (partial_response, existing_context, embedding_client, llm_client)
    """
    ingestion_orchestrator = IngestionOrchestrator()
    partial_response, existing_context = await ingestion_orchestrator.run(request)
    embedding_client, llm_client = ingestion_orchestrator.get_clients()
    
    logger.info("Ingestion phase complete")
    return partial_response, existing_context, embedding_client, llm_client


async def run_synthesis_phase(
    request: Any,
    existing_context: list,
    embedding_client,
    llm_client
) -> Dict[str, Any]:
    """
    Execute steps 5-6: context retrieval and answer synthesis.
    
    Returns:
        Dict with retrieved_context and llm_answer
    """
    synthesis_orchestrator = SynthesisOrchestrator(embedding_client, llm_client)
    synthesis_result = await synthesis_orchestrator.run(request, existing_context)
    
    logger.info("Synthesis phase complete")
    return synthesis_result


async def process_intelligent_query(request: Any) -> Dict[str, Any]:
    """
    Master orchestration for the `/ask` flow.
    Combines ingestion and synthesis phases.
    """
    # Steps 1-4
    partial_response, existing_context, embedding_client, llm_client = await run_ingestion_phase(request)
    
    # Steps 5-6
    synthesis_result = await run_synthesis_phase(request, existing_context, embedding_client, llm_client)
    
    # Merge results
    response: Dict[str, Any] = {
        **partial_response,
        **synthesis_result
    }
    
    logger.info("Orchestration complete")
    return response