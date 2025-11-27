"""
Synthesis Orchestrator - Steps 5-6
Handles context retrieval and LLM answer generation.
"""

import logging
from typing import Dict, Any, List

from clients.embedding_client import EmbeddingClient
from routes import retrieval_service, synthesis_service

logger = logging.getLogger(__name__)


class SynthesisOrchestrator:
    """Orchestrates steps 5-6: retrieval and synthesis."""
    
    def __init__(self, embedding_client: EmbeddingClient, llm_client: Any):
        self.embedding_client = embedding_client
        self.llm_client = llm_client
    
    def _retrieve_context(
        self, 
        request: Any, 
        existing_context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Step 5: RETRIEVE - Get text and visual context.
        
        If cache hit provided existing_context, use that as base.
        Otherwise, search the DB that was just populated.
        """
        if existing_context:
            retrieved_context = existing_context
            
            # Append visual context if requested
            if request.include_image_search:
                visual = retrieval_service.retrieve_all_context(
                    request, self.embedding_client
                )
                visual_only = [
                    x for x in visual 
                    if x.get('metadata', {}).get('type') == 'image'
                ]
                retrieved_context.extend(visual_only)
        else:
            retrieved_context = retrieval_service.retrieve_all_context(
                request, self.embedding_client
            )
        
        return retrieved_context
    
    async def _generate_answer(
        self,
        question: str,
        retrieved_context: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Step 6: SYNTHESIZE - Generate the final answer."""
        return await synthesis_service.generate_answer(
            self.llm_client,
            question,
            retrieved_context,
            temperature,
            max_tokens
        )
    
    async def run(
        self,
        request: Any,
        existing_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute steps 5-6 and return the synthesis results.
        
        Args:
            request: The original request object
            existing_context: Context from cache hit (may be empty)
            
        Returns:
            Dict with retrieved_context and llm_answer
        """
        # Step 5: RETRIEVE
        retrieved_context = self._retrieve_context(request, existing_context)
        logger.info(f"SynthesisOrchestrator: Retrieved {len(retrieved_context)} context items")
        
        # Step 6: SYNTHESIZE
        llm_answer = await self._generate_answer(
            request.question,
            retrieved_context,
            request.temperature,
            request.max_tokens
        )
        logger.info("SynthesisOrchestrator: Answer generation complete")
        
        return {
            "retrieved_context": retrieved_context,
            "llm_answer": llm_answer
        }