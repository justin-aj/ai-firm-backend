import logging
from typing import List, Dict, Any, Tuple
from clients.milvus_client import MilvusClient
from clients.image_analyzer_client import ImageAnalyzerClient

logger = logging.getLogger(__name__)

def check_text_cache(embedding_client, topics, request) -> Tuple[bool, List[Dict]]:
    """Checks if we already have data for these topics."""
    topics_milvus = MilvusClient(collection_name="ai_firm_topics")
    topics_milvus.connect()
    
    topics_text = ", ".join(topics)
    topics_embedding = embedding_client.generate_embedding(topics_text, max_length=128)
    
    # Ensure collection exists before searching
    try: 
        topics_milvus.create_collection(embedding_dim=1024) # fallback dim
        topics_milvus.collection.load()
    except: 
        return True, [] # If load fails, assume no cache, force scrape

    similar_topics = topics_milvus.search(query_embedding=topics_embedding, top_k=1)

    if similar_topics and similar_topics[0]:
        match = similar_topics[0][0]
        if match.get("score", float("inf")) < 0.5:
            # Topic match found, check for actual content
            main_milvus = MilvusClient(collection_name="ai_firm_vectors")
            main_milvus.connect()
            try: main_milvus.collection.load()
            except: pass
            
            q_emb = embedding_client.generate_embedding(request.question, max_length=512)
            existing_context = main_milvus.search(query_embedding=q_emb, top_k=5)
            
            if existing_context:
                logger.info("Service: Cache Hit - Found existing text context.")
                return False, existing_context # Don't scrape, return context

    return True, [] # Scrape needed

def retrieve_all_context(request, embedding_client) -> List[Dict[str, Any]]:
    """Fetches both Text and Visual context."""
    final_context = []
    
    # 1. Text Retrieval
    try:
        milvus_client = MilvusClient() # Default collection
        milvus_client.connect()
        try: milvus_client.collection.load()
        except: pass
        
        q_emb = embedding_client.generate_embedding(request.question, max_length=512)
        text_hits = milvus_client.search(
            query_embedding=q_emb, 
            top_k=5, 
            output_fields=["text", "metadata"]
        )
        final_context.extend(text_hits)
    except Exception as e:
        logger.warning(f"Text retrieval failed: {e}")

    # 2. Visual Retrieval (Passive - Pre-ingested only)
    if request.include_image_search:
        try:
            # We use ImageAnalyzer just as a DB interface here (load_vlm=False)
            image_db = ImageAnalyzerClient(load_vlm=False, enable_embeddings=True)
            image_hits = image_db.search_vectordb(query=request.question, top_k=3)
            
            for img in image_hits:
                final_context.append({
                    "text": f"[VISUAL ANALYSIS from Image: {img.get('image_title', 'Diagram')}]\n{img.get('analysis', '')}",
                    "metadata": {
                        "url": img.get("image_url", ""),
                        "source": "Visual Retrieval",
                        "type": "image"
                    }
                })
        except Exception as e:
            logger.debug(f"Visual retrieval failed: {e}")

    return final_context