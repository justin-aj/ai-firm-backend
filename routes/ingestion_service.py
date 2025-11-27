import logging
import gc
import torch
from typing import List, Dict, Any, Tuple

from clients.question_analyzer_client import QuestionAnalyzerClient
from clients.google_search_client import GoogleCustomSearchClient
from clients.web_scraper_client import WebScraperClient
from clients.milvus_client import MilvusClient
from clients.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)

async def analyze_and_optimize_query(request: Any, llm_client: Any) -> Tuple[List[str], str]:
    """Uses LLM to extract topics and create an optimized Google search query."""
    analyzer = QuestionAnalyzerClient(llm_client=llm_client)
    analysis_result = await analyzer.analyze_question(request.question)
    
    topics = analysis_result.get("topics", [])
    optimized_queries = analysis_result.get("search_queries", [])
    
    if optimized_queries:
        search_query = optimized_queries[0]
    elif topics:
        search_query = " ".join(topics)
    else:
        search_query = request.question
        
    return topics, search_query

async def run_web_ingestion(search_query: str, topics: List[str], embedding_client: EmbeddingClient):
    """Performs Web Search, Scraping, and Storage into Milvus."""
    google_search = GoogleCustomSearchClient()
    web_scraper = WebScraperClient()
    milvus_client = MilvusClient()

    # 1. Web Search
    search_results = google_search.search_detailed(query=search_query, num_results=5)
    urls = [r.get("link") for r in search_results if r.get("link")]
    
    # 2. Scrape
    scraped_content = []
    if urls:
        scraped_content = await web_scraper.scrape_urls(urls=urls, extract_markdown=True)

    # 3. Prepare Data
    texts = []
    metadata_list = []
    successful_scrapes = [s for s in scraped_content if s.get("success") and s.get("markdown")]

    for scrape in successful_scrapes:
        markdown = scrape.get("markdown", "")
        # Hard truncate to prevent token overflows
        if len(markdown) > 50000: markdown = markdown[:50000] + "... [truncated]"
        
        texts.append(markdown)
        metadata_list.append({
            "url": scrape.get("url", ""),
            "query": search_query,
            "topics": topics,
            "metadata": scrape.get("metadata", {})
        })

    # 4. Store in Milvus
    milvus_ids = []
    if texts:
        # Generate Embeddings
        embeddings = [embedding_client.generate_embedding(t, max_length=512) for t in texts]
        
        # Store
        milvus_ids = _store_vectors(milvus_client, embedding_client, texts, embeddings, metadata_list)
        
        # Update Topic Cache
        _update_topic_cache(milvus_client, embedding_client, topics, search_query, len(texts))

    return search_results, scraped_content, milvus_ids

def _store_vectors(milvus_client, embedding_client, texts, embeddings, metadata_list):
    """Internal helper to write to Milvus safely."""
    gc.collect(); torch.cuda.empty_cache()
    milvus_client.connect()
    
    dim = len(embeddings[0]) if embeddings else embedding_client.get_embedding_dimension()
    milvus_client.create_collection(embedding_dim=dim)
    try: milvus_client.create_index(index_type="IVF_FLAT", metric_type="L2")
    except: pass

    total_ids = []
    BATCH_SIZE = 50 
    for i in range(0, len(texts), BATCH_SIZE):
        try:
            ids = milvus_client.insert(
                texts=texts[i:i+BATCH_SIZE], 
                embeddings=embeddings[i:i+BATCH_SIZE], 
                metadata=metadata_list[i:i+BATCH_SIZE]
            )
            total_ids.extend(ids)
        except Exception as e: logger.error(f"Insert failed: {e}")
    
    try: milvus_client.collection.flush()
    except: pass
    return total_ids

def _update_topic_cache(milvus_client, embedding_client, topics, search_query, count):
    """Updates the topics collection."""
    try:
        client = MilvusClient(collection_name="ai_firm_topics")
        client.connect()
        dim = embedding_client.get_embedding_dimension()
        client.create_collection(embedding_dim=dim)
        try: client.create_index()
        except: pass
        
        txt = ", ".join(topics)
        emb = embedding_client.generate_embedding(txt, max_length=128)
        client.insert(texts=[txt], embeddings=[emb], metadata=[{"topics": topics, "query": search_query, "count": count}])
    except Exception: pass