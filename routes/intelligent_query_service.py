"""
Service layer for intelligent_query endpoint.
This module provides helper functions to keep the router small and readable.

Functions:
- process_intelligent_query: Orchestrates the full flow and returns a dict ready for response model

"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from clients.question_analyzer_client import QuestionAnalyzerClient
try:
    # Prefer vLLM for production inference where available
    from clients.vllm_client import VLLMClient as LLMClient
    _VLLM_AVAILABLE = True
except Exception as e:  # pragma: no cover
    # Fall back to GPTOSS if vLLM is not available
    from clients.gpt_oss_client import GPTOSSClient as LLMClient
    _VLLM_AVAILABLE = False
    logger.info(f"vLLM not available, falling back to GPTOSSClient: {e}")
from clients.google_search_client import GoogleCustomSearchClient
from clients.google_image_search_client import GoogleImageSearchClient
from clients.web_scraper_client import WebScraperClient
from clients.milvus_client import MilvusClient
from clients.embedding_client import EmbeddingClient
from clients.image_analyzer_client import ImageAnalyzerClient

logger = logging.getLogger(__name__)

# Use vLLM exclusively for LLM inference — no GPT-OSS fallback
try:
    from clients.vllm_client import VLLMClient as LLMClient
except Exception as e:  # pragma: no cover - explicit failure
    raise ImportError("vLLM client is required for intelligent_query service. Install vllm and related dependencies.")


async def process_intelligent_query(request: Any) -> Dict[str, Any]:
    """Master orchestration for `/ask` flow.

    Returns a dict with all response fields matching `IntelligentQueryResponse` model.
    """
    # Initialize clients
    # Use vLLM where possible for better performance
    llm_client = LLMClient()
    analyzer = QuestionAnalyzerClient(llm_client=llm_client)
    google_search = GoogleCustomSearchClient()
    google_image_search = GoogleImageSearchClient()
    web_scraper = WebScraperClient()
    milvus_client = MilvusClient()
    embedding_client = EmbeddingClient()

    # Defaults for response structure
    response: Dict[str, Any] = {
        "success": True,
        "topics": [],
        "search_results": [],
        "scraped_content": [],
        "image_search_results": [],
        "image_analysis_results": [],
        "stored_in_milvus": False,
        "milvus_ids": [],
        "stored_image_in_milvus": False,
        "image_milvus_ids": [],
        "retrieved_context": [],
        "llm_answer": ""
    }

    # Step 1: Analyze the question
    response["topics"] = await analyze_question(analyzer, request.question)
    topics = response["topics"]

    # Build search query
    search_query = " ".join(topics) if topics else request.question

    # Step 2: Check caches: topics and image collection
    should_scrape = True
    should_image_search = True
    existing_context: List[Dict[str, Any]] = []
    image_existing_context: List[Dict[str, Any]] = []

    try:
        topics_milvus = MilvusClient(collection_name="ai_firm_topics")
        topics_milvus.connect()

        topics_text = ", ".join(topics)
        topics_embedding = embedding_client.generate_embedding(topics_text, max_length=128)

        embedding_dim = embedding_client.get_embedding_dimension()
        topics_milvus.create_collection(embedding_dim=embedding_dim)

        similar_topics = topics_milvus.search(
            query_embedding=topics_embedding,
            top_k=5,
            metric_type="L2",
            search_params={"nprobe": 10},
            output_fields=["text", "metadata"]
        )

        (
            should_scrape,
            existing_context,
            should_image_search,
            image_existing_context,
        ) = check_topic_and_image_cache(
            embedding_client=embedding_client,
            topics=topics,
            request=request,
        )

    except Exception as e:
        logger.warning(f"Service: Error checking topics collection: {e}. Will proceed with scraping.")
        should_scrape = True

    # Step 3: Run web search (if we need fresh content)
    if should_scrape:
        response["search_results"] = web_search(google_search, query=search_query)

    # Step 3b: Image search (optional)
    if request.include_image_search and should_image_search:
        response["image_search_results"] = image_search(google_image_search, query=request.image_query or search_query, num_results=request.image_num_results)

    # If we skipped image search but have existing image context, include that in the response
    if not should_image_search and image_existing_context:
        response["image_analysis_results"].extend(convert_image_existing_context(image_existing_context))

    # Step 4: Scrape URLs (if we have any new URLs to scrape)
    urls = [r.get("link") for r in response["search_results"] if r.get("link")]
    urls = [u for u in urls if u]
    if urls:
        response["scraped_content"] = await scrape_urls(web_scraper, urls)

    # Step 5: Store textual scraped content into Milvus
    successful_scrapes = [s for s in response["scraped_content"] if s.get("success") and s.get("markdown")]
    if successful_scrapes:
        try:
            milvus_client.connect()
            embedding_dim = embedding_client.get_embedding_dimension()
            milvus_client.create_collection(embedding_dim=embedding_dim)
            try:
                milvus_client.create_index(index_type="IVF_FLAT", metric_type="L2", params={"nlist": 128})
            except Exception:
                logger.debug("Service: Milvus index exists or could not be created")

            texts = []
            embeddings = []
            metadata_list = []

            for scrape in successful_scrapes:
                markdown = scrape.get("markdown", "")
                url = scrape.get("url", "")
                if len(markdown) > 60000:
                    markdown = markdown[:60000] + "... [truncated]"
                texts.append(markdown)
                embeddings.append(embedding_client.generate_embedding(markdown, max_length=512))
                metadata_list.append({
                    "url": url,
                    "query": search_query,
                    "topics": topics,
                    "metadata": scrape.get("metadata", {})
                })

            response["milvus_ids"] = store_texts_in_milvus(milvus_client, embedding_client, texts, embeddings, metadata_list)
            response["stored_in_milvus"] = True

            # Record topics in topics collection
            try:
                topics_milvus = MilvusClient(collection_name="ai_firm_topics")
                topics_milvus.connect()
                topics_milvus.create_collection(embedding_dim=embedding_dim)
                try:
                    topics_milvus.create_index(index_type="IVF_FLAT", metric_type="L2", params={"nlist": 128})
                except Exception:
                    pass
                topics_text = ", ".join(topics)
                topics_embedding = embedding_client.generate_embedding(topics_text, max_length=128)
                topics_milvus.insert(
                    texts=[topics_text],
                    embeddings=[topics_embedding],
                    metadata=[{
                        "topics": topics,
                        "query": search_query,
                        "document_count": len(texts),
                        "timestamp": "now"
                    }]
                )
            except Exception:
                logger.warning("Service: Could not store topics in topics collection")

        except Exception as e:
            logger.error(f"Service: Error storing textual scrapes in Milvus: {e}")

    # Step 5b: Image analysis (optional) - only if image search ran and analysis requested
    if request.include_image_search and request.enable_image_analysis and should_image_search and response["image_search_results"]:
        try:
            image_analyzer = ImageAnalyzerClient(load_vlm=False, enable_embeddings=True)
            if request.image_analysis_question:
                img_results = analyze_images(image_analyzer, search_query, request)
            else:
                img_results = image_analyzer.describe_images(query=search_query, num_images=request.image_num_results)

            # Convert results to dict form for API response
            for r in img_results:
                response["image_analysis_results"].extend(convert_img_results_to_dicts(img_results))

            if request.store_image_analysis:
                try:
                    response["image_milvus_ids"], response["stored_image_in_milvus"] = store_image_results(image_analyzer, img_results, query=search_query)
                except Exception as e:
                    logger.warning(f"Service: Failed to store image analysis: {e}")

        except Exception as e:
            logger.warning(f"Service: Image analysis failed: {e}")

    # Step 6: Retrieve relevant context; merge text + image hits
    # If we had existing_context (from topics check), start with it, else search main Milvus
    retrieved_context = existing_context if existing_context else []
    if not retrieved_context:
        try:
            milvus_client.connect()
            question_embedding = embedding_client.generate_embedding(request.question, max_length=512)
            retrieved_context = milvus_client.search(
                query_embedding=question_embedding,
                top_k=5,
                metric_type="L2",
                search_params={"nprobe": 10},
                output_fields=["text", "metadata"]
            )
        except Exception as e:
            logger.warning(f"Service: Retrieval failed: {e}")

    # Add image-based context from image collection (either existing context or search_vectordb)
    if request.include_image_search:
        try:
            retrieved_context = add_image_context_to_retrieved(retrieved_context, request.question)
        except Exception:
            logger.debug("Service: Image vector retrieval failed; continuing")

    response["retrieved_context"] = retrieved_context

    # Step 7: Build prompt content and ask LLM
    context_parts = []
    for i, doc in enumerate(retrieved_context, 1):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        url = metadata.get("url") or metadata.get("image_url") or "Unknown source"
        if len(text) > 2000:
            text = text[:2000] + "... [truncated]"
        context_parts.append(f"[Document {i} from {url}]\n{text}\n")

    context_text = "\n---\n".join(context_parts) if context_parts else "No relevant context found."

    enhanced_prompt = f"""You are a helpful AI assistant. Answer the following question using the provided context from web search results.\n\nQUESTION: {request.question}\n\nCONTEXT FROM WEB SEARCH:\n{context_text}\n\nPlease provide a comprehensive answer based on the context above. If the context doesn't contain relevant information, say so and provide what you know about the topic."""

    response["llm_answer"] = await query_llm(llm_client, enhanced_prompt, request.temperature, request.max_tokens)

    return response


# Singleton and preload helpers for VLM
_image_analyzer_singleton: Optional[ImageAnalyzerClient] = None


def get_image_analyzer_singleton(gpu_memory_utilization: float = 0.9, tensor_parallel_size: int = 1, enable_embeddings: bool = True) -> ImageAnalyzerClient:
    """Return a singleton ImageAnalyzerClient. If not created, create it (no VLM load unless requested).

    If the singleton already exists and parameters differ, a new instance will be created.
    """
    global _image_analyzer_singleton
    if _image_analyzer_singleton is None:
        _image_analyzer_singleton = ImageAnalyzerClient(
            load_vlm=False,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enable_embeddings=enable_embeddings,
        )
    return _image_analyzer_singleton


def preload_image_vlm(tensor_parallel_size: int = 1, force_reload: bool = False) -> Dict[str, Any]:
    """Preload or reload the image VLM (Qwen3-VL).

    Parameters:
        tensor_parallel_size: number of GPUs to use in parallel for VLM
        force_reload: if True, recreate the singleton and reload

    Returns: status dict with whether the VLM is loaded and memory hints
    """
    global _image_analyzer_singleton
    try:
        if force_reload and _image_analyzer_singleton is not None:
            # Attempt to free or remove existing model - best effort
            try:
                _image_analyzer_singleton.vlm = None
            except Exception:
                pass
            _image_analyzer_singleton = None

        if _image_analyzer_singleton is None:
            _image_analyzer_singleton = ImageAnalyzerClient(
                load_vlm=False,
                tensor_parallel_size=tensor_parallel_size,
                enable_embeddings=True
            )

        # Always attempt to initialize VLM now
        _image_analyzer_singleton.tensor_parallel_size = tensor_parallel_size
        _image_analyzer_singleton._initialize_vlm()

        return {
            "status": "loaded" if getattr(_image_analyzer_singleton, 'vlm', None) is not None else "error",
            "tensor_parallel_size": tensor_parallel_size
        }
    except Exception as e:
        logger.error(f"Failed to preload VLM: {e}")
        return {"status": "error", "error": str(e)}


async def analyze_question(analyzer: QuestionAnalyzerClient, question: str) -> List[str]:
    logger.info("Service: Analyzing question")
    analysis = await analyzer.analyze_question(question=question)
    return analysis.get("topics", [])


def check_topic_and_image_cache(embedding_client: EmbeddingClient, topics: List[str], request: Any) -> Tuple[bool, List[Dict[str, Any]], bool, List[Dict[str, Any]]]:
    should_scrape = True
    should_image_search = True
    existing_context: List[Dict[str, Any]] = []
    image_existing_context: List[Dict[str, Any]] = []
    try:
        topics_milvus = MilvusClient(collection_name="ai_firm_topics")
        topics_milvus.connect()
        topics_text = ", ".join(topics)
        topics_embedding = embedding_client.generate_embedding(topics_text, max_length=128)
        embedding_dim = embedding_client.get_embedding_dimension()
        topics_milvus.create_collection(embedding_dim=embedding_dim)

        similar_topics = topics_milvus.search(
            query_embedding=topics_embedding,
            top_k=5,
            metric_type="L2",
            search_params={"nprobe": 10},
            output_fields=["text", "metadata"]
        )
        if similar_topics:
            very_similar = [t for t in similar_topics if t.get("score", float("inf")) < 0.5]
            if very_similar:
                should_scrape = False
                main_milvus = MilvusClient(collection_name="ai_firm_vectors")
                main_milvus.connect()
                main_milvus.create_collection(embedding_dim=embedding_dim)
                q_emb = embedding_client.generate_embedding(request.question, max_length=512)
                existing_context = main_milvus.search(
                    query_embedding=q_emb,
                    top_k=5,
                    metric_type="L2",
                    search_params={"nprobe": 10},
                    output_fields=["text", "metadata"]
                )
                try:
                    image_milvus = MilvusClient(collection_name="image_analysis_retrieval")
                    image_milvus.connect()
                    image_milvus.create_collection(embedding_dim=embedding_dim)
                    similar_images = image_milvus.search(
                        query_embedding=topics_embedding,
                        top_k=5,
                        metric_type="L2",
                        search_params={"nprobe": 10},
                        output_fields=["text", "metadata"]
                    )
                    if similar_images:
                        very_similar_images = [i for i in similar_images if i.get("score", float("inf")) < 0.5]
                        if very_similar_images:
                            should_image_search = False
                            image_existing_context = similar_images
                except Exception:
                    pass
    except Exception:
        pass

    return should_scrape, existing_context, should_image_search, image_existing_context


def web_search(google_search: GoogleCustomSearchClient, query: str) -> List[Dict[str, Any]]:
    logger.info("Service: Running web search")
    return google_search.search_detailed(query=query, num_results=5)


def image_search(google_image_search: GoogleImageSearchClient, query: str, num_results: int) -> List[Dict[str, Any]]:
    try:
        results = google_image_search.search_images(query=query, num_results=num_results)
        logger.info(f"Service: found {len(results)} images for {query}")
        return results
    except Exception as e:
        logger.warning(f"Service: Image search failed: {e}")
        return []


async def scrape_urls(web_scraper: WebScraperClient, urls: List[str]):
    return await web_scraper.scrape_urls(urls=urls, extract_markdown=True, extract_html=False, extract_links=False, max_concurrent=3)


def store_texts_in_milvus(milvus_client: MilvusClient, embedding_client: EmbeddingClient, texts: List[str], embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]) -> List[int]:
    milvus_client.connect()
    milvus_client.create_collection(embedding_dim=len(embeddings[0]) if embeddings else 0)
    try:
        milvus_client.create_index(index_type="IVF_FLAT", metric_type="L2")
    except Exception:
        pass
    ids = milvus_client.insert(texts=texts, embeddings=embeddings, metadata=metadata_list)
    return ids


def analyze_images(image_analyzer: ImageAnalyzerClient, search_query: str, request: Any):
    if request.image_analysis_question:
        return image_analyzer.answer_visual_question(search_query, request.image_analysis_question, num_images=request.image_num_results)
    return image_analyzer.describe_images(query=search_query, num_images=request.image_num_results)


def convert_img_results_to_dicts(img_results: List[Any]) -> List[Dict[str, Any]]:
    converted = []
    for r in img_results:
        converted.append({
            "image_url": r.image_url,
            "image_title": r.image_title,
            "image_source": r.image_source,
            "analysis": r.analysis,
            "has_embedding": r.embedding is not None if getattr(r, 'embedding', None) else False,
            "error": getattr(r, 'error', None)
        })
    return converted


def convert_image_existing_context(image_existing_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for img in image_existing_context:
        md = img.get("metadata", {})
        results.append({
            "image_url": md.get("image_url", ""),
            "image_title": md.get("image_title", ""),
            "image_source": md.get("image_source", ""),
            "analysis": img.get("text", ""),
            "has_embedding": True,
            "error": None
        })
    return results


def store_image_results(image_analyzer: ImageAnalyzerClient, img_results: List[Any], query: str) -> Tuple[List[int], bool]:
    store_resp = image_analyzer.store_in_vectordb(img_results, query=query)
    ids = store_resp.get("ids", []) if isinstance(store_resp, dict) else []
    stored = bool(store_resp.get("stored", 0)) if isinstance(store_resp, dict) else bool(ids)
    return ids, stored


def add_image_context_to_retrieved(retrieved_context: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
    image_analyzer_for_search = ImageAnalyzerClient(load_vlm=False, enable_embeddings=True)
    image_ctx = image_analyzer_for_search.search_vectordb(query=question, top_k=5)
    if image_ctx:
        for img in image_ctx:
            retrieved_context.append({
                "id": img.get("id"),
                "score": img.get("score"),
                "text": img.get("analysis", ""),
                "metadata": {
                    "image_url": img.get("image_url", ""),
                    "image_title": img.get("image_title", ""),
                    "image_source": img.get("image_source", ""),
                    "search_query": img.get("search_query", "")
                }
            })
    return retrieved_context


async def query_llm(llm_client: Any, prompt: str, temperature: float, max_tokens: int) -> str:
    try:
        answer = await llm_client.complete(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
        return answer
    except Exception as e:
        logger.error(f"Service: LLM query error: {e}")
        return f"Error generating answer: {e}"
