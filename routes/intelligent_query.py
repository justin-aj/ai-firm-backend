"""
Intelligent Query Routes
Multi-agent workflow for processing user queries
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from clients.question_analyzer_client import QuestionAnalyzerClient
from clients.gpt_oss_client import GPTOSSClient
from clients.google_search_client import GoogleCustomSearchClient
from clients.web_scraper_client import WebScraperClient
from clients.milvus_client import MilvusClient
from clients.embedding_client import EmbeddingClient
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Intelligent Query"])


class IntelligentQueryRequest(BaseModel):
    """Request model for intelligent query processing"""
    question: str = Field(..., description="The user's question")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1)


class IntelligentQueryResponse(BaseModel):
    """Response model for intelligent query"""
    success: bool
    topics: List[str]
    search_results: List[Dict[str, Any]]
    scraped_content: List[Dict[str, Any]]
    stored_in_milvus: bool
    milvus_ids: List[int]
    retrieved_context: List[Dict[str, Any]]
    llm_answer: str


@router.post("/ask", response_model=IntelligentQueryResponse)
async def intelligent_ask(request: IntelligentQueryRequest):
    """
    Process a user question through multi-agent workflow:
    1. Question Analyzer Agent: Uses LLM to break down question into topics/concepts
    2. Google Search: Search for relevant URLs
    3. Web Scraper: Scrape content from URLs
    4. Milvus Storage: Store scraped content with embeddings
    
    Example:
    ```json
    {
        "question": "What is the use of Triton in ML? How can it help me in my GPU based project?",
        "temperature": 0.7
    }
    ```
    """
    try:
        # Initialize agents
        gpt_oss_client = GPTOSSClient()
        analyzer = QuestionAnalyzerClient(llm_client=gpt_oss_client)
        google_search = GoogleCustomSearchClient()
        web_scraper = WebScraperClient()
        milvus_client = MilvusClient()
        embedding_client = EmbeddingClient()
        
        # Validate input
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Step 1: Analyze the question to get topics
        logger.info("Step 1: Analyzing question")
        analysis = await analyzer.analyze_question(question=request.question)
        topics = analysis.get('topics', [])
        
        # Step 2: Check if we already have similar topics stored
        logger.info(f"Step 2: Checking if topics {topics} already exist in our database")
        should_scrape = True
        existing_context = []
        
        try:
            # Create a topics collection separate from main content
            topics_milvus = MilvusClient(collection_name="ai_firm_topics")
            topics_milvus.connect()
            
            # Generate embedding for the combined topics
            topics_text = ", ".join(topics)
            topics_embedding = embedding_client.generate_embedding(topics_text, max_length=128)
            
            # Get embedding dimension and ensure topics collection exists
            embedding_dim = embedding_client.get_embedding_dimension()
            topics_milvus.create_collection(embedding_dim=embedding_dim)
            
            # Search for similar topics
            similar_topics = topics_milvus.search(
                query_embedding=topics_embedding,
                top_k=5,
                metric_type="L2",
                search_params={"nprobe": 10},
                output_fields=["text", "metadata"]
            )
            
            # If we found very similar topics (distance < 0.5), we already have this content
            if similar_topics and len(similar_topics) > 0:
                very_similar = [t for t in similar_topics if t.get('score', float('inf')) < 0.5]
                if len(very_similar) > 0:
                    should_scrape = False
                    logger.info(f"Found {len(very_similar)} very similar topic sets. Skipping scraping.")
                    logger.info(f"Similar topics: {[t.get('text', '') for t in very_similar[:3]]}")
                    
                    # Retrieve the actual content from main collection using the stored metadata
                    main_milvus = MilvusClient(collection_name="ai_firm_vectors")
                    main_milvus.connect()
                    main_milvus.create_collection(embedding_dim=embedding_dim)
                    
                    # Search main collection with question embedding for relevant documents
                    question_embedding = embedding_client.generate_embedding(request.question, max_length=512)
                    existing_context = main_milvus.search(
                        query_embedding=question_embedding,
                        top_k=5,
                        metric_type="L2",
                        search_params={"nprobe": 10},
                        output_fields=["text", "metadata"]
                    )
                else:
                    logger.info(f"Topics somewhat similar but not close enough. Will scrape fresh content.")
            else:
                logger.info("No similar topics found. Will scrape fresh content.")
                
        except Exception as e:
            logger.warning(f"Error checking topics: {e}. Will proceed with scraping.")
            should_scrape = True
        
        # Initialize these variables for the response
        search_results = []
        scraped_content = []
        stored_in_milvus = False
        milvus_ids = []
        search_query = " ".join(topics)  # Define search_query here for use later
        
        # Step 3: Only scrape if we don't have similar topics already
        if should_scrape:
            # Perform Google Custom Search
            logger.info(f"Step 3: Searching Google for: {search_query}")
            search_results = google_search.search_detailed(query=search_query, num_results=5)
        
        # Step 3: Extract URLs from search results
        urls = [result.get('link') for result in search_results if result.get('link')]
        # Filter out None values
        urls = [url for url in urls if url is not None]
        logger.info(f"Step 3: Scraping {len(urls)} URLs")
        
        # Step 4: Scrape the URLs
        scraped_content = await web_scraper.scrape_urls(
            urls=urls,
            extract_markdown=True,
            extract_html=False,
            extract_links=False,
            max_concurrent=3
        )
        
        # Filter successful scrapes
        successful_scrapes = [
            scrape for scrape in scraped_content 
            if scrape.get('success') and scrape.get('markdown')
        ]
        logger.info(f"Successfully scraped {len(successful_scrapes)}/{len(urls)} URLs")
        
        # Step 5: Store in Milvus with embeddings
        milvus_ids = []
        stored_in_milvus = False
        
        if successful_scrapes:
            try:
                # Connect to Milvus and ensure collection exists
                milvus_client.connect()
                
                # Get embedding dimension
                embedding_dim = embedding_client.get_embedding_dimension()
                
                # Create collection if it doesn't exist
                milvus_client.create_collection(embedding_dim=embedding_dim)
                
                # Create index if needed
                try:
                    milvus_client.create_index(
                        index_type="IVF_FLAT",
                        metric_type="L2",
                        params={"nlist": 128}
                    )
                except Exception as index_error:
                    logger.warning(f"Index might already exist: {index_error}")
                
                # Prepare data for Milvus
                texts = []
                embeddings = []
                metadata_list = []
                
                for scrape in successful_scrapes:
                    markdown = scrape.get('markdown', '')
                    url = scrape.get('url', '')
                    
                    # Truncate text if too long (max 65535 chars for VARCHAR)
                    if len(markdown) > 60000:
                        markdown = markdown[:60000] + "... [truncated]"
                    
                    texts.append(markdown)
                    
                    # Generate embedding
                    embedding = embedding_client.generate_embedding(markdown, max_length=512)
                    embeddings.append(embedding)
                    
                    # Store metadata
                    metadata_list.append({
                        "url": url,
                        "query": search_query,
                        "topics": topics,
                        "metadata": scrape.get('metadata', {})
                    })
                
                # Insert into Milvus
                logger.info(f"Step 5: Storing {len(texts)} documents in Milvus")
                milvus_ids = milvus_client.insert(
                    texts=texts,
                    embeddings=embeddings,
                    metadata=metadata_list
                )
                stored_in_milvus = True
                logger.info(f"Successfully stored {len(milvus_ids)} documents in Milvus")
                
                # Also store the topics in the topics collection for future checks
                try:
                    topics_milvus = MilvusClient(collection_name="ai_firm_topics")
                    topics_milvus.connect()
                    topics_milvus.create_collection(embedding_dim=embedding_dim)
                    
                    try:
                        topics_milvus.create_index(
                            index_type="IVF_FLAT",
                            metric_type="L2",
                            params={"nlist": 128}
                        )
                    except:
                        pass  # Index might already exist
                    
                    # Store topics with their embedding
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
                    logger.info(f"Stored topics '{topics_text}' in topics collection")
                except Exception as topics_error:
                    logger.warning(f"Could not store topics: {topics_error}")
                
            except Exception as milvus_error:
                logger.error(f"Error storing in Milvus: {milvus_error}")
                # Continue even if Milvus storage fails
        
        # Step 6: Retrieve relevant context (use existing_context if we didn't scrape)
        retrieved_context = existing_context if existing_context else []
        
        if not retrieved_context:
            try:
                # Generate embedding for the question
                question_embedding = embedding_client.generate_embedding(request.question, max_length=512)
                
                # Search Milvus for similar documents
                logger.info("Step 6: Searching Milvus for relevant context")
                retrieved_context = milvus_client.search(
                    query_embedding=question_embedding,
                    top_k=5,
                    metric_type="L2",
                    search_params={"nprobe": 10},
                    output_fields=["text", "metadata"]
                )
                logger.info(f"Retrieved {len(retrieved_context)} documents from Milvus")
            except Exception as retrieval_error:
                logger.warning(f"Error retrieving from Milvus: {retrieval_error}")
                # Continue even if retrieval fails
        
        # Step 7: Build context from retrieved documents and query GPT-OSS
        llm_answer = ""
        try:
            # Build context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_context, 1):
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})
                url = metadata.get('url', 'Unknown source')
                
                # Truncate long texts for context
                if len(text) > 2000:
                    text = text[:2000] + "... [truncated]"
                
                context_parts.append(f"[Document {i} from {url}]\n{text}\n")
            
            context_text = "\n---\n".join(context_parts) if context_parts else "No relevant context found."
            
            # Build enhanced prompt for GPT-OSS
            enhanced_prompt = f"""You are a helpful AI assistant. Answer the following question using the provided context from web search results.

QUESTION: {request.question}

CONTEXT FROM WEB SEARCH:
{context_text}

Please provide a comprehensive answer based on the context above. If the context doesn't contain relevant information, say so and provide what you know about the topic."""

            logger.info("Step 7: Querying GPT-OSS with context")
            
            # Query GPT-OSS using the complete method
            llm_answer = await gpt_oss_client.complete(
                prompt=enhanced_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            logger.info("Successfully generated answer from GPT-OSS")
            
        except Exception as llm_error:
            logger.error(f"Error querying GPT-OSS: {llm_error}")
            llm_answer = f"Error generating answer: {str(llm_error)}"
        
        return IntelligentQueryResponse(
            success=True,
            topics=topics,
            search_results=search_results,
            scraped_content=successful_scrapes,
            stored_in_milvus=stored_in_milvus,
            milvus_ids=milvus_ids,
            retrieved_context=retrieved_context,
            llm_answer=llm_answer
        )
        
    except Exception as e:
        logger.error(f"Error in intelligent query processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing intelligent query: {str(e)}"
        )


@router.get("/status")
async def get_intelligent_query_status():
    """
    Get status of the intelligent query system
    """
    try:
        # Test GPT-OSS availability
        gpt_oss = GPTOSSClient()
        is_available = await gpt_oss.is_available()
        
        return {
            "status": "operational" if is_available else "degraded",
            "gpt_oss_available": is_available,
            "agents": {
                "question_analyzer": "ready",
                "google_search": "ready",
                "web_scraper": "ready",
                "milvus": "ready",
                "embeddings": "ready",
                "gpt_oss": "ready" if is_available else "unavailable"
            },
            "workflow_steps": [
                "1. Analyze question and extract topics",
                "2. Search Google for relevant URLs",
                "3. Scrape content from URLs",
                "4. Generate embeddings and store in Milvus",
                "5. Retrieve top 5 relevant documents",
                "6. Generate answer using GPT-OSS with context"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
