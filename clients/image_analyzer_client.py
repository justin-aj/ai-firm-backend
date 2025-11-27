"""
Image Analyzer Client
Integrates Google Image Search with Qwen3-VL for intelligent image analysis

Architecture:
- Single GPU mode: Load VLM once, reuse for all images (memory efficient)
- Multi-GPU mode: Can be expanded with tensor parallelism
"""

import logging
import hashlib
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from clients.google_image_search_client import GoogleImageSearchClient
from clients.qwen3_vl_client import Qwen3VLClient
from clients.embedding_client import EmbeddingClient
from clients.milvus_client import MilvusClient

logger = logging.getLogger(__name__)


@dataclass
class ImageAnalysisResult:
    """Result from analyzing a single image"""
    image_url: str
    image_title: str
    image_source: str
    analysis: str
    embedding: Optional[List[float]] = None
    error: Optional[str] = None


@dataclass
class AnalysisConfig:
    """Configuration for image analysis"""
    num_images: int = 5
    image_size: Optional[str] = "large"  # Prefer high-quality images
    image_type: Optional[str] = "photo"  # Photos for better analysis
    temperature: float = 0.0  # Deterministic for consistency
    max_tokens: int = 512
    batch_size: int = 5  # Process images in batches (GPU memory consideration)


class ImageAnalyzerClient:
    """
    Intelligent image analysis combining Google Image Search + Qwen3-VL
    
    Workflow:
    1. Search for images using Google Custom Search API
    2. Analyze images using Qwen3-VL vision-language model
    3. Extract information, answer questions, or describe images
    
    GPU Strategy:
    - Single GPU: Load VLM once, process images sequentially or in small batches
    - Multi-GPU: Use tensor_parallel_size > 1 in Qwen3VLClient
    """
    
    def __init__(
        self,
        load_vlm: bool = True,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        enable_embeddings: bool = True,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        max_model_len: int = 8192  # <--- ADDED: Default to 8k to save memory
    ):
        """
        Initialize Image Analyzer
        
        Args:
            load_vlm: Whether to load VLM immediately (False for lazy loading)
            gpu_memory_utilization: GPU memory fraction for VLM (0.0-1.0)
            tensor_parallel_size: Number of GPUs (1 for single GPU)
            enable_embeddings: Whether to generate embeddings for analysis results
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            max_model_len: Maximum context length for VLM (lower this to save VRAM)
        """
        self.image_search = GoogleImageSearchClient()
        self.vlm = None
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_embeddings = enable_embeddings
        self.max_model_len = max_model_len  # <--- Store it
        
        # Initialize embedding client if enabled
        self.embedding_client = EmbeddingClient() if enable_embeddings else None
        
        # Initialize Milvus client (lazy connection)
        self.milvus_client = MilvusClient(
            host=milvus_host,
            port=milvus_port,
            collection_name="image_analysis_retrieval"
        )
        
        if load_vlm:
            self._initialize_vlm()
        
        logger.info(f"ImageAnalyzerClient initialized (GPU count: {tensor_parallel_size}, Embeddings: {enable_embeddings})")

    def _initialize_vlm(self):
        """Initialize VLM (lazy loading to save memory)"""
        if self.vlm is None:
            logger.info(f"Loading Qwen3-VL model (Context: {self.max_model_len})...")
            self.vlm = Qwen3VLClient(
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len  # <--- Pass it down
            )
            logger.info("Qwen3-VL loaded successfully")
    
    def search_and_analyze(
        self,
        query: str,
        analysis_question: str,
        config: Optional[AnalysisConfig] = None
    ) -> List[ImageAnalysisResult]:
        """
        Search for images and analyze them with VLM
        
        Args:
            query: Search query for finding images
            analysis_question: Question to ask about each image
            config: Analysis configuration
        
        Returns:
            List of analysis results for each image
        
        Example:
            results = analyzer.search_and_analyze(
                query="AI neural networks diagram",
                analysis_question="Describe the architecture shown in this diagram"
            )
        """
        if config is None:
            config = AnalysisConfig()
        
        # Ensure VLM is loaded
        self._initialize_vlm()
        
        # Step 1: Search for images
        logger.info(f"Searching for '{query}' (up to {config.num_images} images)")
        images = self.image_search.search_images(
            query=query,
            num_results=config.num_images,
            image_size=config.image_size,
            image_type=config.image_type
        )
        
        if not images:
            logger.warning(f"No images found for query: {query}")
            return []
        
        logger.info(f"Found {len(images)} images, analyzing with Qwen3-VL...")
        
        # Step 2: Analyze images (batch processing for efficiency)
        results = []
        for i in range(0, len(images), config.batch_size):
            batch = images[i:i + config.batch_size]
            batch_results = self._analyze_batch(batch, analysis_question, config)
            results.extend(batch_results)
        
        logger.info(f"Completed analysis of {len(results)} images")
        return results
    
    def _analyze_batch(
        self,
        images: List[Dict[str, Any]],
        question: str,
        config: AnalysisConfig
    ) -> List[ImageAnalysisResult]:
        """Analyze a batch of images"""
        results = []
        
        # Prepare batch for VLM (vLLM supports batch inference)
        messages_batch = []
        for img in images:
            messages_batch.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img["link"]},
                        {"type": "text", "text": question}
                    ]
                }
            ])
        
        try:
            # Batch inference (efficient for multiple images)
            analyses = self.vlm.batch_generate(
                messages_batch,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            # Package results with embeddings
            for img, analysis in zip(images, analyses):
                # Generate embedding from combined text (industry best practice)
                embedding = None
                if self.enable_embeddings and self.embedding_client:
                    combined_text = f"{img['title']}. {analysis}"
                    try:
                        embedding = self.embedding_client.generate_embedding(combined_text)
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding: {e}")
                
                results.append(ImageAnalysisResult(
                    image_url=img["link"],
                    image_title=img["title"],
                    image_source=img["displayLink"],
                    analysis=analysis,
                    embedding=embedding,
                    error=None
                ))
        
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            # Fallback to sequential processing
            for img in images:
                result = self._analyze_single(img, question, config)
                results.append(result)
        
        return results
    
    def _analyze_single(
        self,
        image: Dict[str, Any],
        question: str,
        config: AnalysisConfig
    ) -> ImageAnalysisResult:
        """Analyze a single image (fallback)"""
        try:
            analysis = self.vlm.analyze_image(
                image_url=image["link"],
                question=question,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            # Generate embedding
            embedding = None
            if self.enable_embeddings and self.embedding_client:
                combined_text = f"{image['title']}. {analysis}"
                try:
                    embedding = self.embedding_client.generate_embedding(combined_text)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
            
            return ImageAnalysisResult(
                image_url=image["link"],
                image_title=image["title"],
                image_source=image["displayLink"],
                analysis=analysis,
                embedding=embedding,
                error=None
            )
        
        except Exception as e:
            logger.error(f"Failed to analyze image {image['link']}: {e}")
            return ImageAnalysisResult(
                image_url=image["link"],
                image_title=image["title"],
                image_source=image["displayLink"],
                analysis="",
                embedding=None,
                error=str(e)
            )
    
    def extract_text_from_images(
        self,
        query: str,
        num_images: int = 5
    ) -> List[ImageAnalysisResult]:
        """
        Search for images and extract text (OCR)
        
        Args:
            query: Search query
            num_images: Number of images to analyze
        
        Returns:
            List of results with extracted text
        """
        config = AnalysisConfig(
            num_images=num_images,
            temperature=0.0,
            max_tokens=2048  # More tokens for text extraction
        )
        
        return self.search_and_analyze(
            query=query,
            analysis_question="Extract all text visible in this image. Be comprehensive.",
            config=config
        )
    
    def describe_images(
        self,
        query: str,
        num_images: int = 5
    ) -> List[ImageAnalysisResult]:
        """
        Search for images and generate descriptions
        
        Args:
            query: Search query
            num_images: Number of images to describe
        
        Returns:
            List of results with descriptions
        """
        config = AnalysisConfig(
            num_images=num_images,
            temperature=0.3,  # Slightly creative
            max_tokens=512
        )
        
        # CHANGED PROMPT: Optimized for SEMANTIC Vector Retrieval
        # We ask the VLM to "simulate" the documentation.
        # This aligns the vector space of the image description with your actual text docs.
        prompt = """
        Analyze this architecture diagram and describe it as if you were writing the official technical documentation.
        
        1. Identify the key components (e.g. Client, Server, Repository) and explicitly explain how they interact.
        2. Describe the data flow direction (e.g. "Requests enter via gRPC...").
        3. Use precise technical verbs (e.g. "orchestrates," "distributes," "loads," "communicates").
        
        Do not use bullet points. Write a dense, information-rich paragraph describing the system architecture shown.
        """
        
        return self.search_and_analyze(
            query=query,
            analysis_question=prompt,
            config=config
        )
    
    def answer_visual_question(
        self,
        search_query: str,
        question: str,
        num_images: int = 5
    ) -> List[ImageAnalysisResult]:
        """
        Search for images and answer a specific question about them
        
        Args:
            search_query: Query to find relevant images
            question: Specific question to answer about each image
            num_images: Number of images to analyze
        
        Returns:
            List of results with answers
        """
        config = AnalysisConfig(
            num_images=num_images,
            temperature=0.0,
            max_tokens=256
        )
        
        return self.search_and_analyze(
            query=search_query,
            analysis_question=question,
            config=config
        )
    
    def search_vectordb(
        self,
        query: str,
        top_k: int = 5,
        filter_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search vector database for similar image analyses
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            filter_query: Optional metadata filter (e.g., 'category == "image_analysis"')
        
        Returns:
            List of similar image analysis results with scores
        
        Example:
            results = analyzer.search_vectordb(
                query="transformer attention mechanism",
                top_k=5
            )
            for r in results:
                print(f"Score: {r['score']}, Image: {r['image_title']}")
        """
        if not self.enable_embeddings:
            raise ValueError("Embeddings are disabled. Initialize with enable_embeddings=True")
        
        # Connect to Milvus
        self.milvus_client.connect()
        
        # Load collection
        from pymilvus import Collection
        self.milvus_client.collection = Collection("image_analysis_retrieval")
        
        # Generate query embedding
        logger.info(f"Searching for: '{query}'")
        query_embedding = self.embedding_client.generate_embedding(query)
        
        # Search vector database
        search_results = self.milvus_client.search(
            query_embedding=query_embedding,
            top_k=top_k,
            output_fields=["text", "metadata"]
        )
        
        # Format results for easy access
        formatted_results = []
        for result in search_results:
            metadata = result.get("metadata", {})
            formatted_results.append({
                "id": result["id"],
                "score": result["score"],
                "image_url": metadata.get("image_url", ""),
                "image_title": metadata.get("image_title", ""),
                "image_source": metadata.get("image_source", ""),
                "analysis": result["text"],
                "search_query": metadata.get("search_query", ""),
                "category": metadata.get("category", ""),
                "metadata": metadata
            })
        
        logger.info(f"Found {len(formatted_results)} similar results")
        return formatted_results
    
    def store_in_vectordb(
        self,
        results: List[ImageAnalysisResult],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store image analysis results in Milvus vector database
        
        Industry best practice: Batch insert with fallback to individual inserts
        
        Args:
            results: List of analysis results with embeddings
            query: Original search query (stored in metadata)
        
        Returns:
            Storage statistics and inserted IDs
        """
        if not self.enable_embeddings:
            raise ValueError("Embeddings are disabled. Initialize with enable_embeddings=True")
        
        # Filter results with valid embeddings
        valid_results = [r for r in results if r.embedding is not None and r.error is None]
        
        if not valid_results:
            logger.warning("No valid results with embeddings to store")
            return {
                "stored": 0,
                "failed": len(results),
                "ids": []
            }
        
        # Connect and ensure collection exists
        self.milvus_client.connect()
        
        # Create collection if it doesn't exist (1024-dim for BGE-M3)
        embedding_dim = len(valid_results[0].embedding)
        self.milvus_client.create_collection(
            embedding_dim=embedding_dim,
            description="Image analysis results with VLM-generated descriptions"
        )
        self.milvus_client.create_index()
        
        # Prepare data for batch insert
        # Use image URL hash as ID (industry standard for deduplication)
        texts = []
        embeddings = []
        metadata_list = []
        
        for result in valid_results:
            # Combined text: title + analysis (what we embedded)
            combined_text = f"{result.image_title}. {result.analysis}"
            texts.append(combined_text)
            embeddings.append(result.embedding)
            
            # Minimal metadata (following user's example format)
            metadata = {
                "image_url": result.image_url,
                "image_title": result.image_title,
                "image_source": result.image_source,
                "category": "image_analysis"
            }
            
            if query:
                metadata["search_query"] = query
            
            metadata_list.append(metadata)
        
        # Try batch insert first (best practice)
        try:
            logger.info(f"Batch inserting {len(valid_results)} image analysis results")
            ids = self.milvus_client.insert(
                texts=texts,
                embeddings=embeddings,
                metadata=metadata_list
            )
            
            logger.info(f"Successfully stored {len(ids)} results in vector database")
            return {
                "stored": len(ids),
                "failed": 0,
                "ids": ids,
                "collection": "image_analysis_retrieval"
            }
        
        except Exception as e:
            logger.warning(f"Batch insert failed: {e}. Falling back to individual inserts")
            
            # Fallback: Insert one by one
            inserted_ids = []
            failed_count = 0
            
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadata_list)):
                try:
                    ids = self.milvus_client.insert(
                        texts=[text],
                        embeddings=[embedding],
                        metadata=[metadata]
                    )
                    inserted_ids.extend(ids)
                except Exception as e:
                    logger.error(f"Failed to insert result {i}: {e}")
                    failed_count += 1
            
            logger.info(f"Individual insert: {len(inserted_ids)} succeeded, {failed_count} failed")
            return {
                "stored": len(inserted_ids),
                "failed": failed_count,
                "ids": inserted_ids,
                "collection": "image_analysis_retrieval"
            }
    
    def get_summary(self, results: List[ImageAnalysisResult]) -> Dict[str, Any]:
        """
        Generate summary statistics from analysis results
        
        Args:
            results: List of analysis results
        
        Returns:
            Summary statistics
        """
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        with_embeddings = [r for r in successful if r.embedding is not None]
        
        return {
            "total_images": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "with_embeddings": len(with_embeddings),
            "success_rate": len(successful) / len(results) if results else 0,
            "images": [
                {
                    "url": r.image_url,
                    "title": r.image_title,
                    "source": r.image_source,
                    "analysis": r.analysis[:100] + "..." if len(r.analysis) > 100 else r.analysis,
                    "has_embedding": r.embedding is not None
                }
                for r in successful
            ]
        }
