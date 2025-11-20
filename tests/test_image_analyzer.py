"""
Integration tests for Image Analyzer Client
Tests the combination of Google Image Search + Qwen3-VL
"""

import pytest
from clients.image_analyzer_client import (
    ImageAnalyzerClient,
    AnalysisConfig,
    ImageAnalysisResult
)


class TestImageAnalyzerClient:
    """Test suite for ImageAnalyzerClient"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer with lazy loading (don't load VLM in tests)"""
        return ImageAnalyzerClient(load_vlm=False)
    
    @pytest.fixture
    def analyzer_with_vlm(self):
        """Create analyzer with VLM loaded (for integration tests)"""
        return ImageAnalyzerClient(load_vlm=True)
    
    def test_initialization_without_vlm(self, analyzer):
        """Test initialization with lazy loading"""
        assert analyzer.image_search is not None
        assert analyzer.vlm is None
        assert analyzer.tensor_parallel_size == 1
    
    def test_initialization_with_vlm(self):
        """Test initialization with VLM loading"""
        analyzer = ImageAnalyzerClient(load_vlm=True)
        assert analyzer.vlm is not None
        print("\n✓ VLM loaded successfully")
    
    def test_analysis_config_defaults(self):
        """Test default configuration"""
        config = AnalysisConfig()
        assert config.num_images == 5
        assert config.image_size == "large"
        assert config.image_type == "photo"
        assert config.temperature == 0.0
        assert config.max_tokens == 512
        assert config.batch_size == 5
    
    def test_search_and_describe(self, analyzer_with_vlm):
        """Integration test: Search and describe images"""
        results = analyzer_with_vlm.describe_images(
            query="neural network architecture diagram",
            num_images=3
        )
        
        assert len(results) > 0
        assert all(isinstance(r, ImageAnalysisResult) for r in results)
        assert all(r.image_url for r in results)
        assert all(r.analysis for r in results)
        
        print(f"\n✓ Analyzed {len(results)} images")
        for i, result in enumerate(results[:2], 1):
            print(f"\nImage {i}: {result.image_title}")
            print(f"Analysis: {result.analysis[:200]}...")
    
    def test_ocr_extraction(self, analyzer_with_vlm):
        """Integration test: Extract text from images"""
        results = analyzer_with_vlm.extract_text_from_images(
            query="code snippet python",
            num_images=2
        )
        
        assert len(results) > 0
        assert all(isinstance(r, ImageAnalysisResult) for r in results)
        
        print(f"\n✓ Extracted text from {len(results)} images")
        for i, result in enumerate(results, 1):
            print(f"\nImage {i}: {result.image_title}")
            print(f"Extracted text: {result.analysis[:150]}...")
    
    def test_visual_question_answering(self, analyzer_with_vlm):
        """Integration test: Answer questions about images"""
        results = analyzer_with_vlm.answer_visual_question(
            search_query="machine learning model architecture",
            question="What type of architecture or model is shown?",
            num_images=3
        )
        
        assert len(results) > 0
        assert all(r.analysis for r in results if not r.error)
        
        print(f"\n✓ Answered questions for {len(results)} images")
        for i, result in enumerate(results[:2], 1):
            print(f"\nImage {i}: {result.image_title}")
            print(f"Answer: {result.analysis}")
    
    def test_custom_analysis_config(self, analyzer_with_vlm):
        """Test custom analysis configuration"""
        config = AnalysisConfig(
            num_images=2,
            image_size="xlarge",
            image_type="photo",
            temperature=0.3,
            max_tokens=256,
            batch_size=2
        )
        
        results = analyzer_with_vlm.search_and_analyze(
            query="AI technology",
            analysis_question="What technology is depicted?",
            config=config
        )
        
        assert len(results) <= 2
        print(f"\n✓ Custom config test: {len(results)} images analyzed")
    
    def test_summary_generation(self, analyzer_with_vlm):
        """Test summary statistics"""
        results = analyzer_with_vlm.describe_images(
            query="artificial intelligence",
            num_images=3
        )
        
        summary = analyzer_with_vlm.get_summary(results)
        
        assert summary["total_images"] == len(results)
        assert summary["successful"] >= 0
        assert summary["failed"] >= 0
        assert summary["with_embeddings"] >= 0
        assert 0.0 <= summary["success_rate"] <= 1.0
        assert len(summary["images"]) == summary["successful"]
        
        print(f"\n✓ Summary generated")
        print(f"Total: {summary['total_images']}, Success: {summary['successful']}, Failed: {summary['failed']}")
        print(f"With embeddings: {summary['with_embeddings']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
    
    def test_batch_processing(self, analyzer_with_vlm):
        """Test batch processing with multiple images"""
        config = AnalysisConfig(
            num_images=5,
            batch_size=3  # Process 3 at a time
        )
        
        results = analyzer_with_vlm.search_and_analyze(
            query="data visualization charts",
            analysis_question="What type of chart or visualization is shown?",
            config=config
        )
        
        assert len(results) > 0
        print(f"\n✓ Batch processing test: {len(results)} images processed")
    
    def test_error_handling(self, analyzer_with_vlm):
        """Test error handling with invalid images"""
        # Force analysis with potentially problematic URLs
        results = analyzer_with_vlm.search_and_analyze(
            query="test image",
            analysis_question="Describe this",
            config=AnalysisConfig(num_images=2)
        )
        
        # Should handle errors gracefully
        assert isinstance(results, list)
        print(f"\n✓ Error handling: {len(results)} results returned")
    
    def test_multi_gpu_config(self):
        """Test multi-GPU configuration (planning for future)"""
        # Single GPU (current)
        analyzer_single = ImageAnalyzerClient(
            load_vlm=False,
            tensor_parallel_size=1
        )
        assert analyzer_single.tensor_parallel_size == 1
        
        # Multi-GPU (future expansion)
        analyzer_multi = ImageAnalyzerClient(
            load_vlm=False,
            tensor_parallel_size=2  # Would use 2 GPUs when available
        )
        assert analyzer_multi.tensor_parallel_size == 2
        
        print("\n✓ Multi-GPU configuration ready for expansion")
    
    def test_embedding_generation(self, analyzer_with_vlm):
        """Test automatic embedding generation"""
        results = analyzer_with_vlm.describe_images(
            query="machine learning diagram",
            num_images=2
        )
        
        # Check embeddings were generated
        assert len(results) > 0
        for result in results:
            if result.error is None:
                assert result.embedding is not None
                assert len(result.embedding) == 1024  # BGE-M3 dimension
        
        print(f"\n✓ Embeddings generated for {len(results)} images")
    
    def test_vectordb_storage(self, analyzer_with_vlm):
        """Test storing results in vector database"""
        # Analyze images
        results = analyzer_with_vlm.describe_images(
            query="neural network visualization",
            num_images=3
        )
        
        # Store in vector database
        storage_result = analyzer_with_vlm.store_in_vectordb(
            results=results,
            query="neural network visualization"
        )
        
        assert storage_result["stored"] > 0
        assert storage_result["collection"] == "image_analysis_retrieval"
        assert len(storage_result["ids"]) == storage_result["stored"]
        
        print(f"\n✓ Stored {storage_result['stored']} results in vector database")
        print(f"Collection: {storage_result['collection']}")
        print(f"IDs: {storage_result['ids'][:3]}...")
    
    def test_end_to_end_workflow(self, analyzer_with_vlm):
        """Test complete workflow: search → analyze → embed → store"""
        # Step 1: Search and analyze
        results = analyzer_with_vlm.search_and_analyze(
            query="AI technology visualization",
            analysis_question="What AI technology or concept is shown?",
            config=AnalysisConfig(num_images=3)
        )
        
        # Step 2: Verify embeddings
        successful = [r for r in results if r.error is None and r.embedding is not None]
        assert len(successful) > 0
        
        # Step 3: Store in vector database
        storage_result = analyzer_with_vlm.store_in_vectordb(results)
        
        # Step 4: Verify storage
        assert storage_result["stored"] == len(successful)
        
        print(f"\n✓ End-to-end workflow complete")
        print(f"Analyzed: {len(results)} images")
        print(f"With embeddings: {len(successful)}")
        print(f"Stored in DB: {storage_result['stored']}")
    
    def test_vectordb_retrieval(self, analyzer_with_vlm):
        """Test semantic search retrieval from vector database"""
        # First, store some data
        results = analyzer_with_vlm.describe_images(
            query="machine learning architecture",
            num_images=3
        )
        analyzer_with_vlm.store_in_vectordb(results, query="machine learning architecture")
        
        # Now search
        search_results = analyzer_with_vlm.search_vectordb(
            query="neural network diagram",
            top_k=3
        )
        
        assert len(search_results) > 0
        assert all("score" in r for r in search_results)
        assert all("image_url" in r for r in search_results)
        assert all("image_title" in r for r in search_results)
        assert all("analysis" in r for r in search_results)
        
        print(f"\n✓ Retrieved {len(search_results)} similar images")
        for i, r in enumerate(search_results[:2], 1):
            print(f"\n{i}. Score: {r['score']:.3f}")
            print(f"   Title: {r['image_title']}")
            print(f"   Analysis: {r['analysis'][:100]}...")


if __name__ == "__main__":
    # Run quick integration test
    print("Running Image Analyzer integration test...")
    
    analyzer = ImageAnalyzerClient(load_vlm=True)
    
    results = analyzer.describe_images(
        query="neural network visualization",
        num_images=3
    )
    
    print(f"\nAnalyzed {len(results)} images:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.image_title}")
        print(f"   Source: {result.image_source}")
        print(f"   Analysis: {result.analysis[:200]}...")
        print(f"   Has embedding: {result.embedding is not None}")
    
    # Store in vector database
    storage_result = analyzer.store_in_vectordb(results, query="neural network visualization")
    
    summary = analyzer.get_summary(results)
    print(f"\nSuccess rate: {summary['success_rate']:.1%}")
    print(f"Stored in DB: {storage_result['stored']} results")
