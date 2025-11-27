import asyncio
import sys
import os
import torch
import gc
import logging

# Ensure Python can find the 'clients' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports
from clients.google_image_search_client import GoogleImageSearchClient
from clients.image_analyzer_client import ImageAnalyzerClient
from clients.question_analyzer_client import QuestionAnalyzerClient
from routes.intelligent_query_service import run_synthesis_phase
from routes.intelligent_query import IntelligentQuerySynthesisRequest

# Import the LLM Client for Step 1 (vLLM only)
try:
    from clients.vllm_client import VLLMClient as LLMClient
except ImportError as e:
    # vLLM is required for this tool; fail fast with an actionable message
    raise ImportError(
        "vLLM client is required. Please install and configure vLLM. See README for details."
    ) from e

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_gpu():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    torch.cuda.empty_cache()
    print("   [GPU] Memory cleaned.")

async def main():
    # ---------------------------------------------------------
    # INPUT: The raw user intent
    # ---------------------------------------------------------
    RAW_QUESTION = "Explain the architecture of the Triton Inference Server"
    NUM_IMAGES = 4
    
    print(f"\n=== STARTING INTELLIGENT VISUAL PIPELINE ===")
    print(f"Goal: Find diagrams for '{RAW_QUESTION}'\n")

    # ---------------------------------------------------------
    # PHASE 1: GENERATE OPTIMIZED QUERY (Text LLM)
    # ---------------------------------------------------------
    print("--- PHASE 1: Query Optimization (Loading Text LLM) ---")
    clean_gpu()
    
    generated_search_query = RAW_QUESTION # Fallback
    
    try:
        # 1. Initialize Text LLM (Safe settings for V100)
        llm_client = LLMClient(
            gpu_memory_utilization=0.6,
            max_model_len=8192
        )
        analyzer = QuestionAnalyzerClient(llm_client=llm_client)
        
        # 2. Analyze the raw question
        print(f"   [LLM] Analyzing question: '{RAW_QUESTION}'...")
        analysis_result = await analyzer.analyze_question(RAW_QUESTION)
        
        # 3. Extract the best query
        optimized_queries = analysis_result.get("search_queries", [])
        if optimized_queries:
            generated_search_query = optimized_queries[0]
            print(f"   [LLM] Optimization Success! New Query: '{generated_search_query}'")
        else:
            print("   [LLM] No optimized queries found. Using raw topics.")
            topics = analysis_result.get("topics", [])
            if topics:
                generated_search_query = " ".join(topics) + " diagram"

    except Exception as e:
        print(f"   [ERROR] Phase 1 failed: {e}")
        
    finally:
        # 4. CRITICAL: DESTROY TEXT LLM
        print("   [LLM] Unloading Text Model to free VRAM...")
        del analyzer
        del llm_client
        clean_gpu()

    # ---------------------------------------------------------
    # PHASE 2: SEARCH GOOGLE (No GPU needed)
    # ---------------------------------------------------------
    print(f"\n--- PHASE 2: Google Image Search ---")
    print(f"   [SEARCH] Query: '{generated_search_query}'")
    
    google_img_client = GoogleImageSearchClient()
    search_results = google_img_client.search_images(query=generated_search_query, num_results=NUM_IMAGES)
    
    if not search_results:
        print("   [ERROR] No images found. Exiting.")
        return

    print(f"   [SEARCH] Found {len(search_results)} images.")
    for i, res in enumerate(search_results):
        print(f"    {i+1}. {res.get('link')[:60]}...")

    # ---------------------------------------------------------
    # PHASE 3: VISUAL ANALYSIS (Load VLM)
    # ---------------------------------------------------------
    print(f"\n--- PHASE 3: Visual Analysis (Loading VLM) ---")
    
    try:
        # 1. Initialize VLM (High memory utilization allowed now since Text LLM is gone)
        analyzer = ImageAnalyzerClient(
            load_vlm=True, 
            gpu_memory_utilization=0.8, 
            enable_embeddings=True,
            max_model_len=8192 # Critical fix for V100
        )
        
        # 2. Describe Images
        print(f"   [VLM] Analyzing {len(search_results)} images...")
        analyzed_results = analyzer.describe_images(
            query=generated_search_query, 
            num_images=NUM_IMAGES
        )
        
        successful_results = [r for r in analyzed_results if not getattr(r, 'error', None)]
        
        if not successful_results:
            print("   [ERROR] VLM failed to analyze images.")
            return

        print(f"   [VLM] Successfully analyzed {len(successful_results)} images.")
        for res in successful_results:
            preview = res.analysis.replace('\n', ' ')
            print(f"    > {preview}...")

            # Retrieve related context for this visual result from the vector DB
            try:
                related_context = analyzer.search_vectordb(query=preview, top_k=5)
            except Exception as e:
                related_context = []
                print(f"    [WARN] Failed to search vectordb for preview: {e}")

            # Build synthesis request using the image analysis as the question
            synth_request = IntelligentQuerySynthesisRequest(
                question=preview,
                temperature=0.0,
                max_tokens=256,
                include_image_search=False,
                image_query=None,
                image_num_results=0,
                enable_image_analysis=False,
                image_analysis_question=None,
                store_image_analysis=False
            )

                # Use a single lightweight vLLM for per-image synthesis to reduce re-loads
                # NOTE: This LLM should be small and configured to be low-memory. If vLLM isn't available
                # the script will skip synthesis per image.
                if 'synthesis_llm' not in locals():
                    try:
                        synthesis_llm = LLMClient(gpu_memory_utilization=0.05, max_model_len=2048, num_speculative_tokens=1)
                        logger.info("Synthesis vLLM initialized for per-image synthesis")
                    except Exception as e:
                        synthesis_llm = None
                        print(f"    [WARN] Synthesis vLLM initialization failed: {e}")

            # Execute synthesis using the retrieved vectordb context
            if synthesis_llm is not None:
                try:
                    synthesis_result = await run_synthesis_phase(synth_request, related_context, analyzer.embedding_client, synthesis_llm)
                    print(f"    [SYNTHESIS] Answer: {synthesis_result.get('llm_answer', '')}")
                except Exception as e:
                    print(f"    [ERROR] Synthesis failed for image preview: {e}")

        # 3. Store in Vector DB
        print("\n   [DB] Storing into Milvus...")
        store_response = analyzer.store_in_vectordb(successful_results, query=generated_search_query)
        ids = store_response.get("ids", [])
        
        if ids:
            print(f"   [SUCCESS] Stored {len(ids)} visual contexts. IDs: {ids}")
        else:
            print("   [WARNING] Storage reported 0 inserts.")

    except Exception as e:
        print(f"   [ERROR] Phase 3 failed: {e}")
        
    finally:
        # 4. Cleanup
        print("\n   [CLEANUP] Unloading VLM...")
        if 'analyzer' in locals():
            del analyzer
        if 'synthesis_llm' in locals() and synthesis_llm is not None:
            try:
                del synthesis_llm
            except Exception:
                pass
        clean_gpu()
        print("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(main())