import asyncio
import sys
import os
import torch
import gc
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clients.google_image_search_client import GoogleImageSearchClient
from clients.image_analyzer_client import ImageAnalyzerClient
from clients.question_analyzer_client import QuestionAnalyzerClient
from clients.embedding_client import EmbeddingClient
from routes.synthesis_orchestrator import SynthesisOrchestrator
from routes.intelligent_query import IntelligentQuerySynthesisRequest

try:
    from clients.vllm_client import VLLMClient as LLMClient
except ImportError as e:
    raise ImportError(
        "vLLM client is required. Please install and configure vLLM."
    ) from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output file path
OUTPUT_FILE = "pipeline_results.txt"


def write_section_header(f, title, char="="):
    """Write a formatted section header to file"""
    line = char * 80
    f.write(f"\n{line}\n")
    f.write(f"{title}\n")
    f.write(f"{line}\n\n")


def write_result_entry(f, entry_num, image_url, question, context, answer):
    """Write a single result entry to file"""
    f.write(f"{'─' * 80}\n")
    f.write(f"ENTRY #{entry_num}\n")
    f.write(f"{'─' * 80}\n\n")
    
    f.write(f"IMAGE URL:\n{image_url}\n\n")
    
    f.write(f"QUESTION/PROMPT:\n{question}\n\n")
    
    f.write(f"RETRIEVED CONTEXT:\n{context}\n\n")
    
    f.write(f"SYNTHESIZED ANSWER:\n{answer}\n\n")
    
    f.write(f"SYNTHESIZED ANSWER:\n{answer}\n\n")


def clean_gpu():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"   [GPU] Memory cleaned. Free: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")


def print_gpu_memory():
    """Print current GPU memory status"""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        print(f"   [GPU] Used: {used/1e9:.2f} GB / {total/1e9:.2f} GB (Free: {free/1e9:.2f} GB)")


async def main():
    RAW_QUESTION = "Explain the architecture of the Triton Inference Server"
    NUM_IMAGES = 4
    
    print(f"\n=== STARTING INTELLIGENT VISUAL PIPELINE ===")
    print(f"Goal: Find diagrams for '{RAW_QUESTION}'\n")
    print_gpu_memory()

    # Initialize output file
    with open(OUTPUT_FILE, 'w') as f:
        f.write("INTELLIGENT VISUAL PIPELINE RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        write_section_header(f, "PIPELINE CONFIGURATION")
        f.write(f"Original Question: {RAW_QUESTION}\n")
        f.write(f"Number of Images: {NUM_IMAGES}\n")

    # ---------------------------------------------------------
    # PHASE 1: GENERATE OPTIMIZED QUERY (Text LLM)
    # ---------------------------------------------------------
    print("--- PHASE 1: Query Optimization (Loading Text LLM) ---")
    clean_gpu()
    
    generated_search_query = RAW_QUESTION
    llm_client = None
    analyzer = None
    
    try:
        llm_client = LLMClient(
            gpu_memory_utilization=0.6,
            max_model_len=8192
        )
        analyzer = QuestionAnalyzerClient(llm_client=llm_client)
        
        print(f"   [LLM] Analyzing question: '{RAW_QUESTION}'...")
        analysis_result = await analyzer.analyze_question(RAW_QUESTION)
        
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
        print("   [LLM] Unloading Text Model to free VRAM...")
        if analyzer is not None:
            del analyzer
        if llm_client is not None:
            del llm_client
        clean_gpu()
        print_gpu_memory()

    # Log optimized query to file
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"Optimized Search Query: {generated_search_query}\n")

    # ---------------------------------------------------------
    # PHASE 2: SEARCH GOOGLE (No GPU needed)
    # ---------------------------------------------------------
    print(f"\n--- PHASE 2: Google Image Search ---")
    print(f"   [SEARCH] Query: '{generated_search_query}'")
    
    google_img_client = GoogleImageSearchClient()
    search_results = google_img_client.search_images(query=generated_search_query, num_results=NUM_IMAGES)
    
    if not search_results:
        print("   [ERROR] No images found. Exiting.")
        with open(OUTPUT_FILE, 'a') as f:
            f.write("\n[ERROR] No images found. Pipeline terminated.\n")
        return

    print(f"   [SEARCH] Found {len(search_results)} images.")
    
    # Log search results to file
    with open(OUTPUT_FILE, 'a') as f:
        write_section_header(f, "IMAGE SEARCH RESULTS")
        for i, res in enumerate(search_results, 1):
            url = res.get('link', 'N/A')
            title = res.get('title', 'N/A')
            f.write(f"  [{i}] {title}\n      URL: {url}\n\n")
            print(f"    {i}. {url[:60]}...")

    # ---------------------------------------------------------
    # PHASE 3: VISUAL ANALYSIS (Load VLM)
    # ---------------------------------------------------------
    print(f"\n--- PHASE 3: Visual Analysis (Loading VLM) ---")
    print_gpu_memory()
    
    successful_results = []
    image_analyses = []  # Store (preview, image_url) for synthesis
    vlm_analyzer = None
    
    try:
        vlm_analyzer = ImageAnalyzerClient(
            load_vlm=True, 
            gpu_memory_utilization=0.8, 
            enable_embeddings=True,
            max_model_len=8192
        )
        
        print(f"   [VLM] Analyzing {len(search_results)} images from Phase 2...")
        
        # Build the analysis config
        from clients.image_analyzer_client import AnalysisConfig
        config = AnalysisConfig(
            num_images=NUM_IMAGES,
            temperature=0.3,
            max_tokens=512
        )
        
        # Analysis prompt (same as describe_images uses)
        analysis_prompt = """
        Analyze this architecture diagram and describe it as if you were writing the official technical documentation.
        
        1. Identify the key components (e.g. Client, Server, Repository) and explicitly explain how they interact.
        2. Describe the data flow direction (e.g. "Requests enter via gRPC...").
        3. Use precise technical verbs (e.g. "orchestrates," "distributes," "loads," "communicates").
        
        Do not use bullet points. Write a dense, information-rich paragraph describing the system architecture shown.
        """
        
        # Analyze using the SAME URLs from Phase 2 (not a new search)
        analyzed_results = vlm_analyzer._analyze_batch(
            images=search_results,  # Pass the original search results directly
            question=analysis_prompt,
            config=config
        )
        
        successful_results = [r for r in analyzed_results if not getattr(r, 'error', None)]
        
        if not successful_results:
            print("   [ERROR] VLM failed to analyze images.")
            with open(OUTPUT_FILE, 'a') as f:
                f.write("\n[ERROR] VLM failed to analyze images. Pipeline terminated.\n")
            return

        print(f"   [VLM] Successfully analyzed {len(successful_results)} images.")
        
        # Collect image analyses and URLs
        for idx, res in enumerate(successful_results):
            preview = res.analysis.replace('\n', ' ')
            image_url = res.image_url  # Now this is correct - from the same search results
            
            image_analyses.append((preview, image_url))
            print(f"    [{idx+1}] URL: {image_url[:60]}...")
            print(f"        Analysis: {preview[:100]}...")

        # Store in Vector DB
        print("\n   [DB] Storing into Milvus...")
        store_response = vlm_analyzer.store_in_vectordb(successful_results, query=generated_search_query)
        ids = store_response.get("ids", [])
        
        if ids:
            print(f"   [SUCCESS] Stored {len(ids)} visual contexts. IDs: {ids}")
        else:
            print("   [WARNING] Storage reported 0 inserts.")

    except Exception as e:
        print(f"   [ERROR] Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n   [CLEANUP] Unloading VLM before synthesis...")
        if vlm_analyzer is not None:
            del vlm_analyzer
        clean_gpu()
        print_gpu_memory()

    # ---------------------------------------------------------
    # PHASE 4: SYNTHESIS using SynthesisOrchestrator
    # ---------------------------------------------------------
    synthesis_results = []
    
    if image_analyses:
        print(f"\n--- PHASE 4: Synthesis (Loading Synthesis LLM) ---")
        print_gpu_memory()
        
        synthesis_llm = None
        embedding_client = None
        orchestrator = None
        
        try:
            # Initialize embedding client for retrieval
            embedding_client = EmbeddingClient()
            print("   [EMBED] Embedding client initialized")
            
            # Initialize synthesis LLM
            synthesis_llm = LLMClient(
                gpu_memory_utilization=0.6,
                max_model_len=8192
            )
            print("   [LLM] Synthesis LLM initialized")
            
            # Initialize the SynthesisOrchestrator
            orchestrator = SynthesisOrchestrator(
                embedding_client=embedding_client,
                llm_client=synthesis_llm
            )
            print("   [ORCHESTRATOR] SynthesisOrchestrator initialized")
            
            for idx, (preview, image_url) in enumerate(image_analyses):
                print(f"\n   Processing image {idx + 1}/{len(image_analyses)}...")
                
                # Create synthesis request using the image analysis as the question
                synth_request = IntelligentQuerySynthesisRequest(
                    question=preview,  # Use image analysis as the query
                    temperature=0.0,
                    max_tokens=512,
                    include_image_search=True,  # Include visual context
                    image_query=None,
                    image_num_results=0,
                    enable_image_analysis=False,
                    image_analysis_question=None,
                    store_image_analysis=False
                )
                
                try:
                    # Run the orchestrator (Steps 5-6: Retrieve + Synthesize)
                    # Pass empty list for existing_context to trigger fresh retrieval
                    result = await orchestrator.run(
                        request=synth_request,
                        existing_context=[]  # Empty = retrieval_service.retrieve_all_context() will be called
                    )
                    
                    retrieved_context = result.get('retrieved_context', [])
                    answer = result.get('llm_answer', '')
                    
                    print(f"    [RETRIEVE] Found {len(retrieved_context)} context items")
                    print(f"    [SYNTHESIS] Answer: {answer[:150]}...")
                    
                    # Store for file output
                    synthesis_results.append({
                        'image_url': image_url,
                        'question': preview,
                        'context': retrieved_context,
                        'answer': answer
                    })
                    
                except Exception as e:
                    print(f"    [ERROR] Synthesis failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    synthesis_results.append({
                        'image_url': image_url,
                        'question': preview,
                        'context': [],
                        'answer': f"[ERROR] Synthesis failed: {e}"
                    })
                    
        except Exception as e:
            print(f"   [ERROR] Phase 4 failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            print("\n   [CLEANUP] Unloading Synthesis components...")
            if orchestrator is not None:
                del orchestrator
            if synthesis_llm is not None:
                del synthesis_llm
            if embedding_client is not None:
                del embedding_client
            clean_gpu()

    # ---------------------------------------------------------
    # WRITE ALL RESULTS TO FILE
    # ---------------------------------------------------------
    print(f"\n--- Writing Results to {OUTPUT_FILE} ---")
    
    with open(OUTPUT_FILE, 'a') as f:
        write_section_header(f, "SYNTHESIS RESULTS")
        
        if synthesis_results:
            for idx, result in enumerate(synthesis_results, 1):
                write_result_entry(
                    f,
                    entry_num=idx,
                    image_url=result['image_url'],
                    question=result['question'],
                    context=result['context'],
                    answer=result['answer']
                )
        else:
            f.write("No synthesis results generated.\n")
        
        # Summary section
        write_section_header(f, "SUMMARY", char="-")
        f.write(f"Total Images Searched: {len(search_results)}\n")
        f.write(f"Successfully Analyzed: {len(successful_results)}\n")
        f.write(f"Synthesis Results: {len(synthesis_results)}\n")
        f.write(f"Pipeline Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"   [SUCCESS] Results written to {OUTPUT_FILE}")
    
    # ---------------------------------------------------------
    # WRITE INDIVIDUAL MARKDOWN FILES FOR EACH IMAGE
    # ---------------------------------------------------------
    print(f"\n--- Writing Individual Markdown Files ---")
    
    # Create output directory for markdown files
    md_output_dir = "image_reports"
    os.makedirs(md_output_dir, exist_ok=True)
    
    for idx, result in enumerate(synthesis_results, 1):
        md_filename = os.path.join(md_output_dir, f"image_{idx}_report.md")
        
        with open(md_filename, 'w') as f:
            # Title
            f.write(f"# Image Analysis Report #{idx}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Original Query:** {RAW_QUESTION}\n\n")
            f.write("---\n\n")
            
            # Image section
            f.write("## Image\n\n")
            f.write(f"![Image {idx}]({result['image_url']})\n\n")
            f.write(f"**URL:** {result['image_url']}\n\n")
            f.write("---\n\n")
            
            # VLM Analysis section
            f.write("## VLM Analysis (Image Description)\n\n")
            f.write(f"{result['question']}\n\n")
            f.write("---\n\n")
            
            # Synthesized Answer section
            f.write("## Synthesized Answer\n\n")
            f.write(f"{result['answer']}\n\n")
        
        print(f"   [CREATED] {md_filename}")
    
    # Create an index/summary markdown file
    index_filename = os.path.join(md_output_dir, "README.md")
    with open(index_filename, 'w') as f:
        f.write("# Image Analysis Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Query:** {RAW_QUESTION}\n\n")
        f.write(f"**Optimized Search Query:** {generated_search_query}\n\n")
        f.write(f"**Total Images Analyzed:** {len(synthesis_results)}\n\n")
        f.write("---\n\n")
        f.write("## Images\n\n")
        
        for idx, result in enumerate(synthesis_results, 1):
            f.write(f"### {idx}. [Image Report {idx}](image_{idx}_report.md)\n\n")
            f.write(f"![Image {idx}]({result['image_url']})\n\n")
            # Short preview of answer
            answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            f.write(f"**Preview:** {answer_preview}\n\n")
            f.write("---\n\n")
    
    print(f"   [CREATED] {index_filename}")
    print(f"   [SUCCESS] {len(synthesis_results)} markdown reports written to '{md_output_dir}/' directory")
    
    print_gpu_memory()
    print("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    asyncio.run(main())