import logging
import logging

logger = logging.getLogger(__name__)

async def generate_answer(llm_client, question: str, context_list: list, temperature: float, max_tokens: int) -> str:
    """Constructs the prompt and queries the LLM."""
    
    # 1. Format Context
    context_parts = []
    for i, doc in enumerate(context_list, 1):
        text = doc.get("text", "")
        meta = doc.get("metadata", {})
        source = meta.get("url") or meta.get("source") or "Unknown"
        
        # Truncate very long individual documents so they don't hog the context window
        if len(text) > 3000: 
            text = text[:3000] + "... [truncated]"
            
        context_parts.append(f"[Document {i} from {source}]\n{text}\n")

    context_str = "\n---\n".join(context_parts) if context_parts else "No relevant context found."

    # 2. Build Prompt
    enhanced_prompt = f"""You are a helpful AI assistant. Answer the following question using the provided context.

QUESTION: {question}

CONTEXT (Includes Web Search and Visual Analysis):
{context_str}

Please provide a comprehensive answer based on the context above. Citations are encouraged."""

    # 3. Debug logging
    try:
        with open("last_prompt_context.txt", "w", encoding="utf-8") as f:
            f.write(enhanced_prompt)
    except: pass

    # 4. Generate
    try:
        return await llm_client.complete(
            prompt=enhanced_prompt, 
            temperature=temperature, 
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return f"Error generating answer: {e}"