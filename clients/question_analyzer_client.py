"""
Question Analyzer Client
First agent in the pipeline that breaks down user questions into topics/concepts
Uses LLM-based analysis exclusively for accurate, intelligent question breakdown
"""

from typing import List, Dict, Any, Optional
import logging
import json
import re

logger = logging.getLogger(__name__)


class QuestionAnalyzerClient:
    """
    Analyzes user questions and extracts topics, concepts, and intent using LLM
    This is the first agent in the multi-agent workflow
    """
    
    def __init__(self, llm_client):
        """
        Initialize Question Analyzer
        
        Args:
            llm_client: LLM client for intelligent analysis (e.g., VLLMClient) - REQUIRED
        """
        if not llm_client:
            raise ValueError("LLM client is required for Question Analyzer")
        
        self.llm_client = llm_client
        logger.info("Question Analyzer initialized with LLM-based analysis")
    
    async def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze a user question using LLM to extract topics/concepts
        
        Args:
            question: The user's question
            
        Returns:
            Dict containing:
                - original_question: The original question
                - topics: List of identified topics/concepts
                - sub_questions: List of sub-questions extracted
                - technical_terms: Technical terminology identified
                - keywords: Important keywords
                - search_queries: Optimized search queries for Google Custom Search
                - intent: User's intent/goal
                - complexity: Simple/Medium/Complex
        """
        logger.info(f"Analyzing question with LLM: {question[:100]}...")
        
        # LLM-based analysis (only method)
        llm_analysis = await self._llm_analysis(question)
        
        return {
            "original_question": question,
            **llm_analysis,
            "analysis_method": "llm"
        }
    
    async def _llm_analysis(self, question: str) -> Dict[str, Any]:
        """
        Deep analysis using LLM to extract topics and concepts
        Returns structured JSON for reliable parsing
        """
        try:
            # Simple, clear instruction - accept that LLM may add reasoning
            analysis_prompt = f"""Analyze this question and return a JSON object with the analysis:

QUESTION: "{question}"

Return a JSON object with these fields:
- topics: array of main topics/concepts (2-5 items)
- sub_questions: array of sub-questions if complex
- technical_terms: array of technical terms found
- keywords: array of keywords for search
- search_queries: array of 3 optimized Google search queries
- intent: brief description of user intent
- complexity: "simple", "medium", or "complex"

JSON:"""

            logger.info("Requesting LLM to analyze question...")
            response = await self.llm_client.ask(
                question=analysis_prompt,
                temperature=0.2,
                max_tokens=600
            )
            
            logger.info(f"LLM response: {len(response)} characters")
            
            # Parse JSON response
            parsed = self._parse_json_response(response)
            
            return {
                "topics": parsed.get("topics", []),
                "sub_questions": parsed.get("sub_questions", []),
                "technical_terms": parsed.get("technical_terms", []),
                "keywords": parsed.get("keywords", []),
                "search_queries": parsed.get("search_queries", []),
                "intent": parsed.get("intent", ""),
                "complexity": parsed.get("complexity", "medium"),
                "raw_analysis": response
            }
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            raise Exception(f"Failed to analyze question: {e}")
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM with robust extraction
        """
        try:
            cleaned = response.strip()
            
            # Try direct JSON parsing first
            try:
                parsed = json.loads(cleaned)
                logger.info(f"✅ Parsed JSON directly: {len(parsed.get('topics', []))} topics")
                return self._validate_response(parsed)
            except json.JSONDecodeError:
                pass
            
            # Look for JSON object in the response
            # Use regex to find JSON-like structure
            patterns = [
                r'\{[^{}]*"topics"[^{}]*"sub_questions"[^{}]*"technical_terms"[^{}]*"keywords"[^{}]*"search_queries"[^{}]*"intent"[^{}]*"complexity"[^{}]*\}',
                r'\{[^{}]*"topics"[^{}]*"search_queries"[^{}]*\}',
                r'\{(?:[^{}]|\{[^{}]*\})*\}',
            ]
            
            json_text = None
            for pattern in patterns:
                match = re.search(pattern, cleaned, re.DOTALL)
                if match:
                    try:
                        json_text = match.group(0)
                        parsed = json.loads(json_text)
                        logger.info(f"✅ Extracted JSON with regex: {len(parsed.get('topics', []))} topics")
                        return self._validate_response(parsed)
                    except:
                        continue
            
            # Last resort: find first { and matching }
            start_idx = cleaned.find('{')
            if start_idx >= 0:
                brace_count = 0
                for i in range(start_idx, len(cleaned)):
                    if cleaned[i] == '{':
                        brace_count += 1
                    elif cleaned[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_text = cleaned[start_idx:i+1]
                            parsed = json.loads(json_text)
                            logger.info(f"✅ Extracted JSON with brace matching: {len(parsed.get('topics', []))} topics")
                            return self._validate_response(parsed)
            
            raise ValueError("No valid JSON found in response")
            
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response was: {response[:200]}...")
            raise Exception(f"Failed to parse JSON response: {e}")
    
    def _validate_response(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the parsed JSON response"""
        return {
            "topics": parsed.get("topics", []),
            "sub_questions": parsed.get("sub_questions", []),
            "technical_terms": parsed.get("technical_terms", []),
            "keywords": parsed.get("keywords", []),
            "search_queries": parsed.get("search_queries", []),
            "intent": parsed.get("intent", ""),
            "complexity": parsed.get("complexity", "medium")
        }


def create_question_analyzer(llm_client) -> QuestionAnalyzerClient:
    """
    Factory function to create a QuestionAnalyzerClient
    
    Args:
        llm_client: LLM client for intelligent analysis (REQUIRED)
        
    Returns:
        QuestionAnalyzerClient instance
    """
    return QuestionAnalyzerClient(llm_client=llm_client)
