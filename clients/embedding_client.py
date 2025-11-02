import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for generating text embeddings using BGE-M3 model"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialize the embedding model
        
        Args:
            model_name: HuggingFace model identifier
                Default: BAAI/bge-m3 (multilingual, 1024-dim, up to 8192 tokens)
                Alternative: BAAI/bge-base-en-v1.5 (English only, 768-dim)
                Alternative: sentence-transformers/all-MiniLM-L6-v2 (fast, 384-dim)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialized = False
        
    def _lazy_load(self):
        """Lazy load the model to avoid loading on import"""
        if self._initialized:
            return
            
        logger.info(f"Loading embedding model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Model loaded on GPU")
        else:
            logger.info("Model loaded on CPU")
            
        self._initialized = True
        
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Generate embeddings for input text(s)
        
        Args:
            texts: Single text string or list of texts
            max_length: Maximum sequence length
            batch_size: Batch size for processing multiple texts
            
        Returns:
            Tensor of embeddings with shape (num_texts, embedding_dim)
        """
        self._lazy_load()
        
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length
            )
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling for better quality (instead of just CLS token)
                # Mean pooling - take average of all token embeddings
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
            all_embeddings.append(batch_embeddings.cpu())
            
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)
        
        return embeddings
    
    def generate_embedding(self, text: str, max_length: int = 512) -> List[float]:
        """
        Generate embedding for a single text and return as list
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            List of floats representing the embedding
        """
        embedding = self.generate_embeddings(text, max_length=max_length)
        return embedding[0].tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        self._lazy_load()
        # Generate a dummy embedding to get dimension
        dummy_embedding = self.generate_embeddings("test")
        return dummy_embedding.shape[1]
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        embeddings = self.generate_embeddings([text1, text2])
        
        # Cosine similarity
        emb1 = embeddings[0]
        emb2 = embeddings[1]
        
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        )
        
        return similarity.item()
