from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MilvusClient:
    """Client for interacting with Milvus vector database"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "ai_firm_vectors"
    ):
        """
        Initialize Milvus client
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection to use
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self._connected = False
        
    def connect(self):
        """Connect to Milvus server"""
        if self._connected:
            return
            
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
            
    def disconnect(self):
        """Disconnect from Milvus server"""
        if self._connected:
            connections.disconnect(alias="default")
            self._connected = False
            logger.info("Disconnected from Milvus")
            
    def create_collection(
        self,
        embedding_dim: int,
        description: str = "AI Firm vector collection",
        auto_id: bool = True
    ):
        """
        Create a new collection with schema
        
        Args:
            embedding_dim: Dimension of embedding vectors
            description: Collection description
            auto_id: Whether to auto-generate IDs
        """
        self.connect()
        
        # Check if collection exists
        if utility.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists")
            self.collection = Collection(self.collection_name)
            return
            
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=auto_id),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=description
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        logger.info(f"Created collection '{self.collection_name}' with dimension {embedding_dim}")
        
    def create_index(
        self,
        index_type: str = "IVF_FLAT",
        metric_type: str = "L2",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Create index on the embedding field
        
        Args:
            index_type: Type of index (IVF_FLAT, HNSW, etc.)
            metric_type: Distance metric (L2, IP, COSINE)
            params: Index parameters
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection first.")
            
        if params is None:
            params = {"nlist": 128}
            
        index_params = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": params
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info(f"Created index: {index_type} with metric {metric_type}")
        
    def insert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Insert data into collection
        
        Args:
            texts: List of text strings
            embeddings: List of embedding vectors
            metadata: Optional metadata for each text
            
        Returns:
            List of inserted IDs
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection first.")
            
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
            
        # Prepare metadata
        if metadata is None:
            metadata = [{} for _ in texts]
        elif len(metadata) != len(texts):
            raise ValueError("Number of metadata entries must match number of texts")
            
        # Prepare data
        data = [
            texts,
            embeddings,
            metadata
        ]
        
        # Insert
        mr = self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"Inserted {len(texts)} vectors into collection")
        
        return mr.primary_keys
        
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        metric_type: str = "L2",
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            metric_type: Distance metric
            search_params: Search parameters
            output_fields: Fields to return in results
            
        Returns:
            List of search results with scores and data
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection first.")
            
        # Load collection to memory
        self.collection.load()
        
        if search_params is None:
            search_params = {"nprobe": 10}
            
        if output_fields is None:
            output_fields = ["text", "metadata"]
            
        # Search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata", {})
                })
                
        return formatted_results
        
    def delete_collection(self):
        """Delete the collection"""
        self.connect()
        
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            self.collection = None
        else:
            logger.warning(f"Collection '{self.collection_name}' does not exist")
            
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.collection:
            raise ValueError("Collection not initialized")
            
        stats = self.collection.num_entities
        
        return {
            "name": self.collection_name,
            "num_entities": stats,
            "description": self.collection.description
        }
