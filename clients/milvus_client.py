from pymilvus import (
    CollectionSchema,
    FieldSchema,
    DataType,
    MilvusClient as PyMilvusClient,
)
from typing import List, Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class MilvusClient:
    """Client for interacting with Milvus vector database"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "ai_firm_vectors",
        uri: Optional[str] = None,
        token: Optional[str] = None,
        secure: bool = True,
    ):
        """
        Initialize Milvus client
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection to use
            uri, token: Optionally provide a `uri` and `token` for Zilliz Cloud connections. If both are provided,
                        the client will prefer `PyMilvusClient(uri=..., token=...)` for secure cloud connection.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        self._connected = False
        self.uri = uri
        self.token = token
        # user/password removed per request; use uri/token for Zilliz Cloud
        self.secure = secure
        self._pymilvus_client = None
        
    def connect(self):
        """Connect to Milvus server"""
        if self._connected:
            return
        # Prefer URI+token for Zilliz Cloud (PyMilvusClient) exclusively
        uri = self.uri or os.getenv("MILVUS_URI")
        token = self.token or os.getenv("MILVUS_TOKEN")
        if not uri or not token:
            raise ValueError("uri and token are required for PyMilvusClient (Zilliz Cloud) connection")

        try:
            self._pymilvus_client = PyMilvusClient(uri=uri, token=token)
            self._connected = True
            logger.info("Connected to Milvus via PyMilvusClient (uri/token)")
        except Exception as e:
            logger.error("Failed to create PyMilvusClient: %s", e)
            raise

    @classmethod
    def from_env(cls, collection_name: str = "ai_firm_vectors") -> "MilvusClient":
        """
        Create a MilvusClient from environment variables (Zilliz Cloud)

        Required environment variables (Zilliz Cloud):
          - MILVUS_URI
          - MILVUS_TOKEN

        Returns:
            MilvusClient configured to connect via URI + token.
        """
        uri = os.getenv("MILVUS_URI")
        token = os.getenv("MILVUS_TOKEN")
        if not uri or not token:
            raise ValueError("Environment variables MILVUS_URI and MILVUS_TOKEN are required for Zilliz Cloud client")
        return cls(collection_name=collection_name, uri=uri, token=token)
            
    def disconnect(self):
        """Disconnect from Milvus server"""
        if self._connected:
            try:
                if self._pymilvus_client:
                    self._pymilvus_client.close()
            except Exception:
                # ignore
                pass
            self._pymilvus_client = None
            self._connected = False
            logger.info("Disconnected from PyMilvusClient")
            
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
        if self._pymilvus_client.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists")
            return
            
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=auto_id),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        
        schema = self._pymilvus_client.create_schema()
        # Add fields
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=auto_id)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=embedding_dim)
        schema.add_field("metadata", DataType.JSON)
        # Create collection via PyMilvusClient
        self._pymilvus_client.create_collection(self.collection_name, schema=schema)
        
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
        # Validate collection existence via pymilvus
        self.connect()
        if not self._pymilvus_client.has_collection(self.collection_name):
            raise ValueError("Collection not initialized. Call create_collection first.")
            
        if params is None:
            params = {"nlist": 128}
            
        index_params = self._pymilvus_client.prepare_index_params("embedding", index_type=index_type, metric_type=metric_type, params=params or {})
        self._pymilvus_client.create_index(self.collection_name, index_params)
        
        logger.info(f"Created index: {index_type} with metric {metric_type}")
        
    def insert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
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
        self.connect()
        if not self._pymilvus_client.has_collection(self.collection_name):
            raise ValueError("Collection not initialized. Call create_collection first.")
            
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")
            
        # Prepare metadata
        if metadata is None:
            metadata = [{} for _ in texts]
        elif len(metadata) != len(texts):
            raise ValueError("Number of metadata entries must match number of texts")
            
        # Prepare data as a list of dicts for PyMilvusClient.insert
        data = []
        for i, text in enumerate(texts):
            doc = {
                "text": text,
                "embedding": embeddings[i],
                "metadata": metadata[i],
            }
            if ids is not None:
                doc["id"] = ids[i]
            data.append(doc)
        if ids is not None:
            data.insert(0, ids)
        
        # Insert
        res = self._pymilvus_client.insert(self.collection_name, data)
        logger.info(f"Inserted {len(texts)} vectors into collection")
        return res.get("ids", [])
        
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
        self.connect()
        if not self._pymilvus_client.has_collection(self.collection_name):
            raise ValueError("Collection not initialized. Call create_collection first.")
            
        # Load collection to memory
        self._pymilvus_client.load_collection(self.collection_name)
        
        if search_params is None:
            search_params = {"nprobe": 10}
            
        if output_fields is None:
            output_fields = ["text", "metadata"]
            
        # Search
        results = self._pymilvus_client.search(
            self.collection_name,
            data=[query_embedding],
            search_params=search_params,
            anns_field="embedding",
            output_fields=output_fields,
            limit=top_k,
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "score": hit.get("distance") if "distance" in hit else hit.get("score"),
                    "text": hit.get("text"),
                    "metadata": hit.get("metadata", {}),
                })
                
        return formatted_results

    def search_vectors(
        self,
        vectors: List[List[float]],
        top_k: int = 5,
        metric_type: str = "L2",
        search_params: Optional[Dict[str, Any]] = None,
        anns_field: str = "embedding",
    ) -> List[List[Dict[str, Any]]]:
        """
        Search multiple query vectors and return a list of results list.
        """
        self.connect()
        if not self._pymilvus_client.has_collection(self.collection_name):
            raise ValueError("Collection not initialized. Call create_collection first.")
        self._pymilvus_client.load_collection(self.collection_name)
        if search_params is None:
            search_params = {"nprobe": 10}

        results = self._pymilvus_client.search(
            self.collection_name,
            data=vectors,
            search_params=search_params,
            anns_field=anns_field,
            limit=top_k,
            output_fields=["text", "metadata"],
        )

        formatted_batch = []
        for hits in results:
            formatted_hits = []
            for hit in hits:
                formatted_hits.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.entity.get("text") if hasattr(hit, 'entity') else None,
                    "metadata": hit.entity.get("metadata", {}) if hasattr(hit, 'entity') else {},
                })
            formatted_batch.append(formatted_hits)
        return formatted_batch

    # ---- Book collection helpers based on provided example ----
    # ---- Removed book-specific helpers per user request ----
        
    def delete_collection(self):
        """Delete the collection"""
        self.connect()
        if self._pymilvus_client.has_collection(self.collection_name):
            self._pymilvus_client.drop_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            self.collection = None
        else:
            logger.warning(f"Collection '{self.collection_name}' does not exist")
            
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        self.connect()
        if not self._pymilvus_client.has_collection(self.collection_name):
            raise ValueError("Collection not initialized")

        stats = self._pymilvus_client.get_collection_stats(self.collection_name)
        num_entities = stats.get("row_count") or stats.get("row_count", stats.get("num_entities") or 0)
        # Describe collection for description field
        desc = self._pymilvus_client.describe_collection(self.collection_name)
        description = desc.get("description") if isinstance(desc, dict) else None
        return {
            "name": self.collection_name,
            "num_entities": int(num_entities) if num_entities else 0,
            "description": description,
        }

    # ---- Convenience wrappers for some lower-level PyMilvusClient APIs ----
    def flush(self, timeout: Optional[float] = None):
        self.connect()
        self._pymilvus_client.flush(self.collection_name, timeout=timeout)

    def describe_collection(self, timeout: Optional[float] = None):
        self.connect()
        return self._pymilvus_client.describe_collection(self.collection_name, timeout=timeout)

    def has_collection(self):
        self.connect()
        return self._pymilvus_client.has_collection(self.collection_name)

    def list_collections(self):
        self.connect()
        return self._pymilvus_client.list_collections()
