import pytest
from fastapi.testclient import TestClient
from main import app

from typing import List, Dict

client = TestClient(app)


class DummyGoogleImageSearchClient:
    def search_images(self, query: str, num_results: int = 10, **kwargs):
        return [
            {
                "title": f"Image {i}",
                "link": f"https://images.example/{i}.jpg",
                "thumbnail": f"https://images.example/{i}.thumb.jpg",
                "displayLink": "example.com",
            }
            for i in range(1, num_results + 1)
        ]


class DummyImageAnalyzerResult:
    def __init__(self, idx):
        self.image_url = f"https://images.example/{idx}.jpg"
        self.image_title = f"Image {idx}"
        self.image_source = "example.com"
        self.analysis = f"This is a description for image {idx}."
        self.embedding = [0.1, 0.2]  # simplified
        self.error = None


class DummyImageAnalyzerClient:
    def __init__(self, *args, **kwargs):
        self.load_vlm_called = False

    def describe_images(self, query: str, num_images: int = 5, **kwargs):
        return [DummyImageAnalyzerResult(i + 1) for i in range(num_images)]

    def answer_visual_question(self, search_query: str, question: str, num_images: int = 5):
        return [DummyImageAnalyzerResult(i + 1) for i in range(num_images)]

    def store_in_vectordb(self, results: List[DummyImageAnalyzerResult], query: str = None) -> Dict[str, any]:
        # Return fake IDs
        return {"stored": len(results), "ids": list(range(1000, 1000 + len(results))) }

    def search_vectordb(self, query: str, top_k: int = 5, filter_query: str = None):
        return [
            {"id": 2000 + i, "score": 0.01 * i, "analysis": f"analysis {i+1}", "image_url": f"https://images.example/{i+1}.jpg", "image_title": f"Image {i+1}", "image_source": "example.com", "search_query": query}
            for i in range(top_k)
        ]


@pytest.fixture(autouse=True)
def patch_clients(monkeypatch):
    # Patch both GoogleImageSearch and ImageAnalyzerClient constructors used in the route file
    from clients import google_image_search_client as gis
    from clients import image_analyzer_client as iac
    from clients import milvus_client as mc

    monkeypatch.setattr(gis, 'GoogleImageSearchClient', lambda: DummyGoogleImageSearchClient())
    monkeypatch.setattr(iac, 'ImageAnalyzerClient', lambda *args, **kwargs: DummyImageAnalyzerClient())
    # Default Milvus client to avoid external dependency
    class DummyDefaultMilvusClient:
        def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = "ai_firm_vectors"):
            self.collection_name = collection_name
        def connect(self):
            return
        def create_collection(self, embedding_dim: int):
            return
        def create_index(self, *args, **kwargs):
            return
        def insert(self, texts, embeddings, metadata):
            return []
        def search(self, query_embedding, top_k=5, metric_type="L2", search_params=None, output_fields=None):
            return []

    monkeypatch.setattr(mc, 'MilvusClient', lambda *args, **kwargs: DummyDefaultMilvusClient(*args, **kwargs))


def test_ask_endpoint_image_search_only():
    payload = {
        "question": "What is transformer attention?",
        "include_image_search": True,
        "enable_image_analysis": False
    }

    response = client.post('/intelligent-query/ask', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "image_search_results" in data
    assert isinstance(data["image_search_results"], list)
    assert len(data["image_search_results"]) > 0


def test_ask_endpoint_image_analysis_and_store():
    payload = {
        "question": "Show me images of transformers architecture",
        "include_image_search": True,
        "enable_image_analysis": True,
        "image_num_results": 3,
        "store_image_analysis": True
    }

    response = client.post('/intelligent-query/ask', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "image_analysis_results" in data
    assert isinstance(data["image_analysis_results"], list)
    assert len(data["image_analysis_results"]) == 3
    assert data.get("image_milvus_ids") is not None
    assert isinstance(data.get("image_milvus_ids"), list)


def test_image_analysis_included_in_retrieved_context():
    payload = {
        "question": "Tell me about transformer attention with images",
        "include_image_search": True,
        "enable_image_analysis": True,
        "image_num_results": 2,
    }

    response = client.post('/intelligent-query/ask', json=payload)
    assert response.status_code == 200
    data = response.json()
    # Retrieved context should include text and potentially image results
    assert "retrieved_context" in data
    assert isinstance(data["retrieved_context"], list)


def test_skip_image_search_when_image_collection_has_data(monkeypatch):
    # Patch MilvusClient to simulate existing image data and topics data
    from clients import milvus_client as mc

    class DummyMilvusClientForSkip:
        def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = "ai_firm_vectors"):
            self.collection_name = collection_name
        def connect(self):
            return
        def create_collection(self, embedding_dim: int):
            return
        def search(self, query_embedding, top_k=5, metric_type="L2", search_params=None, output_fields=None):
            # If topics collection, return a very similar topic to skip scraping
            if self.collection_name == "ai_firm_topics":
                return [{"id": 1, "score": 0.3, "text": "topic match", "metadata": {}}]
            # If image collection, return existing image analysis entries
            if self.collection_name == "image_analysis_retrieval":
                return [{"id": 10, "score": 0.2, "text": "existing image analysis", "metadata": {"image_url": "https://images.example/1.jpg", "image_title": "Image 1", "image_source": "example.com"}}]
            return []

    monkeypatch.setattr(mc, 'MilvusClient', lambda *args, **kwargs: DummyMilvusClientForSkip(*args, **kwargs))

    payload = {
        "question": "What is transformer attention?",
        "include_image_search": True,
        "enable_image_analysis": True
    }

    response = client.post('/intelligent-query/ask', json=payload)
    assert response.status_code == 200
    data = response.json()
    # Because we simulate existing topics and image data, the route should skip both scraping and image search
    assert data.get("image_search_results") == []
    assert isinstance(data.get("image_analysis_results"), list)
    assert len(data.get("image_analysis_results")) == 1


def test_preload_vlm_endpoint(monkeypatch):
    from clients import image_analyzer_client as iac

    class DummyAnalyzer:
        def __init__(self, *args, **kwargs):
            self.vlm = None
            self.initialized = False
        def _initialize_vlm(self):
            self.vlm = object()
            self.initialized = True

    monkeypatch.setattr(iac, 'ImageAnalyzerClient', lambda *args, **kwargs: DummyAnalyzer())

    payload = {"tensor_parallel_size": 1, "force_reload": True}
    response = client.post('/intelligent-query/preload-vlm', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") in ("loaded", "error")

