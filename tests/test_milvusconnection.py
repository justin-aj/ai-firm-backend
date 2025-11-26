import os
import random
import logging
import pytest

from clients.milvus_client import MilvusClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("milvus_test")


def _get_milvus_creds():
    uri = os.getenv("MILVUS_URI")
    token = os.getenv("MILVUS_TOKEN")
    return uri, token


@pytest.mark.integration
def test_milvus_end_to_end_from_env():
    uri, token = _get_milvus_creds()
    if not (uri and token):
        pytest.skip("MILVUS_URI and MILVUS_TOKEN not set; skipping integration test")

    collection = "book"
    dim = 64
    nb = 20
    insert_rounds = 1

    client = MilvusClient.from_env(collection_name=collection)
    client.connect()

    client.delete_collection()
    # create simple collection with vector dimension
    client.create_collection(embedding_dim=dim, description="book collection", auto_id=True)

    texts = []
    embeddings = []
    metadata = []
    for i in range(nb):
        texts.append(f"book {i}")
        vec = [random.random() for _ in range(dim)]
        embeddings.append(vec)
        metadata.append({"word_count": random.randint(1, 100)})

    ids = client.insert(texts, embeddings, metadata)
    assert len(ids) == len(texts)

    client.flush()
    # Search
    nq = 2
    limit = 2
    search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
    results = client.search_vectors(vectors=search_vectors, top_k=limit, anns_field="embedding")
    assert len(results) == nq
    assert len(results[0]) == limit

    client.delete_collection()
