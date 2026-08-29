import pytest

from app.chunking.chunker import Chunk
from app.embeddings.embedder import HashEmbedder
from app.storage.vector_store import InMemoryVectorStore


def _chunks():
    return [
        Chunk(text="alpha vector database", video_id="dQw4w9WgXcQ", title="A", start_time=0.0, end_time=5.0, chunk_index=0),
        Chunk(text="beta retrieval augmented", video_id="dQw4w9WgXcQ", title="A", start_time=5.0, end_time=10.0, chunk_index=1),
        Chunk(text="gamma reranking model", video_id="abcdefghijk", title="B", start_time=0.0, end_time=5.0, chunk_index=0),
    ]


def _add(store, embedder):
    chunks = _chunks()
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))


def test_memory_store_roundtrip():
    store = InMemoryVectorStore()
    embedder = HashEmbedder(dim=64)
    _add(store, embedder)
    assert store.count() == 3
    assert store.has_video("dQw4w9WgXcQ")
    videos = store.list_videos()
    assert len(videos) == 2
    counts = {v["video_id"]: v["chunk_count"] for v in videos}
    assert counts["dQw4w9WgXcQ"] == 2


def test_memory_store_delete():
    store = InMemoryVectorStore()
    embedder = HashEmbedder(dim=64)
    _add(store, embedder)
    store.delete_video("dQw4w9WgXcQ")
    assert not store.has_video("dQw4w9WgXcQ")
    assert store.count() == 1


def test_chroma_persistence_survives_reopen(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from app.config import Config
    from app.storage.vector_store import ChromaVectorStore

    cfg = Config(
        vector_store="chroma",
        chroma_dir=str(tmp_path / "db"),
        collection_name="test_collection",
    )
    embedder = HashEmbedder(dim=64)

    store = ChromaVectorStore(cfg)
    _add(store, embedder)
    assert store.count() == 3

    # Re-open a brand new client at the same path: data must persist.
    store2 = ChromaVectorStore(cfg)
    assert store2.count() == 3
    assert store2.has_video("dQw4w9WgXcQ")
    results = store2.query(embedder.embed_query("vector database"), n_results=2)
    assert results
    assert "distance" in results[0]
