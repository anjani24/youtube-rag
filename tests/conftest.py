"""Shared pytest fixtures. Everything here runs fully offline."""
from __future__ import annotations

import os
import sys
import tempfile
from typing import List

import pytest

# Ensure the project root is importable when running `pytest` from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Config
from app.embeddings.embedder import HashEmbedder
from app.ingestion.ingest import ingest_video
from app.ingestion.transcript import Segment, TranscriptUnavailable
from app.llm.provider import FakeLLM
from app.rag.qa import RAGPipeline
from app.retrieval.reranker import IdentityReranker
from app.storage.vector_store import InMemoryVectorStore


class FakeTranscriptProvider:
    """Returns canned transcripts; can be told to fail for certain ids."""

    def __init__(self, transcripts: dict, unavailable: set | None = None) -> None:
        self._transcripts = transcripts
        self._unavailable = unavailable or set()

    def fetch(self, video_id: str) -> List[Segment]:
        if video_id in self._unavailable:
            raise TranscriptUnavailable(f"No captions for {video_id}")
        return self._transcripts[video_id]


def _segments(pairs) -> List[Segment]:
    return [Segment(text=t, start=float(s), duration=float(d)) for t, s, d in pairs]


RAG_TRANSCRIPT = _segments(
    [
        ("Welcome to this lecture on retrieval augmented generation.", 0.0, 4.0),
        ("RAG combines retrieval with generation to give an LLM extra context.", 4.0, 5.0),
        ("We first embed documents into vectors using an embedding model.", 12.0, 5.0),
        ("Those vectors are stored in a vector database for similarity search.", 22.0, 5.0),
        ("At query time we retrieve the most similar chunks by cosine similarity.", 35.0, 5.0),
        ("A reranker then reorders candidates to improve relevance before the LLM.", 47.0, 5.0),
        ("Chunking with overlap keeps related context together for retrieval.", 60.0, 5.0),
    ]
)

VECTORDB_TRANSCRIPT = _segments(
    [
        ("This video explains vector databases and indexing in detail.", 0.0, 4.0),
        ("A vector database stores embeddings and supports fast similarity search.", 10.0, 5.0),
        ("Indexing methods like HNSW make nearest neighbor search efficient.", 25.0, 5.0),
        ("We compare exact search versus approximate nearest neighbor indexes.", 40.0, 5.0),
    ]
)


@pytest.fixture
def offline_config() -> Config:
    return Config(
        llm_provider="fake",
        embedding_provider="hash",
        embedding_dim=128,
        reranker_provider="identity",
        vector_store="memory",
        top_k=5,
        vector_top_k=10,
        bm25_top_k=10,
        rerank_top_k=5,
        chunk_size=200,
        chunk_overlap=40,
        min_relevance_score=0.0,
        transcripts_dir=tempfile.mkdtemp(prefix="ytrag_test_"),
    )


@pytest.fixture
def fake_provider() -> FakeTranscriptProvider:
    return FakeTranscriptProvider(
        transcripts={
            "ragLECTURE1": RAG_TRANSCRIPT,
            "vectorDB002": VECTORDB_TRANSCRIPT,
        },
        unavailable={"noCaption01"},
    )


@pytest.fixture
def pipeline(offline_config) -> RAGPipeline:
    return RAGPipeline(
        offline_config,
        store=InMemoryVectorStore(offline_config),
        embedder=HashEmbedder(dim=offline_config.embedding_dim),
        llm=FakeLLM(),
        reranker=IdentityReranker(),
    )


@pytest.fixture
def seeded_pipeline(pipeline, offline_config, fake_provider) -> RAGPipeline:
    for url, vid in [
        ("https://www.youtube.com/watch?v=ragLECTURE1", "ragLECTURE1"),
        ("https://youtu.be/vectorDB002", "vectorDB002"),
    ]:
        res = ingest_video(
            url,
            offline_config,
            pipeline.store,
            pipeline.embedder,
            transcript_provider=fake_provider,
            title_fetcher=lambda v: f"Title for {v}",
        )
        assert res.ok, res.message
    return pipeline
