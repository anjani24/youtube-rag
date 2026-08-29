"""Central configuration for the YouTube Knowledge Intelligence System.

All tunables are read from environment variables (loaded from a local ``.env``
file via python-dotenv). Nothing here is hard-coded to a secret value.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


@dataclass
class Config:
    """Runtime configuration. Instantiate with :meth:`from_env`."""

    # ---- LLM provider (OpenAI-compatible by default) ---------------------
    llm_provider: str = "openai"                 # openai | fake
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None        # override for OpenAI-compatible hosts
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # ---- Embeddings ------------------------------------------------------
    embedding_provider: str = "openai"           # openai | local | hash
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 256                      # only used by the hash embedder

    # ---- Reranking -------------------------------------------------------
    reranker_provider: str = "identity"          # identity | cross_encoder | cohere
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cohere_api_key: Optional[str] = None

    # ---- Vector store ----------------------------------------------------
    vector_store: str = "chroma"                 # chroma | memory
    chroma_dir: str = "chroma_db"
    collection_name: str = "youtube_knowledge"

    # ---- Retrieval knobs -------------------------------------------------
    top_k: int = 5                               # final results returned to the LLM
    vector_top_k: int = 15                        # candidates from vector search
    bm25_top_k: int = 15                          # candidates from BM25
    rerank_top_k: int = 5                          # kept after reranking
    rrf_k: int = 60                               # reciprocal-rank-fusion constant
    min_relevance_score: float = 0.0             # below this -> "insufficient evidence"

    # ---- Chunking --------------------------------------------------------
    chunk_size: int = 800                        # target characters per chunk
    chunk_overlap: int = 150                      # character overlap between chunks

    # ---- Reliability -----------------------------------------------------
    request_timeout: int = 60
    max_retries: int = 3
    retry_backoff: float = 1.5

    # ---- Paths -----------------------------------------------------------
    data_dir: str = "data"
    transcripts_dir: str = field(default="data/transcripts")
    evaluation_dir: str = field(default="data/evaluation")

    # ---- Logging ---------------------------------------------------------
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            llm_provider=_get_str("LLM_PROVIDER", "openai"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            llm_model=_get_str("LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=_get_float("LLM_TEMPERATURE", 0.1),
            llm_max_tokens=_get_int("LLM_MAX_TOKENS", 1024),
            embedding_provider=_get_str("EMBEDDING_PROVIDER", "openai"),
            embedding_model=_get_str("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim=_get_int("EMBEDDING_DIM", 256),
            reranker_provider=_get_str("RERANKER_PROVIDER", "identity"),
            reranker_model=_get_str(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            vector_store=_get_str("VECTOR_STORE", "chroma"),
            chroma_dir=_get_str("CHROMA_DIR", "chroma_db"),
            collection_name=_get_str("COLLECTION_NAME", "youtube_knowledge"),
            top_k=_get_int("TOP_K", 5),
            vector_top_k=_get_int("VECTOR_TOP_K", 15),
            bm25_top_k=_get_int("BM25_TOP_K", 15),
            rerank_top_k=_get_int("RERANK_TOP_K", 5),
            rrf_k=_get_int("RRF_K", 60),
            min_relevance_score=_get_float("MIN_RELEVANCE_SCORE", 0.0),
            chunk_size=_get_int("CHUNK_SIZE", 800),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
            request_timeout=_get_int("REQUEST_TIMEOUT", 60),
            max_retries=_get_int("MAX_RETRIES", 3),
            retry_backoff=_get_float("RETRY_BACKOFF", 1.5),
            data_dir=_get_str("DATA_DIR", "data"),
            transcripts_dir=_get_str("TRANSCRIPTS_DIR", "data/transcripts"),
            evaluation_dir=_get_str("EVALUATION_DIR", "data/evaluation"),
            log_level=_get_str("LOG_LEVEL", "INFO"),
        )

    def require_llm_key(self) -> None:
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file or switch "
                "LLM_PROVIDER=fake for offline experimentation."
            )

    def require_embedding_key(self) -> None:
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file or switch "
                "EMBEDDING_PROVIDER=hash/local for offline experimentation."
            )
