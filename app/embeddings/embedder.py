"""Embedding providers.

The interface is deliberately tiny (``embed_documents`` / ``embed_query``) so
providers are interchangeable:

* ``OpenAIEmbedder``  -- default, works with any OpenAI-compatible endpoint.
* ``LocalEmbedder``   -- sentence-transformers, fully offline (optional dep).
* ``HashEmbedder``    -- deterministic, dependency-free; used for tests and as
                         an offline fallback so the pipeline is always runnable.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

from app.config import Config
from app.utils import log, with_retries

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BaseEmbedder:
    dim: int = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class HashEmbedder(BaseEmbedder):
    """Deterministic bag-of-hashed-tokens embedding, L2-normalised.

    Not competitive with real models, but it is fast, needs no API/GPU, and is
    perfectly reproducible -- ideal for unit tests and offline demos.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


class LocalEmbedder(BaseEmbedder):
    """sentence-transformers embedder (optional heavy dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` or use another provider."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI-compatible embeddings via the official SDK."""

    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: Config) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai is not installed. Run `pip install openai`."
            ) from exc
        config.require_embedding_key()
        self._client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            timeout=config.request_timeout,
        )
        self.model = config.embedding_model
        self.dim = self._DIMS.get(self.model, 1536)
        self._cfg = config

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        @with_retries(max_retries=self._cfg.max_retries, backoff=self._cfg.retry_backoff)
        def _call(batch: List[str]):
            return self._client.embeddings.create(model=self.model, input=batch)

        out: List[List[float]] = []
        # Batch to stay within request limits.
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            resp = _call(batch)
            out.extend([d.embedding for d in resp.data])
        return out


def get_embedder(config: Config) -> BaseEmbedder:
    provider = config.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbedder(config)
    if provider == "local":
        return LocalEmbedder()
    if provider == "hash":
        return HashEmbedder(dim=config.embedding_dim)
    log.warning("Unknown EMBEDDING_PROVIDER '%s', falling back to hash.", provider)
    return HashEmbedder(dim=config.embedding_dim)
