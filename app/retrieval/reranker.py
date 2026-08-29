"""Reranking of a candidate set down to the final top-k.

Retrieval casts a wide net (top-15 from each retriever). A reranker then scores
each *(query, chunk)* pair jointly -- far more accurate than the bag-of-words or
single-vector signals used during first-stage retrieval -- and keeps only the
best ``rerank_top_k``. Providers are modular:

* ``IdentityReranker``      -- keeps fusion order (no extra deps; sane default).
* ``CrossEncoderReranker``  -- sentence-transformers cross-encoder (local).
* ``CohereReranker``        -- Cohere Rerank API.
"""
from __future__ import annotations

from typing import List

from app.config import Config
from app.retrieval.base import RetrievalResult
from app.utils import log, with_retries


class BaseReranker:
    def rerank(
        self, query: str, results: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        raise NotImplementedError


class IdentityReranker(BaseReranker):
    """Trusts the upstream (RRF) ordering; simply truncates to top_k."""

    def rerank(self, query, results, top_k) -> List[RetrievalResult]:
        for rank, r in enumerate(results):
            r.scores["rerank"] = 1.0 / (rank + 1)
        return results[:top_k]


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` or use RERANKER_PROVIDER=identity."
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query, results, top_k) -> List[RetrievalResult]:
        if not results:
            return []
        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)
        for r, s in zip(results, scores):
            r.score = float(s)
            r.scores["rerank"] = float(s)
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]


class CohereReranker(BaseReranker):
    def __init__(self, config: Config) -> None:
        try:
            import cohere  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "cohere is not installed. Run `pip install cohere` "
                "or use RERANKER_PROVIDER=identity."
            ) from exc
        if not config.cohere_api_key:
            raise RuntimeError("COHERE_API_KEY is required for the Cohere reranker.")
        self._client = cohere.Client(config.cohere_api_key)
        self._model = "rerank-english-v3.0"
        self._cfg = config

    def rerank(self, query, results, top_k) -> List[RetrievalResult]:
        if not results:
            return []

        @with_retries(max_retries=self._cfg.max_retries, backoff=self._cfg.retry_backoff)
        def _call():
            return self._client.rerank(
                query=query,
                documents=[r.text for r in results],
                top_n=top_k,
                model=self._model,
            )

        resp = _call()
        reranked: List[RetrievalResult] = []
        for item in resp.results:
            r = results[item.index]
            r.score = float(item.relevance_score)
            r.scores["rerank"] = float(item.relevance_score)
            reranked.append(r)
        return reranked


def get_reranker(config: Config) -> BaseReranker:
    provider = config.reranker_provider.lower()
    if provider == "cross_encoder":
        return CrossEncoderReranker(config.reranker_model)
    if provider == "cohere":
        return CohereReranker(config)
    if provider != "identity":
        log.warning("Unknown RERANKER_PROVIDER '%s', using identity.", provider)
    return IdentityReranker()
