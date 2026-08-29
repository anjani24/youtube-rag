"""BM25 keyword retrieval.

BM25 complements vector search: it excels when the query and the source share
exact terminology (API names, acronyms, code identifiers) that embedding models
sometimes smooth over. The corpus is (re)built from the vector store so the two
retrievers always see the same chunks.
"""
from __future__ import annotations

import re
from typing import List, Optional

from app.retrieval.base import RetrievalResult
from app.storage.vector_store import BaseVectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Search:
    def __init__(self, store: BaseVectorStore) -> None:
        self.store = store
        self._bm25 = None
        self._records: List[dict] = []
        self._signature: Optional[int] = None

    def _build(self, video_ids: Optional[List[str]]) -> None:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "rank-bm25 is not installed. Run `pip install rank-bm25`."
            ) from exc
        self._records = self.store.get_chunks(video_ids=video_ids)
        corpus = [_tokenize(r["text"]) for r in self._records]
        # BM25Okapi requires a non-empty corpus.
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(
        self,
        query: str,
        top_k: int = 15,
        video_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        # Rebuild if the requested scope changed or store size changed.
        signature = hash((tuple(sorted(video_ids)) if video_ids else None, self.store.count()))
        if self._bm25 is None or signature != self._signature:
            self._build(video_ids)
            self._signature = signature
        if not self._bm25 or not self._records:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        results: List[RetrievalResult] = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            rec = self._records[i]
            r = RetrievalResult.from_record(rec, score=float(scores[i]))
            r.scores["bm25"] = float(scores[i])
            results.append(r)
        return results
