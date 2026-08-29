"""Semantic (embedding similarity) retrieval."""
from __future__ import annotations

from typing import List, Optional

from app.embeddings.embedder import BaseEmbedder
from app.retrieval.base import RetrievalResult
from app.storage.vector_store import BaseVectorStore


class VectorSearch:
    def __init__(self, store: BaseVectorStore, embedder: BaseEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 15,
        video_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        embedding = self.embedder.embed_query(query)
        records = self.store.query(embedding, n_results=top_k, video_ids=video_ids)
        results: List[RetrievalResult] = []
        for rec in records:
            distance = rec.get("distance")
            # Cosine distance -> similarity in [0, 1] (clamped).
            similarity = 1.0 - float(distance) if distance is not None else 0.0
            similarity = max(0.0, min(1.0, similarity))
            r = RetrievalResult.from_record(rec, score=similarity)
            r.scores["vector"] = similarity
            results.append(r)
        return results
