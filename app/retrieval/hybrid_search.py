"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Why RRF (documented decision)
-----------------------------
Vector similarity (cosine, ~[0,1]) and BM25 (unbounded, corpus-dependent) live
on completely different scales, so naive score addition lets one retriever
dominate. RRF fuses by *rank* instead of raw score::

    rrf_score(d) = sum_over_retrievers ( 1 / (k + rank_r(d)) )

This is scale-free, needs no per-query normalisation, is robust to outliers,
and empirically strong (Cormack et al., 2009). ``k`` (default 60) damps the
influence of very high ranks. Candidates found by *both* retrievers naturally
accumulate score, which is exactly the behaviour we want from a hybrid system.

We also keep the per-retriever scores on each result for debugging/evaluation,
and merge results by chunk id (deduplication).
"""
from __future__ import annotations

from typing import List, Optional

from app.config import Config
from app.retrieval.base import RetrievalResult
from app.retrieval.bm25_search import BM25Search
from app.retrieval.vector_search import VectorSearch


class HybridSearch:
    def __init__(
        self,
        vector: VectorSearch,
        bm25: BM25Search,
        config: Config,
    ) -> None:
        self.vector = vector
        self.bm25 = bm25
        self.config = config

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        video_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        top_k = top_k or self.config.top_k
        k = self.config.rrf_k

        vec_hits = self.vector.search(
            query, top_k=self.config.vector_top_k, video_ids=video_ids
        )
        bm_hits = self.bm25.search(
            query, top_k=self.config.bm25_top_k, video_ids=video_ids
        )

        fused: dict[str, RetrievalResult] = {}

        def fuse(hits: List[RetrievalResult], source: str) -> None:
            for rank, hit in enumerate(hits):
                contribution = 1.0 / (k + rank + 1)
                if hit.id not in fused:
                    fused[hit.id] = RetrievalResult(
                        id=hit.id,
                        text=hit.text,
                        metadata=hit.metadata,
                        score=0.0,
                        scores={},
                    )
                target = fused[hit.id]
                target.score += contribution
                target.scores[f"{source}_rank"] = rank + 1
                target.scores.update(hit.scores)
                target.scores["rrf"] = target.score

        fuse(vec_hits, "vector")
        fuse(bm_hits, "bm25")

        merged = sorted(fused.values(), key=lambda r: r.score, reverse=True)
        return merged[:top_k]
