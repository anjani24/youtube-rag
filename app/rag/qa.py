"""Question answering pipeline using retrieval + optional reranking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.config import Config
from app.embeddings.embedder import BaseEmbedder, get_embedder
from app.llm.provider import BaseLLM, get_llm
from app.rag.citations import Citation, build_citations, format_answer_with_sources
from app.rag.prompts import (
    INSUFFICIENT_EVIDENCE,
    build_comparison_messages,
    build_qa_messages,
)
from app.retrieval.base import RetrievalResult
from app.retrieval.bm25_search import BM25Search
from app.retrieval.hybrid_search import HybridSearch
from app.retrieval.reranker import BaseReranker, get_reranker
from app.retrieval.vector_search import VectorSearch
from app.storage.vector_store import BaseVectorStore, get_vector_store
from app.utils import log


@dataclass
class Answer:
    text: str
    citations: List[Citation] = field(default_factory=list)
    sources: List[RetrievalResult] = field(default_factory=list)
    sufficient: bool = True

    def render(self) -> str:
        return format_answer_with_sources(self.text, self.citations)


class RAGPipeline:
    def __init__(
        self,
        config: Config,
        store: Optional[BaseVectorStore] = None,
        embedder: Optional[BaseEmbedder] = None,
        llm: Optional[BaseLLM] = None,
        reranker: Optional[BaseReranker] = None,
    ) -> None:
        self.config = config
        self.store = store or get_vector_store(config)
        self.embedder = embedder or get_embedder(config)
        self.llm = llm or get_llm(config)
        self.reranker = reranker or get_reranker(config)
        self.vector = VectorSearch(self.store, self.embedder)
        self.bm25 = BM25Search(self.store)
        self.hybrid = HybridSearch(self.vector, self.bm25, config)

    # ---- retrieval ------------------------------------------------------- #
    def retrieve(
        self,
        query: str,
        video_ids: Optional[List[str]] = None,
        strategy: str = "hybrid_rerank",
    ) -> List[RetrievalResult]:
        """Retrieve using a named strategy. Used by QA and by evaluation."""
        cfg = self.config
        if strategy == "vector":
            return self.vector.search(query, top_k=cfg.top_k, video_ids=video_ids)
        if strategy == "bm25":
            return self.bm25.search(query, top_k=cfg.top_k, video_ids=video_ids)
        if strategy == "hybrid":
            return self.hybrid.search(
                query, top_k=cfg.rerank_top_k, video_ids=video_ids
            )
        # hybrid + rerank (default)
        candidates = self.hybrid.search(
            query, top_k=max(cfg.vector_top_k, cfg.bm25_top_k), video_ids=video_ids
        )
        return self.reranker.rerank(query, candidates, top_k=cfg.rerank_top_k)

    # ---- QA -------------------------------------------------------------- #
    def answer_question(
        self,
        query: str,
        video_ids: Optional[List[str]] = None,
        history: Optional[List[dict]] = None,
    ) -> Answer:
        if not query or not query.strip():
            return Answer(text="Please enter a question.", sufficient=False)

        results = self.retrieve(query, video_ids=video_ids, strategy="hybrid_rerank")

        # Empty-retrieval / low-confidence handling.
        if not results:
            return Answer(text=INSUFFICIENT_EVIDENCE, sufficient=False)
        top_score = max(r.score for r in results)
        if top_score < self.config.min_relevance_score:
            return Answer(
                text=INSUFFICIENT_EVIDENCE, sources=results, sufficient=False
            )

        messages = build_qa_messages(query, results, history=history)
        try:
            text = self.llm.chat(messages)
        except Exception as exc:  # noqa: BLE001
            log.error("LLM call failed: %s", exc)
            return Answer(
                text="The model call failed. Please check the API setup and try again.",
                sources=results,
                sufficient=False,
            )

        citations = build_citations(results)
        sufficient = INSUFFICIENT_EVIDENCE not in text
        return Answer(
            text=text,
            citations=citations if sufficient else [],
            sources=results,
            sufficient=sufficient,
        )

    # ---- Multi-video comparison ----------------------------------------- #
    def compare_videos(self, query: str, video_ids: List[str]) -> Answer:
        if len(video_ids) < 2:
            return Answer(
                text="Please select at least two videos to compare.",
                sufficient=False,
            )
        # Retrieve per video so every video is represented.
        grouped: dict[str, List[RetrievalResult]] = {}
        all_results: List[RetrievalResult] = []
        for vid in video_ids:
            res = self.retrieve(query, video_ids=[vid], strategy="hybrid_rerank")
            if res:
                title = res[0].title
                grouped[title] = res
                all_results.extend(res)

        if not grouped:
            return Answer(text=INSUFFICIENT_EVIDENCE, sufficient=False)

        messages = build_comparison_messages(query, grouped)
        try:
            text = self.llm.chat(messages)
        except Exception as exc:  # noqa: BLE001
            log.error("LLM comparison failed: %s", exc)
            return Answer(text="The model call failed.", sufficient=False)

        citations = build_citations(all_results)
        return Answer(text=text, citations=citations, sources=all_results)


def build_pipeline(config: Config) -> RAGPipeline:
    return RAGPipeline(config)
