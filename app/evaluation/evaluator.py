"""Evaluation: real, computed metrics (never fabricated).

Retrieval metrics are computed against relevance labels derived from each
question's ``expected_video`` and ``expected_keywords``. A retrieved chunk is
judged relevant iff it belongs to an expected video (when specified) AND
contains at least one expected keyword (when specified).

We compute, per strategy (vector / bm25 / hybrid / hybrid+rerank):
  * Recall@K      = relevant retrieved in top-K / total relevant in the corpus
  * Precision@K   = relevant retrieved in top-K / K
  * HitRate@K     = fraction of questions with >= 1 relevant chunk in top-K
  * MRR           = mean reciprocal rank of the first relevant chunk

Generation metrics (skipped for the fake LLM):
  * answer_relevance   = expected-keyword coverage in the answer
  * groundedness       = share of answer content tokens found in the context
  * citation_accuracy  = share of cited timestamps that match retrieved chunks

Performance: retrieval / rerank / LLM / total latency (median ms).
"""
from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import Config
from app.evaluation.dataset import EvalQuestion
from app.rag.qa import RAGPipeline
from app.retrieval.base import RetrievalResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "these",
    "those", "as", "at", "by", "from", "we", "you", "they", "i", "he", "she",
    "our", "your", "their", "its", "can", "will", "would", "should", "could",
    "do", "does", "did", "not", "no", "so", "if", "then", "than", "how", "what",
    "why", "which", "who", "when", "where", "use", "used", "using",
}

STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]
STRATEGY_LABELS = {
    "vector": "Vector Search",
    "bm25": "BM25",
    "hybrid": "Hybrid",
    "hybrid_rerank": "Hybrid + Reranking",
}


def _content_tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3 and t not in _STOP}


def _is_relevant(result: RetrievalResult, q: EvalQuestion) -> bool:
    vid_ok = (not q.expected_video) or (result.video_id in q.expected_video)
    text_low = result.text.lower()
    kw_ok = (not q.expected_keywords) or any(
        kw in text_low for kw in q.expected_keywords
    )
    return vid_ok and kw_ok


def _chunk_relevant(meta_text: str, video_id: str, q: EvalQuestion) -> bool:
    vid_ok = (not q.expected_video) or (video_id in q.expected_video)
    kw_ok = (not q.expected_keywords) or any(
        kw in meta_text.lower() for kw in q.expected_keywords
    )
    return vid_ok and kw_ok


@dataclass
class RetrievalMetrics:
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    hit_rate_at_k: float = 0.0
    mrr: float = 0.0
    median_latency_ms: float = 0.0
    n: int = 0


@dataclass
class GenerationMetrics:
    answer_relevance: float = 0.0
    groundedness: float = 0.0
    citation_accuracy: float = 0.0
    median_llm_latency_ms: float = 0.0
    n: int = 0


@dataclass
class EvalReport:
    k: int
    retrieval: Dict[str, RetrievalMetrics] = field(default_factory=dict)
    generation: Optional[GenerationMetrics] = None


class Evaluator:
    def __init__(self, pipeline: RAGPipeline, config: Config) -> None:
        self.pipeline = pipeline
        self.config = config
        # Precompute corpus for total-relevant counts.
        self._corpus = pipeline.store.get_chunks()

    def _total_relevant(self, q: EvalQuestion) -> int:
        return sum(
            1
            for c in self._corpus
            if _chunk_relevant(c["text"], c["metadata"].get("video_id", ""), q)
        )

    def evaluate_retrieval(self, questions: List[EvalQuestion], k: int) -> Dict[str, RetrievalMetrics]:
        per_strategy: Dict[str, RetrievalMetrics] = {}
        for strategy in STRATEGIES:
            recalls, precisions, hits, rrs, lats = [], [], [], [], []
            for q in questions:
                total_rel = self._total_relevant(q)
                t0 = time.perf_counter()
                results = self.pipeline.retrieve(
                    q.question, strategy=strategy
                )[:k]
                lats.append((time.perf_counter() - t0) * 1000)

                rel_flags = [_is_relevant(r, q) for r in results]
                n_rel = sum(rel_flags)
                precisions.append(n_rel / k if k else 0.0)
                recalls.append(n_rel / total_rel if total_rel else 0.0)
                hits.append(1.0 if n_rel > 0 else 0.0)
                rr = 0.0
                for rank, flag in enumerate(rel_flags, start=1):
                    if flag:
                        rr = 1.0 / rank
                        break
                rrs.append(rr)

            per_strategy[strategy] = RetrievalMetrics(
                recall_at_k=_mean(recalls),
                precision_at_k=_mean(precisions),
                hit_rate_at_k=_mean(hits),
                mrr=_mean(rrs),
                median_latency_ms=_median(lats),
                n=len(questions),
            )
        return per_strategy

    def evaluate_generation(self, questions: List[EvalQuestion]) -> GenerationMetrics:
        rels, grounds, cites, lats = [], [], [], []
        for q in questions:
            results = self.pipeline.retrieve(q.question, strategy="hybrid_rerank")
            context_tokens = set()
            valid_ts = set()
            for r in results:
                context_tokens |= _content_tokens(r.text)
                valid_ts.add(int(r.start_time))
            t0 = time.perf_counter()
            ans = self.pipeline.answer_question(q.question)
            lats.append((time.perf_counter() - t0) * 1000)

            text_low = ans.text.lower()
            # answer relevance: expected keyword coverage
            if q.expected_keywords:
                covered = sum(1 for kw in q.expected_keywords if kw in text_low)
                rels.append(covered / len(q.expected_keywords))
            # groundedness: content tokens of answer present in context
            atoks = _content_tokens(ans.text)
            if atoks:
                grounds.append(len(atoks & context_tokens) / len(atoks))
            # citation accuracy: cited timestamps that exist among retrieved chunks
            if ans.citations:
                good = sum(1 for c in ans.citations if int(c.start_time) in valid_ts)
                cites.append(good / len(ans.citations))

        return GenerationMetrics(
            answer_relevance=_mean(rels),
            groundedness=_mean(grounds),
            citation_accuracy=_mean(cites),
            median_llm_latency_ms=_median(lats),
            n=len(questions),
        )

    def run(
        self,
        questions: List[EvalQuestion],
        k: Optional[int] = None,
        with_generation: bool = True,
    ) -> EvalReport:
        k = k or self.config.top_k
        report = EvalReport(k=k)
        report.retrieval = self.evaluate_retrieval(questions, k)
        if with_generation and self.config.llm_provider != "fake":
            report.generation = self.evaluate_generation(questions)
        return report


def _mean(xs: List[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _median(xs: List[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def format_report(report: EvalReport) -> str:
    lines = []
    lines.append("=" * 44)
    lines.append(" Retrieval Evaluation")
    lines.append("=" * 44)
    lines.append("")
    for strategy in STRATEGIES:
        m = report.retrieval.get(strategy)
        if not m:
            continue
        lines.append(STRATEGY_LABELS[strategy])
        lines.append(f"  Recall@{report.k}:    {m.recall_at_k * 100:5.1f}%")
        lines.append(f"  Precision@{report.k}: {m.precision_at_k * 100:5.1f}%")
        lines.append(f"  HitRate@{report.k}:   {m.hit_rate_at_k * 100:5.1f}%")
        lines.append(f"  MRR:         {m.mrr:5.3f}")
        lines.append(f"  Latency:     {m.median_latency_ms:5.1f} ms (median)")
        lines.append("")
    if report.generation:
        g = report.generation
        lines.append("=" * 44)
        lines.append(" Generation Evaluation")
        lines.append("=" * 44)
        lines.append(f"  Answer relevance:  {g.answer_relevance * 100:5.1f}%")
        lines.append(f"  Groundedness:      {g.groundedness * 100:5.1f}%")
        lines.append(f"  Citation accuracy: {g.citation_accuracy * 100:5.1f}%")
        lines.append(f"  LLM latency:       {g.median_llm_latency_ms:5.1f} ms (median)")
        lines.append("")
    return "\n".join(lines)
