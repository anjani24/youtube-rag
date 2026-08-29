"""End-to-end offline demo — no API keys, no network required.

Seeds two canned transcripts into a real persistent ChromaDB (temp dir) using
the deterministic hash embedder + fake LLM, then runs every major feature:
ingest, retrieve, hybrid+rerank answer, citations, topic search, summary,
comparison, learning/quiz, evaluation, and a persistence reload check.

    python scripts/offline_demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Config
from app.embeddings.embedder import HashEmbedder
from app.evaluation.dataset import EvalQuestion
from app.evaluation.evaluator import Evaluator, format_report
from app.ingestion.ingest import ingest_video
from app.ingestion.transcript import Segment
from app.learning.quiz import learning_mode
from app.llm.provider import FakeLLM
from app.rag.qa import RAGPipeline
from app.rag.topics import search_topic
from app.storage.vector_store import ChromaVectorStore
from app.summarization.summarizer import summarize_video


def seg(t, s, d):
    return Segment(text=t, start=float(s), duration=float(d))


TRANSCRIPTS = {
    "ragLECTURE1": [
        seg("Welcome to this lecture on retrieval augmented generation.", 0, 4),
        seg("RAG combines retrieval with generation to give an LLM extra context.", 12, 6),
        seg("We embed documents into vectors using an embedding model.", 22, 6),
        seg("Vectors are stored in a vector database for similarity search.", 35, 6),
        seg("A reranker reorders candidates to improve relevance before the LLM.", 47, 6),
        seg("Chunking with overlap keeps related context together.", 60, 6),
    ],
    "vectorDB002": [
        seg("This video explains vector databases and indexing.", 0, 4),
        seg("A vector database stores embeddings for fast similarity search.", 10, 6),
        seg("Indexing methods like HNSW make nearest neighbor search efficient.", 25, 6),
        seg("We compare exact search versus approximate nearest neighbor indexes.", 40, 6),
    ],
}


class CannedProvider:
    def fetch(self, video_id):
        return TRANSCRIPTS[video_id]


def hr(title):
    print("\n" + "=" * 64 + f"\n {title}\n" + "=" * 64)


def main():
    tmp = tempfile.mkdtemp(prefix="ytrag_demo_")
    cfg = Config(
        llm_provider="fake",
        embedding_provider="hash",
        embedding_dim=128,
        reranker_provider="identity",
        vector_store="chroma",
        chroma_dir=os.path.join(tmp, "chroma_db"),
        collection_name="demo_collection",
        top_k=4,
        rerank_top_k=4,
        chunk_size=220,
        chunk_overlap=40,
    )
    store = ChromaVectorStore(cfg)
    pipe = RAGPipeline(store=store, config=cfg, embedder=HashEmbedder(dim=128), llm=FakeLLM())

    hr("1) INGEST (canned transcripts -> chunk -> embed -> Chroma)")
    for vid in TRANSCRIPTS:
        res = ingest_video(
            f"https://youtu.be/{vid}", cfg, store, pipe.embedder,
            transcript_provider=CannedProvider(), title_fetcher=lambda v: f"Demo: {v}",
        )
        print(f"  {res.message}")
    print(f"  Total chunks in store: {store.count()}")

    hr("2) HYBRID + RERANK RETRIEVAL")
    for r in pipe.retrieve("what is retrieval augmented generation", strategy="hybrid_rerank"):
        print(f"  score={r.score:.4f} @{r.start_time:>5.0f}s [{r.video_id}] {r.text[:60]}…")

    hr("3) GROUNDED ANSWER + TIMESTAMP CITATIONS")
    ans = pipe.answer_question("How does reranking help retrieval?")
    print(ans.render())

    hr("4) TOPIC SEARCH")
    for i, h in enumerate(search_topic(pipe, "vector database indexing", top_k=3), 1):
        print("  " + h.render(i).replace("\n", "\n  "))

    hr("5) SUMMARY WITH REAL-TIMESTAMP CHAPTERS")
    print(summarize_video("ragLECTURE1", cfg, store, pipe.llm).render())

    hr("6) MULTI-VIDEO COMPARISON")
    cmp = pipe.compare_videos("Compare how these explain vector databases", list(TRANSCRIPTS))
    print(f"  sources: {len(cmp.sources)} chunks across {len(TRANSCRIPTS)} videos")
    print(f"  citations: {len(cmp.citations)}")

    hr("7) LEARNING MODE + QUIZ")
    pack = learning_mode("ragLECTURE1", cfg, store, pipe.llm, n_questions=3)
    print(f"  notes chars: {len(pack.notes)}, quiz questions: {len(pack.quiz)}")

    hr("8) EVALUATION (retrieval strategy comparison)")
    questions = [
        EvalQuestion(question="What is RAG?", expected_keywords=["retrieval", "generation"]),
        EvalQuestion(question="What is a vector database?", expected_keywords=["vector", "database"]),
        EvalQuestion(question="How does reranking work?", expected_keywords=["rerank", "relevance"]),
    ]
    report = Evaluator(pipe, cfg).run(questions, k=4, with_generation=False)
    print(format_report(report))

    hr("9) PERSISTENCE RELOAD CHECK")
    reopened = ChromaVectorStore(cfg)
    print(f"  chunks after reopening the DB: {reopened.count()} "
          f"(videos: {len(reopened.list_videos())})")

    print("\nOffline demo complete. Temp DB at:", tmp)


if __name__ == "__main__":
    main()
