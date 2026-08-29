from app.evaluation.dataset import EvalQuestion
from app.evaluation.evaluator import STRATEGIES, Evaluator, format_report
from app.learning.quiz import learning_mode
from app.rag.topics import search_topic
from app.summarization.summarizer import summarize_video


def _questions():
    return [
        EvalQuestion(question="What is RAG?", expected_keywords=["retrieval", "generation"]),
        EvalQuestion(question="What is a vector database?", expected_keywords=["vector", "database"]),
    ]


def test_evaluator_runs_all_strategies(seeded_pipeline, offline_config):
    ev = Evaluator(seeded_pipeline, offline_config)
    report = ev.run(_questions(), k=5, with_generation=False)
    for s in STRATEGIES:
        assert s in report.retrieval
        m = report.retrieval[s]
        assert 0.0 <= m.recall_at_k <= 1.0
        assert 0.0 <= m.precision_at_k <= 1.0
        assert 0.0 <= m.hit_rate_at_k <= 1.0
        assert m.median_latency_ms >= 0.0


def test_report_formats(seeded_pipeline, offline_config):
    ev = Evaluator(seeded_pipeline, offline_config)
    report = ev.run(_questions(), k=5, with_generation=False)
    text = format_report(report)
    assert "Retrieval Evaluation" in text
    assert "Hybrid + Reranking" in text


def test_hybrid_rerank_hit_rate_reasonable(seeded_pipeline, offline_config):
    ev = Evaluator(seeded_pipeline, offline_config)
    report = ev.run(_questions(), k=5, with_generation=False)
    # With this tiny curated corpus the keyword-bearing chunks should be found.
    assert report.retrieval["hybrid_rerank"].hit_rate_at_k > 0.0


def test_summarize_produces_chapters(seeded_pipeline, offline_config):
    summary = summarize_video(
        "ragLECTURE1", offline_config, seeded_pipeline.store, seeded_pipeline.llm
    )
    assert summary is not None
    assert summary.chapters  # fallback guarantees at least some chapters
    for ch in summary.chapters:
        assert ch.seconds >= 0.0


def test_topic_search_has_clickable_links(seeded_pipeline):
    hits = search_topic(seeded_pipeline, "vector database", top_k=3)
    assert hits
    assert all("youtube.com/watch?v=" in h.url for h in hits)


def test_learning_mode_returns_pack(seeded_pipeline, offline_config):
    pack = learning_mode(
        "ragLECTURE1", offline_config, seeded_pipeline.store, seeded_pipeline.llm, n_questions=3
    )
    assert pack is not None
    assert pack.notes
