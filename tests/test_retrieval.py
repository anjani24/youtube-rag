from app.retrieval.reranker import IdentityReranker


def test_vector_search_returns_scored_results(seeded_pipeline):
    results = seeded_pipeline.vector.search("what is retrieval augmented generation", top_k=5)
    assert results
    for r in results:
        assert 0.0 <= r.score <= 1.0
        assert r.video_id
        assert r.start_time >= 0.0


def test_bm25_finds_exact_terminology(seeded_pipeline):
    results = seeded_pipeline.bm25.search("vector database indexing", top_k=5)
    assert results
    # BM25 should surface the vector-database video for this keyword query.
    assert any(r.video_id == "vectorDB002" for r in results)


def test_hybrid_merges_and_deduplicates(seeded_pipeline):
    results = seeded_pipeline.hybrid.search("vector database similarity search", top_k=5)
    assert results
    ids = [r.id for r in results]
    assert len(ids) == len(set(ids)), "hybrid results must be deduplicated"
    # RRF score present
    assert all("rrf" in r.scores for r in results)


def test_hybrid_scope_filter(seeded_pipeline):
    results = seeded_pipeline.hybrid.search(
        "retrieval", top_k=5, video_ids=["ragLECTURE1"]
    )
    assert results
    assert all(r.video_id == "ragLECTURE1" for r in results)


def test_identity_reranker_truncates():
    from app.retrieval.base import RetrievalResult

    items = [RetrievalResult(id=str(i), text=f"t{i}", metadata={}, score=1.0 - i * 0.1) for i in range(10)]
    out = IdentityReranker().rerank("q", items, top_k=3)
    assert len(out) == 3
    assert out[0].id == "0"
    assert all("rerank" in r.scores for r in out)


def test_retrieve_strategies_all_run(seeded_pipeline):
    for strategy in ["vector", "bm25", "hybrid", "hybrid_rerank"]:
        results = seeded_pipeline.retrieve("embeddings and vectors", strategy=strategy)
        assert isinstance(results, list)
