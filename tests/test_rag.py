from app.llm.provider import FakeLLM
from app.rag.prompts import INSUFFICIENT_EVIDENCE
from app.rag.qa import RAGPipeline
from app.retrieval.reranker import IdentityReranker
from app.storage.vector_store import InMemoryVectorStore
from app.embeddings.embedder import HashEmbedder


def test_relevant_question_produces_answer_and_citations(seeded_pipeline):
    ans = seeded_pipeline.answer_question("What is retrieval augmented generation?")
    assert ans.sufficient
    assert ans.text
    assert ans.sources
    assert ans.citations
    # Every citation URL is a real deep link.
    assert all("youtube.com/watch?v=" in c.url for c in ans.citations)


def test_empty_query_handled(seeded_pipeline):
    ans = seeded_pipeline.answer_question("   ")
    assert not ans.sufficient


def test_insufficient_evidence_when_llm_refuses(offline_config):
    # A fake LLM that always refuses simulates "not enough evidence".
    refusing = FakeLLM(responder=lambda msgs: INSUFFICIENT_EVIDENCE)
    pipe = RAGPipeline(
        offline_config,
        store=InMemoryVectorStore(offline_config),
        embedder=HashEmbedder(dim=offline_config.embedding_dim),
        llm=refusing,
        reranker=IdentityReranker(),
    )
    # No documents indexed -> empty retrieval -> insufficient.
    ans = pipe.answer_question("totally unrelated question about cooking pasta")
    assert not ans.sufficient
    assert ans.text == INSUFFICIENT_EVIDENCE
    assert ans.citations == []


def test_high_min_relevance_triggers_refusal(seeded_pipeline):
    seeded_pipeline.config.min_relevance_score = 999.0  # impossible threshold
    ans = seeded_pipeline.answer_question("What is RAG?")
    assert not ans.sufficient
    assert ans.text == INSUFFICIENT_EVIDENCE


def test_comparison_requires_two_videos(seeded_pipeline):
    ans = seeded_pipeline.compare_videos("compare", ["ragLECTURE1"])
    assert not ans.sufficient


def test_comparison_across_two_videos(seeded_pipeline):
    ans = seeded_pipeline.compare_videos(
        "Compare how these explain vector databases",
        ["ragLECTURE1", "vectorDB002"],
    )
    assert ans.text
    assert ans.sources
