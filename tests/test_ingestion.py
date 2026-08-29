from app.embeddings.embedder import HashEmbedder
from app.ingestion.ingest import ingest_video
from app.ingestion.transcript import TranscriptError
from app.storage.vector_store import InMemoryVectorStore


def _fresh(offline_config):
    return InMemoryVectorStore(offline_config), HashEmbedder(dim=offline_config.embedding_dim)


def test_successful_ingestion(offline_config, fake_provider):
    store, embedder = _fresh(offline_config)
    res = ingest_video(
        "https://youtu.be/ragLECTURE1",
        offline_config,
        store,
        embedder,
        transcript_provider=fake_provider,
        title_fetcher=lambda v: "RAG Lecture",
    )
    assert res.ok
    assert res.video_id == "ragLECTURE1"
    assert res.chunks > 0
    assert store.has_video("ragLECTURE1")


def test_duplicate_video_is_skipped(offline_config, fake_provider):
    store, embedder = _fresh(offline_config)
    kwargs = dict(
        config=offline_config,
        store=store,
        embedder=embedder,
        transcript_provider=fake_provider,
        title_fetcher=lambda v: "RAG Lecture",
    )
    first = ingest_video("https://youtu.be/ragLECTURE1", **kwargs)
    second = ingest_video("https://youtu.be/ragLECTURE1", **kwargs)
    assert first.ok and not first.duplicate
    assert second.ok and second.duplicate


def test_invalid_url_returns_error(offline_config, fake_provider):
    store, embedder = _fresh(offline_config)
    res = ingest_video(
        "https://example.com/not-a-video",
        offline_config,
        store,
        embedder,
        transcript_provider=fake_provider,
    )
    assert not res.ok
    assert "Invalid" in res.message


def test_missing_transcript_handled(offline_config, fake_provider):
    store, embedder = _fresh(offline_config)
    res = ingest_video(
        "https://youtu.be/noCaption01",
        offline_config,
        store,
        embedder,
        transcript_provider=fake_provider,
    )
    assert not res.ok
    assert "No captions" in res.message or "No transcript" in res.message


def test_transcript_api_failure_handled(offline_config):
    store, embedder = _fresh(offline_config)

    class BoomProvider:
        def fetch(self, video_id):
            raise TranscriptError("simulated API failure")

    res = ingest_video(
        "https://youtu.be/dQw4w9WgXcQ",
        offline_config,
        store,
        embedder,
        transcript_provider=BoomProvider(),
    )
    assert not res.ok
    assert "Transcript error" in res.message
