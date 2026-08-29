from app.chunking.chunker import chunk_transcript
from app.ingestion.preprocessing import clean_text, preprocess
from app.ingestion.transcript import Segment


def _seg(text, start, dur):
    return Segment(text=text, start=start, duration=dur)


def test_clean_text_removes_cues_and_dupes():
    assert clean_text("[Music] hello   world") == "hello world"
    assert clean_text("the the the model") == "the model"
    assert clean_text("(applause)") == ""


def test_preprocess_computes_end_and_drops_empty():
    segs = [_seg("[Music]", 0.0, 2.0), _seg("Real content here.", 2.0, 3.0)]
    cleaned = preprocess(segs)
    assert len(cleaned) == 1
    assert cleaned[0]["start"] == 2.0
    assert cleaned[0]["end"] == 5.0


def _sample_cleaned():
    return preprocess(
        [
            _seg("Sentence one about embeddings.", 0.0, 5.0),
            _seg("Sentence two about vectors.", 5.0, 5.0),
            _seg("Sentence three about retrieval.", 10.0, 5.0),
            _seg("Sentence four about reranking.", 15.0, 5.0),
            _seg("Sentence five about chunking.", 20.0, 5.0),
        ]
    )


def test_chunks_not_empty_and_have_text():
    chunks = chunk_transcript(_sample_cleaned(), "vidID000001", "T", 80, 20)
    assert len(chunks) >= 2
    assert all(c.text.strip() for c in chunks)


def test_timestamps_preserved_and_ordered():
    chunks = chunk_transcript(_sample_cleaned(), "vidID000001", "T", 80, 20)
    assert chunks[0].start_time == 0.0
    for c in chunks:
        assert c.end_time >= c.start_time
    # chunk indices are sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # end time never exceeds the last segment end (25.0)
    assert max(c.end_time for c in chunks) <= 25.0


def test_overlap_creates_shared_content():
    chunks = chunk_transcript(_sample_cleaned(), "vidID000001", "T", 70, 40)
    # With overlap, consecutive chunks should share at least one sentence.
    assert len(chunks) >= 2
    shared = set(chunks[0].text.split(".")) & set(chunks[1].text.split("."))
    shared = {s.strip() for s in shared if s.strip()}
    assert shared, "expected overlapping content between consecutive chunks"


def test_empty_input():
    assert chunk_transcript([], "vidID000001", "T") == []


def test_chunk_id_format():
    chunks = chunk_transcript(_sample_cleaned(), "vidID000001", "T", 80, 20)
    assert chunks[0].chunk_id == "vidID000001::0"
