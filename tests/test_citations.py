from app.rag.citations import build_citations, format_answer_with_sources
from app.retrieval.base import RetrievalResult


def _result(vid, start):
    return RetrievalResult(
        id=f"{vid}::0",
        text="some chunk text",
        metadata={"video_id": vid, "title": f"Title {vid}", "start_time": start, "end_time": start + 5},
    )


def test_citations_carry_correct_video_and_timestamp():
    results = [_result("dQw4w9WgXcQ", 754.0), _result("abcdefghijk", 12.0)]
    cites = build_citations(results)
    assert len(cites) == 2
    assert cites[0].video_id == "dQw4w9WgXcQ"
    assert cites[0].timestamp == "12:34"
    assert cites[0].index == 1


def test_citation_urls_are_valid_youtube_links():
    cites = build_citations([_result("dQw4w9WgXcQ", 754.0)])
    assert cites[0].url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=754s"


def test_no_fabricated_timestamps():
    # Citation timestamps must equal the retrieved chunk start times.
    results = [_result("dQw4w9WgXcQ", 60.0)]
    cites = build_citations(results)
    assert cites[0].start_time == results[0].start_time


def test_format_answer_appends_sources():
    cites = build_citations([_result("dQw4w9WgXcQ", 0.0)])
    out = format_answer_with_sources("An answer.", cites)
    assert "An answer." in out
    assert "Sources:" in out
    assert "youtube.com/watch?v=dQw4w9WgXcQ" in out
