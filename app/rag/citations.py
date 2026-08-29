"""Citation construction.

Citations are built *only* from actually retrieved chunks, so a timestamp can
never be fabricated. Each citation is numbered to match the ``[n]`` markers the
LLM was shown, and carries a deep-linked, clickable YouTube URL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.retrieval.base import RetrievalResult
from app.utils import format_timestamp, youtube_timestamp_url


@dataclass
class Citation:
    index: int
    video_id: str
    title: str
    start_time: float
    timestamp: str
    url: str

    def render(self) -> str:
        return f"[{self.index}] {self.title} — {self.timestamp}\n    {self.url}"


def build_citations(results: List[RetrievalResult]) -> List[Citation]:
    citations: List[Citation] = []
    for i, r in enumerate(results, start=1):
        citations.append(
            Citation(
                index=i,
                video_id=r.video_id,
                title=r.title,
                start_time=r.start_time,
                timestamp=format_timestamp(r.start_time),
                url=youtube_timestamp_url(r.video_id, r.start_time),
            )
        )
    return citations


def render_sources_block(citations: List[Citation]) -> str:
    if not citations:
        return ""
    lines = ["Sources:", ""]
    lines.extend(c.render() for c in citations)
    return "\n".join(lines)


def format_answer_with_sources(answer: str, citations: List[Citation]) -> str:
    block = render_sources_block(citations)
    return f"{answer}\n\n{block}" if block else answer
