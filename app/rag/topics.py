"""Topic search: locate where a topic is discussed, with clickable timestamps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.rag.qa import RAGPipeline
from app.utils import format_timestamp, youtube_timestamp_url


@dataclass
class TopicHit:
    title: str
    video_id: str
    timestamp: str
    url: str
    snippet: str

    def render(self, index: int) -> str:
        return (
            f"{index}. {self.timestamp} — {self.title}\n"
            f"    {self.snippet}\n"
            f"    {self.url}"
        )


def search_topic(
    pipeline: RAGPipeline,
    query: str,
    video_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> List[TopicHit]:
    results = pipeline.retrieve(query, video_ids=video_ids, strategy="hybrid_rerank")
    if top_k:
        results = results[:top_k]
    hits: List[TopicHit] = []
    for r in results:
        snippet = " ".join(r.text.split()[:24])
        hits.append(
            TopicHit(
                title=r.title,
                video_id=r.video_id,
                timestamp=format_timestamp(r.start_time),
                url=youtube_timestamp_url(r.video_id, r.start_time),
                snippet=snippet + ("…" if len(r.text.split()) > 24 else ""),
            )
        )
    return hits
