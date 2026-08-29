"""Video summarization and chapter generation.

Chapters are validated against real transcript timestamps: whatever the LLM
proposes is snapped to the nearest actual chunk start time, so a chapter marker
can never point at a fabricated moment. A deterministic fallback produces evenly
spaced chapters if the LLM is unavailable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import Config
from app.llm.provider import BaseLLM
from app.rag.prompts import build_chapters_messages, build_summary_messages
from app.storage.vector_store import BaseVectorStore
from app.utils import format_timestamp, log, youtube_timestamp_url


@dataclass
class Chapter:
    seconds: float
    title: str

    def render(self, video_id: str) -> str:
        return (
            f"{format_timestamp(self.seconds)} {self.title}\n"
            f"    {youtube_timestamp_url(video_id, self.seconds)}"
        )


@dataclass
class VideoSummary:
    video_id: str
    title: str
    overview: str
    chapters: List[Chapter] = field(default_factory=list)

    def render(self) -> str:
        parts = [f"Summary — {self.title}", "=" * 60, "", self.overview, ""]
        if self.chapters:
            parts.append("Chapters:")
            parts.append("")
            parts.extend(c.render(self.video_id) for c in self.chapters)
        return "\n".join(parts)


def _ordered_chunks(store: BaseVectorStore, video_id: str) -> List[dict]:
    chunks = store.get_chunks(video_ids=[video_id])
    chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
    return chunks


def _timed_context(chunks: List[dict], max_chunks: int = 60) -> str:
    lines = []
    for c in chunks[:max_chunks]:
        secs = int(c["metadata"].get("start_time", 0))
        snippet = " ".join(c["text"].split()[:25])
        lines.append(f"{secs} | {snippet}")
    return "\n".join(lines)


def _plain_context(chunks: List[dict], max_chunks: int = 40) -> str:
    lines = []
    for c in chunks[:max_chunks]:
        ts = format_timestamp(c["metadata"].get("start_time", 0))
        lines.append(f"[{ts}] {c['text']}")
    return "\n\n".join(lines)


def _parse_chapters(text: str, valid_starts: List[float]) -> List[Chapter]:
    chapters: List[Chapter] = []
    seen = set()
    for line in text.splitlines():
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*\|\s*(.+)", line.strip())
        if not m:
            continue
        raw_secs = float(m.group(1))
        title = m.group(2).strip().lstrip("-• ").strip()
        if not title:
            continue
        snapped = min(valid_starts, key=lambda s: abs(s - raw_secs))
        if snapped in seen:
            continue
        seen.add(snapped)
        chapters.append(Chapter(seconds=snapped, title=title))
    chapters.sort(key=lambda c: c.seconds)
    return chapters


def _fallback_chapters(chunks: List[dict], n: int = 6) -> List[Chapter]:
    if not chunks:
        return []
    step = max(1, len(chunks) // n)
    chapters = []
    for i in range(0, len(chunks), step):
        c = chunks[i]
        secs = float(c["metadata"].get("start_time", 0))
        title = " ".join(c["text"].split()[:6]) + "…"
        chapters.append(Chapter(seconds=secs, title=title))
    return chapters[:n]


def summarize_video(
    video_id: str,
    config: Config,
    store: BaseVectorStore,
    llm: BaseLLM,
) -> Optional[VideoSummary]:
    chunks = _ordered_chunks(store, video_id)
    if not chunks:
        return None
    title = chunks[0]["metadata"].get("title", video_id)
    valid_starts = [float(c["metadata"].get("start_time", 0)) for c in chunks]

    # Overview.
    try:
        overview = llm.chat(build_summary_messages(title, _plain_context(chunks)))
    except Exception as exc:  # noqa: BLE001
        log.error("Summary LLM call failed: %s", exc)
        overview = "(Summary unavailable — LLM call failed.)"

    # Chapters (validated against real timestamps).
    chapters: List[Chapter] = []
    try:
        raw = llm.chat(build_chapters_messages(title, _timed_context(chunks)))
        chapters = _parse_chapters(raw, valid_starts)
    except Exception as exc:  # noqa: BLE001
        log.warning("Chapter LLM call failed: %s", exc)
    if not chapters:
        chapters = _fallback_chapters(chunks)

    return VideoSummary(
        video_id=video_id, title=title, overview=overview, chapters=chapters
    )
