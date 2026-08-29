"""Timestamp-aware semantic chunking.

Design note (documented decision)
---------------------------------
Off-the-shelf splitters (e.g. LangChain's ``RecursiveCharacterTextSplitter``)
operate on a flat string and would throw away the per-word timing that makes
this project useful. We therefore implement a bespoke chunker that:

1. Builds a character->time map from the cleaned caption segments.
2. Splits the joined text on sentence boundaries (keeping char offsets).
3. Greedily packs whole sentences into chunks up to ``chunk_size`` characters
   with a trailing ``chunk_overlap`` carried into the next chunk, so related
   sentences stay together and context is not cut mid-thought.
4. Derives each chunk's ``start_time``/``end_time`` by interpolating the
   char->time map at the chunk's first and last characters.

This keeps timestamps accurate end-to-end while still producing
semantically coherent, overlapping chunks.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import List

from app.ingestion.preprocessing import CleanSegment

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    video_id: str
    title: str
    start_time: float
    end_time: float
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def chunk_id(self) -> str:
        return f"{self.video_id}::{self.chunk_index}"


class _TimeMap:
    """Maps a character index in the joined text to a wall-clock time."""

    def __init__(self, segments: List[CleanSegment]) -> None:
        self.pieces = []  # (char_start, char_end, t_start, t_end)
        self.text_parts: List[str] = []
        cursor = 0
        for i, seg in enumerate(segments):
            piece = seg["text"]
            if i > 0:
                self.text_parts.append(" ")
                cursor += 1
            start_char = cursor
            self.text_parts.append(piece)
            cursor += len(piece)
            self.pieces.append((start_char, cursor, seg["start"], seg["end"]))
        self.full_text = "".join(self.text_parts)

    def time_at(self, char_index: int) -> float:
        char_index = max(0, min(char_index, len(self.full_text)))
        for c0, c1, t0, t1 in self.pieces:
            if c0 <= char_index <= c1:
                if c1 == c0:
                    return t0
                frac = (char_index - c0) / (c1 - c0)
                return t0 + frac * (t1 - t0)
        return self.pieces[-1][3] if self.pieces else 0.0


def _split_sentences(text: str):
    """Yield (sentence, start_char, end_char) preserving offsets."""
    idx = 0
    for part in _SENTENCE_END_RE.split(text):
        if not part:
            idx += 1
            continue
        start = text.find(part, idx)
        if start == -1:
            start = idx
        end = start + len(part)
        idx = end
        yield part.strip(), start, end


def chunk_transcript(
    segments: List[CleanSegment],
    video_id: str,
    title: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """Chunk cleaned segments into overlapping, timestamp-tagged chunks."""
    if not segments:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    tmap = _TimeMap(segments)
    sentences = list(_split_sentences(tmap.full_text))
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current: List[tuple] = []  # (sentence, start_char, end_char)
    current_len = 0
    chunk_index = 0

    def emit(sent_group: List[tuple]) -> None:
        nonlocal chunk_index
        if not sent_group:
            return
        text = " ".join(s[0] for s in sent_group).strip()
        if not text:
            return
        first_char = sent_group[0][1]
        last_char = sent_group[-1][2]
        chunks.append(
            Chunk(
                text=text,
                video_id=video_id,
                title=title,
                start_time=round(tmap.time_at(first_char), 3),
                end_time=round(tmap.time_at(last_char), 3),
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1

    for sent in sentences:
        s_text = sent[0]
        s_len = len(s_text) + 1
        # A single sentence larger than chunk_size still forms its own chunk.
        if current and current_len + s_len > chunk_size:
            emit(current)
            # Build overlap tail from the end of the just-emitted group.
            overlap: List[tuple] = []
            acc = 0
            for prev in reversed(current):
                acc += len(prev[0]) + 1
                overlap.insert(0, prev)
                if acc >= chunk_overlap:
                    break
            current = overlap[:]
            current_len = sum(len(s[0]) + 1 for s in current)
        current.append(sent)
        current_len += s_len

    emit(current)
    return chunks
