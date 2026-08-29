"""Transcript cleaning and timestamp-preserving normalisation.

Auto-generated captions are noisy: bracketed sound cues, doubled spaces,
newlines mid-phrase, and very short fragments. We clean the text of each
segment while *keeping* its ``start``/``end`` so downstream chunking can still
build accurate timestamps.
"""
from __future__ import annotations

import re
from typing import List, TypedDict

from app.ingestion.transcript import Segment


class CleanSegment(TypedDict):
    text: str
    start: float
    end: float


_CUE_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # [Music], (applause), etc.
_WS_RE = re.compile(r"\s+")
_DUP_RE = re.compile(r"\b(\w+)( \1\b)+", re.IGNORECASE)  # "the the the" -> "the"


def clean_text(text: str) -> str:
    text = _CUE_RE.sub(" ", text)
    text = text.replace("\n", " ")
    text = _DUP_RE.sub(r"\1", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def preprocess(segments: List[Segment]) -> List[CleanSegment]:
    """Clean each segment and compute its end time. Drops now-empty segments."""
    cleaned: List[CleanSegment] = []
    for seg in segments:
        text = clean_text(seg["text"])
        if not text:
            continue
        start = float(seg["start"])
        end = start + float(seg.get("duration", 0.0) or 0.0)
        cleaned.append(CleanSegment(text=text, start=start, end=max(end, start)))
    return cleaned
