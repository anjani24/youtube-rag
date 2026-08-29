"""Transcript retrieval with timestamp preservation.

The provider is an abstraction so tests can inject a fake and so alternative
back-ends (e.g. Whisper on downloaded audio) can be added later without
touching the rest of the pipeline.

A raw transcript segment is a dict::

    {"text": str, "start": float, "duration": float}
"""
from __future__ import annotations

from typing import List, Protocol, TypedDict

from app.utils import log


class Segment(TypedDict):
    text: str
    start: float
    duration: float


class TranscriptError(Exception):
    """Raised when a transcript cannot be retrieved."""


class TranscriptUnavailable(TranscriptError):
    """The video has no usable captions / transcripts."""


class TranscriptProvider(Protocol):
    def fetch(self, video_id: str) -> List[Segment]:  # pragma: no cover - protocol
        ...


class YouTubeTranscriptProvider:
    """Adapter over ``youtube-transcript-api`` that tolerates its API changes.

    Version 0.6.x exposes ``YouTubeTranscriptApi.get_transcript`` (classmethod)
    while 1.x switched to an instance ``fetch`` returning objects. We support
    both and always normalise to plain ``Segment`` dicts.
    """

    def __init__(self, languages: List[str] | None = None) -> None:
        self.languages = languages or ["en", "en-US", "en-GB"]

    def fetch(self, video_id: str) -> List[Segment]:
        try:
            from youtube_transcript_api import (  # type: ignore
                YouTubeTranscriptApi,
            )
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise TranscriptError(
                "youtube-transcript-api is not installed. Run "
                "`pip install youtube-transcript-api`."
            ) from exc

        try:
            from youtube_transcript_api._errors import (  # type: ignore
                NoTranscriptFound,
                TranscriptsDisabled,
                VideoUnavailable,
            )
        except Exception:  # pragma: no cover - error classes moved between versions
            NoTranscriptFound = TranscriptsDisabled = VideoUnavailable = Exception  # type: ignore

        # --- 1.x instance API ------------------------------------------------
        raw = None
        if hasattr(YouTubeTranscriptApi, "__init__") and hasattr(
            YouTubeTranscriptApi(), "fetch"
        ):
            try:
                api = YouTubeTranscriptApi()
                fetched = api.fetch(video_id, languages=self.languages)
                raw = getattr(fetched, "to_raw_data", lambda: fetched)()
            except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as exc:
                raise TranscriptUnavailable(
                    f"No transcript available for video '{video_id}': {exc}"
                ) from exc
            except TypeError:
                raw = None  # signature mismatch -> fall through to classmethod
            except Exception as exc:  # noqa: BLE001
                log.warning("1.x fetch failed for %s: %s", video_id, exc)
                raw = None

        # --- 0.6.x classmethod API ------------------------------------------
        if raw is None and hasattr(YouTubeTranscriptApi, "get_transcript"):
            try:
                raw = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=self.languages
                )
            except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as exc:
                raise TranscriptUnavailable(
                    f"No transcript available for video '{video_id}': {exc}"
                ) from exc
            except Exception as exc:  # noqa: BLE001
                raise TranscriptError(
                    f"Failed to fetch transcript for '{video_id}': {exc}"
                ) from exc

        if raw is None:
            raise TranscriptError(
                f"Unsupported youtube-transcript-api version for '{video_id}'."
            )

        return _normalise(raw, video_id)


def _normalise(raw, video_id: str) -> List[Segment]:
    segments: List[Segment] = []
    for item in raw:
        if isinstance(item, dict):
            text = item.get("text", "")
            start = item.get("start", 0.0)
            duration = item.get("duration", 0.0)
        else:  # transcript objects expose attributes
            text = getattr(item, "text", "")
            start = getattr(item, "start", 0.0)
            duration = getattr(item, "duration", 0.0)
        text = (text or "").strip()
        if not text:
            continue
        segments.append(
            Segment(text=text, start=float(start), duration=float(duration))
        )
    if not segments:
        raise TranscriptUnavailable(
            f"Transcript for '{video_id}' was empty after cleaning."
        )
    return segments
