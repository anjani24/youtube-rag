"""Video ingestion orchestrator.

``ingest_video`` runs the full front half of the pipeline and is deliberately
defensive: invalid URLs, missing captions, empty transcripts, duplicates and
embedding failures all return a structured, non-crashing result.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from app.chunking.chunker import chunk_transcript
from app.config import Config
from app.embeddings.embedder import BaseEmbedder
from app.ingestion.preprocessing import preprocess
from app.ingestion.transcript import (
    TranscriptError,
    TranscriptProvider,
    TranscriptUnavailable,
    YouTubeTranscriptProvider,
)
from app.ingestion.youtube import extract_video_id, get_video_title
from app.storage.vector_store import BaseVectorStore
from app.utils import log


@dataclass
class IngestResult:
    ok: bool
    video_id: Optional[str] = None
    title: Optional[str] = None
    chunks: int = 0
    message: str = ""
    duplicate: bool = False


def ingest_video(
    url: str,
    config: Config,
    store: BaseVectorStore,
    embedder: BaseEmbedder,
    transcript_provider: Optional[TranscriptProvider] = None,
    title_fetcher=get_video_title,
    force: bool = False,
) -> IngestResult:
    # 1. Validate URL / extract id.
    video_id = extract_video_id(url)
    if not video_id:
        return IngestResult(ok=False, message=f"Invalid YouTube URL or id: '{url}'.")

    # 2. Duplicate detection.
    if not force and store.has_video(video_id):
        return IngestResult(
            ok=True,
            video_id=video_id,
            duplicate=True,
            message=f"Video '{video_id}' is already indexed. Skipping.",
        )

    # 3. Fetch transcript (with clear failure modes).
    provider = transcript_provider or YouTubeTranscriptProvider()
    try:
        raw_segments = provider.fetch(video_id)
    except TranscriptUnavailable as exc:
        return IngestResult(ok=False, video_id=video_id, message=str(exc))
    except TranscriptError as exc:
        return IngestResult(
            ok=False, video_id=video_id, message=f"Transcript error: {exc}"
        )
    except Exception as exc:  # noqa: BLE001 - network / unexpected
        return IngestResult(
            ok=False,
            video_id=video_id,
            message=f"Unexpected error fetching transcript: {exc}",
        )

    # 4. Title (best-effort).
    try:
        title = title_fetcher(video_id)
    except Exception:  # noqa: BLE001
        title = video_id

    # 5. Clean + 6. chunk (timestamps preserved throughout).
    cleaned = preprocess(raw_segments)
    if not cleaned:
        return IngestResult(
            ok=False, video_id=video_id, message="Transcript was empty after cleaning."
        )

    chunks = chunk_transcript(
        cleaned,
        video_id=video_id,
        title=title,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    if not chunks:
        return IngestResult(
            ok=False, video_id=video_id, message="Chunking produced no chunks."
        )

    # Persist a raw transcript copy for reference / reproducibility.
    _dump_transcript(config, video_id, title, cleaned)

    # 7. Embed.
    try:
        embeddings = embedder.embed_documents([c.text for c in chunks])
    except Exception as exc:  # noqa: BLE001
        return IngestResult(
            ok=False, video_id=video_id, message=f"Embedding failed: {exc}"
        )

    # 8. Store (idempotent: clear any partial prior state on force).
    if force and store.has_video(video_id):
        store.delete_video(video_id)
    try:
        store.add_chunks(chunks, embeddings)
    except Exception as exc:  # noqa: BLE001
        return IngestResult(
            ok=False, video_id=video_id, message=f"Vector store write failed: {exc}"
        )

    log.info("Indexed '%s' (%s) with %d chunks.", title, video_id, len(chunks))
    return IngestResult(
        ok=True,
        video_id=video_id,
        title=title,
        chunks=len(chunks),
        message=f"Indexed '{title}' with {len(chunks)} chunks.",
    )


def _dump_transcript(config: Config, video_id: str, title: str, cleaned) -> None:
    try:
        os.makedirs(config.transcripts_dir, exist_ok=True)
        path = os.path.join(config.transcripts_dir, f"{video_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {"video_id": video_id, "title": title, "segments": cleaned},
                fh,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as exc:  # noqa: BLE001 - non-fatal
        log.warning("Could not write transcript copy for %s: %s", video_id, exc)
