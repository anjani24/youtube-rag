"""YouTube URL/id helpers and best-effort title retrieval.

Title lookup uses the public oEmbed endpoint, which needs no API key. If the
network call fails we degrade gracefully to using the video id as the title.
"""
from __future__ import annotations

from typing import Optional

import requests

from app.utils import extract_video_id, is_valid_youtube_url, log

__all__ = ["extract_video_id", "is_valid_youtube_url", "get_video_title"]

_OEMBED = "https://www.youtube.com/oembed"


def get_video_title(video_id: str, timeout: int = 15) -> str:
    """Return the human title for a video, or the id if it cannot be fetched."""
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        resp = requests.get(
            _OEMBED,
            params={"url": watch_url, "format": "json"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            title = data.get("title")
            if title:
                return str(title)
        log.warning("oEmbed returned status %s for %s", resp.status_code, video_id)
    except Exception as exc:  # noqa: BLE001 - network best-effort only
        log.warning("Could not fetch title for %s: %s", video_id, exc)
    return video_id
