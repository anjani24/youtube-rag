"""Shared utilities: logging, retry, YouTube URL parsing and timestamp helpers."""
from __future__ import annotations

import functools
import logging
import re
import time
from typing import Callable, Optional, TypeVar
from urllib.parse import parse_qs, urlparse

T = TypeVar("T")

_LOGGER_CONFIGURED = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logging once and return the app logger."""
    global _LOGGER_CONFIGURED
    logger = logging.getLogger("ytrag")
    if not _LOGGER_CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
        _LOGGER_CONFIGURED = True
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


log = setup_logging()


# --------------------------------------------------------------------------- #
# Retry decorator for transient failures (network / API hiccups).
# --------------------------------------------------------------------------- #
def with_retries(
    max_retries: int = 3,
    backoff: float = 1.5,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a callable with exponential backoff on the given exceptions."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001 - deliberate broad retry
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    sleep_for = backoff ** attempt
                    log.warning(
                        "%s failed (attempt %d/%d): %s -- retrying in %.1fs",
                        fn.__name__,
                        attempt,
                        max_retries,
                        exc,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# YouTube URL handling
# --------------------------------------------------------------------------- #
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Return the 11-char YouTube video id from a URL or bare id, else ``None``.

    Supports the common URL shapes::

        https://www.youtube.com/watch?v=ID
        https://youtu.be/ID
        https://www.youtube.com/embed/ID
        https://www.youtube.com/shorts/ID
        https://m.youtube.com/watch?v=ID
        ID   (already a bare id)
    """
    if not url_or_id or not isinstance(url_or_id, str):
        return None
    candidate = url_or_id.strip()

    # Already a bare id.
    if _VIDEO_ID_RE.match(candidate):
        return candidate

    if "://" not in candidate:
        candidate = "https://" + candidate

    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None

    host = (parsed.hostname or "").lower().lstrip("www.")

    if host in {"youtu.be"}:
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid if _VIDEO_ID_RE.match(vid) else None

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [None])[0]
            return vid if vid and _VIDEO_ID_RE.match(vid) else None
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "v", "live"}:
            vid = parts[1]
            return vid if _VIDEO_ID_RE.match(vid) else None

    return None


def is_valid_youtube_url(url: str) -> bool:
    return extract_video_id(url) is not None


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``mm:ss`` or ``h:mm:ss``."""
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def youtube_timestamp_url(video_id: str, seconds: float) -> str:
    """Build a clickable, deep-linked YouTube URL for the given timestamp."""
    t = int(max(0, seconds))
    return f"https://www.youtube.com/watch?v={video_id}&t={t}s"
