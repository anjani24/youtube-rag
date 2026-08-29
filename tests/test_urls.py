from app.utils import (
    extract_video_id,
    format_timestamp,
    is_valid_youtube_url,
    youtube_timestamp_url,
)

VALID = "dQw4w9WgXcQ"


def test_standard_watch_url():
    assert extract_video_id(f"https://www.youtube.com/watch?v={VALID}") == VALID


def test_short_url():
    assert extract_video_id(f"https://youtu.be/{VALID}") == VALID


def test_embed_and_shorts_and_live():
    assert extract_video_id(f"https://www.youtube.com/embed/{VALID}") == VALID
    assert extract_video_id(f"https://www.youtube.com/shorts/{VALID}") == VALID
    assert extract_video_id(f"https://www.youtube.com/live/{VALID}") == VALID


def test_bare_id():
    assert extract_video_id(VALID) == VALID


def test_url_with_extra_params():
    url = f"https://www.youtube.com/watch?v={VALID}&t=42s&list=PLabc"
    assert extract_video_id(url) == VALID


def test_invalid_urls():
    assert extract_video_id("https://example.com/watch?v=abc") is None
    assert extract_video_id("not a url") is None
    assert extract_video_id("") is None
    assert extract_video_id("https://youtu.be/tooShort") is None


def test_is_valid_youtube_url():
    assert is_valid_youtube_url(f"https://youtu.be/{VALID}")
    assert not is_valid_youtube_url("https://example.com")


def test_timestamp_formatting():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3661) == "1:01:01"


def test_timestamp_url():
    assert youtube_timestamp_url(VALID, 754.6) == (
        f"https://www.youtube.com/watch?v={VALID}&t=754s"
    )
