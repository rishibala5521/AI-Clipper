import pytest

from errors import AppError
from video_url import extract_video_id

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtube.com/watch?v={VIDEO_ID}&t=42s",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?si=abc",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
        f"www.youtube.com/watch?v={VIDEO_ID}",
        f"  https://youtu.be/{VIDEO_ID}  ",
    ],
)
def test_valid_urls(url):
    assert extract_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "javascript:alert(1)",
        f"ftp://youtube.com/watch?v={VIDEO_ID}",
        f"https://evil.com/watch?v={VIDEO_ID}",
        f"https://youtube.com.evil.com/watch?v={VIDEO_ID}",
        f"https://youtube.com@evil.com/watch?v={VIDEO_ID}",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/playlist?list=PL12345678901",
    ],
)
def test_invalid_urls(url):
    with pytest.raises(AppError) as exc_info:
        extract_video_id(url)
    assert exc_info.value.code == "invalid_video_url"
    assert exc_info.value.status_code == 400