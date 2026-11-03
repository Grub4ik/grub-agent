"""Search YouTube and open the best title/artist match's watch page."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import webbrowser
from typing import Any


YOUTUBE_PLAY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
            "description": "Exact video/song title and, when known, artist; omit commands like 'play' or 'open'.",
        }
    },
    "required": ["query"],
    "additionalProperties": False,
}


_QUERY_STOP_WORDS = {
    "включи", "включить", "найди", "найти", "открой", "открыть", "поставь", "поставить",
    "запусти", "запустить", "видео", "песню", "песня", "трек", "музыку", "музыка",
    "ютуб", "youtube", "на", "мне", "пожалуйста", "от", "the", "a", "an", "song", "video",
}


def _search_tokens(text: str) -> set[str]:
    """Normalize search text into meaningful words for title/artist matching."""
    return {
        token
        for token in re.findall(r"[\w]+", text.casefold())
        if len(token) > 1 and token not in _QUERY_STOP_WORDS
    }


def _candidate_score(query_tokens: set[str], video: dict[str, Any]) -> tuple[float, int, bool]:
    """Score a result by matching requested words in its title and artist/channel metadata."""
    title_tokens = _search_tokens(str(video.get("title") or ""))
    artist_tokens = _search_tokens(" ".join(
        str(video.get(key) or "") for key in ("artist", "creator", "channel", "uploader")
    ))

    title_matches = query_tokens & title_tokens
    artist_matches = query_tokens & artist_tokens
    # A title match matters most: matching only the artist should not select one of
    # that artist's unrelated songs. Artist metadata helps distinguish same-titled videos.
    score = 3.0 * len(title_matches) + 1.5 * len(artist_matches)
    coverage = len(title_matches | artist_matches)
    return score, coverage, bool(title_matches)


def play_youtube_video(query: str) -> dict[str, str]:
    """Search YouTube, rank returned results against the requested title/artist, and open the best match."""
    query = query.strip()
    if not query:
        raise ValueError("YouTube search query cannot be empty.")
    if len(query) > 200:
        raise ValueError("YouTube search query cannot exceed 200 characters.")

    executable = shutil.which("yt-dlp")
    if executable is None:
        raise RuntimeError("yt-dlp is missing; install it with: python -m pip install yt-dlp")

    command = [
        executable,
        "--ignore-config",
        "--dump-single-json",
        "--flat-playlist",
        "--skip-download",
        "--no-warnings",
        "--no-progress",
        "--socket-timeout", "10",
        "--retries", "0",
        "--extractor-retries", "0",
        f"ytsearch10:{query}",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("YouTube search timed out after 25 seconds.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not start yt-dlp: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:] or f"yt-dlp exited with status {completed.returncode}"
        raise RuntimeError(f"YouTube search failed: {detail}")
    try:
        search_result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("yt-dlp returned malformed search metadata.") from exc

    entries = search_result.get("entries") if isinstance(search_result, dict) else None
    videos = [item for item in entries or [] if isinstance(item, dict)]
    if not videos:
        raise RuntimeError("YouTube search returned no videos.")

    query_tokens = _search_tokens(query)
    if not query_tokens:
        raise ValueError("YouTube search query must contain a title, artist, or topic.")

    ranked_videos = sorted(
        videos,
        key=lambda item: _candidate_score(query_tokens, item),
        reverse=True,
    )
    video = ranked_videos[0]
    best_score, best_coverage, has_title_match = _candidate_score(query_tokens, video)
    if not has_title_match or best_score <= 0 or best_coverage == 0:
        raise RuntimeError("YouTube returned results, but none matched the requested title or artist.")

    video_id = video.get("id")
    if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id):
        raise RuntimeError("YouTube returned an invalid video ID.")

    title = str(video.get("title") or "YouTube video")[:300]
    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        opened = webbrowser.open(url, new=2)
    except Exception as exc:
        raise RuntimeError(f"Could not open the default browser: {exc}") from exc
    if not opened:
        raise RuntimeError("No default browser accepted the YouTube video URL.")

    return {"status": "opened", "query": query, "title": title, "url": url}
