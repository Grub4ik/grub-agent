"""Search YouTube and open the best title/artist match's watch page."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import webbrowser
from typing import Any
from urllib.parse import parse_qs, urlsplit
from urllib.request import getproxies, proxy_bypass

from .web_search import web_search


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
_VERSION_MARKERS = {"acoustic", "live", "cover", "remix", "slowed", "sped", "nightcore", "instrumental"}


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
        str(video.get(key) or "") for key in ("artist", "creator", "channel", "uploader", "snippet")
    ))

    title_matches = query_tokens & title_tokens
    artist_matches = query_tokens & artist_tokens
    # A title match matters most: matching only the artist should not select one of
    # that artist's unrelated songs. Artist metadata helps distinguish same-titled videos.
    score = 3.0 * len(title_matches) + 1.5 * len(artist_matches)
    unrequested_versions = (_search_tokens(str(video.get("title") or "")) & _VERSION_MARKERS) - query_tokens
    score -= 5.0 * len(unrequested_versions)
    coverage = len(title_matches | artist_matches)
    return score, coverage, bool(title_matches)


def _youtube_video_id(url: str) -> str | None:
    """Extract a video ID only from a recognized YouTube watch/video URL."""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"} and parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    elif host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    else:
        return None
    return video_id if isinstance(video_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id) else None


def _search_youtube_web(query: str) -> dict[str, str] | None:
    """Find and score YouTube watch-page results via the existing DuckDuckGo web search."""
    search_result = web_search(f"{query} YouTube")
    query_tokens = _search_tokens(query)
    videos: list[tuple[float, int, dict[str, str]]] = []
    for position, result in enumerate(search_result.get("results", [])):
        video_id = _youtube_video_id(str(result.get("url") or ""))
        if not video_id:
            continue
        title = re.sub(r"\s*-\s*YouTube\s*$", "", str(result.get("title") or ""), flags=re.IGNORECASE)
        candidate = {
            "id": video_id,
            "title": title or "YouTube video",
            "snippet": str(result.get("snippet") or ""),
        }
        score, coverage, has_title_match = _candidate_score(query_tokens, candidate)
        if has_title_match and coverage and score > 0:
            videos.append((score - position * 0.05, position, candidate))

    if not videos:
        return None
    videos.sort(key=lambda item: (-item[0], item[1]))
    return {**videos[0][2], "provider": "DuckDuckGo"}


def _search_youtube_ytdlp(query: str) -> dict[str, str] | None:
    """Search with yt-dlp using the configured HTTPS proxy without downloading media."""
    executable = shutil.which("yt-dlp")
    if executable is None:
        return None
    command = [executable, "--ignore-config"]
    if not proxy_bypass("www.youtube.com"):
        proxies = getproxies()
        proxy = proxies.get("https") or proxies.get("all") or proxies.get("http")
        if proxy:
            command.extend(["--proxy", proxy])
    command.extend([
        "--dump-single-json", "--flat-playlist", "--skip-download",
        "--no-warnings", "--no-progress", "--socket-timeout", "10", "--retries", "0",
        "--extractor-retries", "0", f"ytsearch10:{query}",
    ])
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
        if "handshake operation timed out" in detail.lower() or "timed out" in detail.lower():
            detail = "The HTTPS connection to YouTube timed out; check the VPN/proxy connection and retry."
        raise RuntimeError(f"YouTube search failed: {detail}")
    try:
        search_result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("yt-dlp returned malformed search metadata.") from exc

    entries = search_result.get("entries") if isinstance(search_result, dict) else None
    videos = [item for item in entries or [] if isinstance(item, dict)]
    if not videos:
        return None

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
        return None

    video_id = video.get("id")
    if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id):
        return None

    title = str(video.get("title") or "YouTube video")[:300]
    return {"id": video_id, "title": title, "provider": "yt-dlp"}


def play_youtube_video(query: str) -> dict[str, str]:
    """Search and open the best matching YouTube video, using yt-dlp if web search finds none."""
    query = query.strip()
    if not query:
        raise ValueError("YouTube search query cannot be empty.")
    if len(query) > 200:
        raise ValueError("YouTube search query cannot exceed 200 characters.")
    if not _search_tokens(query):
        raise ValueError("YouTube search query must contain a title, artist, or topic.")

    errors: list[str] = []
    video: dict[str, str] | None = None
    try:
        video = _search_youtube_ytdlp(query)
    except Exception as exc:
        errors.append(str(exc))

    if video is None:
        try:
            video = _search_youtube_web(query)
        except Exception as exc:
            errors.append(f"DuckDuckGo search failed: {exc}")

    if video is None:
        if errors:
            raise RuntimeError("No matching YouTube video could be found. " + "; ".join(errors))
        raise RuntimeError("No matching YouTube video was found.")

    url = f"https://www.youtube.com/watch?v={video['id']}"

    try:
        opened = webbrowser.open(url, new=2)
    except Exception as exc:
        raise RuntimeError(f"Could not open the default browser: {exc}") from exc
    if not opened:
        raise RuntimeError("No default browser accepted the YouTube video URL.")

    return {"status": "opened", "query": query, "title": video["title"], "url": url, "provider": video["provider"]}
