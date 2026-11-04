"""Small read-only web search tool using DuckDuckGo's HTML results page."""

from __future__ import annotations

from html.parser import HTMLParser
import re
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen
from typing import Any


WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 2,
            "maxLength": 200,
            "description": (
                "Search terms that preserve the person's/product's/game's exact name as spoken or written by the user. "
                "Do not replace a similar-sounding name with a better-known one. Include relevant topic words."
            ),
        }
    },
    "required": ["query"],
    "additionalProperties": False,
}

_SEARCH_URL = "https://html.duckduckgo.com/html/"
_TIMEOUT_SECONDS = 12
_MAX_RESPONSE_BYTES = 1_000_000
_FETCH_RESULTS = 12
_MAX_RESULTS = 6
_MAX_FIELD_LENGTH = 1_200
_QUERY_STOP_WORDS = {
    "кто", "что", "такой", "такая", "такое", "это", "про", "расскажи", "найди", "информация",
    "известные", "песни", "песня", "трек", "треков", "пожалуйста", "the", "a", "an", "who", "is",
    "what", "about", "tell", "me", "known", "songs", "song", "music", "information",
}


class _DuckDuckGoHTMLParser(HTMLParser):
    """Collect result links and snippets from DuckDuckGo's non-JavaScript HTML page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._active_field: str | None = None
        self._active_depth = 0
        self._text: list[str] = []

    @staticmethod
    def _has_class(attrs: list[tuple[str, str | None]], class_name: str) -> bool:
        classes = next((value or "" for key, value in attrs if key == "class"), "")
        return class_name in classes.split()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "a" and self._has_class(attrs, "result__a"):
            self.results.append({"title": "", "url": attrs_dict.get("href") or "", "snippet": ""})
            self._start_field("title")
        elif self.results and self._has_class(attrs, "result__snippet"):
            self._start_field("snippet")
        elif self._active_field is not None and tag not in {"br", "img", "meta", "link", "input"}:
            self._active_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._active_field is None:
            return
        if tag == "a" and self._active_field == "title":
            self._finish_field()
        elif tag in {"a", "div", "span"} and self._active_field == "snippet" and self._active_depth == 0:
            self._finish_field()
        elif self._active_depth > 0:
            self._active_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._active_field is not None:
            self._text.append(data)

    def _start_field(self, field: str) -> None:
        self._active_field = field
        self._active_depth = 0
        self._text = []

    def _finish_field(self) -> None:
        value = re.sub(r"\s+", " ", " ".join(self._text)).strip()[:_MAX_FIELD_LENGTH]
        if self.results and self._active_field:
            self.results[-1][self._active_field] = value
        self._active_field = None
        self._active_depth = 0
        self._text = []


def _unwrap_result_url(raw_url: str) -> str | None:
    """Extract a destination URL from DuckDuckGo's redirect link and reject non-web URLs."""
    parsed = urlsplit(raw_url)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
        destination = parse_qs(parsed.query).get("uddg", [None])[0]
        if destination:
            raw_url = destination
            parsed = urlsplit(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return raw_url


def _tokens(text: str) -> set[str]:
    """Return normalized query/content words, excluding common request filler."""
    return {
        token
        for token in re.findall(r"[\w]+", text.casefold())
        if len(token) > 1 and token not in _QUERY_STOP_WORDS
    }


def _rank_results(query: str, results: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Move results containing the user's exact terms above loosely related results."""
    query_tokens = _tokens(query)
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for position, item in enumerate(results):
        title_tokens = _tokens(item["title"])
        snippet_tokens = _tokens(item["snippet"])
        title_matches = sorted(query_tokens & title_tokens)
        snippet_matches = sorted(query_tokens & snippet_tokens)
        score = 5 * len(title_matches) + 2 * len(snippet_matches) - position * 0.05
        ranked.append((score, position, {**item, "matched_query_terms": title_matches + snippet_matches}))
    ranked.sort(key=lambda entry: (-entry[0], entry[1]))
    return [item for _score, _position, item in ranked]


def web_search(query: str) -> dict[str, Any]:
    """Search the public web and return relevant titles, snippets, and source URLs."""
    query = query.strip()
    if len(query) < 2:
        raise ValueError("Search query must contain at least 2 characters.")
    if len(query) > 200:
        raise ValueError("Search query cannot exceed 200 characters.")

    url = f"{_SEARCH_URL}?{urlencode({'q': query, 'kl': 'ru-ru', 's': '0'})}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "ru,en;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            html_bytes = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(f"Web search returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Web search failed: {exc}") from exc

    if len(html_bytes) > _MAX_RESPONSE_BYTES:
        raise RuntimeError("Web search response exceeded the 1 MB safety limit.")

    parser = _DuckDuckGoHTMLParser()
    parser.feed(html_bytes.decode("utf-8", errors="replace"))

    results: list[dict[str, str]] = []
    for item in parser.results:
        result_url = _unwrap_result_url(item["url"])
        if not item["title"] or not result_url:
            continue
        results.append({
            "title": item["title"],
            "url": result_url,
            "snippet": item["snippet"],
        })
        if len(results) >= _FETCH_RESULTS:
            break

    if not results:
        if b"anomaly-modal" in html_bytes or b"Unfortunately, bots use" in html_bytes:
            raise RuntimeError("DuckDuckGo blocked this automated search; try again later.")
        raise RuntimeError("Web search returned no parseable results.")

    return {
        "query": query,
        "provider": "DuckDuckGo",
        "results": _rank_results(query, results)[:_MAX_RESULTS],
    }
