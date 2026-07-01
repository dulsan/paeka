"""
backend/tools/websearch.py
===========================
Web search client — DuckDuckGo implementation.

This is the established web search backend per project policy: SearXNG is
disabled (no local search infra to run/maintain). DuckDuckGo is used via
their unofficial JSON API (no API key required).

For production use or higher volume, swap the _search_duckduckgo() backend
for Brave Search API (free tier: 2000 queries/month with an API key) --
see the commented stub below.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

from backend.shared.config import ToolsSettings

logger = logging.getLogger(__name__)

_SEARCH_TIMEOUT  = 10.0
_FETCH_TIMEOUT   = 8.0
_MAX_RESULT_TEXT = 1500
_USER_AGENT      = "PAEKA/0.11 (+local-assistant)"

# DuckDuckGo unofficial JSON endpoint
_DDG_URL = "https://api.duckduckgo.com/"


@dataclass
class WebResult:
    """Single web search result returned by WebSearchClient."""
    title:      str
    url:        str
    snippet:    str
    content:    str
    trust_tier: str = "web"
    engine:     str = "duckduckgo"


class WebSearchClient:
    """
    Async web search client backed by DuckDuckGo's Instant Answer API.

    Parameters
    ----------
    settings:
        ToolsSettings — reads web_search_enabled and web_search_max_results.
    scanner:
        ContentScanner instance for filtering web content.
    """

    def __init__(self, settings: ToolsSettings, scanner=None) -> None:
        self._enabled = settings.web_search_enabled
        self._scanner = scanner
        self._max     = getattr(settings, "web_search_max_results", 5)
        self._http    = httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=5.0, read=_FETCH_TIMEOUT, write=5.0, pool=5.0
            ),
        )

    async def search(
        self,
        query: str,
        num_results: int | None = None,
        categories: str = "general",   # accepted for compat, unused
        language: str = "en",
    ) -> list[WebResult]:
        """
        Execute a web search and return results with fetched content.

        Returns empty list if web_search_enabled is False.
        """
        if not self._enabled:
            logger.debug("Web search disabled — skipping query.")
            return []

        limit = num_results or self._max
        raw   = await self._search_duckduckgo(query, language)

        results: list[WebResult] = []
        for item in raw[: limit * 2]:
            url     = item.get("url", "")
            title   = item.get("title", "")
            snippet = item.get("snippet", "")

            if not url:
                continue

            # Scan snippet before fetching full page
            if self._scanner:
                scan = self._scanner.scan_web_result(snippet, url=url)
                if scan.is_blocked:
                    continue

            content = await self._fetch_page(url)

            if self._scanner and content:
                scan = self._scanner.scan_web_result(content, url=url)
                if scan.is_blocked:
                    continue
                content = scan.sanitised_text

            results.append(WebResult(
                title=title,
                url=url,
                snippet=snippet,
                content=content or snippet,
            ))

            if len(results) >= limit:
                break

        logger.info(
            "WebSearch: query='%s' → %d results", query[:60], len(results)
        )
        return results

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    async def _search_duckduckgo(
        self, query: str, language: str = "en"
    ) -> list[dict]:
        """
        DuckDuckGo Instant Answer API.

        Returns related topics as search results. This is a best-effort
        interim: for factual one-shot queries it works well. For broad
        research queries it returns fewer results than a full search engine.

        Replace this method with _search_brave() or _search_serper() for
        higher-quality results.
        """
        try:
            resp = await self._http.get(
                _DDG_URL,
                params={
                    "q":   query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                    "kl": f"{language}-{language.upper()}",
                },
                timeout=_SEARCH_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("DuckDuckGo search failed for '%s': %s", query[:60], exc)
            return []

        results = []

        # Abstract answer (the direct answer box)
        if data.get("AbstractText") and data.get("AbstractURL"):
            results.append({
                "title":   data.get("Heading", query),
                "url":     data["AbstractURL"],
                "snippet": data["AbstractText"][:400],
            })

        # Related topics
        for topic in data.get("RelatedTopics", []):
            # Topics can be nested under sub-groups
            if "Topics" in topic:
                for sub in topic["Topics"]:
                    item = _parse_ddg_topic(sub)
                    if item:
                        results.append(item)
            else:
                item = _parse_ddg_topic(topic)
                if item:
                    results.append(item)

        return results

    async def _fetch_page(self, url: str) -> str:
        """Fetch a URL and return extracted plain text (truncated)."""
        try:
            resp = await self._http.get(url, timeout=_FETCH_TIMEOUT)
            resp.raise_for_status()
            return _html_to_text(resp.text)[:_MAX_RESULT_TEXT]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to fetch %s: %s", url, exc)
            return ""


# ---------------------------------------------------------------------------
# Future backend stubs — uncomment and implement when needed
# ---------------------------------------------------------------------------

# async def _search_brave(self, query: str, api_key: str) -> list[dict]:
#     """Brave Search API — free tier 2000 req/month, no tracking."""
#     resp = await self._http.get(
#         "https://api.search.brave.com/res/v1/web/search",
#         params={"q": query, "count": 10},
#         headers={"Accept": "application/json",
#                  "X-Subscription-Token": api_key},
#     )
#     resp.raise_for_status()
#     return [
#         {"title": r["title"], "url": r["url"], "snippet": r.get("description","")}
#         for r in resp.json().get("web", {}).get("results", [])
#     ]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ddg_topic(topic: dict) -> dict | None:
    url  = topic.get("FirstURL", "")
    text = topic.get("Text", "")
    if not url or not text:
        return None
    return {"title": text[:80], "url": url, "snippet": text[:300]}


def _html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                  flags=re.DOTALL | re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    for entity, char in (
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "),
    ):
        html = html.replace(entity, char)
    return re.sub(r"\s+", " ", html).strip()
