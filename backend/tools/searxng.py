"""
backend/tools/searxng.py
=========================
Async SearXNG client.

SearXNG is a self-hosted metasearch engine that proxies to 70+ search
engines simultaneously.  It exposes a plain JSON API with no API keys.

This client:
  - Fetches search results from SearXNG's /search endpoint
  - Fetches the full text of result URLs via httpx
  - Applies content security scanning to all web-sourced text
  - Tags results with trust_tier="web" so the Critic can weight accordingly
  - Respects the web_search_enabled config flag

Reference: https://github.com/searxng/searxng
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from backend.shared.config import ToolsSettings

logger = logging.getLogger(__name__)

_SEARCH_TIMEOUT  = 10.0   # seconds for SearXNG query
_FETCH_TIMEOUT   = 8.0    # seconds for full-page fetch
_MAX_RESULT_TEXT = 1500   # characters kept per fetched page
_USER_AGENT      = "PAEKA/0.9.0 (+local-assistant)"


@dataclass
class WebResult:
    """A single web search result with fetched content."""
    title: str
    url: str
    snippet: str          # SearXNG summary snippet
    content: str          # fetched page body (truncated)
    trust_tier: str = "web"
    engine: str = ""      # which search engine produced this result


class SearXNGClient:
    """
    Async SearXNG search client.

    Parameters
    ----------
    settings:
        ToolsSettings (searxng_url, web_search_enabled).
    scanner:
        ContentScanner instance for filtering web content.
    """

    def __init__(self, settings: ToolsSettings, scanner=None) -> None:
        self._url      = settings.searxng_url.rstrip("/")
        self._enabled  = settings.web_search_enabled
        self._scanner  = scanner
        self._http = httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            timeout=httpx.Timeout(connect=5.0, read=_FETCH_TIMEOUT, write=5.0, pool=5.0),
        )

    async def search(
        self,
        query: str,
        num_results: int = 5,
        categories: str = "general",
        language: str = "en",
    ) -> list[WebResult]:
        """
        Execute a SearXNG search and return fetched results.

        Parameters
        ----------
        query:
            Search query string.
        num_results:
            Maximum results to return (after filtering).
        categories:
            SearXNG category string (general, science, it, files, etc.)
        language:
            Language code for results.

        Returns
        -------
        list[WebResult]
            Results filtered through the content scanner.
            Returns empty list if web_search_enabled is False.
        """
        if not self._enabled:
            logger.debug("Web search disabled — skipping SearXNG query.")
            return []

        try:
            params = {
                "q":          query,
                "format":     "json",
                "categories": categories,
                "language":   language,
                "pageno":     1,
            }
            resp = await self._http.get(
                f"{self._url}/search",
                params=params,
                timeout=_SEARCH_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("SearXNG search failed for '%s': %s", query[:60], exc)
            return []

        raw_results = data.get("results", [])[:num_results * 2]  # fetch extra, filter down
        results: list[WebResult] = []

        for item in raw_results:
            url     = item.get("url", "")
            title   = item.get("title", "")
            snippet = item.get("content", "")
            engine  = item.get("engine", "")

            if not url:
                continue

            # Scan snippet first — fast check before fetching
            if self._scanner:
                scan = self._scanner.scan_web_result(snippet, url=url)
                if scan.is_blocked:
                    logger.info("Web snippet blocked by scanner: %s", url)
                    continue

            # Fetch full page content
            content = await self._fetch_page(url)

            # Scan fetched content
            if self._scanner and content:
                scan = self._scanner.scan_web_result(content, url=url)
                if scan.is_blocked:
                    logger.info("Web page content blocked by scanner: %s", url)
                    continue
                content = scan.sanitised_text

            results.append(WebResult(
                title=title,
                url=url,
                snippet=snippet,
                content=content or snippet,
                engine=engine,
            ))

            if len(results) >= num_results:
                break

        logger.info(
            "SearXNG: query='%s' → %d/%d results after filtering",
            query[:60], len(results), len(raw_results),
        )
        return results

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_page(self, url: str) -> str:
        """Fetch a URL and extract plain text from its HTML body."""
        try:
            resp = await self._http.get(url, timeout=_FETCH_TIMEOUT)
            resp.raise_for_status()
            raw = resp.text
            return _html_to_text(raw)[:_MAX_RESULT_TEXT]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to fetch %s: %s", url, exc)
            return ""


def _html_to_text(html: str) -> str:
    """
    Minimal HTML → plain text conversion.
    Strips tags, collapses whitespace, skips script/style blocks.
    No BeautifulSoup required.
    """
    import re
    # Remove script and style blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.I)
    # Remove all remaining HTML tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    for entity, char in (
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "),
    ):
        html = html.replace(entity, char)
    # Collapse whitespace
    html = re.sub(r"\s+", " ", html).strip()
    return html
