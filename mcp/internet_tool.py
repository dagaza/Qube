# mcp/internet_tool.py

import html
import logging
import re
from typing import List, Dict, Optional, TypedDict
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import requests

logger = logging.getLogger("Qube.MCP.Internet")

DDG_NO_RESULTS_SNIPPET = "No relevant internet results found."
DDG_BOT_CHALLENGE_SNIPPET = (
    "Internet search blocked: DuckDuckGo bot challenge (try again later)."
)
DDG_PACING_TIMEOUT_SNIPPET = (
    "Internet search deferred: discovery pacing timeout (using fallbacks)."
)
DDG_NETWORK_ERROR_PREFIX = "Internet search failed due to network error:"


class DdgInspectResult(TypedDict):
    response_kind: str
    http_status: int | None
    body_len: int
    link_matches: int
    snippet_matches: int
    parsed_rows: int
    urls_with_http: int
    bot_challenge_signals: list[str]
    pace_wait_ms: int


@dataclass(frozen=True)
class InternetSearchResponse:
    rows: list[dict[str, str]]
    inspection: DdgInspectResult | None


def _strip_html_tags(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment or "")).strip()


def _decode_ddg_target_href(href: str) -> str:
    """Resolve DuckDuckGo redirect links to the destination URL when possible."""
    h = (href or "").strip()
    if not h:
        return ""
    if h.startswith("//"):
        h = "https:" + h
    if "uddg=" in h:
        try:
            raw = parse_qs(urlparse(h).query).get("uddg", [""])[0]
            if raw:
                return unquote(raw)
        except Exception:
            pass
    return h

def _count_ddg_serp_markers(html_text: str) -> tuple[int, int]:
    text = html_text or ""
    result_links = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    snippets = re.findall(
        r'<a class="result__snippet"[^>]*>(.*?)</a>',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return len(result_links), len(snippets)


_DDG_BOT_CHALLENGE_MARKERS: tuple[str, ...] = (
    "anomaly-modal",
    "anomaly-modal__image",
    'data-testid="anomaly-modal',
    "verify you're human",
    "verify you are human",
    "confirm you're human",
    "confirm you are human",
)

_DDG_BOT_CHALLENGE_KEYWORDS: tuple[str, ...] = (
    "captcha",
    "unusual traffic",
    "automated",
    "bot detection",
    "are you a robot",
)


def score_ddg_bot_challenge(
    html_text: str,
    *,
    http_status: int | None = None,
    link_matches: int = 0,
    snippet_matches: int = 0,
) -> tuple[bool, list[str]]:
    """Return (is_bot_challenge, matched_signals) using multiple weak signals."""
    body = html_text or ""
    text = body.lower()
    signals: list[str] = []

    if link_matches == 0 and snippet_matches == 0:
        signals.append("no_serp_markers")
    if http_status == 202:
        signals.append("http_202")
    for marker in _DDG_BOT_CHALLENGE_MARKERS:
        if marker in text:
            signals.append(f"marker:{marker}")
    for keyword in _DDG_BOT_CHALLENGE_KEYWORDS:
        if keyword in text:
            signals.append(f"kw:{keyword}")
    if "verify" in text and "human" in text:
        signals.append("kw:verify+human")

    keyword_hits = sum(1 for signal in signals if signal.startswith("kw:"))
    marker_hits = sum(1 for signal in signals if signal.startswith("marker:"))
    is_bot = "no_serp_markers" in signals and (
        marker_hits > 0
        or ("http_202" in signals and keyword_hits >= 1)
        or keyword_hits >= 2
    )
    return is_bot, signals


def inspect_ddg_html_response(
    html_text: str,
    *,
    http_status: int | None = None,
    max_results: int = 3,
) -> DdgInspectResult:
    """Classify a DuckDuckGo HTML body for operational diagnostics."""
    body = html_text or ""
    link_matches, snippet_matches = _count_ddg_serp_markers(body)
    parsed = parse_ddg_html_results(body, max_results=max_results)
    urls_with_http = sum(
        1
        for row in parsed
        if str(row.get("url") or "").startswith(("http://", "https://"))
    )
    bot_challenge, bot_signals = score_ddg_bot_challenge(
        body,
        http_status=http_status,
        link_matches=link_matches,
        snippet_matches=snippet_matches,
    )
    if parsed:
        response_kind = "serp"
    elif bot_challenge:
        response_kind = "bot_challenge"
    else:
        response_kind = "empty_parse"
    return {
        "response_kind": response_kind,
        "http_status": http_status,
        "body_len": len(body),
        "link_matches": link_matches,
        "snippet_matches": snippet_matches,
        "parsed_rows": len(parsed),
        "urls_with_http": urls_with_http,
        "bot_challenge_signals": bot_signals,
        "pace_wait_ms": 0,
    }


def _log_ddg_inspection(query: str, inspection: DdgInspectResult) -> None:
    signals = ",".join(inspection.get("bot_challenge_signals") or []) or "none"
    pace_wait_ms = int(inspection.get("pace_wait_ms") or 0)
    message = (
        "[DDG] query=%r http_status=%s response_kind=%s body_len=%d "
        "link_matches=%d snippet_matches=%d parsed_rows=%d urls_with_http=%d "
        "pace_wait_ms=%d bot_signals=%s"
    )
    args = (
        query,
        inspection["http_status"],
        inspection["response_kind"],
        inspection["body_len"],
        inspection["link_matches"],
        inspection["snippet_matches"],
        inspection["parsed_rows"],
        inspection["urls_with_http"],
        pace_wait_ms,
        signals,
    )
    if inspection["response_kind"] == "serp":
        logger.info(message, *args)
    else:
        logger.warning(message, *args)
        if inspection["response_kind"] == "bot_challenge":
            logger.warning(
                "[DDG] Bot challenge detected. Recovery: retry later or change network."
            )


def parse_ddg_html_results(
    html_text: str,
    *,
    max_results: int = 3,
) -> List[Dict[str, str]]:
    """Parse DuckDuckGo HTML search results into structured snippets."""
    result_links = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html_text or "",
        re.IGNORECASE | re.DOTALL,
    )
    snippets = re.findall(
        r'<a class="result__snippet"[^>]*>(.*?)</a>',
        html_text or "",
        re.IGNORECASE | re.DOTALL,
    )

    results: List[Dict[str, str]] = []
    for i in range(min(max_results, len(snippets))):
        title_clean = ""
        url = ""
        if i < len(result_links):
            href, title_html = result_links[i]
            title_clean = _strip_html_tags(title_html)
            url = _decode_ddg_target_href(href)
        snippet_clean = _strip_html_tags(snippets[i])
        row: Dict[str, str] = {"title": title_clean, "snippet": snippet_clean}
        if url.startswith(("http://", "https://")):
            row["url"] = url
        results.append(row)
    return results


def execute_internet_search(
    query: str,
    max_results: int = 3,
    target_site: Optional[str] = None,
) -> InternetSearchResponse:
    """Perform DDG search and return rows plus inspection metadata."""
    from core.knowledge.discovery.pacing import wait_for_ddg_pace_slot
    from core.knowledge.discovery.session_budget import record_ddg_live_request

    logger.info(
        "Executing Internet Search for: '%s'%s",
        query,
        f" on site: {target_site}" if target_site else "",
    )

    acquired, pace_wait_ms = wait_for_ddg_pace_slot()
    if not acquired:
        inspection: DdgInspectResult = {
            "response_kind": "pacing_timeout",
            "http_status": None,
            "body_len": 0,
            "link_matches": 0,
            "snippet_matches": 0,
            "parsed_rows": 0,
            "urls_with_http": 0,
            "bot_challenge_signals": ["pacing_timeout"],
            "pace_wait_ms": pace_wait_ms,
        }
        _log_ddg_inspection(query, inspection)
        return InternetSearchResponse(
            rows=[{"title": "", "snippet": DDG_PACING_TIMEOUT_SNIPPET}],
            inspection=inspection,
        )

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://html.duckduckgo.com",
        "Referer": "https://html.duckduckgo.com/",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    }

    scoped_query = query
    if target_site:
        scoped_query = f"site:{target_site} {query}"

    try:
        response = requests.post(
            url,
            data={"q": scoped_query},
            headers=headers,
            timeout=5,
        )
        record_ddg_live_request()
        response.raise_for_status()

        inspection = inspect_ddg_html_response(
            response.text,
            http_status=response.status_code,
            max_results=max_results,
        )
        inspection["pace_wait_ms"] = pace_wait_ms
        _log_ddg_inspection(scoped_query, inspection)
        results = parse_ddg_html_results(response.text, max_results=max_results)

        if not results:
            if inspection["response_kind"] == "bot_challenge":
                logger.warning(
                    "[DDG] DuckDuckGo returned a bot/anomaly challenge instead of SERP HTML."
                )
                rows = [{"title": "", "snippet": DDG_BOT_CHALLENGE_SNIPPET}]
            else:
                logger.warning(
                    "[DDG] DuckDuckGo HTML parsed to zero SERP rows (empty_parse)."
                )
                rows = [{"title": "", "snippet": DDG_NO_RESULTS_SNIPPET}]
            return InternetSearchResponse(rows=rows, inspection=inspection)

        logger.debug(
            "Internet Search Success. Retrieved %d snippets (%d with URLs).",
            len(results),
            inspection["urls_with_http"],
        )
        return InternetSearchResponse(rows=results, inspection=inspection)

    except Exception as e:
        logger.error("Internet tool failed: %s", e)
        return InternetSearchResponse(
            rows=[
                {
                    "title": "",
                    "snippet": f"{DDG_NETWORK_ERROR_PREFIX} {e}",
                }
            ],
            inspection=None,
        )


def search_internet(
    query: str,
    max_results: int = 3,
    target_site: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Performs a lightweight web search using DuckDuckGo HTML and returns a list of structured snippets.

    Args:
        query (str): User query string.
        max_results (int): Maximum number of search results to return.
        target_site (Optional[str]): Optional domain restriction (e.g., 'wikipedia.org').

    Returns:
        List[Dict[str, str]]: Each dict contains 'title', 'snippet', and optional 'url'.
    """
    return execute_internet_search(
        query,
        max_results=max_results,
        target_site=target_site,
    ).rows