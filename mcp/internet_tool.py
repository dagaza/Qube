# mcp/internet_tool.py

import html
import logging
import re
from typing import List, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests

logger = logging.getLogger("Qube.MCP.Internet")


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

def search_internet(
    query: str,
    max_results: int = 3,
    target_site: Optional[str] = None
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
    logger.info(f"Executing Internet Search for: '{query}'" + (f" on site: {target_site}" if target_site else ""))
    
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    if target_site:
        query = f"site:{target_site} {query}"
    
    try:
        response = requests.post(url, data={"q": query}, headers=headers, timeout=5)
        response.raise_for_status()
        
        result_links = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            response.text,
            re.IGNORECASE | re.DOTALL,
        )
        snippets = re.findall(
            r'<a class="result__snippet"[^>]*>(.*?)</a>',
            response.text,
            re.IGNORECASE | re.DOTALL,
        )

        results = []
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
        
        if not results:
            logger.debug("No internet results found.")
            return [{"title": "", "snippet": "No relevant internet results found."}]
        
        logger.debug(f"Internet Search Success. Retrieved {len(results)} snippets.")
        return results
        
    except Exception as e:
        logger.error(f"Internet tool failed: {e}")
        return [{"title": "", "snippet": f"Internet search failed due to network error: {e}"}]