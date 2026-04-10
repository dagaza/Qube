# mcp/internet_tool.py

import requests
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("Qube.MCP.Internet")

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
        List[Dict[str, str]]: Each dict contains 'title' and 'snippet'.
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
        
        # Parse result snippets and titles
        titles = re.findall(r'<a class="result__a"[^>]*>(.*?)</a>', response.text, re.IGNORECASE | re.DOTALL)
        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', response.text, re.IGNORECASE | re.DOTALL)
        
        results = []
        for i in range(min(max_results, len(snippets))):
            title_clean = re.sub(r'<[^>]+>', '', titles[i]).strip() if i < len(titles) else ""
            snippet_clean = re.sub(r'<[^>]+>', '', snippets[i]).strip()
            results.append({"title": title_clean, "snippet": snippet_clean})
        
        if not results:
            logger.debug("No internet results found.")
            return [{"title": "", "snippet": "No relevant internet results found."}]
        
        logger.debug(f"Internet Search Success. Retrieved {len(results)} snippets.")
        return results
        
    except Exception as e:
        logger.error(f"Internet tool failed: {e}")
        return [{"title": "", "snippet": f"Internet search failed due to network error: {e}"}]