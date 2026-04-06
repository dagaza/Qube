import requests
import re
import logging

logger = logging.getLogger("Qube.MCP.Internet")

def search_internet(query: str, max_results: int = 3) -> str:
    """Performs a lightweight, dependency-free web search using DuckDuckGo HTML."""
    logger.info(f"Executing Internet Search for: '{query}'")
    
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.post(url, data={"q": query}, headers=headers, timeout=5)
        response.raise_for_status()
        
        # Extremely lightweight regex parsing to extract snippets without needing BeautifulSoup
        snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', response.text, re.IGNORECASE | re.DOTALL)
        
        if not snippets:
            return "No relevant internet results found."

        # Clean HTML tags out of the snippets
        clean_snippets = []
        for snip in snippets[:max_results]:
            clean_text = re.sub(r'<[^>]+>', '', snip).strip()
            clean_snippets.append(clean_text)
            
        result_text = "\n\n".join([f"Result {i+1}: {txt}" for i, txt in enumerate(clean_snippets)])
        logger.debug(f"Internet Search Success. Retrieved {len(clean_snippets)} snippets.")
        return result_text
        
    except Exception as e:
        logger.error(f"Internet tool failed: {e}")
        return f"Internet search failed due to network error: {e}"