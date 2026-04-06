# rag/parsers.py
from pathlib import Path
from pypdf import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import markdown
import fitz  # This is PyMuPDF!

def parse_pdf(path: Path) -> list[str]:
    """
    Upgraded PDF parser using PyMuPDF. 
    Significantly better at handling complex layouts, tables, and weird fonts.
    """
    texts = []
    try:
        # Open the document safely
        with fitz.open(str(path)) as doc:
            for page in doc:
                # Extract text aggressively
                text = page.get_text().strip()
                if text:
                    texts.append(text)
    except Exception as e:
        import logging
        logger = logging.getLogger("Qube.RAG")
        logger.error(f"PyMuPDF failed to read {path.name}: {e}")
        
    return texts

def parse_epub(path: Path) -> list[str]:
    book = epub.read_epub(str(path))
    texts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n").strip()
        if text:
            texts.append(text)
    return texts

def parse_text(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix in (".md", ".markdown"):
        raw = BeautifulSoup(markdown.markdown(raw), "html.parser").get_text()
    return [raw]

def parse_wikipedia_dump(path: Path) -> list[str]:
    # Expects the pre-extracted plain text dump (not raw XML)
    texts = []
    current = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("</doc>"):
                if current:
                    texts.append("\n".join(current))
                    current = []
            elif not line.startswith("<doc") and line.strip():
                current.append(line.strip())
    return texts

def parse_file(path: Path) -> list[str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    elif ext == ".epub":
        return parse_epub(path)
    elif ext in (".txt", ".md", ".markdown"):
        return parse_text(path)
    elif ext in (".xml", ".bz2"):
        return parse_wikipedia_dump(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")