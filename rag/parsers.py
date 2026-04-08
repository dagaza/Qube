# rag/parsers.py
from pathlib import Path
from pypdf import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import markdown
import fitz  # This is PyMuPDF!
import re    # 🔑 Added regex for text cleaning

def parse_pdf(path: Path) -> list[str]:
    """
    Upgraded PDF parser using PyMuPDF. 
    Significantly better at handling complex layouts, tables, and weird fonts.
    Now includes source-level NLP cleaning to prevent hard-line breaks AND Tofu characters!
    """
    texts = []
    try:
        # Open the document safely
        with fitz.open(str(path)) as doc:
            for page in doc:
                # Extract text aggressively
                text = page.get_text().strip()
                if text:
                    # 1. Clean the hard line breaks (Our previous fix)
                    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
                    
                    # --- THE FIX: TOFU SCRUBBER ---
                    # 2. Strip Unicode Replacement Character (\ufffd) and Private Use Area (\ue000-\uf8ff)
                    text = re.sub(r'[\ufffd\ue000-\uf8ff]', '', text)
                    
                    # 3. Strip rogue unprintable ASCII control characters (keeps newlines/tabs)
                    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
                    
                    # 4. Clean up any double spaces created by the merges/deletions
                    text = re.sub(r' +', ' ', text)
                    
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