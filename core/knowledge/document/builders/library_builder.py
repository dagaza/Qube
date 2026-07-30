"""Build canonical Document IR from Library file paths."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub

from core.knowledge.document.types import Document, DocumentMetadata, DocumentSection
from core.knowledge.document.builders.plain_text_sections import split_plain_text_sections
from core.knowledge.document.pdf_text_normalize import normalize_pdf_extracted_text
from core.chunking.markdown_sections import split_markdown_sections_all_levels

logger = logging.getLogger("Qube.RAG")

_LIBRARY_EXTRACTOR = DocumentMetadata(
    extractor_name="LibraryDocumentBuilder",
    extractor_version="1.0.0",
    extractor_confidence=1.0,
    fetch_tier="library",
)

_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _clean_pdf_page_text(text: str) -> str:
    return normalize_pdf_extracted_text(text)


def _sections_from_markdown(text: str) -> list[DocumentSection]:
    parts = split_markdown_sections_all_levels(text)
    sections: list[DocumentSection] = []
    offset = 0
    for part in parts:
        heading: str | None = None
        level = 0
        body_lines: list[str] = []
        for line in part.splitlines():
            match = _HEADING_LINE_RE.match(line.strip())
            if match and not body_lines:
                level = len(match.group(1))
                heading = match.group(2).strip()
                continue
            body_lines.append(line)
        body = "\n".join(body_lines).strip()
        if not body and not heading:
            continue
        sections.append(
            DocumentSection(
                heading=heading,
                level=level,
                text=part,
                char_offset=offset,
            )
        )
        offset += len(part)
    return sections


def _infer_epub_heading(soup: BeautifulSoup) -> tuple[str | None, int]:
    for tag_name in ("h1", "h2", "h3", "title"):
        tag = soup.find(tag_name)
        if tag:
            text = tag.get_text(strip=True)
            if text:
                level = 1 if tag_name == "h1" else 2 if tag_name == "h2" else 3
                return text, level
    return None, 0


def _build_pdf_document(path: Path) -> Document:
    import fitz

    page_texts: list[str] = []
    page_spans: list[dict[str, int]] = []
    char_offset = 0
    try:
        with fitz.open(str(path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = _clean_pdf_page_text(page.get_text().strip())
                if not text:
                    continue
                if page_texts:
                    char_offset += 2
                start = char_offset
                page_texts.append(text)
                char_offset += len(text)
                page_spans.append(
                    {
                        "page": page_num,
                        "char_start": start,
                        "char_end": char_offset,
                    }
                )
    except Exception as exc:
        logger.error("PyMuPDF failed to read %s: %s", path.name, exc)

    full_text = "\n\n".join(page_texts)
    sections = [DocumentSection(heading=None, level=0, text=full_text)] if full_text else []
    return Document(
        url=str(path),
        title=path.stem,
        sections=sections,
        structured_data={"page_spans": page_spans},
        metadata=_LIBRARY_EXTRACTOR,
    )


def _build_epub_document(path: Path) -> Document:
    book = epub.read_epub(str(path))
    sections: list[DocumentSection] = []
    offset = 0
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        heading, level = _infer_epub_heading(soup)
        text = soup.get_text(separator="\n").strip()
        if not text:
            continue
        sections.append(
            DocumentSection(
                heading=heading,
                level=level,
                text=text,
                char_offset=offset,
            )
        )
        offset += len(text)
    return Document(
        url=str(path),
        title=path.stem,
        sections=sections,
        metadata=_LIBRARY_EXTRACTOR,
    )


def _build_plain_document(path: Path, *, raw_text: str | None = None) -> Document:
    text = raw_text if raw_text is not None else path.read_text(encoding="utf-8", errors="ignore")
    text = (text or "").strip()
    split = split_plain_text_sections(text)
    sections = [
        DocumentSection(
            heading=heading,
            level=level,
            text=body,
            char_offset=idx,
        )
        for idx, (heading, level, body) in enumerate(split)
        if body.strip()
    ]
    return Document(
        url=str(path),
        title=path.stem,
        sections=sections,
        metadata=_LIBRARY_EXTRACTOR,
    )


def _build_markdown_document(path: Path, *, raw_text: str | None = None) -> Document:
    text = raw_text if raw_text is not None else path.read_text(encoding="utf-8", errors="ignore")
    text = (text or "").strip()
    sections = _sections_from_markdown(text)
    if not sections and text:
        sections = [DocumentSection(heading=None, level=0, text=text)]
    return Document(
        url=str(path),
        title=path.stem,
        sections=sections,
        metadata=_LIBRARY_EXTRACTOR,
    )


def _build_wikipedia_dump_document(path: Path) -> Document:
    sections: list[DocumentSection] = []
    current: list[str] = []
    offset = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("</doc>"):
                if current:
                    body = "\n".join(current).strip()
                    if body:
                        sections.append(
                            DocumentSection(
                                heading=None,
                                level=0,
                                text=body,
                                char_offset=offset,
                            )
                        )
                        offset += len(body)
                    current = []
            elif not line.startswith("<doc") and line.strip():
                current.append(line.strip())
    return Document(
        url=str(path),
        title=path.stem,
        sections=sections,
        metadata=_LIBRARY_EXTRACTOR,
    )


def build_document_from_path(path: Path) -> Document:
    """Parse a Library file into the shared Document IR."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _build_pdf_document(path)
    if ext == ".epub":
        return _build_epub_document(path)
    if ext in (".md", ".markdown"):
        return _build_markdown_document(path)
    if ext == ".txt":
        return _build_plain_document(path)
    if ext in (".xml", ".bz2"):
        return _build_wikipedia_dump_document(path)
    raise ValueError(f"Unsupported file type: {ext}")


def build_document_from_markdown(
    text: str,
    *,
    title: str | None = None,
    source: str = "help.md",
) -> Document:
    """Build a Document from raw markdown (help corpus and tests)."""
    sections = _sections_from_markdown(text.strip())
    if not sections and text.strip():
        sections = [DocumentSection(heading=None, level=0, text=text.strip())]
    return Document(
        url=source,
        title=title or source,
        sections=sections,
        metadata=_LIBRARY_EXTRACTOR,
    )
