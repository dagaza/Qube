"""Trafilatura-based general article extractor — v1 fallback."""

from __future__ import annotations

import html as html_module
import re

from core.knowledge.document.types import Document, DocumentMetadata, DocumentSection
from core.knowledge.extractors.base import ExtractorMetadata

try:
    import trafilatura
    from trafilatura import bare_extraction
except ImportError:  # pragma: no cover - exercised when dependency missing
    trafilatura = None
    bare_extraction = None

EXTRACTOR_NAME = "TrafilaturaExtractor"
EXTRACTOR_VERSION = "1.0.0"
FALLBACK_CONFIDENCE = 0.3

_HEADING_BLOCK_RE = re.compile(
    r"<h([1-6])[^>]*>(.*?)</h\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(fragment: str) -> str:
    text = _TAG_RE.sub(" ", fragment or "")
    return html_module.unescape(re.sub(r"\s+", " ", text)).strip()


def _sections_from_html(html: str, *, fallback_text: str) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    offset = 0
    matches = list(_HEADING_BLOCK_RE.finditer(html or ""))
    if not matches:
        body = (fallback_text or "").strip()
        if body:
            sections.append(DocumentSection(heading=None, level=0, text=body, char_offset=0))
        return sections

    for index, match in enumerate(matches):
        level = int(match.group(1))
        heading = _strip_tags(match.group(2))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(html)
        block_html = (html or "")[start:end]
        text = _strip_tags(block_html)
        if not text and not heading:
            continue
        sections.append(
            DocumentSection(
                heading=heading or None,
                level=level,
                text=text,
                char_offset=offset,
            )
        )
        offset += len(text)

    if not sections and fallback_text.strip():
        sections.append(
            DocumentSection(
                heading=None,
                level=0,
                text=fallback_text.strip(),
                char_offset=0,
            )
        )
    return sections


def _bare_extraction_fields(html: str, url: str) -> dict[str, str]:
    if bare_extraction is None:
        raise RuntimeError(
            "trafilatura is not installed — add it to requirements.txt"
        )
    extracted = bare_extraction(html, url=url)
    if extracted is None:
        return {}
    if hasattr(extracted, "as_dict"):
        raw = extracted.as_dict()
        return {
            "title": str(raw.get("title") or "").strip(),
            "author": str(raw.get("author") or "").strip(),
            "date": str(raw.get("date") or "").strip(),
            "language": str(raw.get("language") or "").strip(),
            "text": str(raw.get("text") or raw.get("raw_text") or "").strip(),
        }
    if isinstance(extracted, dict):
        return {
            "title": str(extracted.get("title") or "").strip(),
            "author": str(extracted.get("author") or "").strip(),
            "date": str(extracted.get("date") or "").strip(),
            "language": str(extracted.get("language") or "").strip(),
            "text": str(extracted.get("text") or "").strip(),
        }
    return {
        "title": str(getattr(extracted, "title", "") or "").strip(),
        "author": str(getattr(extracted, "author", "") or "").strip(),
        "date": str(getattr(extracted, "date", "") or "").strip(),
        "language": str(getattr(extracted, "language", "") or "").strip(),
        "text": str(getattr(extracted, "text", "") or getattr(extracted, "raw_text", "") or "").strip(),
    }


class TrafilaturaExtractor:
    metadata = ExtractorMetadata(
        name=EXTRACTOR_NAME,
        version=EXTRACTOR_VERSION,
        priority=10,
    )

    def supports(
        self,
        url: str,
        html: str,
        *,
        headers=None,
    ) -> float:
        _ = (url, html, headers)
        return FALLBACK_CONFIDENCE

    def extract(
        self,
        html: str,
        url: str,
        *,
        fetch_tier: str = "http",
    ) -> Document:
        fields = _bare_extraction_fields(html, url)
        title = fields.get("title") or None
        author = fields.get("author") or None
        date = fields.get("date") or None
        language = fields.get("language") or None
        main_text = fields.get("text") or ""

        if not title:
            heading_match = _HEADING_BLOCK_RE.search(html or "")
            if heading_match:
                title = _strip_tags(heading_match.group(2)) or None
            elif main_text:
                title = main_text.split("\n", 1)[0].strip() or None

        sections = _sections_from_html(html, fallback_text=main_text)
        if title and sections and sections[0].heading is None and not sections[0].text:
            sections[0].heading = title
            sections[0].level = 1

        return Document(
            url=url,
            title=title,
            author=author,
            date=date,
            sections=sections,
            metadata=DocumentMetadata(
                extractor_name=EXTRACTOR_NAME,
                extractor_version=EXTRACTOR_VERSION,
                extractor_confidence=FALLBACK_CONFIDENCE,
                fetch_tier=fetch_tier,
                language=language,
            ),
        )
