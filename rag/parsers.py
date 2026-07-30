# rag/parsers.py
"""Legacy text extraction helpers — prefer ``build_document_from_path`` for ingest."""

from __future__ import annotations

from pathlib import Path


def parse_file(path: Path) -> list[str]:
    """Return section bodies from the shared Document IR (backward compatible)."""
    from core.knowledge.document.builders.library_builder import build_document_from_path

    document = build_document_from_path(path)
    return [section.text for section in document.sections if (section.text or "").strip()]
