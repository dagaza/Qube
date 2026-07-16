"""Canonical Document IR between HTML and evidence chunking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentSection:
    heading: str | None
    level: int = 0
    text: str = ""
    list_items: tuple[str, ...] = ()
    char_offset: int = 0


@dataclass
class DocumentTable:
    caption: str | None
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DocumentMetadata:
    extractor_name: str
    extractor_version: str
    extractor_confidence: float
    fetch_tier: str = "http"
    page_count: int = 1
    language: str | None = None


@dataclass
class Document:
    url: str
    title: str | None
    author: str | None = None
    date: str | None = None
    sections: list[DocumentSection] = field(default_factory=list)
    tables: list[DocumentTable] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    structured_data: dict[str, Any] = field(default_factory=dict)
    metadata: DocumentMetadata | None = None

    @property
    def total_text_chars(self) -> int:
        return sum(len(section.text or "") for section in self.sections)
