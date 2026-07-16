"""Extractor protocol and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from core.knowledge.document.types import Document


@dataclass(frozen=True)
class ExtractorMetadata:
    name: str
    version: str
    priority: int = 50


class Extractor(Protocol):
    metadata: ExtractorMetadata

    def supports(
        self,
        url: str,
        html: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> float:
        """Return confidence 0.0–1.0 that this extractor can handle the page."""

    def extract(
        self,
        html: str,
        url: str,
        *,
        fetch_tier: str = "http",
    ) -> Document: ...
