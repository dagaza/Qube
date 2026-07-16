"""Extractor plugin registry and selection."""

from __future__ import annotations

from core.knowledge.document.types import Document, DocumentMetadata
from core.knowledge.extractors.base import Extractor, ExtractorMetadata

_EXTRACTORS: list[Extractor] = []
_FALLBACK_CONFIDENCE = 0.3
_MIN_SPECIALIZED_CONFIDENCE = 0.5


def register_extractor(extractor: Extractor) -> None:
    _EXTRACTORS.append(extractor)
    _EXTRACTORS.sort(key=lambda ext: ext.metadata.priority, reverse=True)


def registered_extractors() -> tuple[Extractor, ...]:
    return tuple(_EXTRACTORS)


def get_fallback_extractor() -> Extractor:
    from core.knowledge.extractors.trafilatura_extractor import TrafilaturaExtractor

    return TrafilaturaExtractor()


def select_best_extractor(
    url: str,
    html: str,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[Extractor, float]:
    """Pick extractor by supports() confidence and priority — no routing table."""
    scores = [
        (ext, ext.supports(url, html, headers=headers))
        for ext in registered_extractors()
    ]
    scores = [(ext, confidence) for ext, confidence in scores if confidence > 0]
    fallback = get_fallback_extractor()
    if not scores:
        return fallback, _FALLBACK_CONFIDENCE

    best_extractor, best_confidence = max(
        scores,
        key=lambda item: (item[1], item[0].metadata.priority),
    )
    if best_confidence < _MIN_SPECIALIZED_CONFIDENCE:
        return fallback, _FALLBACK_CONFIDENCE
    return best_extractor, best_confidence


def extract_document(
    html: str,
    url: str,
    *,
    fetch_tier: str = "http",
    headers: dict[str, str] | None = None,
) -> Document:
    """Select best extractor and produce a canonical Document."""
    extractor, confidence = select_best_extractor(url, html, headers=headers)
    document = extractor.extract(html, url, fetch_tier=fetch_tier)
    if document.metadata is None:
        document.metadata = DocumentMetadata(
            extractor_name=extractor.metadata.name,
            extractor_version=extractor.metadata.version,
            extractor_confidence=confidence,
            fetch_tier=fetch_tier,
        )
    else:
        document.metadata = DocumentMetadata(
            extractor_name=document.metadata.extractor_name or extractor.metadata.name,
            extractor_version=document.metadata.extractor_version or extractor.metadata.version,
            extractor_confidence=confidence,
            fetch_tier=document.metadata.fetch_tier or fetch_tier,
            page_count=document.metadata.page_count,
            language=document.metadata.language,
        )
    return document


def _register_builtin_extractors() -> None:
    from core.knowledge.extractors.recipe_extractor import RecipeExtractor
    from core.knowledge.extractors.trafilatura_extractor import TrafilaturaExtractor

    register_extractor(RecipeExtractor())
    register_extractor(TrafilaturaExtractor())


_register_builtin_extractors()
