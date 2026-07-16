"""Capability-based HTML extractors."""

from core.knowledge.extractors.base import Extractor, ExtractorMetadata
from core.knowledge.extractors.registry import (
    extract_document,
    get_fallback_extractor,
    register_extractor,
    registered_extractors,
    select_best_extractor,
)

__all__ = [
    "Extractor",
    "ExtractorMetadata",
    "extract_document",
    "get_fallback_extractor",
    "register_extractor",
    "registered_extractors",
    "select_best_extractor",
]
