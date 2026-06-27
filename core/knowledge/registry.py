"""Knowledge service registry."""

from __future__ import annotations

from typing import Any, Protocol

from core.knowledge.services.general_web import GeneralWebKnowledgeService
from core.knowledge.services.scientific_evidence import ScientificEvidenceService
from core.knowledge.services.trusted_knowledge import TrustedKnowledgeService
from core.knowledge.services.wikipedia import WikipediaKnowledgeService
from core.knowledge.types import (
    SERVICE_GENERAL_WEB,
    SERVICE_SCIENTIFIC_EVIDENCE,
    SERVICE_TRUSTED_KNOWLEDGE,
    SERVICE_WIKIPEDIA,
)

WEB_COMPOSER_TOOLS = frozenset(
    {"internet", "trusted", "evidence", "wikipedia", "pubmed", "arxiv"}
)


class KnowledgeService(Protocol):
    id: str

    def default_budget(self): ...

    def retrieve(self, ctx): ...


_SERVICES: dict[str, Any] = {
    SERVICE_GENERAL_WEB: GeneralWebKnowledgeService(),
    SERVICE_TRUSTED_KNOWLEDGE: TrustedKnowledgeService(),
    SERVICE_SCIENTIFIC_EVIDENCE: ScientificEvidenceService(),
    SERVICE_WIKIPEDIA: WikipediaKnowledgeService(),
}


def get_knowledge_service(service_id: str) -> KnowledgeService:
    sid = (service_id or SERVICE_GENERAL_WEB).strip().lower()
    service = _SERVICES.get(sid)
    if service is None:
        return _SERVICES[SERVICE_GENERAL_WEB]
    return service


def resolve_turn_knowledge_service(
    *,
    composer_tool: str | None = None,
    composer_trusted: bool = False,
    composer_internet: bool = False,
    default_service: str | None = None,
) -> str:
    """Pick the knowledge service for a web turn."""
    tool = (composer_tool or "").strip().lower()
    if not tool and composer_trusted:
        tool = "trusted"
    if not tool and composer_internet:
        tool = "internet"

    if tool == "evidence":
        return SERVICE_SCIENTIFIC_EVIDENCE
    if tool == "trusted":
        return SERVICE_TRUSTED_KNOWLEDGE
    if tool == "internet":
        return SERVICE_GENERAL_WEB
    if tool == "wikipedia":
        return SERVICE_WIKIPEDIA
    if tool in {"pubmed", "arxiv"}:
        return SERVICE_SCIENTIFIC_EVIDENCE

    fallback = (default_service or SERVICE_GENERAL_WEB).strip().lower()
    if fallback in _SERVICES:
        return fallback
    return SERVICE_GENERAL_WEB


def adapter_filter_for_composer_tool(composer_tool: str | None) -> tuple[str, ...] | None:
    tool = (composer_tool or "").strip().lower()
    if tool == "pubmed":
        return ("pubmed",)
    if tool == "arxiv":
        return ("arxiv",)
    return None
