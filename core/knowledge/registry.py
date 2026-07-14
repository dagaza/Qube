"""Knowledge service registry."""

from __future__ import annotations

from typing import Any, Protocol

from core.knowledge.presets import load_preset, parse_source_pin_tool, parse_user_preset_tool
from core.knowledge.services.finance_knowledge import FinanceKnowledgeService
from core.knowledge.services.legal_knowledge import LegalKnowledgeService
from core.knowledge.services.general_web import GeneralWebKnowledgeService
from core.knowledge.services.internal_corpus import InternalCorpusKnowledgeService
from core.knowledge.services.preset_knowledge import PresetKnowledgeService
from core.knowledge.services.scientific_evidence import ScientificEvidenceService
from core.knowledge.services.trusted_knowledge import TrustedKnowledgeService
from core.knowledge.services.wikipedia import WikipediaKnowledgeService
from core.knowledge.types import (
    SERVICE_GENERAL_WEB,
    SERVICE_FINANCE_KNOWLEDGE,
    SERVICE_INTERNAL_CORPUS,
    SERVICE_LEGAL_KNOWLEDGE,
    SERVICE_PRESET_KNOWLEDGE,
    SERVICE_SCIENTIFIC_EVIDENCE,
    SERVICE_TRUSTED_KNOWLEDGE,
    SERVICE_WIKIPEDIA,
)

WEB_COMPOSER_TOOLS = frozenset(
    {
        "internet",
        "trusted",
        "evidence",
        "science",
        "wikipedia",
        "pubmed",
        "arxiv",
        "finance",
        "legal",
    }
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
    SERVICE_INTERNAL_CORPUS: InternalCorpusKnowledgeService(),
    SERVICE_FINANCE_KNOWLEDGE: FinanceKnowledgeService(),
    SERVICE_LEGAL_KNOWLEDGE: LegalKnowledgeService(),
    SERVICE_PRESET_KNOWLEDGE: PresetKnowledgeService(),
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

    preset_id = parse_user_preset_tool(tool)
    if preset_id and load_preset(preset_id) is not None:
        return SERVICE_PRESET_KNOWLEDGE

    if tool in {"evidence", "science"}:
        return SERVICE_SCIENTIFIC_EVIDENCE
    if tool == "library":
        return SERVICE_INTERNAL_CORPUS
    if tool == "finance":
        return SERVICE_FINANCE_KNOWLEDGE
    if tool == "legal":
        return SERVICE_LEGAL_KNOWLEDGE
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


def resolve_turn_preset_id(composer_tool: str | None) -> str | None:
    return parse_user_preset_tool((composer_tool or "").strip().lower())


def adapter_filter_for_composer_tool(composer_tool: str | None) -> tuple[str, ...] | None:
    tool = (composer_tool or "").strip().lower()
    if tool == "pubmed":
        return ("pubmed",)
    if tool == "arxiv":
        return ("arxiv",)
    source_id = parse_source_pin_tool(tool)
    if source_id:
        return (source_id,)
    preset_id = parse_user_preset_tool(tool)
    if preset_id:
        preset = load_preset(preset_id)
        if preset is not None:
            return tuple(preset.adapters)
    return None
