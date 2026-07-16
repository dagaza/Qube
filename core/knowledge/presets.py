"""Knowledge presets — user-defined composer-facing knowledge environments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.knowledge.adapters.catalog import implemented_adapter_ids
from core.knowledge.types import (
    SERVICE_FINANCE_KNOWLEDGE,
    SERVICE_GENERAL_WEB,
    SERVICE_LEGAL_KNOWLEDGE,
    SERVICE_SCIENTIFIC_EVIDENCE,
)
from core.paths import user_data_root


def presets_dir() -> Path:
    path = user_data_root() / "knowledge" / "presets"
    path.mkdir(parents=True, exist_ok=True)
    return path


PRESET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")
MAX_PRESET_ADAPTERS = 8

RESERVED_PRESET_IDS = frozenset(
    {
        "evidence",
        "science",
        "internet",
        "trusted",
        "finance",
        "legal",
        "library",
        "memory",
        "research",
        "wikipedia",
        "pubmed",
        "arxiv",
        "fetch",
        "recipe",
    }
)

ALLOWED_BASE_SERVICES = frozenset(
    {
        SERVICE_SCIENTIFIC_EVIDENCE,
        SERVICE_FINANCE_KNOWLEDGE,
        SERVICE_LEGAL_KNOWLEDGE,
        SERVICE_GENERAL_WEB,
    }
)

RANKING_PROFILES = frozenset({"generic", "literature", "regulatory", "market_data"})
QUERY_PLANNERS = frozenset({"passthrough", "keyword_extract", "entity_centric"})


def normalize_site_bias_domain(raw: str) -> str:
    """Normalize a user-entered domain for ``site_bias`` lists."""
    value = (raw or "").strip().lower()
    if not value:
        return ""
    for prefix in ("https://", "http://"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    value = value.split("/")[0].split("?")[0].strip()
    if value.startswith("www."):
        value = value[4:]
    return value


def normalize_site_bias(domains: list[str] | tuple[str, ...] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in domains or ():
        domain = normalize_site_bias_domain(str(raw))
        if not domain or domain in seen:
            continue
        seen.add(domain)
        out.append(domain)
    return out


@dataclass
class KnowledgePreset:
    id: str
    label: str
    description: str = ""
    base_service: str = SERVICE_SCIENTIFIC_EVIDENCE
    adapters: list[str] = field(default_factory=list)
    site_bias: list[str] = field(default_factory=list)
    fetch_url_count: int | None = None
    adapter_policy: str = "fixed_order"
    ranking_profile: str = "generic"
    query_planner: str = "passthrough"
    composer_visible: bool = True
    created_at: str = ""
    version: int = 1

    def __post_init__(self) -> None:
        self.id = (self.id or "").strip().lower()
        self.label = (self.label or self.id).strip()
        self.base_service = (self.base_service or SERVICE_SCIENTIFIC_EVIDENCE).strip().lower()
        self.adapter_policy = (self.adapter_policy or "fixed_order").strip().lower()
        self.ranking_profile = (self.ranking_profile or "generic").strip().lower()
        self.query_planner = (self.query_planner or "passthrough").strip().lower()
        self.adapters = [str(a).strip().lower() for a in (self.adapters or []) if str(a).strip()]
        self.site_bias = normalize_site_bias(self.site_bias)
        if self.fetch_url_count is not None:
            self.fetch_url_count = max(0, int(self.fetch_url_count))
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> KnowledgePreset:
        return cls(
            id=str(raw.get("id") or ""),
            label=str(raw.get("label") or ""),
            description=str(raw.get("description") or ""),
            base_service=str(raw.get("base_service") or SERVICE_SCIENTIFIC_EVIDENCE),
            adapters=list(raw.get("adapters") or []),
            site_bias=list(raw.get("site_bias") or []),
            fetch_url_count=(
                int(raw["fetch_url_count"])
                if raw.get("fetch_url_count") is not None
                else None
            ),
            adapter_policy=str(raw.get("adapter_policy") or "fixed_order"),
            ranking_profile=str(raw.get("ranking_profile") or "generic"),
            query_planner=str(raw.get("query_planner") or "passthrough"),
            composer_visible=bool(raw.get("composer_visible", True)),
            created_at=str(raw.get("created_at") or ""),
            version=int(raw.get("version") or 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "base_service": self.base_service,
            "adapters": list(self.adapters),
            "site_bias": list(self.site_bias),
            "fetch_url_count": self.fetch_url_count,
            "adapter_policy": self.adapter_policy,
            "ranking_profile": self.ranking_profile,
            "query_planner": self.query_planner,
            "composer_visible": self.composer_visible,
            "created_at": self.created_at,
            "version": self.version,
        }

    def validate(self) -> None:
        if not PRESET_ID_RE.match(self.id):
            raise ValueError(f"Invalid preset id: {self.id!r}")
        if self.id in RESERVED_PRESET_IDS:
            raise ValueError(f"Preset id reserved: {self.id}")
        if not self.label:
            raise ValueError("Preset label is required")
        if self.base_service not in ALLOWED_BASE_SERVICES:
            raise ValueError(f"Unsupported base service: {self.base_service}")
        if self.base_service == SERVICE_GENERAL_WEB:
            if self.adapters:
                raise ValueError(
                    "Web fetch presets use site_bias domains, not API adapter ids."
                )
            if not self.site_bias:
                raise ValueError(
                    "Web fetch preset must include at least one site_bias domain."
                )
            if self.fetch_url_count is not None and self.fetch_url_count < 0:
                raise ValueError("fetch_url_count must be zero or positive")
        else:
            if self.site_bias:
                raise ValueError(
                    "site_bias is only supported on general_web source profiles."
                )
            if self.fetch_url_count is not None:
                raise ValueError(
                    "fetch_url_count is only supported on general_web source profiles."
                )
            if not self.adapters:
                raise ValueError("Preset must include at least one adapter")
            if len(self.adapters) > MAX_PRESET_ADAPTERS:
                raise ValueError(
                    f"Preset may include at most {MAX_PRESET_ADAPTERS} adapters"
                )
            allowed = set(implemented_adapter_ids(self.base_service))
            from core.knowledge.configured_sources import load_configured_source

            invalid = []
            for a in self.adapters:
                if a in allowed:
                    continue
                source = load_configured_source(a)
                if source is None or source.knowledge_service != self.base_service:
                    invalid.append(a)
            if invalid:
                hint = (
                    f"Unknown source(s): {', '.join(invalid)}. "
                    "Use built-in adapter IDs (e.g. pubmed, arxiv, openalex) or a custom "
                    "source id saved under Settings → Knowledge → Custom sources."
                )
                raise ValueError(hint)
        if self.ranking_profile not in RANKING_PROFILES:
            raise ValueError(f"Unknown ranking profile: {self.ranking_profile}")
        if self.query_planner not in QUERY_PLANNERS:
            raise ValueError(f"Unknown query planner: {self.query_planner}")


def available_preset_adapter_ids(service_id: str) -> tuple[str, ...]:
    """Built-in and saved custom source ids valid for preset adapter lists."""
    from core.knowledge.configured_sources import list_configured_sources

    ids = list(implemented_adapter_ids(service_id))
    for source in list_configured_sources():
        if source.knowledge_service == service_id and source.id not in ids:
            ids.append(source.id)
    return tuple(sorted(ids))


def _preset_path(preset_id: str) -> Path:
    return presets_dir() / f"{preset_id.strip().lower()}.json"


def list_presets(*, composer_visible_only: bool = False) -> list[KnowledgePreset]:
    out: list[KnowledgePreset] = []
    for path in sorted(presets_dir().glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            preset = KnowledgePreset.from_dict(raw)
            preset.validate()
        except Exception:
            continue
        if composer_visible_only and not preset.composer_visible:
            continue
        out.append(preset)
    return out


def load_preset(preset_id: str) -> KnowledgePreset | None:
    path = _preset_path(preset_id)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        preset = KnowledgePreset.from_dict(raw)
        preset.validate()
        return preset
    except Exception:
        return None


def save_preset(preset: KnowledgePreset) -> None:
    preset.validate()
    path = _preset_path(preset.id)
    path.write_text(
        json.dumps(preset.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def delete_preset(preset_id: str) -> bool:
    path = _preset_path(preset_id)
    if path.is_file():
        path.unlink()
        return True
    return False


def parse_user_preset_tool(tool_id: str) -> str | None:
    """Return preset id from composer tool id like ``user:biology``."""
    tool = (tool_id or "").strip().lower()
    if tool.startswith("user:"):
        return tool.split(":", 1)[1].strip() or None
    return None


def preset_retrieval_overrides(preset_id: str) -> dict[str, Any]:
    """Return ``site_bias`` / ``fetch_url_count`` for a general_web source profile."""
    preset = load_preset(preset_id)
    if preset is None or preset.base_service != SERVICE_GENERAL_WEB:
        return {}
    overrides: dict[str, Any] = {"site_bias": tuple(preset.site_bias)}
    if preset.fetch_url_count is not None:
        overrides["fetch_url_count"] = preset.fetch_url_count
    return overrides


def parse_source_pin_tool(tool_id: str) -> str | None:
    """Return source/adapter id from composer tool id like ``source:pubmed``."""
    tool = (tool_id or "").strip().lower()
    if tool.startswith("source:"):
        return tool.split(":", 1)[1].strip() or None
    return None
