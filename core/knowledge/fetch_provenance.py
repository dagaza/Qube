"""Structured fetch provenance for retrieval traces and Inspector Explain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.knowledge.discovery.types import CandidateUrl
from core.knowledge.search_outcome import search_outcome_from_relevance_diag


@dataclass(frozen=True)
class FetchProvenance:
    query: str
    composer_tool: str | None
    site_bias: tuple[str, ...]
    discovery_provider: str
    candidates: tuple[dict[str, Any], ...]
    selected_urls: tuple[str, ...]
    fetch_attempts: tuple[dict[str, Any], ...]
    extractor_name: str | None
    extractor_version: str | None
    extractor_confidence: float | None
    sections_emitted: int
    fetch_url_count: int = 0
    structured_data_type: str | None = None
    document_sections: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "composer_tool": self.composer_tool,
            "site_bias": list(self.site_bias),
            "discovery_provider": self.discovery_provider,
            "candidates": [dict(item) for item in self.candidates],
            "selected_urls": list(self.selected_urls),
            "fetch_attempts": [dict(item) for item in self.fetch_attempts],
            "extractor_name": self.extractor_name,
            "extractor_version": self.extractor_version,
            "extractor_confidence": self.extractor_confidence,
            "sections_emitted": self.sections_emitted,
            "fetch_url_count": self.fetch_url_count,
            "structured_data_type": self.structured_data_type,
            "document_sections": self.document_sections,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> FetchProvenance | None:
        if not isinstance(raw, Mapping):
            return None
        candidates = tuple(dict(item) for item in (raw.get("candidates") or ()) if isinstance(item, Mapping))
        fetch_attempts = tuple(
            dict(item) for item in (raw.get("fetch_attempts") or ()) if isinstance(item, Mapping)
        )
        site_bias = tuple(str(item) for item in (raw.get("site_bias") or ()) if str(item).strip())
        confidence = raw.get("extractor_confidence")
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None
        return cls(
            query=str(raw.get("query") or ""),
            composer_tool=str(raw.get("composer_tool") or "").strip() or None,
            site_bias=site_bias,
            discovery_provider=str(raw.get("discovery_provider") or "duckduckgo"),
            candidates=candidates,
            selected_urls=tuple(str(url) for url in (raw.get("selected_urls") or []) if str(url).strip()),
            fetch_attempts=fetch_attempts,
            extractor_name=str(raw.get("extractor_name") or "").strip() or None,
            extractor_version=str(raw.get("extractor_version") or "").strip() or None,
            extractor_confidence=confidence_value,
            sections_emitted=int(raw.get("sections_emitted") or 0),
            fetch_url_count=int(raw.get("fetch_url_count") or 0),
            structured_data_type=str(raw.get("structured_data_type") or "").strip() or None,
            document_sections=int(raw.get("document_sections") or 0),
        )


def candidate_records(candidates: list[CandidateUrl]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "url": candidate.url,
            "rank": candidate.rank,
            "title": candidate.title,
            "source": candidate.source,
        }
        for candidate in candidates
    )


def primary_extractor_from_fetch_diag(fetch_diag: Mapping[str, Any] | None) -> dict[str, Any]:
    if not fetch_diag:
        return {}
    succeeded = fetch_diag.get("succeeded") or []
    if not succeeded or not isinstance(succeeded, list):
        return {}
    first = succeeded[0]
    return dict(first) if isinstance(first, Mapping) else {}


def build_fetch_provenance(
    *,
    query: str,
    composer_tool: str | None,
    site_bias: tuple[str, ...] | None,
    discovery_provider: str,
    candidates: list[CandidateUrl],
    selected_urls: list[str],
    fetch_diag: Mapping[str, Any] | None,
    sections_emitted: int,
    fetch_url_count: int,
) -> FetchProvenance:
    attempts: list[dict[str, Any]] = []
    for entry in (fetch_diag or {}).get("attempts") or []:
        if isinstance(entry, Mapping):
            attempts.append(dict(entry))

    primary = primary_extractor_from_fetch_diag(fetch_diag)
    structured_type = primary.get("structured_data_type")
    if structured_type is not None:
        structured_type = str(structured_type).strip() or None

    confidence = primary.get("extractor_confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None

    return FetchProvenance(
        query=query,
        composer_tool=(composer_tool or "").strip() or None,
        site_bias=tuple(site_bias or ()),
        discovery_provider=discovery_provider,
        candidates=candidate_records(candidates),
        selected_urls=tuple(selected_urls),
        fetch_attempts=tuple(attempts),
        extractor_name=str(primary.get("extractor") or "").strip() or None,
        extractor_version=str(primary.get("extractor_version") or "").strip() or None,
        extractor_confidence=confidence_value,
        sections_emitted=sections_emitted,
        fetch_url_count=fetch_url_count,
        structured_data_type=structured_type,
        document_sections=int(primary.get("document_sections") or 0),
    )


def build_pipeline_stages_from_provenance(
    provenance: FetchProvenance,
    *,
    rejected_count: int = 0,
    latency_ms: float = 0.0,
) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = [
        {
            "stage": "discovery",
            "adapter": provenance.discovery_provider,
            "outputs_count": len(provenance.candidates),
            "site_bias": list(provenance.site_bias),
        },
        {
            "stage": "relevance_gate",
            "outputs_count": len(provenance.selected_urls),
            "rejected_count": rejected_count,
        },
    ]
    if provenance.fetch_url_count > 0:
        stages.append(
            {
                "stage": "fetch",
                "outputs_count": len(provenance.fetch_attempts),
                "fetch_url_count": provenance.fetch_url_count,
            }
        )
        if provenance.extractor_name:
            stages.append(
                {
                    "stage": "extract",
                    "adapter": provenance.extractor_name,
                    "extractor_version": provenance.extractor_version,
                    "extractor_confidence": provenance.extractor_confidence,
                    "outputs_count": provenance.document_sections,
                    "structured_data_type": provenance.structured_data_type,
                }
            )
        stages.append(
            {
                "stage": "section_rank",
                "outputs_count": provenance.sections_emitted,
            }
        )
    stages.append({"stage": "bundle", "latency_ms": round(latency_ms, 2)})
    return stages


def format_fetch_provenance_text(provenance: FetchProvenance) -> str:
    """Human-readable chain for Inspector Explain."""
    lines = [
        f'Query: "{provenance.query}"',
    ]
    if provenance.composer_tool:
        lines.append(f"Composer: @[tool:{provenance.composer_tool}]")

    from core.knowledge.discovery.policy import discovery_provider_label

    lines.extend(
        [
            "",
            "Discovery:",
            f"  provider: {discovery_provider_label(provenance.discovery_provider)}",
        ]
    )
    if provenance.site_bias:
        lines.append(f"  site_bias: [{', '.join(provenance.site_bias)}]")
    lines.append(f"  candidates: {len(provenance.candidates)} URLs (ranked)")

    if provenance.fetch_url_count <= 0:
        lines.extend(
            [
                "",
                "Fetch:",
                "  skipped (profile/composer fetch_url_count=0; SERP snippets only)",
            ]
        )
        return "\n".join(lines)

    lines.extend(["", "Selected for fetch:"])
    if provenance.selected_urls:
        for index, url in enumerate(provenance.selected_urls[: provenance.fetch_url_count], start=1):
            lines.append(f"  {index}. {url}")
    else:
        lines.append("  (none passed relevance gate)")

    lines.append("")
    lines.append("Fetch:")
    if not provenance.fetch_attempts:
        lines.append("  (no fetch attempts recorded)")
    for attempt in provenance.fetch_attempts:
        url = attempt.get("url", "—")
        tier = attempt.get("tier", "http")
        success = attempt.get("success")
        lines.append(f"  - {url}")
        lines.append(f"    tier: {tier}")
        lines.append(f"    success: {success}")
        if attempt.get("failure_reason"):
            lines.append(f"    failure_reason: {attempt.get('failure_reason')}")
        if attempt.get("status_code") is not None:
            lines.append(f"    status_code: {attempt.get('status_code')}")
        if attempt.get("total_bytes") is not None:
            lines.append(f"    bytes: {attempt.get('total_bytes')}")

    lines.append("")
    lines.append("Extractor:")
    if provenance.extractor_name:
        lines.append(f"  name: {provenance.extractor_name}")
        if provenance.extractor_version:
            lines.append(f"  version: {provenance.extractor_version}")
        if provenance.extractor_confidence is not None:
            lines.append(f"  confidence: {provenance.extractor_confidence:.2f}")
    else:
        lines.append("  (none — fetch did not produce extracted content)")

    lines.extend(["", "Output:"])
    lines.append(f"  document_sections: {provenance.document_sections}")
    if provenance.structured_data_type:
        lines.append(f"  structured_data: {provenance.structured_data_type}")
    lines.append(f"  evidence_objects: {provenance.sections_emitted}")
    return "\n".join(lines)


def format_pipeline_stages_summary(
    pipeline_stages: list[dict[str, Any]] | None,
) -> str:
    """Compact stage chain for qube.log (discovery → gate → fetch → extract → bundle)."""
    if not pipeline_stages:
        return "none"
    parts: list[str] = []
    for stage in pipeline_stages:
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("stage") or "?")
        details: list[str] = []
        if name == "discovery":
            outputs = stage.get("outputs_count")
            if outputs is not None:
                details.append(f"{outputs} urls")
            if stage.get("site_bias"):
                details.append("site_bias")
        elif name == "relevance_gate":
            if stage.get("outputs_count") is not None:
                details.append(f"kept={stage['outputs_count']}")
            rejected = stage.get("rejected_count")
            if rejected:
                details.append(f"dropped={rejected}")
        elif name == "fetch":
            fetch_count = stage.get("fetch_url_count", stage.get("outputs_count"))
            if fetch_count is not None:
                details.append(f"count={fetch_count}")
        elif name == "extract":
            adapter = stage.get("adapter")
            if adapter:
                details.append(str(adapter))
            outputs = stage.get("outputs_count")
            if outputs is not None:
                details.append(f"{outputs} sections")
        elif name == "section_rank":
            outputs = stage.get("outputs_count")
            if outputs is not None:
                details.append(f"{outputs} ranked")
        elif name == "bundle":
            latency = stage.get("latency_ms")
            if latency is not None:
                details.append(f"{latency}ms")
        label = name if not details else f"{name}({', '.join(details)})"
        parts.append(label)
    return " → ".join(parts) if parts else "none"


def summarize_web_pipeline_outcome(
    bundle: Any | None,
    relevance_diag: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Structured web pipeline summary for operational logging."""
    strategy = str(getattr(bundle, "retrieval_strategy", None) or "—")
    warnings = tuple(getattr(bundle, "warnings", None) or ())
    source_count = len(getattr(bundle, "sources", None) or ())
    fetch_url_count = 0
    if isinstance(relevance_diag, Mapping):
        fetch_provenance = relevance_diag.get("fetch_provenance")
        if isinstance(fetch_provenance, Mapping):
            fetch_url_count = int(fetch_provenance.get("fetch_url_count") or 0)
        pipeline_stages = relevance_diag.get("pipeline_stages")
        stages = pipeline_stages if isinstance(pipeline_stages, list) else None
    else:
        stages = None
    search_outcome = search_outcome_from_relevance_diag(
        relevance_diag if isinstance(relevance_diag, Mapping) else None
    )
    search_outcome_kind = (
        search_outcome.kind.value if search_outcome is not None else None
    )
    return {
        "strategy": strategy,
        "fetch_url_count": fetch_url_count,
        "warnings": warnings,
        "stages_summary": format_pipeline_stages_summary(stages),
        "source_count": source_count,
        "search_outcome_kind": search_outcome_kind,
    }


def fetch_provenance_from_relevance_diag(
    relevance_diag: Mapping[str, Any] | None,
) -> FetchProvenance | None:
    if not relevance_diag:
        return None
    raw = relevance_diag.get("fetch_provenance")
    if isinstance(raw, Mapping):
        return FetchProvenance.from_dict(raw)
    return None
