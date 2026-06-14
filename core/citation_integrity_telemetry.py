"""Structured citation-integrity events for ~/.qube/logs/llm_debug.log."""
from __future__ import annotations

from typing import Any

from core.citation_integrity import CitationIntegrityReport
from core.llm_structured_log import structured_llm_log


def log_citation_integrity(
    report: CitationIntegrityReport,
    *,
    phase: str = "worker_finalize",
    execution_route: str = "",
    session_id: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    payload = report.telemetry_dict(
        phase=phase,
        execution_route=execution_route,
        session_id=session_id,
    )
    if extra:
        payload.update(extra)
    structured_llm_log("citation_integrity", payload)


def log_citation_integrity_repair(
    *,
    session_id: str = "",
    execution_route: str = "",
    orphan_ids: list[str] | tuple[str, ...] = (),
    mode: str = "strip",
    chars_before: int = 0,
    chars_after: int = 0,
    retry_reason: str = "",
) -> None:
    structured_llm_log(
        "citation_integrity_repair",
        {
            "session_id": session_id,
            "execution_route": execution_route,
            "citation_orphan_ids": list(orphan_ids),
            "repair_mode": mode,
            "chars_before": chars_before,
            "chars_after": chars_after,
            "retry_reason": retry_reason,
        },
    )
