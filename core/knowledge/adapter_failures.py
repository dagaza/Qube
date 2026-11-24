"""Record adapter failures instead of swallowing them (plan S12, §3.7).

Finance and legal adapter loops used ``except Exception: result = []``, so a
failed SEC or CourtListener call looked exactly like "no filings found". Most
adapters also catch their own errors, log a warning and return ``[]``; the
per-turn HTTP outcome (``http_metrics.failed_sources``) is the record that
survives that. This module joins both into the bundle's coverage rationale and
the relevance diagnostics. Retrieval results are unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from core.knowledge.http_metrics import failed_sources_note


def adapter_error(adapter_id: str, exc: BaseException) -> dict[str, str]:
    return {"adapter": str(adapter_id), "error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def failure_report(
    errors: list[dict[str, str]],
    http_summary: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    """(failure rows for diagnostics, one rationale note or "")."""
    try:
        hosts, note = failed_sources_note(http_summary)
    except Exception:
        hosts, note = [], ""
    raised = list(dict.fromkeys(e["adapter"] for e in errors if e.get("adapter")))
    if raised:
        note += f" Sources that raised an error: {', '.join(raised)}."
    return [*errors, *hosts], note


def with_failure_note(bundle, note: str):
    """Append ``note`` to the bundle's coverage rationale (no other change)."""
    if not note or note.strip() in (bundle.coverage_rationale or ""):
        return bundle
    return replace(bundle, coverage_rationale=f"{bundle.coverage_rationale or ''}{note}".strip())
