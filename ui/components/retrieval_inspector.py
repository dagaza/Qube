"""Read-only Retrieval Inspector — summary, graph, compare, explain."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.knowledge.explain_preset import build_explain_preset, format_explain_preset_text
from core.knowledge.fetch_provenance import (
    fetch_provenance_from_relevance_diag,
    format_fetch_provenance_text,
)
from core.knowledge.search_outcome import (
    format_search_outcome_summary_line,
    search_outcome_from_relevance_diag,
)
from core.knowledge.pipeline_graph import format_pipeline_graph_text
from core.knowledge.retrieval_replay import compare_traces, replay_from_record
from core.knowledge.retrieval_trace_reader import format_retrieval_trace_summary


class RetrievalInspector(QWidget):
    """Tabs: Summary / Graph / Compare / Explain (read-only)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._summary = QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._graph = QPlainTextEdit()
        self._graph.setReadOnly(True)
        self._compare = QPlainTextEdit()
        self._compare.setReadOnly(True)
        self._explain = QPlainTextEdit()
        self._explain.setReadOnly(True)

        for editor, label in (
            (self._summary, "Summary"),
            (self._graph, "Graph"),
            (self._compare, "Compare"),
            (self._explain, "Explain"),
        ):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.addWidget(editor)
            self._tabs.addTab(page, label)

        layout.addWidget(self._tabs)
        self._record: dict | None = None
        self._trace: dict | None = None
        self._routing_record: dict | None = None
        self._db = None

    def set_database(self, db) -> None:
        self._db = db

    def load(
        self,
        *,
        trace: dict | None = None,
        record: dict | None = None,
        preset_id: str | None = None,
        routing_record: dict | None = None,
    ) -> None:
        self._trace = trace
        self._record = record
        self._routing_record = routing_record
        if trace is None and record is None:
            self._summary.setPlainText("No retrieval data for this turn.")
            self._graph.setPlainText("")
            self._compare.setPlainText("")
            self._explain.setPlainText("")
            return

        effective_trace = trace or self._trace_from_record(record)
        self._summary.setPlainText(
            self._format_summary(effective_trace, record, self._routing_record)
        )
        self._graph.setPlainText(format_pipeline_graph_text(effective_trace))
        self._compare.setPlainText(self._format_compare(record))
        self._explain.setPlainText(self._format_explain(effective_trace, preset_id))

    def _trace_from_record(self, record: dict | None) -> dict | None:
        if not record:
            return None
        return {
            "query_raw": record.get("query_raw"),
            "query_resolved": record.get("query_resolved"),
            "knowledge_service": record.get("knowledge_service"),
            "retrieval_strategy": record.get("retrieval_strategy"),
            "adapter_calls": [],
            "evidence_ids_kept": [],
            "candidates_rejected_count": 0,
            "coverage": record.get("coverage"),
            "confidence": record.get("confidence"),
            "latency_ms": record.get("latency_ms"),
            "retrieval_profile": record.get("retrieval_profile"),
            "preset_id": record.get("preset_id"),
        }

    def _format_summary(
        self,
        trace: dict | None,
        record: dict | None,
        routing_record: dict | None = None,
    ) -> str:
        if not trace:
            return "No retrieval trace available."
        lines = [format_retrieval_trace_summary(trace)]
        if record:
            lines.extend(
                [
                    "",
                    f"Request id: {record.get('request_id', '—')}",
                    f"Bundle id: {record.get('bundle_id', '—')}",
                    f"Retrieval profile: {record.get('retrieval_profile', '—')}",
                ]
            )
            fingerprint = record.get("context_fingerprint_json")
            if fingerprint:
                lines.append(f"Context fingerprint: stored")
        warnings = trace.get("warnings") or []
        if warnings:
            lines.append(f"Warnings: {'; '.join(str(w) for w in warnings)}")
        diag = trace.get("relevance_diag") or {}
        search_line = format_search_outcome_summary_line(
            search_outcome_from_relevance_diag(diag)
        )
        if search_line:
            lines.append(search_line)
        provenance = fetch_provenance_from_relevance_diag(diag)
        if provenance:
            if provenance.fetch_url_count > 0 and provenance.extractor_name:
                lines.append(
                    f"Fetch: {provenance.extractor_name} "
                    f"({provenance.sections_emitted} section(s))"
                )
            elif provenance.fetch_url_count <= 0:
                lines.append("Fetch: SERP snippets only")
        if diag:
            diag_bits = []
            for key in ("retrieval_profile", "ranking_profile", "preset_id", "scientific_cache_hit"):
                if key in diag:
                    diag_bits.append(f"{key}={diag[key]}")
            if diag_bits:
                lines.append("Diagnostics: " + ", ".join(diag_bits))
        from core.knowledge.routing_inspect_explain import format_routing_inspect_text

        routing_text = format_routing_inspect_text(routing_record)
        if routing_text:
            lines.append(routing_text)
        return "\n".join(lines)

    def _format_explain(
        self,
        trace: dict | None,
        preset_id: str | None,
    ) -> str:
        diag = (trace or {}).get("relevance_diag") or {}
        search_outcome = search_outcome_from_relevance_diag(diag)
        provenance = fetch_provenance_from_relevance_diag(diag)
        sections: list[str] = []
        if search_outcome is not None:
            from core.knowledge.search_outcome import format_search_outcome_explain_text
            from core.knowledge.discovery.policy import discovery_policy_summary_lines

            sections.append(format_search_outcome_explain_text(search_outcome))
            policy_lines = discovery_policy_summary_lines()
            if policy_lines:
                sections.append("")
                sections.append("Discovery policy:")
                sections.extend(f"  {line}" for line in policy_lines)
        if provenance is not None:
            if sections:
                sections.append("")
            sections.append(format_fetch_provenance_text(provenance))
        if sections:
            return "\n".join(sections)
        if preset_id:
            from core.app_settings import get_retrieval_profile

            explain = build_explain_preset(
                preset_id,
                retrieval_profile=get_retrieval_profile(),
            )
            return format_explain_preset_text(explain)
        return (
            "No fetch provenance recorded for this turn. "
            "Use External knowledge v2 with a Balanced or Thorough profile, "
            "or attach @[tool:fetch] / @[tool:recipe], then inspect a fetch turn."
        )

    def _format_compare(self, record: dict | None) -> str:
        if not record or self._db is None:
            return (
                "Replay compare requires a stored retrieval record. "
                "Run a knowledge turn with external v2 enabled."
            )
        try:
            result = replay_from_record(record, mode="current", db=self._db)
        except Exception as exc:
            return f"Replay failed: {exc}"

        original = {
            "evidence_ids_kept": [],
            "coverage": record.get("coverage"),
            "latency_ms": record.get("latency_ms"),
            "confidence": record.get("confidence"),
        }
        replay_trace = None
        if result.outcome and result.outcome.bundle:
            from core.knowledge.observability import (
                build_retrieval_trace,
                serialize_retrieval_trace,
            )

            replay_trace = serialize_retrieval_trace(
                build_retrieval_trace(
                    result.outcome.bundle,
                    relevance_diag=result.outcome.relevance_diag,
                ),
                sources=result.outcome.bundle.sources,
            )
        cmp = compare_traces(original, replay_trace)
        lines = [
            f"Replay mode: {result.mode}",
            f"Original bundle: {result.original_bundle_id}",
            f"Replay bundle: {result.replay_bundle_id or '—'}",
        ]
        for warning in result.warnings:
            lines.append(f"Note: {warning}")
        if cmp:
            lines.extend(
                [
                    "",
                    f"Coverage: {cmp.get('coverage_before')} → {cmp.get('coverage_after')}",
                    f"Latency: {cmp.get('latency_before_ms')} → {cmp.get('latency_after_ms')} ms",
                    f"Confidence: {cmp.get('confidence_before')} → {cmp.get('confidence_after')}",
                ]
            )
        return "\n".join(lines)


def open_retrieval_inspector_dialog(
    parent,
    *,
    is_dark: bool = True,
    trace: dict | None = None,
    record: dict | None = None,
    preset_id: str | None = None,
    db=None,
    routing_record: dict | None = None,
) -> None:
    from PyQt6.QtWidgets import QDialog, QVBoxLayout

    dlg = QDialog(parent)
    dlg.setWindowTitle("Retrieval Inspector")
    dlg.resize(680, 480)
    layout = QVBoxLayout(dlg)
    inspector = RetrievalInspector(dlg)
    inspector.set_database(db)
    inspector.load(
        trace=trace,
        record=record,
        preset_id=preset_id,
        routing_record=routing_record,
    )
    layout.addWidget(inspector)
    dlg.exec()
