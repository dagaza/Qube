"""Async deep-research worker (Phase 4)."""

from __future__ import annotations

import logging
from queue import Empty, Queue
from typing import Any

from PyQt6.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal

from core.knowledge.deep_research import (
    DeepResearchCancelled,
    DeepResearchProgress,
    DeepResearchResult,
    run_deep_research,
)
from core.knowledge.deep_research_synthesis import (
    DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS,
    compose_deep_research_report,
    synthesize_deep_research_findings,
)
from core.knowledge.evidence_transparency import build_evidence_transparency
from core.knowledge.graph.bundle_codec import bundle_to_dict
from core.knowledge.observability import build_retrieval_trace, record_retrieval_trace
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE
from core.knowledge.ui_adapter import bundle_to_ui_sources
from core.llm_execution_contract import PrimaryEngineTask

logger = logging.getLogger("Qube.DeepResearchWorker")


class DeepResearchWorker(QThread):
    """Background deep-research jobs; does not block chat LLM turns."""

    progress = pyqtSignal(dict)
    finished = pyqtSignal(dict)

    def __init__(self, synthesis_llm=None) -> None:
        super().__init__()
        self.queue: Queue[dict] = Queue()
        self.is_running = True
        self._enabled_mutex = QMutex()
        self._cancel_mutex = QMutex()
        self._is_enabled = False
        self._cancelled_ids: set[str] = set()
        self._active_request_id: str | None = None
        self._synthesis_llm = synthesis_llm

    def set_synthesis_llm(self, llm_worker) -> None:
        self._synthesis_llm = llm_worker

    def enqueue(self, payload: dict) -> None:
        """Queue ``{request_id, session_id, query, knowledge_service?}``."""
        if not self._is_enabled_read():
            logger.debug("[DeepResearch] enqueue ignored — worker disabled")
            return
        if not isinstance(payload, dict):
            return
        if not str(payload.get("query") or "").strip():
            return
        self.queue.put(dict(payload))

    def cancel_request(self, request_id: str) -> None:
        """Mark an in-flight or queued request as cancelled."""
        rid = str(request_id or "").strip()
        if not rid:
            return
        with QMutexLocker(self._cancel_mutex):
            self._cancelled_ids.add(rid)
        if rid == self._active_request_id:
            self._cancel_synthesis_llm()

    def _cancel_synthesis_llm(self) -> None:
        llm = self._synthesis_llm
        if llm is None:
            return
        cancel = getattr(llm, "cancel_generation", None)
        if callable(cancel):
            try:
                cancel()
            except Exception as exc:
                logger.debug("[DeepResearch] synthesis cancel failed: %s", exc)

    def _is_cancelled(self, request_id: str) -> bool:
        with QMutexLocker(self._cancel_mutex):
            return request_id in self._cancelled_ids

    def _clear_cancelled(self, request_id: str) -> None:
        with QMutexLocker(self._cancel_mutex):
            self._cancelled_ids.discard(request_id)

    def set_enabled(self, enabled: bool) -> None:
        with QMutexLocker(self._enabled_mutex):
            self._is_enabled = bool(enabled)

    def _is_enabled_read(self) -> bool:
        with QMutexLocker(self._enabled_mutex):
            return self._is_enabled

    def _emit_progress(self, prog: DeepResearchProgress, *, request_id: str) -> None:
        self.progress.emit(
            {
                "request_id": request_id,
                "phase": prog.phase,
                "message": prog.message,
                "sub_query_index": prog.sub_query_index,
                "sub_query_total": prog.sub_query_total,
                "sources_found": prog.sources_found,
                "sub_queries": list(prog.sub_queries),
            }
        )

    def _decompose_generate(self, system: str, user: str) -> str:
        """LLM callback for ``decompose_query_with_llm`` (positional system, user)."""
        llm = self._synthesis_llm
        if llm is None or not hasattr(llm, "generate"):
            return ""
        try:
            return str(
                llm.generate(
                    task=PrimaryEngineTask.deep_research_decompose,
                    system=system,
                    user=user,
                    temperature=0.15,
                    max_tokens=180,
                    debug_caller="deep_research_decompose",
                )
                or ""
            ).strip()
        except Exception as exc:
            logger.warning("[DeepResearch] LLM decompose failed: %s", exc)
            return ""

    def _record_bundle_trace(
        self,
        bundle,
        *,
        request_id: str,
        session_id: str,
        diagnostics: dict | None = None,
    ) -> None:
        if bundle is None:
            return
        from core.knowledge.deep_research_relevance import build_merge_relevance_diag

        relevance_diag = None
        if diagnostics:
            relevance_diag = build_merge_relevance_diag(diagnostics)
        trace = build_retrieval_trace(
            bundle,
            relevance_diag=relevance_diag,
            request_id=request_id,
            session_id=session_id or None,
        )
        record_retrieval_trace(trace, sources=bundle.sources)

    @staticmethod
    def _embedding_context(query: str, llm) -> tuple:
        if llm is None:
            return None, None
        cache = getattr(llm, "embedding_cache", None)
        embedder = getattr(cache, "embedder", None) if cache is not None else None
        if embedder is None:
            return None, None
        try:
            query_vector = embedder.embed_query(query)
            return embedder.embed_query, query_vector
        except Exception:
            return None, None

    def _synthesis_generate(self, *, system: str, user: str) -> str:
        llm = self._synthesis_llm
        if llm is None or not hasattr(llm, "generate"):
            return ""
        try:
            return str(
                llm.generate(
                    task=PrimaryEngineTask.deep_research_synthesis,
                    system=system,
                    user=user,
                    temperature=0.2,
                    max_tokens=DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS,
                    debug_caller="deep_research_synthesis",
                )
                or ""
            ).strip()
        except Exception as exc:
            logger.warning("[DeepResearch] synthesis LLM failed: %s", exc)
            return ""

    def _finalize_report(
        self,
        result: DeepResearchResult,
        payload: dict,
        *,
        request_id: str,
        session_id: str,
    ) -> tuple[str, list[dict], dict[str, Any], dict[str, Any]]:
        if self._is_cancelled(request_id):
            raise DeepResearchCancelled()

        bundle = result.merged_bundle
        self._record_bundle_trace(
            bundle,
            request_id=request_id,
            session_id=session_id,
            diagnostics=result.diagnostics,
        )

        generate_fn = None
        if bundle and bundle.sources and self._synthesis_llm is not None:
            self._emit_progress(
                DeepResearchProgress(
                    phase="synthesizing",
                    message="Synthesizing findings from evidence…",
                ),
                request_id=request_id,
            )
            generate_fn = self._synthesis_generate

        synthesis = synthesize_deep_research_findings(
            result.query,
            bundle,
            generate_fn=generate_fn,
        )

        if self._is_cancelled(request_id):
            raise DeepResearchCancelled()

        report = compose_deep_research_report(
            query=result.query,
            bundle=bundle,
            sub_queries=result.sub_queries,
            synthesis=synthesis,
        )
        sources = synthesis.ui_sources if synthesis.ui_sources else (
            bundle_to_ui_sources(bundle) if bundle else []
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["synthesis_applied"] = synthesis.synthesized
        transparency = build_evidence_transparency(
            bundle,
            diagnostics=diagnostics,
            sub_queries=result.sub_queries,
        )
        bundle_dict = bundle_to_dict(bundle) if bundle is not None else None
        return report, sources, diagnostics, transparency, bundle_dict

    def run(self) -> None:
        while self.is_running:
            try:
                payload = self.queue.get(timeout=0.5)
            except Empty:
                continue
            if not self._is_enabled_read():
                continue

            request_id = str(payload.get("request_id") or "")
            session_id = str(payload.get("session_id") or "")
            query = str(payload.get("query") or "").strip()
            knowledge_service = str(
                payload.get("knowledge_service") or SERVICE_SCIENTIFIC_EVIDENCE
            )

            if self._is_cancelled(request_id):
                self._clear_cancelled(request_id)
                continue

            self._active_request_id = request_id
            logger.info(
                "[DeepResearch] start request_id=%s session_id=%s query=%r",
                request_id,
                session_id,
                query[:120],
            )

            try:
                embed_fn, query_vector = self._embedding_context(
                    query, self._synthesis_llm
                )
                decompose_fn = None
                if self._synthesis_llm is not None:
                    decompose_fn = self._decompose_generate
                result = run_deep_research(
                    query,
                    knowledge_service=knowledge_service,
                    progress_cb=lambda p: self._emit_progress(
                        p, request_id=request_id
                    ),
                    should_cancel=lambda: self._is_cancelled(request_id),
                    embed_fn=embed_fn,
                    query_vector=query_vector,
                    decompose_generate_fn=decompose_fn,
                )
                report, sources, diagnostics, transparency, bundle_dict = self._finalize_report(
                    result,
                    payload,
                    request_id=request_id,
                    session_id=session_id,
                )
                finished = self._result_payload(result, payload)
                finished["report_markdown"] = report
                finished["sources"] = sources
                finished["diagnostics"] = diagnostics
                finished["evidence_transparency"] = transparency
                finished["bundle_dict"] = bundle_dict
                finished["synthesis_applied"] = diagnostics.get("synthesis_applied", False)
                self.finished.emit(finished)
            except DeepResearchCancelled:
                logger.info("[DeepResearch] cancelled request_id=%s", request_id)
                self.finished.emit(
                    {
                        "request_id": request_id,
                        "session_id": session_id,
                        "query": query,
                        "status": "cancelled",
                    }
                )
            except Exception as exc:
                logger.exception("[DeepResearch] job failed: %s", exc)
                self.finished.emit(
                    {
                        "request_id": request_id,
                        "session_id": session_id,
                        "query": query,
                        "status": "error",
                        "error": str(exc),
                    }
                )
            finally:
                self._active_request_id = None
                self._clear_cancelled(request_id)

    def stop(self) -> None:
        self.is_running = False
        self.wait(3000)

    @staticmethod
    def _result_payload(result: DeepResearchResult, payload: dict) -> dict:
        bundle = result.merged_bundle
        sources: list[dict] = []
        if bundle is not None and bundle.sources:
            sources = bundle_to_ui_sources(bundle)
        return {
            "request_id": str(payload.get("request_id") or ""),
            "session_id": str(payload.get("session_id") or ""),
            "query": result.query,
            "status": "ok" if bundle and bundle.sources else "no_results",
            "sub_queries": list(result.sub_queries),
            "report_markdown": result.report_markdown,
            "latency_ms": round(result.latency_ms, 1),
            "diagnostics": dict(result.diagnostics),
            "bundle_id": bundle.bundle_id if bundle else None,
            "coverage": bundle.coverage if bundle else None,
            "confidence": round(bundle.confidence, 4) if bundle else None,
            "source_count": len(bundle.sources) if bundle else 0,
            "sources": sources,
            "synthesis_applied": False,
        }
