from PyQt6.QtCore import QThread, pyqtSignal
import dataclasses
import requests
import json
import time
import re
import copy
import logging
import uuid
import os
import queue
import threading
from typing import Any
from urllib.parse import urlparse

from core.app_settings import (
    DEFAULT_ENGINE_MODE,
    get_engine_mode,
    get_internal_model_path,
    get_internal_n_gpu_layers,
    get_internal_n_threads,
    get_internal_prompt_layout_override,
    get_llm_chat_history_messages,
    get_llm_context_limit,
    get_llm_min_p,
    get_llm_output_token_limit,
    get_llm_output_token_limit_enabled,
    get_llm_presence_penalty,
    get_llm_repeat_penalty,
    get_llm_temperature,
    get_llm_top_k,
    get_llm_top_p,
    get_mcp_internet_hybrid_enabled,
    get_mcp_rag_auto_activator_enabled,
    get_mcp_rag_enabled,
    get_mcp_rag_strict_enabled,
    get_citation_integrity_enforce,
    get_citation_integrity_missing_retry,
    missing_gguf_shards,
    resolve_internal_model_path,
    set_engine_mode as persist_engine_mode,
    set_llm_chat_history_messages,
    set_llm_context_limit,
    set_llm_min_p,
    set_llm_output_token_limit,
    set_llm_output_token_limit_enabled,
    set_llm_presence_penalty,
    set_llm_repeat_penalty,
    set_llm_temperature,
    set_llm_top_k,
    set_llm_top_p,
    set_mcp_internet_hybrid_enabled,
    set_mcp_rag_auto_activator_enabled,
    set_mcp_rag_enabled,
    set_mcp_rag_strict_enabled,
)
from core.prompt_blocks import build_prompt_blocks, resolve_retrieval_wrapper_mode
from core.skills import activate_skills, build_skill_context
from core.app_settings import get_skill_settings, get_skills_debug_log_enabled
from core.preference_formatters import format_web_snippets
from core.preference_policy import apply_tool_policy, resolve_preference_policy
from core.prompt_renderers import render_messages
from core.prompt_layout import PromptLayoutResolution, resolve_prompt_layout
from core.redacted_thinking_filter import RedactedThinkingStreamFilter
from core.native_meta_leading_strip import LeadingMetaInstructionStripper
from core.output_degeneration import OutputDegenerationStreamObserver
from core.stream_repetition_guard import create_stream_repetition_guard
from core.harmony_degeneration import (
    harmony_tail_degenerate,
    is_harmony_orphan_stream_fragment,
)
from core.harmony_protocol import harmony_stream_parser_enabled, is_harmony_contract
from core.llm_structured_log import structured_llm_log
from core.conversation_health import (
    ConversationHealthState,
    TurnAnomalyOutcome,
    initial_conversation_health,
    resolve_conversation_health_policy,
    update_conversation_health,
)
from core.conversation_health_telemetry import log_conversation_health_update
from core.turn_context import TurnContext, resolve_turn_context
from core.llm_execution_contract import PrimaryEngineTask
from core.harmony_stream_parser import HarmonyStreamParser
from core.gemma_output_strip import (
    GemmaThoughtStreamFilter,
    is_gemma_model_identity,
    strip_gemma_output_artifacts,
)
from core.output_artifact_strip import (
    LeakedHarmonyScaffoldStreamFilter,
    strip_output_artifacts,
)
from core.delimiter_grammar_extractor import (
    DelimiterGrammarStreamFilter,
    extract_delimiter_grammar,
)
from core.output_artifact_report import (
    build_output_artifact_report,
    log_output_artifact_report,
)
from core.template_output_profile import TemplateOutputProfile
from core.completion_output_trace import (
    CompletionOutputSnapshot,
    log_completion_output_trace,
)
from core.llm_truth_diff import (
    bind_llm_worker_truth_diff_hooks,
    get_llm_truth_diff_logger,
    llm_truth_diff_enabled,
)
from core.canonical_request import (
    canonical_trace_export_enabled,
    log_canonical_request_trace,
)
from core.golden_trace_capture import (
    build_golden_trace,
    golden_trace_capture_mode_enabled,
    maybe_capture_golden_trace,
)
from core.llm_debug_markers import (
    log_chat_exchange_begin,
    log_chat_exchange_end,
    next_exchange_id,
)
from core.execution_policy import execution_policy_debug_fields
from core.conversational_follow_up import preserve_streamed_follow_up
from core.output_token_budget import (
    probable_max_tokens_truncation,
    resolve_output_token_budget,
)
from core.output_validation_sanitize import sanitize_output_for_validation
from core.output_validation_trace import log_output_validation_trace
from core.stream_replace_policy import resolve_stream_replacement
from core.memory_filters import (
    detect_recall_intent,
    should_apply_recall_fusion,
    detect_explicit_remember,
    detect_hard_explicit_web_request,
    detect_file_search_intent,
    detect_narrative_intent,
    is_assistant_failure_message,
    is_thin_content,
    query_implies_live_web_intent,
    query_explicitly_requests_library_search,
    query_has_lexical_library_signal,
    should_downgrade_embedding_rag_on_continuation,
    should_downgrade_short_vague_retrieval_on_first_turn,
    library_lane_allowed,
    should_run_internet_search_for_route,
)
from core.discourse_intent import (
    FOLLOW_UP_SUPPRESS_THRESHOLD,
    FollowUpClassification,
    FollowUpKind,
    build_entity_aspect_grounding_suffix,
    build_minimal_referent_fallback_suffix,
    build_referent_salience_suffix,
    build_topic_salience_suffix,
    classify_follow_up,
    discourse_debug_enabled,
    discourse_prompt_hint_enabled,
)
from core.collapse_diagnostics import compute_collapse_diagnostics
from core.collapse_diagnostics_telemetry import log_collapse_diagnostics
from core.citation_integrity import (
    analyze_citations,
    repair_orphan_citations,
    valid_source_ids as citation_valid_source_ids,
)
from core.citation_integrity_telemetry import (
    log_citation_integrity,
    log_citation_integrity_repair,
)
from core.citation_missing_retry import maybe_retry_missing_web_citations
from core.citation_renumber import renumber_citations_by_appearance
from core.generation_debug_capture import (
    GenerationDebugRecorder,
    apply_debug_sampling_overrides,
    generation_debug_enabled,
)
from core.history_degeneration import (
    history_suppression_reason,
    resolve_assistant_history_content,
)
from core.history_degeneration_telemetry import log_history_degeneration_suppression
from core.output_degeneration_telemetry import log_output_degeneration
from core.prior_turn_reliability import (
    build_prior_turn_unreliable_suffix,
    history_contains_suppressed_assistant,
)
from core.discourse_query_rewrite import ResolvedUserQuery, resolve_ambiguous_user_query
from core.discourse_prompt_rewrite import (
    DiscoursePromptRewrite,
    resolve_discourse_prompt_rewrite,
    select_salience_anchor,
)
from core.discourse_telemetry import (
    log_discourse_prompt_rewrite,
    log_discourse_query_rewrite,
    log_discourse_referent_trace,
)
from core.discourse_query import (
    ResolvedRetrievalQuery,
    SearchTargetResult,
    build_resolved_retrieval_query,
    should_veto_ungrounded_web_follow_up,
    web_query_rewrite_failed,
)
from core.app_settings import (
    get_default_knowledge_service,
    get_retrieval_profile,
    research_map_enabled,
)
from core.knowledge.registry import (
    WEB_COMPOSER_TOOLS,
    adapter_filter_for_composer_tool,
    resolve_preset_retrieval_overrides,
    resolve_turn_knowledge_service,
    resolve_turn_preset_id,
)
from core.knowledge.types import SERVICE_INTERNAL_CORPUS
from core.knowledge.web_retrieval import run_web_retrieval
from core.knowledge.fetch_provenance import summarize_web_pipeline_outcome
from core.knowledge.search_outcome import search_outcome_from_relevance_diag
from core.knowledge.adapters.duckduckgo import failure_sentinel_reason
from core.web_search_audit import (
    STATUS_VETOED_TOOL_DISABLED,
    STATUS_VETOED_UNGROUNDED,
    build_audit_event_from_llm_turn,
    record_web_search_audit,
)
from core.discourse_state import (
    DiscourseState,
    promote_referent_after_assistant,
    update_discourse_state,
)
from core.memory_usage_recorder import get_memory_usage_recorder, compute_query_fingerprint
from core.app_settings import (
    get_discourse_grounding_enabled,
    get_enable_chat_personality_nudge,
    get_enable_memory_v7_salvage,
)
from core.app_settings import get_sidecar_query_rewrite_enabled
from core.dual_query_retrieval import merge_memory_search_results, merge_rag_search_results
from core.sidecar_query_rewrite import propose_query_expansion
from core.shadow_retrieval_policy import (
    build_shadow_state_from_worker,
    compute_retrieval_policy,
    get_shadow_retrieval_telemetry,
    shadow_retrieval_policy_enabled,
)
from core.sidecar_telemetry import get_sidecar_telemetry
from core.source_digest import digest_memory_context, digest_rag_context
from core.sidecar_types import QueryExpansion
from core.rag_trigger_routing import (
    apply_custom_rag_trigger_route,
    matches_custom_rag_trigger,
)
from core.composer_attachments import (
    attachment_summary,
    build_referenced_conversation_context,
    is_web_composer_tool,
    resolve_attachment_routing,
)
from core.help_corpus_manifest import HELP_DOC_SOURCE_PREFIX
from core.help_corpus_retrieval import (
    append_canonical_action_block,
    build_canonical_context_block,
    canonical_answer_system_hint,
    help_doc_ids_from_sources,
    log_help_query,
    match_canonical_answer,
)

from mcp.rag_tool import rag_search
from mcp.memory_tool import memory_search
from workers.intent_router import EmbeddingCache

from mcp.cognitive_router import CognitiveRouterV4
from mcp.routing_debug import (
    RoutingDebugBuffer,
    build_chat_contract_trace,
    build_engine_input_trace,
    build_model_router_trace,
    build_record,
    build_retrieval_outcome_snapshot,
    routing_debug_log_enabled,
    routing_debug_log_redact_query,
    routing_debug_log_verbose,
    serialize_record_for_log,
)
from mcp.router_telemetry import RouterTelemetryBrain
from mcp.router_self_tuner import AdaptiveRouterSelfTunerV2
from mcp.router_lane_stats import RouteFeedbackEvent

logger = logging.getLogger("Qube.LLM")
routing_persist_logger = logging.getLogger("Qube.RoutingDebug")

NATIVE_EMPTY_VISIBLE_OUTPUT_MSG = (
    "The model finished without producing any visible text. "
    "Try sending again, adjust Think, or inspect ~/.qube/logs/llm_debug.log."
)


def is_native_empty_visible_output_notice(text: str) -> bool:
    """True for the UI placeholder when native streaming produced no visible text."""
    t = (text or "").strip()
    if not t:
        return False
    msg = NATIVE_EMPTY_VISIBLE_OUTPUT_MSG
    return t == msg or t == msg + msg


class LLMWorker(QThread):
    sentence_ready = pyqtSignal(str, str)
    tts_turn_superseded = pyqtSignal(str)  # session_id — clear in-flight TTS after native retry
    token_streamed = pyqtSignal(str, str)  # session_id, token
    status_update = pyqtSignal(str)
    ttft_latency = pyqtSignal(float)
    tps_metric = pyqtSignal(float)
    context_retrieved = pyqtSignal(bool)
    # active, via_direct (force/manual/@internet), via_hybrid (cognitive router)
    web_search_active = pyqtSignal(bool, bool, bool)
    web_search_outcome_hint = pyqtSignal(str)
    ddg_backoff_started = pyqtSignal(int)  # remaining_seconds
    discovery_tier_b_suggested = pyqtSignal()
    response_finished = pyqtSignal(str, str)
    sources_found = pyqtSignal(str, list)  # session_id, sources
    evidence_transparency_found = pyqtSignal(str, dict)  # session_id, transparency
    router_telemetry_updated = pyqtSignal(dict, dict)  # summary, tuner_state
    sidecar_telemetry_updated = pyqtSignal(dict)
    routing_debug_record_added = pyqtSignal(dict)  # serialized RoutingDebugRecord
    # Phase B: turn-scoped enrichment context (session_id + rag chunk ids + message ids).
    # Emitted once per completed turn, before response_finished, so main.py can
    # forward a rich payload to EnrichmentWorker.enqueue(payload=...).
    enrichment_context_ready = pyqtSignal(dict)
    stream_replaced = pyqtSignal(str, str)  # session_id, full replacement text
    turn_notice = pyqtSignal(str, dict)  # session_id, {"kind": "max_tokens"|"format_retry", ...}

    MAX_TOTAL_RETRIEVAL_CHARS = 4500
    MEMORY_BUDGET = 1500
    RAG_BUDGET = 3000

    # Streaming: read timeout applies between SSE chunks (stall guard); wall cap is absolute safety
    _STREAM_CONNECT_TIMEOUT = 20
    _STREAM_READ_TIMEOUT = 180
    _MAX_STREAM_WALL_SECONDS = 900

    # Per-message cap before sending to the API (single huge assistant/user blobs).
    CHAT_HISTORY_SINGLE_MESSAGE_MAX_CHARS = 14000

    def __init__(self, embedder, store, db_manager, native_engine=None, sidecar_client=None):
        super().__init__()

        self.prompt = ""
        self.session_id = None
        self.api_url = "http://localhost:1234/v1/chat/completions"

        self.embedder = embedder
        self.store = store
        self.db = db_manager
        self._native_engine = native_engine
        self._notify_native_hardware_reload = False
        self._sidecar_client = sidecar_client
        self._last_native_job_cancelled = False
        self.engine_mode = get_engine_mode()
        self._bind_truth_diff_hooks()

        self.embedding_cache = EmbeddingCache(self.embedder)

        # triggers
        try:
            self.cached_custom_triggers = [
                t.lower() for t in self.db.get_rag_triggers()
            ]
        except Exception:
            self.cached_custom_triggers = []

        # ================================
        # BRAIN STACK
        # ================================
        self.cognitive_router = CognitiveRouterV4()
        self.telemetry = RouterTelemetryBrain()
        self._shadow_retrieval_telemetry = get_shadow_retrieval_telemetry()
        self.router_tuner = AdaptiveRouterSelfTunerV2()
        self.routing_debug_buffer = RoutingDebugBuffer()
        self._routing_debug_turn_seq = 0
        self._last_persisted_routing_turn_id: int | None = None

        self.USE_COGNITIVE_ROUTER = True
        self.USE_ADAPTIVE_ROUTER = True
        self.USE_TELEMETRY = True
        _internet_hybrid = get_mcp_internet_hybrid_enabled()
        self.USE_COGNITIVE_ROUTER_INTERNET = _internet_hybrid

        # toggles
        self.mcp_auto_enabled = get_mcp_rag_auto_activator_enabled()
        self.temperature = get_llm_temperature()
        self.context_window = get_llm_context_limit()
        self.output_token_limit_enabled = get_llm_output_token_limit_enabled()
        self.output_token_limit = get_llm_output_token_limit()
        # Sliding window: max DB messages to include in the chat completion (user-controlled).
        self.max_history_messages = get_llm_chat_history_messages()
        self.top_k = get_llm_top_k()
        self.repeat_penalty = get_llm_repeat_penalty()
        self.presence_penalty = get_llm_presence_penalty()
        self.top_p = get_llm_top_p()
        self.min_p = get_llm_min_p()
        self.mcp_rag_enabled = get_mcp_rag_enabled()
        self.mcp_strict_enabled = get_mcp_rag_strict_enabled()
        self.mcp_internet_enabled = _internet_hybrid
        self._force_web_enabled = False

        # Local llama.cpp / LM Studio: align server-side prompt/KV reuse with UI session switches
        self._last_completed_llm_session_id = None
        self._server_kv_cleared_for_session_id = None
        self._discourse_by_session: dict[str, DiscourseState] = {}
        self._conversation_health_by_session: dict[str, ConversationHealthState] = {}
        self._prior_execution_route_by_session: dict[str, str] = {}
        self._prior_web_empty_by_session: dict[str, bool] = {}
        self._push_sampling_to_native()

    def _sampling_payload(self) -> dict:
        payload: dict = {
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self.top_k > 0:
            payload["top_k"] = self.top_k
        if self.min_p > 0:
            payload["min_p"] = self.min_p
        return payload

    def _push_sampling_to_native(self) -> None:
        engine = self._native_engine
        if engine is None or not hasattr(engine, "set_sampling_overrides"):
            return
        engine.set_sampling_overrides(**self._sampling_payload())

    def _is_local_llm_service(self) -> bool:
        """Only localhost inference gets cache_prompt / flush hints (OpenAI cloud may 400 on extras)."""
        try:
            host = (urlparse(self.api_url).hostname or "").lower()
            return host in ("localhost", "127.0.0.1", "::1")
        except Exception:
            return False

    def _uses_external_http(self) -> bool:
        return getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) != "internal"

    def _is_internal_nvidia_family(self) -> bool:
        """Best-effort detection for Nemotron/NVIDIA models loaded in native engine."""
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) != "internal" or not self._native_engine:
            return False
        try:
            snap = self._native_engine.get_model_reasoning_telemetry() or {}
            if not bool(snap.get("loaded")):
                return False
            name = str(snap.get("model_name", "") or "")
            base = str(snap.get("model_basename", "") or "")
            ident = f"{name} {base}".lower()
            return ("nemotron" in ident) or ("nvidia" in ident)
        except Exception:
            return False

    def _reasoning_family_harmony_leak_strip_active(self) -> bool:
        """Strip leaked Harmony scaffold tokens during native streaming/sanitize."""
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) != "internal" or not self._native_engine:
            return False
        try:
            from core.qwen3_thinking_policy import (
                is_reasoning_family_harmony_leak_strip_candidate,
            )

            snap = self._native_engine.get_model_reasoning_telemetry() or {}
            if not bool(snap.get("loaded")):
                return False
            name = str(snap.get("model_name", "") or "")
            base = str(snap.get("model_basename", "") or "")
            path = str(getattr(self._native_engine, "_model_path", "") or "")
            ident_name = f"{name} {base}".strip()
            if is_reasoning_family_harmony_leak_strip_candidate(
                model_path=path,
                model_name=ident_name,
            ):
                return True
            return bool(snap.get("supports_thinking_tokens"))
        except Exception:
            return False

    def _resolve_turn_prompt_layout(self) -> PromptLayoutResolution:
        """
        Resolved layout for this turn (PR1: observability only; messages unchanged).
        Internal engine uses load-time resolution from native telemetry when available.
        """
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal" and self._native_engine:
            try:
                snap = self._native_engine.get_model_reasoning_telemetry() or {}
                layout = snap.get("prompt_layout")
                source = snap.get("prompt_layout_source")
                if layout and source:
                    return PromptLayoutResolution(
                        layout=layout,  # type: ignore[arg-type]
                        source=str(source),
                        degraded=bool(snap.get("prompt_layout_degraded")),
                        evidence=str(snap.get("prompt_layout_evidence") or "")[:240],
                    )
            except Exception:
                pass
        path = resolve_internal_model_path(get_internal_model_path() or "")
        basename = os.path.basename(path) if path else ""
        return resolve_prompt_layout(
            model_id=basename,
            model_display_name=basename,
            model_path=path,
            settings_override=get_internal_prompt_layout_override(),
        )

    def _flush_server_kv_hint(self) -> None:
        """
        Tiny non-streaming completion so llama.cpp/LM Studio advance/rotate prompt cache
        away from the previous conversation. Unique user text avoids prefix-cache hits.
        """
        if not self._uses_external_http():
            return
        if not self._is_local_llm_service():
            return
        token = uuid.uuid4().hex[:10]
        body = {
            "messages": [{"role": "user", "content": f"[qube:ctx:{token}]"}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
            "cache_prompt": False,
        }
        try:
            logger.debug("[LLM] Cross-session server KV / prompt-cache hint (max_tokens=1)")
            r = requests.post(
                self.api_url,
                json=body,
                timeout=(5, 25),
                headers={"Connection": "close"},
            )
            try:
                r.raise_for_status()
            except Exception:
                logger.debug("[LLM] KV hint HTTP status: %s", getattr(r, "status_code", "?"))
            r.close()
        except Exception as e:
            logger.debug("[LLM] KV hint failed (safe to ignore): %s", e)

    def notify_active_session_changed(self, session_id) -> None:
        """
        UI focused a different chat thread while idle: hint the local server to drop reuse
        of the previous thread's prompt/KV state before the user sends another message.
        """
        if not self._uses_external_http():
            return
        if not self._is_local_llm_service():
            return
        if self.isRunning():
            return
        last = self._last_completed_llm_session_id
        if not session_id or last is None or last == session_id:
            return
        cleared = self._server_kv_cleared_for_session_id
        if cleared == session_id:
            return
        self._flush_server_kv_hint()
        self._server_kv_cleared_for_session_id = session_id

    def _ensure_cross_session_server_flush(self) -> None:
        """Before building the next completion, flush if this turn targets a different DB session."""
        if not self._uses_external_http():
            return
        if not self._is_local_llm_service():
            return
        sid = self.session_id
        last = self._last_completed_llm_session_id
        if not sid or last is None or last == sid:
            return
        if self._server_kv_cleared_for_session_id == sid:
            return
        self._flush_server_kv_hint()
        self._server_kv_cleared_for_session_id = sid

    # ============================================================
    # RETRIEVAL BUDGET ENFORCER
    # ============================================================
    def _enforce_retrieval_budget(self, memory_context: str, rag_context: str):

        def trim(t, limit):
            return t[:limit] if t else ""

        memory_context = trim(memory_context, self.MEMORY_BUDGET)

        remaining = self.MAX_TOTAL_RETRIEVAL_CHARS - len(memory_context)
        remaining = max(0, remaining)

        rag_context = trim(rag_context, min(self.RAG_BUDGET, remaining))

        return memory_context, rag_context

    def _apply_sequential_source_ids(self, sources: list, execution_route: str) -> None:
        """Assign globally unique citation ids (1..n) in merge order: memory → RAG → web."""
        if not sources:
            return
        for i, src in enumerate(sources, start=1):
            if isinstance(src, dict):
                src["id"] = i

    def _finalize_citation_integrity_text(
        self,
        text: str,
        sources: list,
        *,
        phase: str = "worker_finalize",
    ) -> str:
        """Log citation integrity; optionally strip orphan tokens before persist/UI."""
        src_list = list(sources or [])
        report = analyze_citations(text or "", src_list)
        try:
            log_citation_integrity(
                report,
                phase=phase,
                execution_route=str(getattr(self, "_turn_execution_route", "") or ""),
                session_id=str(self.session_id or ""),
            )
        except Exception:
            logger.debug("[CitationIntegrity] telemetry failed", exc_info=True)

        if not get_citation_integrity_enforce():
            return text or ""

        repaired, post = repair_orphan_citations(
            text or "",
            src_list,
            mode="strip",
        )
        if report.has_violation:
            try:
                log_citation_integrity_repair(
                    session_id=str(self.session_id or ""),
                    execution_route=str(getattr(self, "_turn_execution_route", "") or ""),
                    orphan_ids=report.orphan_ids,
                    mode="strip",
                    chars_before=len(text or ""),
                    chars_after=len(repaired),
                )
            except Exception:
                logger.debug("[CitationIntegrity] repair telemetry failed", exc_info=True)
        return repaired

    def _apply_cited_only_citation_renumber(
        self, final_text: str, all_ui_sources: list
    ) -> tuple[str, list]:
        """Filter to cited sources only; renumber [1..n] by first appearance in the answer."""
        before = len(all_ui_sources or [])
        new_text, new_sources = renumber_citations_by_appearance(
            final_text or "",
            list(all_ui_sources or []),
        )
        if before or new_sources:
            logger.info(
                "[CitationRenumber] cited-only appearance order: sources %d -> %d",
                before,
                len(new_sources),
            )
        return new_text, new_sources

    def _sync_turn_citation_sources(self, all_ui_sources: list) -> None:
        """Persist isolated source snapshot for UI/telemetry and refresh inline cite links."""
        self._turn_all_ui_sources = copy.deepcopy(all_ui_sources or [])
        self.sources_found.emit(
            self.session_id or "",
            copy.deepcopy(all_ui_sources or []),
        )
        evidence_bundle = getattr(self, "_turn_evidence_bundle", None)
        if evidence_bundle is not None:
            from core.knowledge.evidence_transparency import build_evidence_transparency

            transparency = build_evidence_transparency(evidence_bundle)
            if research_map_enabled():
                from core.knowledge.graph.transparency import (
                    enrich_transparency_with_prior_sessions,
                )

                transparency = enrich_transparency_with_prior_sessions(
                    transparency,
                    db=self.db,
                    session_id=self.session_id,
                    bundle=evidence_bundle,
                )
            self._turn_evidence_transparency = transparency
            self.evidence_transparency_found.emit(
                self.session_id or "",
                copy.deepcopy(transparency),
            )

    def _finalize_turn_citations(
        self,
        final_text: str,
        all_ui_sources: list,
        *,
        messages: list[dict] | None = None,
        execution_route: str = "",
        allow_missing_retry: bool = False,
    ) -> tuple[str, list, bool]:
        """Citation retry (optional), then cited-only renumber + UI source sync."""
        retry_replaced = False
        before_text = final_text or ""
        if (
            allow_missing_retry
            and (final_text or "").strip()
            and all_ui_sources
            and messages is not None
        ):
            final_text, retry_replaced = self._maybe_retry_missing_web_citations(
                final_text,
                all_ui_sources,
                messages,
                execution_route=execution_route,
            )
        final_text, all_ui_sources = self._apply_cited_only_citation_renumber(
            final_text,
            all_ui_sources,
        )
        self._maybe_finalize_capability_cited_trace(final_text, all_ui_sources)
        self._sync_turn_citation_sources(all_ui_sources)
        if (
            (final_text or "").strip() != (before_text or "").strip()
            and (final_text or "").strip()
            and not retry_replaced
        ):
            self.stream_replaced.emit(self.session_id or "", final_text)
        return final_text, all_ui_sources, retry_replaced

    def _maybe_finalize_capability_cited_trace(
        self,
        final_text: str,
        all_ui_sources: list,
    ) -> None:
        ctx = getattr(self, "_turn_capability_trace_ctx", None)
        if ctx is None:
            return
        try:
            from core.app_settings import get_retrieval_profile
            from core.integrations.capability_trace import finalize_capability_cited_trace

            updated = finalize_capability_cited_trace(
                ctx,
                final_text=final_text,
                all_ui_sources=all_ui_sources,
                db=self.db,
                retrieval_profile=get_retrieval_profile(),
            )
            if updated:
                logger.debug(
                    "[LLM Worker] Capability cited step appended (%d steps)",
                    len(updated),
                )
        except Exception:
            logger.debug(
                "[LLM Worker] Capability cited trace finalize failed",
                exc_info=True,
            )

    def _maybe_retry_missing_web_citations(
        self,
        text: str,
        sources: list,
        messages: list[dict],
        *,
        execution_route: str = "",
    ) -> tuple[str, bool]:
        """One-shot citation fixup retry for WEB turns (opt-in via settings)."""
        if not get_citation_integrity_missing_retry():
            return text or "", False
        if str(execution_route or "").upper() != "WEB":
            return text or "", False
        src_list = list(sources or [])
        if not src_list or not (text or "").strip():
            return text or "", False

        native_engine = getattr(self, "_native_engine", None)
        if native_engine is None:
            return text or "", False

        before = text or ""
        outcome = maybe_retry_missing_web_citations(
            native_engine,
            messages,
            before,
            src_list,
        )
        if not outcome.retry_used or outcome.text == before:
            return before, False

        try:
            log_citation_integrity_repair(
                session_id=str(self.session_id or ""),
                execution_route=str(execution_route or ""),
                mode="missing_retry",
                chars_before=len(before),
                chars_after=len(outcome.text),
                retry_reason=outcome.retry_reason,
            )
        except Exception:
            logger.debug("[CitationIntegrity] missing retry telemetry failed", exc_info=True)

        self.stream_replaced.emit(self.session_id or "", outcome.text)
        self.tts_turn_superseded.emit(self.session_id or "")
        return outcome.text, True

    def _append_help_canonical_action_if_needed(self, final_text: str) -> str:
        if getattr(self, "_composer_knowledge_tool", None) != "help":
            return final_text or ""
        entry = getattr(self, "_turn_canonical_help_entry", None)
        if not isinstance(entry, dict):
            return final_text or ""
        return append_canonical_action_block(final_text or "", entry)

    def _record_memory_citations(self, final_text: str, sources: list) -> None:
        """Phase C: scan ``final_text`` for ``[N]`` cites and credit the
        corresponding memory rows.

        Only memory-type sources are credited (web/rag don't need usage
        tracking). The actual disk write is deferred to EnrichmentWorker
        which drains the recorder queue.
        """
        if not final_text or not sources:
            return
        try:
            cited_ids: set[int] = set()
            for m in re.finditer(r"\[(\d+)\]", final_text):
                try:
                    cited_ids.add(int(m.group(1)))
                except Exception:
                    continue
            if not cited_ids:
                return
            recorder = get_memory_usage_recorder()
            for src in sources:
                if not isinstance(src, dict):
                    continue
                if str(src.get("type", "")).lower() != "memory":
                    continue
                cid_id = src.get("id")
                if cid_id in cited_ids:
                    mid = src.get("memory_id")
                    if mid:
                        recorder.record_cited(str(mid))
        except Exception:
            logger.exception("[LLM] memory citation scan failed")

    # ============================================================
    # T3.3: per-turn enrichment skip / mode plumbing.
    #
    # ``_turn_enrichment_mode`` is one of:
    #   - "full"          : run the normal EnrichmentWorker extraction.
    #   - "explicit_only" : skip the extractor LLM call but still let the
    #                       explicit-remember bypass seed its knowledge fact
    #                       (the user's own message is clean even on a
    #                       broken assistant response).
    #   - "skip"          : short-circuit enrichment entirely for this turn.
    #
    # ``_turn_skip_enrichment_reason`` is a short diagnostic string used
    # only for INFO-level logging on the EnrichmentWorker side.
    # ============================================================
    def _reset_turn_enrichment_flags(self) -> None:
        self._turn_enrichment_mode: str = "full"
        self._turn_skip_enrichment_reason: str | None = None

    def _mark_skip_enrichment(self, reason: str) -> None:
        """Mark this turn as ``skip`` enrichment, unless an explicit-remember
        turn has already claimed it (in which case the bypass must still run,
        but we record the secondary cause in the reason for diagnostics).
        """
        if not reason:
            return
        current_mode = getattr(self, "_turn_enrichment_mode", "full")
        if current_mode == "explicit_only":
            if not getattr(self, "_turn_skip_enrichment_reason", None):
                self._turn_skip_enrichment_reason = reason
            return
        self._turn_enrichment_mode = "skip"
        if not getattr(self, "_turn_skip_enrichment_reason", None):
            self._turn_skip_enrichment_reason = reason

    def _mark_explicit_remember_mode(self, reason: str = "explicit_remember_write_only") -> None:
        self._turn_enrichment_mode = "explicit_only"
        self._turn_skip_enrichment_reason = reason

    def _ensure_router_centroids(self) -> None:
        if not getattr(self, "cognitive_router", None):
            return
        embedder = getattr(self.embedding_cache, "embedder", None)
        if embedder is None:
            return
        try:
            from core.router_centroid_install import install_router_centroids

            install_router_centroids(self.cognitive_router, embedder)
        except Exception:
            logger.exception("[LLM Worker] Failed to build router centroids")

    # T4.2: keep the old name as a back-compat alias so any existing
    # call site (e.g. ``_execute_llm_turn``) keeps working without
    # edits, and so out-of-tree callers don't break.
    _ensure_recall_centroid = _ensure_router_centroids

    def _format_sources_for_llm_prompt(
        self,
        sources: list,
        *,
        format_mode: str = "grounded",
    ) -> str:
        """Single numbered block list so [1], [2], … align with UI / DB (no per-tool duplicate ids).

        Thin memory stubs (short memory entries whose content is essentially a
        bare name or < 3 informative words) are annotated when at least one
        non-memory source exists in the same block, so the LLM knows to prefer
        the richer document / web source for detail on "tell me about X"
        style queries.
        """
        background = str(format_mode or "grounded").lower() == "background"
        has_non_memory = any(
            isinstance(s, dict) and str(s.get("type", "")).lower() not in ("memory", "")
            for s in sources
        )

        parts = []
        for src in sources:
            if not isinstance(src, dict):
                continue
            sid = src.get("id")
            name = str(src.get("filename", "Unknown"))
            body = (src.get("content") or "").strip()

            src_type = str(src.get("type", "")).lower()
            if (
                has_non_memory
                and src_type == "memory"
                and is_thin_content(body)
            ):
                name = f"{name} (short memory stub; prefer documents for detail)"

            if background and src_type == "memory":
                parts.append(f"--- Known user context: {name} ---\n{body}")
            else:
                cite_tag = f"[{sid}]" if sid is not None else "[?]"
                parts.append(f"--- {cite_tag}: {name} ---\n{body}")
        return "\n\n".join(parts)

    def _stamp_discourse_on_decision(
        self,
        decision: dict,
        *,
        follow_up: FollowUpClassification,
        discourse_state: DiscourseState | None,
        routing_query: str,
        retrieval_query: str,
        core_memory_suppressed: bool,
        retrieval_wrapper_mode: str,
        inference_user_text: str = "",
        resolved_query: ResolvedUserQuery | None = None,
        prompt_rewrite: DiscoursePromptRewrite | None = None,
        resolved_retrieval: ResolvedRetrievalQuery | None = None,
    ) -> None:
        if not isinstance(decision, dict):
            return
        decision.update(follow_up.to_dict())
        if discourse_state is not None:
            decision.update(discourse_state.to_dict())
        original = (self.prompt or "").strip()
        inference = (inference_user_text or original).strip()
        if inference != original:
            decision["inference_user_text"] = inference
        if resolved_query is not None and resolved_query.succeeded:
            decision["discourse_query_resolved"] = True
            decision["discourse_rewrite_reason"] = resolved_query.rewrite_reason
            decision["discourse_rewrite_confidence"] = round(
                resolved_query.confidence, 3
            )
        if prompt_rewrite is not None:
            decision.update(prompt_rewrite.trace_fields())
        if routing_query != original:
            decision["routing_query"] = routing_query
        if retrieval_query != original:
            decision["retrieval_query"] = retrieval_query
        if resolved_retrieval is not None:
            decision.update(resolved_retrieval.to_telemetry_dict())
        decision["core_memory_suppressed"] = bool(core_memory_suppressed)
        decision["retrieval_wrapper_mode"] = retrieval_wrapper_mode

    def _stamp_query_expansion_on_decision(
        self,
        decision: dict,
        *,
        original_query: str,
        retrieval_query: str,
        expansion: QueryExpansion | None,
    ) -> None:
        if not isinstance(decision, dict):
            return
        decision["original_query"] = original_query
        if expansion is not None:
            decision["expanded_query"] = expansion.expanded_query
            decision["query_expansion_confidence"] = round(expansion.confidence, 3)
            decision["query_expansion_source"] = expansion.topic_source
            decision["sidecar_rewrite_applied"] = True
            if expansion.recommended_target:
                decision["sidecar_recommended_target"] = expansion.recommended_target
        elif retrieval_query != original_query:
            decision["sidecar_rewrite_applied"] = False

    def _memory_search_hybrid(
        self,
        query: str,
        query_vector,
        expansion: QueryExpansion | None,
        **kwargs,
    ) -> dict:
        primary = memory_search(query, query_vector, self.store, **kwargs)
        if expansion is None:
            return primary
        expanded = (expansion.expanded_query or "").strip()
        if not expanded or expanded.lower() == (query or "").strip().lower():
            return primary
        try:
            exp_vector = self.embedding_cache.get_embedding(expanded)
        except Exception as e:
            logger.debug("[Sidecar] expanded memory embedding failed: %s", e)
            return primary
        auxiliary = memory_search(expanded, exp_vector, self.store, **kwargs)
        merged = merge_memory_search_results(primary, auxiliary)
        p_n = len(primary.get("memory_sources") or [])
        m_n = len(merged.get("memory_sources") or [])
        self._sidecar_hybrid_extra_memory = max(0, m_n - p_n)
        return merged

    def _rag_search_hybrid(
        self,
        query: str,
        query_vector,
        expansion: QueryExpansion | None,
        **kwargs,
    ) -> dict:
        primary = rag_search(query, query_vector, self.store, **kwargs)
        if expansion is None:
            return primary
        expanded = (expansion.expanded_query or "").strip()
        if not expanded or expanded.lower() == (query or "").strip().lower():
            return primary
        try:
            exp_vector = self.embedding_cache.get_embedding(expanded)
        except Exception as e:
            logger.debug("[Sidecar] expanded RAG embedding failed: %s", e)
            return primary
        auxiliary = rag_search(expanded, exp_vector, self.store, **kwargs)
        merged = merge_rag_search_results(primary, auxiliary)
        p_n = len(primary.get("sources") or [])
        m_n = len(merged.get("sources") or [])
        self._sidecar_hybrid_extra_rag = max(0, m_n - p_n)
        return merged

    def _log_discourse_debug(
        self,
        *,
        follow_up: FollowUpClassification,
        discourse_state: DiscourseState | None,
        roles: list[str],
        history_chars: int,
        retrieval_chars: int,
        query_chars: int,
        retrieval_wrapper_mode: str,
        core_memory_suppressed: bool,
    ) -> None:
        if not discourse_debug_enabled():
            return
        topic = discourse_state.active_topic if discourse_state else None
        referent = discourse_state.active_referent if discourse_state else None
        ref_source = (
            discourse_state.referent_source if discourse_state else "none"
        )
        ref_conf = (
            discourse_state.referent_confidence if discourse_state else 0.0
        )
        logger.info(
            "[Discourse] follow_up=%s conf=%.2f topic=%r referent=%r "
            "source=%s ref_conf=%.2f wrapper=%s core_memory_suppressed=%s "
            "roles=%s hist_chars=%d retrieval_chars=%d query_chars=%d",
            follow_up.kind.value,
            follow_up.confidence,
            topic,
            referent,
            ref_source,
            ref_conf,
            retrieval_wrapper_mode,
            core_memory_suppressed,
            roles,
            history_chars,
            retrieval_chars,
            query_chars,
        )

    def _promote_discourse_after_assistant(self, final_text: str) -> None:
        if not get_discourse_grounding_enabled() or not self.session_id:
            return
        text = (final_text or "").strip()
        if not text:
            return
        sid = str(self.session_id)
        prior = self._discourse_by_session.get(sid)
        promoted = promote_referent_after_assistant(
            user_prompt=str(self.prompt or ""),
            assistant_text=text,
            prior=prior,
        )
        if promoted.active_referent and (
            prior is None
            or promoted.active_referent != prior.active_referent
            or promoted.referent_source != prior.referent_source
        ):
            log_discourse_referent_trace(
                referent=promoted.active_referent,
                referent_source=promoted.referent_source,
                referent_confidence=promoted.referent_confidence,
                user_prompt_preview=str(self.prompt or ""),
                assistant_preview=text,
            )
        self._discourse_by_session[sid] = promoted

    def _persist_assistant_turn(self, final_text: str, all_ui_sources: list) -> None:
        """Persist assistant output to SQLite with poisoned-history protection."""
        if not self.session_id or not (final_text or "").strip():
            return
        if is_native_empty_visible_output_notice(final_text):
            self._mark_skip_enrichment("empty_visible_output")
            return

        history_content, degeneration = resolve_assistant_history_content(
            final_text,
            stream_cancelled=bool(
                getattr(self, "_turn_stream_degeneration_cancelled", False)
            ),
        )
        sources = list(all_ui_sources or [])
        history_content = self._finalize_citation_integrity_text(
            history_content,
            sources,
            phase="worker_persist",
        )
        self._turn_history_degeneration = degeneration
        self._turn_stored_history_content = history_content

        if degeneration.output_degeneration is not None:
            log_output_degeneration(
                session_id=str(self.session_id or ""),
                result=degeneration.output_degeneration,
                phase="persist",
            )

        src_payload = json.dumps(all_ui_sources) if all_ui_sources else None
        bundle_id = None
        evidence_bundle = getattr(self, "_turn_evidence_bundle", None)
        transparency = None
        if evidence_bundle is not None:
            bundle_id = str(getattr(evidence_bundle, "bundle_id", "") or "") or None
            from core.knowledge.evidence_transparency import build_evidence_transparency
            from core.knowledge.ui_sources_payload import encode_sources_payload

            transparency = build_evidence_transparency(evidence_bundle)
            if research_map_enabled():
                from core.knowledge.graph.transparency import (
                    enrich_transparency_with_prior_sessions,
                )

                transparency = enrich_transparency_with_prior_sessions(
                    transparency,
                    db=self.db,
                    session_id=self.session_id,
                    bundle=evidence_bundle,
                )
            src_payload = encode_sources_payload(
                list(all_ui_sources or []),
                transparency=transparency,
            )
        elif all_ui_sources:
            src_payload = json.dumps(all_ui_sources)
        self._turn_last_assistant_msg_id = self.db.add_message(
            self.session_id,
            "assistant",
            history_content,
            sources_json=src_payload,
            evidence_bundle_id=bundle_id,
        )
        if (
            research_map_enabled()
            and evidence_bundle is not None
            and getattr(self, "_turn_last_assistant_msg_id", None)
        ):
            from core.knowledge.graph.service import record_bundle_in_session_graph

            record_bundle_in_session_graph(
                self.db,
                session_id=str(self.session_id or ""),
                bundle=evidence_bundle,
                message_id=self._turn_last_assistant_msg_id,
            )

        gen_debug = getattr(self, "_active_generation_debug_recorder", None)
        if isinstance(gen_debug, GenerationDebugRecorder):
            gen_debug.record_final_stored(history_content, ui_final=final_text)
            self._active_generation_debug_recorder = None

        try:
            updated = self.routing_debug_buffer.merge_history_degeneration_into_latest(
                degeneration.trace_fields()
            )
            if updated is not None:
                self.routing_debug_record_added.emit(dataclasses.asdict(updated))
                self._persist_routing_debug_record(updated)
        except Exception:
            logger.debug(
                "[HistoryDegeneration] failed to merge routing debug record",
                exc_info=True,
            )

        if degeneration.should_suppress:
            self._mark_skip_enrichment("history_degeneration_suppressed")
            od_fields = (
                degeneration.output_degeneration.trace_fields()
                if degeneration.output_degeneration is not None
                else {}
            )
            log_history_degeneration_suppression(
                session_id=str(self.session_id or ""),
                score=degeneration.score,
                flags=degeneration.flags,
                presented_preview=final_text,
                stored_content=history_content,
                output_degeneration=od_fields,
                stream_cancelled=bool(
                    getattr(self, "_turn_stream_degeneration_cancelled", False)
                ),
                suppression_reason=history_suppression_reason(
                    degeneration,
                    stream_cancelled=bool(
                        getattr(self, "_turn_stream_degeneration_cancelled", False)
                    ),
                ),
            )
            logger.warning(
                "[HistoryDegeneration] suppressed assistant turn score=%.2f flags=%s",
                degeneration.score,
                ",".join(degeneration.flags),
            )
            return

        self._record_memory_citations(final_text, all_ui_sources)
        self._promote_discourse_after_assistant(final_text)

    def _memory_query_fingerprint(
        self,
        query: str,
        *,
        include_preference: bool,
        include_knowledge: bool,
        include_episode: bool,
        include_context: bool,
    ) -> str:
        return compute_query_fingerprint(
            query,
            include_preference=include_preference,
            include_knowledge=include_knowledge,
            include_episode=include_episode,
            include_context=include_context,
        )

    def _bound_session_history(self, history: list[dict]) -> list[dict]:
        """
        Cull session messages for the completion request so the inference server's KV cache
        does not grow without bound on long threads. Window size is user-controlled via
        max_history_messages; single-message truncation remains as a safety cap.
        """
        if not history:
            return []

        max_single = self.CHAT_HISTORY_SINGLE_MESSAGE_MAX_CHARS
        suffix = "\n\n[…message truncated for context window]"

        capped: list[dict] = []
        for m in history:
            role = m.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "user"
            content = m.get("content") or ""
            if len(content) > max_single:
                content = content[: max_single - len(suffix)] + suffix
            capped.append({"role": role, "content": content})

        n_before = len(capped)
        limit = max(2, min(100, int(getattr(self, "max_history_messages", 10))))
        windowed = capped[-limit:] if len(capped) > limit else capped
        # Jinja/Mistral chat templates expect a user turn before assistant; dropping the
        # leading user when windowing leaves assistant-first history and breaks reconstruction.
        while windowed and windowed[0].get("role") == "assistant":
            windowed = windowed[1:]

        if n_before > len(windowed):
            logger.info(
                "[LLM] Chat history windowed: using last %d of %d messages (max_history_messages=%d)",
                len(windowed),
                n_before,
                limit,
            )
            if get_enable_memory_v7_salvage():
                windowed_ids = {m.get("id") for m in windowed if m.get("id")}
                dropped_ids: list[str] = []
                for m in capped:
                    mid = m.get("id")
                    if mid and mid not in windowed_ids:
                        dropped_ids.append(str(mid))
                self._pending_salvage_message_ids = dropped_ids[:24]

        return windowed

    # ============================================================
    def clean_text_for_tts(self, text):
        import re
        text = re.sub(r'[*_]{1,3}', '', text)
        text = re.sub(r'#+\s+', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Strip HTML and Citations (for RAG/Web)
        text = re.sub(r'<[^>]+>', '', text) 
        text = re.sub(r'\[(\d+|W)\]', '', text)
        text = re.sub(
            r"\[\s*format\s+fallback\s+applied\s*\]",
            "",
            text,
            flags=re.IGNORECASE,
        )

        cleaned = text.strip()
        
        # 🔑 THE ULTIMATE FAILSAFE: 
        # If the string contains no letters or numbers (e.g., it's just a ".", "!", or empty), kill it.
        if not re.search(r'[a-zA-Z0-9]', cleaned):
            return ""
            
        return cleaned

    def _reset_tts_dedupe_state(self) -> None:
        self._tts_dedupe_keys: set[str] = set()

    def _normalize_tts_key(self, text: str) -> str:
        cleaned = self.clean_text_for_tts(text)
        return re.sub(r"\s+", " ", cleaned).strip().lower()

    def _queue_tts_sentence(self, raw: str, *, force: bool = False) -> None:
        if bool(getattr(self, "_cancel_requested", False)):
            return
        cleaned = self.clean_text_for_tts(raw)
        if not cleaned:
            return
        key = self._normalize_tts_key(cleaned)
        if not key:
            return
        keys = getattr(self, "_tts_dedupe_keys", None)
        if keys is None:
            self._reset_tts_dedupe_state()
            keys = self._tts_dedupe_keys
        if not force and key in keys:
            return
        keys.add(key)
        self.sentence_ready.emit(cleaned, self.session_id or "")

    def _estimate_output_tokens(self, text: str) -> int:
        """Approximate output token count for UX telemetry (non-billing metric)."""
        return len(re.findall(r"\S+", (text or "").strip()))

    def _emit_output_tps(self, token_count: int, first_token_ts: float | None) -> None:
        if token_count <= 0 or first_token_ts is None:
            self.tps_metric.emit(0.0)
            return
        elapsed = max(0.001, time.time() - float(first_token_ts))
        self.tps_metric.emit(float(token_count) / elapsed)

    def _truth_diff_context(self) -> dict:
        model_name = ""
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal":
            path = resolve_internal_model_path(get_internal_model_path() or "")
            model_name = os.path.basename(path) if path else ""
        else:
            model_name = str(getattr(self, "api_url", "") or "external")
        exchange_id = getattr(self, "_debug_exchange_id", None)
        return {
            "request_id": exchange_id,
            "exchange_id": exchange_id,
            "session_id": str(self.session_id or ""),
            "model_name": model_name,
        }

    def _frontend_raw_request(self) -> dict:
        attachments = getattr(self, "_turn_attachments", None) or []
        return {
            "prompt": self.prompt or "",
            "persist_content": getattr(self, "_persist_content", None) or "",
            "session_id": self.session_id or "",
            "attachments": attachments,
            "engine_mode": str(getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) or ""),
        }

    def _truth_diff_l2_metadata(
        self,
        *,
        template_source: str,
        chat_format_mode: str,
        execution_mode: str,
        prompt_contract_mode: str,
        **extra: Any,
    ) -> dict:
        return {
            **self._truth_diff_context(),
            "template_source": template_source,
            "chat_format_mode": chat_format_mode,
            "execution_mode": execution_mode,
            "prompt_contract_mode": prompt_contract_mode,
            **extra,
        }

    def _safe_truth_diff_l1_raw_request(self) -> None:
        if not llm_truth_diff_enabled():
            return
        try:
            get_llm_truth_diff_logger().log_l1_raw_request(
                self._frontend_raw_request(),
                self._truth_diff_context(),
            )
        except Exception:
            logger.debug("[LLMTruthDiff] L1 raw request log failed", exc_info=True)

    def _reset_turn_trace_capture_state(self) -> None:
        self._turn_engine_request = None
        self._turn_rendered_prompt = ""
        self._turn_history_degeneration = None
        self._turn_prompt_rewrite = None
        self._turn_prior_turn_suppressed = False
        self._turn_stored_history_content = ""
        self._turn_context: TurnContext | None = None
        self._turn_conversation_health: ConversationHealthState | None = None
        self._turn_stream_degeneration_cancelled = False
        self._turn_capability_trace_ctx = None

    def _remember_turn_engine_request(self, request: dict) -> None:
        self._turn_engine_request = copy.deepcopy(request or {})

    def _remember_turn_rendered_prompt(self, prompt: str) -> None:
        self._turn_rendered_prompt = str(prompt or "")

    def _conversation_health_for_session(self) -> ConversationHealthState:
        sid = str(self.session_id or "")
        if not sid:
            return initial_conversation_health()
        return self._conversation_health_by_session.get(sid) or initial_conversation_health()

    def _mark_stream_degeneration_cancelled(self) -> None:
        self._turn_stream_degeneration_cancelled = True

    def _finalize_conversation_health_after_turn(self, final_text: str) -> None:
        sid = str(self.session_id or "")
        if not sid:
            return
        before = self._conversation_health_for_session()
        degeneration = getattr(self, "_turn_history_degeneration", None)
        deg_risk = "LOW"
        history_suppressed = False
        if degeneration is not None:
            history_suppressed = bool(degeneration.should_suppress)
            od = degeneration.output_degeneration
            if od is not None:
                deg_risk = str(od.risk)
            elif degeneration.score >= 0.55:
                deg_risk = "HIGH"
            elif degeneration.score >= 0.35:
                deg_risk = "MEDIUM"
        elif (final_text or "").strip():
            from core.output_degeneration import detect_output_degeneration

            deg_risk = str(detect_output_degeneration(final_text).risk)

        collapse_risk = "LOW"
        stream_cancelled = bool(
            getattr(self, "_turn_stream_degeneration_cancelled", False)
        )
        self_inflicted = stream_cancelled and not history_suppressed
        if not self_inflicted:
            if history_suppressed or deg_risk == "HIGH":
                collapse_risk = "HIGH"
            elif deg_risk == "MEDIUM":
                collapse_risk = "MEDIUM"

        outcome = TurnAnomalyOutcome(
            degeneration_risk=deg_risk,
            history_suppressed=history_suppressed,
            collapse_risk=collapse_risk,
            stream_degeneration_cancelled=bool(
                getattr(self, "_turn_stream_degeneration_cancelled", False)
            ),
        )
        after = update_conversation_health(before, outcome=outcome)
        self._conversation_health_by_session[sid] = after
        self._turn_conversation_health = after
        try:
            log_conversation_health_update(
                session_id=sid,
                before=before,
                after=after,
                outcome=outcome,
            )
        except Exception:
            logger.debug("[ConversationHealth] structured log failed", exc_info=True)
        if before.mode != after.mode or outcome.had_anomaly:
            logger.info(
                "[ConversationHealth] session=%s health %.2f→%.2f mode %s→%s penalty=%.2f",
                sid[:8],
                before.health_score,
                after.health_score,
                before.mode,
                after.mode,
                outcome.anomaly_penalty(),
            )

    def _harmony_model_active(self) -> bool:
        """True when the loaded internal model uses Harmony protocol layers."""
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) != "internal" or not self._native_engine:
            return False
        contract = getattr(self._native_engine, "_last_prompt_contract", None)
        if is_harmony_contract(contract):
            return True
        return bool(getattr(self._native_engine, "_harmony_model_active", False))

    def _resolve_turn_context_for_turn(
        self,
        *,
        execution_route: str,
        follow_up: FollowUpClassification,
        prior_turn_unreliable: bool,
        has_retrieval_sources: bool,
        history_turn_count: int,
        conversation_health: ConversationHealthState | None = None,
    ) -> TurnContext:
        turn_ctx = resolve_turn_context(
            execution_route=execution_route,
            user_query=str(self.prompt or ""),
            follow_up=follow_up,
            prior_turn_unreliable=prior_turn_unreliable,
            has_retrieval_sources=has_retrieval_sources,
            history_turn_count=history_turn_count,
            use_harmony_protocol=self._harmony_model_active(),
            conversation_health=conversation_health,
        )
        self._turn_context = turn_ctx
        try:
            structured_llm_log(
                "turn_context",
                {
                    "session_id": str(self.session_id or ""),
                    **turn_ctx.trace_fields(),
                },
            )
        except Exception:
            logger.debug("[TurnContext] structured log failed", exc_info=True)
        logger.info(
            "[TurnContext] format=%s intent=%s risk=%s history=%s health=%s conflicts=%s",
            turn_ctx.chat_format_mode,
            turn_ctx.reply_shape.format_intent,
            turn_ctx.generation_risk.risk_tier,
            turn_ctx.history_strategy,
            (
                turn_ctx.conversation_health.mode
                if turn_ctx.conversation_health is not None
                else "normal"
            ),
            list(turn_ctx.reply_shape.instruction_conflicts),
        )
        return turn_ctx

    def _generation_params_from_turn_context(
        self,
        turn_ctx: TurnContext,
        *,
        native: bool,
    ) -> dict[str, Any]:
        risk = turn_ctx.generation_risk
        base_max = (
            self._max_tokens_completion(native=native)
        )
        sampling = self._sampling_payload()
        sampling["repeat_penalty"] = risk.effective_repeat_penalty(
            float(sampling.get("repeat_penalty", self.repeat_penalty))
        )
        return {
            "temperature": risk.effective_temperature(self.temperature),
            "max_tokens": risk.effective_max_tokens(base_max),
            "sampling_overrides": sampling,
        }

    def build_last_turn_canonical_trace(
        self,
        *,
        output: str,
        extra_metadata: dict | None = None,
    ):
        """Build CanonicalTrace from the most recently completed turn capture state."""
        engine_req = getattr(self, "_turn_engine_request", None)
        if not engine_req:
            return None
        try:
            snap = getattr(self, "_completion_output_snapshot", None)
            engine_mode = (
                str(getattr(snap, "engine_mode", "") or "")
                if snap
                else str(getattr(self, "engine_mode", "") or "")
            )
            from core.conversation_replay import qube_execution_path_for_engine_mode

            metadata = {
                **self._truth_diff_context(),
                "engine_mode": engine_mode,
                "execution_path": qube_execution_path_for_engine_mode(engine_mode),
                "execution_route": str(getattr(self, "_turn_execution_route", "") or ""),
                **(extra_metadata or {}),
            }
            degeneration = getattr(self, "_turn_history_degeneration", None)
            if degeneration is not None:
                metadata.update(degeneration.trace_fields())
            prompt_rewrite = getattr(self, "_turn_prompt_rewrite", None)
            rewrite_confidence = (
                float(prompt_rewrite.rewrite_confidence) if prompt_rewrite else 0.0
            )
            turn_sources = list(getattr(self, "_turn_all_ui_sources", None) or [])
            collapse = compute_collapse_diagnostics(
                prompt=str(getattr(self, "_turn_rendered_prompt", "") or ""),
                output=str(output or ""),
                user_query=str(self.prompt or ""),
                rewrite_confidence=rewrite_confidence,
                degeneration_score=(
                    float(degeneration.score) if degeneration is not None else None
                ),
                degeneration_flags=degeneration.flags if degeneration else (),
                turn_index=int(getattr(self, "_routing_debug_turn_seq", 0)),
                prior_turn_suppressed=bool(
                    getattr(self, "_turn_prior_turn_suppressed", False)
                ),
                active_referent=(
                    str(
                        getattr(
                            self._discourse_by_session.get(str(self.session_id or "")),
                            "active_referent",
                            "",
                        )
                        or ""
                    )
                ),
                valid_source_ids=citation_valid_source_ids(turn_sources),
            )
            metadata.update(collapse.trace_fields())
            log_collapse_diagnostics(
                session_id=str(self.session_id or ""),
                turn_index=collapse.turn_index,
                collapse_risk=collapse.collapse_risk,
                collapse_score=collapse.collapse_score,
                prompt_length=collapse.prompt_length,
                output_length=collapse.output_length,
                rewrite_confidence=collapse.rewrite_confidence,
                degeneration_score=collapse.degeneration_score,
                hallucination_score=collapse.hallucination_score,
                format_drift_score=collapse.format_drift_score,
                hallucination_flags=collapse.hallucination_flags,
                format_drift_flags=collapse.format_drift_flags,
                prior_turn_suppressed=collapse.prior_turn_suppressed,
            )
            if prompt_rewrite is not None:
                metadata.update(prompt_rewrite.trace_fields())
            turn_ctx = getattr(self, "_turn_context", None)
            if turn_ctx is not None:
                metadata.update(turn_ctx.trace_fields())
            health_after = getattr(self, "_turn_conversation_health", None)
            if health_after is not None:
                metadata.update(health_after.trace_fields())
            return build_golden_trace(
                request=engine_req,
                prompt=str(getattr(self, "_turn_rendered_prompt", "") or ""),
                output=str(output or ""),
                metadata=metadata,
            )
        except Exception:
            logger.debug("[GoldenTraceCapture] build_last_turn_canonical_trace failed", exc_info=True)
            return None

    def _maybe_capture_golden_trace(self, *, output: str) -> None:
        if not golden_trace_capture_mode_enabled():
            return
        try:
            trace = self.build_last_turn_canonical_trace(output=output)
            if trace is not None:
                maybe_capture_golden_trace(trace)
        except Exception:
            logger.debug("[GoldenTraceCapture] worker capture failed", exc_info=True)

    def _safe_log_canonical_request_trace(
        self,
        request: dict,
        *,
        extra_context: dict | None = None,
    ) -> None:
        if not canonical_trace_export_enabled():
            return
        try:
            ctx = {**self._truth_diff_context(), **(extra_context or {})}
            log_canonical_request_trace(request, context=ctx)
        except Exception:
            logger.debug("[CanonicalRequestTrace] log failed", exc_info=True)

    def _safe_truth_diff_l1_engine_request(self, request: dict) -> None:
        self._remember_turn_engine_request(request)
        self._safe_log_canonical_request_trace(request)
        if not llm_truth_diff_enabled():
            return
        try:
            get_llm_truth_diff_logger().log_l1_engine_request(
                request,
                self._truth_diff_context(),
            )
        except Exception:
            logger.debug("[LLMTruthDiff] L1 engine request log failed", exc_info=True)

    def _safe_truth_diff_l2_prompt(self, prompt: str, metadata: dict) -> None:
        self._remember_turn_rendered_prompt(prompt)
        if not llm_truth_diff_enabled():
            return
        try:
            get_llm_truth_diff_logger().log_l2_prompt(prompt, metadata)
        except Exception:
            logger.debug("[LLMTruthDiff] L2 prompt log failed", exc_info=True)

    def _safe_truth_diff_l3(self, *, presented_text: str) -> None:
        if not llm_truth_diff_enabled():
            return
        snap = getattr(self, "_completion_output_snapshot", None)
        if snap is None:
            return
        try:
            stages: list[str] = []
            for value in (
                snap.after_harmony_parser,
                snap.after_worker_filters,
                snap.streamed_incremental,
                snap.worker_return_text,
            ):
                text = str(value or "")
                if text and (not stages or stages[-1] != text):
                    stages.append(text)
            get_llm_truth_diff_logger().log_l3_model_io(
                raw=str(snap.raw_text or ""),
                after_stages=stages,
                final=str(presented_text or ""),
                metadata={
                    **self._truth_diff_context(),
                    "engine_mode": snap.engine_mode or "",
                    "retry_replaced": bool(snap.retry_replaced),
                },
            )
        except Exception:
            logger.debug("[LLMTruthDiff] L3 model I/O log failed", exc_info=True)

    def _native_execution_mode(self) -> str:
        engine = getattr(self, "_native_engine", None)
        if engine is None:
            return ""
        try:
            pol = engine.get_execution_policy()
            return str(getattr(pol, "execution_mode", "") or "")
        except Exception:
            return ""

    def _execution_policy_debug_payload(self) -> dict[str, Any] | None:
        if str(getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) or "").lower() != "internal":
            return None
        engine = getattr(self, "_native_engine", None)
        if engine is None:
            return None
        try:
            pol = engine.get_execution_policy()
            return execution_policy_debug_fields(
                pol,
                reasoning_mode=getattr(engine, "_last_reasoning_mode", None),
                chat_template_kwargs=getattr(engine, "_last_chat_template_kwargs", None),
            )
        except Exception:
            return None

    def _bind_truth_diff_hooks(self) -> None:
        bind_llm_worker_truth_diff_hooks(
            l1_engine_request=self._truth_diff_hook_l1_engine_request,
            l2_prompt=self._truth_diff_hook_l2_prompt,
        )

    def _truth_diff_hook_l1_engine_request(self, request: dict, context: dict) -> None:
        self._remember_turn_engine_request(request)
        self._safe_log_canonical_request_trace(request, extra_context=context)
        if not llm_truth_diff_enabled():
            return
        try:
            merged = {**self._truth_diff_context(), **(context or {})}
            get_llm_truth_diff_logger().log_l1_engine_request(request, merged)
        except Exception:
            logger.debug("[LLMTruthDiff] L1 engine hook failed", exc_info=True)

    def _truth_diff_hook_l2_prompt(self, prompt: str, metadata: dict) -> None:
        self._remember_turn_rendered_prompt(prompt)
        if not llm_truth_diff_enabled():
            return
        try:
            merged = {**self._truth_diff_context(), **(metadata or {})}
            get_llm_truth_diff_logger().log_l2_prompt(prompt, merged)
        except Exception:
            logger.debug("[LLMTruthDiff] L2 hook failed", exc_info=True)

    # ============================================================
    def generate_response(
        self,
        text: str,
        session_id: str,
        *,
        attachments: list | None = None,
        enforced_skills: tuple[str, ...] | list[str] | None = None,
        persist_content: str | None = None,
        input_source: str = "text",
    ):
        """Sets the parameters and starts the thread work."""
        from core.input_source import INPUT_SOURCE_TEXT

        source = (input_source or INPUT_SOURCE_TEXT).strip().lower() or INPUT_SOURCE_TEXT
        if source not in ("text", "voice"):
            source = INPUT_SOURCE_TEXT
        if self.isRunning():
            logger.warning(
                "[LLM] Ignoring new generate_response while previous turn is active "
                "(input_source=%s, session_id=%s).",
                source,
                session_id,
            )
            return

        self._input_source = source
        prompt = (text or "").strip()
        logger.info(
            "[LLM] generate_response accepted input_source=%s session_id=%s prompt_len=%d preview=%r",
            source,
            session_id,
            len(prompt),
            prompt[:80],
        )
        self.prompt = prompt
        self._persist_content = (persist_content or self.prompt).strip()
        self._turn_attachments = list(attachments or [])
        self._turn_enforced_skills = tuple(enforced_skills or ())
        self.session_id = session_id
        self._debug_exchange_id = next_exchange_id()
        self._safe_truth_diff_l1_raw_request()
        self.start() # This automatically triggers the run() method

    # ============================================================
    def was_last_native_job_cancelled(self) -> bool:
        return bool(getattr(self, "_last_native_job_cancelled", False))

    def generate(
        self,
        *,
        task: PrimaryEngineTask,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        debug_caller: str = "helper",
    ) -> str:
        self._last_native_job_cancelled = False
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal" and self._native_engine:
            out: list = []
            ev = threading.Event()
            self._native_engine.enqueue_simple_completion(
                messages,
                temperature,
                max_tokens,
                out,
                ev,
                task=task,
                debug_caller=debug_caller,
            )
            if not ev.wait(120):
                return ""
            result = (out[0] if out else "") or ""
            self._last_native_job_cancelled = bool(
                self._native_engine.consume_last_job_cancelled()
            )
            return result

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self._is_local_llm_service():
            payload["cache_prompt"] = False

        try:
            r = requests.post(
                self.api_url,
                json=payload,
                timeout=120,
                headers={"Connection": "close"},
            )
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            return ""

    # ============================================================
    def run(self):
        self._cancel_requested = False
        self._active_stream_response = None
        self._successfully_finished = False
        self.tps_metric.emit(0.0)
        # T3.3: reset skip/mode flags before the turn begins; _execute_llm_turn
        # re-primes them at the very top but keeping it here is belt-and-braces
        # in case an early exception fires before that method is called.
        self._reset_turn_enrichment_flags()
        self._completion_output_snapshot = None
        self._turn_execution_route = ""
        self._reset_turn_trace_capture_state()
        if getattr(self, "_debug_exchange_id", None) is None:
            self._debug_exchange_id = next_exchange_id()
        self._exchange_worker_started_at = time.monotonic()
        self._engine_enqueue_at: float | None = None
        log_chat_exchange_begin(
            exchange_id=self._debug_exchange_id,
            session_id=str(self.session_id or ""),
            user_prompt=str(getattr(self, "_persist_content", None) or self.prompt or ""),
            engine_mode=str(getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) or ""),
            execution_policy=self._execution_policy_debug_payload(),
        )
        final_text_out = ""
        try:
            final_text_out = self._execute_llm_turn()
        except Exception:
            logger.exception("[LLM] pipeline failure (routing, tools, or stream)")
            # T3.3: a pipeline-level exception means whatever assistant text we
            # have is partial / "Sorry, my brain encountered an error." — do
            # not mine it for memories.
            self._mark_skip_enrichment("pipeline_error")
            if not str(final_text_out).strip():
                final_text_out = "Sorry, my brain encountered an error."
                self.token_streamed.emit(self.session_id or "", "\n\n*(Pipeline Error)*")
        finally:
            self._close_active_stream()
            self._last_completed_llm_session_id = self.session_id
            self._server_kv_cleared_for_session_id = None
            # T3.3: cheap belt-and-suspenders — if the final assistant text
            # looks like a failure / limitation claim, skip extraction even
            # when no upstream trip condition fired.
            try:
                if (
                    getattr(self, "_turn_enrichment_mode", "full") == "full"
                    and is_assistant_failure_message(final_text_out or "")
                ):
                    self._mark_skip_enrichment("assistant_failure_final_text")
            except Exception:
                pass
            try:
                mode = getattr(self, "_turn_enrichment_mode", "full")
                reason = getattr(self, "_turn_skip_enrichment_reason", None)
                enrichment_payload = {
                    "session_id": self.session_id,
                    "last_user_msg_id": getattr(self, "_turn_last_user_msg_id", None),
                    "last_assistant_msg_id": getattr(self, "_turn_last_assistant_msg_id", None),
                    "rag_chunk_ids": list(getattr(self, "_turn_rag_chunk_ids", []) or []),
                    "skip_enrichment": mode == "skip",
                    "enrichment_mode": mode,
                    "skip_reason": reason,
                    "salvage_message_ids": list(getattr(self, "_pending_salvage_message_ids", []) or []),
                    "salvage_reason": "history_window" if getattr(self, "_pending_salvage_message_ids", None) else None,
                }
                self.enrichment_context_ready.emit(enrichment_payload)
            except Exception:
                logger.exception("[LLM] failed to emit enrichment context")
            final_text_out = strip_output_artifacts(
                final_text_out or "",
                harmony_active=self._harmony_model_active(),
                reasoning_family=self._reasoning_family_harmony_leak_strip_active(),
            )
            turn_sources = list(getattr(self, "_turn_all_ui_sources", None) or [])
            final_text_out = self._finalize_citation_integrity_text(
                final_text_out,
                turn_sources,
                phase="worker_finalize",
            )
            try:
                self._finalize_conversation_health_after_turn(final_text_out)
            except Exception:
                logger.debug("[ConversationHealth] finalize failed", exc_info=True)
            self._safe_truth_diff_l3(presented_text=final_text_out)
            self._maybe_capture_golden_trace(output=final_text_out)
            log_completion_output_trace(
                session_id=str(self.session_id or ""),
                snapshot=getattr(self, "_completion_output_snapshot", None),
                presented_text=final_text_out,
            )
            worker_prep_ms = None
            engine_queue_wait_ms = None
            engine_inference_ms = None
            exchange_total_ms = None
            enqueue_at = getattr(self, "_engine_enqueue_at", None)
            started_at = getattr(self, "_exchange_worker_started_at", None)
            if started_at is not None:
                exchange_total_ms = max(
                    0, int(round((time.monotonic() - started_at) * 1000))
                )
            if enqueue_at is not None and started_at is not None:
                worker_prep_ms = max(0, int(round((enqueue_at - started_at) * 1000)))
            native_engine = getattr(self, "_native_engine", None)
            if native_engine is not None:
                job_timing = native_engine.consume_last_job_timing()
                if job_timing is not None:
                    engine_queue_wait_ms = job_timing.queue_wait_ms
                    engine_inference_ms = job_timing.inference_ms
            log_chat_exchange_end(
                exchange_id=self._debug_exchange_id,
                session_id=str(self.session_id or ""),
                route=str(getattr(self, "_turn_execution_route", "") or ""),
                success=bool(self._successfully_finished),
                presented_text=final_text_out,
                worker_prep_ms=worker_prep_ms,
                engine_queue_wait_ms=engine_queue_wait_ms,
                engine_inference_ms=engine_inference_ms,
                exchange_total_ms=exchange_total_ms,
                execution_policy=self._execution_policy_debug_payload(),
            )
            self._completion_output_snapshot = None
            self.context_retrieved.emit(False)
            self.web_search_active.emit(False, False, False)
            self.response_finished.emit(self.session_id, final_text_out)
            if not self._successfully_finished:
                self.status_update.emit("Idle")

    def _execute_llm_turn(self) -> str:
        force_web = bool(getattr(self, "_force_web_enabled", False))

        # Phase B: reset per-turn enrichment context captured during this turn.
        self._turn_rag_chunk_ids: list[str] = []
        self._turn_last_user_msg_id = None
        self._turn_last_assistant_msg_id = None
        # T3.3: reset tool-aware enrichment skip / mode flags for this turn.
        self._reset_turn_enrichment_flags()
        self._pending_salvage_message_ids = []
        self._turn_all_ui_sources: list = []
        self._turn_evidence_bundle = None

        if self.session_id:
            user_content = getattr(self, "_persist_content", None) or self.prompt
            self._turn_last_user_msg_id = self.db.add_message(
                self.session_id, "user", user_content
            )

        self._ensure_cross_session_server_flush()

        history = self.db.get_session_history(self.session_id) if self.session_id else []
        history = self._bound_session_history(history)
        clean_prompt = self.prompt.lower().strip()

        discourse_enabled = get_discourse_grounding_enabled()
        discourse_state: DiscourseState | None = None
        follow_up = FollowUpClassification(FollowUpKind.NONE, 0.0)
        original_query = (self.prompt or "").strip()
        resolved_query: ResolvedUserQuery | None = None
        prompt_rewrite: DiscoursePromptRewrite | None = None
        salience_anchor = ""
        salience_reason = ""
        inference_user_text = original_query
        routing_query = original_query
        retrieval_query = original_query
        resolved_retrieval: ResolvedRetrievalQuery | None = None
        query_expansion: QueryExpansion | None = None
        core_memory_suppressed = False
        self._sidecar_hybrid_extra_memory = 0
        self._sidecar_hybrid_extra_rag = 0
        digest_mem_attempted = False
        digest_mem_applied = False
        digest_rag_attempted = False
        digest_rag_applied = False

        conversation_health = self._conversation_health_for_session()
        health_policy = resolve_conversation_health_policy(conversation_health)
        if health_policy.mode != "normal":
            logger.info(
                "[ConversationHealth] entering turn mode=%s health=%.2f",
                health_policy.mode,
                health_policy.health_score,
            )

        if discourse_enabled:
            prior = (
                self._discourse_by_session.get(str(self.session_id))
                if self.session_id
                else None
            )
            discourse_state = update_discourse_state(history, prior, self.prompt)
            if self.session_id:
                self._discourse_by_session[str(self.session_id)] = discourse_state
            follow_up = classify_follow_up(self.prompt, history, discourse_state)
            resolved_query = resolve_ambiguous_user_query(
                self.prompt, discourse_state, follow_up
            )
            inference_user_text = original_query
            if health_policy.allow_query_rewrite and resolved_query.succeeded:
                inference_user_text = resolved_query.resolved
                log_discourse_query_rewrite(
                    original=resolved_query.original,
                    resolved=resolved_query.resolved,
                    substitutions=resolved_query.substitutions,
                    confidence=resolved_query.confidence,
                    rewrite_reason=resolved_query.rewrite_reason,
                )
            elif resolved_query.succeeded:
                logger.info(
                    "[ConversationHealth] skipped query rewrite mode=%s",
                    health_policy.mode,
                )
            prompt_rewrite = resolve_discourse_prompt_rewrite(
                user_message=original_query,
                resolved_query=resolved_query,
                follow_up=follow_up,
                discourse=discourse_state,
                allow_rewrite=health_policy.allow_discourse_rewrite,
            )
            self._turn_prompt_rewrite = prompt_rewrite
            if (
                health_policy.allow_salience_hints
                and follow_up.active
                and discourse_state
            ):
                salience_anchor, _, salience_reason = select_salience_anchor(
                    discourse=discourse_state,
                    user_message=original_query,
                    resolved_query=resolved_query,
                )
            log_discourse_prompt_rewrite(
                original=prompt_rewrite.original,
                grounded=prompt_rewrite.grounded,
                rewrite_anchor=prompt_rewrite.rewrite_anchor or "",
                rewrite_confidence=prompt_rewrite.rewrite_confidence,
                rewrite_reason=prompt_rewrite.rewrite_reason,
                applied=prompt_rewrite.applied,
                salience_anchor=salience_anchor,
                salience_reason=salience_reason,
            )
            resolved_retrieval = build_resolved_retrieval_query(
                raw_text=original_query,
                inference_text=inference_user_text,
                follow_up=follow_up,
                discourse=discourse_state,
                history=history,
                resolved_query=resolved_query,
            )
            routing_query = resolved_retrieval.routing_text
            retrieval_query = resolved_retrieval.retrieval_text
            if follow_up.active:
                logger.info(
                    "[Discourse] follow_up=%s conf=%.2f topic=%r",
                    follow_up.kind.value,
                    follow_up.confidence,
                    discourse_state.active_topic if discourse_state else None,
                )
            elif follow_up.kind.value != "none":
                logger.info(
                    "[Discourse] follow_up=%s conf=%.2f (below suppress threshold) topic=%r",
                    follow_up.kind.value,
                    follow_up.confidence,
                    discourse_state.active_topic if discourse_state else None,
                )

        if resolved_retrieval is None:
            resolved_retrieval = build_resolved_retrieval_query(
                raw_text=original_query,
                inference_text=inference_user_text,
                follow_up=follow_up,
                discourse=discourse_state if discourse_enabled else None,
                history=history,
                resolved_query=resolved_query,
            )
            routing_query = resolved_retrieval.routing_text
            retrieval_query = resolved_retrieval.retrieval_text

        # ============================================================
        # 0. EXPLICIT-REMEMBER SHORT-CIRCUIT (Memory v6.1)
        # ------------------------------------------------------------
        # When the user asks the assistant to STORE a fact
        # ("please remember that my mom's name is Cornelia",
        # "don't forget my wifi password is ...", etc.) this turn is a
        # write — not a recall. We must:
        #   (a) skip memory / RAG / web retrieval entirely
        #   (b) bypass the cognitive router's semantic recall centroid,
        #       which otherwise scores high on the literal word "remember"
        #       and routes the turn to HYBRID — pulling the web tool into
        #       scope. A failed web fetch then injected a "[W] WEB SEARCH
        #       RESULTS: Internet search failed..." block, causing the LLM
        #       to loop on the "[W]" token (StreamRepetitionGuard cancelled
        #       the stream, producing the visible "[W][W][W]" stub bug).
        # The enrichment worker still picks the fact up asynchronously; we
        # just answer with a brief acknowledgment here.
        # ============================================================
        explicit_remember_body = detect_explicit_remember(self.prompt)
        explicit_remember_active = bool(explicit_remember_body)

        # T3.3: an explicit-remember turn is a write turn — we do NOT want to
        # run the normal extractor over the brief acknowledgement the
        # assistant will emit, because that text is easily misread as a
        # third-party claim. The explicit-remember bypass (synthesised
        # server-side from the user's own message) still runs on the
        # enrichment worker side under the ``explicit_only`` mode.
        if explicit_remember_active:
            self._mark_explicit_remember_mode()

        # ============================================================
        # 0.5 EXPLICIT FILE-SEARCH OVERRIDE (Memory v6.1)
        # ------------------------------------------------------------
        # When the user literally points Qube at their library
        # ("look into my files and tell me if there is a mention of X",
        # "check my documents for ...", "in my notes ...", etc.) we
        # want RAG only — skipping memory + web entirely.
        #
        # Without this, the cognitive router's semantic recall centroid
        # tends to fire on "tell me if there is a mention of <name>"
        # (high cosine similarity to the recall example set) and forces
        # HYBRID. HYBRID then calls ``memory_search`` and injects any
        # top-k memories regardless of topical relevance — a stored
        # "my mom's name is Cornelia" memory ended up in the prompt of
        # a Dr. Evelyn file-lookup query, confusing the LLM into
        # emitting a bare "[2]" citation token.
        #
        # Explicit-remember still beats file-search (a write turn has
        # absolute priority over any retrieval path).
        # ============================================================
        file_search_active = (
            not explicit_remember_active
            and detect_file_search_intent(self.prompt)
        )

        # ============================================================
        # 0.55 COMPOSER @-MENTION ATTACHMENTS
        # ------------------------------------------------------------
        # User-picked Files / Conversations / Tools override NLP routing.
        # ============================================================
        turn_attachments = list(getattr(self, "_turn_attachments", []) or [])
        if turn_attachments:
            logger.info(
                "[LLM Worker] Composer attachments: %s",
                attachment_summary(turn_attachments),
            )
            if self.session_id:
                from core.integrations.agent_scope import (
                    agent_scope_store,
                    build_agent_scope_from_attachments,
                )

                agent_scope_store.set_scope(
                    build_agent_scope_from_attachments(
                        str(self.session_id), turn_attachments
                    )
                )
        attachment_patch = None
        if not explicit_remember_active and turn_attachments:
            attachment_patch = resolve_attachment_routing(turn_attachments)

        attachment_file_active = False
        attachment_conversation_active = False
        self._turn_source_filter = None
        self._turn_source_prefix_filter = None
        self._turn_attachment_context = ""
        self._composer_knowledge_tool = None
        self._turn_canonical_help_entry = None
        self._help_canonical_system_hint = ""
        self._composer_internet_requested = False
        self._composer_trusted_requested = False

        if attachment_patch:
            if attachment_patch.get("attachment_file"):
                attachment_file_active = True
                self._turn_source_filter = attachment_patch.get("source_filter")
            if attachment_patch.get("attachment_conversation"):
                attachment_conversation_active = True
                ref_sid = attachment_patch.get("referenced_session_id")
                if ref_sid:
                    self._turn_attachment_context = build_referenced_conversation_context(
                        ref_sid, self.db
                    )
                    if not (self._turn_attachment_context or "").strip():
                        logger.warning(
                            "[LLM Worker] Conversation @-ref: no transcript loaded "
                            "for session_id=%s",
                            ref_sid,
                        )
                    else:
                        logger.info(
                            "[LLM Worker] Conversation @-ref: loaded transcript "
                            "for session_id=%s (%d chars)",
                            ref_sid,
                            len(self._turn_attachment_context),
                        )
                    self._mark_skip_enrichment("composer_conversation_ref")
            composer_tool = attachment_patch.get("attachment_tool")
            library_knowledge_active = composer_tool == "library"
            help_knowledge_active = composer_tool == "help"
            if library_knowledge_active:
                self._composer_knowledge_tool = "library"
                force_web = True
                if attachment_patch.get("route") != "web":
                    attachment_patch = dict(attachment_patch)
                    attachment_patch["route"] = "web"
                    attachment_patch["strategy"] = "attachment_tool_library"
                logger.info(
                    "[LLM Worker] Composer @library: forcing internal corpus WEB path"
                )
            elif help_knowledge_active:
                self._composer_knowledge_tool = "help"
                self._turn_source_prefix_filter = HELP_DOC_SOURCE_PREFIX
                force_web = True
                if attachment_patch.get("route") != "web":
                    attachment_patch = dict(attachment_patch)
                    attachment_patch["route"] = "web"
                    attachment_patch["strategy"] = "attachment_tool_help"
                logger.info(
                    "[LLM Worker] Composer @help: forcing help corpus WEB path "
                    "(prefix=%s)",
                    HELP_DOC_SOURCE_PREFIX,
                )
            elif composer_tool and is_web_composer_tool(composer_tool):
                self._composer_knowledge_tool = composer_tool
                self._composer_internet_requested = composer_tool == "internet"
                self._composer_trusted_requested = composer_tool == "trusted"
                force_web = True
                if attachment_patch.get("route") != "web":
                    attachment_patch = dict(attachment_patch)
                    attachment_patch["route"] = "web"
                    attachment_patch["strategy"] = f"attachment_tool_{composer_tool}"
                logger.info(
                    "[LLM Worker] Composer @%s: forcing WEB search for this turn",
                    composer_tool,
                )

        if getattr(self, "_composer_knowledge_tool", None) == "help":
            help_entry = match_canonical_answer(self.prompt)
            self._turn_canonical_help_entry = help_entry
            self._help_canonical_system_hint = (
                canonical_answer_system_hint(help_entry) if help_entry else ""
            )

        scoped_library_active = file_search_active or attachment_file_active

        # ============================================================
        # 0.6 T3.2: NARRATIVE / RECAP OVERRIDE
        # ------------------------------------------------------------
        # Narrative recap queries ("what have we been working on?",
        # "recap my session", "where did we leave off?") must route to
        # MEMORY with ``prefer_episode=True`` so the session-summary rows
        # outrank the atomic-fact rows. File-search and explicit-remember
        # both win over narrative (file-search is a document query, and
        # explicit-remember is a write turn).
        # ============================================================
        narrative_active = (
            not explicit_remember_active
            and not scoped_library_active
            and not attachment_conversation_active
            and detect_narrative_intent(self.prompt)
        )

        # ============================================================
        # 1. ROUTING PHASE
        # ============================================================
        self.status_update.emit("Working...")

        intent_vector = None

        if explicit_remember_active:
            logger.info(
                "[LLM Worker] Explicit-remember intent detected; skipping routing/retrieval."
            )
            decision = {
                "route": "none",
                "strategy": "explicit_remember",
                "explicit_remember": True,
            }
        elif attachment_patch:
            decision = {
                k: v
                for k, v in attachment_patch.items()
                if k not in ("source_filter", "referenced_session_id")
            }
            if attachment_patch.get("rag_query") is None and "rag_query" not in decision:
                decision["rag_query"] = self.prompt
            logger.info(
                "[LLM Worker] Composer attachment routing: route=%s strategy=%s",
                decision.get("route"),
                decision.get("strategy"),
            )
        elif file_search_active:
            logger.info(
                "[LLM Worker] Explicit file-search intent detected; forcing RAG, skipping memory/web."
            )
            decision = {
                "route": "rag",
                "strategy": "explicit_file_search",
                "file_search": True,
                "rag_query": self.prompt,
            }
            # The cognitive router is skipped entirely — we don't want its
            # semantic recall centroid or its internet_enabled flag to
            # override a turn the user scoped to document lookup.
        elif narrative_active:
            logger.info(
                "[LLM Worker] Narrative recap intent detected; forcing MEMORY with prefer_episode=True."
            )
            decision = {
                "route": "memory",
                "strategy": "narrative_recap",
                "narrative": True,
                "memory_query": self.prompt,
                "prefer_episode": True,
            }
        elif self.USE_COGNITIVE_ROUTER:
            intent_vector = self.embedding_cache.get_embedding(routing_query)
            if intent_vector is None:
                decision = {"route": "none", "strategy": "no_embedder"}
            else:
                self._ensure_recall_centroid()
                decision = self.cognitive_router.route(
                    routing_query,
                    intent_vector=intent_vector,
                    weights=self.router_tuner.get_weights() if self.USE_ADAPTIVE_ROUTER else None
                )
        else:
            decision = {"route": "none", "strategy": "fallback"}

        execution_route = decision["route"].upper()

        prior_execution_route = "NONE"
        if self.session_id:
            prior_execution_route = self._prior_execution_route_by_session.get(
                str(self.session_id),
                "NONE",
            )
        has_prior_chat = len(history) > 1

        # ------------------------------------------------------------
        # Phase A: recall-intent fusion override.
        # "Tell me about X" / "who is X" / "remind me about X" style queries
        # must consult BOTH memory and documents so the LLM can synthesize
        # from the richer source. Without this override the router will
        # frequently pick pure MEMORY (matching on "remember") or NONE and
        # miss the document chunk that actually describes X.
        # Web route is NOT overridden here — web triggers win below.
        # Explicit-remember is a write, so the fusion override is skipped.
        # ------------------------------------------------------------
        if (
            not explicit_remember_active
            and not scoped_library_active
            and not attachment_patch
            and should_apply_recall_fusion(clean_prompt, decision=decision)
            and execution_route in ("NONE", "MEMORY", "RAG")
        ):
            logger.info("[LLM Worker] Recall intent detected; routing to HYBRID")
            execution_route = "HYBRID"
            decision["recall_fusion"] = True

        if (
            discourse_enabled
            and follow_up.active
            and discourse_state
            and discourse_state.active_topic
            and execution_route in ("MEMORY", "RAG", "HYBRID")
            and not explicit_remember_active
            and not scoped_library_active
            and not narrative_active
            and not decision.get("recall_fusion")
            and not attachment_patch
        ):
            logger.info(
                "[Discourse] follow-up topic %r; downgrading route %s -> NONE",
                discourse_state.active_topic,
                execution_route,
            )
            execution_route = "NONE"
            decision["route_inherited_from_discourse"] = True

        if (
            not explicit_remember_active
            and not scoped_library_active
            and should_downgrade_embedding_rag_on_continuation(
                self.prompt,
                decision=decision if isinstance(decision, dict) else None,
                execution_route=execution_route,
                prior_execution_route=prior_execution_route,
                follow_up_active=follow_up.active,
                has_chat_history=has_prior_chat,
                scoped_library_active=scoped_library_active,
            )
        ):
            logger.info(
                "[LLM Worker] Embedding RAG/HYBRID suppressed on plain-chat "
                "continuation (prior_route=%s); execution_route %s -> NONE",
                prior_execution_route,
                execution_route,
            )
            execution_route = "NONE"
            if isinstance(decision, dict):
                decision["embedding_rag_continuation_suppressed"] = True

        if (
            not explicit_remember_active
            and not scoped_library_active
            and should_downgrade_short_vague_retrieval_on_first_turn(
                self.prompt,
                decision=decision if isinstance(decision, dict) else None,
                execution_route=execution_route,
                has_chat_history=has_prior_chat,
                scoped_library_active=scoped_library_active,
            )
        ):
            logger.info(
                "[LLM Worker] Short vague first-turn retrieval suppressed; "
                "execution_route %s -> NONE",
                execution_route,
            )
            execution_route = "NONE"
            if isinstance(decision, dict):
                decision["short_vague_first_turn_suppressed"] = True

        force_rag_via_trigger = False
        # Custom NLP triggers: upgrade retrieval without clobbering HYBRID.
        if not explicit_remember_active and not scoped_library_active and self.mcp_auto_enabled:
            if matches_custom_rag_trigger(clean_prompt, self.cached_custom_triggers):
                execution_route, force_rag_via_trigger = apply_custom_rag_trigger_route(
                    execution_route,
                    matched=True,
                )
                decision["rag_query"] = self.prompt
                decision["custom_rag_trigger"] = True

        library_bypass = library_lane_allowed(
            mcp_rag_enabled=self.mcp_rag_enabled,
            force_rag_via_trigger=force_rag_via_trigger,
            scoped_library_active=scoped_library_active,
        )
        library_blocked = not library_bypass
        if library_blocked and execution_route == "RAG":
            logger.info(
                "[LLM Worker] Cognitive router picked RAG but Local Knowledge Base "
                "is disabled and no library bypass fired; reverting execution_route "
                "to NONE."
            )
            execution_route = "NONE"
            if isinstance(decision, dict):
                decision["rag_vetoed_tool_disabled"] = True
        elif library_blocked and execution_route == "HYBRID":
            logger.info(
                "[LLM Worker] HYBRID route with Local Knowledge Base disabled; "
                "skipping library leg and downgrading execution_route to MEMORY."
            )
            execution_route = "MEMORY"
            if isinstance(decision, dict):
                decision["rag_vetoed_tool_disabled"] = True
                decision["rag_library_leg_skipped"] = True

        # ------------------------------------------------------------
        # INTERNET TRIGGER (manual + cognitive)
        # ------------------------------------------------------------
        # Skipped on explicit-remember (write turn) and explicit file-search
        # (the user scoped this turn to the local library).
        manual_web = False
        auto_web = False
        hard_web = detect_hard_explicit_web_request(clean_prompt)
        live_web = query_implies_live_web_intent(
            clean_prompt,
            decision=decision if isinstance(decision, dict) else None,
        )
        explicit_web_request = hard_web or live_web
        if not explicit_remember_active and not scoped_library_active:
            # Manual trigger: user explicitly asked to search/check the web.
            manual_web = hard_web

            # Automatic trigger: cognitive router decides internet is needed
            auto_web = getattr(self, "USE_COGNITIVE_ROUTER_INTERNET", False) and decision.get("internet_enabled", False)

            # Final execution decision for WEB
            if (
                execution_route != "CAPABILITY"
                and (
                    force_web
                    or manual_web
                    or auto_web
                    or (live_web and not hard_web)
                )
            ):
                execution_route = "WEB"

            # ------------------------------------------------------------
            # PROACTIVE WEB-ROUTE VETO
            # ------------------------------------------------------------
            # The cognitive router internally promotes ``route`` to
            # ``"web"`` as soon as ``_score_web_intent`` clears its
            # threshold (keywords like "weather" / "today" / "news").
            # That value then flows through ``execution_route =
            # decision["route"].upper()`` above, *before* we ever reach
            # the manual/force/auto gate. So a query like "what's the
            # weather in Copenhagen today?" can arrive here already
            # pinned to WEB even when the user has explicitly disabled
            # the internet tool (``mcp_internet_enabled=False``) and is
            # not force-routing this turn.
            #
            # If neither the force flag, the manual trigger, nor the
            # explicit cognitive-router-internet auto-trigger fired,
            # AND the web tool is disabled, the router's WEB pick has
            # no justification on this turn — revert to NONE so the
            # downstream tool-execution and system-prompt branches
            # don't end up on the WEB path. This prevents the "You
            # have been provided with live web search results" system
            # prompt from firing on a turn that will carry no web
            # sources (the root cause of the hallucinated [W]
            # citation regression).
            if (
                execution_route == "WEB"
                and not force_web
                and not manual_web
                and not auto_web
                and not self.mcp_internet_enabled
            ):
                logger.info(
                    "[LLM Worker] Cognitive router picked WEB but internet "
                    "tool is disabled and no explicit web trigger fired; "
                    "reverting execution_route to NONE."
                )
                execution_route = "NONE"
                decision["web_vetoed_tool_disabled"] = True

            # Deictic follow-up with no resolvable topic cannot produce a
            # meaningful web query ("tips for this" alone). When a topic
            # IS known, WEB stays enabled and the search uses an expanded
            # query (see resolve_web_query below).
            if (
                discourse_enabled
                and should_veto_ungrounded_web_follow_up(follow_up, discourse_state)
                and execution_route == "WEB"
                and not force_web
                and not manual_web
                and not getattr(self, "_composer_knowledge_tool", None)
            ):
                logger.info(
                    "[Discourse] ungrounded follow-up (no topic); "
                    "vetoing WEB route -> NONE"
                )
                execution_route = "NONE"
                decision["discourse_vetoed_web"] = True
            elif (
                discourse_enabled
                and follow_up.active
                and discourse_state
                and discourse_state.active_topic
                and execution_route == "WEB"
            ):
                decision["discourse_web_query_expanded"] = True

        preference_policy = resolve_preference_policy(
            session_overrides=getattr(self, "_session_preference_overrides", None),
        )
        web_vetoed = bool(
            isinstance(decision, dict) and decision.get("web_vetoed_tool_disabled")
        )
        rag_vetoed = bool(
            isinstance(decision, dict) and decision.get("rag_vetoed_tool_disabled")
        )
        web_capability_blocked = bool(
            explicit_web_request and not self.mcp_internet_enabled
        ) or bool(
            web_vetoed and query_implies_live_web_intent(clean_prompt, decision=decision)
        )
        explicit_library_request = query_explicitly_requests_library_search(
            clean_prompt,
            decision=decision if isinstance(decision, dict) else None,
        )
        rag_capability_blocked = bool(
            library_blocked
            and explicit_library_request
            and execution_route in ("NONE", "MEMORY")
            and (
                rag_vetoed
                or detect_file_search_intent(clean_prompt)
                or query_has_lexical_library_signal(clean_prompt)
            )
        )
        if isinstance(decision, dict):
            decision["rag_capability_blocked"] = rag_capability_blocked
        if web_vetoed and not web_capability_blocked:
            logger.info(
                "[LLM Worker] WEB route vetoed (internet disabled) but query has "
                "no live-web intent; using plain chat prompt."
            )
        if rag_vetoed and not rag_capability_blocked:
            logger.info(
                "[LLM Worker] RAG route vetoed (library disabled) but query has "
                "no library intent; using plain chat prompt."
            )

        # ============================================================
        # ROUTING START TIME (telemetry)
        # ============================================================
        route_start = time.time()

        logger.info(f"[Router] route={execution_route}")

        query_expansion = propose_query_expansion(
            inference_user_text,
            follow_up,
            discourse_state if discourse_enabled else None,
            history,
            self._sidecar_client,
            tentative_route=execution_route.lower(),
            retrieval_query=retrieval_query,
        )
        if query_expansion:
            logger.info(
                "[Sidecar] assistive expansion conf=%.2f expanded=%r route=%s",
                query_expansion.confidence,
                query_expansion.expanded_query[:120],
                execution_route,
            )

        retrieval_wrapper_mode = "none"
        self._stamp_discourse_on_decision(
            decision,
            follow_up=follow_up,
            discourse_state=discourse_state if discourse_enabled else None,
            routing_query=routing_query,
            retrieval_query=retrieval_query,
            core_memory_suppressed=core_memory_suppressed,
            retrieval_wrapper_mode=retrieval_wrapper_mode,
            inference_user_text=inference_user_text,
            resolved_query=resolved_query,
            prompt_rewrite=prompt_rewrite,
            resolved_retrieval=resolved_retrieval,
        )
        self._stamp_query_expansion_on_decision(
            decision,
            original_query=original_query,
            retrieval_query=retrieval_query,
            expansion=query_expansion,
        )

        try:
            self._routing_debug_turn_seq += 1
            record = build_record(
                query=self.prompt,
                decision=decision,
                session_id=self.session_id,
                turn_id=self._routing_debug_turn_seq,
                effective_route=execution_route.lower(),
            )
            self.routing_debug_buffer.append(record)
            self.routing_debug_record_added.emit(dataclasses.asdict(record))
        except Exception as e:
            logger.warning("[RoutingDebug] failed to record turn: %s", e)

        # ============================================================
        # SHADOW RETRIEVAL POLICY (observational — no routing/retrieval change)
        # ============================================================
        if shadow_retrieval_policy_enabled():
            try:
                shadow_state = build_shadow_state_from_worker(
                    execution_route=execution_route,
                    decision=decision if isinstance(decision, dict) else {},
                    prompt=clean_prompt,
                    follow_up=follow_up,
                    discourse_state=discourse_state if discourse_enabled else None,
                    memory_enabled=True,
                    rag_enabled=bool(self.mcp_auto_enabled),
                )
                shadow_policy = compute_retrieval_policy(shadow_state)
                baseline_route_norm = str(execution_route or "NONE").strip().lower()
                shadow_route = shadow_policy.get("shadow_decision", "none")
                route_divergence = baseline_route_norm != shadow_route
                if self.USE_TELEMETRY:
                    self.telemetry.log({
                        "route": execution_route,
                        "baseline_route": execution_route,
                        "shadow_retrieval_policy": shadow_policy,
                        "baseline_recall_fusion": shadow_policy.get(
                            "baseline_recall_fusion"
                        ),
                        "shadow_route": shadow_route,
                        "route_divergence": route_divergence,
                        "latency_ms": 0.0,
                        "memory_hits": 0,
                        "rag_hits": 0,
                        "web_hits": 0,
                    })
                self._shadow_retrieval_telemetry.record(
                    baseline_route=execution_route,
                    shadow_policy=shadow_policy,
                    prompt=clean_prompt,
                )
                logger.info(
                    "[ShadowRetrievalPolicy] baseline=%s shadow=%s fusion=%s "
                    "propensity=%.3f divergence=%s",
                    execution_route,
                    shadow_route,
                    shadow_policy.get("baseline_recall_fusion"),
                    float(shadow_policy.get("retrieval_propensity_score") or 0.0),
                    route_divergence,
                )
            except Exception as exc:
                logger.debug("[ShadowRetrievalPolicy] observation failed: %s", exc)

        # ============================================================
        # 2. TOOL EXECUTION
        # ============================================================
        memory_context = ""
        tool_context = ""
        if getattr(self, "_turn_attachment_context", ""):
            tool_context = self._turn_attachment_context.strip() + "\n\n"
        all_ui_sources = []

        # 🔑 THE FIX: Initialize these dictionaries so telemetry doesn't crash
        mem_result = {} 
        rag_result = {}
        web_context = "" # Also initialize this to be safe

        query_vector = None

        if execution_route in ["MEMORY", "RAG", "HYBRID"]:
            query_vector = self.embedding_cache.get_embedding(retrieval_query)
            if query_vector is None:
                logger.warning(
                    "[LLM Worker] Embedder unavailable; retrieval routes disabled for this turn."
                )
                execution_route = "NONE"
                if isinstance(decision, dict):
                    decision["route"] = "none"
                    decision["embedder_unavailable"] = True

        # ---- MEMORY ----
        if execution_route in ["MEMORY", "HYBRID"]:
            # T3.4 tier flags per route (see §3.3 of the plan):
            #  * MEMORY route (router centroid picked ``memory``, which is
            #    recall-leaning by construction) OR HYBRID route
            #    (recall+docs fusion): include_preference + include_knowledge
            #    + include_context. Knowledge rows (third-party facts /
            #    document-derived claims) are exactly what the user wants
            #    when they ask "remind me about X" / "who is X".
            #  * Narrative: additionally include_episode. ``prefer_episode``
            #    alone already forces episode in ``memory_tool``; we pass
            #    ``include_episode=True`` explicitly for clarity and so the
            #    WHERE builder sees the same flag set the caller intended.
            prefer_episode = bool(
                decision.get("prefer_episode") or narrative_active
            )
            include_episode = prefer_episode or narrative_active

            mem_q = decision.get("memory_query") or retrieval_query
            mem_result = self._memory_search_hybrid(
                mem_q,
                query_vector,
                query_expansion,
                prefer_episode=prefer_episode,
                include_preference=True,
                include_knowledge=True,
                include_episode=include_episode,
                include_context=True,
                apply_mmr=True,
                apply_temporal_decay=True,
                query_fingerprint=self._memory_query_fingerprint(
                    mem_q,
                    include_preference=True,
                    include_knowledge=True,
                    include_episode=include_episode,
                    include_context=True,
                ),
            )
            memory_context = mem_result.get("memory_context", "")
            all_ui_sources.extend(mem_result.get("memory_sources", []))
        elif (
            execution_route == "NONE"
            and not explicit_remember_active
            and not scoped_library_active
            and not attachment_conversation_active
            and not getattr(self, "_composer_knowledge_tool", None)
        ):
            # T3.4 §3.3 "default every turn (even CHAT)": on a plain chat
            # turn (router picked ``none``) run a cheap preferences-only
            # retrieval (MemGPT-style core memory). This is the lane where
            # stable user preferences like "I prefer metric units" or
            # "call me by my first name" surface into every conversation
            # without the user having to trigger recall intent.
            #
            # Explicit-remember is a write turn — skip retrieval. File-
            # search scopes to docs; memory retrieval would just dilute
            # the context window.
            if discourse_enabled and follow_up.confidence >= FOLLOW_UP_SUPPRESS_THRESHOLD:
                core_memory_suppressed = True
                logger.info(
                    "[Discourse] follow_up=%s conf=%.2f core_memory=suppressed",
                    follow_up.kind.value,
                    follow_up.confidence,
                )
            else:
                if query_vector is None:
                    query_vector = self.embedding_cache.get_embedding(retrieval_query)
                mem_kwargs: dict = {}
                if (
                    discourse_enabled
                    and follow_up.confidence >= 0.45
                    and follow_up.confidence < FOLLOW_UP_SUPPRESS_THRESHOLD
                ):
                    from core.memory_retrieval_policy import (
                        FOLLOW_UP_CORE_MEMORY_MIN_MARGIN,
                        FOLLOW_UP_CORE_MEMORY_MIN_SCORE,
                    )

                    mem_kwargs["core_memory_min_score"] = FOLLOW_UP_CORE_MEMORY_MIN_SCORE
                    mem_kwargs["core_memory_min_margin"] = FOLLOW_UP_CORE_MEMORY_MIN_MARGIN
                mem_result = self._memory_search_hybrid(
                    retrieval_query,
                    query_vector,
                    query_expansion,
                    prefer_episode=False,
                    include_preference=True,
                    include_knowledge=False,
                    include_episode=False,
                    include_context=True,
                    top_k=3,
                    apply_core_memory_gate=True,
                    query_fingerprint=self._memory_query_fingerprint(
                        retrieval_query,
                        include_preference=True,
                        include_knowledge=False,
                        include_episode=False,
                        include_context=True,
                    ),
                    exclude_presentation_preferences=True,
                    **mem_kwargs,
                )
                memory_context = mem_result.get("memory_context", "")
                all_ui_sources.extend(mem_result.get("memory_sources", []))

        # ---- RAG ----
        if execution_route in ["RAG", "HYBRID"] and (
            self.mcp_rag_enabled
            or force_rag_via_trigger
            or scoped_library_active
        ):
            self.context_retrieved.emit(True)
            rag_q = decision.get("rag_query") or retrieval_query
            rag_result = self._rag_search_hybrid(
                rag_q,
                query_vector,
                query_expansion,
                source_filter=getattr(self, "_turn_source_filter", None),
            )
            # 🔑 Use += to ensure we don't accidentally wipe out other tool data
            tool_context += rag_result.get("llm_context", "")
            rag_sources = rag_result.get("sources", []) or []
            all_ui_sources.extend(rag_sources)

            # Phase B: collect per-turn rag chunk ids for the enrichment
            # context. ``chunk_id`` is populated by rag_tool.rag_search (UI
            # contract additive field). We dedupe while preserving order.
            for s in rag_sources:
                cid = s.get("chunk_id") if isinstance(s, dict) else None
                if cid and cid not in self._turn_rag_chunk_ids:
                    self._turn_rag_chunk_ids.append(str(cid))

        # ---- CAPABILITY (composer @[cap:…]) ----
        if execution_route == "CAPABILITY":
            import time as _cap_time

            from core.app_settings import get_retrieval_profile
            from core.integrations.capability_inspect import build_capability_inspect_trace
            from core.integrations.capability_invoke import invoke_gated_capability
            from core.integrations.capability_trace import (
                CapabilityTraceContext,
                record_capability_retrieval_trace,
            )
            from core.knowledge.bundle_builder import build_generic_bundle
            from core.knowledge.retrieval_records import RetrievalContextFingerprint
            from core.knowledge.ui_adapter import append_turn_evidence_bundle_sources

            cap_urn = ""
            cap_preset_id = ""
            if isinstance(decision, dict):
                cap_urn = str(decision.get("capability_urn") or "")
                cap_preset_id = str(decision.get("capability_preset_id") or "").strip()
            cap_t0 = _cap_time.time()
            cap_turn_id = getattr(self, "_routing_debug_turn_seq", None)
            cap_session_id = str(self.session_id) if self.session_id else None
            from core.integrations.agent_scope import agent_scope_store

            cap_agent_scope = (
                agent_scope_store.get_scope(cap_session_id) if cap_session_id else None
            )
            per_cap_results = ()
            preset_label = cap_preset_id
            if cap_preset_id:
                from core.integrations.preset_capability_alias import (
                    build_preset_capability_inspect_trace,
                    format_preset_bundle_deny_summary,
                    invoke_preset_capability_bundle,
                )
                from core.knowledge.presets import load_preset

                invoke_result, per_cap_results = invoke_preset_capability_bundle(
                    cap_preset_id,
                    original_query or self.prompt,
                    max_results=5,
                    session_id=cap_session_id,
                    turn_id=str(cap_turn_id) if cap_turn_id is not None else None,
                    agent_scope=cap_agent_scope,
                )
                preset = load_preset(cap_preset_id)
                if preset is not None:
                    preset_label = preset.label
                cap_latency_ms = (_cap_time.time() - cap_t0) * 1000.0
                cap_steps = build_preset_capability_inspect_trace(
                    preset_id=cap_preset_id,
                    preset_label=preset_label,
                    query=original_query or self.prompt,
                    per_cap_results=per_cap_results,
                    bundle_result=invoke_result,
                    latency_ms=cap_latency_ms,
                )
            else:
                invoke_result = invoke_gated_capability(
                    cap_urn,
                    original_query or self.prompt,
                    max_results=5,
                    session_id=cap_session_id,
                    turn_id=str(cap_turn_id) if cap_turn_id is not None else None,
                    agent_scope=cap_agent_scope,
                )
                cap_latency_ms = (_cap_time.time() - cap_t0) * 1000.0
                cap_steps = build_capability_inspect_trace(
                    urn=cap_urn,
                    query=original_query or self.prompt,
                    allowed=invoke_result.allowed,
                    reason=invoke_result.reason,
                    rows=invoke_result.rows,
                    bundle_source_count=len(invoke_result.rows) if invoke_result.allowed else 0,
                    rejected_count=0,
                    latency_ms=cap_latency_ms,
                    descriptor=invoke_result.descriptor,
                )
            if isinstance(decision, dict):
                decision["capability_invoke_allowed"] = invoke_result.allowed
                decision["capability_invoke_reason"] = invoke_result.reason
                decision["capability_steps"] = cap_steps

            cap_trace_ctx = CapabilityTraceContext(
                cap_steps=list(cap_steps),
                query_raw=self.prompt,
                query_resolved=original_query or self.prompt,
                latency_ms=cap_latency_ms,
                preset_id=cap_preset_id,
                cap_urn=cap_urn,
                session_id=self.session_id,
                turn_id=cap_turn_id,
                kept_rows=[],
            )
            self._turn_capability_trace_ctx = cap_trace_ctx

            if per_cap_results:
                deny_summary = format_preset_bundle_deny_summary(
                    per_cap_results,
                    preset_label=preset_label,
                )
                if deny_summary:
                    tool_context += deny_summary

            if invoke_result.allowed and invoke_result.rows:
                kept_rows = list(invoke_result.rows)
                self._turn_evidence_bundle = build_generic_bundle(
                    query_raw=self.prompt,
                    query_resolved=original_query or self.prompt,
                    kept_rows=kept_rows,
                    rejected_count=0,
                    latency_ms=cap_latency_ms,
                    knowledge_service="capability",
                    retrieval_strategy="attachment_capability",
                    preset_id=cap_preset_id or None,
                    adapter_calls=tuple(
                        sorted(
                            {
                                str(row.get("_adapter") or "")
                                for row in kept_rows
                                if row.get("_adapter")
                            }
                        )
                    ),
                )
                append_turn_evidence_bundle_sources(
                    all_ui_sources, self._turn_evidence_bundle
                )
                context_parts: list[str] = []
                for row in kept_rows:
                    title = str(row.get("title") or "").strip()
                    snippet = str(row.get("snippet") or "").strip()
                    if title or snippet:
                        context_parts.append(
                            f"{title}\n{snippet}".strip()
                            if title and snippet
                            else (title or snippet)
                        )
                cap_context = "\n\n".join(context_parts)[: self.RAG_BUDGET]
                if cap_context:
                    hdr = "CAPABILITY RESULTS"
                    if tool_context:
                        tool_context = f"{tool_context}\n\n{hdr}:\n{cap_context}"
                    else:
                        tool_context = f"{hdr}:\n{cap_context}"
                logger.info(
                    "[LLM Worker] Capability invoke integrated (%d sources, %d chars)",
                    len(kept_rows),
                    len(cap_context),
                )
                cap_trace_ctx.kept_rows = kept_rows
                cap_trace_ctx.bundle = self._turn_evidence_bundle
                cap_trace_ctx.fingerprint = RetrievalContextFingerprint(
                    query_raw=self.prompt,
                    query_resolved=clean_prompt or self.prompt,
                    knowledge_service="capability",
                    preset_id=cap_preset_id or None,
                    adapter_filter=tuple(
                        sorted(
                            {
                                str(row.get("_adapter") or "")
                                for row in kept_rows
                                if row.get("_adapter")
                            }
                        )
                    ),
                    retrieval_profile=get_retrieval_profile(),
                    connector_config_hashes=(),
                )
                record_capability_retrieval_trace(
                    cap_trace_ctx,
                    db=self.db,
                    retrieval_profile=get_retrieval_profile(),
                )
            elif not invoke_result.allowed:
                logger.warning(
                    "[LLM Worker] Capability invoke denied: %s",
                    invoke_result.reason,
                )
                tool_context += (
                    "\n[CAPABILITY: The attached capability was not permitted "
                    f"({invoke_result.reason}). Tell the user briefly that the "
                    "capability could not run. Do NOT invent results or emit "
                    "citation tokens without sources.]\n"
                )
                record_capability_retrieval_trace(
                    cap_trace_ctx,
                    db=self.db,
                    retrieval_profile=get_retrieval_profile(),
                )
                self._mark_skip_enrichment("capability_invoke_denied")
            else:
                logger.warning(
                    "[LLM Worker] Capability invoke returned no sources for %s",
                    cap_urn,
                )
                tool_context += (
                    "\n[CAPABILITY: The attached capability returned no results. "
                    "Tell the user briefly that nothing was retrieved. Do NOT "
                    "invent facts or emit citation tokens without sources.]\n"
                )
                record_capability_retrieval_trace(
                    cap_trace_ctx,
                    db=self.db,
                    retrieval_profile=get_retrieval_profile(),
                )
                self._mark_skip_enrichment("capability_invoke_empty")

        # ---- WEB + HYBRID ----
        web_search_attempted = False
        _web_audit_common = {
            "force_web": force_web,
            "manual_web": manual_web,
            "auto_web": auto_web,
            "composer_internet": bool(getattr(self, "_composer_internet_requested", False)),
            "composer_trusted": bool(getattr(self, "_composer_trusted_requested", False)),
            "composer_evidence": getattr(self, "_composer_knowledge_tool", None)
            == "evidence",
            "query_raw": resolved_retrieval.raw_text,
            "query_resolved": apply_tool_policy(
                resolved_retrieval.web_text,
                preference_policy,
                tool="internet",
            ),
            "query_rewrite_reason": resolved_retrieval.web_rewrite_reason,
            "query_rewrite_failed": web_query_rewrite_failed(
                resolved_retrieval.raw_text,
                follow_up,
                resolved_retrieval.web_text,
                explicit_web=explicit_web_request,
            ),
        }
        if isinstance(decision, dict):
            if decision.get("web_vetoed_tool_disabled"):
                self._emit_web_search_audit(
                    **_web_audit_common,
                    execution_route="WEB",
                    veto_status=STATUS_VETOED_TOOL_DISABLED,
                )
            elif decision.get("discourse_vetoed_web"):
                self._emit_web_search_audit(
                    **_web_audit_common,
                    execution_route="WEB",
                    veto_status=STATUS_VETOED_UNGROUNDED,
                )
        if should_run_internet_search_for_route(
            execution_route,
            clean_prompt,
            decision=decision if isinstance(decision, dict) else None,
            force_web=force_web,
            manual_web=manual_web,
            auto_web=auto_web,
            composer_web_tool=bool(getattr(self, "_composer_knowledge_tool", None)),
            composer_internet=bool(getattr(self, "_composer_internet_requested", False)),
            composer_trusted=bool(getattr(self, "_composer_trusted_requested", False)),
        ) and (self.mcp_internet_enabled or force_web):
            web_search_attempted = True
            web_audit_t0 = time.time()
            web_results_raw_for_audit = None
            web_results_kept_for_audit = None
            web_audit_rel_diag = None
            via_direct = bool(
                force_web
                or manual_web
                or getattr(self, "_composer_knowledge_tool", None)
            )
            via_hybrid = bool(
                auto_web
                or (live_web and not hard_web and not force_web and not manual_web)
            )
            self.web_search_active.emit(True, via_direct, via_hybrid)
            self.status_update.emit("🌐 Searching the Web...")

            search_target = SearchTargetResult(
                resolved_retrieval.web_text,
                resolved_retrieval.web_rewrite_reason,
            )
            web_semantic = resolved_retrieval.web_text
            web_query = apply_tool_policy(
                web_semantic,
                preference_policy,
                tool="internet",
            )
            raw_prompt = resolved_retrieval.raw_text
            rewrite_failed = web_query_rewrite_failed(
                raw_prompt,
                follow_up,
                web_semantic,
                explicit_web=explicit_web_request,
            )
            if isinstance(decision, dict):
                decision["web_search_attempted"] = True
                decision["web_query_raw"] = raw_prompt
                decision["web_query_resolved"] = web_query
                decision["web_query_rewrite_reason"] = search_target.rewrite_reason
                decision["web_query_rewrite_failed"] = rewrite_failed
                if search_target.rewritten:
                    decision["web_query"] = web_query
                    if search_target.rewrite_reason == "topic_expansion":
                        decision["discourse_web_query_expanded"] = True
                    elif search_target.rewrite_reason == "meta_prior_turn":
                        decision["web_query_rewritten_from_meta"] = True

            if search_target.rewritten:
                logger.info(
                    "[WebPipeline] query_resolved raw=%r resolved=%r reason=%s",
                    raw_prompt[:120],
                    web_semantic[:120],
                    search_target.rewrite_reason,
                )
                if search_target.rewrite_reason == "topic_expansion":
                    logger.info(
                        "[Discourse] web search query expanded for follow-up "
                        "(topic=%r)",
                        discourse_state.active_topic if discourse_state else None,
                    )
                elif search_target.rewrite_reason == "meta_prior_turn":
                    logger.info(
                        "[Discourse] web search query rewritten from meta "
                        "web request (prior=%r)",
                        web_semantic[:120],
                    )
            if rewrite_failed:
                logger.warning(
                    "[WebPipeline] unresolved meta web request; search may be "
                    "off-topic (raw=%r)",
                    raw_prompt[:120],
                )

            try:
                web_query_vector = self.embedding_cache.get_embedding(
                    web_semantic or web_query
                )
            except Exception:
                web_query_vector = None

            turn_knowledge_service = resolve_turn_knowledge_service(
                composer_tool=getattr(self, "_composer_knowledge_tool", None),
                composer_trusted=bool(
                    getattr(self, "_composer_trusted_requested", False)
                ),
                composer_internet=bool(
                    getattr(self, "_composer_internet_requested", False)
                ),
                default_service=get_default_knowledge_service(),
            )
            turn_adapter_filter = adapter_filter_for_composer_tool(
                getattr(self, "_composer_knowledge_tool", None)
            )
            turn_preset_id = resolve_turn_preset_id(
                getattr(self, "_composer_knowledge_tool", None)
            )
            preset_overrides = resolve_preset_retrieval_overrides(
                getattr(self, "_composer_knowledge_tool", None)
            )
            library_store = (
                self.store
                if turn_knowledge_service == SERVICE_INTERNAL_CORPUS
                else None
            )
            corpus_source_filter = (
                getattr(self, "_turn_source_filter", None)
                if turn_knowledge_service == SERVICE_INTERNAL_CORPUS
                else None
            )
            corpus_source_prefix_filter = (
                getattr(self, "_turn_source_prefix_filter", None)
                if turn_knowledge_service == SERVICE_INTERNAL_CORPUS
                else None
            )

            _web_outcome = run_web_retrieval(
                query=web_query,
                semantic_query=web_semantic or web_query,
                query_vector=web_query_vector,
                embed_fn=self.embedding_cache.get_embedding,
                knowledge_service=turn_knowledge_service,
                adapter_filter=turn_adapter_filter,
                session_id=self.session_id,
                turn_id=getattr(self, "_routing_debug_turn_seq", None),
                library_store=library_store,
                source_filter=corpus_source_filter,
                source_prefix_filter=corpus_source_prefix_filter,
                preset_id=turn_preset_id,
                retrieval_profile=get_retrieval_profile(),
                db=self.db,
                composer_tool=getattr(self, "_composer_knowledge_tool", None),
                site_bias=preset_overrides.get("site_bias"),
                fetch_url_count=preset_overrides.get("fetch_url_count"),
            )
            web_results = _web_outcome.web_results
            web_results_raw_for_audit = _web_outcome.web_results_raw_for_audit
            web_results_kept_for_audit = _web_outcome.web_results_kept_for_audit
            web_audit_rel_diag = _web_outcome.relevance_diag
            self._turn_evidence_bundle = _web_outcome.bundle

            _pipeline_summary = summarize_web_pipeline_outcome(
                _web_outcome.bundle,
                _web_outcome.relevance_diag,
            )
            _search_outcome = search_outcome_from_relevance_diag(
                _web_outcome.relevance_diag
            )
            _search_outcome_kind = (
                _search_outcome.kind.value if _search_outcome is not None else "—"
            )
            logger.info(
                "[WebPipeline] outcome strategy=%s search_outcome=%s "
                "fetch_url_count=%d warnings=%s stages=%s sources=%d",
                _pipeline_summary["strategy"],
                _search_outcome_kind,
                _pipeline_summary["fetch_url_count"],
                ",".join(_pipeline_summary["warnings"]) or "none",
                _pipeline_summary["stages_summary"],
                _pipeline_summary["source_count"],
            )
            if _search_outcome is not None and _search_outcome.is_failure:
                if _search_outcome.kind.value == "bot_challenge":
                    self.status_update.emit("🌐 Web search blocked (provider challenge)")
                    self.web_search_outcome_hint.emit(
                        "Web search blocked — search engine bot challenge"
                    )
                else:
                    self.status_update.emit("🌐 Web search returned no usable results")
                    self.web_search_outcome_hint.emit(
                        f"Web search failed — {_search_outcome.label.lower()}"
                    )
            elif (
                _search_outcome is not None
                and _search_outcome.fallback_from
                and not _web_outcome.skip_enrichment
            ):
                self.web_search_outcome_hint.emit(
                    "Web search via "
                    f"{_search_outcome.provider} "
                    f"(fallback from {_search_outcome.fallback_from})"
                )

            from core.knowledge.discovery.backoff import consume_ddg_backoff_notification

            _should_notify_ddg_backoff, _ddg_backoff_remaining = (
                consume_ddg_backoff_notification()
            )
            if _should_notify_ddg_backoff:
                self.ddg_backoff_started.emit(_ddg_backoff_remaining)

            from core.knowledge.discovery.health import consume_tier_b_suggestion

            if consume_tier_b_suggestion():
                self.discovery_tier_b_suggested.emit()

            if _web_outcome.skip_enrichment:
                if _web_outcome.relevance_diag is None:
                    _sentinel_reason = failure_sentinel_reason(
                        _web_outcome.web_results_raw_for_audit
                    )
                    logger.info(
                        "[LLM Worker] Web results dropped (empty / failure sentinel"
                        + (f", reason={_sentinel_reason}" if _sentinel_reason else "")
                        + "); not injecting [W] context."
                    )
                else:
                    kept = _web_outcome.relevance_diag.get("web_results_kept_count", 0)
                    dropped = len(
                        _web_outcome.relevance_diag.get("web_relevance_dropped") or []
                    )
                    gate_mode = _web_outcome.relevance_diag.get(
                        "web_relevance_gate_mode", "strict"
                    )
                    logger.info(
                        "[WebPipeline] relevance_gate kept=%d dropped=%d "
                        "min_overlap=%.2f mode=%s",
                        kept,
                        dropped,
                        _web_outcome.relevance_diag.get(
                            "web_relevance_min_overlap", 0.15
                        ),
                        gate_mode,
                    )
                    logger.info(
                        "[LLM Worker] Web results dropped (relevance gate); "
                        "not injecting [W] context."
                    )
                if execution_route in ("WEB", "INTERNET", "HYBRID"):
                    self._mark_skip_enrichment("web_tool_failure")
                if _web_outcome.relevance_diag and isinstance(decision, dict):
                    decision.update(_web_outcome.relevance_diag)
                if _web_outcome.relevance_diag and hasattr(self, "routing_debug_buffer"):
                    try:
                        self.routing_debug_buffer.merge_web_pipeline_into_latest(
                            {
                                "web_query_resolved": web_query,
                                "web_query_rewrite_reason": search_target.rewrite_reason,
                                "web_query_rewrite_failed": rewrite_failed,
                                **_web_outcome.relevance_diag,
                            }
                        )
                    except Exception:
                        pass
            elif web_audit_rel_diag:
                kept = web_audit_rel_diag.get("web_results_kept_count", 0)
                dropped = len(web_audit_rel_diag.get("web_relevance_dropped") or [])
                gate_mode = web_audit_rel_diag.get("web_relevance_gate_mode", "strict")
                if web_audit_rel_diag.get("web_relevance_gate_skipped"):
                    logger.info(
                        "[WebPipeline] relevance_gate skipped mode=%s kept=%d",
                        gate_mode,
                        kept,
                    )
                else:
                    logger.info(
                        "[WebPipeline] relevance_gate kept=%d dropped=%d "
                        "min_overlap=%.2f mode=%s",
                        kept,
                        dropped,
                        web_audit_rel_diag.get("web_relevance_min_overlap", 0.15),
                        gate_mode,
                    )
                if isinstance(decision, dict):
                    decision.update(web_audit_rel_diag)
                if hasattr(self, "routing_debug_buffer"):
                    try:
                        self.routing_debug_buffer.merge_web_pipeline_into_latest(
                            {
                                "web_query_resolved": web_query,
                                "web_query_rewrite_reason": search_target.rewrite_reason,
                                "web_query_rewrite_failed": rewrite_failed,
                                **web_audit_rel_diag,
                            }
                        )
                    except Exception:
                        pass
            if _web_outcome.bundle is not None and isinstance(decision, dict):
                decision["knowledge_service"] = (
                    _web_outcome.bundle.knowledge_service
                    if _web_outcome.bundle is not None
                    else turn_knowledge_service
                )
                decision["evidence_bundle_id"] = _web_outcome.bundle.bundle_id
                decision["evidence_coverage"] = _web_outcome.bundle.coverage
                decision["evidence_confidence"] = round(
                    _web_outcome.bundle.confidence, 4
                )

            if web_results:
                web_items: list[dict] = []
                if isinstance(web_results, list):
                    web_items = [r for r in web_results if isinstance(r, dict)]
                else:
                    web_items = [
                        {
                            "title": "Live Web Search",
                            "snippet": str(web_results),
                        }
                    ]
                web_items = format_web_snippets(web_items, preference_policy)

                web_context_parts: list[str] = []
                for item in web_items:
                    title = str(item.get("title") or "").strip()
                    snippet = str(item.get("snippet") or "").strip()
                    if title or snippet:
                        web_context_parts.append(
                            f"{title}\n{snippet}".strip() if title and snippet else (title or snippet)
                        )
                web_context = "\n\n".join(web_context_parts)[: self.RAG_BUDGET]

                _turn_bundle = getattr(self, "_turn_evidence_bundle", None)
                if _turn_bundle is not None and _turn_bundle.sources:
                    from core.knowledge.ui_adapter import append_turn_evidence_bundle_sources

                    append_turn_evidence_bundle_sources(all_ui_sources, _turn_bundle)
                else:
                    for idx, item in enumerate(web_items, start=1):
                        title = (
                            str(item.get("title") or "").strip() or f"Web result {idx}"
                        )
                        snippet = str(item.get("snippet") or "").strip()
                        src: dict = {
                            "filename": title,
                            "content": snippet,
                            "type": "web",
                        }
                        url = str(item.get("url") or "").strip()
                        if url.startswith(("http://", "https://")):
                            src["url"] = url
                        all_ui_sources.append(src)

                web_hdr = (
                    "QUBE HELP DOCUMENTATION"
                    if getattr(self, "_composer_knowledge_tool", None) == "help"
                    else "LIBRARY SEARCH RESULTS"
                    if turn_knowledge_service == SERVICE_INTERNAL_CORPUS
                    else "WEB SEARCH RESULTS"
                )
                if tool_context:
                    tool_context = f"{tool_context}\n\n{web_hdr}:\n{web_context}"
                else:
                    tool_context = f"{web_hdr}:\n{web_context}"

                logger.info(
                    "[LLM Worker] Web search integrated (%d sources, %d chars)",
                    len(web_items),
                    len(web_context),
                )

            if getattr(self, "_composer_knowledge_tool", None) == "help":
                help_entry = getattr(self, "_turn_canonical_help_entry", None)
                if help_entry and tool_context.strip():
                    canonical_block = build_canonical_context_block(help_entry)
                    tool_context = f"{canonical_block}\n\n{tool_context}"
                elif help_entry:
                    tool_context = build_canonical_context_block(help_entry)
                log_help_query(
                    query=self.prompt,
                    retrieved_doc_ids=help_doc_ids_from_sources(all_ui_sources),
                    canonical_id=(
                        str(help_entry.get("id")) if isinstance(help_entry, dict) else None
                    ),
                    session_id=self.session_id,
                )

            if web_results_kept_for_audit is None and web_results:
                web_results_kept_for_audit = [
                    dict(r) for r in web_results if isinstance(r, dict)
                ]

            self._emit_web_search_audit(
                **_web_audit_common,
                execution_route=execution_route,
                web_results_raw=web_results_raw_for_audit,
                web_results_kept=web_results_kept_for_audit,
                relevance_diag=web_audit_rel_diag,
                latency_ms=(time.time() - web_audit_t0) * 1000,
            )

        digest_mem_eligible = bool(
            memory_context and mem_result.get("memory_sources")
        )
        digest_mem_attempted = False
        digest_mem_applied = False
        digest_mem_chars_before = 0
        digest_mem_chars_after = 0
        digest_mem_source_count = 0
        digest_mem_skip_reason = ""
        if digest_mem_eligible:
            mem_digest = digest_memory_context(
                memory_context,
                mem_result.get("memory_sources") or [],
                self._sidecar_client,
            )
            digest_mem_chars_before = mem_digest.chars_before
            digest_mem_chars_after = mem_digest.chars_after
            digest_mem_source_count = mem_digest.source_count
            digest_mem_skip_reason = mem_digest.skip_reason
            digest_mem_attempted = digest_mem_eligible and mem_digest.skip_reason not in (
                "disabled",
                "below_threshold",
                "no_context",
            )
            digest_mem_applied = bool(mem_digest.applied)
            memory_context = mem_digest.text
            if digest_mem_applied:
                logger.info(
                    "[Sidecar] memory digest applied chars %d -> %d (sources=%d)",
                    digest_mem_chars_before,
                    digest_mem_chars_after,
                    digest_mem_source_count,
                )
            elif digest_mem_skip_reason == "below_threshold":
                logger.debug(
                    "[Sidecar] memory digest skipped below_threshold chars=%d",
                    digest_mem_chars_before,
                )

        digest_rag_eligible = bool(tool_context and rag_result.get("sources"))
        digest_rag_attempted = False
        digest_rag_applied = False
        digest_rag_chars_before = 0
        digest_rag_chars_after = 0
        digest_rag_source_count = 0
        digest_rag_skip_reason = ""
        if digest_rag_eligible:
            rag_digest = digest_rag_context(
                tool_context,
                rag_result.get("sources") or [],
                self._sidecar_client,
            )
            digest_rag_chars_before = rag_digest.chars_before
            digest_rag_chars_after = rag_digest.chars_after
            digest_rag_source_count = rag_digest.source_count
            digest_rag_skip_reason = rag_digest.skip_reason
            digest_rag_attempted = digest_rag_eligible and rag_digest.skip_reason not in (
                "disabled",
                "below_threshold",
                "no_context",
            )
            digest_rag_applied = bool(rag_digest.applied)
            tool_context = rag_digest.text
            if digest_rag_applied:
                logger.info(
                    "[Sidecar] RAG digest applied chars %d -> %d (sources=%d)",
                    digest_rag_chars_before,
                    digest_rag_chars_after,
                    digest_rag_source_count,
                )
            elif digest_rag_skip_reason == "below_threshold":
                logger.debug(
                    "[Sidecar] RAG digest skipped below_threshold chars=%d",
                    digest_rag_chars_before,
                )

        # Sequential ids + emit isolated snapshots (UI must not share worker list refs)
        self._apply_sequential_source_ids(all_ui_sources, execution_route)
        self._turn_all_ui_sources = copy.deepcopy(all_ui_sources)
        if all_ui_sources:
            self.sources_found.emit(self.session_id or "", copy.deepcopy(all_ui_sources))
        evidence_bundle = getattr(self, "_turn_evidence_bundle", None)
        if evidence_bundle is not None:
            from core.knowledge.evidence_transparency import build_evidence_transparency

            transparency = build_evidence_transparency(evidence_bundle)
            if research_map_enabled():
                from core.knowledge.graph.transparency import (
                    enrich_transparency_with_prior_sessions,
                )

                transparency = enrich_transparency_with_prior_sessions(
                    transparency,
                    db=self.db,
                    session_id=self.session_id,
                    bundle=evidence_bundle,
                )
            self._turn_evidence_transparency = transparency
            self.evidence_transparency_found.emit(
                self.session_id or "",
                copy.deepcopy(transparency),
            )

        # ============================================================
        # TELEMETRY + SELF TUNING
        # ============================================================
        latency_ms = (time.time() - route_start) * 1000

        memory_hits_count = len(mem_result.get("memory_sources", []))
        rag_hits_count = len(rag_result.get("sources", []))
        web_hits_count = sum(
            1
            for s in all_ui_sources
            if isinstance(s, dict) and s.get("type") == "web"
        )

        if self.USE_TELEMETRY:
            self.telemetry.log({
                "route": execution_route,
                "memory_hits": memory_hits_count,
                "rag_hits": rag_hits_count,
                "web_hits": web_hits_count,
                "web_search_attempted": bool(web_search_attempted),
                "latency_ms": latency_ms,
                "memory_chars": len(memory_context),
                "rag_chars": len(tool_context),
            })

            self.router_tuner.observe({
                "route": execution_route,
                "memory_hits": memory_hits_count,
                "rag_hits": rag_hits_count,
                "latency_ms": latency_ms,
            })
            
            try:
                summary = self.telemetry.summarize()
                tuner_state = self.router_tuner.get_weights()
                self.router_telemetry_updated.emit(summary, tuner_state)
            except Exception as e:
                logger.error(f"Failed to emit router telemetry: {e}")

        # 🔑 NEW: Feed the Cognitive V4 Router its learning data!
        if self.USE_COGNITIVE_ROUTER and hasattr(self, 'cognitive_router'):
            # V4 expects latency in seconds, not milliseconds
            latency_seconds = latency_ms / 1000.0 
            # Did we actually use RAG this turn?
            rag_was_used = len(rag_result.get("sources", [])) > 0
            
            self.cognitive_router.record_latency(latency_seconds)
            self.cognitive_router.record_rag_used(rag_was_used)
            logger.debug(f"[Router Feedback] Logged latency: {latency_seconds:.2f}s | RAG used: {rag_was_used}")

        # ============================================================
        # 2.75 T4.1: POST-RETRIEVAL ROUTE DOWNGRADE
        # ------------------------------------------------------------
        # If we routed into a retrieval lane (MEMORY / RAG / HYBRID /
        # WEB / INTERNET) but every channel came back empty or
        # below-floor (rag_tool's MIN_RAG_SEMANTIC_SCORE gate killed
        # all vector candidates, memory_tool's MIN_SEMANTIC_SCORE +
        # proper-noun gate killed all memory candidates, or
        # search_internet was skipped/sentinel-cleared), downgrade
        # this turn to NONE.
        #
        # Why: the prompt-build branch at §3 currently has TWO modes
        # for a retrieval route — the citation-disciplined "you MUST
        # cite your sources" branch (when ``all_ui_sources`` is
        # populated) and the NO_SOURCES fallback. The fallback already
        # existed, but even the NO_SOURCES suffix carries a subtle
        # "you were meant to answer from retrieved sources" framing
        # that biases small LLMs towards "I couldn't find anything in
        # my sources." responses on general-knowledge questions. By
        # downgrading to NONE here, the turn is treated as a plain
        # chat turn and gets the base system prompt + no retrieval
        # wrapper in the user message — the LLM answers from its own
        # knowledge as if no retrieval had been attempted.
        #
        # WEB / INTERNET are included here because the WEB system-
        # prompt branch at §3 asserts "You have just been provided
        # with real-time, live web search results" and instructs the
        # model to cite with ``[W]``. When ``all_ui_sources`` is
        # empty (internet tool disabled, or ``search_internet``
        # returned the "Internet search failed" sentinel and the
        # guard at §2 cleared ``web_results``), the prompt is lying
        # to the model about context that doesn't exist — a small
        # LLM then fabricates both an answer and the ``[W]``
        # citation, which the UI correctly flags as a missing
        # source. Downgrading to NONE on the WEB path lands the
        # turn on the base "You are Qube, be concise" prompt with
        # no ``[W]`` instruction, so the model either answers
        # conservatively from its own parameters or honestly says
        # it can't check live data right now.
        #
        # We do this AFTER telemetry so ``router_telemetry`` still
        # records the original executed route (useful for tuning the
        # cognitive router's thresholds over time). On the WEB path
        # we also mark ``skip_enrichment`` for the same reason
        # ``web_tool_failure`` does on the sentinel path: a turn
        # where the assistant said "I can't check without internet
        # access" should not be mined for user facts.
        # ============================================================
        execution_route_pre_downgrade = execution_route
        tier3_success_flag: bool | None = None

        if (
            execution_route in ("MEMORY", "RAG", "HYBRID", "WEB", "INTERNET")
            and not all_ui_sources
            and not getattr(self, "_composer_knowledge_tool", None)
        ):
            logger.info(
                "[LLM Worker] All retrieval channels empty after relevance "
                "gates; downgrading route %s -> NONE for prompt build.",
                execution_route,
            )
            if execution_route in ("WEB", "INTERNET"):
                self._mark_skip_enrichment("web_route_no_sources")
            execution_route = "NONE"
        elif (
            getattr(self, "_composer_knowledge_tool", None)
            and execution_route in ("WEB", "INTERNET")
            and not all_ui_sources
        ):
            composer_tool = str(getattr(self, "_composer_knowledge_tool", "") or "").lower()
            tool_label = f"@{composer_tool or 'internet'}"
            logger.warning(
                "[LLM Worker] Composer %s: web search returned no "
                "sources; keeping WEB route with empty-results guidance.",
                tool_label,
            )
            self._mark_skip_enrichment("web_route_no_sources")
            if composer_tool == "legal":
                tool_context += (
                    "\n[@legal: No case law sources were retrieved. Preferred legal "
                    "sources may be disabled in Settings → Knowledge. Tell the user "
                    "briefly that case law could not be retrieved. Do NOT answer from "
                    "general model knowledge about cases, holdings, or citations. "
                    "Do NOT emit [1], [2], or [W].]\n"
                )
            elif composer_tool == "finance":
                tool_context += (
                    "\n[@finance: No SEC or finance sources were retrieved. Preferred "
                    "finance sources may be disabled in Settings → Knowledge. Tell the "
                    "user briefly that finance filings could not be retrieved. Do NOT "
                    "answer from general model knowledge about filings or tickers. "
                    "Do NOT emit [1], [2], or [W].]\n"
                )
            elif composer_tool == "help":
                tool_context += (
                    "\n[@help: No matching Qube help documentation was retrieved. "
                    "Tell the user briefly that help docs did not match their question. "
                    "Suggest Settings → Help → Open Qube documentation or rephrasing. "
                    "Do NOT search or cite the user's Main library uploads. "
                    "Do NOT invent settings paths. Do NOT emit [1], [2], or [W].]\n"
                )
                help_entry = getattr(self, "_turn_canonical_help_entry", None)
                canonical_id = (
                    str(help_entry.get("id"))
                    if isinstance(help_entry, dict) and help_entry.get("id")
                    else None
                )
                log_help_query(
                    query=self.prompt,
                    retrieved_doc_ids=[],
                    canonical_id=canonical_id,
                    session_id=self.session_id,
                )
            else:
                tool_context += (
                    "\n[WEB SEARCH: No live results were returned for this query. "
                    "Your first sentence must state that the web search did not return "
                    "usable results right now. Do NOT claim you lack internet access. "
                    "Do NOT invent facts or emit [W] citations without sources.]\n"
                )

        explicit_web_empty_results = bool(
            web_search_attempted
            and not all_ui_sources
            and self.mcp_internet_enabled
            and execution_route in ("NONE", "WEB", "INTERNET")
        )
        prior_web_empty = False
        if self.session_id:
            prior_web_empty = bool(
                self._prior_web_empty_by_session.get(str(self.session_id), False)
            )
        composer_tool_name = str(
            getattr(self, "_composer_knowledge_tool", "") or ""
        ).lower()
        composer_web_empty = bool(
            explicit_web_empty_results
            and composer_tool_name
            and is_web_composer_tool(composer_tool_name)
        )
        prior_web_empty_follow_up = bool(
            follow_up.active
            and prior_web_empty
            and not all_ui_sources
            and not web_capability_blocked
            and execution_route == "NONE"
            and not explicit_web_empty_results
        )
        scientific_medical_disclaimer = False
        financial_disclaimer = False
        legal_disclaimer = False
        composer_tool = str(getattr(self, "_composer_knowledge_tool", "") or "").lower()
        legal_sources_empty = composer_tool == "legal" and not all_ui_sources
        finance_sources_empty = composer_tool == "finance" and not all_ui_sources
        help_sources_empty = composer_tool == "help" and not all_ui_sources
        composer_help_attached = composer_tool == "help"
        _evidence_bundle = getattr(self, "_turn_evidence_bundle", None)
        if _evidence_bundle is not None:
            warnings = _evidence_bundle.warnings or ()
            scientific_medical_disclaimer = "medical_disclaimer" in warnings
            financial_disclaimer = (
                "not_financial_advice" in warnings and not finance_sources_empty
            )
            legal_disclaimer = (
                "not_legal_advice" in warnings and not legal_sources_empty
            )
        self._turn_execution_route = execution_route
        if (
            isinstance(decision, dict)
            and execution_route != "CAPABILITY"
            and not str(decision.get("capability_urn") or "").strip()
            and not str(decision.get("capability_preset_id") or "").strip()
        ):
            try:
                from core.app_settings import get_router_integration_suggestions_enabled
                from core.integrations.router_capability_suggestions import (
                    suggest_integration_capabilities,
                )

                if get_router_integration_suggestions_enabled():
                    suggestions = suggest_integration_capabilities(
                        clean_prompt or self.prompt
                    )
                    if suggestions:
                        decision["integration_capability_suggestions"] = suggestions
            except Exception:
                logger.debug(
                    "[Router] integration capability suggestions failed",
                    exc_info=True,
                )
        if self.session_id:
            self._prior_execution_route_by_session[str(self.session_id)] = execution_route
            self._prior_web_empty_by_session[str(self.session_id)] = (
                explicit_web_empty_results
            )

        # ============================================================
        # 2.76 TIER 3: emit RouteFeedbackEvent for the cognitive
        # router's bounded adaptive calibration layer.
        # ------------------------------------------------------------
        # MUST run AFTER the post-retrieval downgrade above so the
        # ``success`` signal reflects the genuine post-gate state
        # — exactly what Tier 1's downgrade itself trusts.
        #
        # Skipped when:
        #   * ``USE_COGNITIVE_ROUTER`` is False (no router instance),
        #   * ``decision["drift"]`` is True (retrieval was suppressed
        #     for an unrelated reason; signal is not informative),
        #   * the original routed lane was ``none`` (no retrieval was
        #     attempted, so there is nothing to calibrate against).
        #
        # ``per_lane_hits`` uses the same channel counts the existing
        # ``router_tuner.observe(...)`` block reads, plus a deterministic
        # ``web_hits`` derived from ``all_ui_sources`` (web items the
        # UI actually received this turn). For ``hybrid`` the registry
        # credits each retrieval lane independently from this dict, so
        # a hybrid where only RAG returned data correctly credits RAG
        # with success and MEMORY with failure.
        #
        # Wrapped in try/except: a calibration-record failure must
        # NEVER crash a user-facing turn. Mirrors the existing
        # try/except around ``router_telemetry_updated.emit(...)``.
        # ============================================================
        if (
            self.USE_COGNITIVE_ROUTER
            and hasattr(self, 'cognitive_router')
            and isinstance(decision, dict)
        ):
            original_route = str(decision.get("route") or "none").lower()
            is_drift = bool(decision.get("drift", False))
            if not is_drift and original_route != "none":
                try:
                    per_lane_hits = {
                        "memory": memory_hits_count,
                        "rag":    rag_hits_count,
                        "web":    web_hits_count,
                    }

                    if original_route == "hybrid":
                        success_flag = (memory_hits_count > 0) or (rag_hits_count > 0)
                    elif original_route in ("memory", "rag", "web"):
                        success_flag = per_lane_hits[original_route] > 0
                    else:
                        success_flag = False

                    tier3_success_flag = bool(success_flag)

                    feedback_event = RouteFeedbackEvent(
                        route=original_route,
                        top_intent=str(decision.get("top_intent") or original_route),
                        top_source=str(decision.get("top_intent_source") or "substring"),
                        confidence_margin=float(decision.get("confidence_margin") or 0.0),
                        latency_ms=float(latency_ms),
                        success=bool(success_flag),
                        drift=False,
                        per_lane_hits=per_lane_hits,
                    )
                    self.cognitive_router.observe_feedback(feedback_event)
                except Exception as e:
                    logger.warning(f"[Tier3 Feedback] Failed to emit RouteFeedbackEvent: {e}")

        retrieval_outcome_snapshot: dict | None = None
        try:
            if isinstance(decision, dict):
                retrieval_outcome_snapshot = build_retrieval_outcome_snapshot(
                    decision=decision,
                    execution_route_pre_downgrade=str(execution_route_pre_downgrade),
                    execution_route_final=str(execution_route),
                    memory_hits=memory_hits_count,
                    rag_hits=rag_hits_count,
                    web_hits=web_hits_count,
                    hybrid_extra_memory=int(
                        getattr(self, "_sidecar_hybrid_extra_memory", 0) or 0
                    ),
                    hybrid_extra_rag=int(
                        getattr(self, "_sidecar_hybrid_extra_rag", 0) or 0
                    ),
                    tier3_success=tier3_success_flag,
                )
                updated = self.routing_debug_buffer.merge_retrieval_outcome_into_latest(
                    retrieval_outcome_snapshot
                )
                if updated:
                    self.routing_debug_record_added.emit(dataclasses.asdict(updated))
                    self._persist_routing_debug_record(updated)
        except Exception as e:
            logger.warning("[RoutingDebug] failed to merge retrieval outcome: %s", e)

        try:
            rewrite_attempted = bool(
                discourse_enabled
                and get_sidecar_query_rewrite_enabled()
                and follow_up.active
            )
            retrieval_meta: dict = {}
            if isinstance(retrieval_outcome_snapshot, dict):
                retrieval_meta = {
                    "router_route": retrieval_outcome_snapshot.get("router_route"),
                    "execution_route_final": retrieval_outcome_snapshot.get(
                        "execution_route_final"
                    ),
                    "downgrade_fired": retrieval_outcome_snapshot.get("downgrade_fired"),
                    "memory_hits": retrieval_outcome_snapshot.get("memory_hits"),
                    "rag_hits": retrieval_outcome_snapshot.get("rag_hits"),
                }
            get_sidecar_telemetry().record_turn(
                rewrite_attempted=rewrite_attempted,
                rewrite_applied=query_expansion is not None,
                rewrite_confidence=(
                    float(query_expansion.confidence)
                    if query_expansion is not None
                    else 0.0
                ),
                digest_memory_attempted=digest_mem_attempted,
                digest_memory_applied=digest_mem_applied,
                digest_rag_attempted=digest_rag_attempted,
                digest_rag_applied=digest_rag_applied,
                digest_memory_chars_before=digest_mem_chars_before,
                digest_memory_chars_after=digest_mem_chars_after,
                digest_rag_chars_before=digest_rag_chars_before,
                digest_rag_chars_after=digest_rag_chars_after,
                digest_memory_skip_reason=digest_mem_skip_reason,
                digest_rag_skip_reason=digest_rag_skip_reason,
                hybrid_extra_memory=int(
                    getattr(self, "_sidecar_hybrid_extra_memory", 0) or 0
                ),
                hybrid_extra_rag=int(
                    getattr(self, "_sidecar_hybrid_extra_rag", 0) or 0
                ),
                meta=retrieval_meta or None,
            )
            self.sidecar_telemetry_updated.emit(get_sidecar_telemetry().summarize())
        except Exception as e:
            logger.debug("Failed to emit sidecar telemetry: %s", e)

        # ============================================================
        # 2.5 UNIFIED RETRIEVAL PROMPT (order: memory → RAG → web; ids [1]..[n] match UI)
        # ============================================================
        retrieval_prompt_body = self._format_sources_for_llm_prompt(
            all_ui_sources,
            format_mode=(
                "background"
                if execution_route == "NONE"
                and all_ui_sources
                and all(
                    str(s.get("type", "")).lower() == "memory"
                    for s in all_ui_sources
                    if isinstance(s, dict)
                )
                else "grounded"
            ),
        )
        attachment_ctx = getattr(self, "_turn_attachment_context", "").strip()
        if attachment_ctx:
            if retrieval_prompt_body:
                retrieval_prompt_body = (
                    f"{attachment_ctx}\n\n{retrieval_prompt_body}"
                )
            else:
                retrieval_prompt_body = attachment_ctx
            logger.info(
                "[LLM Worker] Injected composer attachment context (%d chars)",
                len(attachment_ctx),
            )
        if tool_context.strip() and (
            execution_route == "CAPABILITY"
            or (
                getattr(self, "_composer_knowledge_tool", None)
                and not all_ui_sources
            )
        ):
            if retrieval_prompt_body:
                retrieval_prompt_body = (
                    f"{tool_context.strip()}\n\n{retrieval_prompt_body}"
                )
            else:
                retrieval_prompt_body = tool_context.strip()
            logger.info(
                "[LLM Worker] Injected capability/tool context (%d chars, route=%s)",
                len(tool_context.strip()),
                execution_route,
            )
        if retrieval_prompt_body:
            retrieval_prompt_body = retrieval_prompt_body[: self.MAX_TOTAL_RETRIEVAL_CHARS]

        has_retrieval_prompt = bool((retrieval_prompt_body or "").strip())
        effective_has_retrieval = bool(all_ui_sources) or has_retrieval_prompt

        # Conversation @-ref: do not send unrelated prior turns from *this* session.
        # Otherwise the model answers from current-thread noise instead of the transcript.
        prompt_history = history
        if attachment_conversation_active:
            question = (self.prompt or "").strip()
            if not question:
                question = "What is the attached conversation about?"
            prompt_history = [{"role": "user", "content": question}]
            if len(history) > 1:
                logger.info(
                    "[LLM Worker] Conversation @-ref: isolated prompt turn "
                    "(omitted %d other messages from active session)",
                    len(history) - 1,
                )
        elif discourse_enabled and history and history[-1].get("role") == "user":
            grounded = list(history)
            last = dict(grounded[-1])
            if prompt_rewrite is not None and prompt_rewrite.applied:
                last["content"] = prompt_rewrite.grounded
            grounded[-1] = last
            prompt_history = grounded

        # ============================================================
        # 3. PROMPT BUILD
        # ============================================================
        pl_res = self._resolve_turn_prompt_layout()
        logger.info(
            "[PromptLayout] turn layout=%s source=%s degraded=%s route=%s",
            pl_res.layout,
            pl_res.source,
            pl_res.degraded,
            execution_route,
        )

        memory_only_sources = bool(all_ui_sources) and all(
            str(s.get("type", "")).lower() == "memory"
            for s in all_ui_sources
            if isinstance(s, dict)
        )
        retrieval_wrapper_mode = resolve_retrieval_wrapper_mode(
            execution_route,
            effective_has_retrieval,
            memory_only_sources=memory_only_sources,
        )
        self._stamp_discourse_on_decision(
            decision,
            follow_up=follow_up,
            discourse_state=discourse_state if discourse_enabled else None,
            routing_query=routing_query,
            retrieval_query=retrieval_query,
            core_memory_suppressed=core_memory_suppressed,
            retrieval_wrapper_mode=retrieval_wrapper_mode,
            inference_user_text=inference_user_text,
            resolved_query=resolved_query,
            prompt_rewrite=prompt_rewrite,
            resolved_retrieval=resolved_retrieval,
        )
        self._stamp_query_expansion_on_decision(
            decision,
            original_query=original_query,
            retrieval_query=retrieval_query,
            expansion=query_expansion,
        )

        topic_salience = ""
        prior_turn_unreliable = ""
        self._turn_prior_turn_suppressed = history_contains_suppressed_assistant(
            prompt_history
        )
        if self._turn_prior_turn_suppressed:
            prior_turn_unreliable = build_prior_turn_unreliable_suffix()
            logger.info(
                "[HistoryDegeneration] prior assistant turn suppressed; "
                "injecting reliability hint"
            )
        if discourse_enabled and follow_up.active and discourse_state and salience_anchor:
            if discourse_prompt_hint_enabled() and salience_reason.startswith(
                "referent_salience"
            ):
                topic_salience = build_referent_salience_suffix(
                    salience_anchor,
                    referent_type=discourse_state.referent_type,
                )
            elif salience_reason.startswith("referent_salience"):
                topic_salience = build_entity_aspect_grounding_suffix(
                    salience_anchor,
                    aspect=(discourse_state.active_aspect or ""),
                    entity_type=discourse_state.referent_type,
                )
            elif salience_reason == "topic_salience":
                topic_salience = build_topic_salience_suffix(
                    salience_anchor,
                    topic_type=discourse_state.topic_type,
                )

        turn_ctx = self._resolve_turn_context_for_turn(
            execution_route=execution_route,
            follow_up=follow_up,
            prior_turn_unreliable=bool(prior_turn_unreliable),
            has_retrieval_sources=effective_has_retrieval,
            history_turn_count=len(prompt_history),
            conversation_health=conversation_health,
        )

        skill_ctx = build_skill_context(
            user_query=self.prompt,
            clean_query=clean_prompt,
            execution_route=execution_route,
            all_ui_sources=all_ui_sources,
            follow_up_active=follow_up.active,
            explicit_remember_active=explicit_remember_active,
            file_search_active=file_search_active,
            narrative_active=narrative_active,
            decision=decision if isinstance(decision, dict) else None,
            query_embedding=(
                query_vector if query_vector is not None else intent_vector
            ),
            web_capability_blocked=web_capability_blocked,
            explicit_web_empty_results=explicit_web_empty_results,
            rag_capability_blocked=rag_capability_blocked,
            knowledge_service=(
                str(decision.get("knowledge_service"))
                if isinstance(decision, dict) and decision.get("knowledge_service")
                else None
            ),
            evidence_summary=(
                _evidence_bundle.summary_for_skills()
                if _evidence_bundle is not None
                else None
            ),
        )
        skill_result = activate_skills(
            skill_ctx,
            settings=get_skill_settings(),
            forced_skill_ids=tuple(
                dict.fromkeys(
                    (
                        *(getattr(self, "_turn_enforced_skills", ()) or ()),
                        *(
                            ("scientific_research",)
                            if _evidence_bundle is not None
                            and _evidence_bundle.knowledge_service == "scientific_evidence"
                            and _evidence_bundle.sources
                            else ()
                        ),
                    )
                )
            ),
        )
        if get_skills_debug_log_enabled():
            from core.skills.debug_sink import attach_skills_debug_file_sink, log_skill_activation

            attach_skills_debug_file_sink()
            log_skill_activation(
                {
                    "query": (clean_prompt or "")[:200],
                    "route": execution_route,
                    **skill_result.telemetry_dict(),
                }
            )
        if hasattr(self, "routing_debug_buffer"):
            updated_skills = self.routing_debug_buffer.merge_skills_into_latest(
                skill_result.telemetry_dict()
            )
            if updated_skills is not None:
                self.routing_debug_record_added.emit(dataclasses.asdict(updated_skills))
                self._persist_routing_debug_record(updated_skills)

        prompt_blocks = build_prompt_blocks(
            execution_route=execution_route,
            explicit_remember_active=explicit_remember_active,
            explicit_remember_body=explicit_remember_body or "",
            file_search_active=file_search_active,
            narrative_active=narrative_active,
            has_retrieval_sources=effective_has_retrieval,
            engine_mode=getattr(self, "engine_mode", DEFAULT_ENGINE_MODE),
            internal_nvidia_family=self._is_internal_nvidia_family(),
            retrieval_context=retrieval_prompt_body,
            conversation_history=prompt_history,
            composer_conversation_ref=attachment_conversation_active,
            web_capability_blocked=web_capability_blocked,
            explicit_web_empty_results=explicit_web_empty_results,
            composer_web_empty=composer_web_empty,
            prior_web_empty_follow_up=prior_web_empty_follow_up,
            scientific_medical_disclaimer=scientific_medical_disclaimer,
            financial_disclaimer=financial_disclaimer,
            legal_disclaimer=legal_disclaimer,
            legal_sources_empty=legal_sources_empty,
            finance_sources_empty=finance_sources_empty,
            composer_help_attached=composer_help_attached,
            help_sources_empty=help_sources_empty,
            help_canonical_hint=getattr(self, "_help_canonical_system_hint", ""),
            rag_capability_blocked=rag_capability_blocked,
            strict_isolation_enabled=self.mcp_strict_enabled,
            preference_context=preference_policy.compact_prompt_context(
                query=self.prompt,
                route=execution_route,
            ),
            apply_preference_suffix=preference_policy.has_presentation_prefs(),
            retrieval_wrapper_mode=retrieval_wrapper_mode,
            topic_salience_hint=topic_salience,
            follow_up_active=follow_up.active,
            prior_turn_unreliable_hint=prior_turn_unreliable,
            chat_personality_enabled=get_enable_chat_personality_nudge(),
            reply_shape_hint=turn_ctx.reply_shape.system_reply_hint,
            skill_guidance=skill_result.prompt_block,
            retrieval_source_count=len(all_ui_sources),
            web_hit_count=sum(
                1
                for s in all_ui_sources
                if isinstance(s, dict) and str(s.get("type", "")).lower() == "web"
            ),
        )
        if prompt_blocks.no_sources_mode:
            logger.info(
                "[LLM Worker] No sources survived retrieval filtering; "
                "switching to NO_SOURCES system prompt (route=%s).",
                execution_route,
            )

        messages = render_messages(prompt_blocks, pl_res.layout)
        roles = [str(m.get("role", "")) for m in messages]
        history_chars = sum(len(str(m.get("content") or "")) for m in prompt_history)
        self._log_discourse_debug(
            follow_up=follow_up,
            discourse_state=discourse_state if discourse_enabled else None,
            roles=roles,
            history_chars=history_chars,
            retrieval_chars=len(retrieval_prompt_body or ""),
            query_chars=len((self.prompt or "")),
            retrieval_wrapper_mode=retrieval_wrapper_mode,
            core_memory_suppressed=core_memory_suppressed,
        )
        logger.info(
            "[PromptLayout] rendered layout=%s roles=%s has_system=%s",
            pl_res.layout,
            roles,
            "system" in roles,
        )
        if retrieval_prompt_body and messages and messages[-1].get("role") == "user":
            logger.debug("Successfully injected unified retrieval context into the final prompt.")

        # ============================================================
        # 4. LLM STREAMING
        # ============================================================
        self.status_update.emit("Synthesizing...")

        final_text = ""

        gen_params = self._generation_params_from_turn_context(
            turn_ctx,
            native=getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal",
        )

        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal" and self._native_engine:
            final_text = self._stream_via_native(
                messages,
                all_ui_sources,
                retrieval_context=retrieval_prompt_body,
                execution_route=execution_route,
                turn_ctx=turn_ctx,
                gen_params=gen_params,
            )
            return final_text

        chat_format_mode = turn_ctx.chat_format_mode
        self._safe_truth_diff_l2_prompt(
            json.dumps(messages, ensure_ascii=False),
            self._truth_diff_l2_metadata(
                template_source=f"worker_messages/{pl_res.layout}",
                chat_format_mode=chat_format_mode,
                execution_mode="",
                prompt_contract_mode="messages",
                prompt_layout=pl_res.layout,
            ),
        )

        payload = {
            "messages": messages,
            "temperature": gen_params["temperature"],
            "max_tokens": gen_params["max_tokens"],
            "stream": True,
            **gen_params["sampling_overrides"],
        }
        if self._uses_external_http() and self._is_local_llm_service():
            # llama.cpp server: avoid unbounded prompt-prefix / KV reuse across unrelated requests
            payload["cache_prompt"] = False

        self._safe_truth_diff_l1_engine_request(payload)

        current_sentence = ""
        final_text = ""
        external_finish_reason = ""
        start = time.time()
        first_token = False
        first_token_ts: float | None = None
        output_token_count = 0
        self._reset_tts_dedupe_state()

        try:
            self._active_stream_response = requests.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=(self._STREAM_CONNECT_TIMEOUT, self._STREAM_READ_TIMEOUT),
                headers={"Connection": "close"},
            )
            r = self._active_stream_response
            r.raise_for_status()

            stream_wall_start = time.time()
            repetition_guard = create_stream_repetition_guard(turn_ctx.generation_risk)
            _health = turn_ctx.conversation_health
            degeneration_observer = OutputDegenerationStreamObserver(
                rescore_every=(
                    _health.degeneration_rescore_every if _health is not None else 120
                )
            )

            for line in r.iter_lines(decode_unicode=False):
                if time.time() - stream_wall_start > self._MAX_STREAM_WALL_SECONDS:
                    logger.error("[LLM] SSE stream exceeded wall-time cap; closing.")
                    break
                if getattr(self, "_cancel_requested", False):
                    break

                if not line:
                    continue

                data = line.decode("utf-8")

                if data.startswith("data: "):
                    chunk = data[6:]
                    if chunk.strip() == "[DONE]":
                        break

                    try:
                        packet = json.loads(chunk)
                        choice0 = packet["choices"][0]
                        delta = choice0.get("delta", {}).get("content", "")
                        finish = choice0.get("finish_reason")
                        if finish:
                            external_finish_reason = str(finish)

                        if delta:
                            if not first_token:
                                self.ttft_latency.emit((time.time() - start) * 1000)
                                first_token = True
                                first_token_ts = time.time()

                            current_sentence += delta
                            final_text += delta
                            output_token_count += self._estimate_output_tokens(delta)
                            self.token_streamed.emit(self.session_id or "", delta)

                            if any(p in delta for p in ".!?"):
                                self._queue_tts_sentence(current_sentence)
                                current_sentence = ""

                            if repetition_guard.observe(delta):
                                logger.error(
                                    "[LLM] SSE stream degeneration detected (%s); cancelling.",
                                    repetition_guard.trip_reason,
                                )
                                self._mark_stream_degeneration_cancelled()
                                self._mark_skip_enrichment("stream_repetition_cancelled")
                                break

                            if degeneration_observer.observe(delta):
                                logger.error(
                                    "[LLM] Output degeneration HIGH during stream (%s); cancelling.",
                                    degeneration_observer.trip_reason,
                                )
                                self._mark_stream_degeneration_cancelled()
                                self._mark_skip_enrichment("output_degeneration_stream_cancelled")
                                break

                    except json.JSONDecodeError:
                        continue

            raw_external_text = final_text
            if final_text:
                final_text = strip_output_artifacts(final_text, harmony_active=False)

            if final_text.strip():
                trunc_reason = probable_max_tokens_truncation(
                    final_text,
                    stream_finish_reason=str(external_finish_reason or ""),
                    max_tokens=int(gen_params.get("max_tokens", 0) or 0),
                    limit_enabled=bool(
                        getattr(self, "output_token_limit_enabled", True)
                    ),
                    completion_token_count=output_token_count or None,
                )
                if trunc_reason:
                    self.turn_notice.emit(
                        self.session_id or "",
                        {"kind": "max_tokens", "reason": trunc_reason},
                    )

            if current_sentence.strip():
                self._queue_tts_sentence(current_sentence)

            self._completion_output_snapshot = CompletionOutputSnapshot(
                engine_mode="external",
                raw_text=raw_external_text or "",
                after_worker_filters=final_text or "",
                worker_return_text=final_text or "",
            )

            if self.session_id and final_text.strip():
                final_text = self._append_help_canonical_action_if_needed(final_text)
                final_text, all_ui_sources, _ = self._finalize_turn_citations(
                    final_text,
                    all_ui_sources,
                )
                self._persist_assistant_turn(final_text, all_ui_sources)

            self._successfully_finished = True

        except requests.exceptions.Timeout:
            logger.error("LLM Connection Error: Request timed out.")
            final_text = "Sorry, my brain disconnected (Timeout)."
            self.token_streamed.emit(self.session_id or "", "\n\n*(Connection Timeout)*")

        except Exception as e:
            logger.error(f"LLM Connection Error: {e}")
            final_text = "Sorry, my brain encountered an error."
            self.token_streamed.emit(self.session_id or "", "\n\n*(Connection Error)*")

        finally:
            self._close_active_stream()
            self._emit_output_tps(output_token_count, first_token_ts)

        self._persist_latest_routing_debug_record()
        return final_text

    def _max_tokens_completion(self, *, native: bool) -> int:
        """Budget for new completion tokens (not n_ctx). Native reclamps after prompt."""
        return resolve_output_token_budget(
            context_window=max(512, int(getattr(self, "context_window", 4096))),
            limit_enabled=bool(getattr(self, "output_token_limit_enabled", True)),
            user_limit=int(getattr(self, "output_token_limit", 4096)),
        )

    def _max_tokens_native_completion(self) -> int:
        return self._max_tokens_completion(native=True)

    def _stream_via_native(
        self,
        messages: list[dict],
        all_ui_sources: list,
        *,
        retrieval_context: str = "",
        execution_route: str = "NONE",
        turn_ctx: TurnContext | None = None,
        gen_params: dict[str, Any] | None = None,
        chat_format_mode: str | None = None,
    ) -> str:
        """Stream native output after a small leading-meta/thinking gate.

        The first few chunks may contain "Provide final answer" / thinking tags; filters may
        briefly buffer those openers, but once real answer text starts, UI and TTS both stream
        the same cleaned fragments normally.
        """
        self._reset_tts_dedupe_state()
        token_queue: queue.Queue = queue.Queue()
        done_event = threading.Event()
        if turn_ctx is not None:
            chat_format_mode = turn_ctx.chat_format_mode
        elif chat_format_mode is None:
            turn_ctx = getattr(self, "_turn_context", None)
            chat_format_mode = (
                turn_ctx.chat_format_mode if turn_ctx else "structured"
            )
        if gen_params is None:
            if turn_ctx is None:
                turn_ctx = getattr(self, "_turn_context", None)
            gen_params = (
                self._generation_params_from_turn_context(turn_ctx, native=True)
                if turn_ctx is not None
                else {
                    "temperature": self.temperature,
                    "max_tokens": self._max_tokens_native_completion(),
                    "sampling_overrides": self._sampling_payload(),
                }
            )
        if generation_debug_enabled():
            gen_params = apply_debug_sampling_overrides(gen_params)
        turn_id = int(getattr(self, "_routing_debug_turn_seq", 0))
        native_engine = self._native_engine
        preflight = getattr(native_engine, "_last_trace_preflight", None) or {}
        merged_stops = getattr(native_engine, "_last_merged_stops", None)
        gen_debug = GenerationDebugRecorder.maybe_start(
            turn_id=turn_id,
            session_id=str(self.session_id or ""),
            user_query=str(self.prompt or ""),
            gen_params=gen_params,
            native_preflight=preflight,
            merged_stops=list(merged_stops or []),
        )
        self._active_generation_debug_recorder = gen_debug
        logger.info(
            "[ChatFormatMode] input_source=%s mode=%s route=%s query_preview=%r risk=%s",
            getattr(self, "_input_source", "text"),
            chat_format_mode,
            execution_route,
            (str(self.prompt or "")[:80]),
            (
                turn_ctx.generation_risk.risk_tier
                if turn_ctx is not None
                else "unknown"
            ),
        )
        self._engine_enqueue_at = time.monotonic()
        self._native_engine.enqueue_generation(
            messages,
            float(gen_params["temperature"]),
            int(gen_params["max_tokens"]),
            token_queue,
            done_event,
            retrieval_context=(retrieval_context or "").strip(),
            chat_format_mode=chat_format_mode,
            reply_shape_policy=(
                turn_ctx.reply_shape if turn_ctx is not None else None
            ),
            sampling_overrides=gen_params.get("sampling_overrides"),
            output_token_limit_enabled=bool(
                getattr(self, "output_token_limit_enabled", True)
            ),
            context_window=int(getattr(self, "context_window", 4096)),
            debug_caller="chat",
            debug_exchange_id=getattr(self, "_debug_exchange_id", None),
            debug_session_id=str(self.session_id or ""),
        )

        _native_exec_policy = self._native_engine.get_execution_policy()
        cot_filter = (
            RedactedThinkingStreamFilter()
            if bool(_native_exec_policy.strip_thinking_output)
            else None
        )
        meta_filter = LeadingMetaInstructionStripper()
        _native_telemetry = self._native_engine.get_model_reasoning_telemetry() or {}
        use_gemma_strip = is_gemma_model_identity(
            model_name=str(_native_telemetry.get("model_name") or ""),
            model_path=str(getattr(self._native_engine, "_model_path", "") or ""),
        )
        gemma_filter = GemmaThoughtStreamFilter() if use_gemma_strip else None
        risk_profile = turn_ctx.generation_risk if turn_ctx is not None else None
        repetition_guard = create_stream_repetition_guard(risk_profile)
        _health = turn_ctx.conversation_health if turn_ctx is not None else None
        degeneration_observer = OutputDegenerationStreamObserver(
            rescore_every=(
                _health.degeneration_rescore_every if _health is not None else 120
            )
        )
        prompt_contract = getattr(self._native_engine, "_last_prompt_contract", None)
        use_harmony_parser = bool(
            is_harmony_contract(prompt_contract) and harmony_stream_parser_enabled()
        )
        harmony_parser = HarmonyStreamParser() if use_harmony_parser else None
        harmony_active = self._harmony_model_active()
        output_profile: TemplateOutputProfile | None = None
        if self._native_engine is not None:
            output_profile = self._native_engine.get_template_output_profile()
        delimiter_filter = (
            DelimiterGrammarStreamFilter(output_profile)
            if output_profile is not None and output_profile.grammar_tier == "delimiter"
            else None
        )
        reasoning_family_harmony_leak_strip = (
            self._reasoning_family_harmony_leak_strip_active()
            and delimiter_filter is None
        )
        harmony_scaffold_filter = (
            LeakedHarmonyScaffoldStreamFilter()
            if harmony_parser is None and not harmony_active and delimiter_filter is None
            else None
        )
        current_sentence = ""
        final_text = ""
        raw_parts: list[str] = []
        native_end_text = ""
        native_load_error_text = ""
        stream_output_superseded = False
        streamed_before_replace = ""
        start = time.time()
        first_token = False
        first_token_ts: float | None = None
        stream_wall_start = time.time()
        output_token_count = 0
        harmony_cut_cancelled = False

        def _sanitize_complete_native_text(raw_text: str) -> str:
            return sanitize_output_for_validation(
                raw_text,
                harmony_active=harmony_active,
                policy=_native_exec_policy,
                reasoning_family=reasoning_family_harmony_leak_strip,
            )

        def _apply_stream_thinking_filter(text: str) -> str:
            if cot_filter is None:
                return text
            return cot_filter.feed(text)

        def _abort_harmony_tts_tail() -> None:
            nonlocal current_sentence
            current_sentence = ""
            self._reset_tts_dedupe_state()
            self.tts_turn_superseded.emit(self.session_id or "")

        def _emit_filtered(fragment: str, *, speak: bool = True) -> None:
            nonlocal current_sentence, final_text, first_token, first_token_ts, output_token_count
            if not fragment:
                return
            if harmony_parser is not None and is_harmony_orphan_stream_fragment(fragment):
                return
            if harmony_parser is None and delimiter_filter is None:
                fragment = strip_output_artifacts(
                    fragment,
                    harmony_active=harmony_active,
                    reasoning_family=reasoning_family_harmony_leak_strip,
                )
            if not fragment:
                return
            if not first_token:
                self.ttft_latency.emit((time.time() - start) * 1000)
                first_token = True
                first_token_ts = time.time()
            final_text += fragment
            output_token_count += self._estimate_output_tokens(fragment)
            self.token_streamed.emit(self.session_id or "", fragment)
            current_sentence += fragment
            if speak and any(p in fragment for p in ".!?"):
                self._queue_tts_sentence(current_sentence)
                current_sentence = ""

        def _flush_tail() -> None:
            tail = ""
            if harmony_parser is not None:
                tail = harmony_parser.flush()
            elif delimiter_filter is not None:
                tail = delimiter_filter.flush()
            if gemma_filter is not None:
                tail = gemma_filter.feed(tail)
                tail += gemma_filter.flush()
            if cot_filter is not None:
                tail = cot_filter.feed(tail)
                tail += cot_filter.flush()
            tail = meta_filter.feed(tail) + meta_filter.flush()
            if harmony_scaffold_filter is not None:
                tail = harmony_scaffold_filter.feed(tail)
                tail += harmony_scaffold_filter.flush()
            _emit_filtered(tail)

        saw_end = False
        while True:
            if time.time() - stream_wall_start > self._MAX_STREAM_WALL_SECONDS:
                logger.error("[LLM] Native stream exceeded wall-time cap.")
                self._native_engine.request_cancel_generation()
                break
            if getattr(self, "_cancel_requested", False):
                self._native_engine.request_cancel_generation()
            try:
                kind, data = token_queue.get(timeout=0.2)
            except queue.Empty:
                if done_event.is_set() and token_queue.empty():
                    break
                continue

            if kind == "delta":
                raw = data
                raw_text = str(raw or "")
                raw_parts.append(raw_text)
                if harmony_parser is not None:
                    stream_in = harmony_parser.feed(raw_text)
                elif delimiter_filter is not None:
                    stream_in = delimiter_filter.feed(raw_text)
                else:
                    stream_in = raw_text
                stream_piece = stream_in
                if gemma_filter is not None:
                    stream_piece = gemma_filter.feed(stream_in)
                clean_piece = meta_filter.feed(_apply_stream_thinking_filter(stream_piece))
                if harmony_scaffold_filter is not None:
                    clean_piece = harmony_scaffold_filter.feed(clean_piece)
                chunk_events: dict[str, Any] = {}
                if gen_debug is not None:
                    repair_triggers: list[str] = []
                    if harmony_parser is not None and stream_in != raw_text:
                        repair_triggers.append("harmony_parser")
                    if delimiter_filter is not None and stream_in != raw_text:
                        repair_triggers.append("delimiter_grammar")
                    if gemma_filter is not None and stream_piece != stream_in:
                        repair_triggers.append("gemma_thought_filter")
                    if clean_piece != stream_in:
                        repair_triggers.append("cot_or_meta_filter")
                    if repair_triggers:
                        chunk_events["repair_triggers"] = repair_triggers
                _emit_filtered(clean_piece)
                if gen_debug is not None:
                    gen_debug.record_delta(
                        delta=raw_text,
                        cumulative_raw="".join(raw_parts),
                        cumulative_filtered=final_text,
                        events=chunk_events,
                    )
                if harmony_parser is not None and final_text.strip():
                    if harmony_parser.degeneration_detected or harmony_tail_degenerate(
                        harmony_parser.raw_seen
                    ):
                        logger.info(
                            "[LLM] Harmony degeneration detected; cancelling generation."
                        )
                        harmony_cut_cancelled = True
                        if gen_debug is not None:
                            gen_debug.note_stream_cancel("harmony_degeneration")
                        self._mark_stream_degeneration_cancelled()
                        self._mark_skip_enrichment("harmony_degeneration_cancelled")
                        _abort_harmony_tts_tail()
                        self._native_engine.request_cancel_generation()
                        saw_end = True
                        break
                if clean_piece and repetition_guard.observe(clean_piece):
                    logger.error(
                        "[LLM] Native stream degeneration detected (%s); cancelling.",
                        repetition_guard.trip_reason,
                    )
                    if gen_debug is not None:
                        gen_debug.note_stream_cancel(
                            f"stream_repetition:{repetition_guard.trip_reason}"
                        )
                    self._mark_stream_degeneration_cancelled()
                    self._mark_skip_enrichment("stream_repetition_cancelled")
                    self._native_engine.request_cancel_generation()
                    _flush_tail()
                    saw_end = True
                    break
                if clean_piece and degeneration_observer.observe(clean_piece):
                    logger.error(
                        "[LLM] Output degeneration HIGH during native stream (%s); cancelling.",
                        degeneration_observer.trip_reason,
                    )
                    if gen_debug is not None:
                        gen_debug.note_stream_cancel(
                            f"output_degeneration:{degeneration_observer.trip_reason}"
                        )
                    self._mark_stream_degeneration_cancelled()
                    self._mark_skip_enrichment("output_degeneration_stream_cancelled")
                    self._native_engine.request_cancel_generation()
                    _abort_harmony_tts_tail()
                    saw_end = True
                    break
            elif kind == "recovery":
                raw_text = str(data or "")
                if raw_text:
                    raw_parts.append(raw_text)
                if harmony_parser is not None:
                    stream_in = harmony_parser.feed(raw_text)
                elif delimiter_filter is not None:
                    stream_in = delimiter_filter.feed(raw_text)
                else:
                    stream_in = raw_text
                stream_piece = stream_in
                if gemma_filter is not None:
                    stream_piece = gemma_filter.feed(stream_in)
                clean_piece = meta_filter.feed(_apply_stream_thinking_filter(stream_piece))
                if harmony_scaffold_filter is not None:
                    clean_piece = harmony_scaffold_filter.feed(clean_piece)
                _emit_filtered(clean_piece, speak=False)
            elif kind == "replace":
                replacement = str(data or "").strip()
                streamed_snapshot = strip_output_artifacts(
                    final_text,
                    harmony_active=harmony_active,
                    reasoning_family=reasoning_family_harmony_leak_strip,
                ).strip()
                streamed_before_replace = streamed_snapshot
                resolved, rejection = resolve_stream_replacement(
                    replacement,
                    streamed_snapshot,
                    harmony_active=harmony_active,
                    reasoning_family=reasoning_family_harmony_leak_strip,
                )
                val_trace = getattr(
                    self._native_engine, "_last_output_validation_trace", None
                )
                if val_trace is not None:
                    val_trace.streamed_visible_len = len(streamed_snapshot)
                if rejection:
                    if val_trace is not None:
                        val_trace.replacement_suppressed = True
                        val_trace.replacement_rejection_reason = rejection
                        log_output_validation_trace(
                            session_id=str(self.session_id or ""),
                            trace=val_trace,
                            phase="stream_replace",
                        )
                    continue
                replacement = resolved
                stream_output_superseded = True
                native_end_text = replacement
                final_text = replacement
                raw_parts.clear()
                if replacement:
                    raw_parts.append(replacement)
                current_sentence = ""
                self._reset_tts_dedupe_state()
                self.tts_turn_superseded.emit(self.session_id or "")
                self.stream_replaced.emit(self.session_id or "", replacement)
            elif kind == "notice":
                payload = data if isinstance(data, dict) else {"kind": str(data or "")}
                self.turn_notice.emit(self.session_id or "", payload)
            elif kind == "error":
                self.token_streamed.emit(self.session_id or "", f"\n\n*({data})*")
                err_txt = str(data or "")
                if "native model not loaded" in err_txt.lower():
                    # Persist this as the assistant turn in SQLite so the next
                    # user message does not leave back-to-back user roles in
                    # chat history (breaks Mistral flatten_user prompts).
                    native_load_error_text = err_txt.strip()
                    self._mark_skip_enrichment("native_model_not_loaded")
                    self.status_update.emit("Load a Model")
                    self._queue_tts_sentence(err_txt)
            elif kind == "end":
                native_end_text = str(data or "")
                _flush_tail()
                saw_end = True
                break

        if not saw_end:
            _flush_tail()

        if (
            not stream_output_superseded
            and current_sentence.strip()
            and not harmony_cut_cancelled
        ):
            self._queue_tts_sentence(current_sentence)
            current_sentence = ""

        emitted_text = (
            final_text.strip()
            if delimiter_filter is not None
            else strip_output_artifacts(
                final_text,
                harmony_active=harmony_active,
                reasoning_family=reasoning_family_harmony_leak_strip,
            ).strip()
        )
        raw_complete_text = native_end_text or "".join(raw_parts)
        if harmony_parser is not None:
            cut = harmony_parser.degeneration_cut
            raw_for_parse = (
                raw_complete_text[:cut] if cut is not None else raw_complete_text
            )
            replay = HarmonyStreamParser()
            after_harmony_text = ""
            for chunk in raw_for_parse:
                after_harmony_text += replay.feed(chunk)
            after_harmony_text += replay.flush()
            parse_path = "harmony_channel"
        elif delimiter_filter is not None and output_profile is not None:
            parsed = extract_delimiter_grammar(raw_complete_text, output_profile)
            after_harmony_text = parsed.visible_text
            parse_path = "delimiter_grammar"
        else:
            after_harmony_text = raw_complete_text
            parse_path = "fallback_strip"
        authoritative_text = (
            _sanitize_complete_native_text(after_harmony_text or raw_complete_text)
            if raw_complete_text
            else emitted_text
        )
        if stream_output_superseded:
            # Engine ``end`` carries the unmerged retry; re-apply follow-up preservation.
            final_text = preserve_streamed_follow_up(
                authoritative_text or emitted_text,
                streamed_before_replace or emitted_text,
                harmony_active=harmony_active,
            )
            if final_text.strip():
                self._queue_tts_sentence(final_text)
        else:
            if harmony_parser is not None and authoritative_text:
                # Prefer sanitized replay over a polluted incremental stream.
                final_text = authoritative_text
                current_sentence = ""
            elif authoritative_text and authoritative_text != emitted_text:
                if emitted_text and authoritative_text.startswith(emitted_text):
                    _emit_filtered(authoritative_text[len(emitted_text) :], speak=True)
                elif not emitted_text or not emitted_text.strip():
                    _emit_filtered(authoritative_text, speak=True)
                else:
                    final_text = authoritative_text
                    current_sentence = ""
            elif authoritative_text:
                final_text = authoritative_text
            if current_sentence.strip():
                self._queue_tts_sentence(current_sentence)
                current_sentence = ""
            if not (harmony_parser is not None and authoritative_text):
                final_text = authoritative_text or emitted_text
        if not final_text.strip() and native_load_error_text:
            final_text = native_load_error_text
        if not final_text.strip():
            _emit_filtered(NATIVE_EMPTY_VISIBLE_OUTPUT_MSG)

        citation_retry_replaced = False
        if final_text.strip():
            final_text = self._append_help_canonical_action_if_needed(final_text)
            final_text, all_ui_sources, citation_retry_replaced = (
                self._finalize_turn_citations(
                    final_text,
                    all_ui_sources,
                    messages=messages,
                    execution_route=execution_route,
                    allow_missing_retry=bool(all_ui_sources),
                )
            )
            if citation_retry_replaced:
                stream_output_superseded = True

        trace_extra: dict = {}
        val_trace = getattr(self._native_engine, "_last_output_validation_trace", None)
        if val_trace is not None:
            trace_extra.update(val_trace.trace_fields())
        if harmony_parser is not None:
            trace_extra.update(
                {
                    "harmony_parser": True,
                    "harmony_channel": harmony_parser.current_channel,
                }
            )
        if delimiter_filter is not None:
            trace_extra["delimiter_grammar_parser"] = True
        artifact_report = build_output_artifact_report(
            raw_text=raw_complete_text or "",
            visible_text=final_text or "",
            profile=output_profile,
            parse_path=parse_path,
            parse_confidence="high" if parse_path != "fallback_strip" else "low",
        )
        log_output_artifact_report(artifact_report)
        trace_extra["output_artifact_report"] = artifact_report.to_dict()

        self._completion_output_snapshot = CompletionOutputSnapshot(
            engine_mode="internal",
            raw_text=raw_complete_text or "",
            after_harmony_parser=after_harmony_text or "",
            after_worker_filters=authoritative_text or "",
            streamed_incremental=emitted_text or "",
            worker_return_text=final_text or "",
            engine_end_text=native_end_text or "",
            retry_replaced=bool(stream_output_superseded),
            extra=trace_extra,
        )

        if gen_debug is not None:
            native_engine = self._native_engine
            fresh_preflight = getattr(native_engine, "_last_trace_preflight", None) or {}
            fresh_stops = list(getattr(native_engine, "_last_merged_stops", None) or [])
            gen_debug.finalize_stream(
                snapshot=self._completion_output_snapshot,
                gt_token_ids=list(getattr(native_engine, "_gt_token_ids", []) or []),
                gt_token_texts=list(getattr(native_engine, "_gt_token_texts", []) or []),
                finish_reason=(
                    "retry_replaced" if stream_output_superseded else gen_debug._finish_reason
                ),
                native_preflight=fresh_preflight,
                merged_stops=fresh_stops,
            )
            if not (self.session_id and final_text.strip()):
                gen_debug.write_artifacts(ui_final=final_text or "")

        if self.session_id and final_text.strip():
            self._persist_assistant_turn(final_text, all_ui_sources)

        try:
            mr_trace = build_model_router_trace(self._native_engine)
            updated = self.routing_debug_buffer.merge_model_router_into_latest(mr_trace)
            cc_trace = build_chat_contract_trace(self._native_engine)
            updated_cc = self.routing_debug_buffer.merge_chat_contract_into_latest(cc_trace)
            ei_trace = build_engine_input_trace(self._native_engine)
            updated_ei = self.routing_debug_buffer.merge_engine_input_into_latest(ei_trace)
            merged = updated_ei or updated_cc or updated
            if merged is not None:
                self.routing_debug_record_added.emit(dataclasses.asdict(merged))
                self._persist_routing_debug_record(merged)
            else:
                self._persist_latest_routing_debug_record()
        except Exception as e:
            logger.debug("[RoutingDebug] native post-trace merge failed: %s", e)
            self._persist_latest_routing_debug_record()

        self._successfully_finished = True
        self._emit_output_tps(output_token_count, first_token_ts)
        return final_text

    # --- SETTERS FOR THE UI BLUEPRINT ---
    def set_provider(self, port: int):
        self.api_url = f"http://localhost:{port}/v1/chat/completions"
        self.status_update.emit(f"Switched LLM Provider (Port: {port})")
        logger.info(f"LLM Provider API URL updated to: {self.api_url}")

    def set_temperature(self, val: float):
        self.temperature = max(0.0, min(2.0, float(val)))
        set_llm_temperature(self.temperature)
        logger.debug(f"Temperature updated to {self.temperature}")

    def set_context_window(self, val: int):
        new_val = max(1024, min(128000, int(val)))
        if new_val == self.context_window:
            return
        self.context_window = new_val
        set_llm_context_limit(self.context_window)
        logger.debug(f"Context Window updated to {self.context_window}")
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal":
            self.refresh_native_model_from_settings(notify_hardware_reload=True)

    def set_output_token_limit_enabled(self, enabled: bool) -> None:
        self.output_token_limit_enabled = bool(enabled)
        set_llm_output_token_limit_enabled(self.output_token_limit_enabled)
        logger.debug(
            "Output token limit enabled=%s", self.output_token_limit_enabled
        )

    def set_output_token_limit(self, val: int) -> None:
        self.output_token_limit = max(256, min(32768, int(val)))
        set_llm_output_token_limit(self.output_token_limit)
        logger.debug("Output token limit updated to %s", self.output_token_limit)

    def set_max_history_messages(self, val: int):
        self.max_history_messages = max(2, min(100, int(val)))
        set_llm_chat_history_messages(self.max_history_messages)
        logger.debug(f"Max chat history messages updated to {self.max_history_messages}")

    def set_top_k(self, val: int):
        self.top_k = max(0, min(200, int(val)))
        set_llm_top_k(self.top_k)
        self._push_sampling_to_native()
        logger.debug(f"Top-K updated to {self.top_k}")

    def set_repeat_penalty(self, val: float):
        self.repeat_penalty = max(0.0, min(2.0, float(val)))
        set_llm_repeat_penalty(self.repeat_penalty)
        self._push_sampling_to_native()
        logger.debug(f"Repeat penalty updated to {self.repeat_penalty}")

    def set_presence_penalty(self, val: float):
        self.presence_penalty = max(0.0, min(2.0, float(val)))
        set_llm_presence_penalty(self.presence_penalty)
        self._push_sampling_to_native()
        logger.debug(f"Presence penalty updated to {self.presence_penalty}")

    def set_top_p(self, val: float):
        self.top_p = max(0.0, min(1.0, float(val)))
        set_llm_top_p(self.top_p)
        self._push_sampling_to_native()
        logger.debug(f"Top-P updated to {self.top_p}")

    def set_min_p(self, val: float):
        self.min_p = max(0.0, min(1.0, float(val)))
        set_llm_min_p(self.min_p)
        self._push_sampling_to_native()
        logger.debug(f"Min-P updated to {self.min_p}")

    def set_mcp_rag(self, enabled: bool):
        self.mcp_rag_enabled = bool(enabled)
        set_mcp_rag_enabled(self.mcp_rag_enabled)

    def set_mcp_strict(self, enabled: bool):
        self.mcp_strict_enabled = bool(enabled)
        set_mcp_rag_strict_enabled(self.mcp_strict_enabled)
        logger.debug(f"Strict Isolation Mode set to: {enabled}")

    def set_mcp_auto(self, enabled: bool):
        self.mcp_auto_enabled = bool(enabled)
        set_mcp_rag_auto_activator_enabled(self.mcp_auto_enabled)
        logger.debug(f"NLP Auto-Activator set to: {enabled}")

    def refresh_rag_triggers(self) -> None:
        """Reload custom NLP RAG trigger phrases from SQLite."""
        try:
            self.cached_custom_triggers = [
                t.lower() for t in self.db.get_rag_triggers()
            ]
        except Exception:
            self.cached_custom_triggers = []
        logger.debug(
            "Refreshed RAG trigger cache (%d phrases)",
            len(self.cached_custom_triggers),
        )
        
    def set_mcp_internet(self, enabled: bool):
        self.set_mcp_internet_hybrid(enabled)

    def set_mcp_internet_hybrid(self, enabled: bool) -> None:
        """Enable web search tool and cognitive auto-web routing together."""
        enabled = bool(enabled)
        self.mcp_internet_enabled = enabled
        self.USE_COGNITIVE_ROUTER_INTERNET = enabled
        set_mcp_internet_hybrid_enabled(enabled)

    def set_force_web_enabled(self, enabled: bool) -> None:
        """Sticky UI override: force web search on every turn until disabled."""
        self._force_web_enabled = bool(enabled)

    def set_force_web_next_turn(self, enabled: bool) -> None:
        """Alias for :meth:`set_force_web_enabled` (legacy call sites)."""
        self.set_force_web_enabled(enabled)

    def _close_active_stream(self):
        r = getattr(self, "_active_stream_response", None)
        if r is not None:
            try:
                r.close()
            except Exception:
                pass
            self._active_stream_response = None

    def _emit_web_search_audit(
        self,
        *,
        execution_route: str,
        query_raw: str,
        query_resolved: str,
        query_rewrite_reason: str | None,
        query_rewrite_failed: bool,
        force_web: bool,
        manual_web: bool,
        auto_web: bool,
        composer_internet: bool,
        composer_trusted: bool = False,
        composer_evidence: bool = False,
        veto_status: str | None = None,
        web_results_raw=None,
        web_results_kept=None,
        relevance_diag=None,
        latency_ms: float | None = None,
    ) -> None:
        try:
            event = build_audit_event_from_llm_turn(
                session_id=self.session_id,
                turn_id=getattr(self, "_routing_debug_turn_seq", None),
                user_prompt=self.prompt or "",
                execution_route=execution_route,
                internet_tool_enabled=bool(self.mcp_internet_enabled),
                force_web=force_web,
                manual_web=manual_web,
                auto_web=auto_web,
                composer_internet=composer_internet,
                composer_trusted=composer_trusted,
                composer_evidence=composer_evidence,
                query_raw=query_raw,
                query_resolved=query_resolved,
                query_rewrite_reason=query_rewrite_reason,
                query_rewrite_failed=query_rewrite_failed,
                veto_status=veto_status,
                web_results_raw=web_results_raw,
                web_results_kept=web_results_kept,
                relevance_diag=relevance_diag,
                latency_ms=latency_ms,
            )
            record_web_search_audit(event)
        except Exception:
            pass

    def _persist_routing_debug_record(self, record) -> None:
        """
        Persist one compact JSONL routing-debug event (single final write per turn).
        Never raises.
        """
        if record is None:
            return
        if not routing_debug_log_enabled():
            return
        turn_id = getattr(record, "turn_id", None)
        if turn_id is not None and self._last_persisted_routing_turn_id == turn_id:
            return
        try:
            payload = serialize_record_for_log(
                record,
                verbose=routing_debug_log_verbose(),
                redact_query=routing_debug_log_redact_query(),
            )
            routing_persist_logger.info(
                json.dumps(payload, ensure_ascii=False, default=str)
            )
            if turn_id is not None:
                self._last_persisted_routing_turn_id = int(turn_id)
        except Exception as e:
            logger.debug("[RoutingDebug] file persist failed: %s", e)

    def _persist_latest_routing_debug_record(self) -> None:
        try:
            latest = self.routing_debug_buffer.latest()
        except Exception:
            latest = None
        self._persist_routing_debug_record(latest)

    def cancel_generation(self):
        """Best-effort cancel: unblocks streaming reads; run() still finishes via finally."""
        logger.info(
            "[LLM] Cancel requested (engine_mode=%s, thread_running=%s).",
            getattr(self, "engine_mode", "unknown"),
            self.isRunning(),
        )
        self._cancel_requested = True
        self._close_active_stream()
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal" and self._native_engine:
            self._native_engine.request_cancel_generation()

    def set_engine_mode(self, mode: str) -> None:
        """Switch between external OpenAI-compatible server and in-process llama.cpp."""
        m = "internal" if str(mode).lower().strip() == "internal" else "external"
        persist_engine_mode(m)
        self.engine_mode = m
        if self.isRunning():
            self.cancel_generation()
        if m == "external":
            # One brain at a time: release native llama.cpp VRAM before external server use.
            if self._native_engine:
                self._native_engine.unload_model()
            self.status_update.emit("Engine: External (localhost) — native model unloaded (VRAM released)")
        else:
            self.status_update.emit("Engine: Internal (native)")
            # Do not auto-load here; startup/engine transitions decide this via settings.

    def eject_loaded_native_model(self) -> None:
        """Unload the in-process GGUF without clearing the saved model path."""
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) != "internal" or not self._native_engine:
            return
        self.cancel_generation()
        if self.isRunning():
            for _ in range(40):
                if not self.isRunning():
                    break
                time.sleep(0.05)
        self._native_engine.unload_model()

    def refresh_native_model_from_settings(
        self,
        *,
        autoload: bool = False,
        notify_hardware_reload: bool = False,
    ) -> "NativeModelRefreshOutcome":
        """Load or reload the native .gguf from QSettings (path, GPU layers, context)."""
        from core.native_model_autoload import (
            NativeModelRefreshOutcome,
            evaluate_native_model_refresh,
        )

        noop = NativeModelRefreshOutcome(attempted=False)
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) != "internal" or not self._native_engine:
            return noop
        if notify_hardware_reload:
            self._notify_native_hardware_reload = True
        if self.isRunning():
            self.cancel_generation()
            # Give the current turn a brief window to unwind so model load can proceed quickly.
            for _ in range(20):
                if not self.isRunning():
                    break
                time.sleep(0.05)
        path = resolve_internal_model_path(get_internal_model_path())
        outcome = evaluate_native_model_refresh(path, autoload=autoload)
        if outcome.missing_display_name:
            if self._native_engine:
                self._native_engine.unload_model()
            if outcome.missing_shards:
                self.status_update.emit(
                    f"Native engine: missing shard files for {outcome.missing_display_name}"
                )
            else:
                self.status_update.emit("Native engine: select a .gguf in Model Manager")
            return outcome
        if not path or not os.path.isfile(path):
            if self._native_engine:
                self._native_engine.unload_model()
            self.status_update.emit("Native engine: select a .gguf in Model Manager")
            return NativeModelRefreshOutcome(attempted=autoload)
        n_gpu = get_internal_n_gpu_layers()
        n_threads = get_internal_n_threads()
        n_ctx = int(getattr(self, "context_window", 4096))
        self._native_engine.load_model(path, n_gpu, n_ctx, n_threads)
        return NativeModelRefreshOutcome(attempted=autoload)

    def reload_model(self):
        """External: status only; Internal: reload .gguf with current settings."""
        logger.info("Model reload triggered by UI.")
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal":
            self.refresh_native_model_from_settings()
        else:
            self.status_update.emit("Model Context Updated")