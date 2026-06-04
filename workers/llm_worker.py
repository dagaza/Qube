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
from urllib.parse import urlparse

from core.app_settings import (
    DEFAULT_ENGINE_MODE,
    get_engine_mode,
    get_internal_model_path,
    get_internal_n_gpu_layers,
    get_internal_n_threads,
    get_internal_prompt_layout_override,
    missing_gguf_shards,
    resolve_internal_model_path,
    set_engine_mode as persist_engine_mode,
)
from core.prompt_blocks import build_prompt_blocks, resolve_retrieval_wrapper_mode
from core.preference_formatters import format_web_snippets
from core.preference_policy import apply_tool_policy, resolve_preference_policy
from core.prompt_renderers import render_messages
from core.prompt_layout import PromptLayoutResolution, resolve_prompt_layout
from core.redacted_thinking_filter import RedactedThinkingStreamFilter
from core.native_meta_leading_strip import LeadingMetaInstructionStripper
from core.stream_repetition_guard import StreamRepetitionGuard
from core.harmony_degeneration import (
    harmony_tail_degenerate,
    is_harmony_orphan_stream_fragment,
    polish_harmony_visible_text,
)
from core.harmony_protocol import harmony_stream_parser_enabled, is_harmony_contract
from core.harmony_stream_parser import HarmonyStreamParser
from core.output_artifact_strip import strip_harmony_oss_artifacts
from core.completion_output_trace import (
    CompletionOutputSnapshot,
    log_completion_output_trace,
)
from core.conversational_follow_up import preserve_streamed_follow_up
from core.memory_filters import (
    detect_recall_intent,
    detect_explicit_remember,
    detect_explicit_web_request,
    detect_file_search_intent,
    detect_narrative_intent,
    is_assistant_failure_message,
    is_thin_content,
    query_implies_live_web_intent,
    should_run_internet_search_for_route,
)
from core.discourse_intent import (
    FOLLOW_UP_SUPPRESS_THRESHOLD,
    FollowUpClassification,
    FollowUpKind,
    build_topic_salience_suffix,
    classify_follow_up,
    discourse_debug_enabled,
)
from core.discourse_query import (
    resolve_retrieval_query,
    resolve_routing_query,
    resolve_search_target,
    should_veto_ungrounded_web_follow_up,
    web_query_rewrite_failed,
)
from core.retrieval_relevance import filter_web_results
from core.discourse_state import DiscourseState, update_discourse_state
from core.memory_usage_recorder import get_memory_usage_recorder, compute_query_fingerprint
from core.app_settings import (
    get_discourse_grounding_enabled,
    get_enable_chat_personality_nudge,
    get_enable_memory_v7_salvage,
)
from core.app_settings import get_sidecar_query_rewrite_enabled
from core.dual_query_retrieval import merge_memory_search_results, merge_rag_search_results
from core.sidecar_query_rewrite import propose_query_expansion
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
    resolve_attachment_routing,
)

from mcp.rag_tool import rag_search
from mcp.internet_tool import search_internet
from mcp.memory_tool import memory_search
from workers.intent_router import EmbeddingCache

from mcp.cognitive_router import CognitiveRouterV4
from mcp.routing_debug import (
    RoutingDebugBuffer,
    build_chat_contract_trace,
    build_engine_input_trace,
    build_model_router_trace,
    build_record,
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


class LLMWorker(QThread):
    sentence_ready = pyqtSignal(str, str)
    tts_turn_superseded = pyqtSignal(str)  # session_id — clear in-flight TTS after native retry
    token_streamed = pyqtSignal(str, str)  # session_id, token
    status_update = pyqtSignal(str)
    ttft_latency = pyqtSignal(float)
    tps_metric = pyqtSignal(float)
    context_retrieved = pyqtSignal(bool)
    response_finished = pyqtSignal(str, str)
    sources_found = pyqtSignal(str, list)  # session_id, sources
    router_telemetry_updated = pyqtSignal(dict, dict)  # summary, tuner_state
    sidecar_telemetry_updated = pyqtSignal(dict)
    routing_debug_record_added = pyqtSignal(dict)  # serialized RoutingDebugRecord
    # Phase B: turn-scoped enrichment context (session_id + rag chunk ids + message ids).
    # Emitted once per completed turn, before response_finished, so main.py can
    # forward a rich payload to EnrichmentWorker.enqueue(payload=...).
    enrichment_context_ready = pyqtSignal(dict)
    stream_replaced = pyqtSignal(str, str)  # session_id, full replacement text

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
        self._sidecar_client = sidecar_client
        self.engine_mode = get_engine_mode()

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
        self.router_tuner = AdaptiveRouterSelfTunerV2()
        self.routing_debug_buffer = RoutingDebugBuffer()
        self._routing_debug_turn_seq = 0
        self._last_persisted_routing_turn_id: int | None = None

        self.USE_COGNITIVE_ROUTER = True
        self.USE_ADAPTIVE_ROUTER = True
        self.USE_TELEMETRY = True
        self.USE_COGNITIVE_ROUTER_INTERNET = True # For hybrid internet mode

        # toggles
        self.mcp_auto_enabled = True
        self.temperature = 0.7
        self.context_window = 4096
        # Sliding window: max DB messages to include in the chat completion (user-controlled).
        self.max_history_messages = 10
        self.mcp_rag_enabled = True
        self.mcp_strict_enabled = False
        self.mcp_internet_enabled = False
        self._force_web_next_turn = False

        # Local llama.cpp / LM Studio: align server-side prompt/KV reuse with UI session switches
        self._last_completed_llm_session_id = None
        self._server_kv_cleared_for_session_id = None
        self._discourse_by_session: dict[str, DiscourseState] = {}

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
        if execution_route in ("WEB", "INTERNET") and len(sources) == 1:
            if str(sources[0].get("type", "")).lower() == "web":
                return
        for i, src in enumerate(sources, start=1):
            if isinstance(src, dict):
                src["id"] = i

    # Phase B: curated recall examples used to build the semantic centroid
    # consumed by ``CognitiveRouterV4._score_recall_intent``. Kept short so
    # the one-time embedding pass at first use is cheap.
    _RECALL_INTENT_EXAMPLES = (
        "tell me about Alice",
        "who is John Smith?",
        "what do you know about my brother?",
        "remind me about the project deadline",
        "what did we say about the proposal yesterday?",
        "summarize what you know about the trip plans",
        "do you remember anything about my coffee preference?",
        "refresh my memory on the Berlin meeting",
        "recall what I told you about my thesis",
        "what is the user's preferred coding style?",
    )

    # T4.2: curated chat / general-knowledge examples used to build the
    # NEGATIVE-class centroid consumed by
    # ``CognitiveRouterV4._score_chat_intent``. Deliberately avoids
    # "remember" / "recall" / "tell me about" / "who is" tokens so the
    # centroid sits visibly away from the recall centroid in embedding
    # space. Mix is factual / general-knowledge / chitchat / task /
    # coding, 10 short prompts (≤ ~60 chars each) to mirror the shape of
    # ``_RECALL_INTENT_EXAMPLES``.
    _CHAT_INTENT_EXAMPLES = (
        "Why is the sky blue?",
        "How does photosynthesis work?",
        "What is the speed of light in a vacuum?",
        "Explain how a transformer neural network works.",
        "Write me a haiku about the sea.",
        "Give me a Python snippet to reverse a string.",
        "Translate 'good morning' into Spanish.",
        "What's the capital of Australia?",
        "Summarize the plot of Macbeth in two sentences.",
        "How do I convert 32 degrees Fahrenheit to Celsius?",
    )

    # Tier 2: curated phrase sets used to build the per-lane embedding
    # centroids consumed by ``CognitiveRouterV4._score_*_intent_embedding``.
    # Kept at ~10 short prompts each to mirror the recall/chat sets and
    # keep the one-time embedding pass cheap. Each set deliberately uses
    # vocabulary that the substring trigger lists DO NOT cover, so the
    # ``max(substring, embedding)`` fusion adds genuine semantic recall
    # rather than echoing the keyword list.
    _MEMORY_INTENT_EXAMPLES = (
        "what did I tell you about my work last week?",
        "do you recall the name of my dog?",
        "bring up what we agreed on yesterday",
        "what are my dietary restrictions?",
        "what timezone do I live in again?",
        "what was the address I gave you?",
        "show me the notes I shared earlier",
        "what's the password hint I told you?",
        "what's my usual sleep schedule?",
        "remind me of my favorite movies list",
    )

    _RAG_INTENT_EXAMPLES = (
        "summarize the attached PDF",
        "what does the contract say about termination?",
        "according to the report, what is the revenue?",
        "in the document, find the section about safety",
        "quote the relevant passage from the manual",
        "what does the spec define for retry behavior?",
        "based on the file I uploaded, who are the authors?",
        "find the clause about confidentiality in the agreement",
        "extract the conclusions from the paper",
        "what does chapter three of the book cover?",
    )

    _WEB_INTENT_EXAMPLES = (
        "search the internet for the latest iPhone release date",
        "look up today's weather in Madrid",
        "what's currently trending on Hacker News?",
        "find recent news about the federal reserve",
        "google the price of bitcoin right now",
        "what's the live score of the soccer match?",
        "look online for flight delays at JFK today",
        "search for recent reviews of this restaurant",
        "what is the current exchange rate for USD to EUR?",
        "fetch the latest stock price of Tesla",
    )

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
        """T4.2: lazily build and install BOTH the RECALL and CHAT
        (negative-class) semantic centroids on the cognitive router.

        Called once on the first turn that uses the cognitive router.
        Each centroid is only built if it has not been installed yet,
        so the method is cheap to call on every turn. The router falls
        back to substring detection for recall if anything fails here;
        an unset chat centroid simply returns ``chat_score = 0.0`` and
        leaves the margin gate trivially satisfied (backwards compatible
        with the single-centroid pre-T4.2 behaviour).
        """
        if not getattr(self, "cognitive_router", None):
            return
        embedder = getattr(self.embedding_cache, "embedder", None)
        if embedder is None:
            return
        try:
            from workers.intent_router import build_centroid
            if self.cognitive_router.recall_centroid is None:
                self.cognitive_router.set_recall_centroid(
                    build_centroid(embedder, list(self._RECALL_INTENT_EXAMPLES))
                )
                logger.info("[LLM Worker] Recall centroid installed.")
            if self.cognitive_router.chat_centroid is None:
                self.cognitive_router.set_chat_centroid(
                    build_centroid(embedder, list(self._CHAT_INTENT_EXAMPLES))
                )
                logger.info("[LLM Worker] Chat centroid installed.")
            # Tier 2: install the per-lane embedding centroids. Each
            # is gated by ``is None`` so we never stomp a manually
            # installed centroid (e.g. in tests) and so the build
            # cost is paid exactly once per worker lifetime. Until at
            # least one of these is installed, the router's confidence
            # layer stays dormant via the ``any_embedding_centroid``
            # gate in ``CognitiveRouterV4.route(...)``.
            if self.cognitive_router.memory_centroid is None:
                self.cognitive_router.set_memory_centroid(
                    build_centroid(embedder, list(self._MEMORY_INTENT_EXAMPLES))
                )
                logger.info("[LLM Worker] Memory centroid installed.")
            if self.cognitive_router.rag_centroid is None:
                self.cognitive_router.set_rag_centroid(
                    build_centroid(embedder, list(self._RAG_INTENT_EXAMPLES))
                )
                logger.info("[LLM Worker] RAG centroid installed.")
            if self.cognitive_router.web_centroid is None:
                self.cognitive_router.set_web_centroid(
                    build_centroid(embedder, list(self._WEB_INTENT_EXAMPLES))
                )
                logger.info("[LLM Worker] Web centroid installed.")
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
    ) -> None:
        if not isinstance(decision, dict):
            return
        decision.update(follow_up.to_dict())
        if discourse_state is not None:
            decision.update(discourse_state.to_dict())
        if routing_query != (self.prompt or "").strip():
            decision["routing_query"] = routing_query
        if retrieval_query != (self.prompt or "").strip():
            decision["retrieval_query"] = retrieval_query
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
        logger.info(
            "[Discourse] follow_up=%s conf=%.2f topic=%r wrapper=%s "
            "core_memory_suppressed=%s roles=%s hist_chars=%d retrieval_chars=%d query_chars=%d",
            follow_up.kind.value,
            follow_up.confidence,
            topic,
            retrieval_wrapper_mode,
            core_memory_suppressed,
            roles,
            history_chars,
            retrieval_chars,
            query_chars,
        )

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

    # ============================================================
    def generate_response(
        self,
        text: str,
        session_id: str,
        *,
        attachments: list | None = None,
        persist_content: str | None = None,
    ):
        """Sets the parameters and starts the thread work."""
        if self.isRunning():
            logger.warning(
                "[LLM] Ignoring new generate_response while previous turn is active (session_id=%s).",
                session_id,
            )
            return

        self.prompt = (text or "").strip()
        self._persist_content = (persist_content or self.prompt).strip()
        self._turn_attachments = list(attachments or [])
        self.session_id = session_id
        self.start() # This automatically triggers the run() method

    # ============================================================
    def generate(self, prompt: str) -> str:
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal" and self._native_engine:
            out: list = []
            ev = threading.Event()
            self._native_engine.enqueue_simple_completion(
                [{"role": "user", "content": prompt}],
                0.1,
                1000,
                out,
                ev,
            )
            if not ev.wait(120):
                return ""
            return (out[0] if out else "") or ""

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000,
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
            final_text_out = strip_harmony_oss_artifacts(final_text_out or "")
            log_completion_output_trace(
                session_id=str(self.session_id or ""),
                snapshot=getattr(self, "_completion_output_snapshot", None),
                presented_text=final_text_out,
            )
            self._completion_output_snapshot = None
            self.response_finished.emit(self.session_id, final_text_out)
            if not self._successfully_finished:
                self.status_update.emit("Idle")

    def _execute_llm_turn(self) -> str:
        force_web = bool(getattr(self, "_force_web_next_turn", False))
        self._force_web_next_turn = False

        # Phase B: reset per-turn enrichment context captured during this turn.
        self._turn_rag_chunk_ids: list[str] = []
        self._turn_last_user_msg_id = None
        self._turn_last_assistant_msg_id = None
        # T3.3: reset tool-aware enrichment skip / mode flags for this turn.
        self._reset_turn_enrichment_flags()
        self._pending_salvage_message_ids = []

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
        routing_query = original_query
        retrieval_query = original_query
        query_expansion: QueryExpansion | None = None
        core_memory_suppressed = False
        self._sidecar_hybrid_extra_memory = 0
        self._sidecar_hybrid_extra_rag = 0
        digest_mem_attempted = False
        digest_mem_applied = False
        digest_rag_attempted = False
        digest_rag_applied = False

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
            routing_query = resolve_routing_query(self.prompt, follow_up, discourse_state)
            retrieval_query = resolve_retrieval_query(self.prompt, follow_up, discourse_state)
            query_expansion = propose_query_expansion(
                original_query,
                follow_up,
                discourse_state,
                history,
                self._sidecar_client,
            )
            if query_expansion:
                logger.info(
                    "[Sidecar] assistive expansion conf=%.2f expanded=%r",
                    query_expansion.confidence,
                    query_expansion.expanded_query[:120],
                )
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
        attachment_patch = None
        if not explicit_remember_active and turn_attachments:
            attachment_patch = resolve_attachment_routing(turn_attachments)

        attachment_file_active = False
        attachment_conversation_active = False
        self._turn_source_filter = None
        self._turn_attachment_context = ""
        self._composer_internet_requested = False

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
            if attachment_patch.get("attachment_tool") == "internet":
                self._composer_internet_requested = True
                # Explicit @internet must behave like the toolbar Web toggle:
                # force a web search for this turn (not gated on Settings).
                force_web = True
                if attachment_patch.get("route") != "web":
                    attachment_patch = dict(attachment_patch)
                    attachment_patch["route"] = "web"
                    attachment_patch["strategy"] = "attachment_tool_internet"
                logger.info(
                    "[LLM Worker] Composer @internet: forcing WEB search for this turn"
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
        self.status_update.emit("Thinking...")

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
            self._ensure_recall_centroid()
            decision = self.cognitive_router.route(
                routing_query,
                intent_vector=intent_vector,
                weights=self.router_tuner.get_weights() if self.USE_ADAPTIVE_ROUTER else None
            )
        else:
            decision = {"route": "none", "strategy": "fallback"}

        execution_route = decision["route"].upper()

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
            and detect_recall_intent(clean_prompt)
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

        # ------------------------------------------------------------
        # INTERNET TRIGGER (manual + cognitive)
        # ------------------------------------------------------------
        # Skipped on explicit-remember (write turn) and explicit file-search
        # (the user scoped this turn to the local library).
        manual_web = False
        auto_web = False
        explicit_web_request = detect_explicit_web_request(clean_prompt)
        if not explicit_remember_active and not scoped_library_active:
            # Manual trigger: user explicitly asked to search/check the web.
            manual_web = explicit_web_request

            # Automatic trigger: cognitive router decides internet is needed
            auto_web = getattr(self, "USE_COGNITIVE_ROUTER_INTERNET", False) and decision.get("internet_enabled", False)

            # Final execution decision for WEB
            if force_web or manual_web or auto_web:
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
                and not getattr(self, "_composer_internet_requested", False)
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
        web_capability_blocked = bool(
            explicit_web_request and not self.mcp_internet_enabled
        ) or bool(
            web_vetoed and query_implies_live_web_intent(clean_prompt, decision=decision)
        )
        if web_vetoed and not web_capability_blocked:
            logger.info(
                "[LLM Worker] WEB route vetoed (internet disabled) but query has "
                "no live-web intent; using plain chat prompt."
            )

        # ============================================================
        # ROUTING START TIME (telemetry)
        # ============================================================
        route_start = time.time()

        logger.info(f"[Router] route={execution_route}")

        retrieval_wrapper_mode = "none"
        self._stamp_discourse_on_decision(
            decision,
            follow_up=follow_up,
            discourse_state=discourse_state if discourse_enabled else None,
            routing_query=routing_query,
            retrieval_query=retrieval_query,
            core_memory_suppressed=core_memory_suppressed,
            retrieval_wrapper_mode=retrieval_wrapper_mode,
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
            and not getattr(self, "_composer_internet_requested", False)
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
            self.mcp_rag_enabled or force_rag_via_trigger
        ):
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

        # ---- WEB + HYBRID ----
        web_search_attempted = False
        if should_run_internet_search_for_route(
            execution_route,
            clean_prompt,
            decision=decision if isinstance(decision, dict) else None,
            force_web=force_web,
            manual_web=manual_web,
            auto_web=auto_web,
            composer_internet=bool(getattr(self, "_composer_internet_requested", False)),
        ) and (self.mcp_internet_enabled or force_web):
            web_search_attempted = True
            self.status_update.emit("🌐 Searching the Web...")

            search_target = resolve_search_target(
                self.prompt,
                follow_up,
                discourse_state,
                history,
            )
            web_semantic = search_target.query
            web_query = apply_tool_policy(
                web_semantic,
                preference_policy,
                tool="internet",
            )
            raw_prompt = (self.prompt or "").strip()
            rewrite_failed = web_query_rewrite_failed(
                self.prompt,
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

            web_results = search_internet(web_query)

            # Defensive guard: when search_internet fails (e.g. DNS /
            # connection reset / no-result sentinel) it still returns a
            # list of the shape [{"title": "", "snippet": "Internet search
            # failed..."}]. Previously we injected that sentinel into the
            # prompt with a "[W]" tag, and the small-LLM happily looped
            # "[W][W][W]" until StreamRepetitionGuard cancelled the turn.
            # Treat any such sentinel as "no web data for this turn".
            if isinstance(web_results, list):
                _snips = " ".join(
                    str((r or {}).get("snippet") or "") if isinstance(r, dict) else str(r or "")
                    for r in web_results
                )
                if (
                    "Internet search failed" in _snips
                    or "No relevant internet results found" in _snips
                    or not _snips.strip()
                ):
                    logger.info(
                        "[LLM Worker] Web results dropped (empty / failure sentinel); not injecting [W] context."
                    )
                    web_results = None
                    # T3.3: a web-route turn without web data is effectively
                    # a capability failure — skip enrichment so the thin /
                    # "I couldn't find anything online" style reply is not
                    # mined as a user fact.
                    if execution_route in ("WEB", "INTERNET", "HYBRID"):
                        self._mark_skip_enrichment("web_tool_failure")
                elif web_results:
                    try:
                        web_query_vector = self.embedding_cache.get_embedding(
                            web_semantic or web_query
                        )
                    except Exception:
                        web_query_vector = None
                    filtered, rel_diag = filter_web_results(
                        web_semantic or web_query,
                        [r for r in web_results if isinstance(r, dict)],
                        query_vector=web_query_vector,
                        embed_text_fn=self.embedding_cache.get_embedding,
                        use_embedding_gate=True,
                    )
                    if isinstance(decision, dict):
                        decision.update(rel_diag)
                    kept = rel_diag.get("web_results_kept_count", 0)
                    dropped = len(rel_diag.get("web_relevance_dropped") or [])
                    logger.info(
                        "[WebPipeline] relevance_gate kept=%d dropped=%d "
                        "min_overlap=%.2f",
                        kept,
                        dropped,
                        rel_diag.get("web_relevance_min_overlap", 0.15),
                    )
                    if filtered:
                        web_results = filtered
                    else:
                        logger.info(
                            "[LLM Worker] Web results dropped (relevance gate); "
                            "not injecting [W] context."
                        )
                        web_results = None
                        if execution_route in ("WEB", "INTERNET", "HYBRID"):
                            self._mark_skip_enrichment("web_tool_failure")
                    if hasattr(self, "routing_debug_buffer"):
                        try:
                            self.routing_debug_buffer.merge_web_pipeline_into_latest(
                                {
                                    "web_query_resolved": web_query,
                                    "web_query_rewrite_reason": search_target.rewrite_reason,
                                    "web_query_rewrite_failed": rewrite_failed,
                                    **rel_diag,
                                }
                            )
                        except Exception:
                            pass

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

                single_web = len(web_items) == 1
                for idx, item in enumerate(web_items, start=1):
                    title = str(item.get("title") or "").strip() or f"Web result {idx}"
                    snippet = str(item.get("snippet") or "").strip()
                    src: dict = {
                        "filename": title,
                        "content": snippet,
                        "type": "web",
                    }
                    url = str(item.get("url") or "").strip()
                    if url.startswith(("http://", "https://")):
                        src["url"] = url
                    if single_web and execution_route in ("WEB", "INTERNET"):
                        src["id"] = "W"
                    all_ui_sources.append(src)

                web_hdr = (
                    "[W] WEB SEARCH RESULTS"
                    if single_web and execution_route in ("WEB", "INTERNET")
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

        digest_mem_attempted = bool(
            memory_context and mem_result.get("memory_sources")
        )
        digest_mem_applied = False
        if digest_mem_attempted:
            digested, applied = digest_memory_context(
                memory_context,
                mem_result.get("memory_sources") or [],
                self._sidecar_client,
            )
            digest_mem_applied = bool(applied)
            if applied:
                logger.info(
                    "[Sidecar] memory digest applied chars %d -> %d",
                    len(memory_context),
                    len(digested),
                )
                memory_context = digested

        digest_rag_attempted = bool(tool_context and rag_result.get("sources"))
        digest_rag_applied = False
        if digest_rag_attempted:
            digested_rag, applied_rag = digest_rag_context(
                tool_context,
                rag_result.get("sources") or [],
                self._sidecar_client,
            )
            digest_rag_applied = bool(applied_rag)
            if applied_rag:
                logger.info(
                    "[Sidecar] RAG digest applied chars %d -> %d",
                    len(tool_context),
                    len(digested_rag),
                )
                tool_context = digested_rag

        # Sequential ids + emit isolated snapshots (UI must not share worker list refs)
        self._apply_sequential_source_ids(all_ui_sources, execution_route)
        if all_ui_sources:
            self.sources_found.emit(self.session_id or "", copy.deepcopy(all_ui_sources))

        # ============================================================
        # TELEMETRY + SELF TUNING
        # ============================================================
        latency_ms = (time.time() - route_start) * 1000

        if self.USE_TELEMETRY:
            web_hits = sum(
                1
                for s in all_ui_sources
                if isinstance(s, dict) and s.get("type") == "web"
            )
            self.telemetry.log({
                "route": execution_route,
                "memory_hits": len(mem_result.get("memory_sources", [])),
                "rag_hits": len(rag_result.get("sources", [])),
                "web_hits": web_hits,
                "web_search_attempted": bool(web_search_attempted),
                "latency_ms": latency_ms,
                "memory_chars": len(memory_context),
                "rag_chars": len(tool_context),
            })

            self.router_tuner.observe({
                "route": execution_route,
                "memory_hits": len(mem_result.get("memory_sources", [])),
                "rag_hits": len(rag_result.get("sources", [])),
                "latency_ms": latency_ms,
            })
            
            try:
                summary = self.telemetry.summarize()
                tuner_state = self.router_tuner.get_weights()
                self.router_telemetry_updated.emit(summary, tuner_state)
            except Exception as e:
                logger.error(f"Failed to emit router telemetry: {e}")

        try:
            rewrite_attempted = bool(
                discourse_enabled
                and get_sidecar_query_rewrite_enabled()
                and follow_up.active
            )
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
                hybrid_extra_memory=int(
                    getattr(self, "_sidecar_hybrid_extra_memory", 0) or 0
                ),
                hybrid_extra_rag=int(
                    getattr(self, "_sidecar_hybrid_extra_rag", 0) or 0
                ),
            )
            self.sidecar_telemetry_updated.emit(get_sidecar_telemetry().summarize())
        except Exception as e:
            logger.debug("Failed to emit sidecar telemetry: %s", e)

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
        if (
            execution_route in ("MEMORY", "RAG", "HYBRID", "WEB", "INTERNET")
            and not all_ui_sources
            and not getattr(self, "_composer_internet_requested", False)
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
            getattr(self, "_composer_internet_requested", False)
            and execution_route in ("WEB", "INTERNET")
            and not all_ui_sources
        ):
            logger.warning(
                "[LLM Worker] Composer @internet: web search returned no "
                "sources; keeping WEB route with empty-results guidance."
            )
            self._mark_skip_enrichment("web_route_no_sources")
            tool_context += (
                "\n[WEB SEARCH: No live results were returned for this query. "
                "Tell the user you could not retrieve web results right now. "
                "Do NOT invent facts or emit [W] citations without sources.]\n"
            )

        explicit_web_empty_results = bool(
            explicit_web_request
            and web_search_attempted
            and not all_ui_sources
            and self.mcp_internet_enabled
            and execution_route == "NONE"
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
                    memory_hits = len(mem_result.get("memory_sources", []))
                    rag_hits   = len(rag_result.get("sources", []))
                    web_hits   = sum(
                        1
                        for s in all_ui_sources
                        if isinstance(s, dict) and s.get("type") == "web"
                    )

                    per_lane_hits = {
                        "memory": memory_hits,
                        "rag":    rag_hits,
                        "web":    web_hits,
                    }

                    if original_route == "hybrid":
                        success_flag = (memory_hits > 0) or (rag_hits > 0)
                    elif original_route in ("memory", "rag", "web"):
                        success_flag = per_lane_hits[original_route] > 0
                    else:
                        success_flag = False

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
        if (
            getattr(self, "_composer_internet_requested", False)
            and not all_ui_sources
            and tool_context.strip()
        ):
            retrieval_prompt_body = tool_context.strip()
            logger.info(
                "[LLM Worker] Composer @internet: injected web guidance (%d chars)",
                len(retrieval_prompt_body),
            )
        if retrieval_prompt_body:
            retrieval_prompt_body = retrieval_prompt_body[: self.MAX_TOTAL_RETRIEVAL_CHARS]

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
        elif (
            discourse_enabled
            and follow_up.active
            and history
            and history[-1].get("role") == "user"
        ):
            grounded = list(history)
            last = dict(grounded[-1])
            if discourse_state and discourse_state.active_topic:
                last["content"] = (
                    f"[Continuing our discussion of {discourse_state.active_topic}]\n\n"
                    f"{last.get('content') or ''}"
                )
            elif len(grounded) >= 2:
                last["content"] = (
                    "[Continuing from the conversation above]\n\n"
                    f"{last.get('content') or ''}"
                )
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
            bool(all_ui_sources),
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
        )
        self._stamp_query_expansion_on_decision(
            decision,
            original_query=original_query,
            retrieval_query=retrieval_query,
            expansion=query_expansion,
        )

        topic_salience = ""
        if (
            discourse_enabled
            and follow_up.active
            and discourse_state
            and discourse_state.active_topic
        ):
            topic_salience = build_topic_salience_suffix(
                discourse_state.active_topic,
                topic_type=discourse_state.topic_type,
            )

        prompt_blocks = build_prompt_blocks(
            execution_route=execution_route,
            explicit_remember_active=explicit_remember_active,
            explicit_remember_body=explicit_remember_body or "",
            file_search_active=file_search_active,
            narrative_active=narrative_active,
            has_retrieval_sources=bool(all_ui_sources),
            engine_mode=getattr(self, "engine_mode", DEFAULT_ENGINE_MODE),
            internal_nvidia_family=self._is_internal_nvidia_family(),
            retrieval_context=retrieval_prompt_body,
            conversation_history=prompt_history,
            composer_conversation_ref=attachment_conversation_active,
            web_capability_blocked=web_capability_blocked,
            explicit_web_empty_results=explicit_web_empty_results,
            preference_context=preference_policy.compact_prompt_context(
                query=self.prompt,
                route=execution_route,
            ),
            apply_preference_suffix=preference_policy.has_presentation_prefs(),
            retrieval_wrapper_mode=retrieval_wrapper_mode,
            topic_salience_hint=topic_salience,
            follow_up_active=follow_up.active,
            chat_personality_enabled=get_enable_chat_personality_nudge(),
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

        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal" and self._native_engine:
            final_text = self._stream_via_native(
                messages,
                all_ui_sources,
                retrieval_context=retrieval_prompt_body,
            )
            return final_text

        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.context_window,
            "stream": True,
        }
        if self._uses_external_http() and self._is_local_llm_service():
            # llama.cpp server: avoid unbounded prompt-prefix / KV reuse across unrelated requests
            payload["cache_prompt"] = False

        current_sentence = ""
        final_text = ""
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
            repetition_guard = StreamRepetitionGuard()

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
                        delta = packet["choices"][0].get("delta", {}).get("content", "")

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
                                # T3.3: truncated / degenerate assistant text
                                # must not be mined for memories.
                                self._mark_skip_enrichment("stream_repetition_cancelled")
                                break

                    except json.JSONDecodeError:
                        continue

            raw_external_text = final_text
            if final_text:
                final_text = strip_harmony_oss_artifacts(final_text)

            if current_sentence.strip():
                self._queue_tts_sentence(current_sentence)

            self._completion_output_snapshot = CompletionOutputSnapshot(
                engine_mode="external",
                raw_text=raw_external_text or "",
                after_worker_filters=final_text or "",
                worker_return_text=final_text or "",
            )

            if self.session_id and final_text.strip():
                src_payload = json.dumps(all_ui_sources) if all_ui_sources else None
                self._turn_last_assistant_msg_id = self.db.add_message(
                    self.session_id, "assistant", final_text, sources_json=src_payload
                )
                self._record_memory_citations(final_text, all_ui_sources)

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

    def _max_tokens_native_completion(self) -> int:
        """
        Budget for *new* completion tokens in create_chat_completion (not n_ctx).
        Passing the full context window as max_tokens harms quality and can stall streaming.
        """
        ctx = max(512, int(getattr(self, "context_window", 4096)))
        return min(4096, max(256, ctx // 2))

    def _stream_via_native(
        self,
        messages: list[dict],
        all_ui_sources: list,
        *,
        retrieval_context: str = "",
    ) -> str:
        """Stream native output after a small leading-meta/thinking gate.

        The first few chunks may contain "Provide final answer" / thinking tags; filters may
        briefly buffer those openers, but once real answer text starts, UI and TTS both stream
        the same cleaned fragments normally.
        """
        self._reset_tts_dedupe_state()
        token_queue: queue.Queue = queue.Queue()
        done_event = threading.Event()
        self._native_engine.enqueue_generation(
            messages,
            self.temperature,
            self._max_tokens_native_completion(),
            token_queue,
            done_event,
            retrieval_context=(retrieval_context or "").strip(),
        )

        cot_filter = RedactedThinkingStreamFilter()
        meta_filter = LeadingMetaInstructionStripper()
        repetition_guard = StreamRepetitionGuard()
        prompt_contract = getattr(self._native_engine, "_last_prompt_contract", None)
        use_harmony_parser = bool(
            is_harmony_contract(prompt_contract) and harmony_stream_parser_enabled()
        )
        harmony_parser = HarmonyStreamParser() if use_harmony_parser else None
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
            if not raw_text:
                return ""
            complete_cot = RedactedThinkingStreamFilter()
            complete_meta = LeadingMetaInstructionStripper()
            cleaned = complete_cot.feed(raw_text)
            cleaned += complete_cot.flush()
            cleaned = complete_meta.feed(cleaned) + complete_meta.flush()
            return strip_harmony_oss_artifacts(
                polish_harmony_visible_text(cleaned)
            ).strip()

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
            if harmony_parser is None:
                fragment = strip_harmony_oss_artifacts(fragment)
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
            tail = cot_filter.feed(tail)
            tail += cot_filter.flush()
            tail = meta_filter.feed(tail) + meta_filter.flush()
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
                else:
                    stream_in = raw_text
                clean_piece = meta_filter.feed(cot_filter.feed(stream_in))
                _emit_filtered(clean_piece)
                if harmony_parser is not None and final_text.strip():
                    if harmony_parser.degeneration_detected or harmony_tail_degenerate(
                        harmony_parser.raw_seen
                    ):
                        logger.info(
                            "[LLM] Harmony degeneration detected; cancelling generation."
                        )
                        harmony_cut_cancelled = True
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
                    # T3.3: truncated / degenerate assistant text must not be
                    # mined for memories.
                    self._mark_skip_enrichment("stream_repetition_cancelled")
                    self._native_engine.request_cancel_generation()
                    _flush_tail()
                    saw_end = True
                    break
            elif kind == "recovery":
                raw_text = str(data or "")
                if raw_text:
                    raw_parts.append(raw_text)
                if harmony_parser is not None:
                    stream_in = harmony_parser.feed(raw_text)
                else:
                    stream_in = raw_text
                clean_piece = meta_filter.feed(cot_filter.feed(stream_in))
                _emit_filtered(clean_piece, speak=False)
            elif kind == "replace":
                replacement = str(data or "").strip()
                streamed_snapshot = strip_harmony_oss_artifacts(final_text).strip()
                streamed_before_replace = streamed_snapshot
                replacement = preserve_streamed_follow_up(replacement, streamed_snapshot)
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

        emitted_text = strip_harmony_oss_artifacts(final_text).strip()
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
        else:
            after_harmony_text = raw_complete_text
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
            empty_msg = (
                "The model finished without producing any visible text. "
                "Try sending again, adjust Think, or inspect logs/llm_debug.log."
            )
            final_text = empty_msg
            _emit_filtered(empty_msg)

        self._completion_output_snapshot = CompletionOutputSnapshot(
            engine_mode="internal",
            raw_text=raw_complete_text or "",
            after_harmony_parser=after_harmony_text or "",
            after_worker_filters=authoritative_text or "",
            streamed_incremental=emitted_text or "",
            worker_return_text=final_text or "",
            engine_end_text=native_end_text or "",
            retry_replaced=bool(stream_output_superseded),
            extra=(
                {"harmony_parser": True, "harmony_channel": harmony_parser.current_channel}
                if harmony_parser is not None
                else {}
            ),
        )

        if self.session_id and final_text.strip():
            src_payload = json.dumps(all_ui_sources) if all_ui_sources else None
            self._turn_last_assistant_msg_id = self.db.add_message(
                self.session_id, "assistant", final_text, sources_json=src_payload
            )
            self._record_memory_citations(final_text, all_ui_sources)

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
        self.temperature = val
        logger.debug(f"Temperature updated to {val}")

    def set_context_window(self, val: int):
        self.context_window = val
        logger.debug(f"Context Window updated to {val}")
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal":
            self.refresh_native_model_from_settings()

    def set_max_history_messages(self, val: int):
        self.max_history_messages = max(2, min(100, int(val)))
        logger.debug(f"Max chat history messages updated to {self.max_history_messages}")

    def set_mcp_rag(self, enabled: bool):
        self.mcp_rag_enabled = enabled

    def set_mcp_strict(self, enabled: bool):
        self.mcp_strict_enabled = enabled
        logger.debug(f"Strict Isolation Mode set to: {enabled}")

    def set_mcp_auto(self, enabled: bool):
        self.mcp_auto_enabled = enabled
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
        self.mcp_internet_enabled = enabled

    def set_force_web_next_turn(self, enabled: bool) -> None:
        """One-shot UI override for the next user prompt."""
        self._force_web_next_turn = bool(enabled)

    def _close_active_stream(self):
        r = getattr(self, "_active_stream_response", None)
        if r is not None:
            try:
                r.close()
            except Exception:
                pass
            self._active_stream_response = None

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

    def refresh_native_model_from_settings(self) -> None:
        """Load or reload the native .gguf from QSettings (path, GPU layers, context)."""
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) != "internal" or not self._native_engine:
            return
        if self.isRunning():
            self.cancel_generation()
            # Give the current turn a brief window to unwind so model load can proceed quickly.
            for _ in range(20):
                if not self.isRunning():
                    break
                time.sleep(0.05)
        path = resolve_internal_model_path(get_internal_model_path())
        n_gpu = get_internal_n_gpu_layers()
        n_threads = get_internal_n_threads()
        n_ctx = int(getattr(self, "context_window", 4096))
        if not path or not os.path.isfile(path):
            if self._native_engine:
                self._native_engine.unload_model()
            self.status_update.emit("Native engine: select a .gguf in Model Manager")
            return
        missing = missing_gguf_shards(path)
        if missing:
            if self._native_engine:
                self._native_engine.unload_model()
            self.status_update.emit(
                f"Native engine: missing shard files ({len(missing)} missing) - download all parts"
            )
            return
        self._native_engine.load_model(path, n_gpu, n_ctx, n_threads)

    def reload_model(self):
        """External: status only; Internal: reload .gguf with current settings."""
        logger.info("Model reload triggered by UI.")
        if getattr(self, "engine_mode", DEFAULT_ENGINE_MODE) == "internal":
            self.refresh_native_model_from_settings()
        else:
            self.status_update.emit("Model Context Updated")