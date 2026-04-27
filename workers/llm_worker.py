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
    get_engine_mode,
    get_internal_model_path,
    get_internal_n_gpu_layers,
    get_internal_n_threads,
    missing_gguf_shards,
    resolve_internal_model_path,
    set_engine_mode as persist_engine_mode,
)
from core.redacted_thinking_filter import RedactedThinkingStreamFilter
from core.native_meta_leading_strip import LeadingMetaInstructionStripper
from core.stream_repetition_guard import StreamRepetitionGuard
from core.output_artifact_strip import strip_harmony_oss_artifacts
from core.memory_filters import (
    detect_recall_intent,
    detect_explicit_remember,
    detect_file_search_intent,
    detect_narrative_intent,
    is_assistant_failure_message,
    is_thin_content,
    RECALL_FUSION_SYSTEM_SUFFIX,
    FILE_SEARCH_SYSTEM_SUFFIX,
    NARRATIVE_RECALL_SYSTEM_SUFFIX,
    CITATION_DISCIPLINE_SUFFIX,
    GROUNDED_ANSWER_SYSTEM_SUFFIX,
    NO_SOURCES_SYSTEM_SUFFIX,
)
from core.memory_usage_recorder import get_memory_usage_recorder

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
    token_streamed = pyqtSignal(str, str)  # session_id, token
    status_update = pyqtSignal(str)
    ttft_latency = pyqtSignal(float)
    context_retrieved = pyqtSignal(bool)
    response_finished = pyqtSignal(str, str)
    sources_found = pyqtSignal(str, list)  # session_id, sources
    router_telemetry_updated = pyqtSignal(dict, dict)  # summary, tuner_state
    routing_debug_record_added = pyqtSignal(dict)  # serialized RoutingDebugRecord
    # Phase B: turn-scoped enrichment context (session_id + rag chunk ids + message ids).
    # Emitted once per completed turn, before response_finished, so main.py can
    # forward a rich payload to EnrichmentWorker.enqueue(payload=...).
    enrichment_context_ready = pyqtSignal(dict)

    MAX_TOTAL_RETRIEVAL_CHARS = 4500
    MEMORY_BUDGET = 1500
    RAG_BUDGET = 3000

    # Streaming: read timeout applies between SSE chunks (stall guard); wall cap is absolute safety
    _STREAM_CONNECT_TIMEOUT = 20
    _STREAM_READ_TIMEOUT = 180
    _MAX_STREAM_WALL_SECONDS = 900

    # Per-message cap before sending to the API (single huge assistant/user blobs).
    CHAT_HISTORY_SINGLE_MESSAGE_MAX_CHARS = 14000

    def __init__(self, embedder, store, db_manager, native_engine=None):
        super().__init__()

        self.prompt = ""
        self.session_id = None
        self.api_url = "http://localhost:1234/v1/chat/completions"

        self.embedder = embedder
        self.store = store
        self.db = db_manager
        self._native_engine = native_engine
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

    def _is_local_llm_service(self) -> bool:
        """Only localhost inference gets cache_prompt / flush hints (OpenAI cloud may 400 on extras)."""
        try:
            host = (urlparse(self.api_url).hostname or "").lower()
            return host in ("localhost", "127.0.0.1", "::1")
        except Exception:
            return False

    def _uses_external_http(self) -> bool:
        return getattr(self, "engine_mode", "external") != "internal"

    def _is_internal_nvidia_family(self) -> bool:
        """Best-effort detection for Nemotron/NVIDIA models loaded in native engine."""
        if getattr(self, "engine_mode", "external") != "internal" or not self._native_engine:
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

    def _format_sources_for_llm_prompt(self, sources: list) -> str:
        """Single numbered block list so [1], [2], … align with UI / DB (no per-tool duplicate ids).

        Thin memory stubs (short memory entries whose content is essentially a
        bare name or < 3 informative words) are annotated when at least one
        non-memory source exists in the same block, so the LLM knows to prefer
        the richer document / web source for detail on "tell me about X"
        style queries.
        """
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

            parts.append(f"--- SOURCE {sid}: {name} ---\n{body}")
        return "\n\n".join(parts)

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

        if n_before > len(windowed):
            logger.info(
                "[LLM] Chat history windowed: using last %d of %d messages (max_history_messages=%d)",
                len(windowed),
                n_before,
                limit,
            )

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
        
        cleaned = text.strip()
        
        # 🔑 THE ULTIMATE FAILSAFE: 
        # If the string contains no letters or numbers (e.g., it's just a ".", "!", or empty), kill it.
        if not re.search(r'[a-zA-Z0-9]', cleaned):
            return ""
            
        return cleaned

    # ============================================================
    def generate_response(self, text: str, session_id: str):
        """Sets the parameters and starts the thread work."""
        if self.isRunning():
            logger.warning(
                "[LLM] Ignoring new generate_response while previous turn is active (session_id=%s).",
                session_id,
            )
            return

        self.prompt = text
        self.session_id = session_id
        self.start() # This automatically triggers the run() method

    # ============================================================
    def generate(self, prompt: str) -> str:
        if getattr(self, "engine_mode", "external") == "internal" and self._native_engine:
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
        # T3.3: reset skip/mode flags before the turn begins; _execute_llm_turn
        # re-primes them at the very top but keeping it here is belt-and-braces
        # in case an early exception fires before that method is called.
        self._reset_turn_enrichment_flags()
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
                    # T3.3: per-turn enrichment gate. ``skip_enrichment`` is
                    # the primary boolean honoured by EnrichmentWorker;
                    # ``enrichment_mode`` carries the finer tri-state so the
                    # explicit-remember bypass can still seed its knowledge
                    # fact on an "explicit_only" turn.
                    "skip_enrichment": mode == "skip",
                    "enrichment_mode": mode,
                    "skip_reason": reason,
                }
                self.enrichment_context_ready.emit(enrichment_payload)
            except Exception:
                logger.exception("[LLM] failed to emit enrichment context")
            final_text_out = strip_harmony_oss_artifacts(final_text_out or "")
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

        if self.session_id:
            self._turn_last_user_msg_id = self.db.add_message(
                self.session_id, "user", self.prompt
            )

        self._ensure_cross_session_server_flush()

        history = self.db.get_session_history(self.session_id) if self.session_id else []
        # API expects only role/content; DB rows may include "sources" for UI persistence
        history = [{"role": m["role"], "content": m["content"]} for m in history]
        history = self._bound_session_history(history)
        clean_prompt = self.prompt.lower().strip()

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
            and not file_search_active
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
            intent_vector = self.embedding_cache.get_embedding(self.prompt)
            self._ensure_recall_centroid()
            decision = self.cognitive_router.route(
                self.prompt,
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
            and not file_search_active
            and detect_recall_intent(clean_prompt)
            and execution_route in ("NONE", "MEMORY", "RAG")
        ):
            logger.info("[LLM Worker] Recall intent detected; routing to HYBRID")
            execution_route = "HYBRID"
            decision["recall_fusion"] = True

        # custom triggers override
        if not explicit_remember_active and not file_search_active and self.mcp_auto_enabled:
            if any(t in clean_prompt for t in self.cached_custom_triggers):
                execution_route = "RAG"
                decision["rag_query"] = self.prompt

        # ------------------------------------------------------------
        # INTERNET TRIGGER (manual + cognitive)
        # ------------------------------------------------------------
        # Skipped on explicit-remember (write turn) and explicit file-search
        # (the user scoped this turn to the local library).
        if not explicit_remember_active and not file_search_active:
            # Manual trigger: user text contains known web commands
            web_triggers = ["search the internet", "who won", "current news", "weather"]
            manual_web = any(t in clean_prompt for t in web_triggers) and self.mcp_internet_enabled

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

        # ============================================================
        # ROUTING START TIME (telemetry)
        # ============================================================
        route_start = time.time()

        logger.info(f"[Router] route={execution_route}")

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
        all_ui_sources = []

        # 🔑 THE FIX: Initialize these dictionaries so telemetry doesn't crash
        mem_result = {} 
        rag_result = {}
        web_context = "" # Also initialize this to be safe

        query_vector = None

        if execution_route in ["MEMORY", "RAG", "HYBRID"]:
            query_vector = self.embedding_cache.get_embedding(self.prompt)

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

            mem_result = memory_search(
                decision.get("memory_query") or self.prompt,
                query_vector,
                self.store,
                prefer_episode=prefer_episode,
                include_preference=True,
                include_knowledge=True,
                include_episode=include_episode,
                include_context=True,
            )
            memory_context = mem_result.get("memory_context", "")
            all_ui_sources.extend(mem_result.get("memory_sources", []))
        elif (
            execution_route == "NONE"
            and not explicit_remember_active
            and not file_search_active
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
            if query_vector is None:
                query_vector = self.embedding_cache.get_embedding(self.prompt)
            mem_result = memory_search(
                self.prompt,
                query_vector,
                self.store,
                prefer_episode=False,
                include_preference=True,
                include_knowledge=False,
                include_episode=False,
                include_context=True,
                top_k=3,
            )
            memory_context = mem_result.get("memory_context", "")
            all_ui_sources.extend(mem_result.get("memory_sources", []))

        # ---- RAG ----
        if execution_route in ["RAG", "HYBRID"] and self.mcp_rag_enabled:
            rag_result = rag_search(
                decision.get("rag_query") or self.prompt,
                query_vector,
                self.store
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
        if execution_route in ["WEB", "INTERNET", "HYBRID"] and (self.mcp_internet_enabled or force_web):
            from mcp.internet_tool import search_internet
            self.status_update.emit("🌐 Searching the Web...")
            
            web_results = search_internet(self.prompt)

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

            if web_results:
                if isinstance(web_results, list):
                    web_results = "\n\n".join([str(item) for item in web_results])
                else:
                    web_results = str(web_results)

                web_context = web_results[:self.RAG_BUDGET]
                
                all_ui_sources.append({
                    "id": "W", 
                    "filename": "Live Web Search", 
                    "content": web_context, 
                    "type": "web"
                })
                
                # 🔑 Append Web results to tool_context
                if tool_context:
                    tool_context = f"{tool_context}\n\n[W] WEB SEARCH RESULTS:\n{web_context}"
                else:
                    tool_context = f"[W] WEB SEARCH RESULTS:\n{web_context}"
                    
                logger.info(f"[LLM Worker] Web search integrated ({len(web_context)} chars)")

        # Sequential ids + emit isolated snapshots (UI must not share worker list refs)
        self._apply_sequential_source_ids(all_ui_sources, execution_route)
        if all_ui_sources:
            self.sources_found.emit(self.session_id or "", copy.deepcopy(all_ui_sources))

        # ============================================================
        # TELEMETRY + SELF TUNING
        # ============================================================
        latency_ms = (time.time() - route_start) * 1000

        if self.USE_TELEMETRY:
            self.telemetry.log({
                "route": execution_route,
                "memory_hits": len(mem_result.get("memory_sources", [])),
                "rag_hits": len(rag_result.get("sources", [])),
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
        ):
            logger.info(
                "[LLM Worker] All retrieval channels empty after relevance "
                "gates; downgrading route %s -> NONE for prompt build.",
                execution_route,
            )
            if execution_route in ("WEB", "INTERNET"):
                self._mark_skip_enrichment("web_route_no_sources")
            execution_route = "NONE"

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
        retrieval_prompt_body = self._format_sources_for_llm_prompt(all_ui_sources)
        if retrieval_prompt_body:
            retrieval_prompt_body = retrieval_prompt_body[: self.MAX_TOTAL_RETRIEVAL_CHARS]

        # ============================================================
        # 3. PROMPT BUILD
        # ============================================================
        system_prompt = (
            "You are Qube, a highly capable offline AI assistant. "
            "Answer naturally and accurately."
        )

        if explicit_remember_active:
            # Write-intent turn: acknowledge briefly, don't retrieve, don't cite.
            quoted_fact = (explicit_remember_body or "").strip()
            system_prompt = (
                "You are Qube. The user has just asked you to remember a fact for future reference. "
                "Acknowledge briefly — one short sentence — that you've made a note of it, "
                "and optionally paraphrase the fact naturally. "
                "Do NOT use bracket tokens like [1], [2], or [W]. "
                "Do NOT cite sources. "
                "Do NOT say you cannot remember things; Qube persists long-term memories automatically."
            )
            if quoted_fact:
                system_prompt += f" The fact to acknowledge is: \"{quoted_fact}\"."
        elif execution_route in ["RAG", "HYBRID", "MEMORY"]:
            # v6.1: if retrieval ran but nothing survived filtering
            # (proper-noun gate, semantic floor, soft L2 gate, empty
            # tables) we flip into the "no sources" mode. Without this,
            # the normal "you MUST cite your sources" instruction below
            # pushes a small LLM into fabricating a plausible-sounding
            # answer and inventing a citation (see: the "Dr. Evelyn
            # Vogel" confabulation against a Cornelia memory).
            if not all_ui_sources:
                logger.info(
                    "[LLM Worker] No sources survived retrieval filtering; "
                    "switching to NO_SOURCES system prompt (route=%s).",
                    execution_route,
                )
                system_prompt = (
                    "You are Qube, a highly capable offline AI assistant. "
                    "Answer naturally and accurately."
                )
                system_prompt += NO_SOURCES_SYSTEM_SUFFIX
            else:
                system_prompt += (
                    " You MUST cite your sources inline using brackets and the ID, like [1] or [2]. "
                    "Write citations as plain bracket tokens only—do not wrap them in Markdown links, "
                    "do not add URLs in parentheses after the token, and do not put them inside code fences or backticks."
                )
                system_prompt += RECALL_FUSION_SYSTEM_SUFFIX
                system_prompt += CITATION_DISCIPLINE_SUFFIX
                system_prompt += GROUNDED_ANSWER_SYSTEM_SUFFIX
                # File-search turns get an extra steering block so the model
                # answers strictly from numbered document sources (and says
                # so plainly when nothing matches) — never from stored
                # memories that happen to surface.
                if file_search_active:
                    system_prompt += FILE_SEARCH_SYSTEM_SUFFIX
                # T3.2: narrative / recap turns prefer the EPISODE-labelled
                # memory sources over atomic facts. memory_tool tags those
                # sources inline so the LLM can see the [EPISODE] label.
                if narrative_active:
                    system_prompt += NARRATIVE_RECALL_SYSTEM_SUFFIX

        # 🔑 THE FIX: Make the LLM hide its scratchpad and just talk normally!
        elif execution_route in ["WEB", "INTERNET"]:
            system_prompt = (
                "You are Qube. You have just been provided with real-time, live web search results. "
                "You MUST use the TOOLS context provided below to answer the user's query. "
                "Do not state that you are offline or cannot browse the internet. "
                "CRITICAL: Respond directly to the user in a natural, conversational tone. "
                "Do NOT output your internal reasoning, 'Step 1' thoughts, or search metadata. "
                "Write only the user-facing response. "
                "Cite the web sources inline using a plain [W] token only—no Markdown hyperlink syntax, "
                "no URL in parentheses after [W], and no backticks around [W]. "
                "Use [W] at most once at the end of each sentence that relies on the web results, "
                "and never output [W] two or more times in a row."
            )
            system_prompt += CITATION_DISCIPLINE_SUFFIX

        # Native llama.cpp path: behavioral alignment (LM Studio often adds server-side polish).
        if getattr(self, "engine_mode", "external") == "internal":
            if self._is_internal_nvidia_family():
                system_prompt += (
                    " Start directly with the answer content in natural language. "
                    "Do not narrate instructions, planning notes, request analysis, or hidden reasoning. "
                    "Write only what the user should see. "
                    "Prioritize clarity and completeness. "
                    "Use short answers for simple questions, but give fuller explanations when the user asks to explain, compare, or summarize."
                )
            else:
                system_prompt += (
                    " Start directly with the answer content in natural language. "
                    "Do not include preamble, planning, or meta commentary. "
                    "Do not restate or analyze the user's request. "
                    "Write only what the user should see. "
                    "Keep the response natural and focused."
                )

        messages = [{"role": "system", "content": system_prompt}] + history

        # 🔑 Inject retrieval in the same order as all_ui_sources / citation ids
        if retrieval_prompt_body and messages and messages[-1]["role"] == "user":
            original_query = messages[-1]["content"]

            messages[-1]["content"] = (
                f"=== SYSTEM RETRIEVED CONTEXT ===\n"
                f"Use the following numbered sources to answer the query. "
                f"In the prose of your reply, cite with plain tokens [1], [2], or [W] only (no markdown links).\n\n"
                f"{retrieval_prompt_body}\n"
                f"================================\n\n"
                f"USER QUERY:\n{original_query}"
            )

            logger.debug("Successfully injected unified retrieval context into the final prompt.")

        # ============================================================
        # 4. LLM STREAMING
        # ============================================================
        self.status_update.emit("Synthesizing...")

        final_text = ""

        if getattr(self, "engine_mode", "external") == "internal" and self._native_engine:
            final_text = self._stream_via_native(messages, all_ui_sources)
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

                            current_sentence += delta
                            final_text += delta
                            self.token_streamed.emit(self.session_id or "", delta)

                            if any(p in delta for p in ".!?"):
                                clean = self.clean_text_for_tts(current_sentence)
                                if clean:
                                    if not bool(getattr(self, "_cancel_requested", False)):
                                        self.sentence_ready.emit(clean, self.session_id)
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

            if final_text:
                final_text = strip_harmony_oss_artifacts(final_text)

            if current_sentence.strip():
                clean = self.clean_text_for_tts(current_sentence)
                if clean:
                    if not bool(getattr(self, "_cancel_requested", False)):
                        self.sentence_ready.emit(clean, self.session_id)

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

        self._persist_latest_routing_debug_record()
        return final_text

    def _max_tokens_native_completion(self) -> int:
        """
        Budget for *new* completion tokens in create_chat_completion (not n_ctx).
        Passing the full context window as max_tokens harms quality and can stall streaming.
        """
        ctx = max(512, int(getattr(self, "context_window", 4096)))
        return min(4096, max(256, ctx // 2))

    def _stream_via_native(self, messages: list[dict], all_ui_sources: list) -> str:
        """Stream native output after a small leading-meta/thinking gate.

        The first few chunks may contain "Provide final answer" / thinking tags; filters may
        briefly buffer those openers, but once real answer text starts, UI and TTS both stream
        the same cleaned fragments normally.
        """
        token_queue: queue.Queue = queue.Queue()
        done_event = threading.Event()
        self._native_engine.enqueue_generation(
            messages,
            self.temperature,
            self._max_tokens_native_completion(),
            token_queue,
            done_event,
        )

        cot_filter = RedactedThinkingStreamFilter()
        meta_filter = LeadingMetaInstructionStripper()
        repetition_guard = StreamRepetitionGuard()
        current_sentence = ""
        final_text = ""
        raw_parts: list[str] = []
        native_end_text = ""
        start = time.time()
        first_token = False
        stream_wall_start = time.time()

        def _sanitize_complete_native_text(raw_text: str) -> str:
            if not raw_text:
                return ""
            complete_cot = RedactedThinkingStreamFilter()
            complete_meta = LeadingMetaInstructionStripper()
            cleaned = complete_cot.feed(raw_text)
            cleaned += complete_cot.flush()
            cleaned = complete_meta.feed(cleaned) + complete_meta.flush()
            return strip_harmony_oss_artifacts(cleaned).strip()

        def _emit_filtered(fragment: str) -> None:
            nonlocal current_sentence, final_text, first_token
            if not fragment:
                return
            fragment = strip_harmony_oss_artifacts(fragment)
            if not fragment:
                return
            if not first_token:
                self.ttft_latency.emit((time.time() - start) * 1000)
                first_token = True
            final_text += fragment
            self.token_streamed.emit(self.session_id or "", fragment)
            current_sentence += fragment
            if any(p in fragment for p in ".!?"):
                spoken = self.clean_text_for_tts(current_sentence)
                if spoken and not bool(getattr(self, "_cancel_requested", False)):
                    self.sentence_ready.emit(spoken, self.session_id)
                current_sentence = ""

        def _flush_tail() -> None:
            tail = cot_filter.flush()
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
                clean_piece = meta_filter.feed(cot_filter.feed(raw_text))
                _emit_filtered(clean_piece)
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
            elif kind == "error":
                self.token_streamed.emit(self.session_id or "", f"\n\n*({data})*")
                err_txt = str(data or "")
                if "native model not loaded" in err_txt.lower():
                    self.status_update.emit("Load a Model")
                    spoken = self.clean_text_for_tts(err_txt)
                    if spoken:
                        self.sentence_ready.emit(spoken, self.session_id)
            elif kind == "end":
                native_end_text = str(data or "")
                _flush_tail()
                saw_end = True
                break

        if not saw_end:
            _flush_tail()

        if current_sentence.strip():
            spoken = self.clean_text_for_tts(current_sentence)
            if spoken and not bool(getattr(self, "_cancel_requested", False)):
                self.sentence_ready.emit(spoken, self.session_id)
            current_sentence = ""

        emitted_text = strip_harmony_oss_artifacts(final_text).strip()
        raw_complete_text = native_end_text or "".join(raw_parts)
        authoritative_text = (
            _sanitize_complete_native_text(raw_complete_text)
            if raw_complete_text
            else emitted_text
        )
        if authoritative_text and authoritative_text != emitted_text:
            if emitted_text and authoritative_text.startswith(emitted_text):
                _emit_filtered(authoritative_text[len(emitted_text) :])
            elif not emitted_text or not emitted_text.strip():
                _emit_filtered(authoritative_text)
            else:
                # The UI will reconcile the bubble on response_finished; emitting here prevents
                # "spoken but never printed" when native deltas only carried a meta preface.
                _emit_filtered("\n" + authoritative_text)
        if current_sentence.strip():
            spoken = self.clean_text_for_tts(current_sentence)
            if spoken and not bool(getattr(self, "_cancel_requested", False)):
                self.sentence_ready.emit(spoken, self.session_id)
            current_sentence = ""
        final_text = authoritative_text or emitted_text

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
        if getattr(self, "engine_mode", "external") == "internal":
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
        if getattr(self, "engine_mode", "external") == "internal" and self._native_engine:
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

    def refresh_native_model_from_settings(self) -> None:
        """Load or reload the native .gguf from QSettings (path, GPU layers, context)."""
        if getattr(self, "engine_mode", "external") != "internal" or not self._native_engine:
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
        if getattr(self, "engine_mode", "external") == "internal":
            self.refresh_native_model_from_settings()
        else:
            self.status_update.emit("Model Context Updated")