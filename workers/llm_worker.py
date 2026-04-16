from PyQt6.QtCore import QThread, pyqtSignal
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
from core.memory_filters import (
    detect_recall_intent,
    detect_explicit_remember,
    detect_file_search_intent,
    is_thin_content,
    RECALL_FUSION_SYSTEM_SUFFIX,
    FILE_SEARCH_SYSTEM_SUFFIX,
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
from mcp.router_telemetry import RouterTelemetryBrain
from mcp.router_self_tuner import AdaptiveRouterSelfTunerV2

logger = logging.getLogger("Qube.LLM")


class LLMWorker(QThread):
    sentence_ready = pyqtSignal(str, str)
    token_streamed = pyqtSignal(str, str)  # session_id, token
    status_update = pyqtSignal(str)
    ttft_latency = pyqtSignal(float)
    context_retrieved = pyqtSignal(bool)
    response_finished = pyqtSignal(str, str)
    sources_found = pyqtSignal(str, list)  # session_id, sources
    router_telemetry_updated = pyqtSignal(dict, dict)  # summary, tuner_state
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

    def _ensure_recall_centroid(self) -> None:
        """Lazily build and install the RECALL semantic centroid on the
        cognitive router. Called once on the first turn that uses the
        cognitive router. No-op on subsequent calls and on any failure
        (the router falls back to substring detection)."""
        if not getattr(self, "cognitive_router", None):
            return
        if getattr(self.cognitive_router, "recall_centroid", None) is not None:
            return
        embedder = getattr(self.embedding_cache, "embedder", None)
        if embedder is None:
            return
        try:
            from workers.intent_router import build_centroid
            centroid = build_centroid(embedder, list(self._RECALL_INTENT_EXAMPLES))
            self.cognitive_router.set_recall_centroid(centroid)
            logger.info("[LLM Worker] Recall semantic centroid installed.")
        except Exception:
            logger.exception("[LLM Worker] Failed to build recall centroid")

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
        final_text_out = ""
        try:
            final_text_out = self._execute_llm_turn()
        except Exception:
            logger.exception("[LLM] pipeline failure (routing, tools, or stream)")
            if not str(final_text_out).strip():
                final_text_out = "Sorry, my brain encountered an error."
                self.token_streamed.emit(self.session_id or "", "\n\n*(Pipeline Error)*")
        finally:
            self._close_active_stream()
            self._last_completed_llm_session_id = self.session_id
            self._server_kv_cleared_for_session_id = None
            try:
                enrichment_payload = {
                    "session_id": self.session_id,
                    "last_user_msg_id": getattr(self, "_turn_last_user_msg_id", None),
                    "last_assistant_msg_id": getattr(self, "_turn_last_assistant_msg_id", None),
                    "rag_chunk_ids": list(getattr(self, "_turn_rag_chunk_ids", []) or []),
                }
                self.enrichment_context_ready.emit(enrichment_payload)
            except Exception:
                logger.exception("[LLM] failed to emit enrichment context")
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

        # ============================================================
        # ROUTING START TIME (telemetry)
        # ============================================================
        route_start = time.time()

        logger.info(f"[Router] route={execution_route}")

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
            mem_result = memory_search(
                decision.get("memory_query") or self.prompt,
                query_vector,
                self.store
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
            "Be concise and accurate."
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
                    "Be concise and accurate."
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

        # 🔑 THE FIX: Make the LLM hide its scratchpad and just talk normally!
        elif execution_route in ["WEB", "INTERNET"]:
            system_prompt = (
                "You are Qube. You have just been provided with real-time, live web search results. "
                "You MUST use the TOOLS context provided below to answer the user's query. "
                "Do not state that you are offline or cannot browse the internet. "
                "CRITICAL: Respond directly to the user in a natural, conversational tone. "
                "Do NOT output your internal reasoning, 'Step 1' thoughts, or search metadata. "
                "Output ONLY your final answer. "
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
                    " Respond directly with the final answer and avoid meta preambles. "
                    "Perform any reasoning internally and do not expose chain-of-thought. "
                    "Prioritize clarity and completeness. "
                    "Use short answers for simple questions, but give fuller explanations when the user asks to explain, compare, or summarize."
                )
            else:
                system_prompt += (
                    " Respond directly with the final answer, starting immediately with the answer content. "
                    "Do not include any preamble, planning, or meta commentary. "
                    "Do not restate or analyze the user's request. "
                    "Do not include phrases such as \"Provide an answer\", \"We need to\", \"We should\", "
                    "\"Step 1\", or similar instructional language. "
                    "Perform any reasoning internally and only output the final result. "
                    "Keep the response natural, concise, and focused."
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
                                break

                    except json.JSONDecodeError:
                        continue

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

        return final_text

    def _max_tokens_native_completion(self) -> int:
        """
        Budget for *new* completion tokens in create_chat_completion (not n_ctx).
        Passing the full context window as max_tokens harms quality and can stall streaming.
        """
        ctx = max(512, int(getattr(self, "context_window", 4096)))
        return min(4096, max(256, ctx // 2))

    def _stream_via_native(self, messages: list[dict], all_ui_sources: list) -> str:
        """Stream tokens from NativeLlamaEngine (llama-cpp-python) on a dedicated thread."""
        token_queue: queue.Queue = queue.Queue()
        done_event = threading.Event()
        self._native_engine.enqueue_generation(
            messages,
            self.temperature,
            self._max_tokens_native_completion(),
            token_queue,
            done_event,
        )

        policy = self._native_engine.get_execution_policy()
        strip_thinking = policy.strip_thinking_output
        show_thinking = not strip_thinking
        cot_filter = RedactedThinkingStreamFilter()
        meta_disp = LeadingMetaInstructionStripper()
        meta_tts = LeadingMetaInstructionStripper()
        repetition_guard = StreamRepetitionGuard()
        current_sentence = ""
        final_text = ""
        start = time.time()
        first_token = False
        stream_wall_start = time.time()

        def _emit_filtered(display_fragment: str, tts_fragment: str) -> None:
            nonlocal current_sentence, final_text, first_token
            if not display_fragment and not tts_fragment:
                return
            if display_fragment and not first_token:
                self.ttft_latency.emit((time.time() - start) * 1000)
                first_token = True
            current_sentence += tts_fragment
            final_text += display_fragment
            if display_fragment:
                self.token_streamed.emit(self.session_id or "", display_fragment)
            if tts_fragment and any(p in tts_fragment for p in ".!?"):
                clean = self.clean_text_for_tts(current_sentence)
                if clean:
                    if not bool(getattr(self, "_cancel_requested", False)):
                        self.sentence_ready.emit(clean, self.session_id)
                current_sentence = ""

        def _flush_tail() -> None:
            cot_tail = cot_filter.flush()
            if show_thinking:
                _emit_filtered(meta_disp.flush(), meta_tts.feed(cot_tail))
                _emit_filtered("", meta_tts.flush())
            else:
                _emit_filtered(meta_disp.feed(cot_tail), meta_tts.feed(cot_tail))
                _emit_filtered(meta_disp.flush(), meta_tts.flush())

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
                if show_thinking:
                    disp = meta_disp.feed(raw)
                    tts_piece = meta_tts.feed(cot_filter.feed(raw))
                else:
                    stripped = cot_filter.feed(raw)
                    disp = meta_disp.feed(stripped)
                    tts_piece = meta_tts.feed(stripped)
                _emit_filtered(disp, tts_piece)
                if disp and repetition_guard.observe(disp):
                    logger.error(
                        "[LLM] Native stream degeneration detected (%s); cancelling.",
                        repetition_guard.trip_reason,
                    )
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
                _flush_tail()
                saw_end = True
                break

        if not saw_end:
            _flush_tail()

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