from PyQt6.QtCore import QThread, QMutex, QMutexLocker
import logging
import time
import numpy as np
import json
import re
from queue import Queue, Empty

from core.memory_filters import (
    detect_explicit_remember,
    is_assistant_failure_message,
    is_thin_content,
)
from core.memory_usage_recorder import (
    KIND_CITED,
    KIND_RETRIEVED,
    get_memory_usage_recorder,
)
from core.memory_negative_list import (
    DEFAULT_REJECT_DISTANCE,
    get_memory_negative_list,
)

logger = logging.getLogger("Qube.EnrichmentWorker")


# Allowed enum values for the Phase A typed extraction schema.
_ALLOWED_SUBJECTS = frozenset({"user", "third_party", "system"})
_ALLOWED_SOURCE_ROLES = frozenset({"user", "assistant", "derived"})
_ALLOWED_DURABILITY = frozenset({"long_term", "session", "transient"})
_ALLOWED_CATEGORIES = frozenset(
    {"preference", "identity", "project", "knowledge", "context"}
)


class EnrichmentWorker(QThread):
    """
    Async Atomic Memory Worker (Memory v6 - Phase A hardening).

    v6 Phase A upgrades on top of the v5.1 base:
    - Typed extraction schema (``subject``, ``source_role``, ``durability``,
      ``provenance_quote``) with explicit negative examples for the
      ``person_name`` and ``no internet`` regressions.
    - Role-aware preprocessing: assistant refusal / limitation messages are
      scrubbed before the extraction prompt is built so they cannot become
      memories.
    - Server-side validation that drops third-party content unless the user
      explicitly asked to remember it, drops system-subject claims, drops
      thin one-token stubs, and drops facts without provenance.
    - Explicit ``remember that ...`` bypass so that user-requested
      third-party / knowledge memories survive the subject filter.
    """

    def __init__(self, llm, embedder, store, db):
        super().__init__()

        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.db = db

        self.queue = Queue()
        self.is_running = True
        self.is_processing = False

        self._enabled_mutex = QMutex()
        self._is_enabled = True

        self.MAX_MESSAGES = 12

        # Quality controls
        self.MIN_CONFIDENCE = 0.65
        self.MAX_FACT_LENGTH = 300

        # Similarity threshold for deduplication
        self.DUPLICATE_DISTANCE_THRESHOLD = 0.22

        # Reinforcement safety cap (prevents infinite growth)
        self.MAX_STRENGTH = 50

        # Phase B: nearest-neighbor cluster join threshold.
        # If the closest existing memory is within this vector distance we
        # adopt its cluster id; otherwise we mint a new cluster. Tuned a
        # little looser than the duplicate threshold so related-but-distinct
        # facts ("I like dark roast" vs "my favorite is arabica") can share
        # a cluster and trigger the contradiction judge.
        self.CLUSTER_JOIN_DISTANCE = 0.30

        # Phase C: maintenance cadences + thresholds.
        # USAGE_DRAIN_INTERVAL_SEC controls how often the recorder queue is
        # flushed to disk. DECAY_SWEEP_INTERVAL_SEC controls the periodic
        # decay/purge sweep. DECAY_PURGE_THRESHOLD is the cutoff below
        # which a memory is removed.
        self.USAGE_DRAIN_INTERVAL_SEC = 30.0
        self.DECAY_SWEEP_INTERVAL_SEC = 24 * 60 * 60.0
        self.DECAY_PURGE_THRESHOLD = 0.15
        self._last_usage_drain = 0.0
        self._last_decay_sweep = 0.0

    # ============================================================
    # CLUSTERING (NOW ACTUALLY USED)
    # ============================================================

    def _get_cluster_key(self, content: str, category: str) -> str:
        """Legacy keyword-based cluster key.

        Retained as a fallback when the embedding-based cluster assignment
        in :meth:`_assign_cluster` fails (e.g. table not yet populated,
        vector search error). Phase B prefers the embedding join.
        """
        words = content.lower().split()

        key_terms = [w for w in words if len(w) > 6][:3]

        if not key_terms:
            key_terms = [category]

        return f"{category}::" + "_".join(key_terms)

    def _assign_cluster(self, vector, category: str) -> str | None:
        """Phase B: embedding-based cluster assignment.

        Finds the nearest existing memory by vector distance on the full
        memory table. If the top hit is within ``CLUSTER_JOIN_DISTANCE`` we
        adopt its ``cluster`` id so related facts share a cluster (enabling
        the contradiction judge to see them together). Otherwise we mint a
        new uuid-based cluster id. No centroid storage is needed — cluster
        identity is purely nearest-neighbor join, which LanceDB supports
        efficiently.

        Returns ``None`` on error so the caller can fall back to the
        keyword-length heuristic.
        """
        try:
            results = (
                self.store.table
                .search(vector)
                .where("source LIKE 'qube_memory::%'")
                .limit(3)
                .to_list()
            )
        except Exception as e:
            logger.debug(f"[Memory v6] _assign_cluster search failed: {e}")
            return None

        if results:
            top = results[0]
            dist = top.get("_distance", 1.0)
            if dist is not None and dist < self.CLUSTER_JOIN_DISTANCE:
                try:
                    old_payload = json.loads(top.get("text", "{}") or "{}")
                except Exception:
                    old_payload = {}
                old_cluster = old_payload.get("cluster")
                if old_cluster:
                    return str(old_cluster)

        import uuid as _uuid
        return f"{category}::c_{_uuid.uuid4().hex[:12]}"

    # ============================================================
    # CONTRADICTION DETECTION
    # ============================================================

    def _is_contradiction(self, a: str, b: str) -> bool:
        """Legacy rule-based contradiction detector.

        Retained as a last-resort fallback; Phase B's
        :meth:`_judge_contradiction` prefers a Jaccard fast-path and a
        short LLM micro-call.
        """
        a, b = (a or "").lower(), (b or "").lower()

        pairs = [
            ("likes", "dislikes"),
            ("prefer", "avoid"),
            ("use", "stopped"),
            ("is", "is not"),
            ("works with", "does not use"),
        ]

        return any(x in a and y in b or y in a and x in b for x, y in pairs)

    def _jaccard(self, a: str, b: str) -> float:
        """Simple token-set Jaccard similarity, used as the fast path of
        the two-stage contradiction judge."""
        ta = {t for t in re.findall(r"\w+", (a or "").lower()) if t}
        tb = {t for t in re.findall(r"\w+", (b or "").lower()) if t}
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _judge_contradiction(self, old_content: str, new_content: str) -> str:
        """Phase B two-stage classifier.

        Returns one of:
          - ``"duplicate"``    : same fact, reinforce strength (do not re-insert)
          - ``"contradiction"``: new fact replaces old (delete old, insert new)
          - ``"complement"``   : facts coexist (keep old, insert new)

        Stage 1 — cheap Jaccard fast path:
          - near-identical text (>= 0.85) is treated as duplicate.
        Stage 2 — LLM micro-call:
          - short prompt, single-word answer, ~10 max tokens. Falls back to
            the legacy rule-based ``_is_contradiction`` on any LLM failure.
        """
        old_s = (old_content or "").strip()
        new_s = (new_content or "").strip()
        if not old_s or not new_s:
            return "complement"

        jaccard = self._jaccard(old_s, new_s)
        if jaccard >= 0.85:
            return "duplicate"

        judge_prompt = (
            "You classify the relationship between two short facts ABOUT THE SAME USER.\n"
            "Respond with EXACTLY one word: duplicate, contradiction, or complement.\n"
            " - duplicate    : A and B assert the same thing (paraphrase).\n"
            " - contradiction: A and B directly conflict; B should replace A.\n"
            " - complement   : A and B are compatible and both can be true.\n\n"
            f"A: {old_s}\n"
            f"B: {new_s}\n\n"
            "Answer:"
        )
        try:
            raw = (self.llm.generate(judge_prompt) or "").strip().lower()
        except Exception as e:
            logger.debug(f"[Memory v6] contradiction judge LLM failed: {e}")
            return "contradiction" if self._is_contradiction(old_s, new_s) else "duplicate"

        raw_first = raw.split()[0] if raw else ""
        if raw_first.startswith("duplicate"):
            return "duplicate"
        if raw_first.startswith("contradict"):
            return "contradiction"
        if raw_first.startswith("complement"):
            return "complement"

        if "contradict" in raw:
            return "contradiction"
        if "duplicate" in raw:
            return "duplicate"
        if "complement" in raw or "compatible" in raw:
            return "complement"

        return "contradiction" if self._is_contradiction(old_s, new_s) else "duplicate"

    # ============================================================
    # PUBLIC API
    # ============================================================

    def enqueue(self, session_id_or_payload) -> None:
        """Queue a session for extraction.

        Accepts either a bare ``session_id`` (legacy) or a dict payload
        containing ``{session_id, last_user_msg_id, last_assistant_msg_id,
        rag_chunk_ids}`` (Phase B per-turn context). When a payload is
        provided the worker will extract only from the exact user+assistant
        pair that ended the turn, and each stored fact will carry the
        matching provenance + ``links_to_document_ids``.
        """
        if isinstance(session_id_or_payload, dict):
            payload = dict(session_id_or_payload)
            if not payload.get("session_id"):
                return
            self.queue.put(payload)
        else:
            if not session_id_or_payload:
                return
            self.queue.put(str(session_id_or_payload))

    def set_enabled(self, enabled: bool) -> None:
        """Toggle enrichment from the main thread; worker stays alive and idles when disabled."""
        with QMutexLocker(self._enabled_mutex):
            self._is_enabled = bool(enabled)
        logger.debug(f"[Memory v5.1] enrichment enabled={self._is_enabled}")

    def _is_enabled_read(self) -> bool:
        with QMutexLocker(self._enabled_mutex):
            return self._is_enabled

    def _wait_for_chat_llm_idle(self) -> bool:
        """
        Avoid concurrent blocking requests to the local LLM (single-slot servers).
        Returns False if we give up after a long wait (skip this extraction).
        """
        deadline = time.time() + 300.0
        while self.llm.isRunning():
            if time.time() > deadline:
                logger.warning(
                    "[Memory v5.1] Chat LLM still busy after 300s; skipping extraction for this turn"
                )
                return False
            self.msleep(100)
        return True

    def stop(self):
        self.is_running = False

    # ============================================================
    # MAIN LOOP
    # ============================================================

    def run(self):
        logger.info("[Memory v6] Worker started.")

        # Run the first decay sweep early so old payloads (without Phase C
        # fields) get migrated and stale rows are pruned at startup.
        self._last_decay_sweep = time.time()

        while self.is_running:
            if not self._is_enabled_read():
                self.msleep(100)
                continue
            try:
                try:
                    item = self.queue.get(timeout=0.5)
                except Empty:
                    # Idle tick: run periodic Phase C maintenance.
                    self._maybe_drain_usage_recorder()
                    self._maybe_run_decay_sweep()
                    continue

                self.is_processing = True
                if isinstance(item, dict):
                    self._process_turn(item)
                else:
                    self._process_session(item)

            except Exception as e:
                logger.error(f"[Memory v6] Loop error: {e}")

            finally:
                self.is_processing = False
                self._maybe_drain_usage_recorder()
                self._maybe_run_decay_sweep()

    # ============================================================
    # PIPELINE
    # ============================================================

    def _process_turn(self, payload: dict) -> None:
        """Phase B per-turn extraction path.

        Narrows extraction to just the user + assistant pair that ended this
        turn so ``source_message_ids`` on each stored memory points at
        exactly those two messages, and so ``links_to_document_ids``
        accurately reflects the RAG chunks that were in context.
        """
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            return

        last_user_msg_id = payload.get("last_user_msg_id")
        last_assistant_msg_id = payload.get("last_assistant_msg_id")
        rag_chunk_ids = list(payload.get("rag_chunk_ids") or [])

        if not self._wait_for_chat_llm_idle():
            return

        try:
            all_messages = self.db.get_session_history(session_id) or []
        except Exception as e:
            logger.error(f"[Memory v6] DB error (turn): {e}")
            return

        if not all_messages:
            return

        # Scope to just the user+assistant pair for this turn when we have ids.
        turn_messages: list[dict]
        if last_user_msg_id or last_assistant_msg_id:
            turn_messages = [
                m
                for m in all_messages
                if m.get("id") in {last_user_msg_id, last_assistant_msg_id}
            ]
            turn_messages.sort(
                key=lambda m: (
                    0 if m.get("id") == last_user_msg_id else 1
                )
            )
        else:
            turn_messages = all_messages[-2:]

        if not turn_messages:
            return

        self._extract_and_store(
            session_id=session_id,
            messages=turn_messages,
            last_user_msg_id=last_user_msg_id,
            last_assistant_msg_id=last_assistant_msg_id,
            rag_chunk_ids=rag_chunk_ids,
        )

    def _process_session(self, session_id: str):
        if not self._wait_for_chat_llm_idle():
            return

        try:
            all_messages = self.db.get_session_history(session_id)
            messages = all_messages[-self.MAX_MESSAGES:] if all_messages else []
        except Exception as e:
            logger.error(f"[Memory v6] DB error: {e}")
            return

        if not messages:
            return

        self._extract_and_store(
            session_id=session_id,
            messages=messages,
            last_user_msg_id=None,
            last_assistant_msg_id=None,
            rag_chunk_ids=[],
        )

    def _extract_and_store(
        self,
        *,
        session_id: str,
        messages: list[dict],
        last_user_msg_id,
        last_assistant_msg_id,
        rag_chunk_ids: list[str],
    ) -> None:
        """Shared extraction path used by both whole-session and per-turn modes.

        All provenance + link fields are threaded down to ``_store_facts`` so
        Phase B's richer payload schema is populated uniformly.
        """
        prompt = self._build_prompt(messages)
        raw = self._generate_memory(prompt)

        facts = self._extract_json_facts(raw) if raw else []

        # Explicit "remember that ..." bypass. Even if the LLM extractor
        # returns nothing (or drops the user's explicit ask because
        # subject=third_party would otherwise be filtered), we pre-seed a
        # knowledge-category fact so the user-requested memory survives.
        last_user = next(
            (m for m in reversed(messages) if (m.get("role") or "") == "user"),
            None,
        )
        if last_user:
            user_text = last_user.get("content") or ""
            body = detect_explicit_remember(user_text)
            if body:
                facts = list(facts) + [
                    {
                        "subject": "third_party",
                        "source_role": "user",
                        "durability": "long_term",
                        "category": "knowledge",
                        "content": body,
                        "provenance_quote": user_text.strip(),
                        "confidence": 0.95,
                        "_explicit_remember": True,
                    }
                ]
                logger.info(
                    "[Memory v6] explicit-remember bypass seeded: %r",
                    body[:80],
                )

        if not facts:
            return

        source_message_ids: list = []
        if last_user_msg_id:
            source_message_ids.append(last_user_msg_id)
        if last_assistant_msg_id:
            source_message_ids.append(last_assistant_msg_id)
        if not source_message_ids:
            # Fallback: use the ids we have from the message rows themselves.
            for m in messages:
                mid = m.get("id")
                if mid:
                    source_message_ids.append(mid)

        # Pass a normalized copy of the conversation text down to
        # ``_validate_fact`` so each candidate's ``provenance_quote`` can be
        # verified as a real substring of what was actually said. This kills
        # the small-LLM failure mode where the extractor copies a proper-noun
        # fact (e.g. "the user's sister is called <name>") straight from the
        # few-shot POSITIVE examples in the prompt.
        conversation_text = "\n".join(
            [(m.get("content") or "") for m in messages or []]
        )

        turn_context = {
            "session_id": session_id,
            "source_message_ids": source_message_ids,
            "rag_chunk_ids": [str(c) for c in (rag_chunk_ids or []) if c],
            "conversation_text": conversation_text,
        }
        self._store_facts(facts, turn_context=turn_context)

    # ============================================================
    # PROMPT
    # ============================================================

    def _build_prompt(self, messages: list) -> str:
        scrubbed = self._scrub_assistant_failures(messages)
        conversation = "\n".join(
            [f"{m['role']}: {m['content']}" for m in scrubbed]
        )

        return f"""You extract DURABLE FACTS ABOUT THE USER from a conversation between the user and an AI assistant. Return a JSON array. If nothing qualifies, return an empty array [].

A durable fact about the user is something like:
- a stable preference ("the user prefers dark mode")
- an identity trait ("the user is a PhD student in linguistics")
- an ongoing project ("the user is writing a thesis on X")
- an explicit thing the user asked you to remember (see POSITIVE example below)

ANTI-LEAKAGE (CRITICAL):
The EXAMPLES block below is for illustration ONLY. It contains placeholder
names, places, and preferences that are NOT part of the real conversation.
You MUST NOT copy any name, place, relationship, or detail from the examples
into your output. Only extract facts that literally appear in the
CONVERSATION block at the very end of this prompt. If the CONVERSATION block
does not contain a durable user fact, return [].

STRICT RULES:
- NEVER extract names, topics, or entities that the user merely asked about.
  A question like "can you tell me about <person>?" does NOT mean that person
  is a fact about the user.
- NEVER extract statements that describe the assistant's capabilities or
  limitations (e.g. "has no internet access", "cannot browse the web",
  "is offline"). Those are stale system claims, not user facts.
- NEVER extract apologies, refusals, or error messages.
- NEVER extract a single bare proper noun as a fact on its own.
- NEVER summarize the conversation.
- NEVER invent facts. Every provenance_quote MUST be a verbatim substring of
  the actual conversation below — if you cannot quote the supporting
  sentence word-for-word from the CONVERSATION block, drop the fact.
- Content must be a complete sentence of at least 3 words with real
  information.

Return JSON ONLY in exactly this schema:

[
  {{
    "subject": "user" | "third_party" | "system",
    "source_role": "user" | "assistant" | "derived",
    "durability": "long_term" | "session" | "transient",
    "category": "preference" | "identity" | "project" | "knowledge" | "context",
    "content": "full sentence describing the fact",
    "provenance_quote": "exact sentence from the conversation that supports this fact",
    "confidence": 0.0
  }}
]

NEGATIVE EXAMPLES (illustrative only — do NOT copy names / details):

  Conversation:
    user: Can you tell me about <EXAMPLE_PERSON>?
    assistant: <EXAMPLE_PERSON> is a researcher mentioned in your documents.
  -> []
  (The user is asking a question. The named person is not a user fact.)

  Conversation:
    user: Check the weather online.
    assistant: I don't have access to the internet right now.
  -> []
  (Assistant refusal is not a user fact.)

  Conversation:
    user: Summarize the introduction of the PDF.
    assistant: The introduction mentions <EXAMPLE_NAME> and 19th century England.
  -> []
  (Document content is not a user fact.)

POSITIVE EXAMPLES (illustrative only — do NOT copy these facts verbatim):

  Conversation:
    user: Please remember that my driving license expires next July.
  -> [
    {{
      "subject": "user",
      "source_role": "user",
      "durability": "long_term",
      "category": "knowledge",
      "content": "The user's driving license expires next July.",
      "provenance_quote": "Please remember that my driving license expires next July.",
      "confidence": 0.95
    }}
  ]

  Conversation:
    user: I always take my coffee black, no sugar.
  -> [
    {{
      "subject": "user",
      "source_role": "user",
      "durability": "long_term",
      "category": "preference",
      "content": "The user drinks coffee black with no sugar.",
      "provenance_quote": "I always take my coffee black, no sugar.",
      "confidence": 0.92
    }}
  ]

===== END OF EXAMPLES — everything below is the REAL conversation =====

Conversation:
{conversation}
"""

    def _scrub_assistant_failures(self, messages: list) -> list:
        """Replace assistant refusal / limitation messages with a placeholder.

        Prevents the extraction LLM from seeing sentences like "I don't have
        access to the internet" and turning them into memories. User messages
        and normal assistant replies pass through unchanged.
        """
        scrubbed: list[dict] = []
        for m in messages or []:
            role = m.get("role") or "user"
            content = m.get("content") or ""
            if role == "assistant" and is_assistant_failure_message(content):
                scrubbed.append(
                    {"role": "assistant", "content": "[failure message omitted]"}
                )
            else:
                scrubbed.append({"role": role, "content": content})
        return scrubbed

    # ============================================================
    # LLM CALL
    # ============================================================

    def _generate_memory(self, prompt: str) -> str:
        try:
            return self.llm.generate(prompt).strip()
        except Exception as e:
            logger.error(f"[Memory v5.1] LLM error: {e}")
            return ""

    # ============================================================
    # JSON EXTRACTION
    # ============================================================

    def _extract_json_facts(self, raw: str):
        try:
            match = re.search(r'\[[\s\S]*\]', raw)

            if match:
                return json.loads(match.group(0))

            return json.loads(raw)

        except Exception:
            logger.warning("[Memory v5.1] JSON parse failed")
            return []

    # ============================================================
    # DUPLICATION SEARCH
    # ============================================================

    def _find_duplicate(self, vector):
        try:
            results = (
                self.store.table
                .search(vector)
                .where("source LIKE 'qube_memory::%'")
                .limit(1)
                .to_list()
            )

            if not results:
                return None

            if results[0].get("_distance", 1.0) < self.DUPLICATE_DISTANCE_THRESHOLD:
                return results[0]

            return None

        except Exception:
            return None

    # ============================================================
    # STORAGE ENGINE (SAFE + FULL FIDELITY)
    # ============================================================

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        """Lower + collapse whitespace so provenance substring checks are
        tolerant to line-wrapping and trailing punctuation differences."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _validate_fact(self, fact: dict, conversation_text: str = "") -> tuple[bool, str]:
        """Apply Phase A server-side validation.

        Returns ``(ok, reason)``. When ``ok`` is False, ``reason`` is a short
        DEBUG-logged explanation of why the candidate was dropped.

        When ``conversation_text`` is provided, candidates whose
        ``provenance_quote`` does not appear (case / whitespace insensitive)
        inside that text are rejected as hallucinations. Candidates that
        were pre-seeded via the explicit-remember bypass skip this check
        (they were synthesized server-side from the user's own message).
        """
        if not isinstance(fact, dict):
            return False, "not a dict"

        explicit_remember = bool(fact.get("_explicit_remember", False))

        content = (fact.get("content") or "").strip()
        if not content:
            return False, "empty content"
        if len(content) > self.MAX_FACT_LENGTH:
            return False, "content exceeds MAX_FACT_LENGTH"

        try:
            confidence = float(fact.get("confidence", 0.0))
        except Exception:
            return False, "non-numeric confidence"
        if confidence < self.MIN_CONFIDENCE:
            return False, f"confidence {confidence:.2f} < {self.MIN_CONFIDENCE}"

        subject = str(fact.get("subject") or "").strip().lower()
        source_role = str(fact.get("source_role") or "").strip().lower()
        durability = str(fact.get("durability") or "").strip().lower()
        category = str(fact.get("category") or "").strip().lower() or "context"

        if subject not in _ALLOWED_SUBJECTS:
            return False, f"invalid subject={subject!r}"
        if source_role not in _ALLOWED_SOURCE_ROLES:
            return False, f"invalid source_role={source_role!r}"
        if durability not in _ALLOWED_DURABILITY:
            return False, f"invalid durability={durability!r}"
        if category not in _ALLOWED_CATEGORIES:
            return False, f"invalid category={category!r}"

        if subject == "system":
            return False, "subject=system is never stored"
        if source_role == "assistant" and not explicit_remember:
            return False, "source_role=assistant without explicit-remember bypass"
        if subject == "third_party" and category != "knowledge":
            return False, "third_party subject requires category=knowledge"
        if durability != "long_term":
            return False, f"durability={durability} (only long_term is stored)"

        if is_thin_content(content):
            return False, "thin content (bare name / < 3 words / stopwords)"
        if is_assistant_failure_message(content):
            return False, "content matches assistant-failure pattern"

        provenance = (fact.get("provenance_quote") or "").strip()
        if not provenance:
            return False, "missing provenance_quote"

        # Hallucination guard: the provenance MUST literally appear in the
        # real conversation. This catches small-LLM failures where the
        # extractor echoes a sentence from the few-shot examples in the
        # prompt (e.g. copying "Remember that my sister is called <name>"
        # verbatim when the user only said something short or ambiguous).
        # Skipped for explicit-remember bypass facts, which are synthesized
        # by the worker from the user's own message.
        if conversation_text and not explicit_remember:
            norm_prov = self._normalize_for_match(provenance)
            norm_conv = self._normalize_for_match(conversation_text)
            if norm_prov and norm_prov not in norm_conv:
                return False, "provenance_quote not found verbatim in conversation"

        return True, "ok"

    def _store_facts(self, facts, turn_context: dict | None = None):
        turn_context = turn_context or {}
        session_id = str(turn_context.get("session_id") or "")
        source_message_ids = list(turn_context.get("source_message_ids") or [])
        links_to_document_ids = list(turn_context.get("rag_chunk_ids") or [])
        conversation_text = str(turn_context.get("conversation_text") or "")
        has_rag_context = bool(links_to_document_ids)

        records_to_add = []

        for fact in facts:
            try:
                ok, reason = self._validate_fact(fact, conversation_text=conversation_text)
                if not ok:
                    logger.debug(
                        "[Memory v6] dropped candidate (%s): %r",
                        reason,
                        (fact.get("content") or "")[:80],
                    )
                    continue

                content = (fact["content"] or "").strip()
                category = str(fact.get("category") or "context").strip().lower()
                confidence = float(fact.get("confidence", 0.0))
                subject = str(fact.get("subject") or "").strip().lower()
                source_role = str(fact.get("source_role") or "").strip().lower()
                durability = str(fact.get("durability") or "long_term").strip().lower()
                provenance_quote = (fact.get("provenance_quote") or "").strip()

                vector = self.embedder.embed_query(content)

                # Phase C: reject candidates that the user has already
                # explicitly removed via the Memory Manager. Cheap L2
                # check against an in-memory matrix kept in sync with
                # ``~/.qube/memory_negatives.json``.
                try:
                    neg_list = get_memory_negative_list()
                    if neg_list.is_negative(vector, threshold=DEFAULT_REJECT_DISTANCE):
                        logger.info(
                            "[Memory v6] dropped candidate (negative_list): %r",
                            content[:80],
                        )
                        continue
                except Exception:
                    pass

                existing = self._find_duplicate(vector)

                # Phase B: embedding-based cluster assignment replaces the
                # keyword-length heuristic. Falls back to the legacy keyword
                # cluster when the vector nearest-neighbor lookup fails.
                cluster_key = self._assign_cluster(vector, category) or self._get_cluster_key(content, category)

                # Derive per-memory Phase B provenance fields.
                if fact.get("_explicit_remember"):
                    origin = "user_stated"
                elif subject == "user" and source_role == "user":
                    origin = "user_stated"
                elif subject == "user" and source_role == "assistant":
                    origin = "user_confirmed"
                elif has_rag_context:
                    origin = "document_derived"
                else:
                    origin = "system_derived"

                # ========================================================
                # MEMORY PAYLOAD (v6 Phase B)
                #
                # Phase A introduced the typed schema (subject / source_role
                # / durability / provenance_quote). Phase B adds explicit
                # provenance (``source_session_id`` / ``source_message_ids``
                # / ``origin``) and link-back to the RAG chunks that were in
                # context when the memory was formed
                # (``links_to_document_ids``). Phase C will add usage
                # counters + decay bookkeeping on top.
                # ========================================================

                now_ts = int(time.time())
                base_payload = {
                    "type": "fact",
                    "category": category,
                    "content": content,
                    "confidence": confidence,

                    # typed schema (Phase A)
                    "subject": subject,
                    "source_role": source_role,
                    "durability": durability,
                    "provenance_quote": provenance_quote,

                    # Phase B provenance
                    "source_session_id": session_id,
                    "source_message_ids": list(source_message_ids),
                    "origin": origin,
                    "links_to_document_ids": list(links_to_document_ids),

                    # v5 core
                    "strength": 1,

                    # clustering (now embedding-based)
                    "cluster": cluster_key,

                    # v6 importance + decay
                    "importance": confidence,
                    "decay": 1.0,

                    # Phase C usage counters + reflection bookkeeping
                    "times_retrieved": 0,
                    "times_cited_positively": 0,
                    "last_used_at": now_ts,
                    "last_reflected_at": 0,
                    "flagged_for_review": False,

                    "timestamp": now_ts
                }

                skip_insert = False
                if existing:
                    old_payload = {}
                    try:
                        old_payload = json.loads(existing.get("text", "{}"))
                    except Exception:
                        pass

                    old_strength = min(old_payload.get("strength", 1), self.MAX_STRENGTH)
                    old_content = old_payload.get("content", "") or ""

                    # Phase B two-stage judge (Jaccard fast-path + LLM micro-call).
                    verdict = self._judge_contradiction(old_content, content)
                    logger.debug(
                        "[Memory v6] contradiction judge verdict=%s old=%r new=%r",
                        verdict, old_content[:60], content[:60],
                    )

                    if verdict == "duplicate":
                        # Reinforce strength on the existing row, do NOT insert a
                        # new row. Use SAFE delete+re-add so we can bump strength.
                        try:
                            self.store.table.delete(f"id = '{existing.get('id')}'")
                        except Exception:
                            pass
                        base_payload["strength"] = min(old_strength + 1, self.MAX_STRENGTH)
                        # Preserve the original cluster identity on duplicates.
                        if old_payload.get("cluster"):
                            base_payload["cluster"] = old_payload["cluster"]
                        # Preserve cumulative usage counters across reinforcement.
                        base_payload["times_retrieved"] = int(old_payload.get("times_retrieved", 0))
                        base_payload["times_cited_positively"] = int(old_payload.get("times_cited_positively", 0))
                        base_payload["last_used_at"] = int(old_payload.get("last_used_at", now_ts))
                    elif verdict == "contradiction":
                        # Replace old with new.
                        try:
                            self.store.table.delete(f"id = '{existing.get('id')}'")
                        except Exception:
                            pass
                        base_payload["strength"] = 1
                    else:
                        # Complement: keep the old row AND insert the new one.
                        # Share the same cluster so future retrieval groups
                        # them together.
                        if old_payload.get("cluster"):
                            base_payload["cluster"] = old_payload["cluster"]
                        base_payload["strength"] = 1

                if not skip_insert:
                    records_to_add.append({
                        "text": json.dumps(base_payload),
                        "vector": vector,
                        "source": f"qube_memory::{category}",
                        "chunk_id": 0
                    })

            except Exception as e:
                logger.debug(f"[Memory v6] fact error: {e}")

        if records_to_add:
            try:
                self.store.table.add(records_to_add)
                logger.info(f"[Memory v6] stored {len(records_to_add)} facts")
            except Exception as e:
                logger.error(f"[Memory v6] write failed: {e}")

    # ============================================================
    # PHASE C: USAGE COUNTERS + DECAY MAINTENANCE
    # ============================================================

    def _load_memory_row(self, memory_id: str) -> dict | None:
        """Fetch a single memory row by lance ``id``."""
        if not memory_id:
            return None
        try:
            safe_id = memory_id.replace("'", "''")
            rows = (
                self.store.table.search()
                .where(f"id = '{safe_id}'")
                .limit(1)
                .to_list()
            )
        except Exception as e:
            logger.debug(f"[Memory v6] _load_memory_row({memory_id}) failed: {e}")
            return None
        return rows[0] if rows else None

    def _rewrite_memory_row(self, row: dict, payload: dict) -> bool:
        """Safely replace ``row`` with the same vector + new ``payload`` JSON.

        Mirrors the existing delete+add pattern used by ``_store_facts``.
        Returns True on success.
        """
        rid = row.get("id")
        if not rid:
            return False
        try:
            safe_id = str(rid).replace("'", "''")
            self.store.table.delete(f"id = '{safe_id}'")
        except Exception as e:
            logger.debug(f"[Memory v6] _rewrite delete failed for {rid}: {e}")
            return False
        try:
            self.store.table.add([{
                "text": json.dumps(payload),
                "vector": row.get("vector"),
                "source": row.get("source"),
                "chunk_id": row.get("chunk_id", 0),
            }])
            return True
        except Exception as e:
            logger.error(f"[Memory v6] _rewrite re-add failed for {rid}: {e}")
            return False

    def _maybe_drain_usage_recorder(self) -> None:
        """Drain the recorder queue at most once every USAGE_DRAIN_INTERVAL_SEC.

        Aggregates all events for a given memory id into a single
        delete+re-add so we don't write amplify when the same row appears
        multiple times in a burst.
        """
        now = time.time()
        if now - self._last_usage_drain < self.USAGE_DRAIN_INTERVAL_SEC:
            return
        self._last_usage_drain = now

        recorder = get_memory_usage_recorder()
        events = recorder.drain(max_items=512)
        if not events:
            return

        deltas: dict[str, dict[str, int]] = {}
        for kind, mid in events:
            d = deltas.setdefault(mid, {"retrieved": 0, "cited": 0})
            if kind == KIND_RETRIEVED:
                d["retrieved"] += 1
            elif kind == KIND_CITED:
                d["cited"] += 1

        applied = 0
        for mid, d in deltas.items():
            row = self._load_memory_row(mid)
            if not row:
                continue
            try:
                payload = json.loads(row.get("text", "{}") or "{}")
            except Exception:
                continue
            payload["times_retrieved"] = int(payload.get("times_retrieved", 0)) + d["retrieved"]
            payload["times_cited_positively"] = int(payload.get("times_cited_positively", 0)) + d["cited"]
            payload["last_used_at"] = int(time.time())
            if self._rewrite_memory_row(row, payload):
                applied += 1

        if applied:
            logger.info(
                f"[Memory v6] drained {len(events)} usage events into {applied} memory row(s)"
            )

    def _maybe_run_decay_sweep(self) -> None:
        """Run the periodic decay/purge sweep at most every
        DECAY_SWEEP_INTERVAL_SEC. Purges rows whose decay drops below
        DECAY_PURGE_THRESHOLD."""
        now = time.time()
        if now - self._last_decay_sweep < self.DECAY_SWEEP_INTERVAL_SEC:
            return
        self._last_decay_sweep = now

        try:
            rows = (
                self.store.table.search()
                .where("source LIKE 'qube_memory::%'")
                .limit(10000)
                .to_list()
            )
        except Exception as e:
            logger.debug(f"[Memory v6] decay sweep load failed: {e}")
            return

        if not rows:
            return

        purged = 0
        rewritten = 0
        for row in rows:
            try:
                payload = json.loads(row.get("text", "{}") or "{}")
            except Exception:
                continue

            last_used = int(payload.get("last_used_at") or payload.get("timestamp") or now)
            days_idle = max(0.0, (now - last_used) / 86400.0)
            retrieved = int(payload.get("times_retrieved", 0))
            cited = int(payload.get("times_cited_positively", 0))
            usefulness = (cited + 1) / (retrieved + 1)

            old_decay = float(payload.get("decay", 1.0))
            # Exponential decay tempered by usefulness:
            # rows that get cited slow their decay; rows that are retrieved
            # but never cited lose ground faster than the base rate.
            new_decay = old_decay * pow(0.99, days_idle) * (0.5 + 0.5 * usefulness)
            new_decay = max(0.0, min(1.0, new_decay))

            if new_decay < self.DECAY_PURGE_THRESHOLD:
                rid = row.get("id")
                if rid:
                    try:
                        safe_id = str(rid).replace("'", "''")
                        self.store.table.delete(f"id = '{safe_id}'")
                        purged += 1
                    except Exception:
                        pass
                continue

            if abs(new_decay - old_decay) > 0.005:
                payload["decay"] = new_decay
                if self._rewrite_memory_row(row, payload):
                    rewritten += 1

        if purged or rewritten:
            logger.info(
                f"[Memory v6] decay sweep: purged={purged} rewritten={rewritten} (n={len(rows)})"
            )