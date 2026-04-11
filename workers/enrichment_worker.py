from PyQt6.QtCore import QThread, QMutex, QMutexLocker
import logging
import time
import numpy as np
import json
import re
from queue import Queue, Empty

logger = logging.getLogger("Qube.EnrichmentWorker")


class EnrichmentWorker(QThread):
    """
    Async Atomic Memory Worker (Memory v5.1 - Full Fidelity Safe Rebuild)

    ✔ Atomic fact extraction
    ✔ Reinforcement scoring
    ✔ Contradiction replacement
    ✔ Cluster-aware memory grouping
    ✔ Scoped deduplication (memory-only)
    ✔ Schema-safe LanceDB writes (no update dependency)
    ✔ v6-ready fields (decay + importance hooks)
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

    # ============================================================
    # CLUSTERING (NOW ACTUALLY USED)
    # ============================================================

    def _get_cluster_key(self, content: str, category: str) -> str:
        """
        Lightweight semantic clustering heuristic.
        Used to group related facts for future retrieval optimization.
        """
        words = content.lower().split()

        key_terms = [w for w in words if len(w) > 6][:3]

        if not key_terms:
            key_terms = [category]

        return f"{category}::" + "_".join(key_terms)

    # ============================================================
    # CONTRADICTION DETECTION
    # ============================================================

    def _is_contradiction(self, a: str, b: str) -> bool:
        a, b = (a or "").lower(), (b or "").lower()

        pairs = [
            ("likes", "dislikes"),
            ("prefer", "avoid"),
            ("use", "stopped"),
            ("is", "is not"),
            ("works with", "does not use"),
        ]

        return any(x in a and y in b or y in a and x in b for x, y in pairs)

    # ============================================================
    # PUBLIC API
    # ============================================================

    def enqueue(self, session_id: str):
        self.queue.put(session_id)

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
        logger.info("[Memory v5.1] Worker started.")

        while self.is_running:
            if not self._is_enabled_read():
                self.msleep(100)
                continue
            try:
                try:
                    session_id = self.queue.get(timeout=0.5)
                except Empty:
                    continue

                self.is_processing = True
                self._process_session(session_id)

            except Exception as e:
                logger.error(f"[Memory v5.1] Loop error: {e}")

            finally:
                self.is_processing = False

    # ============================================================
    # PIPELINE
    # ============================================================

    def _process_session(self, session_id: str):
        if not self._wait_for_chat_llm_idle():
            return

        try:
            all_messages = self.db.get_session_history(session_id)
            messages = all_messages[-self.MAX_MESSAGES:] if all_messages else []
        except Exception as e:
            logger.error(f"[Memory v5.1] DB error: {e}")
            return

        if not messages:
            return

        prompt = self._build_prompt(messages)
        raw = self._generate_memory(prompt)

        if not raw:
            return

        facts = self._extract_json_facts(raw)

        if not facts:
            return

        self._store_facts(facts)

    # ============================================================
    # PROMPT
    # ============================================================

    def _build_prompt(self, messages: list) -> str:
        conversation = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages]
        )

        return f"""
Extract atomic facts from the conversation.

Rules:
- Facts must be durable (not conversational fluff)
- Keep them short and reusable
- No summarization

Return JSON ONLY:

[
  {{
    "category": "preference|project|identity|context",
    "content": "...",
    "confidence": 0.0-1.0
  }}
]

Conversation:
{conversation}
"""

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

    def _store_facts(self, facts):

        records_to_add = []

        for fact in facts:

            try:
                content = (fact.get("content") or "").strip()
                category = fact.get("category", "context")
                confidence = float(fact.get("confidence", 0.0))

                if (
                    not content
                    or len(content) > self.MAX_FACT_LENGTH
                    or confidence < self.MIN_CONFIDENCE
                ):
                    continue

                vector = self.embedder.embed_query(content)

                existing = self._find_duplicate(vector)

                cluster_key = self._get_cluster_key(content, category)

                # ========================================================
                # MEMORY PAYLOAD (v5.1 + v6 READY)
                # ========================================================

                base_payload = {
                    "type": "fact",
                    "category": category,
                    "content": content,
                    "confidence": confidence,

                    # v5 core
                    "strength": 1,

                    # clustering (now active)
                    "cluster": cluster_key,

                    # v6 READY (not used yet)
                    "importance": confidence,
                    "decay": 1.0,

                    "timestamp": int(time.time())
                }

                if existing:

                    old_payload = {}
                    try:
                        old_payload = json.loads(existing.get("text", "{}"))
                    except Exception:
                        pass

                    old_strength = min(old_payload.get("strength", 1), self.MAX_STRENGTH)

                    # contradiction → reset reinforcement
                    if self._is_contradiction(old_payload.get("content", ""), content):
                        base_payload["strength"] = 1

                    else:
                        base_payload["strength"] = min(old_strength + 1, self.MAX_STRENGTH)

                    # SAFE replace (no update dependency)
                    try:
                        self.store.table.delete(f"id = '{existing.get('id')}'")
                    except Exception:
                        pass

                records_to_add.append({
                    "text": json.dumps(base_payload),
                    "vector": vector,
                    "source": f"qube_memory::{category}",
                    "chunk_id": 0
                })

            except Exception as e:
                logger.debug(f"[Memory v5.1] fact error: {e}")

        # ============================================================
        # WRITE
        # ============================================================

        if records_to_add:
            try:
                self.store.table.add(records_to_add)
                logger.info(f"[Memory v5.1] stored {len(records_to_add)} facts")
            except Exception as e:
                logger.error(f"[Memory v5.1] write failed: {e}")