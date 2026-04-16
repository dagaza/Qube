"""
Memory Reflection Worker (Phase C: Tier 2.4 / 3.1).

Periodically audits the LanceDB memory store with the same titler-class
LLM the enrichment pipeline uses. Each batch the worker grabs the
oldest-reflected (or never-reflected) memories, asks the LLM to label
them, and writes ``flagged_for_review = True`` for the ones that look
suspect. The worker NEVER auto-deletes — flagged memories surface in
the Memory Manager view's "Flagged for review" section so the user
makes the final call.

Cadence
-------
- Wakes every ``REFLECT_INTERVAL_SEC`` (default 6 h).
- Each cycle inspects up to ``BATCH_SIZE`` (default 10) memories.
- Skips memories that were reflected within the last
  ``MIN_REFLECT_AGE_SEC`` (default 7 d) so we don't burn LLM cycles on
  the same row over and over.

Labels
------
The LLM must output one of:
- ``durable_user_fact``  -> looks like a stable fact about the user
- ``third_party_stub``   -> bare reference to someone else with no detail
- ``system_claim``       -> claim about the assistant / app limitations
- ``transient``          -> a one-off / time-bound state
- ``unclear``            -> insufficient context to judge

Anything other than ``durable_user_fact`` flips ``flagged_for_review``.

Concurrency
-----------
This worker shares the LanceDB ``store`` with ``EnrichmentWorker``. To
avoid stepping on the enrichment flush path it pauses for a short jitter
and uses the same delete+re-add mechanism (the LanceDB writes are
single-row and the operations are idempotent).
"""
from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Optional

from PyQt6.QtCore import QThread

logger = logging.getLogger("Qube.MemoryReflectionWorker")


# Cadence / batching defaults (constants so unit tests can monkeypatch).
REFLECT_INTERVAL_SEC = 6 * 60 * 60.0
BATCH_SIZE = 10
MIN_REFLECT_AGE_SEC = 7 * 24 * 60 * 60.0   # 7 days
LOOP_TICK_SEC = 30.0
SCAN_LIMIT = 500   # Cap rows pulled into the reflection candidate scan.

VALID_LABELS = {
    "durable_user_fact",
    "third_party_stub",
    "system_claim",
    "transient",
    "unclear",
    # T3.4 structural labels set deterministically by
    # ``_structural_label_for`` BEFORE the LLM judge runs. They are
    # added to the valid set so ``_parse_label`` never rewrites them
    # back to "unclear" on an LLM echo / round-trip.
    "tier_mismatch",
    "orphan_knowledge",
}


class MemoryReflectionWorker(QThread):
    """Periodic LLM-driven self-audit for the memory store."""

    def __init__(self, llm, store, parent=None) -> None:
        super().__init__(parent)
        self.llm = llm
        self.store = store
        self._running = True
        # Stagger the first cycle so we don't slam the LLM on app start.
        self._next_run_at = time.time() + random.uniform(60.0, 300.0)

    # ------------------------- public API -------------------------

    def shutdown(self) -> None:
        self._running = False

    # ------------------------- thread loop -------------------------

    def run(self) -> None:
        while self._running:
            try:
                now = time.time()
                if now >= self._next_run_at:
                    self._run_cycle()
                    self._next_run_at = time.time() + REFLECT_INTERVAL_SEC
            except Exception as e:
                logger.exception("[MemoryReflection] cycle failed: %s", e)

            # Sleep in short ticks so shutdown is responsive.
            for _ in range(int(LOOP_TICK_SEC)):
                if not self._running:
                    return
                self.sleep(1)

    # ------------------------- cycle -------------------------

    def _run_cycle(self) -> None:
        if self.store is None or getattr(self.store, "table", None) is None:
            return
        if self.llm is None:
            return

        candidates = self._fetch_candidates(BATCH_SIZE)
        if not candidates:
            logger.info("[MemoryReflection] no candidates this cycle")
            return

        logger.info(
            "[MemoryReflection] reflecting on %d memories",
            len(candidates),
        )
        for cand in candidates:
            if not self._running:
                return
            try:
                self._reflect_one(cand)
            except Exception as e:
                logger.debug(
                    "[MemoryReflection] _reflect_one(%s) failed: %s",
                    cand.get("id"),
                    e,
                )

    # ------------------------- candidate selection -------------------------

    def _fetch_candidates(self, n: int) -> list[dict]:
        """Return up to ``n`` memory rows that need reflection.

        Selection rule: type=fact rows whose ``last_reflected_at`` is 0
        OR older than ``MIN_REFLECT_AGE_SEC``. Sorted oldest-reflected
        first so we make even progress over the whole store.
        """
        try:
            rows = (
                self.store.table.search()
                .limit(SCAN_LIMIT)
                .to_list()
            )
        except Exception as e:
            logger.debug("[MemoryReflection] scan failed: %s", e)
            return []

        cutoff = time.time() - MIN_REFLECT_AGE_SEC
        out: list[dict] = []
        for r in rows:
            text = r.get("text")
            if not isinstance(text, str) or not text.startswith("{"):
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("type") != "fact":
                continue
            # T3.2: episode summaries are regenerated by the enrichment
            # worker on cadence + idle triggers — they are not the kind
            # of "durable user fact" the reflection judge is meant to
            # rate, and flagging them would just clutter the Memory
            # Manager with false positives.
            if str(payload.get("category") or "").lower() == "episode":
                continue
            # Already flagged? Skip — user must act on it.
            if payload.get("flagged_for_review"):
                continue
            last = float(payload.get("last_reflected_at") or 0)
            if last >= cutoff:
                continue
            out.append({
                "id": str(r.get("id") or ""),
                "vector": r.get("vector"),
                "source": r.get("source") or "qube_memory",
                "chunk_id": int(r.get("chunk_id") or 0),
                "payload": payload,
            })

        # Oldest-reflected first.
        out.sort(key=lambda c: float((c.get("payload") or {}).get("last_reflected_at") or 0))
        return out[:n]

    # ------------------------- per-row reflection -------------------------

    def _reflect_one(self, cand: dict) -> None:
        payload = cand.get("payload") or {}
        content = (payload.get("content") or "").strip()
        if not content:
            return

        # T3.4: structural labels trump the LLM judge. If the stored
        # ``source`` tier disagrees with what ``derive_memory_tier``
        # would assign today (e.g. a row tagged as preference but
        # whose subject is "third_party"), flag as ``tier_mismatch``
        # without spending an LLM cycle. Likewise, a knowledge row
        # that has lost all provenance + links + explicit-remember
        # origin is an ``orphan_knowledge`` candidate.
        structural = self._structural_label_for(cand, payload)
        if structural is not None:
            label = structural
        else:
            prompt = self._build_prompt(payload)
            raw = self._call_llm(prompt)
            label = self._parse_label(raw)
        flagged = label != "durable_user_fact"

        new_payload = dict(payload)
        new_payload["last_reflected_at"] = int(time.time())
        new_payload["flagged_for_review"] = bool(flagged)
        new_payload["reflection_label"] = label  # surfaced in UI tooltip

        ok = self._rewrite_memory(cand, new_payload)
        if ok:
            logger.info(
                "[MemoryReflection] %s -> label=%s flagged=%s",
                (cand.get("id") or "?")[:8],
                label,
                flagged,
            )

    def _structural_label_for(self, cand: dict, payload: dict) -> Optional[str]:
        """Cheap deterministic tier / orphan check (T3.4).

        Runs BEFORE the LLM judge so we never waste a reflection cycle
        on rows whose structural metadata alone is enough to flag them.
        Returns ``None`` when there is no structural issue and the LLM
        judge should run instead.
        """
        source = str(cand.get("source") or "").strip().lower()
        subject = str(payload.get("subject") or "").strip().lower()

        # Tier mismatch: a row parked in the preference namespace must
        # be about the user. A preference row with subject=third_party
        # (or missing subject) is structurally wrong and the user
        # should review it.
        if source.startswith("qube_memory::preference::") and subject != "user":
            return "tier_mismatch"

        # A knowledge-tier row that has lost every piece of evidence
        # it was stored for (provenance quote, RAG document link, or
        # the explicit-remember origin stamp) is an orphan and almost
        # always a stale scrape.
        if source.startswith("qube_memory::knowledge::"):
            prov = str(payload.get("provenance_quote") or "").strip()
            links = payload.get("links_to_document_ids") or []
            origin = str(payload.get("origin") or "").strip().lower()
            explicit = bool(payload.get("_explicit_remember", False))
            if (
                not prov
                and not (isinstance(links, list) and links)
                and origin != "explicit_remember"
                and not explicit
            ):
                return "orphan_knowledge"

        return None

    def _build_prompt(self, payload: dict) -> str:
        content = (payload.get("content") or "").strip()
        subject = str(payload.get("subject") or "").strip().lower() or "unknown"
        category = str(payload.get("category") or "").strip().lower() or "context"
        origin = str(payload.get("origin") or "").strip().lower() or "unknown"
        prov = (payload.get("provenance_quote") or "").strip()
        prov_excerpt = prov[:300] if prov else "(none)"

        return f"""You are auditing a single stored memory in a personal AI assistant.
Decide if the memory is the kind of durable fact about the user that the
assistant should keep referencing, or whether it is a stub / system claim
/ transient note that should be flagged for the user to review.

Return STRICT JSON of the form:
{{ "label": "<one of: durable_user_fact, third_party_stub, system_claim, transient, unclear>" }}

Definitions:
- durable_user_fact: a stable fact about the user (preference, identity,
  ongoing project, long-lived context they would want the assistant to
  remember next month).
- third_party_stub: a bare reference to someone or something other than
  the user, with no useful detail (e.g. just a name).
- system_claim: a claim about the assistant, the app, or its tools
  (e.g. "the assistant doesn't have internet access", "RAG is slow").
- transient: a momentary state, single event, or time-bound note (e.g.
  "the user is hungry right now", "the user is downloading a model").
- unclear: insufficient context to judge.

Memory metadata:
- subject: {subject}
- category: {category}
- origin: {origin}
- provenance quote: "{prov_excerpt}"

Memory content:
\"\"\"{content}\"\"\"

Return ONLY the JSON object. No prose. No markdown."""

    def _call_llm(self, prompt: str) -> str:
        try:
            out = self.llm.generate(prompt)
            return (out or "").strip()
        except Exception as e:
            logger.debug("[MemoryReflection] llm.generate failed: %s", e)
            return ""

    def _parse_label(self, raw: str) -> str:
        """Robust parse: prefer JSON, fall back to keyword scan."""
        if not raw:
            return "unclear"
        # Try strict JSON first.
        match = re.search(r"\{[^{}]*\}", raw)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    label = str(obj.get("label") or "").strip().lower()
                    if label in VALID_LABELS:
                        return label
            except Exception:
                pass
        # Fallback: keyword scan.
        lower = raw.lower()
        for label in (
            "durable_user_fact",
            "third_party_stub",
            "system_claim",
            "transient",
            "unclear",
        ):
            if label in lower:
                return label
        return "unclear"

    # ------------------------- writeback -------------------------

    def _rewrite_memory(self, cand: dict, new_payload: dict) -> bool:
        rid = cand.get("id")
        if not rid:
            return False
        try:
            safe = str(rid).replace("'", "''")
            self.store.table.delete(f"id = '{safe}'")
            self.store.table.add([{
                "text": json.dumps(new_payload),
                "vector": cand.get("vector"),
                "source": cand.get("source") or "qube_memory",
                "chunk_id": int(cand.get("chunk_id") or 0),
            }])
            return True
        except Exception as e:
            logger.warning(
                "[MemoryReflection] rewrite %s failed: %s", rid, e
            )
            return False


__all__ = [
    "MemoryReflectionWorker",
    "REFLECT_INTERVAL_SEC",
    "BATCH_SIZE",
    "MIN_REFLECT_AGE_SEC",
    "VALID_LABELS",
]
