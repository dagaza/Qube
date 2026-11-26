"""Plan S34 (D48, D49): cross-encoder shadow scores for library and memory recall.

During a worker turn, plain-chat library search, ``@library`` and memory recall
offer their candidate pools (before the floors) and their kept rows to
``core.knowledge.ranking.library_memory_shadow``. After the reply, a daemon
thread scores them and logs one ``[RerankShadow]`` line per search: scores and
counts only. Nothing a search keeps changes, and nothing is collected outside a
turn or without the reranker. A fake scorer stands in for the model.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.ranking import library_memory_shadow as shadow  # noqa: E402
from core.knowledge.ranking import rerank  # noqa: E402
from mcp.memory_tool import memory_search  # noqa: E402
from mcp.rag_tool import rag_search  # noqa: E402

LOGGER = "Qube.Knowledge.RerankShadow"


def _scorer_from(table: dict[str, float], default: float = -11.0):
    """Score a passage by the first table key it contains."""

    def score(_question: str, docs: list[str]) -> list[float]:
        return [next((v for k, v in table.items() if k in d), default) for d in docs]

    return score


class _ShadowCase(unittest.TestCase):
    def setUp(self):
        shadow.end_turn()
        rerank.set_reranker_scorer_for_tests(_scorer_from({"cat": 5.0, "deposit": -5.0}))

    def tearDown(self):
        shadow.end_turn()
        rerank.set_reranker_scorer_for_tests(None)


# ---------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------


class TestCollection(_ShadowCase):
    def test_nothing_is_collected_without_the_reranker(self):
        rerank.set_reranker_scorer_for_tests(None)
        with patch.object(rerank, "reranker_available", return_value=False):
            self.assertFalse(shadow.begin_turn())
        shadow.offer(shadow.PATH_LIBRARY_CHAT, "can I keep a cat", [("a cat clause", True)], top_k=5)
        self.assertEqual(shadow.end_turn(), [])

    def test_nothing_is_collected_outside_a_turn(self):
        shadow.offer(shadow.PATH_LIBRARY_CHAT, "can I keep a cat", [("a cat clause", True)], top_k=5)
        self.assertEqual(shadow.end_turn(), [])

    def test_core_and_narrative_are_not_shadow_paths(self):
        self.assertTrue(shadow.begin_turn())
        for path in ("memory_core", "memory_narrative", "help"):
            shadow.offer(path, "q", [("text", True)], top_k=5)
        self.assertEqual(shadow.end_turn(), [])

    def test_kept_rows_survive_the_candidate_cap(self):
        self.assertTrue(shadow.begin_turn())
        rows = [(f"other row {i}", False) for i in range(shadow.MAX_CANDIDATES + 5)] + [("kept row", True)]
        shadow.offer(shadow.PATH_LIBRARY_CHAT, "q", rows, top_k=5)
        (job,) = shadow.end_turn()
        self.assertEqual(len(job.candidates), shadow.MAX_CANDIDATES)
        self.assertTrue(job.candidates[0].kept)


class TestRecord(_ShadowCase):
    def _job(self, question="Can I keep a cat in the flat?"):
        return shadow.ShadowJob(
            path=shadow.PATH_LIBRARY_CHAT,
            question=question,
            top_k=5,
            scoped=False,
            candidates=(
                shadow.ShadowCandidate("lease_notes.md. One cat is allowed.", True),
                shadow.ShadowCandidate("lease_notes.md. A pet deposit is not required.", True),
                shadow.ShadowCandidate("budget.md. Rent is due on the 1st.", False),
            ),
        )

    def test_counts_and_scores(self):
        rec = shadow.score_job(self._job())
        self.assertIsNone(rec["skipped_reason"])
        self.assertEqual((rec["candidates"], rec["kept"]), (3, 2))
        self.assertEqual(rec["kept_scores"], [5.0, -5.0])
        self.assertEqual(rec["pool_max"], 5.0)
        self.assertEqual(rec["pool_at_or_above"], {"-6": 2, "-4": 1, "0": 1})
        self.assertEqual(rec["kept_below"], {"-6": 0, "-4": 1, "0": 1})
        self.assertEqual((rec["calibrated_keep"], rec["would_keep"], rec["would_abstain"]), (-4.0, 1, False))

    def test_the_log_line_holds_no_text(self):
        job = self._job()
        with self.assertLogs(LOGGER, level="INFO") as logs:
            shadow.score_jobs([job])
        (line,) = logs.output
        payload = json.loads(line.split("[RerankShadow] ", 1)[1])
        self.assertEqual(payload["path"], shadow.PATH_LIBRARY_CHAT)
        for word in ("cat", "flat", "lease_notes", "deposit", "budget", "Rent"):
            self.assertNotIn(word, line)

    def test_a_skipped_question_logs_the_reason_only(self):
        rec = shadow.score_job(self._job("¿Puedo tener un gato en el piso?"))
        self.assertEqual(rec["skipped_reason"], rerank.SKIP_NON_ENGLISH)
        self.assertNotIn("kept_scores", rec)

    def test_a_scorer_failure_never_raises(self):
        def boom(_q, _d):
            raise RuntimeError("onnx")

        rerank.set_reranker_scorer_for_tests(boom)
        rec = shadow.score_job(self._job())
        self.assertEqual(rec["skipped_reason"], rerank.SKIP_ERROR)

    def test_async_scoring_logs_one_line_per_job(self):
        with self.assertLogs(LOGGER, level="INFO") as logs:
            thread = shadow.score_jobs_async([self._job(), self._job()])
            thread.join(10)
        self.assertEqual(len(logs.output), 2)
        self.assertIsNone(shadow.score_jobs_async([]))


# ---------------------------------------------------------------------------
# rag_search and memory_search: offers, and nothing kept changes
# ---------------------------------------------------------------------------


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_a):
        return self

    def where(self, *_a, **_k):
        return self

    def limit(self, _n):
        return self

    def to_list(self):
        return [dict(r) for r in self._rows]


class _Store:
    def __init__(self, vector_rows, fts_rows=()):
        self.table = SimpleNamespace(
            search=lambda q=None, query_type=None, **_k: _Query(list(fts_rows) if query_type == "fts" else vector_rows)
        )


def _chunk(source, text, cid, distance=None):
    row = {"source": source, "text": text, "chunk_id": cid}
    if distance is not None:
        row["_distance"] = distance
    return row


LEASE = "lease_notes.md"
MEMORY_ROW = _chunk("qube_memory::preference::x", json.dumps({"content": "The user has a cat."}), 0, 0.2)


class TestRagSearchShadow(_ShadowCase):
    VECTOR = [
        _chunk(LEASE, "## Pets\n\nOne cat is allowed.", 1, 0.30),
        _chunk(LEASE, "## Deposit\n\nNo pet deposit is required.", 2, 0.45),
        _chunk("budget.md", "## Fixed costs\n\nRent is due on the 1st.", 0, 0.80),  # below the 0.30 floor
        MEMORY_ROW,
    ]
    FTS = [_chunk(LEASE, "## Ending the tenancy\n\nThe cat leaves with the tenant.", 6)]

    def _run(self, shadow_path=None):
        store = _Store(self.VECTOR, self.FTS)
        return rag_search("can I keep a cat", np.zeros(8, dtype=np.float32), store, top_k=5, shadow_path=shadow_path)

    def test_kept_rows_are_identical_with_shadow_on(self):
        off = self._run()
        self.assertTrue(shadow.begin_turn())
        on = self._run(shadow.PATH_LIBRARY_CHAT)
        self.assertEqual(on, off)

    def test_the_pool_is_every_document_candidate_before_the_floor(self):
        self.assertTrue(shadow.begin_turn())
        out = self._run(shadow.PATH_LIBRARY_CHAT)
        (job,) = shadow.end_turn()
        self.assertEqual((job.path, job.question, job.top_k, job.scoped), (shadow.PATH_LIBRARY_CHAT, "can I keep a cat", 5, False))
        passages = {c.passage: c.kept for c in job.candidates}
        self.assertEqual(len(passages), 4)  # 3 vector document rows + 1 FTS row; the memory row is not a document
        self.assertFalse(any("The user has a cat" in p for p in passages))
        self.assertTrue(all(p.startswith(("lease_notes.md. ", "budget.md. ")) for p in passages))
        kept = [p for p, k in passages.items() if k]
        self.assertEqual(len(kept), len(out["sources"]))
        self.assertFalse(passages["budget.md. ## Fixed costs Rent is due on the 1st."])

    def test_all_rows_below_the_floor_offers_none_kept(self):
        self.assertTrue(shadow.begin_turn())
        store = _Store([_chunk("budget.md", "Rent is due on the 1st.", 0, 0.9)])
        out = rag_search("can I keep a cat", np.zeros(8, dtype=np.float32), store, shadow_path=shadow.PATH_LIBRARY_CHAT)
        self.assertEqual(out["sources"], [])
        (job,) = shadow.end_turn()
        self.assertEqual([c.kept for c in job.candidates], [False])

    def test_no_shadow_path_offers_nothing(self):
        self.assertTrue(shadow.begin_turn())
        self._run()
        self.assertEqual(shadow.end_turn(), [])


def _memory(rowid, content, distance, tier="preference"):
    payload = {"content": content, "confidence": 0.8, "category": tier, "strength": 1, "decay": 1.0}
    return {"_rowid": rowid, "text": json.dumps(payload), "source": f"qube_memory::{tier}::{rowid}", "_distance": distance}


class TestMemorySearchShadow(_ShadowCase):
    ROWS = [
        _memory(1, "The user's cat is called Miso.", 0.20, "knowledge"),
        _memory(2, "The user prefers metric units.", 0.50),
        _memory(3, "The user bakes sourdough.", 0.90),  # below the 0.35 floor
    ]

    def _run(self, shadow_path=None):
        return memory_search(
            "what is my cat called",
            np.zeros(8, dtype=np.float32),
            _Store(self.ROWS),
            include_knowledge=True,
            apply_mmr=True,
            apply_temporal_decay=True,
            shadow_path=shadow_path,
        )

    def test_kept_memories_are_identical_with_shadow_on(self):
        off = self._run()
        self.assertTrue(shadow.begin_turn())
        on = self._run(shadow.PATH_MEMORY_RECALL)
        self.assertEqual(on, off)

    def test_the_pool_is_every_candidate_before_the_floor(self):
        self.assertTrue(shadow.begin_turn())
        out = self._run(shadow.PATH_MEMORY_RECALL)
        (job,) = shadow.end_turn()
        passages = {c.passage: c.kept for c in job.candidates}
        self.assertEqual(set(passages), {r and json.loads(r["text"])["content"] for r in self.ROWS})
        self.assertEqual({p for p, k in passages.items() if k}, {s["content"] for s in out["memory_sources"]})
        kept_in_order = [c.passage for c in job.candidates if c.kept]
        self.assertEqual(kept_in_order, [s["content"] for s in out["memory_sources"]])  # rank order
        self.assertFalse(passages["The user bakes sourdough."])
        self.assertEqual(job.top_k, 5)


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------


class TestWorkerWiring(_ShadowCase):
    def test_hybrid_search_offers_from_the_primary_search_only(self):
        from workers.llm_worker import LLMWorker

        calls = []

        def fake_rag(query, _vec, _store, **kwargs):
            calls.append((query, kwargs.get("shadow_path")))
            return {"llm_context": "", "sources": []}

        fake = SimpleNamespace(store=None, embedding_cache=SimpleNamespace(get_embedding=lambda _t: np.zeros(8)))
        expansion = SimpleNamespace(expanded_query="can I keep a cat under the lease")
        with patch("workers.llm_worker.rag_search", side_effect=fake_rag), patch(
            "workers.llm_worker.memory_search", side_effect=lambda q, v, s, **kw: calls.append((q, kw.get("shadow_path"))) or {}
        ):
            LLMWorker._rag_search_hybrid(fake, "can I keep a cat", np.zeros(8), expansion, shadow_path=shadow.PATH_LIBRARY_CHAT)
            LLMWorker._memory_search_hybrid(fake, "my cat", np.zeros(8), expansion, shadow_path=shadow.PATH_MEMORY_RECALL)
        self.assertEqual(
            [p for _q, p in calls],
            [shadow.PATH_LIBRARY_CHAT, None, shadow.PATH_MEMORY_RECALL, None],
        )

    def test_worker_source_wires_the_paths_and_the_after_reply_scoring(self):
        src = (ROOT / "workers" / "llm_worker.py").read_text(encoding="utf-8")
        run_body = src[src.index("    def run(self):") : src.index("    def _execute_llm_turn(self)")]
        self.assertIn("shadow_lm.begin_turn()", run_body)
        end = run_body.index("shadow_lm.score_jobs_async(shadow_lm.end_turn())")
        self.assertLess(end, run_body.index("self.response_finished.emit(self.session_id, final_text_out)"))
        self.assertGreater(end, run_body.index("finally:"))
        self.assertIn("shadow_path=None if prefer_episode else shadow_lm.PATH_MEMORY_RECALL", src)
        self.assertIn("shadow_path=shadow_lm.PATH_LIBRARY_CHAT", src)
        core_call = src[src.index("top_k=3,\n                    apply_core_memory_gate=True") :][:600]
        self.assertNotIn("shadow_path", core_call)
        library = (ROOT / "core" / "knowledge" / "adapters" / "lancedb_library.py").read_text(encoding="utf-8")
        self.assertTrue(re.search(r"shadow_path=PATH_LIBRARY_TOOL", library))


if __name__ == "__main__":
    unittest.main()
