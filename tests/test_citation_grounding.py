"""Plan S11: post-generation citation grounding telemetry (log-only)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import citation_grounding as cg  # noqa: E402
from core.knowledge.ranking.rerank import set_reranker_scorer_for_tests  # noqa: E402

SOURCES = [
    {"id": 1, "filename": "Heat pump", "content": "A heat pump moves heat from outside air into a building using a refrigerant cycle."},
    {"id": 2, "filename": "Boiler costs", "content": "Gas boilers cost less to install but more to run in most climates."},
]


def _keyword_scorer(_q: str, docs: list[str]) -> list[float]:
    """Fake cross-encoder: 6 if the doc mentions the claim's topic, else -6."""
    q = _q.lower()
    out = []
    for d in docs:
        dl = d.lower()
        out.append(6.0 if ("heat pump" in q and "refrigerant" in dl) or ("boiler" in q and "boiler" in dl) else -6.0)
    return out


class ClaimExtractionTests(unittest.TestCase):
    def test_sentence_with_citations(self) -> None:
        claims = cg.extract_cited_claims("A heat pump moves heat [1]. Boilers cost more to run [2]. No cite here.")
        self.assertEqual([c.text for c in claims], ["A heat pump moves heat.", "Boilers cost more to run."])
        self.assertEqual([c.cited_ids for c in claims], [("1",), ("2",)])

    def test_trailing_citation_stack_belongs_to_previous_sentence(self) -> None:
        claims = cg.extract_cited_claims("Earnings grow tax-free. Withdrawals after 59 are tax-free. [1][2][3]")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].text, "Withdrawals after 59 are tax-free.")
        self.assertEqual(claims[0].cited_ids, ("1", "2", "3"))
        self.assertEqual(claims[0].span, "Earnings grow tax-free. Withdrawals after 59 are tax-free.")

    def test_span_stops_at_previous_citation_and_paragraph(self) -> None:
        text = "First fact is here [1]. Second uncited sentence. Third cited sentence [2].\nNew paragraph claim [3]."
        claims = cg.extract_cited_claims(text)
        self.assertEqual(claims[0].span, "First fact is here.")
        self.assertEqual(claims[1].span, "Second uncited sentence. Third cited sentence.")
        self.assertEqual(claims[2].span, "New paragraph claim.")

    def test_markdown_and_abbreviations(self) -> None:
        text = "- **Cost:** about $10k for most homes [1]\n1. Efficiency is high, e.g. a COP of 3 [2]."
        claims = cg.extract_cited_claims(text)
        self.assertEqual([c.text for c in claims], ["Cost: about $10k for most homes", "Efficiency is high, e.g. a COP of 3."])

    def test_web_token_and_dedup(self) -> None:
        claims = cg.extract_cited_claims("The rate rose in October [W][W].")
        self.assertEqual(claims[0].cited_ids, ("W",))

    def test_short_or_empty(self) -> None:
        self.assertEqual(cg.extract_cited_claims(""), [])
        self.assertEqual(cg.extract_cited_claims("Yes [1]."), [])
        self.assertEqual(cg.extract_cited_claims("[1][2]"), [])


class SourceWindowTests(unittest.TestCase):
    def test_short_source_is_one_text(self) -> None:
        self.assertEqual(cg.source_windows(SOURCES[0]), ["Heat pump. " + SOURCES[0]["content"]])

    def test_long_source_windows_overlap_and_are_bounded(self) -> None:
        body = " ".join(f"Sentence number {i} says something specific about topic {i}." for i in range(60))
        src = {"id": 1, "filename": "T", "content": body}
        windows = cg.source_windows(src, window_chars=200, max_windows=4)
        self.assertEqual(len(windows), 4)
        self.assertTrue(all(w.startswith("T. ") for w in windows))
        self.assertTrue(all(len(w) <= 3 + 800 for w in windows))
        head = cg.source_windows(src, mode=cg.MODE_HEAD)
        self.assertEqual(len(head), 1)
        self.assertLessEqual(len(head[0]), cg.DEFAULT_TEXT_CHARS)

    def test_empty_source(self) -> None:
        self.assertEqual(cg.source_windows({"id": 1}), [])


class ScoringTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_reranker_scorer_for_tests(None)

    def _score(self, text, sources=SOURCES, **kw):
        set_reranker_scorer_for_tests(_keyword_scorer)
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True):
            return cg.score_citation_grounding(text, sources, **kw)

    def test_supported_and_miscited_pairs(self) -> None:
        report = self._score("A heat pump moves heat with refrigerant [1]. Boilers cost more to run [1][2].")
        self.assertEqual(report.status, "scored")
        scores = {(p.claim_index, p.source_id): p.score for p in report.pairs}
        self.assertEqual(scores, {(0, "1"): 6.0, (1, "1"): -6.0, (1, "2"): 6.0})
        summary = report.summary()
        self.assertEqual(summary["claims"], 2)
        self.assertEqual(summary["pairs_scored"], 3)
        self.assertEqual(summary["claims_best_below_0"], 0)

    def test_best_window_wins(self) -> None:
        filler = " ".join(f"Unrelated sentence {i} about weather patterns and rainfall totals." for i in range(20))
        src = {"id": 1, "filename": "Long", "content": filler + " A refrigerant cycle moves heat."}
        window = self._score("A heat pump uses a cycle [1].", [src])
        head = self._score("A heat pump uses a cycle [1].", [src], mode=cg.MODE_HEAD)
        self.assertEqual(window.pairs[0].score, 6.0)
        self.assertEqual(head.pairs[0].score, -6.0)

    def test_span_unit_uses_paragraph_context(self) -> None:
        text = "Heat pump systems are efficient. They suit most homes [2]."
        sentence = self._score(text)
        span = self._score(text, claim_unit=cg.UNIT_SPAN)
        self.assertEqual(sentence.pairs[0].score, -6.0)  # "They suit most homes" alone
        self.assertEqual(span.claim_unit, cg.UNIT_SPAN)
        self.assertEqual(span.pairs[0].score, -6.0)  # source 2 is about boilers
        self.assertIn("Heat pump systems", span.claims[0].span)

    def test_orphan_and_pair_budget(self) -> None:
        report = self._score("A heat pump moves heat [1][9]. Boilers cost more [2].", max_pairs=1)
        reasons = {(p.claim_index, p.source_id): p.unscored_reason for p in report.pairs if p.score is None}
        self.assertEqual(reasons, {(0, "9"): "orphan", (1, "2"): "pair_budget"})

    def test_text_budget_bounds_scored_windows(self) -> None:
        body = " ".join(f"Sentence {i} says something specific about topic {i}." for i in range(80))
        srcs = [{"id": i, "filename": f"S{i}", "content": body} for i in (1, 2)]
        calls: list[int] = []

        def counting(_q, docs):
            calls.append(len(docs))
            return [0.0] * len(docs)

        set_reranker_scorer_for_tests(counting)
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True):
            report = cg.score_citation_grounding(
                "First claim about topic one [1][2]. Second claim about topic two [1][2].", srcs, max_texts=8
            )
        self.assertEqual(sum(calls), 8)
        reasons = [p.unscored_reason for p in report.pairs if p.score is None]
        self.assertEqual(reasons, ["pair_budget", "pair_budget"])

    def test_skips(self) -> None:
        self.assertEqual(self._score("No citations at all.").skip_reason, cg.SKIP_NO_CITATIONS)
        self.assertEqual(self._score("A heat pump moves heat [1].", [{"id": 1}]).skip_reason, cg.SKIP_NO_SOURCE_TEXT)
        set_reranker_scorer_for_tests(None)
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=False):
            report = cg.score_citation_grounding("A heat pump moves heat [1].", SOURCES)
        self.assertEqual(report.status, "skipped")
        self.assertTrue(report.skip_reason.startswith(cg.SKIP_RERANKER))

    def test_scorer_failure_never_raises(self) -> None:
        def boom(_q, _docs):
            raise RuntimeError("onnx")

        set_reranker_scorer_for_tests(boom)
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True):
            report = cg.score_citation_grounding("A heat pump moves heat [1].", SOURCES)
        self.assertEqual(report.status, "skipped")
        self.assertTrue(report.skip_reason.startswith(cg.SKIP_RERANKER))

    def test_telemetry_payload(self) -> None:
        report = self._score("A heat pump moves heat with refrigerant [1][7].")
        payload = report.telemetry_dict(session_id="s", turn_id=4, execution_route="WEB")
        self.assertEqual(payload["status"], "scored")
        self.assertEqual(payload["session_id"], "s")
        self.assertEqual(payload["turn_id"], 4)
        self.assertEqual(payload["grounding_scores"][0]["scores"], {"1": 6.0, "7": None})
        self.assertEqual(payload["grounding_scores"][0]["unscored"], ["orphan"])
        self.assertIn("latency_ms", payload)
        json.dumps(payload)


class SchedulingTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_reranker_scorer_for_tests(None)

    def test_route_gate(self) -> None:
        text = "A heat pump moves heat [1]."
        self.assertTrue(cg.should_ground_turn(text, SOURCES, "WEB"))
        self.assertTrue(cg.should_ground_turn(text, SOURCES, "internet"))
        self.assertFalse(cg.should_ground_turn(text, SOURCES, "RAG"))
        self.assertFalse(cg.should_ground_turn(text, [], "WEB"))
        self.assertFalse(cg.should_ground_turn("No cites.", SOURCES, "WEB"))

    def test_background_thread_logs_and_never_touches_text(self) -> None:
        set_reranker_scorer_for_tests(_keyword_scorer)
        events: list[tuple[str, dict]] = []
        text = "A heat pump moves heat with refrigerant [1]."
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True), patch(
            "core.llm_structured_log.structured_llm_log", side_effect=lambda e, p: events.append((e, p))
        ):
            thread = cg.run_citation_grounding_telemetry(text, SOURCES, execution_route="WEB", session_id="s", turn_id=2)
            self.assertIsInstance(thread, threading.Thread)
            self.assertNotEqual(thread.ident, threading.get_ident())
            thread.join(5)
        self.assertEqual(text, "A heat pump moves heat with refrigerant [1].")
        self.assertEqual([e for e, _ in events], [cg.GROUNDING_EVENT])
        self.assertEqual(events[0][1]["turn_id"], 2)
        self.assertEqual(events[0][1]["status"], "scored")

    def test_not_started_for_uncited_or_non_web_turns(self) -> None:
        self.assertIsNone(cg.run_citation_grounding_telemetry("No cites.", SOURCES, execution_route="WEB"))
        self.assertIsNone(cg.run_citation_grounding_telemetry("Cited [1].", SOURCES, execution_route="NONE"))

    def test_skipped_turn_is_still_logged(self) -> None:
        """'Telemetry on 100 % of cited WEB turns': a turn the reranker cannot score is logged as skipped."""
        events: list[dict] = []
        with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=False), patch(
            "core.llm_structured_log.structured_llm_log", side_effect=lambda e, p: events.append(p)
        ):
            cg.run_citation_grounding_telemetry(
                "A heat pump moves heat [1].", SOURCES, execution_route="WEB", background=False
            )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "skipped")


class WorkerHookTests(unittest.TestCase):
    def _fake(self):
        from workers.llm_worker import LLMWorker

        calls: list[tuple[str, list]] = []
        fake = SimpleNamespace(session_id="s", _turn_execution_route="WEB", _routing_debug_turn_seq=3)
        fake._schedule_citation_grounding = lambda text, sources: calls.append((text, sources))
        return fake, LLMWorker, calls

    def test_finalize_schedules_once_and_returns_text_unchanged(self) -> None:
        fake, worker, calls = self._fake()
        text = "A heat pump moves heat [1]. Orphan claim here [5]."
        with patch("workers.llm_worker.get_citation_integrity_enforce", return_value=False):
            out = worker._finalize_citation_integrity_text(fake, text, SOURCES, phase="worker_finalize")
            persisted = worker._finalize_citation_integrity_text(fake, text, SOURCES, phase="worker_persist")
        self.assertEqual(out, text)
        self.assertEqual(persisted, text)
        self.assertEqual(len(calls), 1)  # only the finalize phase

    def test_schedule_respects_setting(self) -> None:
        from workers.llm_worker import LLMWorker

        fake = SimpleNamespace(session_id="s", _turn_execution_route="WEB", _routing_debug_turn_seq=3)
        with patch("core.app_settings.get_citation_grounding_telemetry_enabled", return_value=False), patch(
            "core.citation_grounding.run_citation_grounding_telemetry"
        ) as run:
            LLMWorker._schedule_citation_grounding(fake, "A heat pump moves heat [1].", SOURCES)
        run.assert_not_called()
        with patch("core.app_settings.get_citation_grounding_telemetry_enabled", return_value=True), patch(
            "core.citation_grounding.run_citation_grounding_telemetry"
        ) as run:
            LLMWorker._schedule_citation_grounding(fake, "A heat pump moves heat [1].", SOURCES)
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["turn_id"], 3)
        self.assertEqual(run.call_args.kwargs["execution_route"], "WEB")


class SettingsTests(unittest.TestCase):
    def test_default_on_and_follows_reranker(self) -> None:
        from core import app_settings

        store = SimpleNamespace(get=lambda _k, default=None: default)
        with patch.object(app_settings, "_store", return_value=store):
            with patch.object(app_settings, "get_knowledge_reranker_enabled", return_value=True):
                self.assertTrue(app_settings.get_citation_grounding_telemetry_enabled())
            with patch.object(app_settings, "get_knowledge_reranker_enabled", return_value=False):
                self.assertFalse(app_settings.get_citation_grounding_telemetry_enabled())

    def test_reset_section_includes_key(self) -> None:
        from core import app_settings
        from core.settings_section_reset import SECTION_SETTING_KEYS

        self.assertIn(app_settings.KEY_KNOWLEDGE_CITATION_GROUNDING_ENABLED, SECTION_SETTING_KEYS["knowledge"])


class TraceJoinTests(unittest.TestCase):
    def _write(self, path: Path, payloads: list[dict], *, prefixed: bool) -> None:
        lines = []
        for p in payloads:
            body = json.dumps(p)
            lines.append(f"[2026-11-24 10:00:00] [INFO] [Qube.WebSearchAudit] {body}" if prefixed else body)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_reader_parses_sink_lines_and_joins_grounding(self) -> None:
        from core.knowledge.retrieval_trace_reader import format_retrieval_trace_summary, read_retrieval_traces

        trace = {"event": "retrieval_trace", "session_id": "s", "turn_id": 3, "knowledge_service": "general_web"}
        other = {"event": "retrieval_trace", "session_id": "s", "turn_id": 4}
        grounding = {
            "event": "citation_grounding", "session_id": "s", "turn_id": 3, "status": "scored",
            "claims": 2, "pairs_scored": 3, "score_median": 1.5, "score_min": -2.0, "claims_best_below_0": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "web_search.log"
            for prefixed in (True, False):
                self._write(path, [trace, other, grounding], prefixed=prefixed)
                traces = read_retrieval_traces(log_path=path)
                self.assertEqual(len(traces), 2, prefixed)
                by_turn = {t["turn_id"]: t for t in traces}
                self.assertEqual(by_turn[3]["citation_grounding"]["claims"], 2)
                self.assertNotIn("citation_grounding", by_turn[4])
                summary = format_retrieval_trace_summary(by_turn[3])
                self.assertIn("Citation grounding (relevance proxy, not a verdict): 2 cited sentence(s)", summary)

    def test_event_recorded_only_when_audit_enabled(self) -> None:
        from core.knowledge import observability

        with patch.object(observability, "web_search_audit_log_enabled", return_value=False), patch(
            "logging.Logger.info"
        ) as info:
            observability.record_citation_grounding_event({"status": "scored"})
        info.assert_not_called()
        with patch.object(observability, "web_search_audit_log_enabled", return_value=True), patch(
            "logging.Logger.info"
        ) as info:
            observability.record_citation_grounding_event({"status": "scored", "turn_id": 1})
        body = json.loads(info.call_args.args[0])
        self.assertEqual(body["event"], "citation_grounding")
        self.assertEqual(body["turn_id"], 1)
        row = {"claim": 0, "preview": "private answer text", "scores": {"1": 2.0}}
        with patch.object(observability, "web_search_audit_log_enabled", return_value=True), patch(
            "core.web_search_audit.web_search_audit_redact_enabled", return_value=True
        ), patch("logging.Logger.info") as info:
            observability.record_citation_grounding_event({"status": "scored", "grounding_scores": [row]})
        body = json.loads(info.call_args.args[0])
        self.assertEqual(body["grounding_scores"], [{"claim": 0, "scores": {"1": 2.0}}])
        self.assertTrue(body["previews_redacted"])

    def test_skipped_line(self) -> None:
        from core.knowledge.retrieval_trace_reader import format_citation_grounding_line

        self.assertEqual(format_citation_grounding_line({}), "")
        self.assertEqual(
            format_citation_grounding_line({"status": "skipped", "skip_reason": "no_citations"}),
            "Citation grounding: not scored (no_citations)",
        )


class CalibrationCorpusTests(unittest.TestCase):
    """The committed S11 calibration set stays well-formed (no reranker needed)."""

    def test_corpus_shape(self) -> None:
        path = ROOT / "eval" / "citation_grounding" / "v1_labelled_pairs.json"
        corpus = json.loads(path.read_text(encoding="utf-8"))
        pairs = corpus["pairs"]
        self.assertGreaterEqual(len(pairs), 150)
        self.assertEqual(len({p["pair_id"] for p in pairs}), len(pairs))
        for p in pairs:
            self.assertIn(p["label"], ("supported", "partial", "unsupported"))
            self.assertEqual(p["label_source"], "model")
            self.assertIn(p["kind"], ("cited", "uncited_same_answer"))
            self.assertTrue(p["claim"] and p["source_text"])
        # Every cited pair is reproducible from its stored answer.
        answers = {(a["fixture_id"], a["model"]): a for a in corpus["answers"]}
        for p in pairs:
            if p["kind"] != "cited":
                continue
            a = answers[(p["fixture_id"], p["model"])]
            claims = {c.text: c.cited_ids for c in cg.extract_cited_claims(a["answer"])}
            self.assertIn(p["source_id"], claims.get(p["claim"], ()), p["pair_id"])

    def test_eval_metrics(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        import evaluate_citation_grounding as ev

        self.assertEqual(ev.auc([3.0, 2.0], [1.0, 0.0]), 1.0)
        self.assertEqual(ev.auc([1.0], [1.0]), 0.5)
        self.assertIsNone(ev.auc([], [1.0]))
        self.assertAlmostEqual(ev.spearman([1.0, 2.0, 3.0], [0, 1, 2]), 1.0)
        current = {"default_mode": "sentence/window", "modes": {"sentence/window": {"auc_supported_vs_unsupported": 0.90}}}
        base = {"modes": {"sentence/window": {"auc_supported_vs_unsupported": 0.95}}}
        self.assertEqual(len(ev.compare(current, base)), 1)
        base["modes"]["sentence/window"]["auc_supported_vs_unsupported"] = 0.91
        self.assertEqual(ev.compare(current, base), [])


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    unittest.main()
