"""Plan S13: A/B harness for the scientific heuristics (routing arms, ablations, capture)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge import quality_eval as qe  # noqa: E402
from core.knowledge.scientific_adapters import (  # noqa: E402
    ROUTE_ARMS,
    ROUTE_NO_DISCIPLINE,
    ROUTE_NO_QUERY_TYPE,
    ROUTE_PRODUCTION,
    apply_scientific_adapter_policy,
    scientific_adapter_routes,
)
from core.knowledge.source_preferences import get_effective_enabled_adapters  # noqa: E402

ENABLED = get_effective_enabled_adapters("scientific_evidence", stored_preferences={})


def _load_capture_tool():
    spec = importlib.util.spec_from_file_location(
        "capture_knowledge_fixture_s13", ROOT / "tools" / "capture_knowledge_fixture.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(title: str, adapter: str, rank: int, *, label: str = "relevant", added: bool = False, **extra):
    hit = {"query": "production", "adapter": adapter, "rank": rank}
    if added:
        hit["route_added"] = True
    return {"title": title, "_adapter": adapter, "label": label, "label_source": "model",
            "retrieved": [hit], **extra}


class RouteArmTests(unittest.TestCase):
    def test_production_arm_is_the_adapter_policy(self) -> None:
        for q in ("Which catalysts work best for the hydrogen evolution reaction?",
                  "What is the standard of care for plantar fasciitis?", "Effect of sleep on memory"):
            with self.subTest(q=q):
                routes = scientific_adapter_routes(ENABLED, query=q)
                self.assertEqual(set(routes), set(ROUTE_ARMS))
                self.assertEqual(routes[ROUTE_PRODUCTION], apply_scientific_adapter_policy(ENABLED, query=q))

    def test_no_discipline_arm_routes_like_general_science(self) -> None:
        general = apply_scientific_adapter_policy(ENABLED, query="Effect of sleep on memory")
        routes = scientific_adapter_routes(ENABLED, query="Does quantitative easing raise inflation?")
        self.assertEqual(routes[ROUTE_PRODUCTION][0], "repec")
        self.assertEqual(routes[ROUTE_NO_DISCIPLINE], general)

    def test_no_query_type_arm_drops_added_institutional_adapters(self) -> None:
        routes = scientific_adapter_routes(ENABLED, query="What is the standard of care for plantar fasciitis?")
        self.assertIn("nice", routes[ROUTE_PRODUCTION])
        self.assertNotIn("nice", routes[ROUTE_NO_QUERY_TYPE])


class FixturePoolArmTests(unittest.TestCase):
    def _fixture(self) -> dict:
        return {
            "id": "fx", "query_clean": "q", "production_depth": {"a": 2, "b": 2, "c": 2},
            "routes": {"production": {ROUTE_PRODUCTION: ["a", "b"], ROUTE_NO_DISCIPLINE: ["c", "a"],
                                      ROUTE_NO_QUERY_TYPE: ["b", "a"]}},
            "candidates": [
                _row("a0", "a", 0), _row("b0", "b", 0), _row("c0", "c", 0, added=True),
                _row("a2", "a", 2),  # beyond production depth
            ],
        }

    def test_default_pool_leaves_out_route_added_hits(self) -> None:
        self.assertEqual([r["title"] for r in qe.fixture_pool(self._fixture())], ["a0", "b0"])

    def test_arm_pool_uses_the_arm_adapters_in_arm_order(self) -> None:
        fx = self._fixture()
        self.assertEqual(
            [r["title"] for r in qe.fixture_pool(fx, route_arm=ROUTE_NO_DISCIPLINE)], ["c0", "a0"]
        )
        self.assertEqual(
            [r["title"] for r in qe.fixture_pool(fx, route_arm=ROUTE_NO_QUERY_TYPE)], ["b0", "a0"]
        )
        self.assertTrue(qe.fixture_has_routes(fx))

    def test_fixture_without_routes_keeps_its_first_pool(self) -> None:
        fx = self._fixture()
        fx.pop("routes")
        self.assertFalse(qe.fixture_has_routes(fx))
        self.assertEqual(
            [r["title"] for r in qe.fixture_pool(fx, route_arm=ROUTE_NO_DISCIPLINE)], ["a0", "b0"]
        )

    def test_deferred_capture_fails_validation(self) -> None:
        fx = self._fixture()
        fx["deferred_adapters"] = {"production": ["openalex"]}
        self.assertTrue(any("deferred" in e for e in qe.validate_fixture(fx)))


class RankingAblationTests(unittest.TestCase):
    """The replay mirrors ``pipeline_scientific.run``: trial boost + discipline slots."""

    def _rank(self, query: str, rows: list[dict], ablate: tuple[str, ...] = ()) -> list[str]:
        token = qe._EVAL_ABLATE.set(frozenset(ablate))
        try:
            out = qe._token_ranker(query, [dict(r) for r in rows], 1, "scientific_evidence")
        finally:
            qe._EVAL_ABLATE.reset(token)
        return [r["title"] for r in out.kept]

    def test_trial_boost_is_applied_and_can_be_ablated(self) -> None:
        q = "What were the main results of the DAPA-HF trial with dapagliflozin in heart failure?"
        rows = [
            {"title": "Real-world dapagliflozin in heart failure", "snippet": "observational registry dapagliflozin heart failure results", "_adapter": "pubmed"},
            {"title": "DAPA-HF", "snippet": "commentary", "_adapter": "pubmed", "publication_types": ["Randomized Controlled Trial"]},
        ]
        self.assertEqual(self._rank(q, rows), ["DAPA-HF"])
        self.assertEqual(
            self._rank(q, rows, ablate=(qe.ABLATE_TRIAL_GROUNDING,)),
            ["Real-world dapagliflozin in heart failure"],
        )

    def test_chemistry_slot_is_applied_and_can_be_ablated(self) -> None:
        q = "Which catalysts work best for the hydrogen evolution reaction in water splitting?"
        rows = [
            {"title": "Catalysts for the hydrogen evolution reaction in water splitting", "snippet": "hydrogen evolution reaction catalysts water splitting", "_adapter": "openalex"},
            {"title": "Water (CID 962)", "snippet": "Molecular formula: H2O", "_adapter": "pubchem"},
        ]
        self.assertEqual(self._rank(q, rows), ["Water (CID 962)"])
        self.assertEqual(
            self._rank(q, rows, ablate=(qe.ABLATE_DISCIPLINE_SLOTS,)),
            ["Catalysts for the hydrogen evolution reaction in water splitting"],
        )


class CorpusArmLabelTests(unittest.TestCase):
    def test_every_arm_pool_is_labelled(self) -> None:
        corpus = qe.load_json("eval/retrieval_corpus/v1_knowledge_quality.json")
        checked = 0
        for entry in corpus["queries"]:
            if entry.get("path") != "scientific_evidence":
                continue
            fx = qe.load_json(entry["fixture"])
            for mode in ("production", "standalone"):
                if not qe.fixture_has_routes(fx, mode):
                    continue
                for arm in ROUTE_ARMS:
                    with self.subTest(case=entry["id"], mode=mode, arm=arm):
                        pool = qe.fixture_pool(fx, query_mode=mode, route_arm=arm)
                        self.assertEqual([r["title"] for r in pool if qe.candidate_gain(r) is None], [])
                        checked += 1
        self.assertGreaterEqual(checked, 3 * 22)  # 15 S13 cases + 7 extended fixtures


class CaptureAddRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = _load_capture_tool()

    def _fixture(self) -> dict:
        return {
            "id": "fx", "path": "scientific_evidence", "query_clean": "q", "retrieval_query": "q",
            "production_depth": {"pubmed": 3},
            "adapter_queries": {"production": {"pubmed": "q"}},
            "routes": {"production": {ROUTE_PRODUCTION: ["pubmed", "openalex"],
                                      ROUTE_NO_DISCIPLINE: ["openalex", "crossref"],
                                      ROUTE_NO_QUERY_TYPE: ["pubmed", "openalex"]}},
            "deferred_adapters": {"production": ["openalex"]},
            "candidates": [_row("kept", "pubmed", 0, label="partial")],
        }

    def _search(self, results: dict[str, list[dict]]):
        def get_search_function(aid):
            rows = results.get(aid)
            if rows is None:
                return lambda q, max_results=3: (_ for _ in ()).throw(RuntimeError("down"))
            return lambda q, max_results=3: [dict(r) for r in rows]
        return get_search_function

    def test_fills_deferred_into_the_pool_and_marks_arm_only_hits(self) -> None:
        fx = self._fixture()
        results = {"openalex": [{"title": "oa"}], "crossref": [{"title": "kept"}, {"title": "cr"}]}
        with patch("core.knowledge.adapters.registry.get_search_function", self._search(results)), \
                patch.object(self.tool, "_pace", lambda _aid: None):
            added = self.tool.add_routes(fx, depth=3)
        self.assertEqual(added, 2)
        self.assertNotIn("deferred_adapters", fx)
        by_title = {c["title"]: c for c in fx["candidates"]}
        self.assertEqual(by_title["kept"]["label"], "partial")  # existing label untouched
        self.assertEqual(by_title["kept"]["retrieved"][-1], {"query": "production", "adapter": "crossref", "rank": 0, "route_added": True})
        self.assertNotIn("route_added", by_title["oa"]["retrieved"][0])  # deferred production adapter
        self.assertTrue(by_title["cr"]["retrieved"][0]["route_added"])
        self.assertEqual([r["title"] for r in qe.fixture_pool(fx)], ["kept", "oa"])

    def test_adapter_that_does_not_answer_stays_deferred(self) -> None:
        fx = self._fixture()
        results = {"crossref": [{"title": "cr"}]}  # openalex raises
        with patch("core.knowledge.adapters.registry.get_search_function", self._search(results)), \
                patch.object(self.tool, "_pace", lambda _aid: None):
            self.tool.add_routes(fx, depth=3)
        self.assertEqual(fx["deferred_adapters"], {"production": ["openalex"]})
        self.assertNotIn("openalex", fx["adapter_queries"]["production"])


def _load_ab_tool():
    spec = importlib.util.spec_from_file_location("evaluate_heuristic_ab_s13", ROOT / "tools" / "evaluate_heuristic_ab.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AbOutcomeTests(unittest.TestCase):
    """Kept rows are counted by label even when the report cuts their title."""

    def setUp(self) -> None:
        self.tool = _load_ab_tool()

    def test_long_kept_title_is_counted(self) -> None:
        long_title = "Efficacy of dapagliflozin " + "x" * 200
        fx = {"id": "fx", "candidates": [_row(long_title, "openalex", 0, label="relevant"),
                                         _row("short", "pubmed", 0, label="irrelevant")]}
        out =self.tool._outcome({"kept_titles": [long_title[:120], "short"]}, fx)
        self.assertEqual((out["relevant"], out["irrelevant"], out["gain"]), (1, 1, 1))

    def test_unlabelled_or_ambiguous_kept_row_is_refused(self) -> None:
        fx = {"id": "fx", "candidates": [_row("a", "pubmed", 0)]}
        with self.assertRaises(ValueError):
            self.tool._outcome({"kept_titles": ["missing"]}, fx)
        prefix = "p" * 120
        fx = {"id": "fx", "candidates": [_row(prefix + "1", "pubmed", 0, label="relevant"),
                                         _row(prefix + "2", "pubmed", 1, label="irrelevant")]}
        with self.assertRaises(ValueError):
            self.tool._outcome({"kept_titles": [prefix]}, fx)


if __name__ == "__main__":
    unittest.main()
