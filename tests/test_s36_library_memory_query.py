"""Plan S36: library and memory search resolve follow-up questions (D51–D54).

``retrieval_text`` (plain-chat library search, memory search and, since S36,
``@library``) uses S6's subject + facet when the subject tracker says the turn
continues the conversation, and otherwise puts the previous user question in
front of a turn that leans on the conversation without naming a subject.
Topic switches are never anchored, request frames stay (D54), and routing
keeps the legacy text (D53). Two discourse fixes (D52) stop "What's" and
"Yes." from becoming referents.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.discourse_intent import FollowUpClassification, FollowUpKind  # noqa: E402
from core.discourse_query import build_resolved_retrieval_query, resolve_routing_query  # noqa: E402
from core.discourse_referent_policy import extract_entity_and_aspect, validate_referent_candidate  # noqa: E402
from core.discourse_subject import context_signals, refers_inside_turn  # noqa: E402
from core.knowledge import library_memory_eval as lme  # noqa: E402
from core.query_resolution_evaluation import build_discourse_resolution  # noqa: E402

MODES = ("balanced", "fast", "power")
BASELINE_DIR = ROOT / "eval" / "library_memory_corpus" / "baselines"


def _u(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def _a(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


def _corpus_cases() -> dict[str, dict]:
    return {c["id"]: c for c in lme.load_corpus()["cases"]}


def _resolve(case: dict):
    return build_discourse_resolution(case["query"], list(case.get("history") or []))


# S33 follow-ups: (case id, library and memory text, reason).
RESOLVED = [
    ("lib_mt_pet_deposit", "Can I keep a cat in my flat? Is there an extra deposit for it?", "prior_question"),
    ("lib_mt_hx9_reset", "my heat pump how do I reset it?", "subject"),
    ("lib_mt_heron_who_decided", "When is the Heron app launching? Who decided that?", "prior_question"),
    ("lib_mt_bike_tarmac", "What tyre pressure should I run on gravel? And on tarmac?", "prior_question"),
    ("lib_mt_ins_theft", "my contents insurance Does it cover theft from the bike store?", "subject"),
    ("lib_mt_svd_pca", "singular value decomposition How does it relate to PCA?", "subject"),
    (
        "lib_mt_soil_fertiliser",
        "What did the soil study find about nitrogen? Could farmers use less fertiliser then?",
        "prior_question",
    ),
    ("mem_mt_cat_age", "What's my cat called? How old is she?", "prior_question"),
    ("mem_mt_supervisor_field", "Who was my master's supervisor? What field was that in?", "prior_question"),
    ("mem_mt_sister_city", "Where does Hana work?", "none"),  # the pronoun rewrite already resolved it
    # Plan S27 (V91, D64): "What's the …?" is a facet question like "What is the …?".
    # S36 searched these two as typed only because of the contraction.
    ("lib_mt_led_page_size", "pagination work in the Ledger API What's the maximum page size?", "subject"),
    ("lib_mt_parking_absent", "my flat what's the parking fee?", "subject"),
]
# Follow-ups with no reference signal: searched as typed (V76).
AS_TYPED = [
    "lib_mt_hb_sick_days",
    "lib_mt_lis_palace_entry",
    "lib_mt_tram_eastern",
    "mem_mt_api_decision",
]


def multi_turn_sets() -> list[tuple[str, list, str]]:
    """(set, history, turn) for the five other multi-turn fixture sets (S6, follow-up subjects, KQ, router, QR)."""
    from test_discourse_subject import FOLLOW_UPS, TOPIC_SWITCHES

    out = [("s6", h, t) for h, t, _s in FOLLOW_UPS] + [("s6", h, t) for h, t, _o in TOPIC_SWITCHES]
    fs = json.loads((ROOT / "eval/router_corpus/followup_subjects_v1.json").read_text(encoding="utf-8"))
    out += [("fs", c["history"], c["prompt"]) for c in fs["cases"]]
    kq = json.loads((ROOT / "eval/retrieval_corpus/v1_knowledge_quality.json").read_text(encoding="utf-8"))
    out += [("kq", q["history"], q["query"]) for q in kq["queries"] if q.get("history")]
    for name, path, key in (("router", "v1_baseline.json", "prompt"), ("qr", "query_resolution_v1.json", "prompt")):
        data = json.loads((ROOT / "eval/router_corpus" / path).read_text(encoding="utf-8"))
        cases = data if isinstance(data, list) else (data.get("cases") or data.get("queries") or [])
        out += [(name, c["history"], c.get(key) or c.get("query")) for c in cases if c.get("history")]
    return out


def synthetic_switches() -> list[dict]:
    """Each S33 single-turn case asked after a multi-turn history about other documents (V77)."""
    corpus = lme.load_corpus()

    def docs(case):
        return {lab.get("doc") or lab.get("memory") for lab in case.get("labels") or []}

    multi = [c for c in corpus["cases"] if c.get("history")]
    out = []
    for i, case in enumerate(c for c in corpus["cases"] if not c.get("history")):
        pool = [m for m in multi if m["path"] == case["path"] and not (docs(m) & docs(case))]
        if pool:
            m = pool[i % len(pool)]
            out.append(dict(case, history=m["history"]))
    return out


class TestLibraryMemoryText(unittest.TestCase):
    def test_s33_follow_ups_are_resolved(self):
        cases = _corpus_cases()
        for case_id, text, reason in RESOLVED:
            with self.subTest(case=case_id):
                _fu, _st, r = _resolve(cases[case_id])
                self.assertEqual(r.retrieval_text, text)
                self.assertEqual(r.retrieval_rewrite_reason, reason)
                self.assertNotIn("Regarding", r.retrieval_text)

    def test_follow_ups_without_a_reference_are_searched_as_typed(self):
        cases = _corpus_cases()
        for case_id in AS_TYPED:
            with self.subTest(case=case_id):
                _fu, _st, r = _resolve(cases[case_id])
                self.assertEqual(r.retrieval_text, cases[case_id]["query"])
                self.assertEqual(r.retrieval_rewrite_reason, "none")

    def test_s6_follow_ups_keep_the_subject(self):
        from test_discourse_subject import FOLLOW_UPS

        for history, turn, subject in FOLLOW_UPS:
            with self.subTest(turn=turn):
                _fu, _st, r = build_discourse_resolution(turn, history)
                self.assertIn(subject, r.retrieval_text.lower())
                self.assertNotIn("Regarding", r.retrieval_text)

    def test_topic_switches_are_never_anchored(self):
        from test_discourse_subject import TOPIC_SWITCHES

        switches = [(h, t) for h, t, _old in TOPIC_SWITCHES]
        fs = json.loads((ROOT / "eval/router_corpus/followup_subjects_v1.json").read_text(encoding="utf-8"))
        switches += [(c["history"], c["prompt"]) for c in fs["cases"] if c["category"] == "topic_switch"]
        switches += [(c["history"], c["query"]) for c in synthetic_switches()]
        self.assertEqual(len(switches), 115)
        for history, turn in switches:
            with self.subTest(turn=turn):
                _fu, _st, r = build_discourse_resolution(turn, history)
                self.assertEqual(r.retrieval_rewrite_reason, "none")
                self.assertEqual(r.retrieval_text, r.inference_text)

    def test_today_s_wrong_anchors_on_switches_are_gone(self):
        # The legacy expansion anchored these to a word from the last turn (V77).
        cases = {c["id"]: c for c in synthetic_switches()}
        for case_id in ("lib_omega_count", "lib_bud_house_saving", "lib_lis_sintra", "core_dinner"):
            with self.subTest(case=case_id):
                _fu, _st, r = _resolve(cases[case_id])
                self.assertEqual(r.retrieval_text, cases[case_id]["query"])
                # The anchor came from the "What" referent (V78); D52 removes it from routing too.
                self.assertNotIn("Regarding", r.routing_text)

    def test_a_pronoun_inside_its_own_turn_is_not_a_back_reference(self):
        for turn in (
            "How much is the deposit and how is it protected?",
            "Is my bike covered if it's stolen from the shared bike store?",
            "When is the Sintra day trip and how do we get there?",
        ):
            with self.subTest(turn=turn):
                self.assertTrue(refers_inside_turn(turn))
                self.assertEqual(context_signals(turn), ())
        for turn in ("Is there an extra deposit for it?", "And how do I reset it?", "Who decided that?"):
            with self.subTest(turn=turn):
                self.assertFalse(refers_inside_turn(turn))

    def test_context_signals(self):
        self.assertEqual(context_signals("Who decided that?"), ("back_reference",))
        self.assertEqual(context_signals("How old is she?"), ("back_reference",))
        self.assertEqual(context_signals("And on tarmac?"), ("leading_connective",))
        self.assertEqual(context_signals("Could farmers use less fertiliser then?"), ("final_then",))
        # A definite noun phrase starts most new questions too (V77).
        self.assertEqual(context_signals("Why is the sky blue?"), ())
        self.assertEqual(context_signals("What time is the palace entry?"), ())
        self.assertEqual(context_signals("Let's talk about it then"), ())  # topic change

    def test_request_frames_stay(self):
        # D54: "Tell me about Dr. Evelyn." -> "Dr. Evelyn" pulled a wrong memory (V79).
        _fu, _st, r = build_discourse_resolution("Tell me about Dr. Evelyn.", [])
        self.assertEqual(r.retrieval_text, "Tell me about Dr. Evelyn.")

    def test_discourse_disabled_searches_the_turn(self):
        cases = _corpus_cases()
        case = cases["lib_mt_heron_who_decided"]
        _fu, _st, r = build_discourse_resolution(case["query"], case["history"], discourse_enabled=False)
        self.assertEqual(r.retrieval_text, case["query"])
        self.assertEqual(r.retrieval_rewrite_reason, "none")

    def test_acronyms_defined_earlier_are_expanded(self):
        history = [_u("What is the Ledger API Rate Limiter (LARL)?"), _a("It throttles clients per key.")]
        r = build_resolved_retrieval_query(
            raw_text="How does LARL handle bursts?",
            inference_text="How does LARL handle bursts?",
            follow_up=FollowUpClassification(FollowUpKind.NONE, 0.0),
            discourse=None,
            history=history,
        )
        self.assertIn("Ledger API Rate Limiter (LARL)", r.retrieval_text)

    def test_routing_keeps_the_legacy_text(self):
        sets = multi_turn_sets() + [("s33", c["history"], c["query"]) for c in _corpus_cases().values() if c.get("history")]
        self.assertEqual(len(sets), 88)
        for name, history, turn in sets:
            with self.subTest(set=name, turn=turn):
                fu, st, r = build_discourse_resolution(turn, history)
                self.assertEqual(r.routing_text, resolve_routing_query(r.inference_text, fu, st))

    def test_telemetry_names_the_rewrite(self):
        cases = _corpus_cases()
        _fu, _st, r = _resolve(cases["lib_mt_heron_who_decided"])
        self.assertEqual(r.to_telemetry_dict()["retrieval_query_rewrite_reason"], "prior_question")
        _fu, _st, plain = build_discourse_resolution("What is my monthly rent?", [])
        self.assertNotIn("retrieval_query_rewrite_reason", plain.to_telemetry_dict())


class TestReferents(unittest.TestCase):
    """D52: two discourse fixes shared with web and routing."""

    def test_a_wh_word_before_s_is_a_contraction(self):
        for text in ("What's my cat called?", "And what's the parking fee?", "Who's my manager?", "Let's go"):
            with self.subTest(text=text):
                self.assertIsNone(extract_entity_and_aspect(text).entity)
        parse = extract_entity_and_aspect("What about Kathmandu's flora and fauna?")
        self.assertEqual((parse.entity, parse.aspect), ("Kathmandu", "flora and fauna"))
        self.assertEqual(extract_entity_and_aspect("Kathmandu's music").entity, "Kathmandu")

    def test_answer_particles_and_wh_words_are_not_referents(self):
        for word in ("Yes", "No", "Sure", "What", "And what", "Exactly"):
            with self.subTest(word=word):
                usable, reason = validate_referent_candidate(word, source="assistant_answer")
                self.assertFalse(usable)
                self.assertEqual(reason, "non_referent_word")
        self.assertTrue(validate_referent_candidate("Kathmandu", source="assistant_answer")[0])

    def test_no_more_how_old_is_what(self):
        cases = _corpus_cases()
        _fu, _st, r = _resolve(cases["mem_mt_cat_age"])
        self.assertNotIn("What?", r.inference_text)
        self.assertNotIn("What?", r.web_text)
        _fu, _st, r = _resolve(cases["lib_mt_pet_deposit"])
        self.assertEqual(r.web_text, "Is there an extra deposit for it?")
        self.assertNotIn("Yes", r.routing_text)

    def test_the_what_referent_no_longer_anchors_the_insurance_follow_up(self):
        # On the six multi-turn sets (88 turns) the fixes change web or routing
        # text only on lib_mt_pet_deposit, lib_mt_ins_theft and mem_mt_cat_age (V78).
        cases = _corpus_cases()
        _fu, _st, r = _resolve(cases["lib_mt_ins_theft"])
        self.assertEqual(r.web_text, "my contents insurance Does it cover theft from the bike store?")
        self.assertEqual(r.routing_text, "Does it cover theft from the bike store?")


class TestWorkerWiring(unittest.TestCase):
    def test_library_tool_searches_the_library_text(self):
        src = (ROOT / "workers" / "llm_worker.py").read_text(encoding="utf-8")
        block = re.search(
            r"turn_knowledge_service == SERVICE_INTERNAL_CORPUS\s+and resolved_retrieval\.retrieval_text.*?"
            r"web_semantic = resolved_retrieval\.retrieval_text.*?web_query_vector = self\.embedding_cache"
            r"\.get_embedding\(web_semantic\).*?_web_outcome = run_web_retrieval\(\s+query=web_query,",
            src,
            re.S,
        )
        self.assertIsNotNone(block, "@library must search retrieval_text with its own vector")

    def test_harness_mirrors_it(self):
        case = _corpus_cases()["lib_mt_heron_who_decided"]
        q = lme.case_query(case, lme.RUN_LIBRARY_TOOL)
        self.assertEqual(q.search_text, "When is the Heron app launching? Who decided that?")
        self.assertEqual(q.vector_text, q.search_text)


class TestS36Findings(unittest.TestCase):
    """V76, read from the committed reports."""

    def _mt(self, stage: str, mode: str, run: str) -> dict:
        report = lme.load_json(BASELINE_DIR / f"{stage}_{mode}.json")
        return report["summary"][run]["by_category"]["multi_turn"]

    def test_multi_turn_library_search_improves_on_every_embedder(self):
        for mode in MODES:
            for run in lme.LIBRARY_RUNS:
                with self.subTest(mode=mode, run=run):
                    before, after = self._mt("lm_s41", mode, run), self._mt("lm_s36", mode, run)
                    self.assertGreater(after["ndcg_at_5"], before["ndcg_at_5"] + 0.08)
                    self.assertGreater(after["relevant_recall"], before["relevant_recall"])

    def test_no_rule_7_regression(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                now = lme.load_json(BASELINE_DIR / f"lm_s36_{mode}.json")
                self.assertEqual(lme.compare_reports(now, lme.load_json(BASELINE_DIR / f"lm_s41_{mode}.json")), [])


if __name__ == "__main__":
    unittest.main()
