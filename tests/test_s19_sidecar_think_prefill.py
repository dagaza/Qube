"""Plan S19 (D9): opt-in Qwen3 empty-think pre-fill for raw-path sidecar tasks.

Part 1 checks the mechanism. Part 2 replays completions recorded from the bundled
Qwen3-1.7B (``tests/fixtures/sidecar_tasks_recorded_s19.json``) through the real
parsers and ``EnrichmentWorker._judge_contradiction``, so the evaluation numbers
in ``docs/knowledge_retrieval_followups.md`` (S19) are pinned by tests.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import core.sidecar_prompts as sp  # noqa: E402
from core.cognition_prompt_adapter import (  # noqa: E402
    CHATML_ASSISTANT_START,
    QWEN3_EMPTY_THINK,
    apply_qwen3_no_think_to_prompt,
    build_cognition_prompt,
    prefill_qwen3_empty_think,
)
from core.sidecar_types import SidecarTask  # noqa: E402

_QWEN = "/m/cognition/Qwen3-1.7B-Q6_K.gguf"
_OTHER = "/m/cognition/SmolLM2-1.7B-Instruct-Q6_K.gguf"
_JUDGE = {"old_content": "User lives in Berlin.", "new_content": "User moved to Munich last month."}
_DIGEST = {"sources_text": "[1] a.md:\nThe pump has a COP of 3.4.", "expected_ids": [1]}


def _judge_prompt(**kw) -> str:
    return sp.build_prompt_for_task(SidecarTask.contradiction_judge, **kw, **_JUDGE)


# --- Part 1: mechanism ---------------------------------------------------------


def test_helper_prefills_only_qwen3_chatml_at_the_assistant_turn() -> None:
    rendered = build_cognition_prompt("System.", "User.", "chatml", model_path=_QWEN)
    assert rendered.endswith(CHATML_ASSISTANT_START)
    assert prefill_qwen3_empty_think(rendered, "chatml", _QWEN) == rendered + QWEN3_EMPTY_THINK
    # other models, other formats, and prompts not ending at the assistant header
    other = build_cognition_prompt("System.", "User.", "chatml", model_path=_OTHER)
    assert prefill_qwen3_empty_think(other, "chatml", _OTHER) == other
    for fmt in ("llama-3", "phi", "gemma"):
        p = build_cognition_prompt("System.", "User.", fmt, model_path=_QWEN)
        assert prefill_qwen3_empty_think(p, fmt, _QWEN) == p
    assert prefill_qwen3_empty_think("free text", "chatml", _QWEN) == "free text"
    once = prefill_qwen3_empty_think(rendered, "chatml", _QWEN)
    assert prefill_qwen3_empty_think(once, "chatml", _QWEN) == once  # idempotent


def test_shipped_opt_in_set() -> None:
    """Changes only with an owner D-row per task (D9; S19: D29, D30; S20: D31-D33)."""
    assert sp.QWEN3_THINK_PREFILL_TASKS == frozenset({SidecarTask.ingest_blurb, SidecarTask.episode_summary})
    assert not sp.QWEN3_THINK_PREFILL_TASKS & {SidecarTask.contradiction_judge, SidecarTask.source_digest}


@pytest.mark.parametrize(
    ("task", "payload"),
    [
        (SidecarTask.contradiction_judge, _JUDGE),
        (SidecarTask.source_digest, _DIGEST),
        (SidecarTask.ingest_blurb, {"sample_text": "A short manual for a heat pump."}),
        (SidecarTask.episode_summary, {"conversation": "user: hi\nassistant: hello"}),
        (SidecarTask.query_rewrite, {"original_query": "how much is it?", "topic": "heat pump"}),
    ],
)
def test_tasks_outside_the_set_are_unchanged(task, payload) -> None:
    with patch.object(sp, "QWEN3_THINK_PREFILL_TASKS", frozenset()):
        p = sp.build_prompt_for_task(task, chat_format="chatml", model_path=_QWEN, **payload)
    assert p.endswith(CHATML_ASSISTANT_START)
    assert "<think>" not in p


def test_opted_in_task_ends_with_the_empty_think_block() -> None:
    with patch.object(sp, "QWEN3_THINK_PREFILL_TASKS", frozenset({SidecarTask.contradiction_judge})):
        p = _judge_prompt(chat_format="chatml", model_path=_QWEN)
        digest = sp.build_prompt_for_task(SidecarTask.source_digest, chat_format="chatml", model_path=_QWEN, **_DIGEST)
        phi = _judge_prompt(chat_format="phi", model_path=_QWEN)
        smol = _judge_prompt(chat_format="chatml", model_path=_OTHER)
    assert p.endswith(CHATML_ASSISTANT_START + QWEN3_EMPTY_THINK)
    assert "<think>" not in digest  # the opt-in is per task
    assert "<think>" not in phi and "<think>" not in smol
    # The worker's no-think pass leaves it alone: /no_think once, pre-fill still last.
    final = apply_qwen3_no_think_to_prompt(p, _QWEN)
    assert final == p
    assert final.count("/no_think") == 1


def test_title_prompt_is_never_prefilled() -> None:
    with patch.object(sp, "QWEN3_THINK_PREFILL_TASKS", frozenset(SidecarTask) - {SidecarTask.title}):
        p = sp.build_prompt_for_task(SidecarTask.title, chat_format="chatml", model_path=_QWEN,
                                     user_prompt="Heat pumps?", assistant_reply="They move heat.")
    assert "<think>" not in p


def test_slot_prompt_is_unchanged_by_the_shared_helper() -> None:
    from core.query_slots_llm import SLOT_SUBJECT_PREFIX, build_slot_prompt

    p = build_slot_prompt(history=[], prompt="What does it cost?", model_path=_QWEN)
    assert p.endswith(CHATML_ASSISTANT_START + "<think>\n\n</think>\n\n" + SLOT_SUBJECT_PREFIX)
    plain = build_slot_prompt(history=[], prompt="What does it cost?", model_path=_OTHER)
    assert plain.endswith(CHATML_ASSISTANT_START + SLOT_SUBJECT_PREFIX)


def test_worker_sends_the_prefilled_prompt() -> None:
    pytest.importorskip("PyQt6")
    from workers.sidecar_llm_worker import SidecarLlmWorker

    sent: list[str] = []

    class _FakeLlama:
        def __call__(self, prompt, **kw):
            sent.append(prompt)
            return {"choices": [{"text": "contradiction"}]}

    worker = SidecarLlmWorker.__new__(SidecarLlmWorker)
    worker.model = _FakeLlama()
    worker.active_model_path = _QWEN
    worker.active_chat_format = "chatml"
    worker._sync_telemetry_runtime = lambda **kw: None
    out: list = []
    with patch.object(sp, "QWEN3_THINK_PREFILL_TASKS", frozenset({SidecarTask.contradiction_judge})):
        SidecarLlmWorker._do_task(
            worker,
            {"task": SidecarTask.contradiction_judge, "payload": dict(_JUDGE), "out": out, "done_event": threading.Event()},
        )
    assert sent[0].endswith(QWEN3_EMPTY_THINK)
    assert out[0].ok and out[0].parsed == {"verdict": "contradiction"}


# --- Part 2: recorded completions (bundled Qwen3-1.7B, S18 threads) -------------

import json  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402

_FIXTURE = Path(__file__).parent / "fixtures" / "sidecar_tasks_recorded_s19.json"
_REC = json.loads(_FIXTURE.read_text(encoding="utf-8"))
_ARMS = ("as_shipped", "prefill", "prefill_definitions")
# EnrichmentWorker: duplicate and contradiction both delete the old row and insert
# the new content (strength +1 vs reset); complement keeps both rows.
_ACTION = {"duplicate": "replace", "contradiction": "replace", "complement": "keep_both"}


class _ReplayClient:
    def __init__(self, raw: str) -> None:
        self.raw = raw

    def complete(self, task, **kw):
        return sp.parse_task_output(task, self.raw, **kw)


def _verdict(row: dict, arm: str) -> str:
    """The verdict the production code path reaches for a recorded completion."""
    pytest.importorskip("PyQt6")
    from workers.enrichment_worker import EnrichmentWorker

    rec = row[arm] if arm == "as_shipped" else row[arm][0]
    worker = EnrichmentWorker.__new__(EnrichmentWorker)
    worker.cognition_llm = _ReplayClient(rec["raw"])
    return worker._judge_contradiction(row["a"], row["b"])


def test_as_shipped_both_tasks_return_only_think() -> None:
    """V-row S19: the raw path's "\\n\\n" stop ends every completion at "<think>"."""
    for row in _REC["judge"]:
        rec = row["as_shipped"]
        assert rec["raw"] == "<think>" and rec["final_prompt_tail"].endswith(CHATML_ASSISTANT_START)
        assert sp.parse_task_output(SidecarTask.contradiction_judge, rec["raw"]).error == "empty_output"
    for case in _REC["digest"]:
        assert case["as_shipped"]["raw"] == "<think>"
        res = sp.parse_task_output(SidecarTask.source_digest, "<think>", expected_ids=[1])
        assert (res.ok, res.error) == (False, "empty_output")
    for row in _REC["judge"]:
        for rec in row["prefill"] + row["prefill_definitions"]:
            assert rec["final_prompt_tail"].endswith(QWEN3_EMPTY_THINK)


def test_judge_gate_counts() -> None:
    reach = Counter(r["gold"] for r in _REC["judge"] if r["gate"]["reaches_judge"])
    assert dict(reach) == {"duplicate": 11, "contradiction": 6, "complement": 2}
    assert not any(r["gate"]["jaccard_fast_path"] for r in _REC["judge"])


def test_judge_label_accuracy_per_arm() -> None:
    def acc(arm: str, rows: list) -> int:
        return sum(_verdict(r, arm) == r["gold"] for r in rows)

    rows = _REC["judge"]
    reach = [r for r in rows if r["gate"]["reaches_judge"]]
    assert [acc(a, rows) for a in _ARMS] == [13, 21, 22]  # of 36
    assert [acc(a, reach) for a in _ARMS] == [12, 10, 7]  # of 19
    # Today: everything is "duplicate" except the one keyword-rule hit.
    shipped = Counter(_verdict(r, "as_shipped") for r in rows)
    assert shipped == Counter({"duplicate": 35, "contradiction": 1})


def test_judge_storage_actions_on_pairs_that_reach_it() -> None:
    reach = [r for r in _REC["judge"] if r["gate"]["reaches_judge"]]
    right = [sum(_ACTION[_verdict(r, a)] == _ACTION[r["gold"]] for r in reach) for a in _ARMS]
    assert right == [17, 11, 8]  # of 19: the working judge stores worse than today's fallback
    kept_dupes = sum(r["gold"] == "duplicate" and _verdict(r, "prefill") == "complement" for r in reach)
    assert kept_dupes == 7


def test_judge_completions_are_stable_and_fast() -> None:
    rows = _REC["judge"]
    assert sum(len({x["raw"] for x in r["prefill"]}) == 1 for r in rows) == 36
    assert sum(len({x["raw"] for x in r["prefill_definitions"]}) == 1 for r in rows) == 35
    assert max(x["ms"] for r in rows for x in r["prefill"]) < 1000


def _cited(raw: str) -> set[int]:
    return {int(i) for i in re.findall(r"\[(\d+)\]", raw)}


def test_prefilled_digest_stops_after_the_first_source() -> None:
    """The blank line between source sections hits the "\\n\\n" stop."""
    for case in _REC["digest"]:
        ids = [s["id"] for s in case["sources"]]
        for rec in case["prefill"]:
            res = sp.parse_task_output(SidecarTask.source_digest, rec["raw"], expected_ids=ids)
            assert (res.ok, res.error) == (False, "citation_ids_missing")
            assert _cited(rec["raw"]) == {1} and rec["finish_reason"] == "stop"
            assert rec["ms"] > 7 * 1500  # far beyond the foreground timeout


def test_digest_without_the_blank_line_stop() -> None:
    ok = invented = 0
    for case in _REC["digest"]:
        rec = case["prefill_no_blank_line_stop"]
        ids = [s["id"] for s in case["sources"]]
        ok += sp.parse_task_output(SidecarTask.source_digest, rec["raw"], expected_ids=ids).ok
        invented += any(i not in ids for i in _cited(rec["raw"]))
        assert rec["finish_reason"] == "length" and rec["ms"] > 25_000
        assert len(rec["raw"].strip()) > 1200  # the parser keeps only the first 1200 chars
    assert (ok, invented) == (3, 2)


def test_shipped_digest_costs_the_turn_its_timeout() -> None:
    cost = _REC["shipped_turn_cost"]
    assert cost["foreground_timeout_ms"] == 1500
    for run in cost["runs"]:
        assert run["skip_reason"] == "timeout" and not run["applied"]
        assert 1500 <= run["turn_blocked_ms"] < 1700
        assert run["sidecar_busy_ms"] > 8000  # keeps evaluating after the turn gave up


# --- Part 3: source digest default (D30) --------------------------------------


class _Store(dict):
    pass


def test_digest_is_off_by_default_and_honours_an_explicit_setting() -> None:
    import core.app_settings as app_settings

    with patch.object(app_settings, "get_sidecar_enabled", return_value=True):
        with patch.object(app_settings, "_store", return_value=_Store()):
            assert app_settings.get_sidecar_source_digest_enabled() is False
        with patch.object(app_settings, "_store", return_value=_Store({app_settings.KEY_SIDECAR_SOURCE_DIGEST: True})):
            assert app_settings.get_sidecar_source_digest_enabled() is True
    with patch.object(app_settings, "get_sidecar_enabled", return_value=False), patch.object(
        app_settings, "_store", return_value=_Store({app_settings.KEY_SIDECAR_SOURCE_DIGEST: True})
    ):
        assert app_settings.get_sidecar_source_digest_enabled() is False


def test_default_turn_does_not_wait_on_the_digest() -> None:
    import core.app_settings as app_settings
    import core.source_digest as sd

    case = _REC["digest"][0]
    context = sd._format_sources_for_sidecar(case["sources"])
    client = type("C", (), {"calls": 0, "complete": lambda self, *a, **k: setattr(self, "calls", self.calls + 1)})()
    with patch.object(app_settings, "get_sidecar_enabled", return_value=True), patch.object(
        app_settings, "_store", return_value=_Store()
    ):
        res = sd.digest_rag_context(context, case["sources"], client)
    assert (res.applied, res.skip_reason, res.text) == (False, "disabled", context)
    assert client.calls == 0


def test_opt_in_covers_builder_made_prompts() -> None:
    """S20: companion lines come from their own builders; the opt-in still applies."""
    with patch.object(sp, "QWEN3_THINK_PREFILL_TASKS", frozenset({SidecarTask.companion_line})):
        p = sp.build_prompt_for_task(SidecarTask.companion_line, chat_format="chatml", model_path=_QWEN, trigger="idle")
    assert p.endswith(CHATML_ASSISTANT_START + QWEN3_EMPTY_THINK)
    p = sp.build_prompt_for_task(SidecarTask.companion_line, chat_format="chatml", model_path=_QWEN, trigger="idle")
    assert "<think>" not in p
