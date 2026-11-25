"""Plan S20 (D31-D33): the remaining raw-path sidecar tasks.

Replays completions recorded from the bundled Qwen3-1.7B
(``tests/fixtures/sidecar_tasks_recorded_s20.json``) through the production
parsers, the reflection worker's own ``_call_llm`` / ``_parse_label``, and the
episode path of ``EnrichmentWorker``, and checks the two shipped changes: the
joined ingest-blurb sample and the small-talk guard for episodes.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import core.sidecar_prompts as sp  # noqa: E402
from core.cognition_prompt_adapter import CHATML_ASSISTANT_START, QWEN3_EMPTY_THINK  # noqa: E402
from core.sidecar_types import SidecarTask  # noqa: E402

_REC = json.loads((Path(__file__).parent / "fixtures" / "sidecar_tasks_recorded_s20.json").read_text(encoding="utf-8"))
_QWEN = "/m/cognition/Qwen3-1.7B-Q6_K.gguf"
_STOP = set("this that with from document outlines provides explains various their about including into using".split())


def _ground(blurb: str, sample: str) -> float:
    words = {w for w in re.findall(r"[a-z]{4,}", blurb.lower()) if w not in _STOP}
    source = set(re.findall(r"[a-z]{4,}", sample.lower()))
    return len(words & source) / len(words) if words else 0.0


class _Replay:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.calls = 0

    def isRunning(self) -> bool:
        return False

    def complete(self, task, **kw):
        self.calls += 1
        return sp.parse_task_output(task, self.raw, **kw)


# --- shipped opt-in ------------------------------------------------------------


def test_prefill_set_after_s20() -> None:
    assert sp.QWEN3_THINK_PREFILL_TASKS == frozenset({SidecarTask.ingest_blurb, SidecarTask.episode_summary})
    for task, kw in [
        (SidecarTask.ingest_blurb, {"sample_text": "A manual for a heat pump."}),
        (SidecarTask.episode_summary, {"conversation": "user: hi\nassistant: hello"}),
    ]:
        assert sp.build_prompt_for_task(task, chat_format="chatml", model_path=_QWEN, **kw).endswith(
            CHATML_ASSISTANT_START + QWEN3_EMPTY_THINK
        )
    # Not pre-filled: companion lines (D33), and the reflection worker's flat prompt.
    assert "<think>" not in sp.build_prompt_for_task(
        SidecarTask.companion_line, chat_format="chatml", model_path=_QWEN, trigger="idle"
    )
    flat = sp.build_prompt_for_task(SidecarTask.reflection_label, chat_format="chatml", model_path=_QWEN, prompt="Audit.")
    assert "<think>" not in flat and "<|im_start|>" not in flat


def test_recorded_tails_match_the_arms() -> None:
    for section, arms in (("episode", ("prefill",)), ("blurb", ("prefill",)), ("companion", ("prefill",))):
        for row in _REC[section]:
            assert row["as_shipped"]["final_prompt_tail"].endswith(CHATML_ASSISTANT_START)
            for arm in arms:
                recs = row[arm] if isinstance(row[arm], list) else [row[arm]]
                assert all(r["final_prompt_tail"].endswith(QWEN3_EMPTY_THINK) for r in recs)


# --- reflection_label (V49: works as shipped) -----------------------------------


def _reflection_label(raw: str) -> str:
    pytest.importorskip("PyQt6")
    from workers.memory_reflection_worker import MemoryReflectionWorker

    worker = MemoryReflectionWorker.__new__(MemoryReflectionWorker)
    worker.llm = _Replay(raw)
    return worker._parse_label(worker._call_llm("unused"))


def test_reflection_works_as_shipped() -> None:
    rows = _REC["reflection"]
    for arm in ("as_shipped", "chatml_prefill"):
        labels = [_reflection_label(r[arm]["raw"]) for r in rows]
        assert sum(lab == r["gold"] for lab, r in zip(labels, rows)) == 16
        # Flag decisions (anything but durable_user_fact is flagged) are all right.
        assert all((lab != "durable_user_fact") == (r["gold"] != "durable_user_fact") for lab, r in zip(labels, rows))
        assert Counter((r["gold"], lab) for lab, r in zip(labels, rows) if lab != r["gold"]) == {("unclear", "transient"): 4}
    assert not any(r["as_shipped"]["raw"].strip().startswith("<think>") for r in rows)


# --- episode_summary (V51, D32) -------------------------------------------------


class _DB:
    def __init__(self, messages):
        self.messages = messages

    def get_session_history(self, _sid):
        return list(self.messages)


class _Table:
    def __init__(self, store):
        self.store = store

    def add(self, records):
        self.store.rows.extend(dict(r) for r in records)

    def delete(self, where):
        pass

    def search(self, *a, **k):
        return self

    def where(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def select(self, *a, **k):
        return self

    def to_list(self):
        return []


class _Store:
    def __init__(self):
        self.rows: list[dict] = []

    @property
    def table(self):
        return _Table(self)


class _Embedder:
    def embed_query(self, _text):
        return [0.1, 0.2, 0.3]


def _messages(conversation: str) -> list[dict]:
    out = []
    for i, line in enumerate(conversation.splitlines()):
        role, _, content = line.partition(": ")
        out.append({"id": f"m{i}", "role": role, "content": content})
    return out


def _summarise(row: dict, raw: str) -> tuple[int, int]:
    """(episode rows written, LLM calls) for one recorded conversation."""
    pytest.importorskip("PyQt6")
    from workers.enrichment_worker import EnrichmentWorker

    llm = _Replay(raw)
    worker = EnrichmentWorker(llm=llm, cognition_llm=llm, embedder=_Embedder(), store=_Store(), db=_DB(_messages(row["conversation"])))
    worker._summarise_session_now(f"sess-{row['id']}", reason="idle")
    return len(worker.store.rows), llm.calls


def test_episode_outcomes_with_the_small_talk_guard() -> None:
    from workers.enrichment_worker import EPISODE_SUMMARY_MIN_USER_WORDS

    assert EPISODE_SUMMARY_MIN_USER_WORDS == 20
    for row in _REC["episode"]:
        assert _summarise(row, row["as_shipped"]["raw"])[0] == 0  # "<think>": nothing, as before
        written, calls = _summarise(row, row["prefill"][0]["raw"])
        if row["gold"] == "summary":
            assert (written, calls) == (1, 1), row["id"]
        else:
            assert (written, calls) == (0, 0), row["id"]  # skipped before the sidecar call


def test_without_the_guard_small_talk_would_be_stored() -> None:
    """The recorded completions never say SKIP; the guard is what keeps them out."""
    for row in _REC["episode"]:
        res = sp.parse_task_output(SidecarTask.episode_summary, row["prefill"][0]["raw"])
        summary = str((res.parsed or {}).get("summary") or "")
        assert res.ok and summary and summary.upper() != "SKIP", row["id"]
        if row["gold"] == "summary":
            low = summary.lower()
            assert sum(t in low for t in row["key_terms"]) >= len(row["key_terms"]) - 1
            assert 20 <= len(summary.split()) <= 120


# --- ingest_blurb (V50, D31) ----------------------------------------------------


def test_blurbs_as_shipped_and_prefilled() -> None:
    rows = _REC["blurb"]
    assert len(rows) == 19
    for row in rows:
        assert sp.parse_task_output(SidecarTask.ingest_blurb, row["as_shipped"]["raw"]).error == "empty_output"
        assert sp.parse_task_output(SidecarTask.ingest_blurb, row["prefill"]["raw"]).ok
    short = [r for r in rows if len(r["sample_text"]) < 100]
    longer = [r for r in rows if len(r["sample_text"]) >= 100]
    assert len(short) == 6 and len(longer) == 13
    # Heading-only first chunks: the blurb is invented; with the joined sample it is grounded.
    assert max(_ground(r["prefill"]["raw"], r["sample_text"]) for r in short) < 0.3
    assert min(_ground(r["prefill_joined"]["raw"], r["joined_sample_text"]) for r in short) >= 0.6
    assert min(_ground(r["prefill"]["raw"], r["sample_text"]) for r in longer) >= 0.4
    assert all(r["prefill_joined"]["finish_reason"] == "stop" for r in short)


def test_blurb_sample_joins_chunks_up_to_the_limit() -> None:
    assert sp.build_ingest_blurb_sample([]) == ""
    assert sp.build_ingest_blurb_sample(["## Prerequisites", "", "  - A maintainer account.  "]) == (
        "## Prerequisites\n\n- A maintainer account."
    )
    long = sp.build_ingest_blurb_sample(["x" * 1500, "y" * 1500, "z" * 1500])
    assert len(long) == sp.INGEST_BLURB_SAMPLE_CHARS and "z" not in long
    assert sp.build_ingest_blurb_sample(["a" * 3000]) == "a" * 2500
    # The recorded joined samples are exactly what the helper builds today.
    for row in _REC["blurb"]:
        if "joined_sample_text" in row:
            assert len(row["joined_sample_text"]) == sp.INGEST_BLURB_SAMPLE_CHARS


def test_ingestion_worker_uses_the_joined_sample() -> None:
    source = (Path(_ROOT) / "workers" / "ingestion_worker.py").read_text(encoding="utf-8")
    assert "sample = build_ingest_blurb_sample(chunks)" in source
    assert "chunks[0][:2500]" not in source


# --- companion_line (V52, D33) --------------------------------------------------


def test_companion_copies_its_prompt_example() -> None:
    from core.companion_verbal_prompts import build_companion_line_prompt

    example = '{"line":"Still here if you need me.","kind":"idle_quip"}'
    assert example in build_companion_line_prompt(model_path=_QWEN, trigger="ingest_complete")
    lines = []
    for row in _REC["companion"]:
        assert sp.parse_task_output(SidecarTask.companion_line, row["as_shipped"]["raw"], **row["payload"]).error == "empty_output"
        for rec in row["prefill"]:
            res = sp.parse_task_output(SidecarTask.companion_line, rec["raw"], **row["payload"])
            assert res.ok
            lines.append(res.text)
    assert len(lines) == 18 and set(lines) == {"Still here if you need me."}


def test_raw_prompt_path_has_no_production_callers() -> None:
    """V53: SidecarLlmClient.generate is unused outside tests."""
    hits = []
    for folder in ("core", "workers", "ui", "mcp"):
        for path in (Path(_ROOT) / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"(sidecar\w*|cognition_llm)\.generate\(", text):
                hits.append(str(path))
    assert hits == []
