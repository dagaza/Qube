"""Tests for generation debug capture helpers."""
from __future__ import annotations

import json
import os
import tempfile

from core.generation_debug_capture import (
    GenerationDebugRecorder,
    analyze_corruption_origin,
    apply_debug_stop_mode,
    build_diagnostic_summary,
)
from core.completion_output_trace import CompletionOutputSnapshot


def test_apply_debug_stop_mode_minimal(monkeypatch):
    monkeypatch.setenv("QUBE_GENERATION_DEBUG_STOP_MODE", "minimal")
    stops = apply_debug_stop_mode(
        ["<|return|>", "\nWe need to", "extra"],
        eos_token="</s>",
    )
    assert "<|return|>" in stops
    assert "</s>" in stops
    assert "\nWe need to" not in stops


def test_corruption_origin_detects_raw_generation():
    raw = "1. Item one\n2.\n3. **broken"
    clean = "1. Item one\n2. Item two"
    origin = analyze_corruption_origin(
        raw=raw,
        after_harmony=raw,
        after_filters=clean,
        worker_return=clean,
        stored=clean,
    )
    assert origin["likely_stage"] == "raw_generation"
    assert origin["likely_cause"] == "sampling_or_model_generation"


def test_recorder_writes_artifacts(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("QUBE_GENERATION_DEBUG", "1")
        monkeypatch.setenv("QUBE_GENERATION_DEBUG_DIR", tmp)
        rec = GenerationDebugRecorder.maybe_start(
            turn_id=5,
            session_id="sess-1",
            user_query="List UNESCO sites",
            gen_params={"temperature": 0.7, "max_tokens": 512},
            merged_stops=["<|return|>"],
        )
        assert rec is not None
        rec.record_delta(
            delta="Hello",
            cumulative_raw="Hello",
            cumulative_filtered="Hello",
        )
        snap = CompletionOutputSnapshot(
            engine_mode="internal",
            raw_text="Hello world",
            after_harmony_parser="Hello world",
            after_worker_filters="Hello world",
            worker_return_text="Hello world",
        )
        rec.finalize_stream(snapshot=snap, merged_stops=["<|return|>"])
        rec.record_final_stored("Hello world", ui_final="Hello world")

        assert os.path.isfile(os.path.join(tmp, "turn5_raw_stream.txt"))
        assert os.path.isfile(os.path.join(tmp, "turn5_postprocess.txt"))
        assert os.path.isfile(os.path.join(tmp, "turn5_final.txt"))
        assert os.path.isfile(os.path.join(tmp, "turn5_meta.json"))
        assert os.path.isfile(os.path.join(tmp, "turn5_trace_analysis.json"))

        with open(os.path.join(tmp, "turn5_meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["turn_id"] == 5
        assert meta["temperature"] == 0.7


def test_build_diagnostic_summary(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("QUBE_GENERATION_DEBUG", "1")
        monkeypatch.setenv("QUBE_GENERATION_DEBUG_DIR", tmp)
        rec = GenerationDebugRecorder.maybe_start(
            turn_id=6,
            user_query="test",
            gen_params={"temperature": 0.3},
        )
        assert rec is not None
        snap = CompletionOutputSnapshot(
            engine_mode="internal",
            raw_text="bad <|channel|> soup",
            worker_return_text="bad <|channel|> soup",
        )
        rec.finalize_stream(snapshot=snap)
        rec.record_final_stored("bad <|channel|> soup")

        summary = build_diagnostic_summary(tmp)
        assert summary["turn_count"] == 1
        assert summary["first_collapse_turn"] == 6
        assert os.path.isfile(os.path.join(tmp, "diagnostic_summary.json"))
