"""Tests for core/bootstrap_trace.py."""

from __future__ import annotations

import json
from unittest.mock import patch

from core.bootstrap_trace import (
    bootstrap_trace_path,
    configure_bootstrap_trace,
    record_bootstrap_trace,
    record_startup_progress,
    reset_bootstrap_trace_for_tests,
)


def test_bootstrap_trace_disabled_by_default(tmp_path, monkeypatch):
    reset_bootstrap_trace_for_tests()
    monkeypatch.setattr("core.bootstrap_trace.bootstrap_trace_path", lambda: tmp_path / "t.jsonl")
    record_bootstrap_trace("ignored")
    assert not (tmp_path / "t.jsonl").exists()


def test_configure_bootstrap_trace_from_cli_flag(tmp_path, monkeypatch):
    reset_bootstrap_trace_for_tests()
    trace_file = tmp_path / "bootstrap-trace.jsonl"
    monkeypatch.setattr("core.bootstrap_trace.bootstrap_trace_path", lambda: trace_file)
    monkeypatch.setattr(
        "core.bootstrap_trace.bootstrap_state_path",
        lambda: tmp_path / "bootstrap-state.json",
    )

    class _Args:
        bootstrap_trace = True

    configure_bootstrap_trace(_Args())
    record_bootstrap_trace("splash_presented", selected_count=3)

    lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    enabled = json.loads(lines[0])
    presented = json.loads(lines[1])
    assert enabled["event"] == "trace_enabled"
    assert presented["event"] == "splash_presented"
    assert presented["selected_count"] == 3


def test_record_startup_progress_forwards_to_winget_smoke(tmp_path, monkeypatch):
    reset_bootstrap_trace_for_tests()
    monkeypatch.setenv("QUBE_BOOTSTRAP_TRACE", "1")
    monkeypatch.setenv("QUBE_WINGET_VALIDATION", "1")
    monkeypatch.setattr("core.bootstrap_trace.bootstrap_trace_path", lambda: tmp_path / "t.jsonl")
    monkeypatch.setattr(
        "core.bootstrap_trace.bootstrap_state_path",
        lambda: tmp_path / "state.json",
    )
    configure_bootstrap_trace(None)

    with patch("core.winget_validation.record_boot_state") as winget_record:
        record_startup_progress("downloads_start", pending_count=2)

    winget_record.assert_called_once_with("downloads_start", pending_count=2)
    assert (tmp_path / "t.jsonl").is_file()
