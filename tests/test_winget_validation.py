"""Tests for WinGet validation guard (core/winget_validation.py)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from core import llama_cpp_import as llama_mod
from core.winget_validation import (
    boot_state_path,
    boot_trace_path,
    is_winget_smoke_validation,
    is_winget_validation_mode,
    record_boot_state,
    reset_winget_validation_state_for_tests,
    smoke_result_path,
    write_smoke_failure,
    write_smoke_result,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_winget_validation_state_for_tests()
    llama_mod.reset_llama_import_state_for_tests()
    monkeypatch.delenv("QUBE_WINDOWS_VARIANT", raising=False)


def test_explicit_env_enables_validation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUBE_WINGET_VALIDATION", "1")
    assert is_winget_validation_mode() is True


def test_explicit_env_can_disable_install_grace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QUBE_WINGET_VALIDATION", "0")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Qube.exe"), raising=False)
    (tmp_path / ".qube-windows-variant").write_text("cuda", encoding="utf-8")
    (tmp_path / ".qube-install-ts").write_text("1", encoding="utf-8")
    assert is_winget_validation_mode() is False


def test_cuda_install_grace_enables_validation_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exe = tmp_path / "Qube.exe"
    exe.write_text("", encoding="utf-8")
    (tmp_path / ".qube-windows-variant").write_text("cuda", encoding="utf-8")
    marker = tmp_path / ".qube-install-ts"
    marker.write_text("1", encoding="utf-8")
    now = time.time()
    os.utime(marker, (now, now))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe), raising=False)
    assert is_winget_validation_mode() is True


def test_get_llama_class_skipped_without_import_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUBE_WINGET_VALIDATION", "1")
    assert llama_mod.get_llama_class() is None
    assert llama_mod.llama_import_was_attempted() is False


def test_merge_native_telemetry_skips_hardware_probe_in_validation_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUBE_WINGET_VALIDATION", "1")
    from core.inference_transparency import merge_native_telemetry_snapshot

    with patch("core.inference_transparency.get_hardware_profile_snapshot") as hardware:
        snap = merge_native_telemetry_snapshot(None)
    hardware.assert_not_called()
    assert snap["hardware"]["gpu_memory_kind"] == "none"


def test_write_smoke_result_records_no_llama_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QUBE_WINGET_VALIDATION", "1")
    with patch("core.paths.user_data_root", return_value=tmp_path):
        write_smoke_result(boot_complete=True)
    payload = json.loads((tmp_path / ".winget-validation-smoke.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["stage"] == "boot_complete"
    assert payload["llama_import_attempted"] is False
    boot_state = json.loads((tmp_path / ".winget-validation-boot-state.json").read_text(encoding="utf-8"))
    assert boot_state["state"] == "boot_complete"


def test_write_smoke_failure_records_stage_and_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QUBE_WINGET_VALIDATION", "1")
    with patch("core.paths.user_data_root", return_value=tmp_path):
        write_smoke_failure(stage="phase_2", error="boom")
    payload = json.loads((tmp_path / ".winget-validation-smoke.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["stage"] == "phase_2"
    assert payload["error"] == "boom"
    boot_state = json.loads((tmp_path / ".winget-validation-boot-state.json").read_text(encoding="utf-8"))
    assert boot_state["state"] == "boot_failed"


def test_record_boot_state_skipped_outside_validation_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with patch("core.paths.user_data_root", return_value=tmp_path):
        record_boot_state("phase_start", phase=0)
    assert not (tmp_path / ".winget-validation-boot-state.json").exists()


def test_record_boot_state_writes_when_validation_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QUBE_WINGET_VALIDATION", "1")
    with patch("core.paths.user_data_root", return_value=tmp_path):
        record_boot_state("phase_start", phase=1)
        assert smoke_result_path().parent == tmp_path
        assert boot_state_path().parent == tmp_path
        assert boot_trace_path().parent == tmp_path
    payload = json.loads((tmp_path / ".winget-validation-boot-state.json").read_text(encoding="utf-8"))
    assert payload["state"] == "phase_start"
    assert payload["phase"] == 1
    trace_lines = (
        (tmp_path / ".winget-validation-boot-trace.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .split("\n")
    )
    assert len(trace_lines) == 1
    assert json.loads(trace_lines[0])["state"] == "phase_start"


def test_record_boot_state_appends_boot_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QUBE_WINGET_VALIDATION", "1")
    with patch("core.paths.user_data_root", return_value=tmp_path):
        record_boot_state("phase_start", phase=0)
        record_boot_state("phase_complete", phase=0)
    trace_lines = (
        (tmp_path / ".winget-validation-boot-trace.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .split("\n")
    )
    assert len(trace_lines) == 2
    assert json.loads(trace_lines[0])["state"] == "phase_start"
    assert json.loads(trace_lines[1])["state"] == "phase_complete"
    last_state = json.loads(
        (tmp_path / ".winget-validation-boot-state.json").read_text(encoding="utf-8")
    )
    assert last_state["state"] == "phase_complete"


def test_cuda_install_grace_defers_cuda_backend_but_keeps_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("PyQt6")
    exe = tmp_path / "Qube.exe"
    exe.write_text("", encoding="utf-8")
    (tmp_path / ".qube-windows-variant").write_text("cuda", encoding="utf-8")
    marker = tmp_path / ".qube-install-ts"
    marker.write_text("1", encoding="utf-8")
    now = time.time()
    os.utime(marker, (now, now))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe), raising=False)

    from core.bootstrap_selection import (
        effective_bootstrap_selection,
        is_bootstrap_completed,
        should_show_bootstrap_consent,
    )
    from core.winget_validation import apply_winget_validation_bootstrap_shortcut

    assert is_winget_validation_mode() is True
    assert is_winget_smoke_validation() is False
    assert apply_winget_validation_bootstrap_shortcut() is False
    assert is_bootstrap_completed() is False
    assert should_show_bootstrap_consent() is True
    assert effective_bootstrap_selection() != set()


def test_validation_mode_skips_bootstrap_consent_and_default_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("PyQt6")
    monkeypatch.setenv("QUBE_WINGET_VALIDATION", "1")
    from core.bootstrap_selection import (
        effective_bootstrap_selection,
        should_show_bootstrap_consent,
    )

    assert should_show_bootstrap_consent() is False
    assert effective_bootstrap_selection() == set()


def test_boot_storage_skips_embedder_in_validation_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("PyQt6")
    monkeypatch.setenv("QUBE_WINGET_VALIDATION", "1")
    embedder_calls: list[int] = []

    class _FakeEmbedder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            embedder_calls.append(1)

    class _FakeStore:
        dim_mismatch = False

        def __init__(self, **kwargs: object) -> None:
            pass

    class _FakeDb:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr("rag.embedder.EmbeddingModel", _FakeEmbedder)
    monkeypatch.setattr("rag.store.DocumentStore", _FakeStore)
    monkeypatch.setattr("core.database.DatabaseManager", _FakeDb)

    from main import Qube

    qube = Qube.__new__(Qube)
    qube._boot_storage(lambda _msg: None, None)
    assert embedder_calls == []
    assert qube.embedder is None


def test_maybe_reset_stale_shell_bootstrap_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("PyQt6")
    import tempfile

    from core.bootstrap_selection import (
        KEY_COMPLETED,
        KEY_SELECTED,
        maybe_reset_stale_shell_bootstrap_completion,
        should_show_bootstrap_consent,
    )
    from core.settings_store import SettingsStore

    schema_path = (
        Path(__file__).resolve().parent.parent / "assets" / "config" / "settings.schema.json"
    )
    with tempfile.TemporaryDirectory() as tmp:
        store = SettingsStore(user_path=Path(tmp) / "settings.json", schema_path=schema_path)
        store.set(KEY_COMPLETED, True)
        store.set(KEY_SELECTED, "[]")
        monkeypatch.setattr(
            "core.bootstrap_selection.get_settings_store",
            lambda: store,
        )
        monkeypatch.setattr(
            "core.bootstrap_download.infer_installed_selection",
            lambda: None,
        )
        monkeypatch.setattr(
            "core.bootstrap_download.model_is_present",
            lambda _mid: False,
        )

        assert maybe_reset_stale_shell_bootstrap_completion() is True
        assert store.get(KEY_COMPLETED) is not True
        assert should_show_bootstrap_consent() is True
