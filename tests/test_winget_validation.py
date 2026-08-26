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
    is_winget_validation_mode,
    reset_winget_validation_state_for_tests,
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
    assert payload["llama_import_attempted"] is False
