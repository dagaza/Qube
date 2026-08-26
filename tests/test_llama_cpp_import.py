"""Tests for core/llama_cpp_import.py."""

from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path
from unittest.mock import patch

from core import llama_cpp_import as mod


def test_prepare_llama_cpp_runtime_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    mod.reset_llama_import_state_for_tests()
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    monkeypatch.setattr(mod, "llama_cpp_lib_dir", lambda: lib_dir)

    mod.prepare_llama_cpp_runtime()
    first_path = os.environ.get("PATH", "")
    mod.prepare_llama_cpp_runtime()
    assert os.environ.get("PATH", "") == first_path


def test_llama_cpp_lib_dir_uses_internal_path_when_frozen(monkeypatch, tmp_path: Path) -> None:
    mod.reset_llama_import_state_for_tests()
    exe = tmp_path / "Qube.exe"
    exe.write_text("", encoding="utf-8")
    lib_dir = tmp_path / "_internal" / "llama_cpp" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "llama.dll").write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe), raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert mod.llama_cpp_lib_dir() == lib_dir


def test_get_llama_class_caches_failure(monkeypatch) -> None:
    mod.reset_llama_import_state_for_tests()
    monkeypatch.setattr(mod, "prepare_llama_cpp_runtime", lambda: None)

    real_import = builtins.__import__
    calls = {"count": 0}

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "llama_cpp":
            calls["count"] += 1
            raise OSError("dll load failed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    first = mod.get_llama_class()
    second = mod.get_llama_class()
    assert first is None
    assert second is None
    assert calls["count"] == 1
    assert isinstance(mod.llama_import_error(), OSError)


def test_worker_modules_do_not_eager_load_llama() -> None:
    """Startup must not import llama_cpp until a model load is requested."""
    repo = Path(__file__).resolve().parents[1]
    for rel in (
        "workers/sidecar_llm_worker.py",
        "workers/native_llama_engine.py",
        "main.py",
    ):
        for line in (repo / rel).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped == "Llama = get_llama_class()":
                assert line[:1] in {" ", "\t"}, (
                    f"{rel} must not assign Llama at module scope"
                )
            if rel == "main.py" and stripped == "prepare_llama_cpp_runtime()":
                assert line[:1] in {" ", "\t"}, (
                    f"{rel} must not call prepare_llama_cpp_runtime() at module scope"
                )


def test_sidecar_defers_cognition_load_until_first_command() -> None:
    text = (Path(__file__).resolve().parents[1] / "workers/sidecar_llm_worker.py").read_text(
        encoding="utf-8"
    )
    assert "_try_load_cognition_model_if_needed" in text
    run_section = text.split("def run(self)", 1)[1].split("\n    def _run_degraded", 1)[0]
    assert "_load_cognition_model(path)" not in run_section


def test_merge_native_telemetry_skips_build_probe_before_model_load() -> None:
    from core.inference_transparency import merge_native_telemetry_snapshot

    with patch("core.inference_transparency.get_build_snapshot") as build, patch(
        "core.inference_transparency._static_build_snapshot",
        return_value={"backend_hint": "cuda", "probe_deferred": True},
    ), patch(
        "core.inference_transparency.get_hardware_profile_snapshot",
        return_value={},
    ), patch(
        "core.inference_transparency.get_settings_snapshot",
        return_value={},
    ):
        merge_native_telemetry_snapshot(None)
    build.assert_not_called()
