"""Tests for core/llama_cpp_import.py."""

from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path

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
