"""Tests for scripts/stage_vulkan_runtime_libs.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_stage():
    path = Path(__file__).resolve().parents[1] / "scripts" / "stage_vulkan_runtime_libs.py"
    spec = importlib.util.spec_from_file_location("stage_vulkan_runtime_libs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage_copies_vulkan_loader(tmp_path: Path, monkeypatch) -> None:
    mod = _load_stage()
    loader = tmp_path / "vulkan-1.dll"
    loader.write_bytes(b"loader")

    monkeypatch.setattr(mod, "find_vulkan_loader", lambda: loader)

    dest = tmp_path / "bundle" / "lib"
    assert mod.stage(dest) == 0
    assert (dest / "vulkan-1.dll").read_bytes() == b"loader"
