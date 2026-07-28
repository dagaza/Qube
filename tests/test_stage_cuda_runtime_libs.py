"""Tests for scripts/stage_cuda_runtime_libs.py."""

from __future__ import annotations

from pathlib import Path


def test_stage_cuda_script_resolves_repo_root():
    script = Path(__file__).resolve().parents[1] / "scripts" / "stage_cuda_runtime_libs.py"
    repo_root = script.resolve().parents[1]
    assert (repo_root / "core" / "nvidia_wheel_lib_dirs.py").is_file()
