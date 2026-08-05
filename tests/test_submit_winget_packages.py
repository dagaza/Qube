"""Tests for scripts/release/submit_winget_packages.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "release" / "submit_winget_packages.py"
_spec = importlib.util.spec_from_file_location("submit_winget_packages", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)


def test_submit_winget_packages_submits_all_rendered_manifests(tmp_path):
    wingetcreate = tmp_path / "wingetcreate.exe"
    wingetcreate.write_text("", encoding="utf-8")
    manifest_root = tmp_path / "manifests"
    for package_id in ("dagaza.Qube", "dagaza.Qube.Vulkan", "dagaza.Qube.CUDA"):
        (manifest_root / package_id).mkdir(parents=True)

    calls: list[list[str]] = []

    def _run(command, check=True):
        calls.append(command)
        return mock.Mock(returncode=0)

    with mock.patch("subprocess.run", side_effect=_run):
        _mod.submit_winget_packages(
            version="1.3.0",
            token="token",
            wingetcreate=wingetcreate,
            manifest_root=manifest_root,
        )

    assert len(calls) == 3
    assert all(call[1] == "submit" for call in calls)
    assert calls[0][2].endswith("dagaza.Qube")
    assert calls[1][2].endswith("dagaza.Qube.Vulkan")
    assert calls[2][2].endswith("dagaza.Qube.CUDA")


def test_submit_winget_packages_requires_manifest_dir(tmp_path):
    wingetcreate = tmp_path / "wingetcreate.exe"
    wingetcreate.write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Rendered manifest folder missing"):
        _mod.submit_winget_packages(
            version="1.3.0",
            token="token",
            wingetcreate=wingetcreate,
            manifest_root=tmp_path / "empty",
        )
