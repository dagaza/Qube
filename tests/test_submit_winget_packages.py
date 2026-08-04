"""Tests for scripts/release/submit_winget_packages.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "release" / "submit_winget_packages.py"
_spec = importlib.util.spec_from_file_location("submit_winget_packages", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)


def test_package_manifest_path():
    assert _mod._package_manifest_path("dagaza.Qube") == "manifests/d/dagaza/Qube"
    assert _mod._package_manifest_path("dagaza.Qube.Vulkan") == "manifests/d/dagaza/Qube.Vulkan"
    assert _mod._package_manifest_path("dagaza.Qube.CUDA") == "manifests/d/dagaza/Qube.CUDA"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (404, b"{}", False),
        (200, json.dumps([{"name": "1.0.0"}]).encode(), True),
    ],
)
def test_package_exists_in_winget_pkgs(status, body, expected):
    response = mock.Mock()
    response.read.return_value = body
    response.__enter__ = mock.Mock(return_value=response)
    response.__exit__ = mock.Mock(return_value=False)

    def _urlopen(request, timeout=30):
        if status == 404:
            import urllib.error

            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        return response

    with mock.patch("urllib.request.urlopen", side_effect=_urlopen):
        assert _mod.package_exists_in_winget_pkgs("dagaza.Qube") is expected


def test_submit_winget_packages_update_vs_submit(tmp_path):
    wingetcreate = tmp_path / "wingetcreate.exe"
    wingetcreate.write_text("", encoding="utf-8")
    manifest_root = tmp_path / "manifests"
    (manifest_root / "dagaza.Qube.Vulkan").mkdir(parents=True)
    (manifest_root / "dagaza.Qube.CUDA").mkdir(parents=True)

    calls: list[list[str]] = []

    def _run(command, check=True):
        calls.append(command)
        return mock.Mock(returncode=0)

    exists = {"dagaza.Qube": True, "dagaza.Qube.Vulkan": False, "dagaza.Qube.CUDA": False}

    with (
        mock.patch.object(_mod, "package_exists_in_winget_pkgs", side_effect=lambda pid: exists[pid]),
        mock.patch("subprocess.run", side_effect=_run),
    ):
        _mod.submit_winget_packages(
            version="1.2.9",
            token="token",
            wingetcreate=wingetcreate,
            manifest_root=manifest_root,
        )

    assert len(calls) == 3
    assert calls[0][:3] == [str(wingetcreate), "update", "dagaza.Qube"]
    assert calls[1][:3] == [str(wingetcreate), "submit", str(manifest_root / "dagaza.Qube.Vulkan")]
    assert calls[2][:3] == [str(wingetcreate), "submit", str(manifest_root / "dagaza.Qube.CUDA")]
