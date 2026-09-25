"""Tests for the fail-closed license gate.

Stdlib + pytest only (no training environment needed) so this can run in CI as the
required provenance check for the wakeword/ project.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lib import licenses  # noqa: E402


def _write_manifest(root: Path, name: str, **overrides) -> Path:
    payload = {
        "asset": f"background-noise/{name}.wav",
        "dataset": "MUSAN",
        "source_url": "https://www.openslr.org/17/",
        "license": "CC-BY-4.0",
        "commercial_use": True,
        "retrieved": "2026-06-16",
    }
    payload.update(overrides)
    path = root / f"{name}{licenses.MANIFEST_SUFFIX}"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_commercial_manifest_passes(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "ok")
    result = licenses.run_gate(tmp_path, require_commercial=True)
    assert result.ok
    assert result.checked == 1
    assert not result.errors


def test_non_commercial_license_fails(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "acav", license="CC-BY-NC-SA-4.0", commercial_use=False)
    result = licenses.run_gate(tmp_path, require_commercial=True)
    assert not result.ok
    assert any("allowlist" in e for e in result.errors)


def test_commercial_flag_false_fails_even_if_license_ok(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "weird", license="CC-BY-4.0", commercial_use=False)
    result = licenses.run_gate(tmp_path, require_commercial=True)
    assert not result.ok
    assert any("commercial_use=false" in e for e in result.errors)


def test_missing_required_field_fails(tmp_path: Path) -> None:
    path = tmp_path / f"bad{licenses.MANIFEST_SUFFIX}"
    path.write_text(json.dumps({"asset": "x.wav"}), encoding="utf-8")
    result = licenses.run_gate(tmp_path, require_commercial=True)
    assert not result.ok
    assert any("missing required field" in e for e in result.errors)


def test_share_alike_warns_but_does_not_fail(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "sa", license="CC-BY-SA-4.0")
    result = licenses.run_gate(tmp_path, require_commercial=True)
    assert result.ok
    assert any("share-alike" in w for w in result.warnings)


def test_example_manifests_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / f"MUSAN.example{licenses.MANIFEST_SUFFIX}"
    path.write_text(json.dumps({"asset": "x"}), encoding="utf-8")
    result = licenses.run_gate(tmp_path, require_commercial=True)
    assert result.checked == 0


def test_presence_only_mode_skips_commercial_checks(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "nc", license="CC-BY-NC-4.0", commercial_use=False)
    result = licenses.run_gate(tmp_path, require_commercial=False)
    assert result.ok


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
