"""Tests for scripts/prepare_release.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prepare_release():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_release.py"
    spec = importlib.util.spec_from_file_location("prepare_release", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_version_accepts_v_prefix():
    mod = _load_prepare_release()
    assert mod._normalize_version("v1.2.3") == "1.2.3"


def test_changelog_issues_when_section_missing(monkeypatch):
    mod = _load_prepare_release()
    root = Path(__file__).resolve().parents[1] / ".pytest_tmp" / "prepare_release_test"
    root.mkdir(parents=True, exist_ok=True)
    (root / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n### Added\n- item\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_repo_root", lambda: root)
    issues = mod._changelog_issues("1.0.1")
    assert any("1.0.1" in i for i in issues)
