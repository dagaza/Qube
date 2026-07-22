"""Tests for core/uninstall_paths.py."""

from __future__ import annotations

from pathlib import Path

from core.uninstall_paths import (
    default_app_bundle_paths,
    homebrew_zap_paths,
    support_file_paths,
    uninstall_targets,
    user_data_paths,
)


def test_default_app_bundle_paths_include_applications(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    paths = default_app_bundle_paths()
    assert Path("/Applications/Qube.app") in paths
    assert tmp_path / "Applications" / "Qube.app" in paths


def test_user_data_paths_use_qube_root(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert user_data_paths() == [tmp_path / ".qube"]


def test_support_file_paths_include_preferences(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    paths = support_file_paths()
    assert tmp_path / "Library" / "Preferences" / "com.dagaza.Qube.plist" in paths


def test_uninstall_targets_include_app_and_data(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    targets = uninstall_targets(include_user_data=True)
    assert Path("/Applications/Qube.app") in targets
    assert tmp_path / ".qube" in targets


def test_uninstall_targets_can_skip_user_data(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    targets = uninstall_targets(include_user_data=False)
    assert tmp_path / ".qube" not in targets
    assert Path("/Applications/Qube.app") in targets


def test_homebrew_zap_paths_use_tilde_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    zap = homebrew_zap_paths()
    assert "/Applications/Qube.app" in zap
    assert "~/.qube" in zap
    assert all(path.startswith("~/") or path.startswith("/") for path in zap)
