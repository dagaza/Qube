"""Tests for core/uninstall_paths.py."""

from __future__ import annotations

import sys
from pathlib import Path

from core.uninstall_paths import (
    deb_runtime_dependencies,
    default_app_bundle_paths,
    homebrew_zap_paths,
    linux_app_paths,
    support_file_paths,
    uninstall_targets,
    user_data_paths,
)


def test_default_app_bundle_paths_include_applications(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    paths = default_app_bundle_paths()
    assert Path("/Applications/Qube.app") in paths
    assert tmp_path / "Applications" / "Qube.app" in paths


def test_linux_app_paths_use_opt_qube():
    assert linux_app_paths() == [Path("/opt/qube")]


def test_user_data_paths_use_qube_root(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert user_data_paths() == [tmp_path / ".qube"]


def test_support_file_paths_include_preferences_on_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    paths = support_file_paths()
    assert tmp_path / "Library" / "Preferences" / "com.dagaza.Qube.plist" in paths


def test_support_file_paths_include_desktop_entries_on_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    paths = support_file_paths()
    assert Path("/usr/share/applications/qube.desktop") in paths
    assert tmp_path / ".local/share/applications/qube.desktop" in paths


def test_uninstall_targets_include_app_and_data_on_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    targets = uninstall_targets(include_user_data=True)
    assert Path("/Applications/Qube.app") in targets
    assert tmp_path / ".qube" in targets


def test_uninstall_targets_include_opt_qube_on_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    targets = uninstall_targets(include_user_data=True)
    assert Path("/opt/qube") in targets
    assert tmp_path / ".qube" in targets


def test_uninstall_targets_can_skip_user_data(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
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


def test_deb_runtime_dependencies_are_non_empty():
    assert "libportaudio2" in deb_runtime_dependencies()
