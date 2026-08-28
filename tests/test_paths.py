"""Tests for frozen-aware path resolution."""

from __future__ import annotations

import sys
from unittest import mock

from core import paths


def test_install_root_dev_points_at_repo_root():
    root = paths.install_root()
    assert (root / "main.py").is_file()
    assert (root / "core" / "paths.py").is_file()


def test_resource_path_resolves_assets():
    schema = paths.resource_path("assets", "config", "settings.schema.json")
    assert schema.is_file()


def test_user_data_root_is_writable(tmp_path, monkeypatch):
    def _root():
        path = tmp_path / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(paths, "user_data_root", _root)
    root = paths.user_data_root()
    assert root.is_dir()
    db = paths.default_db_path()
    assert db.parent == root
    assert db.name == "qube_data.db"


def test_install_root_frozen_uses_executable_parent(tmp_path):
    fake_exe = tmp_path / "Qube.exe"
    fake_exe.touch()
    with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
        sys, "executable", str(fake_exe)
    ):
        assert paths.install_root() == tmp_path


def test_resource_path_frozen_uses_meipass(tmp_path):
    internal_assets = tmp_path / "_internal" / "assets" / "config"
    internal_assets.mkdir(parents=True)
    schema = internal_assets / "settings.schema.json"
    schema.write_text("{}", encoding="utf-8")
    fake_exe = tmp_path / "Qube.exe"
    fake_exe.touch()
    with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
        sys, "executable", str(fake_exe)
    ), mock.patch.object(sys, "_MEIPASS", str(tmp_path / "_internal"), create=True):
        assert (
            paths.resource_path("assets", "config", "settings.schema.json") == schema
        )
