"""Tests for core/windows_install_mutex.py."""

from __future__ import annotations

from pathlib import Path

from core import windows_install_mutex as mod


def test_install_mutex_name_matches_inno_script():
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "installer" / "qube.iss").read_text(encoding="utf-8")
    assert mod.INSTALL_MUTEX_NAME in text
    assert 'MySetupMutex   "dagaza.Qube.SetupMutex"' in text
    assert "SetupMutex={#MySetupMutex}" in text


def test_initialize_setup_kills_running_qube_before_appmutex():
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "installer" / "qube.iss").read_text(encoding="utf-8")
    setup_fn = text.split("function InitializeSetup(): Boolean;", 1)[1]
    body = setup_fn.split("end;", 1)[0]
    assert "KillRunningQube();" in body


def test_acquire_install_mutex_is_noop_off_windows(monkeypatch):
    mod._handle = None
    monkeypatch.setattr(mod.sys, "platform", "linux")
    mod.acquire_install_mutex()
    assert mod._handle is None
    mod.release_install_mutex()


def test_release_install_mutex_is_noop_off_windows(monkeypatch):
    mod._handle = None
    monkeypatch.setattr(mod.sys, "platform", "linux")
    mod.release_install_mutex()
    assert mod._handle is None
