"""Tests for core/platform/macos_dock.py."""

from __future__ import annotations

import sys

from core.platform import macos_dock as mod


def test_install_macos_dock_reopen_handler_noop_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(mod, "_handler_installed", False)
    monkeypatch.setattr(mod, "_restore_fn", None)

    called: list[int] = []
    mod.install_macos_dock_reopen_handler(lambda: called.append(1))

    assert called == []
    assert mod._restore_fn is None


def test_install_macos_dock_reopen_handler_stores_callback_on_darwin(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(mod, "_handler_installed", True)

    called: list[int] = []
    mod.install_macos_dock_reopen_handler(lambda: called.append(1))

    assert mod._restore_fn is not None
    mod._restore_fn()
    assert called == [1]
