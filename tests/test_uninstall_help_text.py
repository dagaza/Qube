"""Tests for core/uninstall_help_text.py."""

from __future__ import annotations

import sys

from core import uninstall_help_text as mod


def test_uninstall_help_paragraphs_cover_all_platforms():
    text = "\n".join(mod.uninstall_help_paragraphs())
    assert "Windows" in text
    assert "macOS" in text
    assert "Linux" in text
    assert "winget uninstall" in text
    assert "qube-uninstall" in text
    assert "Uninstall Qube.app" in text


def test_uninstall_help_paragraphs_list_current_platform_first(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    paragraphs = mod.uninstall_help_paragraphs()
    assert paragraphs[1].startswith("Linux (this device)")


def test_uninstall_help_paragraphs_include_backup_guidance():
    paragraphs = mod.uninstall_help_paragraphs()
    intro = paragraphs[0].lower()
    assert "backup & restore" in intro
    assert "state backup" in intro
    assert "knowledge pack" in intro


def test_uninstall_help_mentions_silent_deleteuserdata_on_windows():
    text = "\n".join(mod.uninstall_help_paragraphs())
    assert "DELETEUSERDATA=1" in text
    assert "%USERPROFILE%\\.qube" in text
