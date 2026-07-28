"""Tests for core/linux_uninstall.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from core import linux_uninstall as mod


def test_resolve_uninstall_script_path_non_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert mod.resolve_uninstall_script_path() is None


def test_resolve_uninstall_script_path_frozen_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    exe = tmp_path / "opt" / "qube" / "Qube"
    script = tmp_path / "opt" / "qube" / "uninstall" / "uninstall.sh"
    exe.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(exe))

    assert mod.resolve_uninstall_script_path() == script


def test_launch_linux_uninstall_starts_script(monkeypatch, tmp_path):
    script = tmp_path / "uninstall.sh"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setattr(mod, "resolve_uninstall_script_path", lambda: script)

    with mock.patch.object(mod.subprocess, "Popen") as popen:
        ok, message = mod.launch_linux_uninstall(keep_user_data=True)

    assert ok is True
    assert message == ""
    popen.assert_called_once()
    args = popen.call_args.args[0]
    assert args[:2] == ["/bin/bash", str(script)]
    assert "--confirmed" in args
    assert args[-1] == "--keep-data"


def test_request_linux_uninstall_quits_app(monkeypatch, tmp_path):
    script = tmp_path / "uninstall.sh"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    monkeypatch.setattr(mod, "resolve_uninstall_script_path", lambda: script)

    app = mock.Mock()
    monkeypatch.setitem(
        sys.modules,
        "PyQt6.QtWidgets",
        mock.Mock(QApplication=mock.Mock(instance=mock.Mock(return_value=app))),
    )

    with mock.patch.object(mod.subprocess, "Popen"):
        ok, message = mod.request_linux_uninstall()

    assert ok is True
    assert message == ""
    app.quit.assert_called_once()
