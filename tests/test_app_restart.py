"""Tests for application relaunch command construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

from core.app_restart import (
    build_relaunch_command,
    manual_restart_instructions,
    restart_prompt_body,
    _resolve_gui_program,
)


class TestAppRestart(unittest.TestCase):
    def test_falls_back_to_main_py_when_argv_empty(self) -> None:
        root = Path(__file__).resolve().parent.parent
        main_py = root / "main.py"
        if not main_py.is_file():
            self.skipTest("main.py missing")

        with mock.patch.object(sys, "argv", ["/usr/bin/python3"]), mock.patch.object(
            sys, "frozen", False, create=True
        ), mock.patch("core.app_restart.os.getcwd", return_value=str(root)):
            program, args, cwd = build_relaunch_command()

        self.assertIn("python", program.lower())
        self.assertEqual(len(args), 1)
        self.assertTrue(args[0].endswith("main.py"))
        self.assertEqual(Path(cwd), root)

    def test_preserves_script_argv_when_present(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["/usr/bin/python3", "main.py", "--routing-debug"],
        ), mock.patch.object(sys, "frozen", False, create=True):
            _program, args, _cwd = build_relaunch_command()

        self.assertEqual(args[1:], ["--routing-debug"])
        self.assertTrue(args[0].endswith("main.py"))

    def test_windows_prefers_pythonw(self) -> None:
        with mock.patch.object(sys, "platform", "win32"):
            resolved = _resolve_gui_program(r"C:\Python313\python.exe")
        self.assertTrue(resolved.lower().endswith("pythonw.exe") or resolved.endswith("python.exe"))

    def test_manual_restart_instructions_mention_terminal_on_linux(self) -> None:
        with mock.patch.object(sys, "platform", "linux"), mock.patch.object(
            sys, "frozen", False, create=True
        ):
            text = manual_restart_instructions()
        self.assertIn("terminal", text.lower())
        self.assertIn("python3", text)

    def test_manual_restart_instructions_mention_powershell_on_windows(self) -> None:
        with mock.patch.object(sys, "platform", "win32"), mock.patch.object(
            sys, "frozen", False, create=True
        ):
            text = manual_restart_instructions()
        self.assertIn("PowerShell", text)

    def test_restart_prompt_body_is_platform_specific(self) -> None:
        with mock.patch.object(sys, "platform", "darwin"):
            body = restart_prompt_body()
        self.assertIn("Cmd+Q", body)


if __name__ == "__main__":
    unittest.main()
