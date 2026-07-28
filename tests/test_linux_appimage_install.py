"""Tests for Linux AppImage install helpers."""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

from core.linux_appimage_install import (
    linux_appimage_install_plan,
    parse_appimage_filename,
    render_appimage_desktop_entry,
    stale_appimage_install_files,
)


class LinuxAppImageInstallTests(unittest.TestCase):
    def test_parse_release_filename(self) -> None:
        parsed = parse_appimage_filename("Qube-1.2.3-x86_64-vulkan.AppImage")
        self.assertEqual(parsed, ("1.2.3", "vulkan"))

    def test_install_plan_paths(self) -> None:
        plan = linux_appimage_install_plan(
            "/tmp/Qube-1.0.0-x86_64-cuda.AppImage",
            home=Path("/home/tester"),
        )
        self.assertEqual(plan.variant, "cuda")
        self.assertEqual(
            plan.install_path,
            Path("/home/tester/.local/opt/qube/Qube.AppImage"),
        )
        self.assertEqual(plan.launcher_path, Path("/home/tester/.local/bin/qube-appimage"))

    def test_desktop_entry_includes_extract_and_run(self) -> None:
        plan = linux_appimage_install_plan(
            "Qube-9.9.9-x86_64-cpu.AppImage",
            home=Path("/home/tester"),
        )
        desktop = render_appimage_desktop_entry(plan)
        self.assertIn("APPIMAGE_EXTRACT_AND_RUN=1", desktop)
        self.assertIn("StartupWMClass=Qube", desktop)

    def test_stale_appimage_install_files_remove_other_appimages(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            install_dir = home / ".local" / "opt" / "qube"
            install_dir.mkdir(parents=True)
            old_release = install_dir / "Qube-1.0.0-x86_64-cpu.AppImage"
            other = install_dir / "Other.AppImage"
            old_release.write_bytes(b"old")
            other.write_bytes(b"other")

            plan = linux_appimage_install_plan(
                "Qube-1.1.0-x86_64-cpu.AppImage",
                home=home,
            )
            stale = stale_appimage_install_files(plan)
            self.assertEqual(
                sorted(stale),
                sorted([old_release.resolve(), other.resolve()]),
            )

    @unittest.skipUnless(
        sys.platform.startswith("linux") and shutil.which("bash"),
        "install_appimage.sh dry-run requires Linux bash",
    )
    def test_install_script_dry_run(self) -> None:
        import subprocess
        import tempfile

        repo = Path(__file__).resolve().parents[1]
        script = repo / "scripts" / "linux" / "install_appimage.sh"
        with tempfile.TemporaryDirectory() as tmp:
            appimage = Path(tmp) / "Qube-1.0.0-x86_64-vulkan.AppImage"
            appimage.write_bytes(b"#!/bin/sh\n")
            result = subprocess.run(
                [str(script), "--dry-run", str(appimage)],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("qube-appimage", result.stdout)


if __name__ == "__main__":
    unittest.main()
