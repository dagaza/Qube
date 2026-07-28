"""Launch the macOS uninstall script bundled with Qube.app."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from core.paths import install_root

logger = logging.getLogger("Qube.MacOSUninstall")


def resolve_uninstall_script_path() -> Path | None:
    """Return ``uninstall.sh`` when this install supports in-app uninstall."""
    if sys.platform != "darwin":
        return None

    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve()
        bundled = exe.parent.parent / "Resources" / "uninstall" / "uninstall.sh"
        if bundled.is_file():
            return bundled

    for candidate in (
        install_root() / "dist" / "Uninstall Qube.app" / "Contents" / "Resources" / "uninstall.sh",
        install_root() / "dist" / "Qube.app" / "Contents" / "Resources" / "uninstall" / "uninstall.sh",
    ):
        if candidate.is_file():
            return candidate
    return None


def is_macos_uninstall_available() -> bool:
    return resolve_uninstall_script_path() is not None


def launch_macos_uninstall(*, keep_user_data: bool = False) -> tuple[bool, str]:
    """
    Start the detached uninstall script.

    On success the caller should quit the Qt application — the script waits for
    Qube to exit before removing files.
    """
    script = resolve_uninstall_script_path()
    if script is None:
        return False, "Uninstall script not found in this installation."

    args = ["/bin/bash", str(script)]
    if keep_user_data:
        args.append("--keep-data")

    try:
        subprocess.Popen(
            args,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        logger.warning("Failed to start macOS uninstaller: %s", exc)
        return False, f"Could not start uninstaller: {exc}"

    return True, ""


def request_macos_uninstall(*, keep_user_data: bool = False) -> tuple[bool, str]:
    """Launch uninstall and quit the running app when successful."""
    ok, message = launch_macos_uninstall(keep_user_data=keep_user_data)
    if not ok:
        return ok, message

    try:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.quit()
    except Exception as exc:
        logger.debug("Quit after uninstall request failed: %s", exc)

    return True, ""
