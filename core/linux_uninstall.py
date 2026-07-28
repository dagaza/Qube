"""Launch the Linux .deb uninstall script bundled with packaged installs."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from core.paths import install_root

logger = logging.getLogger("Qube.LinuxUninstall")

_LINUX_OPT_SCRIPT = Path("/opt/qube/uninstall/uninstall.sh")
_LINUX_CLI = Path("/usr/bin/qube-uninstall")


def resolve_uninstall_script_path() -> Path | None:
    """Return the uninstall script when this install supports in-app uninstall."""
    if not sys.platform.startswith("linux"):
        return None

    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(install_root() / "uninstall" / "uninstall.sh")
    candidates.extend([_LINUX_OPT_SCRIPT, _LINUX_CLI])

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def is_linux_uninstall_available() -> bool:
    return resolve_uninstall_script_path() is not None


def _script_argv(script: Path, *, keep_user_data: bool) -> list[str]:
    if script.name == "qube-uninstall":
        argv = [str(script)]
    else:
        argv = ["/bin/bash", str(script)]
    argv.append("--confirmed")
    if keep_user_data:
        argv.append("--keep-data")
    return argv


def launch_linux_uninstall(*, keep_user_data: bool = False) -> tuple[bool, str]:
    """
    Start the detached uninstall script.

    On success the caller should quit the Qt application — the script waits for
    Qube to exit before removing files.
    """
    script = resolve_uninstall_script_path()
    if script is None:
        return False, "Uninstall script not found in this installation."

    try:
        subprocess.Popen(
            _script_argv(script, keep_user_data=keep_user_data),
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        logger.warning("Failed to start Linux uninstaller: %s", exc)
        return False, f"Could not start uninstaller: {exc}"

    return True, ""


def request_linux_uninstall(*, keep_user_data: bool = False) -> tuple[bool, str]:
    """Launch uninstall and quit the running app when successful."""
    ok, message = launch_linux_uninstall(keep_user_data=keep_user_data)
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
