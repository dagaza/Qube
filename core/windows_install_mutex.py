"""Named mutex pairing with installer/qube.iss AppMutex for clean uninstall."""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes

logger = logging.getLogger("Qube.WindowsInstallMutex")

# Keep in sync with #define MyAppMutex in installer/qube.iss
INSTALL_MUTEX_NAME = "dagaza.Qube.AppMutex"

_handle: int | None = None


def acquire_install_mutex() -> None:
    """Hold a process-wide mutex so Inno CloseApplications can stop Qube."""
    global _handle
    if sys.platform != "win32" or _handle is not None:
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.CreateMutexW(None, wintypes.BOOL(False), INSTALL_MUTEX_NAME)
    if not handle:
        logger.debug("CreateMutexW failed (error %s)", ctypes.get_last_error())
        return
    _handle = int(handle)


def release_install_mutex() -> None:
    """Release the install mutex so silent uninstall can remove files."""
    global _handle
    if sys.platform != "win32" or _handle is None:
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle(wintypes.HANDLE(_handle))
    _handle = None
