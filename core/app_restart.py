"""Relaunch the running Qube process and quit the current instance."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication

from core.paths import install_root

logger = logging.getLogger("Qube.Restart")


def _project_root() -> Path:
    return install_root()


def platform_display_name() -> str:
    if sys.platform == "win32":
        return "Windows"
    if sys.platform == "darwin":
        return "macOS"
    if sys.platform.startswith("linux"):
        return "Linux"
    return sys.platform


def _running_under_pyqt_inspect() -> bool:
    """True when Qube was started through PyQtInspect (pqi.py wrapper)."""
    for arg in sys.argv:
        normalized = str(arg).replace("\\", "/")
        if "PyQtInspect" in normalized or normalized.endswith("/pqi.py"):
            return True
    return "PyQtInspect.pqi" in sys.modules


def build_relaunch_command() -> tuple[str, list[str], str]:
    """
    Build (program, args, working_directory) for relaunch.

    Debuggers and some IDE wrappers replace ``sys.argv`` with only the Python
    executable, which would otherwise open an interactive REPL instead of Qube.
    """
    program = sys.executable
    root = _project_root()
    cwd = os.getcwd() or str(root)
    main_py = root / "main.py"

    argv_tail = [str(a) for a in sys.argv[1:] if str(a).strip()]

    if getattr(sys, "frozen", False):
        return program, argv_tail, cwd

    if argv_tail and (argv_tail[0].endswith(".py") or argv_tail[0] == "-m"):
        if not Path(argv_tail[0]).is_absolute() and argv_tail[0].endswith(".py"):
            candidate = Path(cwd) / argv_tail[0]
            if candidate.is_file():
                argv_tail = [str(candidate.resolve()), *argv_tail[1:]]
        return program, argv_tail, cwd

    if main_py.is_file():
        return program, [str(main_py.resolve())], str(root)

    return program, argv_tail, cwd


def _resolve_gui_program(program: str) -> str:
    """On Windows prefer pythonw.exe so relaunch does not flash a console window."""
    if sys.platform != "win32":
        return program
    exe = Path(program)
    if exe.name.lower() == "python.exe":
        pythonw = exe.with_name("pythonw.exe")
        if pythonw.is_file():
            return str(pythonw)
    return program


def _start_detached_windows_createprocess(program: str, args: list[str], workdir: str) -> bool:
    """Launch directly via kernel32 CreateProcessW (no shell, no PyQtInspect hooks)."""
    import ctypes
    from ctypes import wintypes

    CREATE_NO_WINDOW = 0x08000000
    CREATE_UNICODE_ENVIRONMENT = 0x00000400
    DETACHED_PROCESS = 0x00000008
    STARTF_USESHOWWINDOW = 0x00000001
    SW_HIDE = 0

    class STARTUPINFO(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD),
            ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD),
            ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
            ("hStdInput", wintypes.HANDLE),
            ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        ]

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("hProcess", wintypes.HANDLE),
            ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD),
            ("dwThreadId", wintypes.DWORD),
        ]

    si = STARTUPINFO()
    si.cb = ctypes.sizeof(STARTUPINFO)
    si.dwFlags = STARTF_USESHOWWINDOW
    si.wShowWindow = SW_HIDE
    pi = PROCESS_INFORMATION()

    cmdline = subprocess.list2cmdline([program, *args])
    cmd_buf = ctypes.create_unicode_buffer(cmdline)

    ok = ctypes.windll.kernel32.CreateProcessW(
        program,
        cmd_buf,
        None,
        None,
        False,
        CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_UNICODE_ENVIRONMENT,
        None,
        workdir,
        ctypes.byref(si),
        ctypes.byref(pi),
    )
    if not ok:
        logger.error("CreateProcessW relaunch failed: %s", ctypes.get_last_error())
        return False
    ctypes.windll.kernel32.CloseHandle(pi.hProcess)
    ctypes.windll.kernel32.CloseHandle(pi.hThread)
    return True


def _start_detached_windows_bypass_debugger(program: str, args: list[str], workdir: str) -> bool:
    """
    Relaunch without PyQtInspect injecting pqi.py into the child process.

    Uses kernel32 CreateProcessW directly so neither QProcess, subprocess, nor
    PowerShell are involved (no console/shell flicker).
    """
    return _start_detached_windows_createprocess(program, args, workdir)


def _start_detached_windows(program: str, args: list[str], workdir: str) -> bool:
    if _running_under_pyqt_inspect():
        logger.info(
            "PyQtInspect detected — relaunching via CreateProcessW (debug session will end)"
        )
        return _start_detached_windows_bypass_debugger(program, args, workdir)

    ok = QProcess.startDetached(program, args, workdir)
    if ok:
        return True
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
        subprocess.Popen(
            [program, *args],
            cwd=workdir,
            close_fds=True,
            creationflags=creationflags,
        )
        return True
    except OSError as exc:
        logger.error("Windows detached relaunch failed: %s", exc)
        return False


def restart_action_label() -> str:
    return "Restart now"


def restart_prompt_body(*, purpose: str = "apply changes") -> str:
    if sys.platform == "win32":
        return (
            f"Restart Qube now to {purpose}. "
            "You can also close Qube and open it again from the Start menu or desktop shortcut."
        )
    if sys.platform == "darwin":
        return (
            f"Restart Qube now to {purpose}. "
            "You can also quit with Cmd+Q and reopen Qube from Applications."
        )
    return (
        f"Restart Qube now to {purpose}. "
        "You can also close the window and launch Qube from your app menu or desktop entry."
    )


def manual_restart_instructions() -> str:
    """Fallback copy when automatic relaunch fails."""
    root = _project_root()
    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            return "Close Qube completely, then start it again from the Start menu or desktop shortcut."
        if sys.platform == "darwin":
            return "Quit Qube (Cmd+Q), then reopen it from Applications or the Dock."
        return "Close Qube completely, then open it again from your applications menu."

    py = "python3" if sys.platform != "win32" else "python"
    main_py = root / "main.py"
    script_line = f"{py} {main_py.name}"
    if sys.platform == "win32":
        return (
            "Close Qube, open PowerShell or Command Prompt, then run:\n\n"
            f"cd \"{root}\"\n"
            f"{script_line}"
        )
    if sys.platform == "darwin":
        return (
            "Quit Qube (Cmd+Q), open Terminal, then run:\n\n"
            f"cd \"{root}\"\n"
            f"{script_line}"
        )
    return (
        "Close Qube, open a terminal, then run:\n\n"
        f"cd \"{root}\"\n"
        f"{script_line}"
    )


def relaunch_and_quit() -> bool:
    """Restart Qube using the most natural mechanism for the current OS."""
    app = QApplication.instance()
    if app is None:
        return False

    program, args, workdir = build_relaunch_command()
    program = _resolve_gui_program(program)
    argv = [program, *args]

    logger.info(
        "Restart requested on %s — relaunching %s (cwd=%s)",
        platform_display_name(),
        " ".join(argv),
        workdir,
    )

    if sys.platform == "win32":
        if not _start_detached_windows(program, args, workdir):
            return False
        app.quit()
        return True

    def _relaunch_after_quit() -> None:
        try:
            os.chdir(workdir)
            os.execv(program, argv)
        except OSError as exc:
            logger.error("execv relaunch failed on %s: %s", platform_display_name(), exc)

    app.aboutToQuit.connect(_relaunch_after_quit)
    app.quit()
    return True
