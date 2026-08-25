"""Hard process exit helpers for aborted startup (splash close before app ready).

Closing the frameless bootstrap splash only tears down the splash window. Phased boot
may already have started non-daemon QThreads / native engine workers, and
``setQuitOnLastWindowClosed(False)`` means a vanished splash does not imply process
death. Arm a daemon failsafe so the OS process cannot linger without UI.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger("Qube.StartupExit")

_FORCE_EXIT_ARMED = False
_EXIT_REQUESTED = False
_DEFAULT_FORCE_EXIT_DELAY_S = 0.75


def startup_exit_requested() -> bool:
    return _EXIT_REQUESTED


def mark_startup_exit_requested() -> None:
    global _EXIT_REQUESTED
    _EXIT_REQUESTED = True


def arm_force_process_exit(*, delay_s: float = _DEFAULT_FORCE_EXIT_DELAY_S, code: int = 0) -> None:
    """Schedule ``os._exit`` if the interpreter hangs on leftover worker threads."""
    global _FORCE_EXIT_ARMED
    mark_startup_exit_requested()
    if _FORCE_EXIT_ARMED:
        return
    _FORCE_EXIT_ARMED = True

    def _kill() -> None:
        logger.info("Forcing process exit after splash abort (code=%s).", code)
        os._exit(code)

    timer = threading.Timer(max(0.0, float(delay_s)), _kill)
    timer.daemon = True
    timer.start()


def force_process_exit_now(code: int = 0) -> None:
    """Immediate hard exit; use after ``app.exec()`` returns from a splash abort."""
    mark_startup_exit_requested()
    logger.info("Forcing process exit now after splash abort (code=%s).", code)
    os._exit(code)
