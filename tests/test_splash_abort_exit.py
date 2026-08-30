"""Tests for splash/bootstrap close aborting the process cleanly."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QObject

import core.startup_exit as startup_exit
from core.startup_exit import (
    arm_force_process_exit,
    mark_startup_exit_requested,
    startup_exit_requested,
)
from ui.splash_overlay import StartupSplashController, _PhasedQubeRunner


@pytest.fixture(autouse=True)
def _reset_startup_exit_flags() -> None:
    startup_exit._FORCE_EXIT_ARMED = False
    startup_exit._EXIT_REQUESTED = False
    yield
    startup_exit._FORCE_EXIT_ARMED = False
    startup_exit._EXIT_REQUESTED = False


def _abortable_splash_stub() -> StartupSplashController:
    """Minimal controller for abort path without building splash widgets."""
    splash = StartupSplashController.__new__(StartupSplashController)
    QObject.__init__(splash)
    splash._exit_requested = False
    splash._dismiss_scheduled = False
    splash._ready_callback = lambda _q: None
    splash._bootstrap_fn = lambda **_kwargs: None
    splash._bootstrap_running = True
    splash._embedder_poll = MagicMock()
    splash._phased_runner = None
    splash._shell = None
    splash._stop_spinner = MagicMock()
    return splash


def test_arm_force_process_exit_marks_requested_and_schedules_timer() -> None:
    with patch("core.startup_exit.threading.Timer") as timer_cls:
        timer = MagicMock()
        timer_cls.return_value = timer
        arm_force_process_exit(delay_s=1.25, code=3)

    assert startup_exit_requested() is True
    timer_cls.assert_called_once()
    assert timer_cls.call_args.args[0] == 1.25
    assert timer.daemon is True
    timer.start.assert_called_once()


def test_phased_runner_cancel_stops_further_phases(qapp_cls) -> None:
    _ = qapp_cls.instance() or qapp_cls([])
    phases: list[int] = []

    def on_phase(step_index: int, percent: int) -> None:
        phases.append(step_index)

    runner = _PhasedQubeRunner(
        embedder=None,
        enable_routing_debug_tool=False,
        on_phase=on_phase,
        on_complete=lambda _q: None,
    )
    runner.cancel()
    runner._run_next()
    assert phases == []


def test_splash_close_aborts_and_quits_app(qapp_cls, monkeypatch: pytest.MonkeyPatch) -> None:
    app = qapp_cls.instance() or qapp_cls([])

    quit_calls: list[bool] = []
    monkeypatch.setattr(app, "quit", lambda: quit_calls.append(True))

    splash = _abortable_splash_stub()
    splash._phased_runner = MagicMock()
    splash._phased_runner.partial_qube.return_value = None

    with patch("ui.splash_overlay.arm_force_process_exit") as arm_exit:
        splash.abort_startup_and_exit()

    assert splash.exit_requested() is True
    assert startup_exit_requested() is True
    assert quit_calls == [True]
    arm_exit.assert_called_once()
    splash._phased_runner.cancel.assert_called_once()
    splash._embedder_poll.stop.assert_called_once()
    splash._stop_spinner.assert_called_once()


def test_splash_abort_signals_partial_workers() -> None:
    class _Engine:
        def __init__(self) -> None:
            self.stop_engine = MagicMock()

    class _Tray:
        def __init__(self) -> None:
            self.hide_tray = MagicMock()

    class _Window:
        def __init__(self) -> None:
            self.tray_controller = _Tray()

    class _PartialQube:
        def __init__(self) -> None:
            self.native_llama_engine = _Engine()
            self.window = _Window()

    qube = _PartialQube()
    StartupSplashController._signal_partial_qube_stop(qube, blocking=True)

    qube.window.tray_controller.hide_tray.assert_called_once()
    qube.native_llama_engine.stop_engine.assert_called_once_with(wait_ms=30_000)


def test_splash_abort_nonblocking_skips_native_engine_wait() -> None:
    class _Engine:
        def __init__(self) -> None:
            self.stop_engine = MagicMock()

    class _PartialQube:
        def __init__(self) -> None:
            self.native_llama_engine = _Engine()

    qube = _PartialQube()
    StartupSplashController._signal_partial_qube_stop(qube, blocking=False)

    qube.native_llama_engine.stop_engine.assert_called_once_with(wait_ms=0)


def test_mark_startup_exit_requested() -> None:
    assert startup_exit_requested() is False
    mark_startup_exit_requested()
    assert startup_exit_requested() is True
