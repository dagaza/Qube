"""
Startup splash overlay: presentation + fade-in, separate from app bootstrap.

The circle spinner is timer-driven (decorative). The step list and progress bar jump
at phase boundaries via :class:`_PhasedQubeRunner` (``QTimer.singleShot(0)`` between
phases so the spinner can repaint when work is not blocking the GUI thread).

Heavy ``EmbeddingModel`` init runs on a stdlib ``threading.Thread``; remaining boot
phases run synchronously on the main thread one phase per event-loop tick.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from PyQt6.QtCore import QEasingCurve, QObject, QPropertyAnimation, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

from ui.splash_widget import QubeSplashCard, resolve_splash_logo_path

logger = logging.getLogger("Qube.UI.Splash")

T = TypeVar("T")

_FADE_IN_MS = 260
_MIN_VISIBLE_MS = 380
_BOOTSTRAP_FALLBACK_MS = _FADE_IN_MS + 200
_SHELL_CHROME_MARGIN = 12
_EMBEDDER_POLL_MS = 40
_SPINNER_INTERVAL_MS = 16
_EMBEDDER_DONE_PERCENT = 12
# Step indices 1–7 in SPLASH_STEP_LABELS; percent jumps at each phase start.
_PHASE_STEPS = (1, 2, 3, 4, 5, 6, 7)
_PHASE_PERCENTS = (22, 38, 52, 70, 82, 92, 100)

SplashPhaseCallback = Callable[[int, int], None]
SplashBuildCallback = Callable[..., None]


class StartupSplashController(QObject):
    """Owns the floating splash shell, fade-in, and dismiss coordination."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        logo_path: Path | str | None = None,
        compact: bool = True,
    ) -> None:
        app = QApplication.instance()
        super().__init__(app if isinstance(app, QObject) else None)
        self._repo_root = repo_root or Path(__file__).resolve().parent.parent
        resolved_logo = Path(logo_path) if logo_path else resolve_splash_logo_path(self._repo_root)
        self._logo_path = str(resolved_logo) if resolved_logo else None

        self._shell = QWidget(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._shell.setObjectName("QubeStartupSplashShell")
        self._shell.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._shell.setWindowOpacity(0.0)

        shell_layout = QVBoxLayout(self._shell)
        shell_layout.setContentsMargins(
            _SHELL_CHROME_MARGIN,
            _SHELL_CHROME_MARGIN,
            _SHELL_CHROME_MARGIN,
            _SHELL_CHROME_MARGIN,
        )
        shell_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._card = QubeSplashCard(
            logo_path=self._logo_path,
            compact=compact,
            parent=self._shell,
        )
        shell_layout.addWidget(self._card)

        self._fade_in_anim: QPropertyAnimation | None = None
        self._fade_in_done = False
        self._bootstrap_kicked = False
        self._first_shown_mono: float | None = None
        self._bootstrap_fn: SplashBuildCallback | None = None
        self._bootstrap_running = False
        self._ready_callback: Callable[[Any], None] | None = None
        self._bootstrap_result: Any = None
        self._dismiss_scheduled = False

        self._embedder_thread: threading.Thread | None = None
        self._embedder_outcome: tuple[bool, object] | None = None
        self._phased_runner: _PhasedQubeRunner | None = None
        self._embedder_poll = QTimer(self)
        self._embedder_poll.setInterval(_EMBEDDER_POLL_MS)
        self._embedder_poll.timeout.connect(self._poll_embedder_thread)

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(_SPINNER_INTERVAL_MS)
        self._spinner_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._spinner_timer.timeout.connect(self._advance_spinner)

    def _advance_spinner(self) -> None:
        self._card.spinner.advance(float(self._spinner_timer.interval()))

    def _on_phase(self, step_index: int, percent: int) -> None:
        if step_index > 0:
            self._card.complete_step(step_index - 1)
        self._card.set_active_step(step_index)
        self._card.set_progress_percent(percent)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _start_spinner(self) -> None:
        self._advance_spinner()
        self._spinner_timer.start()

    def _stop_spinner(self) -> None:
        self._spinner_timer.stop()

    def present(self) -> None:
        """Show the floating card and begin fade-in."""
        self._shell.adjustSize()
        self._center_on_primary_screen()
        self._shell.show()
        self._shell.raise_()
        self._first_shown_mono = time.monotonic()
        self._start_spinner()
        logger.info("Splash card presented.")
        QTimer.singleShot(0, self._start_fade_in)
        QTimer.singleShot(_BOOTSTRAP_FALLBACK_MS, self._bootstrap_fallback)

    def run_bootstrap(
        self,
        fn: SplashBuildCallback,
        *,
        on_ready: Callable[[Any], None],
    ) -> None:
        """Queue startup work to run after fade-in completes (or fallback timer)."""
        self._bootstrap_fn = fn
        self._ready_callback = on_ready
        if self._fade_in_done:
            self._kick_bootstrap()

    def _center_on_primary_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self._shell.frameGeometry()
        frame.moveCenter(available.center())
        self._shell.move(frame.topLeft())

    def _start_fade_in(self) -> None:
        anim = QPropertyAnimation(self._shell, b"windowOpacity", self)
        anim.setDuration(_FADE_IN_MS)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(self._on_fade_in_finished)
        self._fade_in_anim = anim
        anim.start()

    def _on_fade_in_finished(self) -> None:
        self._fade_in_done = True
        self._kick_bootstrap()

    def _bootstrap_fallback(self) -> None:
        if self._bootstrap_kicked or self._bootstrap_fn is None:
            return
        if not self._fade_in_done:
            logger.warning(
                "Splash fade-in did not finish in %dms; starting bootstrap anyway.",
                _BOOTSTRAP_FALLBACK_MS,
            )
            self._fade_in_done = True
        self._kick_bootstrap()

    def _kick_bootstrap(self) -> None:
        if self._bootstrap_kicked or self._bootstrap_fn is None:
            return
        self._bootstrap_kicked = True
        QTimer.singleShot(0, self._begin_embedder_load)

    @staticmethod
    def _load_embedder_worker() -> object:
        from rag.embedder import EmbeddingModel

        return EmbeddingModel()

    def _begin_embedder_load(self) -> None:
        if self._bootstrap_running:
            return
        self._bootstrap_running = True
        self._embedder_outcome = None
        self._embedder_thread = threading.Thread(
            target=self._embedder_thread_main,
            name="QubeSplashEmbedder",
            daemon=True,
        )
        self._embedder_thread.start()
        self._embedder_poll.start()

    def _embedder_thread_main(self) -> None:
        try:
            embedder = self._load_embedder_worker()
            self._embedder_outcome = (True, embedder)
        except Exception as exc:
            self._embedder_outcome = (False, exc)

    def _poll_embedder_thread(self) -> None:
        thread = self._embedder_thread
        if thread is not None and thread.is_alive():
            return
        self._embedder_poll.stop()
        outcome = self._embedder_outcome
        self._embedder_thread = None
        if outcome is None:
            self._bootstrap_running = False
            logger.error("Embedder thread exited without a result.")
            return
        ok, payload = outcome
        if not ok:
            self._bootstrap_running = False
            logger.error("Embedder init failed: %s", payload)
            if isinstance(payload, BaseException):
                raise payload
            raise RuntimeError(f"Embedder init failed: {payload!r}")
        QTimer.singleShot(0, lambda: self._finish_bootstrap(payload))

    def _finish_bootstrap(self, embedder: object) -> None:
        fn = self._bootstrap_fn
        if fn is None:
            self._bootstrap_running = False
            return
        self._card.complete_step(0)
        self._card.set_progress_percent(_EMBEDDER_DONE_PERCENT)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        logger.info("Splash bootstrap started (phased).")
        self._phased_runner = fn(
            embedder=embedder,
            on_phase=self._on_phase,
            on_complete=self._on_phased_bootstrap_complete,
        )

    def _on_phased_bootstrap_complete(self, qube: object) -> None:
        self._bootstrap_running = False
        self._bootstrap_result = qube
        self._card.complete_step(7)
        self._card.set_progress_percent(100)
        logger.info("Splash bootstrap finished.")
        self._schedule_dismiss()

    def _schedule_dismiss(self) -> None:
        if self._dismiss_scheduled:
            return
        self._dismiss_scheduled = True
        if self._first_shown_mono is None:
            self._dismiss_now()
            return
        elapsed_ms = (time.monotonic() - self._first_shown_mono) * 1000.0
        wait_ms = max(0, int(_MIN_VISIBLE_MS - elapsed_ms))
        if wait_ms <= 0:
            self._dismiss_now()
        else:
            QTimer.singleShot(wait_ms, self._dismiss_now)

    def _dismiss_now(self) -> None:
        self._embedder_poll.stop()
        self._stop_spinner()
        logger.info("Splash dismissed.")
        self._shell.hide()
        self._shell.deleteLater()
        if self._ready_callback is not None:
            self._ready_callback(self._bootstrap_result)  # type: ignore[arg-type]


class _PhasedQubeRunner(QObject):
    """Runs ``Qube._boot_*`` one phase per event-loop tick for splash UI updates."""

    def __init__(
        self,
        *,
        embedder: object,
        enable_routing_debug_tool: bool,
        enable_trace_diff_debug_tool: bool = False,
        on_phase: SplashPhaseCallback,
        on_complete: Callable[[object], None],
    ) -> None:
        app = QApplication.instance()
        super().__init__(app if isinstance(app, QObject) else None)
        self._embedder = embedder
        self._enable_routing = enable_routing_debug_tool
        self._enable_trace_diff = enable_trace_diff_debug_tool
        self._on_phase = on_phase
        self._on_complete = on_complete
        self._phase = 0
        self._qube: object | None = None

    def start(self) -> None:
        QTimer.singleShot(0, self._run_next)

    def _run_next(self) -> None:
        if self._phase >= len(_PHASE_STEPS):
            if self._qube is not None:
                self._on_complete(self._qube)
            return
        step_index = _PHASE_STEPS[self._phase]
        percent = _PHASE_PERCENTS[self._phase]
        self._on_phase(step_index, percent)
        try:
            self._run_phase(self._phase)
        except Exception:
            logger.exception("Phased Qube bootstrap failed at phase %d.", self._phase)
            raise
        self._phase += 1
        QTimer.singleShot(0, self._run_next)

    def _run_phase(self, phase: int) -> None:
        from main import Qube

        noop_tick: Callable[[str], None] = lambda _msg: None
        if self._qube is None:
            self._qube = Qube.__new__(Qube)
        qube = self._qube
        if phase == 0:
            qube._boot_storage(noop_tick, self._embedder)  # type: ignore[attr-defined]
        elif phase == 1:
            qube._boot_core_workers(noop_tick)  # type: ignore[attr-defined]
        elif phase == 2:
            qube._boot_memory_workers(noop_tick)  # type: ignore[attr-defined]
        elif phase == 3:
            qube._boot_main_window(noop_tick, self._enable_routing, self._enable_trace_diff)  # type: ignore[attr-defined]
        elif phase == 4:
            qube._boot_connect_and_sync(noop_tick)  # type: ignore[attr-defined]
        elif phase == 5:
            qube._boot_autoload_model(noop_tick)  # type: ignore[attr-defined]
        elif phase == 6:
            qube._boot_runtime(noop_tick)  # type: ignore[attr-defined]


def start_phased_qube_build(
    *,
    embedder: object,
    enable_routing_debug_tool: bool,
    enable_trace_diff_debug_tool: bool = False,
    on_phase: SplashPhaseCallback,
    on_complete: Callable[[object], None],
) -> _PhasedQubeRunner:
    """Build ``Qube`` in boot phases; ``on_phase(step_index, percent)`` before each."""
    runner = _PhasedQubeRunner(
        embedder=embedder,
        enable_routing_debug_tool=enable_routing_debug_tool,
        enable_trace_diff_debug_tool=enable_trace_diff_debug_tool,
        on_phase=on_phase,
        on_complete=on_complete,
    )
    runner.start()
    return runner


def bootstrap_with_splash(
    *,
    repo_root: Path,
    build_app_fn: SplashBuildCallback,
    on_ready: Callable[[Any], None],
) -> StartupSplashController:
    """Present splash, then build the app on the GUI thread when startup completes."""
    splash = StartupSplashController(repo_root=repo_root, compact=True)
    splash.present()
    splash.run_bootstrap(build_app_fn, on_ready=on_ready)
    return splash
