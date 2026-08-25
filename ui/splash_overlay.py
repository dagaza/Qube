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
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from PyQt6.QtCore import QEasingCurve, QObject, QPropertyAnimation, QSize, Qt, QTimer, QEvent, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QMouseEvent, QShowEvent
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.bootstrap_manifest import BOOTSTRAP_MODELS, BootstrapModelId
from core.bootstrap_download import (
    bootstrap_download_should_mock,
    download_bootstrap_models,
    format_download_detail,
    model_is_present,
    simulate_bootstrap_downloads,
)
from core.bootstrap_selection import effective_bootstrap_selection, save_bootstrap_selection
from core.platform.window_activation import (
    activate_toplevel_window,
    center_widget_on_screen,
    splash_window_flags,
)
from ui.bootstrap_consent_dialog import BootstrapConsentPanel
from ui.components.prestige_dialog import PrestigeDialog
from ui.splash_widget import (
    QubeFirstRunSplitSplash,
    QubeSplashCard,
    RotatingQubeCube,
    resolve_splash_logo_path,
)

logger = logging.getLogger("Qube.UI.Splash")

T = TypeVar("T")

_FADE_IN_MS = 260
_MIN_VISIBLE_MS = 380
_BOOTSTRAP_FALLBACK_MS = _FADE_IN_MS + 200
_EMBEDDER_POLL_MS = 40
_SPINNER_INTERVAL_MS = 16
_DOWNLOAD_DONE_PERCENT = 10
_EMBEDDER_DONE_PERCENT = 22
# Step indices 1–7 in SPLASH_STEP_LABELS; percent jumps at each phase start.
_PHASE_STEPS = (1, 2, 3, 4, 5, 6, 7)
_PHASE_PERCENTS = (22, 38, 52, 70, 82, 92, 100)
_SPLASH_CHROME_MARGIN = 8
_SPLASH_CHROME_BTN_GAP = 6
_SPLASH_MINIMIZE_BTN_SIZE = 28

SplashPhaseCallback = Callable[[int, int], None]
SplashBuildCallback = Callable[..., None]


_DRAG_BLOCK_TYPES: tuple[type[QWidget], ...] = (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSlider,
    QTextEdit,
)


def _widget_blocks_splash_drag(widget: QWidget | None) -> bool:
    w = widget
    while w is not None:
        if isinstance(w, RotatingQubeCube):
            return True
        if w.metaObject().className() in ("_BootstrapModelRow", "_CollapsibleHeader"):
            return True
        if isinstance(w, _DRAG_BLOCK_TYPES):
            return True
        w = w.parentWidget()
    return False


class _SplashWindowDragFilter(QObject):
    """Drag the frameless splash shell from non-interactive chrome."""

    def __init__(self, shell: QWidget) -> None:
        super().__init__(shell)
        self._shell = shell
        self._drag_offset = None

    def register_widget_tree(self, root: QWidget) -> None:
        root.installEventFilter(self)
        for child in root.findChildren(QWidget):
            child.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if not isinstance(watched, QWidget):
            return False
        if event.type() == QEvent.Type.MouseButtonPress:
            mouse = event
            if (
                isinstance(mouse, QMouseEvent)
                and mouse.button() == Qt.MouseButton.LeftButton
                and not _widget_blocks_splash_drag(watched)
            ):
                self._drag_offset = (
                    mouse.globalPosition().toPoint() - self._shell.frameGeometry().topLeft()
                )
            return False
        if event.type() == QEvent.Type.MouseMove:
            mouse = event
            if (
                isinstance(mouse, QMouseEvent)
                and self._drag_offset is not None
                and mouse.buttons() & Qt.MouseButton.LeftButton
            ):
                self._shell.move(mouse.globalPosition().toPoint() - self._drag_offset)
                return True
            return False
        if event.type() == QEvent.Type.MouseButtonRelease:
            mouse = event
            if isinstance(mouse, QMouseEvent) and mouse.button() == Qt.MouseButton.LeftButton:
                self._drag_offset = None
            return False
        return False


def _splash_minimize_leading_edge() -> bool:
    """True for top-left placement (macOS); False for top-right (Windows/Linux)."""
    return sys.platform == "darwin"


class _DownloadProgressBridge(QObject):
    """Thread-safe bridge for download progress callbacks from worker threads."""

    progress = pyqtSignal(str, str, int, str, str)

    def publish(
        self,
        step_label: str,
        filename: str,
        percent: int,
        source_display: str,
        detail: str,
    ) -> None:
        self.progress.emit(step_label, filename, percent, source_display, detail)


class _StartupSplashShell(QWidget):
    """Frameless splash window; quitting early during first-run consent is allowed."""

    def __init__(self, controller: "StartupSplashController") -> None:
        super().__init__(None, splash_window_flags())
        self._controller = controller
        self.setWindowTitle("Qube")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")
        self._minimize_btn = self._create_chrome_button(
            "QubeSplashMinimizeButton",
            "fa5s.minus",
            "Minimise",
            self.showMinimized,
        )
        self._close_btn = self._create_chrome_button(
            "QubeSplashCloseButton",
            "fa5s.times",
            "Close",
            self.close,
        )

    def _create_chrome_button(
        self,
        object_name: str,
        icon_name: str,
        tooltip: str,
        on_click: Callable[[], None],
    ) -> QPushButton:
        import qtawesome as qta

        from ui.branded_theme import SPLASH_CHROME_ICON, splash_overlay_chrome_button_qss

        btn = QPushButton(self)
        btn.setObjectName(object_name)
        btn.setProperty("class", "WindowControlButton")
        btn.setFixedSize(_SPLASH_MINIMIZE_BTN_SIZE, _SPLASH_MINIMIZE_BTN_SIZE)
        btn.setIcon(qta.icon(icon_name, color=SPLASH_CHROME_ICON))
        btn.setIconSize(QSize(12, 12))
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.CursorShape.ArrowCursor)
        btn.clicked.connect(on_click)
        btn.setStyleSheet(splash_overlay_chrome_button_qss(object_name))
        return btn

    def _position_chrome_buttons(self) -> None:
        margin = _SPLASH_CHROME_MARGIN
        gap = _SPLASH_CHROME_BTN_GAP
        if _splash_minimize_leading_edge():
            x = margin
            for btn in (self._minimize_btn, self._close_btn):
                btn.move(x, margin)
                btn.raise_()
                x += btn.width() + gap
        else:
            x = self.width() - margin
            for btn in (self._close_btn, self._minimize_btn):
                x -= btn.width()
                btn.move(x, margin)
                btn.raise_()
                x -= gap

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_chrome_buttons()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self._position_chrome_buttons()
        activate_toplevel_window(self)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        if event is not None:
            if self._controller.consent_pending():
                logger.info("First-run consent dismissed from splash; exiting.")
            else:
                logger.info("Startup splash closed before app ready; exiting.")
            event.accept()
            app = QApplication.instance()
            if app is not None:
                app.quit()
            return
        super().closeEvent(event)


class StartupSplashController(QObject):
    """Owns the floating splash shell, fade-in, and dismiss coordination."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        logo_path: Path | str | None = None,
        compact: bool = True,
        selected_models: set[BootstrapModelId] | None = None,
        needs_consent: bool = False,
        mock_downloads: bool = False,
    ) -> None:
        app = QApplication.instance()
        super().__init__(app if isinstance(app, QObject) else None)
        self._repo_root = repo_root or Path(__file__).resolve().parent.parent
        self._selected_models = selected_models or effective_bootstrap_selection()
        self._needs_consent = needs_consent
        self._consent_resolved = not needs_consent
        self._mock_downloads = bootstrap_download_should_mock(explicit_mock=mock_downloads)
        self._logo_rotating = False
        resolved_logo = Path(logo_path) if logo_path else resolve_splash_logo_path(self._repo_root)
        self._logo_path = str(resolved_logo) if resolved_logo else None

        self._shell = _StartupSplashShell(self)
        self._shell.setObjectName("QubeStartupSplashShell")
        self._shell.setWindowOpacity(0.0)

        shell_layout = QVBoxLayout(self._shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._card: QubeSplashCard | None = None
        self._split: QubeFirstRunSplitSplash | None = None
        self._consent_panel: BootstrapConsentPanel | None = None

        if self._needs_consent:
            self._split = QubeFirstRunSplitSplash(logo_path=self._logo_path, parent=self._shell)
            self._consent_panel = BootstrapConsentPanel(
                parent=self._split,
                split_embedded=True,
            )
            self._consent_panel.selection_confirmed.connect(self._on_consent_confirmed)
            self._split.set_consent_widget(self._consent_panel)
            shell_layout.addWidget(self._split)
        else:
            self._card = QubeSplashCard(
                logo_path=self._logo_path,
                compact=compact,
                parent=self._shell,
            )
            shell_layout.addWidget(self._card)

        self._drag_filter = _SplashWindowDragFilter(self._shell)
        self._drag_filter.register_widget_tree(self._shell)

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
        self._download_thread: threading.Thread | None = None
        self._embedder_outcome: tuple[bool, object] | None = None
        self._download_outcome: tuple[bool, object] | None = None
        self._phased_runner: _PhasedQubeRunner | None = None
        self._embedder_poll = QTimer(self)
        self._embedder_poll.setInterval(_EMBEDDER_POLL_MS)
        self._embedder_poll.timeout.connect(self._poll_background_threads)

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(_SPINNER_INTERVAL_MS)
        self._spinner_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._spinner_timer.timeout.connect(self._advance_spinner)

        self._download_progress_bridge = _DownloadProgressBridge(self)
        self._download_progress_bridge.progress.connect(self._apply_download_progress)

    def _apply_download_progress(
        self,
        _step_label: str,
        _filename: str,
        percent: int,
        _source_display: str,
        detail: str,
    ) -> None:
        self._set_logo_rotating(True)
        self._view.set_download_detail(detail)
        overall = int(percent * _DOWNLOAD_DONE_PERCENT / 100)
        self._view.set_progress_percent(overall)

    @property
    def _view(self) -> QubeSplashCard | QubeFirstRunSplitSplash:
        if self._split is not None:
            return self._split
        assert self._card is not None
        return self._card

    def consent_pending(self) -> bool:
        return self._needs_consent and not self._consent_resolved

    def _on_consent_confirmed(self, selected: set[BootstrapModelId]) -> None:
        save_bootstrap_selection(selected)
        self._selected_models = set(selected)
        self._needs_consent = False
        self._consent_resolved = True
        if self._split is not None:
            self._split.dismiss_consent_side()
        self._recenter_on_primary_screen()
        self._set_logo_rotating(True)
        self._start_spinner()
        logger.info("First-run consent confirmed; starting downloads.")
        QTimer.singleShot(0, self._kick_bootstrap_after_consent)

    def _kick_bootstrap_after_consent(self) -> None:
        if self._fade_in_done:
            self._kick_bootstrap()
        else:
            QTimer.singleShot(50, self._kick_bootstrap_after_consent)

    def _set_logo_rotating(self, rotating: bool) -> None:
        self._logo_rotating = rotating
        if self._split is not None:
            self._split.set_logo_rotating(rotating)

    def _advance_spinner(self) -> None:
        interval = float(self._spinner_timer.interval())
        if self._split is not None:
            if self._logo_rotating:
                self._split.advance_logo(interval)
        elif self._card is not None:
            self._card.spinner.advance(interval)

    def _on_phase(self, step_index: int, percent: int) -> None:
        view = self._view
        view.setUpdatesEnabled(False)
        try:
            if step_index > 0:
                view.complete_step(step_index - 1)
            view.set_active_step(step_index)
            view.set_progress_percent(percent)
            if self._split is not None and percent > 0:
                self._set_logo_rotating(True)
        finally:
            view.setUpdatesEnabled(True)
            view.update()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _start_spinner(self) -> None:
        self._advance_spinner()
        self._spinner_timer.start()

    def _stop_spinner(self) -> None:
        self._spinner_timer.stop()
        self._set_logo_rotating(False)

    def present(self) -> None:
        """Show the floating card and begin fade-in."""
        self._recenter_on_primary_screen()
        self._shell.show()
        activate_toplevel_window(self._shell)
        self._first_shown_mono = time.monotonic()
        if not self.consent_pending():
            self._start_spinner()
        logger.info(
            "Splash presented (first_run=%s, mock_downloads=%s).",
            self._needs_consent,
            self._mock_downloads,
        )
        QTimer.singleShot(0, self._start_fade_in)
        QTimer.singleShot(_BOOTSTRAP_FALLBACK_MS, self._bootstrap_fallback)

    def request_activation(self) -> None:
        activate_toplevel_window(self._shell)

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
        center_widget_on_screen(self._shell)

    def _recenter_on_primary_screen(self) -> None:
        """Re-center after layout settles so the splash sits in screen middle."""
        self._center_on_primary_screen()
        QTimer.singleShot(0, self._center_on_primary_screen)

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
        activate_toplevel_window(self._shell)
        if not self.consent_pending():
            self._kick_bootstrap()

    def _bootstrap_fallback(self) -> None:
        if self._bootstrap_kicked or self._bootstrap_fn is None:
            return
        if self.consent_pending():
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
        QTimer.singleShot(0, self._begin_model_downloads)

    def _models_needing_download(self) -> set[BootstrapModelId]:
        return {mid for mid in self._selected_models if not model_is_present(mid)}

    def _begin_model_downloads(self) -> None:
        if self._bootstrap_running:
            return
        self._bootstrap_running = True
        self._set_logo_rotating(True)
        self._view.set_active_step(0)
        self._view.set_progress_percent(0)
        if self._mock_downloads:
            pending = set(self._selected_models)
        else:
            pending = self._models_needing_download()
        if not pending:
            QTimer.singleShot(0, self._begin_embedder_load)
            return
        self._download_outcome = None
        self._download_thread = threading.Thread(
            target=self._download_thread_main,
            name="QubeSplashDownloads",
            daemon=True,
        )
        self._download_thread.start()
        self._embedder_poll.start()

    def _download_thread_main(self) -> None:
        try:
            if self._mock_downloads:
                errors = simulate_bootstrap_downloads(
                    self._selected_models,
                    on_progress=self._emit_download_progress,
                )
            else:
                errors = download_bootstrap_models(
                    self._selected_models,
                    on_progress=self._emit_download_progress,
                )
            if errors:
                self._download_outcome = (False, errors)
            else:
                self._download_outcome = (True, None)
        except Exception as exc:
            self._download_outcome = (False, exc)

    def _emit_download_progress(
        self,
        step_label: str,
        filename: str,
        percent: int,
        source_display: str,
    ) -> None:
        size_label = ""
        for mid in self._selected_models:
            spec = BOOTSTRAP_MODELS.get(mid)
            if spec and filename in {spec.hf_filename, spec.label, "Whisper Small"}:
                from core.bootstrap_manifest import format_byte_size

                size_label = format_byte_size(spec.size_bytes)
                break
        detail = format_download_detail(filename, percent, source_display, size_label)
        self._download_progress_bridge.publish(
            step_label,
            filename,
            percent,
            source_display,
            detail,
        )

    @staticmethod
    def _load_embedder_worker() -> object | None:
        from rag.embedder import EmbeddingModel

        try:
            return EmbeddingModel()
        except Exception as exc:
            logger.warning("Search model load failed during splash: %s", exc)
            return None

    def _splash_should_load_embedder(self) -> bool:
        from core.bootstrap_manifest import BootstrapModelId
        from core.embedding_models import gguf_override_available

        if BootstrapModelId.SEARCH_PRESET_BALANCED in self._selected_models:
            return True
        return gguf_override_available()

    def _begin_embedder_load(self) -> None:
        if not self._splash_should_load_embedder():
            self._view.set_download_detail("Skipping search model load (not selected).")
            self._view.set_progress_percent(_DOWNLOAD_DONE_PERCENT)
            QTimer.singleShot(0, lambda: self._finish_bootstrap(None))
            return
        self._set_logo_rotating(True)
        self._view.set_download_detail("Preparing search models (Balanced)…")
        self._view.set_progress_percent(_DOWNLOAD_DONE_PERCENT)
        self._embedder_outcome = None
        self._embedder_thread = threading.Thread(
            target=self._embedder_thread_main,
            name="QubeSplashEmbedder",
            daemon=True,
        )
        self._embedder_thread.start()
        if not self._embedder_poll.isActive():
            self._embedder_poll.start()

    def _embedder_thread_main(self) -> None:
        try:
            embedder = self._load_embedder_worker()
            self._embedder_outcome = (True, embedder)
        except Exception as exc:
            self._embedder_outcome = (False, exc)

    def _poll_background_threads(self) -> None:
        download_thread = self._download_thread
        if download_thread is not None and download_thread.is_alive():
            return
        if download_thread is not None:
            self._download_thread = None
            outcome = self._download_outcome
            self._download_outcome = None
            if outcome is None:
                self._bootstrap_running = False
                logger.error("Download thread exited without a result.")
                return
            ok, payload = outcome
            if not ok:
                logger.error("Bootstrap downloads failed: %s", payload)
                if isinstance(payload, list):
                    self._view.set_download_detail(
                        "Some downloads failed — continuing with available models.\n"
                        + "\n".join(str(item) for item in payload)
                    )
                else:
                    self._view.set_download_detail(
                        f"Download failed — continuing with available models.\n{payload}"
                    )
            self._view.set_progress_percent(_DOWNLOAD_DONE_PERCENT)
            QTimer.singleShot(0, self._begin_embedder_load)
            return

        embedder_thread = self._embedder_thread
        if embedder_thread is not None and embedder_thread.is_alive():
            return
        if embedder_thread is None:
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
        self._view.complete_step(0)
        self._view.set_progress_percent(_EMBEDDER_DONE_PERCENT)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        logger.info("Splash bootstrap started (phased).")
        self._phased_runner = fn(
            embedder=embedder,
            on_phase=self._on_phase,
            on_complete=self._on_phased_bootstrap_complete,
            on_failed=self._on_phased_bootstrap_failed,
        )

    def _on_phased_bootstrap_failed(self, exc: BaseException) -> None:
        self._bootstrap_running = False
        self._stop_spinner()
        self._view.set_download_detail(f"Startup failed:\n{exc}")
        logger.error("Splash bootstrap failed: %s", exc)
        PrestigeDialog(
            self._shell,
            "Qube could not start",
            str(exc),
            is_dark=True,
            tone="danger",
            show_cancel=False,
            confirm_text="OK",
        ).exec()
        app = QApplication.instance()
        if app is not None:
            app.quit()
        else:
            sys.exit(1)

    def _on_phased_bootstrap_complete(self, qube: object) -> None:
        self._bootstrap_running = False
        self._bootstrap_result = qube
        self._view.complete_step(7)
        self._view.set_progress_percent(100)
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
        on_failed: Callable[[BaseException], None] | None = None,
        theme_manager=None,
    ) -> None:
        app = QApplication.instance()
        super().__init__(app if isinstance(app, QObject) else None)
        self._embedder = embedder
        self._enable_routing = enable_routing_debug_tool
        self._enable_trace_diff = enable_trace_diff_debug_tool
        self._on_phase = on_phase
        self._on_complete = on_complete
        self._on_failed = on_failed
        self._theme_manager = theme_manager
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
        except Exception as exc:
            logger.exception("Phased Qube bootstrap failed at phase %d.", self._phase)
            if self._on_failed is not None:
                self._on_failed(exc)
                return
            raise
        self._phase += 1
        QTimer.singleShot(0, self._run_next)

    def _run_phase(self, phase: int) -> None:
        from main import Qube

        noop_tick: Callable[[str], None] = lambda _msg: None
        if self._qube is None:
            self._qube = Qube.__new__(Qube)
            self._qube._theme_manager = self._theme_manager  # type: ignore[attr-defined]
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
    on_failed: Callable[[BaseException], None] | None = None,
    theme_manager=None,
) -> _PhasedQubeRunner:
    """Build ``Qube`` in boot phases; ``on_phase(step_index, percent)`` before each."""
    runner = _PhasedQubeRunner(
        embedder=embedder,
        enable_routing_debug_tool=enable_routing_debug_tool,
        enable_trace_diff_debug_tool=enable_trace_diff_debug_tool,
        on_phase=on_phase,
        on_complete=on_complete,
        on_failed=on_failed,
        theme_manager=theme_manager,
    )
    runner.start()
    return runner


def bootstrap_with_splash(
    *,
    repo_root: Path,
    build_app_fn: SplashBuildCallback,
    on_ready: Callable[[Any], None],
    selected_models: set[BootstrapModelId] | None = None,
    needs_consent: bool = False,
    mock_downloads: bool = False,
    early_splash: Any | None = None,
) -> StartupSplashController:
    """Present splash; first-run uses split consent + branded left pane."""
    if early_splash is not None and hasattr(early_splash, "dismiss"):
        early_splash.dismiss()
    splash = StartupSplashController(
        repo_root=repo_root,
        compact=True,
        selected_models=selected_models,
        needs_consent=needs_consent,
        mock_downloads=mock_downloads,
    )
    splash.present()
    splash.run_bootstrap(build_app_fn, on_ready=on_ready)
    return splash
