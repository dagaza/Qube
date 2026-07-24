"""Apply resolved themes to the running application."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from core.paths import resource_path
from core.theme.feature_flags import is_generated_theme_enabled
from core.theme.tokens import ResolvedTheme, ThemeMode

if TYPE_CHECKING:
    from core.theme_toggle_profile import ThemeToggleProfiler

logger = logging.getLogger("Qube.ThemeApplicator")


class ThemeApplicator:
    """Apply ``ResolvedTheme`` to ``QApplication`` (rendered or static QSS)."""

    def __init__(
        self,
        *,
        use_generated_stylesheet: bool | None = None,
        main_window_ref: Callable[[], object | None] | None = None,
    ) -> None:
        self._use_generated = (
            use_generated_stylesheet
            if use_generated_stylesheet is not None
            else is_generated_theme_enabled()
        )
        self._main_window_ref = main_window_ref
        self._last_applied: ResolvedTheme | None = None

    @property
    def last_applied(self) -> ResolvedTheme | None:
        return self._last_applied

    def apply(
        self,
        resolved: ResolvedTheme,
        *,
        profiler: ThemeToggleProfiler | None = None,
    ) -> None:
        """Record and push stylesheet to ``QApplication`` (no caching)."""
        self._last_applied = resolved
        step = profiler.step if profiler is not None else lambda _name: nullcontext()
        if self._use_generated:
            from core.theme.stylesheet import render_stylesheet

            with step("qss_load"):
                qss = render_stylesheet(resolved)
        else:
            with step("qss_load"):
                qss = self._load_static_stylesheet(resolved.mode)
        if qss is not None:
            with step("qss_apply"):
                self._set_app_stylesheet(qss)

    def _static_qss_path(self, mode: ThemeMode) -> Path:
        filename = "base.qss" if mode.is_dark else "light.qss"
        return resource_path("assets", "styles", filename)

    def _load_static_stylesheet(self, mode: ThemeMode) -> str | None:
        path = self._static_qss_path(mode)
        if not path.is_file():
            logger.warning("Static theme stylesheet not found: %s", path)
            return None
        return path.read_text(encoding="utf-8")

    def _set_app_stylesheet(self, qss: str) -> None:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            logger.debug("ThemeApplicator skipped QSS apply — no QApplication")
            return
        app.setStyleSheet(qss)
