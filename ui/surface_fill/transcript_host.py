"""Transcript region host that paints wallpaper + overlay behind content."""

from __future__ import annotations

import json
from typing import Any

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from core.surface_fill.constants import SURFACE_CHAT_TRANSCRIPT
from core.surface_fill.models import SurfaceProfile
from core.surface_fill.renderer import SurfaceFillRenderer
from core.surface_fill.serialization import surface_profile_to_dict, wallpaper_to_dict
from core.theme.tokens import ResolvedTheme
from core.theme.view_theme import view_resolved_theme
from core.ui_perf_probe import ui_perf_count, ui_perf_enabled, ui_perf_measure
from ui.surface_fill.background_build import BackgroundBuildBridge, BackgroundBuildRequest


class TranscriptWallpaperHost(QWidget):
    """Hosts transcript content with a composited wallpaper background."""

    def __init__(
        self,
        surface_id: str,
        content: QWidget,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._surface_id = surface_id
        self._content = content
        self._renderer = SurfaceFillRenderer()
        self._overlay_boost = 0
        self._suppressed = False
        self._preview_profile: SurfaceProfile | None = None
        self._preview_resolved_wallpaper = None
        self._preview_theme: ResolvedTheme | None = None
        self._refresh_registered = False
        self._surface_refresh_callback = None
        self._surface_refresh_manager = None
        self._background_cache: QPixmap | None = None
        self._background_cache_key: tuple[Any, ...] | None = None
        self._cache_build_pending = False
        self._resize_debounce = QTimer(self)
        self._resize_debounce.setSingleShot(True)
        self._resize_debounce.setInterval(50)
        self._resize_debounce.timeout.connect(self._on_resize_debounced)
        self._expensive_resize_paused = False
        self._pending_resize_work = False
        self._background_build_generation = 0
        self._async_build_bridge = BackgroundBuildBridge(self)
        self._async_build_bridge.build_finished.connect(self._on_async_background_built)

        self.setObjectName("TranscriptWallpaperHost")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(content)

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self._ensure_transparent_content(content)

    @property
    def surface_id(self) -> str:
        return self._surface_id

    def set_expensive_resize_paused(self, paused: bool) -> None:
        paused = bool(paused)
        if paused == self._expensive_resize_paused:
            return
        self._expensive_resize_paused = paused
        if paused:
            self._resize_debounce.stop()
            self._background_build_generation += 1
            self._cache_build_pending = False
            if not self._background_cache_is_current():
                self._pending_resize_work = True

    def flush_pending_resize_work(self) -> None:
        if not self._pending_resize_work and self._background_cache_is_current():
            return
        self._pending_resize_work = False
        if self.width() <= 0 or self.height() <= 0:
            return
        self._resize_debounce.stop()
        self._rebuild_background_cache_for_size_change()
        self.update()

    def set_overlay_boost(self, boost: int) -> None:
        value = max(0, int(boost))
        if value == self._overlay_boost:
            return
        self._overlay_boost = value
        self._invalidate_background_cache()
        self.update()

    def set_suppressed(self, suppressed: bool) -> None:
        value = bool(suppressed)
        if value == self._suppressed:
            return
        self._suppressed = value
        self._invalidate_background_cache()
        self.update()

    def set_preview_profile(
        self,
        profile: SurfaceProfile | None,
        *,
        resolved_wallpaper=None,
        theme: ResolvedTheme | None = None,
    ) -> None:
        """Override manager-backed profile (Settings preview only)."""
        self._preview_profile = profile
        self._preview_resolved_wallpaper = resolved_wallpaper
        self._preview_theme = theme
        self._invalidate_background_cache()
        self.update()

    def refresh_surface_fill(self) -> None:
        self._invalidate_background_cache()
        self.update()
        self._schedule_background_cache_build()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._register_theme_refresh()
        self._schedule_background_cache_build()

    def _resize_builds_blocked(self) -> bool:
        return bool(self._expensive_resize_paused)

    def _schedule_background_cache_build(self) -> None:
        if self._resize_builds_blocked():
            self._pending_resize_work = True
            self._cache_build_pending = False
            return
        if self._cache_build_pending:
            return
        if self._background_cache_is_current():
            return
        self._cache_build_pending = True
        QTimer.singleShot(0, self._build_background_cache_deferred)

    def _build_background_cache_deferred(self) -> None:
        if self._resize_builds_blocked():
            self._cache_build_pending = False
            self._pending_resize_work = True
            return
        if self.width() <= 0 or self.height() <= 0:
            self._cache_build_pending = False
            return
        if self._background_cache_is_current():
            self._cache_build_pending = False
            return

        context = self._paint_context()
        if context is None:
            self._cache_build_pending = False
            return
        theme, profile, resolved_wallpaper = context
        cache_key = self._background_cache_key_for(
            theme, profile, resolved_wallpaper
        )
        if (
            self._background_cache is not None
            and not self._background_cache.isNull()
            and cache_key == self._background_cache_key
        ):
            self._cache_build_pending = False
            return

        self._background_build_generation += 1
        generation = self._background_build_generation
        ui_perf_count("wallpaper.background_cache_build_async")
        request = BackgroundBuildRequest(
            generation=generation,
            width=self.width(),
            height=self.height(),
            cache_key=cache_key,
            theme=theme,
            profile=profile,
            resolved_wallpaper=resolved_wallpaper,
            overlay_boost=self._overlay_boost,
            suppressed=self._suppressed,
        )
        self._async_build_bridge.start(request, renderer=self._renderer)

    def _on_async_background_built(
        self,
        generation: int,
        image: QImage,
        cache_key: tuple[Any, ...],
    ) -> None:
        self._cache_build_pending = False
        if self._resize_builds_blocked():
            self._pending_resize_work = True
            ui_perf_count("wallpaper.background_cache_build_deferred_apply")
            return
        if generation != self._background_build_generation:
            ui_perf_count("wallpaper.background_cache_build_stale_drop")
            return
        if image is None or image.isNull():
            return
        if self.width() <= 0 or self.height() <= 0:
            return

        context = self._paint_context()
        if context is None:
            return
        theme, profile, resolved_wallpaper = context
        expected_key = self._background_cache_key_for(
            theme, profile, resolved_wallpaper
        )
        if cache_key != expected_key:
            ui_perf_count("wallpaper.background_cache_build_stale_drop")
            return
        if image.width() != self.width() or image.height() != self.height():
            ui_perf_count("wallpaper.background_cache_build_stale_drop")
            self._schedule_background_cache_build()
            return

        with ui_perf_measure("wallpaper.background_pixmap_apply"):
            pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            return
        self._background_cache = pixmap
        self._background_cache_key = cache_key
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if event.size() == event.oldSize():
            return
        ui_perf_count("wallpaper.resize")
        if self._resize_builds_blocked():
            self._pending_resize_work = True
            self.update()
            return
        self._resize_debounce.start()
        self.update()

    def _on_resize_debounced(self) -> None:
        if self._background_cache_key is not None:
            cached_w, cached_h = self._background_cache_key[0], self._background_cache_key[1]
            if cached_w == self.width() and cached_h == self.height():
                return
        ui_perf_count("wallpaper.debounced_rebuild_schedule")
        self._rebuild_background_cache_for_size_change()

    def _rebuild_background_cache_for_size_change(self) -> None:
        """Schedule a new cache at the current size without clearing the stale pixmap."""
        if self._resize_builds_blocked():
            self._pending_resize_work = True
            return
        if self._background_cache_is_current():
            return
        with ui_perf_measure("wallpaper.debounced_rebuild_schedule"):
            self._background_build_generation += 1
            self._cache_build_pending = False
            self._schedule_background_cache_build()

    def paintEvent(self, event) -> None:
        perf = ui_perf_enabled()
        if perf:
            ui_perf_count("wallpaper.paint")
        resizing = self._resize_builds_blocked()
        painter = QPainter(self)
        try:
            if resizing:
                painter.setRenderHint(
                    QPainter.RenderHint.SmoothPixmapTransform, False
                )
            pixmap = self._background_cache
            if pixmap is not None and not pixmap.isNull():
                if pixmap.width() == self.width() and pixmap.height() == self.height():
                    if perf:
                        with ui_perf_measure("wallpaper.paint_cached_blit"):
                            painter.drawPixmap(0, 0, pixmap)
                    else:
                        painter.drawPixmap(0, 0, pixmap)
                    return
                if perf:
                    ui_perf_count("wallpaper.paint_stale_stretch")
                if perf:
                    with ui_perf_measure("wallpaper.paint_stale_stretch"):
                        painter.drawPixmap(self.rect(), pixmap)
                else:
                    painter.drawPixmap(self.rect(), pixmap)
                if resizing:
                    self._pending_resize_work = True
                else:
                    self._schedule_background_cache_build()
                return
            if perf:
                ui_perf_count("wallpaper.paint_fast_fallback")
            if perf:
                with ui_perf_measure("wallpaper.paint_fast_fallback"):
                    self._paint_fast_fallback(painter)
            else:
                self._paint_fast_fallback(painter)
            if resizing:
                self._pending_resize_work = True
            else:
                self._schedule_background_cache_build()
        finally:
            painter.end()

    def _paint_fast_fallback(self, painter: QPainter) -> None:
        context = self._paint_context()
        if context is None:
            return
        theme, profile, resolved_wallpaper = context
        self._renderer.paint(
            painter,
            self.rect(),
            profile,
            theme=theme,
            overlay_boost=self._overlay_boost,
            suppressed=True,
            resolved_wallpaper=resolved_wallpaper,
        )

    def _invalidate_background_cache(self) -> None:
        self._background_build_generation += 1
        self._background_cache = None
        self._background_cache_key = None
        self._cache_build_pending = False

    def _background_cache_is_current(self) -> bool:
        pixmap = self._background_cache
        if pixmap is None or pixmap.isNull():
            return False
        if pixmap.width() != self.width() or pixmap.height() != self.height():
            return False
        if self._background_cache_key is None:
            return False
        cached_w, cached_h = self._background_cache_key[0], self._background_cache_key[1]
        return cached_w == self.width() and cached_h == self.height()

    def _background_pixmap(self) -> QPixmap | None:
        """Synchronous cache build (tests and diagnostics)."""
        context = self._paint_context()
        if context is None:
            return None
        theme, profile, resolved_wallpaper = context
        cache_key = self._background_cache_key_for(
            theme, profile, resolved_wallpaper
        )
        if (
            self._background_cache is not None
            and not self._background_cache.isNull()
            and cache_key == self._background_cache_key
        ):
            return self._background_cache

        if self.width() <= 0 or self.height() <= 0:
            return None

        ui_perf_count("wallpaper.background_cache_build_sync")
        with ui_perf_measure("wallpaper.background_pixmap_build"):
            image = self._renderer.render_to_image(
                self.size(),
                profile,
                theme=theme,
                overlay_boost=self._overlay_boost,
                suppressed=self._suppressed,
                resolved_wallpaper=resolved_wallpaper,
            )
            pixmap = QPixmap.fromImage(image)

        self._background_cache = pixmap
        self._background_cache_key = cache_key
        self._cache_build_pending = False
        return pixmap

    def _paint_context(self) -> tuple[ResolvedTheme, SurfaceProfile, Any] | None:
        theme = self._preview_theme or view_resolved_theme(self)
        if self._preview_profile is not None:
            profile = self._preview_profile
            resolved_wallpaper = (
                self._preview_resolved_wallpaper or profile.wallpaper
            )
            return theme, profile, resolved_wallpaper

        manager = self._theme_manager()
        if manager is None:
            return None
        profile = manager.resolved_effective_surface_profile(self._surface_id)
        resolved_wallpaper = profile.wallpaper
        return theme, profile, resolved_wallpaper

    def _background_cache_key_for(
        self,
        theme: ResolvedTheme,
        profile: SurfaceProfile,
        resolved_wallpaper,
    ) -> tuple[Any, ...]:
        return (
            self.width(),
            self.height(),
            self._overlay_boost,
            self._suppressed,
            theme.scheme_id,
            theme.is_dark,
            theme.background,
            json.dumps(surface_profile_to_dict(profile), sort_keys=True),
            json.dumps(wallpaper_to_dict(resolved_wallpaper), sort_keys=True),
        )

    def _register_theme_refresh(self) -> None:
        if self._refresh_registered:
            return
        manager = self._theme_manager()
        if manager is None:
            return
        callback = self.refresh_surface_fill
        manager.register_surface_refresh(callback)
        self._surface_refresh_callback = callback
        self._surface_refresh_manager = manager
        self._refresh_registered = True
        self.destroyed.connect(self._on_destroyed_unregister_surface_refresh)

    def _unregister_theme_refresh(self) -> None:
        if not self._refresh_registered:
            return
        manager = self._surface_refresh_manager
        callback = self._surface_refresh_callback
        if manager is not None and callback is not None:
            manager.unregister_surface_refresh(callback)
        self._refresh_registered = False
        self._surface_refresh_callback = None
        self._surface_refresh_manager = None

    def _on_destroyed_unregister_surface_refresh(self, *_args) -> None:
        self._unregister_theme_refresh()

    def closeEvent(self, event) -> None:
        self._unregister_theme_refresh()
        super().closeEvent(event)

    def _theme_manager(self):
        window = self.window()
        if window is not None and hasattr(window, "theme_manager"):
            return window.theme_manager
        return None

    def _ensure_transparent_content(self, widget: QWidget) -> None:
        widget.setAutoFillBackground(False)
        viewport = getattr(widget, "viewport", None)
        if callable(viewport):
            vp = viewport()
            if vp is not None:
                vp.setAutoFillBackground(False)
                vp.setStyleSheet("background: transparent;")
        name = widget.objectName()
        if name:
            widget.setStyleSheet(f"#{name} {{ background: transparent; border: none; }}")
        else:
            widget.setStyleSheet("background: transparent; border: none;")


def bind_transcript_wallpaper_readability(
    host: TranscriptWallpaperHost | None,
    *,
    high_contrast: bool,
    reader_focus: bool,
) -> None:
    """Sync high-contrast suppression and reader-focus overlay boost."""
    if host is None:
        return
    host.set_suppressed(high_contrast)
    host.set_overlay_boost(0 if high_contrast else (1 if reader_focus else 0))


def chat_transcript_surface_id() -> str:
    return SURFACE_CHAT_TRANSCRIPT