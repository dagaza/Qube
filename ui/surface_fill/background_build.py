"""Off-thread wallpaper cache composition for transcript hosts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from core.surface_fill.models import SurfaceProfile
from core.surface_fill.renderer import SurfaceFillRenderer
from core.theme.tokens import ResolvedTheme


@dataclass(frozen=True)
class BackgroundBuildRequest:
    """Snapshot of inputs needed to compose a host background off the GUI thread."""

    generation: int
    width: int
    height: int
    cache_key: tuple[Any, ...]
    theme: ResolvedTheme
    profile: SurfaceProfile
    resolved_wallpaper: Any
    overlay_boost: int
    suppressed: bool


class _BackgroundBuildRunnable(QRunnable):
    def __init__(
        self,
        *,
        bridge: "BackgroundBuildBridge",
        request: BackgroundBuildRequest,
        renderer: SurfaceFillRenderer,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._bridge = bridge
        self._request = request
        self._renderer = renderer

    def run(self) -> None:
        req = self._request
        from PyQt6.QtCore import QSize

        image = self._renderer.render_to_image(
            QSize(req.width, req.height),
            req.profile,
            theme=req.theme,
            overlay_boost=req.overlay_boost,
            suppressed=req.suppressed,
            resolved_wallpaper=req.resolved_wallpaper,
        )
        self._bridge.build_finished.emit(req.generation, image, req.cache_key)


class BackgroundBuildBridge(QObject):
    """Delivers completed QImages back on the GUI thread via queued signal delivery."""

    build_finished = pyqtSignal(int, object, object)

    def start(
        self,
        request: BackgroundBuildRequest,
        *,
        renderer: SurfaceFillRenderer,
    ) -> None:
        QThreadPool.globalInstance().start(
            _BackgroundBuildRunnable(
                bridge=self,
                request=request,
                renderer=renderer,
            )
        )