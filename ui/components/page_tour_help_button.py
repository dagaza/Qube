"""Consistent ? button to launch a page guided tour."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import QPushButton
import qtawesome as qta

from core.theme.accessors import theme_for
from core.theme.widget_styles import MUTED_ICON
from ui.components.ghost_icon_button import apply_ghost_icon_button_style


class PageTourHelpButton(QPushButton):
    """Icon button that requests a registered page tour by id."""

    tour_requested = pyqtSignal(str)

    def __init__(
        self,
        tour_id: str,
        *,
        area_display_name: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._tour_id = tour_id
        self._area_display_name = area_display_name or tour_id
        self.setObjectName("PageTourHelpButton")
        self.setProperty("class", "IconButton")
        self.setFixedSize(28, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Start guided tour")
        self.setIconSize(QSize(16, 16))
        self.clicked.connect(self._on_clicked)
        self.apply_theme()

    @property
    def tour_id(self) -> str:
        return self._tour_id

    @property
    def area_display_name(self) -> str:
        return self._area_display_name

    def set_tour(self, tour_id: str, *, area_display_name: str | None = None) -> None:
        self._tour_id = tour_id
        if area_display_name is not None:
            self._area_display_name = area_display_name

    def apply_theme(self, is_dark: bool | None = None) -> None:
        window = self.window()
        if is_dark is None and window is not None and hasattr(window, "_is_dark_theme"):
            is_dark = bool(window._is_dark_theme)
        theme = theme_for(is_dark=bool(is_dark if is_dark is not None else True))
        apply_ghost_icon_button_style(self, theme)
        self.setIcon(
            qta.icon("fa5s.question-circle", color=theme.color(MUTED_ICON))
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        window = self.window()
        if window is not None and hasattr(window, "_is_dark_theme"):
            self.apply_theme(bool(window._is_dark_theme))

    def _on_clicked(self) -> None:
        win = self.window()
        if win is not None and hasattr(win, "request_page_tour"):
            win.request_page_tour(self._tour_id, area_display_name=self._area_display_name)
            return
        self.tour_requested.emit(self._tour_id)
