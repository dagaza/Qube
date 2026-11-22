"""Drag-and-drop targets for conversation-first library document upload."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from core.library_ingest_ui_flow import filter_library_ingest_paths, paths_from_drop_urls
from core.theme.view_theme import view_resolved_theme


class ConversationDocumentDropFilter(QObject):
    """Accepts local file drops on transcript/composer surfaces."""

    def __init__(
        self,
        *,
        on_files: Callable[[list[Path]], None],
        on_drag_active: Callable[[], None] | None = None,
        on_drag_end: Callable[[], None] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_files = on_files
        self._on_drag_active = on_drag_active
        self._on_drag_end = on_drag_end
        self._drag_depth = 0

    def _paths_from_event(self, event) -> list[Path]:
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return []
        return paths_from_drop_urls(mime.urls())

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        et = event.type()
        if et == QEvent.Type.DragEnter:
            paths = self._paths_from_event(event)
            if not paths:
                return False
            event.acceptProposedAction()
            self._drag_depth += 1
            if self._drag_depth == 1 and self._on_drag_active:
                self._on_drag_active()
            return True
        if et == QEvent.Type.DragMove:
            paths = self._paths_from_event(event)
            if not paths:
                return False
            event.acceptProposedAction()
            return True
        if et == QEvent.Type.DragLeave:
            self._drag_depth = max(0, self._drag_depth - 1)
            if self._drag_depth == 0 and self._on_drag_end:
                self._on_drag_end()
            return False
        if et == QEvent.Type.Drop:
            paths = self._paths_from_event(event)
            self._drag_depth = 0
            if self._on_drag_end:
                self._on_drag_end()
            if paths:
                event.acceptProposedAction()
                self._on_files(filter_library_ingest_paths(paths))
                return True
            return False
        return False


def install_conversation_document_drop(
    widget: QWidget,
    *,
    on_files: Callable[[list[Path]], None],
    on_drag_active: Callable[[], None] | None = None,
    on_drag_end: Callable[[], None] | None = None,
) -> ConversationDocumentDropFilter:
    widget.setAcceptDrops(True)
    filt = ConversationDocumentDropFilter(
        on_files=on_files,
        on_drag_active=on_drag_active,
        on_drag_end=on_drag_end,
        parent=widget,
    )
    widget.installEventFilter(filt)
    return filt


class ConversationDocumentDropOverlay(QFrame):
    """Visual hint while dragging files over the chat mainstage."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("ConversationDocumentDropOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        self._label = QLabel("Drop to add to Library and attach")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        layout.addWidget(self._label)
        layout.addStretch(1)

    def apply_theme(self, is_dark: bool) -> None:
        theme = view_resolved_theme(self, is_dark=is_dark)
        border = theme.accent
        bg = theme.surface_elevated
        self.setStyleSheet(
            f"""
            QFrame#ConversationDocumentDropOverlay {{
                background-color: {bg};
                border: 2px dashed {border};
                border-radius: 12px;
            }}
            QLabel {{
                color: {theme.text_primary};
                font-size: 15px;
                font-weight: 600;
            }}
            """
        )

    def sync_geometry(self, host: QWidget) -> None:
        if host is None:
            return
        self.setGeometry(host.rect())