"""Vertical timeline rail for jumping between user prompts in long chat transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter
from PyQt6.QtWidgets import QWidget

from core.theme.accessors import theme_for

_RAIL_WIDTH_PX = 22
TRANSCRIPT_TIMELINE_RAIL_WIDTH_PX = _RAIL_WIDTH_PX
_MARKER_HEIGHT = 2
_MARKER_HEIGHT_ACTIVE = 3
_MARKER_WIDTH_INACTIVE = 10
_MARKER_WIDTH_HOVER = 14
_MARKER_WIDTH_ACTIVE = 20
_MARKER_GAP_MIN = 3
_MARKER_GAP_MAX = 7
_STACK_INSET_PX = 20
_HIT_SLOP_PX = 8
_LABEL_MAX_LEN = 80


def truncate_waypoint_label(text: str, *, max_len: int = _LABEL_MAX_LEN) -> str:
    """Plain-text preview for a user prompt waypoint."""
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1].rstrip() + "…"


def format_waypoint_tooltip(index: int, total: int, label: str) -> str:
    """Plain-text tooltip for a turn marker (shown via setToolTip)."""
    turn_line = f"Turn {int(index) + 1} of {max(1, int(total))}"
    preview = str(label or "").strip()
    if preview:
        return f"{turn_line}\n{preview}"
    return turn_line


def compute_scroll_target_for_waypoint_y(
    waypoint_y: int,
    *,
    margin: int = 24,
    scroll_min: int = 0,
    scroll_max: int = 0,
) -> int:
    """Scroll-bar value that places a waypoint near the top of the viewport."""
    target = int(waypoint_y) - int(margin)
    return max(int(scroll_min), min(int(scroll_max), target))


def transcript_timeline_should_show(
    container_height: int,
    viewport_height: int,
    *,
    waypoint_count: int = 0,
) -> bool:
    """True when the transcript overflows the viewport and has at least one waypoint."""
    if waypoint_count < 1:
        return False
    if viewport_height <= 0 or container_height <= 0:
        return False
    return container_height > viewport_height


def compute_active_waypoint_index(
    scroll_top: int,
    waypoint_ys: Sequence[int],
    *,
    viewport_margin: int = 24,
) -> int:
    """Index of the user turn nearest the top of the visible viewport."""
    if not waypoint_ys:
        return 0
    target = max(0, int(scroll_top) + int(viewport_margin))
    active = 0
    for idx, y in enumerate(waypoint_ys):
        if int(y) <= target:
            active = idx
        else:
            break
    return active


def compute_stacked_marker_centers(
    count: int,
    rail_height: int,
    *,
    marker_height: float = _MARKER_HEIGHT,
    gap_min: float = _MARKER_GAP_MIN,
    gap_max: float = _MARKER_GAP_MAX,
    inset: float = _STACK_INSET_PX,
) -> list[float]:
    """Evenly stack marker center-Y positions in a compact vertical block."""
    if count <= 0:
        return []
    if count == 1:
        return [float(rail_height) / 2.0]

    usable = max(float(marker_height), float(rail_height) - 2.0 * inset)
    gap = float(gap_max)
    total = count * marker_height + (count - 1) * gap
    if total > usable:
        gap = max(
            gap_min,
            (usable - count * marker_height) / max(1, count - 1),
        )
        total = count * marker_height + (count - 1) * gap

    block_top = (float(rail_height) - total) / 2.0
    centers: list[float] = []
    cursor = block_top + marker_height / 2.0
    for _ in range(count):
        centers.append(cursor)
        cursor += marker_height + gap
    return centers


def nearest_waypoint_index_for_y(
    local_y: float,
    marker_center_ys: Sequence[float],
    *,
    hit_slop: float = _HIT_SLOP_PX,
) -> int:
    """Return the closest marker index within hit slop, else -1."""
    if not marker_center_ys:
        return -1
    best_idx = -1
    best_dist = float(hit_slop)
    for idx, center_y in enumerate(marker_center_ys):
        dist = abs(float(local_y) - float(center_y))
        if dist <= best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


@dataclass(frozen=True)
class TranscriptWaypointEntry:
    """Geometry + label for one user-turn marker."""

    y: int
    label: str


class TranscriptTimelineRail(QWidget):
    """Compact stacked turn index — one tick per user prompt, active tick enlarges on scroll."""

    waypoint_clicked = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TranscriptTimelineRail")
        self.setFixedWidth(_RAIL_WIDTH_PX)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._is_dark = True
        self._entries: list[TranscriptWaypointEntry] = []
        self._overflow_visible = False
        self._active_index = 0
        self._hover_index: int | None = None

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        self.update()

    def set_geometry_from_container(
        self,
        entries: list[TranscriptWaypointEntry],
        *,
        container_height: int,
        show: bool,
    ) -> None:
        del container_height  # retained for caller API; layout is index-based, not minimap
        self._entries = list(entries)
        self._overflow_visible = bool(show)
        self.setVisible(show and bool(entries))
        self.update()

    def set_active_index(self, index: int) -> None:
        if not self._entries:
            self._active_index = 0
            return
        clamped = max(0, min(int(index), len(self._entries) - 1))
        if clamped == self._active_index:
            return
        self._active_index = clamped
        self.update()

    def _marker_center_ys(self) -> list[float]:
        return compute_stacked_marker_centers(len(self._entries), self.height())

    def _marker_width(self, idx: int) -> int:
        if idx == self._active_index:
            return _MARKER_WIDTH_ACTIVE
        if idx == self._hover_index:
            return _MARKER_WIDTH_HOVER
        return _MARKER_WIDTH_INACTIVE

    def _marker_height(self, idx: int) -> int:
        return _MARKER_HEIGHT_ACTIVE if idx == self._active_index else _MARKER_HEIGHT

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt API
        if not self._overflow_visible or not self._entries:
            return super().paintEvent(event)

        theme = theme_for(is_dark=self._is_dark)
        inactive = QColor(theme.text_muted)
        inactive.setAlpha(150)
        active = QColor(theme.accent)
        hover = QColor(theme.accent)
        hover.setAlpha(200)

        centers = self._marker_center_ys()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)

        center_x = self.width() / 2.0
        for idx, center_y in enumerate(centers):
            is_active = idx == self._active_index
            is_hover = idx == self._hover_index
            width = self._marker_width(idx)
            height = self._marker_height(idx)
            if is_active:
                color = active
            elif is_hover:
                color = hover
            else:
                color = inactive

            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(
                int(center_x - width / 2),
                int(center_y - height / 2),
                width,
                height,
                height / 2,
                height / 2,
            )

        painter.end()
        super().paintEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt API
        if (
            not self._overflow_visible
            or not self._entries
            or event.button() != Qt.MouseButton.LeftButton
        ):
            return super().mousePressEvent(event)
        idx = nearest_waypoint_index_for_y(
            event.position().y(),
            self._marker_center_ys(),
        )
        if idx >= 0:
            self.waypoint_clicked.emit(idx)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt API
        if not self._overflow_visible or not self._entries:
            return super().mouseMoveEvent(event)
        idx = nearest_waypoint_index_for_y(
            event.position().y(),
            self._marker_center_ys(),
        )
        if idx != self._hover_index:
            self._hover_index = idx if idx >= 0 else None
            if self._hover_index is not None and self._entries:
                tip = format_waypoint_tooltip(
                    self._hover_index,
                    len(self._entries),
                    self._entries[self._hover_index].label,
                )
                self.setToolTip(tip)
            else:
                self.setToolTip("")
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if idx >= 0
                else Qt.CursorShape.ArrowCursor
            )
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 — Qt API
        if self._hover_index is not None:
            self._hover_index = None
            self.setToolTip("")
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
        super().leaveEvent(event)
