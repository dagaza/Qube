"""Context chip strip for the chat composer and user message bubbles."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRect, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import qtawesome as qta

from core.composer_draft import (
    ComposerDraft,
    routing_chip_icon,
    routing_chip_tooltip,
    skill_chip_tooltip,
)

# Match PrestigeToggle active track + telemetry accent (emerald-500 family).
_EMERALD = "#10b981"
_EMERALD_MUTED = "#34d399"
_EMERALD_DEEP = "#059669"


class _FlowLayout(QLayout):
    """Minimal flow layout for wrapping chip rows on narrow widths."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientations:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize(0, 0)
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        m = self.contentsMargins()
        start_x = rect.x() + m.left()
        x = start_x
        y = rect.y() + m.top()
        max_right = rect.right() - m.right()
        line_items: list[tuple[QLayoutItem, QSize]] = []
        line_h = 0

        def flush_line() -> None:
            nonlocal x, y, line_h, line_items
            if not line_items:
                return
            cx = start_x
            for it, sz in line_items:
                if not test_only:
                    dy = (line_h - sz.height()) // 2
                    it.setGeometry(QRect(cx, y + dy, sz.width(), sz.height()))
                cx += sz.width() + self.spacing()
            y += line_h + self.spacing()
            line_h = 0
            line_items.clear()

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width()
            if line_items and next_x > max_right:
                flush_line()
                next_x = x + hint.width()
            line_items.append((item, hint))
            line_h = max(line_h, hint.height())
            x = next_x + self.spacing()

        flush_line()
        return y + line_h - rect.y() + m.bottom()


class _ComposerContextChip(QFrame):
    """Single routing or skill chip."""

    remove_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        label: str,
        icon_name: str,
        tooltip: str,
        chip_role: str,
        is_primary: bool,
        editable: bool,
        is_dark: bool,
        compact: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ComposerContextChip")
        self.setProperty("chipRole", chip_role)
        self.setProperty("chipPrimary", "true" if is_primary else "false")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setToolTip(tooltip)

        layout = QHBoxLayout(self)
        pad_v = 2 if compact else 4
        pad_h = 6 if compact else 8
        layout.setContentsMargins(pad_h, pad_v, pad_h, pad_v)
        layout.setSpacing(6)

        icon_lbl = QLabel()
        icon_lbl.setObjectName("ComposerContextChipIcon")
        icon_lbl.setPixmap(
            qta.icon(icon_name, color=self._icon_color(is_dark, chip_role)).pixmap(12, 12)
        )
        icon_lbl.setFixedSize(12, 12)
        layout.addWidget(icon_lbl)

        text_lbl = QLabel(label)
        text_lbl.setObjectName("ComposerContextChipLabel")
        text_lbl.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        layout.addWidget(text_lbl)

        self._remove_btn: QPushButton | None = None
        if editable:
            btn = QPushButton()
            btn.setObjectName("ComposerContextChipRemove")
            btn.setFixedSize(16, 16)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip("Remove")
            btn.setIcon(qta.icon("fa5s.times", color=self._muted_fg(is_dark)))
            btn.setIconSize(QSize(10, 10))
            btn.clicked.connect(self.remove_clicked.emit)
            layout.addWidget(btn)
            self._remove_btn = btn

        self._icon = icon_lbl
        self._chip_role = chip_role
        self._is_primary = is_primary
        self._compact = compact
        self._is_dark = is_dark
        self._icon_name = icon_name
        self.apply_theme(is_dark)

    @staticmethod
    def _icon_color(is_dark: bool, chip_role: str) -> str:
        if chip_role == "skill":
            return "#c4b5fd" if is_dark else "#6d28d9"
        return _EMERALD_MUTED if is_dark else _EMERALD

    @staticmethod
    def _muted_fg(is_dark: bool) -> str:
        return "#a6adc8" if is_dark else "#64748b"

    def _routing_chip_colors(self, is_dark: bool) -> tuple[str, str, str]:
        """Return (background, border, foreground) for knowledge/routing chips."""
        strong = self._is_primary or (self._compact and self._chip_role == "routing")
        if is_dark:
            if strong:
                return (
                    "rgba(16, 185, 129, 0.55)",
                    _EMERALD,
                    "#ecfdf5",
                )
            return (
                "rgba(16, 185, 129, 0.20)",
                "rgba(16, 185, 129, 0.48)",
                "#a7f3d0",
            )
        if strong:
            return (
                "rgba(16, 185, 129, 0.18)",
                _EMERALD,
                _EMERALD_DEEP,
            )
        return (
            "rgba(16, 185, 129, 0.08)",
            "rgba(16, 185, 129, 0.38)",
            "#047857",
        )

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        icon_color = self._icon_color(is_dark, self._chip_role)
        self._icon.setPixmap(qta.icon(self._icon_name, color=icon_color).pixmap(12, 12))

        if self._chip_role == "skill":
            bg = "rgba(139, 92, 246, 0.22)" if is_dark else "rgba(139, 92, 246, 0.12)"
            border = "rgba(139, 92, 246, 0.55)" if is_dark else "rgba(109, 40, 217, 0.45)"
            fg = "#e9d5ff" if is_dark else "#5b21b6"
        elif self._chip_role == "routing":
            bg, border, fg = self._routing_chip_colors(is_dark)
        else:
            bg = "rgba(255, 255, 255, 0.05)" if is_dark else "rgba(0, 0, 0, 0.03)"
            border = "rgba(148, 163, 184, 0.55)" if is_dark else "rgba(148, 163, 184, 0.55)"
            fg = "#bac2de" if is_dark else "#475569"

        radius = 8 if self._compact else 10
        self.setStyleSheet(
            f"""
            QFrame#ComposerContextChip {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: {radius}px;
            }}
            QLabel#ComposerContextChipLabel {{
                color: {fg};
                background: transparent;
                border: none;
                font-size: {"10px" if self._compact else "11px"};
                font-weight: 600;
                padding: 0px;
            }}
            QLabel#ComposerContextChipIcon {{
                background: transparent;
                border: none;
                padding: 0px;
            }}
            QPushButton#ComposerContextChipRemove {{
                background: transparent;
                border: none;
                padding: 0px;
            }}
            QPushButton#ComposerContextChipRemove:hover {{
                background-color: {"rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.06)"};
                border-radius: 8px;
            }}
            """
        )
        if self._remove_btn is not None:
            self._remove_btn.setIcon(
                qta.icon("fa5s.times", color=self._muted_fg(is_dark))
            )


def _elide_label(text: str, *, max_len: int = 28) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


class ComposerContextChipStrip(QWidget):
    """Draft or transcript chip row for routing attachments and skills."""

    routing_removed = pyqtSignal(int)
    skill_removed = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ComposerContextChipStrip")
        self._is_dark = True
        self._editable = True
        self._compact = False
        self._draft = ComposerDraft()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self._routing_host = QWidget()
        self._routing_flow = _FlowLayout(self._routing_host, spacing=6)
        self._routing_host.setLayout(self._routing_flow)

        self._skills_host = QWidget()
        self._skills_flow = _FlowLayout(self._skills_host, spacing=6)
        self._skills_host.setLayout(self._skills_flow)

        root.addWidget(self._routing_host)
        root.addWidget(self._skills_host)
        self.setVisible(False)

    def set_draft(
        self,
        draft: ComposerDraft,
        *,
        editable: bool | None = None,
        compact: bool | None = None,
    ) -> None:
        if editable is not None:
            self._editable = editable
        if compact is not None:
            self._compact = compact
        self._draft = draft.clone()
        self._rebuild()

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        self._rebuild()

    def has_chips(self) -> bool:
        return bool(self._draft.routing or self._draft.skills)

    def _clear_layout(self, layout: _FlowLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _rebuild(self) -> None:
        self._clear_layout(self._routing_flow)
        self._clear_layout(self._skills_flow)

        for idx, att in enumerate(self._draft.routing):
            chip = _ComposerContextChip(
                label=_elide_label(att.label, max_len=24 if self._compact else 28),
                icon_name=routing_chip_icon(att),
                tooltip=routing_chip_tooltip(att, is_primary=(idx == 0)),
                chip_role="routing",
                is_primary=(idx == 0),
                editable=self._editable,
                is_dark=self._is_dark,
                compact=self._compact,
            )
            if self._editable:
                chip.remove_clicked.connect(
                    lambda _checked=False, i=idx: self.routing_removed.emit(i)
                )
            self._routing_flow.addWidget(chip)

        for idx, skill in enumerate(self._draft.skills):
            chip = _ComposerContextChip(
                label=_elide_label(skill.label, max_len=24 if self._compact else 28),
                icon_name="fa5s.brain",
                tooltip=skill_chip_tooltip(skill),
                chip_role="skill",
                is_primary=False,
                editable=self._editable,
                is_dark=self._is_dark,
                compact=self._compact,
            )
            if self._editable:
                chip.remove_clicked.connect(
                    lambda _checked=False, i=idx: self.skill_removed.emit(i)
                )
            self._skills_flow.addWidget(chip)

        has_routing = bool(self._draft.routing)
        has_skills = bool(self._draft.skills)
        self._routing_host.setVisible(has_routing)
        self._skills_host.setVisible(has_skills)
        self.setVisible(has_routing or has_skills)
        self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for host in (self._routing_host, self._skills_host):
            host.updateGeometry()
