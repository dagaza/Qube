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


from core.composer_draft import (
    ComposerDraft,
    routing_chip_icon,
    routing_chip_tooltip,
    routing_chip_unavailable_message,
    skill_chip_tooltip,
)
from core.theme.color_utils import adjust_lightness, with_alpha
from core.theme.tokens import ResolvedTheme
from core.theme.view_theme import view_resolved_theme
from core.theme.svg_icons import themed_fa_icon, themed_fa_pixmap


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
        is_unavailable: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ComposerContextChip")
        self.setProperty("chipRole", chip_role)
        self.setProperty("chipPrimary", "true" if is_primary else "false")
        self.setProperty("chipUnavailable", "true" if is_unavailable else "false")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setToolTip(tooltip)

        layout = QHBoxLayout(self)
        pad_v = 2 if compact else 4
        pad_h = 6 if compact else 8
        layout.setContentsMargins(pad_h, pad_v, pad_h, pad_v)
        layout.setSpacing(6)

        init_theme = view_resolved_theme(parent, is_dark=is_dark)
        icon_lbl = QLabel()
        icon_lbl.setObjectName("ComposerContextChipIcon")
        icon_lbl.setPixmap(
            themed_fa_pixmap(
                icon_name,
                self._icon_color(
                    init_theme,
                    chip_role,
                    is_unavailable=is_unavailable,
                ),
                12,
            )
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
            btn.setIcon(
                themed_fa_icon("fa5s.times", self._muted_fg(init_theme), 10)
            )
            btn.setIconSize(QSize(10, 10))
            btn.clicked.connect(self.remove_clicked.emit)
            layout.addWidget(btn)
            self._remove_btn = btn

        self._icon = icon_lbl
        self._chip_role = chip_role
        self._is_primary = is_primary
        self._compact = compact
        self._is_dark = is_dark
        self._is_unavailable = is_unavailable
        self._icon_name = icon_name
        self.apply_theme(is_dark)

    @staticmethod
    def _icon_color(
        theme: ResolvedTheme,
        chip_role: str,
        *,
        is_unavailable: bool = False,
    ) -> str:
        if is_unavailable:
            return adjust_lightness(theme.warning, 0.12 if theme.is_dark else -0.12)
        if chip_role == "skill":
            return theme.accent if theme.is_dark else adjust_lightness(theme.accent, -0.12)
        return adjust_lightness(theme.success, 0.12 if theme.is_dark else 0.0)

    @staticmethod
    def _muted_fg(theme: ResolvedTheme) -> str:
        return theme.text_muted if theme.is_dark else theme.text_secondary

    def _unavailable_chip_colors(self, theme: ResolvedTheme) -> tuple[str, str, str]:
        warn = theme.warning
        if theme.is_dark:
            return (
                with_alpha(warn, 0.18),
                with_alpha(warn, 0.48),
                adjust_lightness(warn, 0.35),
            )
        return (
            with_alpha(warn, 0.10),
            with_alpha(warn, 0.42),
            adjust_lightness(warn, -0.18),
        )

    def _routing_chip_colors(self, theme: ResolvedTheme) -> tuple[str, str, str]:
        """Return (background, border, foreground) for knowledge/routing chips."""
        if self._is_unavailable:
            return self._unavailable_chip_colors(theme)
        strong = self._is_primary or (self._compact and self._chip_role == "routing")
        success = theme.success
        if theme.is_dark:
            if strong:
                return (
                    with_alpha(success, 0.55),
                    success,
                    theme.text_on_accent,
                )
            return (
                with_alpha(success, 0.20),
                with_alpha(success, 0.48),
                adjust_lightness(success, 0.35),
            )
        if strong:
            return (
                with_alpha(success, 0.18),
                success,
                adjust_lightness(success, -0.18),
            )
        return (
            with_alpha(success, 0.08),
            with_alpha(success, 0.38),
            adjust_lightness(success, -0.22),
        )

    def apply_theme(self, is_dark: bool) -> None:
        theme = view_resolved_theme(self, is_dark=is_dark)
        self._is_dark = theme.is_dark
        icon_color = self._icon_color(
            theme,
            self._chip_role,
            is_unavailable=self._is_unavailable,
        )
        self._icon.setPixmap(themed_fa_pixmap(self._icon_name, icon_color, 12))

        if self._is_unavailable and self._chip_role == "routing":
            bg, border, fg = self._unavailable_chip_colors(theme)
        elif self._chip_role == "skill":
            bg = with_alpha(theme.accent, 0.22 if theme.is_dark else 0.12)
            border = with_alpha(theme.accent, 0.55 if theme.is_dark else 0.45)
            fg = adjust_lightness(theme.accent, 0.35 if theme.is_dark else -0.22)
        elif self._chip_role == "routing":
            bg, border, fg = self._routing_chip_colors(theme)
        else:
            bg = with_alpha(theme.text_primary, 0.05 if theme.is_dark else 0.03)
            border = with_alpha(theme.text_muted, 0.55)
            fg = theme.text_secondary if theme.is_dark else theme.text_muted

        radius = 8 if self._compact else 10
        remove_hover = with_alpha(theme.text_primary, 0.08 if theme.is_dark else 0.06)
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
                background-color: {remove_hover};
                border-radius: 8px;
            }}
            """
        )
        if self._remove_btn is not None:
            self._remove_btn.setIcon(
                themed_fa_icon("fa5s.times", self._muted_fg(theme), 10)
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
        theme = view_resolved_theme(self, is_dark=is_dark)
        self._is_dark = theme.is_dark
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
            unavailable_msg = routing_chip_unavailable_message(att)
            chip = _ComposerContextChip(
                label=_elide_label(att.label, max_len=24 if self._compact else 28),
                icon_name=routing_chip_icon(att),
                tooltip=routing_chip_tooltip(att, is_primary=(idx == 0)),
                chip_role="routing",
                is_primary=(idx == 0),
                editable=self._editable,
                is_dark=self._is_dark,
                compact=self._compact,
                is_unavailable=bool(unavailable_msg),
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
