"""Floating @-mention picker for the chat composer (Files / Conversations / Tools)."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt, QTimer, pyqtSignal, QEvent
from PyQt6.QtGui import QColor, QBrush, QFontMetrics, QKeyEvent, QPalette, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QStyle,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from core.composer_attachments import (
    ComposerAttachment,
    composer_tool_by_id,
    composer_tool_tooltip,
    composer_tools_for_palette,
)
from core.integrations.search import CapabilityPaletteEntry, capability_palette_tooltip
from core.composer_commands import COMPOSER_COMMANDS, ComposerCommand
from core.composer_mention_search import (
    ComposerPaletteView,
    resolve_scoped_filter,
    search_composer_mentions,
    section_label,
)
from core.composer_mention_trigger import root_row_index_for_query
from core.composer_skills import ComposerSkillMention, list_skill_mentions_for_palette
from core.platform.frameless_window import apply_translucent_window_chrome
from core.theme.color_utils import with_alpha
from core.theme.view_theme import view_resolved_theme
from core.theme.svg_icons import themed_fa_pixmap

_ROOT_ROWS = (
    ("file", "Files", "Reference a library document", "fa5s.file-alt"),
    ("conversation", "Conversations", "Reference another chat", "fa5s.comments"),
    ("tool", "Tools", "Internet, library, or memory", "fa5s.tools"),
    ("skill", "Skills", "Reasoning frameworks", "fa5s.brain"),
    ("command", "Commands", "App actions and guidance", "fa5s.terminal"),
)

_ROOT_ROW_TOOLTIPS: dict[str, str] = {
    "file": "Attach an indexed library document. Inserts @[file:filename] and scopes search to that file.",
    "conversation": "Attach another chat's transcript. Inserts @[chat:session-id::Title] for this turn only.",
    "tool": "Choose a routing tool (Internet, Library, or Memory). Inserts @[tool:…] to control how Qube searches.",
    "skill": "Add a reasoning-framework skill. Inserts @[skill:…] as prompt guidance (not routing).",
    "command": "Run an app action immediately. Commands are not sent to the model.",
}

_FILTER_TOOLTIPS: dict[str, str] = {
    "file": "Search indexed library documents by filename. Backspace returns to categories when empty.",
    "conversation": "Search conversations by title. Backspace returns to categories when empty.",
    "tool": "Filter tools by name or description. Backspace returns to categories when empty.",
    "skill": "Search skills by name or description. Backspace returns to categories when empty.",
    "command": "Filter commands by name or description. Backspace returns to categories when empty.",
}

_BACK_LINK_TOOLTIP = "Back to categories (Backspace)"

_ROOT_LIST_TOOLTIP = (
    "Type @ to browse categories, or keep typing to search everything."
)
_SEARCH_LIST_TOOLTIP = (
    "Global search across tools, files, conversations, skills, and commands. "
    "Arrow keys, Enter, or Tab to select."
)
_SCOPED_LIST_TOOLTIP = (
    "Browsing one category. Keep typing to filter within it. "
    "Backspace at empty filter returns to search."
)

_ROOT_ROW_HEIGHT = 56
_DRILL_LIST_HEIGHT = 220
_TYPEAHEAD_RESET_MS = 900


def _root_kind_meta(kind: str) -> tuple[str, str, str]:
    """Return ``(title, subtitle, icon_name)`` for a root category kind."""
    for k, title, subtitle, icon_name in _ROOT_ROWS:
        if k == kind:
            return title, subtitle, icon_name
    return kind.title(), "", "fa5s.circle"


class _ComposerClickableLabel(QLabel):
    """Compact clickable crumb — avoids global QPushButton padding."""

    activated = pyqtSignal()

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def mouseReleaseEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
        ):
            self.activated.emit()
        super().mouseReleaseEvent(event)


class _ComposerContextHeader(QFrame):
    """Eyebrow + breadcrumb trail showing where the user is in the @ palette."""

    root_activated = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("ComposerMentionContextHeader")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 0, 2, 6)
        outer.setSpacing(3)

        self._eyebrow = QLabel("ATTACH")
        self._eyebrow.setObjectName("ComposerMentionEyebrow")
        outer.addWidget(self._eyebrow)

        trail = QHBoxLayout()
        trail.setContentsMargins(0, 0, 0, 0)
        trail.setSpacing(5)

        self._root_link = _ComposerClickableLabel("Categories")
        self._root_link.setObjectName("ComposerMentionCrumbRoot")
        self._root_link.setToolTip(_BACK_LINK_TOOLTIP)
        self._root_link.activated.connect(self.root_activated.emit)

        self._sep = QLabel()
        self._sep.setObjectName("ComposerMentionCrumbSep")
        self._sep.setFixedSize(10, 10)
        self._sep.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._icon = QLabel()
        self._icon.setFixedSize(14, 14)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._current = QLabel("Categories")
        self._current.setObjectName("ComposerMentionCrumbCurrent")

        self._query_hint = QLabel()
        self._query_hint.setObjectName("ComposerMentionCrumbQuery")

        trail.addWidget(self._root_link, 0, Qt.AlignmentFlag.AlignVCenter)
        trail.addWidget(self._sep, 0, Qt.AlignmentFlag.AlignVCenter)
        trail.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignVCenter)
        trail.addWidget(self._current, 0, Qt.AlignmentFlag.AlignVCenter)
        trail.addWidget(self._query_hint, 0, Qt.AlignmentFlag.AlignVCenter)
        trail.addStretch(1)
        outer.addLayout(trail)

        self._accent_color = ""
        self._icon_color = ""
        self._section_kind: str | None = None

    def apply_theme(
        self,
        *,
        is_dark: bool,
        fg: str,
        sub: str,
        accent: str,
        hover_bg: str,
        border: str,
    ) -> None:
        self._accent_color = accent
        self._icon_color = accent
        chip_bg = hover_bg if not is_dark else with_alpha(fg, 0.06)
        chip_border = with_alpha(border, 0.85) if is_dark else border
        self._eyebrow.setStyleSheet(
            f"color: {sub}; font-size: 10px; font-weight: 600; "
            "letter-spacing: 0.06em; background: transparent; border: none; padding: 0px;"
        )
        self._root_link.setStyleSheet(
            f"color: {sub}; font-size: 12px; font-weight: 500; "
            "background: transparent; border: none; padding: 0px;"
        )
        self._current.setStyleSheet(
            f"color: {fg}; font-size: 13px; font-weight: 600; "
            "background: transparent; border: none; padding: 0px;"
        )
        self._query_hint.setStyleSheet(
            f"color: {sub}; font-size: 11px; font-weight: 500; "
            f"background-color: {chip_bg}; border: 1px solid {chip_border}; "
            "border-radius: 5px; padding: 1px 7px; margin-left: 2px;"
        )
        self.setStyleSheet(
            f"""
            QFrame#ComposerMentionContextHeader {{
                background: transparent;
                border: none;
                border-bottom: 1px solid {chip_border};
                margin-bottom: 2px;
                padding-bottom: 2px;
            }}
            QLabel#ComposerMentionCrumbRoot:hover {{
                color: {fg};
            }}
            """
        )
        self._refresh_sep_icon()
        if self._section_kind is not None:
            self._refresh_section_icon(self._section_kind)

    def set_palette_context(
        self,
        *,
        view_mode: ComposerPaletteView,
        scoped_kind: str | None = None,
        query: str = "",
        scoped_filter: str = "",
    ) -> None:
        self._section_kind = scoped_kind
        q = (query or "").strip()
        filt = (scoped_filter or "").strip()

        if view_mode == ComposerPaletteView.SEARCH:
            self._eyebrow.setText("ATTACH")
            self._root_link.hide()
            self._sep.hide()
            self._icon.hide()
            self._current.setText("Search")
            self._current.show()
            if q:
                self._query_hint.setText(self._format_query_chip(q))
                self._query_hint.show()
            else:
                self._query_hint.hide()
            return

        if view_mode == ComposerPaletteView.SCOPED and scoped_kind:
            title, _subtitle, icon_name = _root_kind_meta(scoped_kind)
            self._eyebrow.setText("ATTACH")
            self._root_link.setText("Categories")
            self._root_link.show()
            self._sep.show()
            self._refresh_sep_icon()
            self._icon.show()
            self._refresh_section_icon(scoped_kind, icon_name=icon_name)
            self._current.setText(title)
            self._current.show()
            if filt:
                self._query_hint.setText(self._format_query_chip(filt))
                self._query_hint.show()
            else:
                self._query_hint.hide()
            return

        self._root_link.hide()
        self._sep.hide()
        self._icon.hide()
        self._current.setText("Categories")
        self._current.show()
        self._query_hint.hide()
        self._eyebrow.setText("ATTACH")

    def set_scoped_breadcrumb_tooltip(self, text: str) -> None:
        self._current.setToolTip(text)
        self._icon.setToolTip(text)

    def set_context(self, mode: str | None, *, query: str = "") -> None:
        if mode is None:
            view = ComposerPaletteView.SEARCH if (query or "").strip() else ComposerPaletteView.BROWSE
            self.set_palette_context(view_mode=view, query=query)
        else:
            self.set_palette_context(
                view_mode=ComposerPaletteView.SCOPED,
                scoped_kind=mode,
                query=query,
                scoped_filter=resolve_scoped_filter(mode, query),
            )

    def _format_query_chip(self, query: str) -> str:
        text = query if len(query) <= 24 else f"{query[:21]}…"
        return text

    def _refresh_sep_icon(self) -> None:
        self._sep.setPixmap(
            themed_fa_pixmap("fa5s.chevron-right", self._accent_color, 10)
        )

    def _refresh_section_icon(
        self, kind: str | None, *, icon_name: str | None = None
    ) -> None:
        if kind is None:
            return
        if icon_name is None:
            _title, _sub, icon_name = _root_kind_meta(kind)
        self._icon.setPixmap(
            themed_fa_pixmap(icon_name, self._icon_color, 14)
        )


class _ComposerMentionItemDelegate(QStyledItemDelegate):
    """Paint drill rows entirely ourselves so Qt never draws a second list frame."""

    def __init__(self, popup: "ComposerMentionPopup") -> None:
        super().__init__(popup._list)
        self._popup = popup

    def paint(self, painter, option, index) -> None:
        opt = QStyleOptionViewItem(option)
        opt.state &= ~QStyle.StateFlag.State_HasFocus
        is_drill = self._popup._view_mode != ComposerPaletteView.BROWSE

        if is_drill:
            self._paint_drill_row(painter, opt, index)
            return

        colors = getattr(self._popup, "_theme_colors", None)
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        if colors and (selected or hovered):
            _fg, _border, hover_bg = colors
            rect = opt.rect.adjusted(2, 1, -2, -1)
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            fill = QColor(hover_bg)
            if self._popup._is_dark:
                fill.setAlpha(140)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 8, 8)
            painter.restore()

        opt.state &= ~(
            QStyle.StateFlag.State_Selected | QStyle.StateFlag.State_MouseOver
        )
        opt.backgroundBrush = QBrush(Qt.GlobalColor.transparent)
        super().paint(painter, opt, index)

    def _paint_drill_row(self, painter, opt, index) -> None:
        colors = getattr(self._popup, "_theme_colors", None)
        if colors is None:
            return
        fg, _border, hover_bg = colors
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        enabled = bool(opt.state & QStyle.StateFlag.State_Enabled)
        rect = opt.rect.adjusted(2, 1, -2, -1)

        if enabled and (selected or hovered):
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(hover_bg))
            painter.drawRoundedRect(rect, 8, 8)
            painter.restore()

        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        text_rect = opt.rect.adjusted(12, 0, -10, 0)
        painter.save()
        color = QColor(fg)
        if not enabled:
            color.setAlpha(140)
        painter.setPen(color)
        painter.setFont(opt.font)
        elided = QFontMetrics(opt.font).elidedText(
            text,
            Qt.TextElideMode.ElideRight,
            max(0, text_rect.width()),
        )
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )
        painter.restore()


class ComposerMentionPopup(QWidget):
    """Frameless popup: root categories or searchable drill-down list."""

    item_selected = pyqtSignal(object)  # ComposerAttachment
    skill_selected = pyqtSignal(object)  # ComposerSkillMention
    command_selected = pyqtSignal(object)  # ComposerCommand
    dismissed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            parent,
            Qt.WindowType.Popup
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setObjectName("ComposerMentionPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._is_dark = True
        self._view_mode = ComposerPaletteView.BROWSE
        self._scoped_kind: str | None = None
        self._composer_query = ""
        self._active_session_id: str | None = None
        self._db = None
        self._store = None
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._run_search)
        self._type_buffer = ""
        self._type_reset_timer = QTimer(self)
        self._type_reset_timer.setSingleShot(True)
        self._type_reset_timer.timeout.connect(self._clear_type_buffer)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._shell = QFrame(self)
        self._shell.setObjectName("ComposerMentionShell")
        self._shell.setFrameShape(QFrame.Shape.NoFrame)
        self._shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._shell.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        outer.addWidget(self._shell)

        layout = QVBoxLayout(self._shell)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._layout = layout

        self._context_header = _ComposerContextHeader(self._shell)
        self._context_header.root_activated.connect(self._navigate_to_root)
        layout.addWidget(self._context_header)

        self._filter = QLineEdit()
        self._filter.setObjectName("ComposerMentionFilter")
        self._filter.setPlaceholderText("Filter…")
        self._filter.hide()
        self._filter.textChanged.connect(self._schedule_search)
        layout.addWidget(self._filter)

        self._list = QListWidget()
        self._list.setObjectName("ComposerMentionList")
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setLineWidth(0)
        self._list.setMidLineWidth(0)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._list.setItemDelegate(_ComposerMentionItemDelegate(self))
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.setUniformItemSizes(False)
        self._list.setSpacing(2)
        self._list.setFixedHeight(_DRILL_LIST_HEIGHT)
        self._filter.installEventFilter(self)
        self._list.installEventFilter(self)
        layout.addWidget(self._list)

        self.setToolTip(_ROOT_LIST_TOOLTIP)
        self._shell.setToolTip(_ROOT_LIST_TOOLTIP)
        self._list.setToolTip(_ROOT_LIST_TOOLTIP)
        self._filter.setToolTip("")

        self._search_debounce_ms = 280
        self._anchor_global_pos: QPoint | None = None
        self._window_margin = 8
        self.apply_theme(view_resolved_theme(self).is_dark)

    def set_context(
        self,
        *,
        db,
        store=None,
        active_session_id: str | None = None,
    ) -> None:
        self._db = db
        self._store = store
        self._active_session_id = active_session_id

    def apply_theme(self, is_dark: bool) -> None:
        theme = view_resolved_theme(self, is_dark=is_dark)
        self._theme = theme
        self._is_dark = theme.is_dark
        is_dark = theme.is_dark
        bg = theme.background
        fg = theme.text_primary
        border = theme.border_subtle if is_dark else theme.border
        hover = theme.surface_hover if is_dark else theme.surface
        sub = theme.text_muted if is_dark else theme.text_secondary

        palette = QPalette()
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
            palette.setColor(role, QColor(bg))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(fg))
        palette.setColor(QPalette.ColorRole.Text, QColor(fg))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(hover))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(fg))
        self.setPalette(palette)
        self._shell.setPalette(palette)
        self._filter.setPalette(palette)
        self._list.setPalette(palette)

        chevron = theme.accent
        self._context_header.apply_theme(
            is_dark=is_dark,
            fg=fg,
            sub=sub,
            accent=chevron,
            hover_bg=hover,
            border=border,
        )
        shell_ss = f"""
            QFrame#ComposerMentionShell {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 12px;
            }}
        """
        filter_ss = f"""
            QLineEdit#ComposerMentionFilter {{
                background-color: {hover};
                color: {fg};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 6px 10px;
            }}
        """
        self._theme_colors = (fg, border, hover)
        self.setStyleSheet("background: transparent; border: none;")
        self._shell.setStyleSheet(shell_ss)
        self._filter.setStyleSheet(filter_ss)
        self._apply_list_stylesheet()
        self._sync_context_header()
        self._rebuild_visible_list()

    def _sync_context_header(self) -> None:
        scoped_filter = ""
        if self._view_mode == ComposerPaletteView.SCOPED and self._scoped_kind:
            scoped_filter = resolve_scoped_filter(self._scoped_kind, self._composer_query)
        self._context_header.set_palette_context(
            view_mode=self._view_mode,
            scoped_kind=self._scoped_kind,
            query=self._composer_query,
            scoped_filter=scoped_filter,
        )

    def _resolve_view_from_query(self) -> None:
        q = (self._composer_query or "").strip()
        if not q:
            self._scoped_kind = None
            self._view_mode = ComposerPaletteView.BROWSE
            return
        if self._view_mode == ComposerPaletteView.SCOPED and self._scoped_kind:
            return
        self._view_mode = ComposerPaletteView.SEARCH
        self._scoped_kind = None

    def _search_query(self) -> str:
        if self._view_mode == ComposerPaletteView.SCOPED and self._scoped_kind:
            return resolve_scoped_filter(self._scoped_kind, self._composer_query)
        return (self._composer_query or self._filter.text()).strip()

    def _apply_popup_chrome(self) -> None:
        apply_translucent_window_chrome(self, transparent_stylesheet=True)

    def _apply_list_stylesheet(self) -> None:
        colors = getattr(self, "_theme_colors", None)
        if colors is None:
            return
        fg, border, hover = colors
        if self._view_mode == ComposerPaletteView.BROWSE:
            item_rules = """
            QListWidget#ComposerMentionList::item {
                padding: 0px;
                border: none;
                outline: none;
                background: transparent;
            }
            """
        else:
            item_rules = f"""
            QListWidget#ComposerMentionList::item {{
                padding: 8px 10px;
                border: none;
                outline: none;
                background: transparent;
                font-weight: normal;
            }}
            QListWidget#ComposerMentionList::item:selected,
            QListWidget#ComposerMentionList::item:hover {{
                background: transparent;
                border: none;
                outline: none;
                font-weight: normal;
            }}
            """
        list_ss = f"""
            QListWidget#ComposerMentionList {{
                background-color: transparent;
                color: {fg};
                border: none;
                outline: none;
            }}
            QListWidget#ComposerMentionList:focus {{
                border: none;
                outline: none;
            }}
            QListWidget#ComposerMentionList::viewport {{
                border: none;
                outline: none;
                background: transparent;
            }}
            {item_rules}
        """
        self._list.setStyleSheet(list_ss)
        self._list.setAutoFillBackground(False)
        self._list.viewport().setAutoFillBackground(False)

    def _set_drill_chrome_visible(self, visible: bool) -> None:
        self._filter.setVisible(visible)
        self._layout.setSpacing(6)

    def _restore_composer_focus(self) -> None:
        """``Qt.Popup`` grabs the keyboard on show; composer must keep typing focus."""
        parent = self.parentWidget()
        if parent is None:
            return
        self.releaseKeyboard()
        parent.setFocus(Qt.FocusReason.OtherFocusReason)
        top = parent.window()
        if top is not None and top is not self:
            top.activateWindow()

    def close_mention(self) -> None:
        """Hide popup and reset state (e.g. user deleted ``@``)."""
        self._view_mode = ComposerPaletteView.BROWSE
        self._scoped_kind = None
        self._composer_query = ""
        self._set_drill_chrome_visible(False)
        self._filter.clear()
        self._clear_type_buffer()
        self.hide()

    def _sync_filter_from_composer_query(self) -> None:
        self._filter.blockSignals(True)
        self._filter.setText(self._composer_query)
        self._filter.blockSignals(False)

    def set_composer_query(self, query: str, global_pos=None) -> None:
        """Rebuild list for current mode using composer-typed ``@`` suffix."""
        self._composer_query = query or ""
        self._sync_filter_from_composer_query()
        anchor = global_pos
        if anchor is not None:
            self._anchor_global_pos = QPoint(anchor)
        elif self.parentWidget() is not None and hasattr(
            self.parentWidget(), "_mention_global_pos"
        ):
            self._anchor_global_pos = QPoint(self.parentWidget()._mention_global_pos())

        self._resolve_view_from_query()
        self._apply_list_stylesheet()

        if self._view_mode == ComposerPaletteView.SEARCH:
            self._list.setFixedHeight(_DRILL_LIST_HEIGHT)
            self._schedule_search()
        elif self._view_mode == ComposerPaletteView.SCOPED:
            self._list.setFixedHeight(_DRILL_LIST_HEIGHT)
            if self._scoped_kind in ("file", "conversation"):
                self._schedule_search()
            else:
                self._rebuild_visible_list()
        else:
            self._rebuild_visible_list()

        self._select_first_actionable_row()
        if self._anchor_global_pos is not None:
            self._position_at(self._anchor_global_pos)
        self._sync_context_header()
        self._sync_panel_tooltips()

    def _clear_type_buffer(self) -> None:
        self._type_buffer = ""
        self._type_reset_timer.stop()

    def seed_type_buffer(self, text: str) -> None:
        """Seed prefix matcher (e.g. pasted ``@Files``) and highlight."""
        cleaned = "".join(ch for ch in (text or "").lower() if ch.isalpha())
        self._type_buffer = cleaned
        if self._type_buffer:
            self._type_reset_timer.start(_TYPEAHEAD_RESET_MS)
        self._apply_type_buffer()

    def _append_type_char(self, ch: str) -> None:
        self._type_buffer += ch.lower()
        self._type_reset_timer.start(_TYPEAHEAD_RESET_MS)
        self._apply_type_buffer()

    def _pop_type_char(self) -> None:
        if self._type_buffer:
            self._type_buffer = self._type_buffer[:-1]
            self._type_reset_timer.start(_TYPEAHEAD_RESET_MS)
        self._apply_type_buffer()

    def _apply_type_buffer(self) -> None:
        if self._view_mode != ComposerPaletteView.BROWSE or not self.isVisible():
            return
        if not self._type_buffer:
            self._select_first_actionable_row()
            return
        target_kind = _ROOT_ROWS[root_row_index_for_query(self._type_buffer)][0]
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, tuple) and data[0] in ("root", "category") and data[1] == target_kind:
                self._list.setCurrentRow(row)
                self._list.scrollToItem(item)
                return

    def show_root(self, global_pos) -> None:
        self._anchor_global_pos = QPoint(global_pos)
        self._view_mode = ComposerPaletteView.BROWSE
        self._scoped_kind = None
        self._set_drill_chrome_visible(False)
        self._apply_list_stylesheet()
        self._filter.clear()
        self._clear_type_buffer()
        self._rebuild_visible_list()
        self._sync_context_header()
        self._sync_panel_tooltips()
        self._select_first_actionable_row()
        self._position_at(self._anchor_global_pos)
        self.show()

    def apply_root_query(self, query: str) -> None:
        """Legacy hook: apply composer ``@`` suffix to the root menu."""
        if not self.isVisible():
            return
        self.set_composer_query(query)

    def enter_scoped_browse(self, kind: str, global_pos) -> None:
        if global_pos is not None:
            self._anchor_global_pos = QPoint(global_pos)
        self._view_mode = ComposerPaletteView.SCOPED
        self._scoped_kind = kind
        self._set_drill_chrome_visible(False)
        self._apply_list_stylesheet()
        self._list.setFixedHeight(_DRILL_LIST_HEIGHT)
        self._rebuild_visible_list()
        self._sync_context_header()
        self._sync_panel_tooltips()
        self._select_first_actionable_row()
        anchor = self._anchor_global_pos or QPoint(global_pos)
        self._position_at(anchor)
        self.show()

    def show_drill_down(self, kind: str, global_pos, *, query: str = "") -> None:
        """Legacy alias for explicit category browse."""
        self._composer_query = query or self._composer_query
        self.enter_scoped_browse(kind, global_pos)

    def eventFilter(self, watched, event) -> bool:
        if (
            self.isVisible()
            and event.type() == QEvent.Type.KeyPress
            and watched in (self._filter, self._list)
        ):
            if self._handle_navigation_key(event, from_filter=(watched is self._filter)):
                return True
        return super().eventFilter(watched, event)

    def _host_window_rect(self) -> QRect | None:
        """Global geometry of the Qube top-level window (not the whole screen).

        This widget is a ``Qt.Popup`` window, so ``self.window()`` is the popup
        itself — use the composer parent's ``window()`` (``MainWindow``).
        """
        anchor_widget = self.parentWidget()
        if anchor_widget is not None:
            top = anchor_widget.window()
            if top is not None and top is not self and top.isVisible():
                return top.frameGeometry()
        return None

    def _clamp_to_window(self, anchor: QPoint, width: int, height: int) -> QPoint:
        """Keep the popup inside the app window; prefer below the composer anchor."""
        bounds = self._host_window_rect()
        m = self._window_margin
        x = anchor.x()
        y = anchor.y()
        if bounds is not None and bounds.isValid():
            width = min(width, max(200, bounds.width() - 2 * m))
            # Below anchor (composer caret), else above.
            if y + height > bounds.bottom() - m:
                y = anchor.y() - height - 4
            if y < bounds.top() + m:
                y = bounds.top() + m
            if y + height > bounds.bottom() - m:
                y = max(bounds.top() + m, bounds.bottom() - m - height)
            if x + width > bounds.right() - m:
                x = bounds.right() - m - width
            if x < bounds.left() + m:
                x = bounds.left() + m
        return QPoint(x, y)

    def _position_at(self, global_pos) -> None:
        if global_pos is not None:
            self._anchor_global_pos = QPoint(global_pos)
        anchor = self._anchor_global_pos or QPoint(0, 0)
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        self._shell.setMinimumHeight(0)
        self._shell.setMaximumHeight(16777215)
        self._shell.adjustSize()
        self.adjustSize()
        w = max(280, self.sizeHint().width())
        bounds = self._host_window_rect()
        if bounds is not None and bounds.isValid():
            w = min(w, max(200, bounds.width() - 2 * self._window_margin))
        self.setFixedWidth(w)
        self.adjustSize()
        h = self.sizeHint().height()
        pt = self._clamp_to_window(anchor, w, h)
        self.move(pt)
        self.resize(w, h)

    def _sync_panel_tooltips(self) -> None:
        if self._view_mode == ComposerPaletteView.BROWSE:
            panel_tip = _ROOT_LIST_TOOLTIP
        elif self._view_mode == ComposerPaletteView.SEARCH:
            panel_tip = _SEARCH_LIST_TOOLTIP
        else:
            panel_tip = ""
        self.setToolTip(panel_tip)
        self._shell.setToolTip(panel_tip)
        self._list.setToolTip(panel_tip)

        if self._view_mode == ComposerPaletteView.SCOPED and self._scoped_kind:
            scoped_tip = _SCOPED_LIST_TOOLTIP
            self._context_header.set_scoped_breadcrumb_tooltip(scoped_tip)
            self._filter.setToolTip(_FILTER_TOOLTIPS.get(self._scoped_kind, scoped_tip))
        else:
            self._context_header.set_scoped_breadcrumb_tooltip("")
            if self._view_mode != ComposerPaletteView.SCOPED:
                self._filter.setToolTip("")

    def _apply_row_tooltip(self, row: QListWidgetItem, text: str) -> None:
        if not text:
            return
        row.setToolTip(text)
        widget = self._list.itemWidget(row)
        if widget is None:
            return
        widget.setToolTip(text)
        for child in widget.findChildren(QWidget):
            child.setToolTip(text)

    def _schedule_search(self) -> None:
        if self._view_mode in (ComposerPaletteView.SEARCH, ComposerPaletteView.SCOPED):
            self._search_timer.start(self._search_debounce_ms)

    def _run_search(self) -> None:
        self._rebuild_visible_list()
        self._select_first_actionable_row()
        if self._anchor_global_pos is not None:
            self._position_at(self._anchor_global_pos)

    def _rebuild_visible_list(self) -> None:
        self._list.clear()
        if self._view_mode == ComposerPaletteView.BROWSE:
            self._populate_root()
        elif self._view_mode == ComposerPaletteView.SEARCH:
            self._populate_search()
        elif self._scoped_kind == "file":
            self._populate_files()
        elif self._scoped_kind == "conversation":
            self._populate_conversations()
        elif self._scoped_kind == "tool":
            self._populate_tools()
        elif self._scoped_kind == "skill":
            self._populate_skills()
        elif self._scoped_kind == "command":
            self._populate_commands()

    def _populate_search(self) -> None:
        q = (self._composer_query or "").strip()
        if not q:
            self._add_empty_row("Type to search")
            self._list.setFixedHeight(_DRILL_LIST_HEIGHT)
            return
        hits = search_composer_mentions(
            q,
            db=self._db,
            store=self._store,
            active_session_id=self._active_session_id,
        )
        if not hits:
            self._add_empty_row("No matching results")
            self._list.setFixedHeight(_DRILL_LIST_HEIGHT)
            return
        theme = getattr(self, "_theme", view_resolved_theme(self, is_dark=self._is_dark))
        sub_color = theme.text_muted if theme.is_dark else theme.text_secondary
        last_section: str | None = None
        for hit in hits:
            if hit.section != last_section:
                last_section = hit.section
                header = QListWidgetItem(section_label(hit.section))
                header.setFlags(Qt.ItemFlag.NoItemFlags)
                header.setData(Qt.ItemDataRole.UserRole, ("section", hit.section))
                header.setForeground(QBrush(QColor(sub_color)))
                self._list.addItem(header)
            text = f"{hit.label} — {hit.subtitle}" if hit.subtitle else hit.label
            row = QListWidgetItem(text)
            row.setData(Qt.ItemDataRole.UserRole, hit.payload)
            if isinstance(hit.payload, tuple) and hit.payload[0] == "category":
                kind = hit.payload[1]
                self._apply_row_tooltip(row, _ROOT_ROW_TOOLTIPS.get(kind, hit.subtitle))
            elif isinstance(hit.payload, ComposerAttachment):
                att = hit.payload
                if att.kind == "file":
                    tip = f"Attach {att.label}. Search will be scoped to this document."
                elif att.kind == "conversation":
                    tip = (
                        f'Attach "{att.label}". Includes that chat\'s transcript in this turn (~7000 chars).'
                    )
                elif att.kind == "tool":
                    tool = composer_tool_by_id(att.id)
                    tip = composer_tool_tooltip(tool) if tool is not None else f"Attach {att.label}."
                else:
                    tip = f"Attach {att.label}."
                self._apply_row_tooltip(row, tip)
            elif isinstance(hit.payload, CapabilityPaletteEntry):
                entry = hit.payload
                row.setData(Qt.ItemDataRole.UserRole, entry.to_attachment())
                self._apply_row_tooltip(row, capability_palette_tooltip(entry))
            self._list.addItem(row)
        self._list.setFixedHeight(_DRILL_LIST_HEIGHT)

    def _populate_root(self) -> None:
        theme = getattr(self, "_theme", view_resolved_theme(self, is_dark=self._is_dark))
        sub_color = theme.text_muted if theme.is_dark else theme.text_secondary
        fg_color = theme.text_primary
        list_w = max(260, self._list.viewport().width())
        visible_indices = list(range(len(_ROOT_ROWS)))
        for idx, (kind, title, subtitle, icon_name) in enumerate(_ROOT_ROWS):
            if idx not in visible_indices:
                continue
            row = QListWidgetItem()
            row.setData(Qt.ItemDataRole.UserRole, ("category", kind))
            tip = _ROOT_ROW_TOOLTIPS.get(kind, subtitle)
            widget = QWidget()
            widget.setMinimumHeight(_ROOT_ROW_HEIGHT - 8)
            hl = QHBoxLayout(widget)
            hl.setContentsMargins(8, 6, 8, 6)
            hl.setSpacing(10)
            ic = QLabel()
            ic.setFixedSize(20, 20)
            ic.setPixmap(themed_fa_pixmap(icon_name, sub_color, 20))
            ic.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            col = QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet(
                f"color: {fg_color}; font-weight: 600; font-size: 13px;"
            )
            s = QLabel(subtitle)
            s.setStyleSheet(f"color: {sub_color}; font-size: 11px;")
            s.setWordWrap(False)
            col.addWidget(t)
            col.addWidget(s)
            hl.addWidget(ic, alignment=Qt.AlignmentFlag.AlignVCenter)
            hl.addLayout(col, stretch=1)
            widget.adjustSize()
            row_h = max(_ROOT_ROW_HEIGHT, widget.sizeHint().height())
            row.setSizeHint(QSize(list_w, row_h))
            self._list.addItem(row)
            self._list.setItemWidget(row, widget)
            self._apply_row_tooltip(row, tip)
        if self._list.count() == 0:
            self._add_empty_row("No matching categories")
        n = max(1, self._list.count())
        self._list.setFixedHeight(n * _ROOT_ROW_HEIGHT + max(0, (n - 1) * self._list.spacing()) + 6)

    def _populate_files(self) -> None:
        if not self._db:
            self._add_empty_row("Database unavailable")
            return
        q = self._search_query()
        try:
            if q:
                docs = self._db.get_library_documents_for_sidebar_search(q, limit=80)
            else:
                docs = self._db.get_library_documents(limit=80, offset=0)
        except Exception:
            docs = []
        shown = 0
        for doc in docs:
            filename = str(doc.get("filename") or "").strip()
            if not filename:
                continue
            chunk_count = int(doc.get("chunk_count") or 0)
            indexed = chunk_count > 0
            if self._store is not None:
                try:
                    indexed = indexed or self._store.source_exists(filename)
                except Exception:
                    pass
            if not indexed:
                continue
            row = QListWidgetItem(filename)
            row.setData(
                Qt.ItemDataRole.UserRole,
                ComposerAttachment(kind="file", id=filename, label=filename),
            )
            self._apply_row_tooltip(
                row,
                f"Attach {filename}. Search will be scoped to this document.",
            )
            self._list.addItem(row)
            shown += 1
        if shown == 0:
            self._add_empty_row("No indexed documents" if not q else "No matching documents")

    def _populate_conversations(self) -> None:
        if not self._db:
            self._add_empty_row("Database unavailable")
            return
        q = self._search_query()
        try:
            if q:
                sessions = self._db.get_sessions_for_sidebar_search(q, limit=80)
            else:
                _folders, grouped = self._db.get_sessions_for_sidebar_by_folder()
                sessions = []
                for rows in grouped.values():
                    sessions.extend(rows)
                sessions.sort(
                    key=lambda s: str(s.get("updated_at") or ""),
                    reverse=True,
                )
                sessions = sessions[:80]
        except Exception:
            sessions = []
        shown = 0
        for sess in sessions:
            sid = str(sess.get("id") or "")
            if not sid or sid == self._active_session_id:
                continue
            title = str(sess.get("title") or "Untitled").strip()
            row = QListWidgetItem(title)
            row.setData(
                Qt.ItemDataRole.UserRole,
                ComposerAttachment(kind="conversation", id=sid, label=title),
            )
            self._apply_row_tooltip(
                row,
                f'Attach "{title}". Includes that chat\'s transcript in this turn (~7000 chars).',
            )
            self._list.addItem(row)
            shown += 1
        if shown == 0:
            self._add_empty_row("No other conversations" if not q else "No matching conversations")

    def _populate_tools(self) -> None:
        q = self._search_query().lower()
        for tool in composer_tools_for_palette(q):
            label = tool["label"]
            desc = tool["description"]
            text = f"{label} — {desc}"
            row = QListWidgetItem(text)
            row.setData(
                Qt.ItemDataRole.UserRole,
                ComposerAttachment(kind="tool", id=tool["id"], label=label),
            )
            self._apply_row_tooltip(row, composer_tool_tooltip(tool))
            self._list.addItem(row)

    def _populate_skills(self) -> None:
        q = self._search_query()
        from core.skills.registry import get_skill

        mentions = list_skill_mentions_for_palette(query=q)
        for mention in mentions:
            skill = get_skill(mention.id)
            desc = skill.description if skill is not None else ""
            text = f"{mention.label} — {desc}" if desc else mention.label
            row = QListWidgetItem(text)
            row.setData(Qt.ItemDataRole.UserRole, mention)
            if desc:
                tip = (
                    f"{mention.label}. {desc} "
                    f"Inserts @[skill:{mention.id}] as prompt guidance."
                )
            else:
                tip = f"{mention.label}. Inserts @[skill:{mention.id}] as prompt guidance."
            self._apply_row_tooltip(row, tip)
            self._list.addItem(row)
        if self._list.count() == 0:
            self._add_empty_row("No matching skills")

    def _populate_commands(self) -> None:
        q = self._search_query().lower()
        for command in COMPOSER_COMMANDS:
            if q and q not in command.label.lower() and q not in command.description.lower() and q not in command.id:
                continue
            text = f"{command.label} — {command.description}"
            row = QListWidgetItem(text)
            row.setData(Qt.ItemDataRole.UserRole, command)
            self._apply_row_tooltip(
                row,
                f"{command.label}. {command.description} Runs immediately when selected.",
            )
            self._list.addItem(row)
        if self._list.count() == 0:
            self._add_empty_row("No matching commands")

    def _add_empty_row(self, text: str) -> None:
        row = QListWidgetItem(text)
        row.setFlags(Qt.ItemFlag.NoItemFlags)
        self._list.addItem(row)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if data is None:
            return
        if isinstance(data, tuple):
            if data[0] == "section":
                return
            if data[0] in ("root", "category"):
                kind = data[1]
                anchor = self._anchor_global_pos
                parent = self.parentWidget()
                if anchor is None and parent is not None and hasattr(parent, "_mention_global_pos"):
                    anchor = parent._mention_global_pos()
                self.enter_scoped_browse(kind, anchor)
                return
        if isinstance(data, ComposerCommand):
            self.command_selected.emit(data)
            self.hide()
            return
        if isinstance(data, ComposerSkillMention):
            self.skill_selected.emit(data)
            self.hide()
            return
        if isinstance(data, ComposerAttachment):
            self.item_selected.emit(data)
            self.hide()

    def _select_first_actionable_row(self) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsEnabled:
                self._list.setCurrentRow(row)
                return

    def _navigate_to_root(self) -> None:
        anchor = self._anchor_global_pos
        parent = self.parentWidget()
        if anchor is None and parent is not None and hasattr(parent, "_mention_global_pos"):
            anchor = parent._mention_global_pos()
        self._scoped_kind = None
        q = self._composer_query
        if parent is not None and hasattr(parent, "_active_mention_query"):
            active = parent._active_mention_query()
            if active:
                q = active[1]
        self._view_mode = (
            ComposerPaletteView.SEARCH if (q or "").strip() else ComposerPaletteView.BROWSE
        )
        self._apply_list_stylesheet()
        self._rebuild_visible_list()
        self._sync_context_header()
        self._sync_panel_tooltips()
        self._select_first_actionable_row()
        if anchor is not None:
            self._position_at(anchor)

    def _activate_current_item(self) -> None:
        cur = self._list.currentItem()
        if cur is None and self._list.count() > 0:
            self._select_first_actionable_row()
            cur = self._list.currentItem()
        if cur is None:
            return
        if not (cur.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        self._on_item_clicked(cur)

    def handle_navigation_key(self, event, *, from_filter: bool = False) -> bool:
        """Return True if the key was consumed (composer keeps focus)."""
        if not self.isVisible():
            return False
        return self._handle_navigation_key(event, from_filter=from_filter)

    def handle_key(self, event) -> bool:
        """Return True if the key was consumed (composer still has focus)."""
        return self.handle_navigation_key(event, from_filter=False)

    def _handle_navigation_key(self, event: QKeyEvent, *, from_filter: bool) -> bool:
        key = event.key()

        if key == Qt.Key.Key_Backspace:
            if self._view_mode == ComposerPaletteView.SCOPED and self._scoped_kind:
                if from_filter and self._filter.text():
                    return False
                if not resolve_scoped_filter(self._scoped_kind, self._composer_query):
                    self._navigate_to_root()
                    event.accept()
                    return True
                return False
            return False

        if (
            self._view_mode == ComposerPaletteView.BROWSE
            and self._try_activate_root_by_number(key, event)
        ):
            return True

        if key == Qt.Key.Key_Escape:
            if self._view_mode == ComposerPaletteView.SCOPED:
                self._navigate_to_root()
                event.accept()
                return True
            self.hide()
            self.dismissed.emit()
            event.accept()
            return True

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Tab):
            self._activate_current_item()
            event.accept()
            return True

        if key == Qt.Key.Key_Up:
            self._advance_actionable_row(-1)
            event.accept()
            return True

        if key == Qt.Key.Key_Down:
            self._advance_actionable_row(1)
            event.accept()
            return True

        return False

    def _try_activate_root_by_number(self, key: int, event: QKeyEvent) -> bool:
        """Browse menu: ``1``–``5`` enter scoped category rows."""
        if self._view_mode != ComposerPaletteView.BROWSE:
            return False
        idx = -1
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_5:
            idx = key - Qt.Key.Key_1
        else:
            keypad = getattr(Qt.Key, "Keypad1", None)
            if keypad is not None and Qt.Key.Keypad1 <= key <= Qt.Key.Keypad5:
                idx = key - Qt.Key.Keypad1
        if idx < 0 or idx >= len(_ROOT_ROWS):
            return False
        kind = _ROOT_ROWS[idx][0]
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is None:
                continue
            data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, tuple) and data[0] in ("root", "category") and data[1] == kind:
                self._list.setCurrentRow(row)
                self._activate_current_item()
                event.accept()
                return True
        return False

    def _advance_actionable_row(self, delta: int) -> None:
        if self._list.count() == 0:
            return
        start = self._list.currentRow()
        if start < 0:
            start = 0 if delta > 0 else self._list.count()
        idx = start
        for _ in range(self._list.count()):
            idx += delta
            if idx < 0 or idx >= self._list.count():
                break
            item = self._list.item(idx)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsEnabled:
                self._list.setCurrentRow(idx)
                break

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_popup_chrome()
        QTimer.singleShot(0, self._restore_composer_focus)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Forward keys grabbed by ``Qt.Popup`` to the composer input."""
        composer = self.parentWidget()
        if composer is not None:
            composer.keyPressEvent(event)
            if event.isAccepted():
                return
        event.ignore()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        composer = self.parentWidget()
        if composer is not None:
            composer.keyReleaseEvent(event)
            if event.isAccepted():
                return
        event.ignore()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._view_mode = ComposerPaletteView.BROWSE
        self._scoped_kind = None
        self._set_drill_chrome_visible(False)
        self.dismissed.emit()
