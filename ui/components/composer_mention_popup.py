"""Floating @-mention picker for the chat composer (Files / Conversations / Tools)."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt, QTimer, pyqtSignal, QEvent
from PyQt6.QtGui import QColor, QKeyEvent, QPalette
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import qtawesome as qta

from core.composer_attachments import COMPOSER_TOOLS, ComposerAttachment
from core.composer_commands import COMPOSER_COMMANDS, ComposerCommand
from core.composer_skills import ComposerSkillMention, list_skill_mentions_for_palette

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
    "Type letters to jump to a category, or use arrow keys and Enter to select."
)
_DRILL_LIST_TOOLTIP = (
    "Use arrow keys and Enter to select. Backspace returns to categories."
)

_ROOT_ROW_HEIGHT = 56
_DRILL_LIST_HEIGHT = 220
_TYPEAHEAD_RESET_MS = 900


class _ComposerBackLink(QLabel):
    """Compact text link — avoids global QPushButton padding/min-height."""

    activated = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("ComposerMentionBack")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def mouseReleaseEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
        ):
            self.activated.emit()
        super().mouseReleaseEvent(event)


def root_row_index_for_query(query: str) -> int:
    """Best root-menu row for a type-ahead prefix (0-based index into _ROOT_ROWS)."""
    q = (query or "").strip().lower()
    if not q:
        return 0
    matches: list[int] = []
    for idx, (kind, title, _subtitle, _icon) in enumerate(_ROOT_ROWS):
        title_l = title.lower()
        kind_l = kind.lower()
        if title_l.startswith(q) or kind_l.startswith(q):
            matches.append(idx)
    return matches[0] if matches else 0


class ComposerMentionPopup(QWidget):
    """Frameless popup: root categories or searchable drill-down list."""

    item_selected = pyqtSignal(object)  # ComposerAttachment
    skill_selected = pyqtSignal(object)  # ComposerSkillMention
    command_selected = pyqtSignal(object)  # ComposerCommand
    dismissed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._is_dark = True
        self._mode: str | None = None  # None = root, else file|conversation|tool
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

        shell = QFrame(self)
        shell.setObjectName("ComposerMentionShell")
        shell.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._shell = shell
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(shell)

        layout = QVBoxLayout(shell)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._shell_layout = layout

        self._back_link = _ComposerBackLink()
        self._back_link.hide()
        self._back_link.activated.connect(self._navigate_to_root)
        layout.addWidget(
            self._back_link,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self._filter = QLineEdit()
        self._filter.setObjectName("ComposerMentionFilter")
        self._filter.setPlaceholderText("Filter…")
        self._filter.hide()
        self._filter.textChanged.connect(self._schedule_search)
        layout.addWidget(self._filter)

        self._list = QListWidget()
        self._list.setObjectName("ComposerMentionList")
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
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
        self._back_link.setToolTip(_BACK_LINK_TOOLTIP)

        self._search_debounce_ms = 280
        self._anchor_global_pos: QPoint | None = None
        self._window_margin = 8
        self.apply_theme(True)

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
        self._is_dark = is_dark
        if is_dark:
            bg, fg, border, hover = "#1e1e2e", "#cdd6f4", "rgba(255,255,255,0.1)", "#313244"
            sub = "#a6adc8"
        else:
            bg, fg, border, hover = "#ffffff", "#1e293b", "#cbd5e1", "#f1f5f9"
            sub = "#64748b"

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

        chevron = "#94a3b8" if is_dark else "#64748b"
        self._back_link.setText("")
        self._back_link.setPixmap(qta.icon("fa5s.chevron-left", color=chevron).pixmap(12, 12))
        self._back_link.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._back_link.setFixedHeight(16)
        back_ss = f"""
            QLabel#ComposerMentionBack {{
                color: {fg};
                background: transparent;
                border: none;
                padding: 2px 0px;
                margin: 0px;
            }}
            QLabel#ComposerMentionBack:hover {{
                color: {chevron};
            }}
        """
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
        list_ss = f"""
            QListWidget#ComposerMentionList {{
                background-color: transparent;
                color: {fg};
                border: none;
                outline: none;
            }}
            QListWidget#ComposerMentionList::item {{
                padding: 8px 10px;
                border-radius: 8px;
            }}
            QListWidget#ComposerMentionList::item:selected {{
                background-color: {hover};
            }}
        """
        self._shell.setStyleSheet(shell_ss)
        self._back_link.setStyleSheet(back_ss)
        self._filter.setStyleSheet(filter_ss)
        self._list.setStyleSheet(list_ss)
        self._rebuild_visible_list()

    def _set_drill_chrome_visible(self, visible: bool) -> None:
        self._back_link.setVisible(visible)
        self._filter.setVisible(visible)
        self._shell_layout.setSpacing(4 if visible else 6)

    def close_mention(self) -> None:
        """Hide popup and reset drill-down state (e.g. user deleted ``@``)."""
        self._mode = None
        self._set_drill_chrome_visible(False)
        self._filter.clear()
        self._clear_type_buffer()
        self.hide()

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
        if self._mode is not None or not self.isVisible():
            return
        idx = root_row_index_for_query(self._type_buffer)
        if idx < 0 or idx >= self._list.count():
            return
        item = self._list.item(idx)
        if item is None or not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        self._list.setCurrentRow(idx)
        self._list.scrollToItem(item)

    def show_root(self, global_pos) -> None:
        self._anchor_global_pos = QPoint(global_pos)
        self._mode = None
        self._set_drill_chrome_visible(False)
        self._filter.clear()
        self._clear_type_buffer()
        self._rebuild_visible_list()
        self._sync_panel_tooltips()
        self._select_first_actionable_row()
        self._position_at(self._anchor_global_pos)
        self.show()
        self._list.setFocus()

    def apply_root_query(self, query: str) -> None:
        """Legacy hook: seed the popup type-ahead buffer from composer ``@`` suffix."""
        if self._mode is not None or not self.isVisible():
            return
        if query:
            self.seed_type_buffer(query)
        else:
            self._clear_type_buffer()
            self._select_first_actionable_row()

    def show_drill_down(self, kind: str, global_pos, *, query: str = "") -> None:
        if global_pos is not None:
            self._anchor_global_pos = QPoint(global_pos)
        self._mode = kind
        self._set_drill_chrome_visible(True)
        self._filter.setText(query)
        placeholders = {
            "file": "Search documents…",
            "conversation": "Search conversations…",
            "tool": "Filter tools…",
            "skill": "Search skills…",
            "command": "Filter commands…",
        }
        self._filter.setPlaceholderText(placeholders.get(kind, "Filter…"))
        self._filter.setToolTip(_FILTER_TOOLTIPS.get(kind, "Filter this list."))
        self._list.setFixedHeight(_DRILL_LIST_HEIGHT)
        self._rebuild_visible_list()
        self._sync_panel_tooltips()
        self._select_first_actionable_row()
        anchor = self._anchor_global_pos or QPoint(global_pos)
        self._position_at(anchor)
        self.show()
        self._filter.setFocus()

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
        if self._mode is None:
            panel_tip = _ROOT_LIST_TOOLTIP
        else:
            panel_tip = _DRILL_LIST_TOOLTIP
        self.setToolTip(panel_tip)
        self._shell.setToolTip(panel_tip)
        self._list.setToolTip(panel_tip)

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
        if self._mode:
            self._search_timer.start(self._search_debounce_ms)

    def _run_search(self) -> None:
        self._rebuild_visible_list()

    def _rebuild_visible_list(self) -> None:
        self._list.clear()
        if self._mode is None:
            self._populate_root()
        elif self._mode == "file":
            self._populate_files()
        elif self._mode == "conversation":
            self._populate_conversations()
        elif self._mode == "tool":
            self._populate_tools()
        elif self._mode == "skill":
            self._populate_skills()
        elif self._mode == "command":
            self._populate_commands()

    def _populate_root(self) -> None:
        sub_color = "#a6adc8" if self._is_dark else "#64748b"
        list_w = max(260, self._list.viewport().width())
        for kind, title, subtitle, icon_name in _ROOT_ROWS:
            row = QListWidgetItem()
            row.setData(Qt.ItemDataRole.UserRole, ("root", kind))
            tip = _ROOT_ROW_TOOLTIPS.get(kind, subtitle)
            widget = QWidget()
            widget.setMinimumHeight(_ROOT_ROW_HEIGHT - 8)
            hl = QHBoxLayout(widget)
            hl.setContentsMargins(8, 6, 8, 6)
            hl.setSpacing(10)
            ic = QLabel()
            ic.setFixedSize(20, 20)
            ic.setPixmap(qta.icon(icon_name, color=sub_color).pixmap(20, 20))
            ic.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            col = QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet(
                f"color: {'#cdd6f4' if self._is_dark else '#1e293b'}; "
                "font-weight: 600; font-size: 13px;"
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
        # Shrink list to fit three category rows (avoids huge empty area + clipping).
        n = max(1, self._list.count())
        self._list.setFixedHeight(n * _ROOT_ROW_HEIGHT + max(0, (n - 1) * self._list.spacing()) + 6)

    def _populate_files(self) -> None:
        if not self._db:
            self._add_empty_row("Database unavailable")
            return
        q = self._filter.text().strip()
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
            row.setToolTip(
                f"Attach {filename}. Search will be scoped to this document."
            )
            self._list.addItem(row)
            shown += 1
        if shown == 0:
            self._add_empty_row("No indexed documents" if not q else "No matching documents")

    def _populate_conversations(self) -> None:
        if not self._db:
            self._add_empty_row("Database unavailable")
            return
        q = self._filter.text().strip()
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
            row.setToolTip(
                f'Attach "{title}". Includes that chat\'s transcript in this turn (~7000 chars).'
            )
            self._list.addItem(row)
            shown += 1
        if shown == 0:
            self._add_empty_row("No other conversations" if not q else "No matching conversations")

    def _populate_tools(self) -> None:
        q = self._filter.text().strip().lower()
        for tool in COMPOSER_TOOLS:
            label = tool["label"]
            desc = tool["description"]
            if q and q not in label.lower() and q not in desc.lower() and q not in tool["id"]:
                continue
            text = f"{label} — {desc}"
            row = QListWidgetItem(text)
            row.setData(
                Qt.ItemDataRole.UserRole,
                ComposerAttachment(kind="tool", id=tool["id"], label=label),
            )
            row.setToolTip(
                f"{label}: {desc}. Inserts @[tool:{tool['id']}] to route this message."
            )
            self._list.addItem(row)

    def _populate_skills(self) -> None:
        q = self._filter.text().strip()
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
                    f"{mention.label}: {desc}. "
                    f"Inserts @[skill:{mention.id}] as prompt guidance."
                )
            else:
                tip = f"Inserts @[skill:{mention.id}] as prompt guidance."
            row.setToolTip(tip)
            self._list.addItem(row)
        if self._list.count() == 0:
            self._add_empty_row("No matching skills")

    def _populate_commands(self) -> None:
        q = self._filter.text().strip().lower()
        for command in COMPOSER_COMMANDS:
            if q and q not in command.label.lower() and q not in command.description.lower() and q not in command.id:
                continue
            text = f"{command.label} — {command.description}"
            row = QListWidgetItem(text)
            row.setData(Qt.ItemDataRole.UserRole, command)
            row.setToolTip(
                f"{command.label}: {command.description}. Runs immediately when selected."
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
        if isinstance(data, tuple) and data[0] == "root":
            kind = data[1]
            parent = self.parent()
            query = ""
            if parent is not None and hasattr(parent, "_active_mention_query"):
                active = parent._active_mention_query()
                if active:
                    query = active[1]
            if not query and self._type_buffer:
                query = self._type_buffer
            # Reuse composer caret anchor — not popup bottom-left (drifts off-window).
            anchor = self._anchor_global_pos
            if anchor is None and parent is not None and hasattr(parent, "_mention_global_pos"):
                anchor = parent._mention_global_pos()
            self.show_drill_down(kind, anchor, query=query)
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
        parent = self.parent()
        if anchor is None and parent is not None and hasattr(parent, "_mention_global_pos"):
            anchor = parent._mention_global_pos()
        self.show_root(anchor)

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

    def handle_key(self, event) -> bool:
        """Return True if the key was consumed (composer still has focus)."""
        if not self.isVisible():
            return False
        return self._handle_navigation_key(event, from_filter=False)

    def _handle_navigation_key(self, event: QKeyEvent, *, from_filter: bool) -> bool:
        key = event.key()

        if key == Qt.Key.Key_Backspace:
            if self._mode is not None:
                if from_filter and self._filter.text():
                    return False
                self._navigate_to_root()
                event.accept()
                return True
            if self._type_buffer:
                self._pop_type_char()
                event.accept()
                return True
            self.hide()
            event.accept()
            return True

        if self._mode is None and self._try_typeahead_key(event):
            return True

        if self._mode is None and self._try_activate_root_by_number(key, event):
            return True

        if key == Qt.Key.Key_Escape:
            if self._mode is not None:
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
            if from_filter:
                self._list.setFocus(Qt.FocusReason.OtherFocusReason)
            row = max(0, self._list.currentRow() - 1)
            self._list.setCurrentRow(row)
            event.accept()
            return True

        if key == Qt.Key.Key_Down:
            if from_filter:
                self._list.setFocus(Qt.FocusReason.OtherFocusReason)
            start = self._list.currentRow()
            if start < 0 and self._list.count() > 0:
                start = -1
            row = min(self._list.count() - 1, start + 1)
            self._list.setCurrentRow(row)
            event.accept()
            return True

        return False

    def _try_typeahead_key(self, event: QKeyEvent) -> bool:
        """Root menu: letter keys build a prefix buffer and highlight the best row."""
        if event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return False
        text = event.text()
        if not text or len(text) != 1 or not text.isalpha():
            return False
        self._append_type_char(text)
        event.accept()
        return True

    def _try_activate_root_by_number(self, key: int, event: QKeyEvent) -> bool:
        """Root menu: ``1``–``4`` activate Files / Conversations / Tools / Commands."""
        idx = -1
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_4:
            idx = key - Qt.Key.Key_1
        else:
            keypad = getattr(Qt.Key, "Keypad1", None)
            if keypad is not None and Qt.Key.Keypad1 <= key <= Qt.Key.Keypad4:
                idx = key - Qt.Key.Keypad1
        if idx < 0 or idx >= self._list.count():
            return False
        item = self._list.item(idx)
        if item is None or not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return False
        self._list.setCurrentRow(idx)
        self._activate_current_item()
        event.accept()
        return True

    def hideEvent(self, event):
        super().hideEvent(event)
        # Reset mode so a new ``@`` always opens the root menu, not a stale drill-down.
        self._mode = None
        self._set_drill_chrome_visible(False)
        self.dismissed.emit()
