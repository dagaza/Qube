from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QLineEdit, QPushButton, QListWidget, QScrollArea, QSizePolicy, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QEvent
import qtawesome as qta
import logging
from urllib.parse import unquote
import copy
import unicodedata
import re as _re_cite

from ui.components.prestige_dialog import PrestigeDialog
from ui.components.source_viewer import SourcePreviewer

logger = logging.getLogger("Qube.UI.Conversations")

# In-text citation links must survive Qt's Markdown → anchor step. Qt's importer is flaky for
# custom schemes with numeric path segments (qube://cite/1); https + .invalid is linkified reliably.
CITATION_HREF_PREFIX = "https://qube.invalid/cite/"

# log_agent_token(..., citation_sources=...) — distinguish "not passed" (live stream) from explicit None (DB row with no sources)
_UNSET_SOURCES = object()


def _normalize_citation_id(value) -> str:
    """Single canonical form for matching cite tokens across JSON (int/float/str), Qt URLs, and LLM [W]."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).strip()
    s = str(value).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except ValueError:
        return s


def _source_citation_match_keys(src: dict) -> set[str]:
    """Normalized id values to compare against a clicked cite token (handles alternate keys)."""
    out: set[str] = set()
    if not isinstance(src, dict):
        return out
    for key in ("id", "cite_id", "source_id"):
        if key not in src:
            continue
        n = _normalize_citation_id(src.get(key))
        if n:
            out.add(n)
    return out


def _normalize_stored_source_id(src: dict) -> None:
    """Ensure citation ids are JSON-stable scalars; web citations use the string 'W' (not list/tuple)."""
    if not isinstance(src, dict):
        return
    rid = src.get("id")
    if isinstance(rid, (list, tuple)) and len(rid) == 1:
        rid = rid[0]
        src["id"] = rid
    if isinstance(rid, str) and rid.strip().upper() == "W":
        src["id"] = "W"
        return
    st = str(src.get("type", "")).lower()
    if st == "web" and rid in (None, ""):
        src["id"] = "W"


def _snapshot_citation_sources(sources) -> list:
    """Deep copy so each bubble owns an isolated list/dict graph (no cross-bubble mutation)."""
    if not sources:
        return []
    out = copy.deepcopy(list(sources))
    for src in out:
        _normalize_stored_source_id(src)
    return out


def _prepare_stream_for_qt_citation_links(raw: str) -> str:
    """
    Normalize citations before QLabel Markdown linkify. Later turns often emit
    [1](url), [[1]], or `[1]` which would otherwise stack with our link syntax and break Qt.
    """
    if not raw:
        return raw
    s = unicodedata.normalize("NFKC", raw)
    s = s.replace("\uff3b", "[").replace("\uff3d", "]")
    # Model-authored markdown links for citations → plain [n] / [W]
    s = _re_cite.sub(
        r"\[(\d+|[wW])\]\([^\)]*\)",
        lambda m: "[W]" if m.group(1).lower() == "w" else f"[{m.group(1)}]",
        s,
    )
    # Double-bracket wrappers e.g. [[1]] → [1] (repeat — models sometimes nest)
    for _ in range(4):
        ns = _re_cite.sub(
            r"\[\[(\d+|[wW])\]\]",
            lambda m: "[W]" if m.group(1).lower() == "w" else f"[{m.group(1)}]",
            s,
        )
        if ns == s:
            break
        s = ns

    def _unwrap_bt(m):
        inner = m.group(1)
        return "[W]" if inner.lower() == "[w]" else inner

    s = _re_cite.sub(r"`(\[\d+\]|\[[wW]\])`", _unwrap_bt, s)
    return s


def _markdown_cite_link_replacement(match) -> str:
    token = match.group(1)
    key = "W" if str(token).lower() == "w" else str(token)
    return f"[[{key}]](<{CITATION_HREF_PREFIX}{key}>)"


class ChatLabel(QLabel):
    """A QLabel that remembers its unwrapped width to force native layout expansion."""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.setMinimumWidth(0) # CRITICAL: Allows Qt to shrink the label gracefully
        
        self._cached_text = ""
        self._cached_ideal_width = 0
        self._citation_sources: list = []
        self._conversations_view = None

    def attach_citation_handling(self, conversations_view):
        """Bind link clicks to this label's own snapshot (no sender() / late-binding issues)."""
        self._conversations_view = conversations_view
        self.setOpenExternalLinks(False)
        self.linkActivated.connect(self._on_citation_link_activated)

    def _on_citation_link_activated(self, link_text: str):
        view = self._conversations_view
        if view is not None and hasattr(view, "_resolve_citation_link_for_label"):
            view._resolve_citation_link_for_label(self, link_text)

    def sizeHint(self):
        from PyQt6.QtGui import QTextDocument
        from PyQt6.QtCore import QSize, Qt
        
        hint = super().sizeHint()
        current_text = self.text()
        
        # Calculate the pure, unwrapped width of the text so the layout knows we WANT to expand
        if current_text != self._cached_text:
            doc = QTextDocument()
            if self.textFormat() == Qt.TextFormat.MarkdownText:
                doc.setMarkdown(current_text)
            else:
                doc.setPlainText(current_text)
                
            doc.setDefaultFont(self.font())
            self._cached_ideal_width = int(doc.idealWidth()) + 15 # 15px buffer for safety
            self._cached_text = current_text
        
        # Tell the layout to give us the unwrapped width (capped eventually by maximumWidth)
        return QSize(max(self._cached_ideal_width, hint.width()), hint.height())

    def heightForWidth(self, w: int) -> int:
        return super().heightForWidth(w)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateGeometry()
class MessageWrapper(QWidget):
    """An autonomous layout row that takes full width and safely manages bubble expansion."""
    def __init__(self, bubble: QWidget, is_user: bool, parent=None):
        super().__init__(parent)
        self.bubble = bubble
        self.is_user = is_user  # 🔑 Save this state to use during resizing
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if self.is_user:
            # User: Pushed to the right by the stretch
            layout.addStretch(1)
            layout.addWidget(bubble, 0)
        else:
            # 🔑 Agent: Give the bubble all the stretch so it fills the screen
            layout.addWidget(bubble, 1)
            # We removed the layout.addStretch(1) that was trapping it on the left!

    def resizeEvent(self, event):
        super().resizeEvent(event)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._update_width)

    def _update_width(self):
        # 🔑 User gets capped at 80% of the screen. Agent gets the full 100%.
        ratio = 0.8 if self.is_user else 1.0
        safe_max = max(int(self.width() * ratio), 100)
        
        self.bubble.setMaximumWidth(safe_max)


class ConversationsView(QWidget):
    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        self.llm = workers.get("llm")
        self.tts = workers.get("tts")
        self._pending_citation_sources = None
        self._user_turn_id = 0

        self._setup_ui()
        self._start_new_chat() 

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1) 

        self.history_pane = self._build_history_pane()
        layout.addWidget(self.history_pane)

        self.chat_stage = self._build_chat_stage()

        # 🔑 THE FIX: Lock the horizontal boundaries of the chat stage
        self.chat_stage.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        layout.addWidget(self.chat_stage, stretch=1) 

        # --- ADD THIS LINE AT THE BOTTOM ---
        # Forces the buttons to load with the default Dark Mode purple on startup
        self.refresh_button_themes(is_dark=True)

    # --------------------------------------------------------- #
    #  PANEL BUILDERS                                           #
    # --------------------------------------------------------- #

    def _build_history_pane(self) -> QFrame:
        frame = QFrame()
        frame.setFixedWidth(280)
        frame.setObjectName("HistorySidebar")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)

        header_layout = QHBoxLayout()
        
        # --- THE FIX: Change 'title' to 'self.list_title' ---
        self.list_title = QLabel("CONVERSATIONS")
        self.list_title.setProperty("class", "SidebarTitle")
        
        self.new_chat_btn = QPushButton()
        self.new_chat_btn.setIcon(qta.icon('fa5s.plus'))
        self.new_chat_btn.setProperty("class", "IconButton")
        
        # --- THE FIX: Make sure you add 'self.list_title' to the layout here ---
        header_layout.addWidget(self.list_title)
        
        header_layout.addStretch()
        header_layout.addWidget(self.new_chat_btn)
        layout.addLayout(header_layout)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search history...")
        self.search_bar.setObjectName("HistorySearch")
        layout.addWidget(self.search_bar)

        self.history_list = QListWidget()
        self.history_list.setObjectName("HistoryList")
        layout.addWidget(self.history_list)

        self.new_chat_btn.clicked.connect(self._start_new_chat)
        self.history_list.itemClicked.connect(self._load_selected_chat)
        self.history_list.itemSelectionChanged.connect(self._update_row_colors)

        # 🔑 NEW: Wire up the scrollbar for infinite scrolling
        self.history_offset = 0
        self.is_loading_history = False
        self.history_list.verticalScrollBar().valueChanged.connect(self._on_history_scroll)

        return frame

    def _build_chat_stage(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("ChatStage")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        # 1. The New Architecture: A Scroll Area containing a vertical list of message widgets
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("ChatScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.installEventFilter(self)

        # Container widget
        self.transcript_container = QWidget()
        self.transcript_container.setObjectName("ChatTranscriptContainer")
        
        # 🔑 The line that was crashing is perfectly safe now
        self.transcript_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.transcript_layout = QVBoxLayout(self.transcript_container)
        self.transcript_layout.setContentsMargins(20, 20, 20, 20)
        self.transcript_layout.setSpacing(25)
        self.transcript_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.transcript_container)
        layout.addWidget(self.scroll_area)

        # 2. Input Bar Area
        input_container = QFrame()
        input_container.setObjectName("ChatInputContainer")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(10, 5, 5, 5)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type a message to Qube...")
        self.text_input.setObjectName("ChatTextInput")
        
        self.send_btn = QPushButton()
        self.send_btn.setIcon(qta.icon('fa5s.paper-plane'))
        self.send_btn.setFixedSize(35, 35)
        self.send_btn.setProperty("class", "SendButton")

        input_layout.addWidget(self.text_input)
        input_layout.addWidget(self.send_btn)
        layout.addWidget(input_container)

        # 3. Latency Metrics Footer
        latency_layout = QHBoxLayout()
        self.stt_latency_lbl = QLabel("STT: -- ms")
        self.ttft_latency_lbl = QLabel("TTFT: -- ms")
        self.tts_latency_lbl = QLabel("TTS: -- ms")
        
        for lbl in [self.stt_latency_lbl, self.ttft_latency_lbl, self.tts_latency_lbl]:
            lbl.setProperty("class", "MiniLatencyLabel")
            latency_layout.addWidget(lbl)
            
        latency_layout.addStretch()
        layout.addLayout(latency_layout)

        self.send_btn.clicked.connect(self._handle_text_submit)
        self.text_input.returnPressed.connect(self._handle_text_submit)
        
        return frame
    
    # --------------------------------------------------------- #
    #  UI UPDATE RECEIVERS (The Magic Happens Here)             #
    # --------------------------------------------------------- #

    def log_user_message(self, text: str) -> None:
        self._clear_placeholders()
        # New user turn: drop stale assistant pointer so Turn N+1 tools cannot overwrite Turn N bubbles.
        self._user_turn_id += 1
        self.current_agent_msg = None

        bubble = QFrame()
        # 🔑 FIX 2: Allow bubble to expand horizontally, minimum vertically
        bubble.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        bubble.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bubble.setStyleSheet("background-color: #89b4fa; border-radius: 18px;")

        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(16, 12, 16, 12)

        lbl = ChatLabel(text)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl.setStyleSheet("background: transparent; border: none; font-size: 14px; color: #11111b;")
        
        bubble_layout.addWidget(lbl)

        wrapper = MessageWrapper(bubble, is_user=True)
        self.transcript_layout.addWidget(wrapper)
        
        # Initial width enforcement
        wrapper._update_width()

        self._is_agent_typing = False
        self._scroll_to_bottom()

    def log_agent_token(self, token: str, *, citation_sources=_UNSET_SOURCES) -> None:
        self._clear_placeholders()

        is_dark = True
        if self.window() and hasattr(self.window(), '_is_dark_theme'):
            is_dark = self.window()._is_dark_theme
            
        text_color = "#cdd6f4" if is_dark else "#1e293b"
        header_color = "#8b5cf6" if is_dark else "#8839ef" 

        if not getattr(self, '_is_agent_typing', False):
            header = QLabel("QUBE")
            header.setStyleSheet(f"color: {header_color}; font-weight: bold; font-size: 11px; margin-top: 15px; background: transparent;")
            self.transcript_layout.addWidget(header)

            self.agent_msg_container = QFrame()
            self.agent_msg_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            
            container_layout = QVBoxLayout(self.agent_msg_container)
            container_layout.setContentsMargins(0, 0, 0, 20)

            self.current_agent_msg = ChatLabel()
            self.current_agent_msg._assistant_turn_id = self._user_turn_id
            self.current_agent_msg.setTextFormat(Qt.TextFormat.MarkdownText)
            
            # 🔑 CHANGE 1: Enable Link Interaction
            self.current_agent_msg.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextBrowserInteraction | 
                Qt.TextInteractionFlag.LinksAccessibleByMouse
            )
            self.current_agent_msg.attach_citation_handling(self)
            
            from PyQt6.QtGui import QPalette, QColor
            palette = self.current_agent_msg.palette()
            palette.setColor(QPalette.ColorRole.WindowText, QColor(text_color))
            palette.setColor(QPalette.ColorRole.Text, QColor(text_color))
            self.current_agent_msg.setPalette(palette)
            self.current_agent_msg.setStyleSheet("font-size: 14px; background: transparent; border: none;")

            # Per-bubble citation context (survives new turns and session reloads)
            if citation_sources is not _UNSET_SOURCES:
                self.current_agent_msg._citation_sources = _snapshot_citation_sources(citation_sources)
            else:
                self._attach_pending_citation_sources(self.current_agent_msg)

            container_layout.addWidget(self.current_agent_msg)

            wrapper = MessageWrapper(self.agent_msg_container, is_user=False)
            self.transcript_layout.addWidget(wrapper)
            
            wrapper._update_width()

            self._agent_text_buffer = ""
            self._is_agent_typing = True

        self._agent_text_buffer += token

        # Strip model markdown around cites, then linkify plain [n]/[W] for Qt MarkdownText.
        prepared = _prepare_stream_for_qt_citation_links(self._agent_text_buffer)
        rich_text = _re_cite.sub(
            r"\[\s*(\d+|[wW])\s*\]",
            _markdown_cite_link_replacement,
            prepared,
        )

        self.current_agent_msg.setText(rich_text)

        self.current_agent_msg.updateGeometry()
        self._scroll_to_bottom()

    def _clear_placeholders(self):
        if hasattr(self, 'placeholder_lbl') and self.placeholder_lbl:
            self.placeholder_lbl.hide()
            self.placeholder_lbl.deleteLater()
            self.placeholder_lbl = None

    def _clear_transcript(self):
        """Destroys all message widgets to prepare for a new chat."""
        while self.transcript_layout.count():
            child = self.transcript_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self._pending_citation_sources = None
        self.current_agent_msg = None

    # --------------------------------------------------------- #
    #  INTERACTION & LOGIC                                      #
    # --------------------------------------------------------- #

    def _handle_text_submit(self):
        text = self.text_input.text().strip()
        if not text: return
        self.text_input.clear()
        self.set_input_enabled(False)

        self.log_user_message(text)

        if not hasattr(self, 'active_session_id'):
            recent_sessions = self.db.get_recent_sessions(limit=1)
            if recent_sessions:
                self.active_session_id = recent_sessions[0]['id']
            else:
                self.active_session_id = self.db.create_session("Text Conversation")

        if self.llm:
            self.llm.generate_response(text, self.active_session_id)

    def update_stt_latency(self, ms: float) -> None:
        self.stt_latency_lbl.setText(f"STT: {ms:.0f} ms")

    def update_ttft_latency(self, ms: float) -> None:
        self.ttft_latency_lbl.setText(f"TTFT: {ms:.0f} ms")

    def update_tts_latency(self, ms: float) -> None:
        self.tts_latency_lbl.setText(f"TTS: {ms:.0f} ms")

    def _refresh_history_list(self):
        """Clears the list, resets the offset, and pulls the first batch."""
        self.history_list.clear()
        self.history_offset = 0
        self.is_loading_history = False

        # Silent garbage collection for empty sessions
        current_active = getattr(self, 'active_session_id', None)
        if hasattr(self.db, 'cleanup_empty_sessions'):
            self.db.cleanup_empty_sessions(current_active)

        # Update the Title Count
        count = self.db.get_session_count()
        display_count = "999+" if count > 999 else str(count)
        if hasattr(self, 'list_title'):
            self.list_title.setText(f"CONVERSATIONS ({display_count})")
            
        # Load the initial batch!
        self._load_history_batch()

    def _on_history_scroll(self, value):
        """Triggered every time the user scrolls."""
        bar = self.history_list.verticalScrollBar()
        # If the scrollbar hits the absolute maximum, load the next batch!
        if value == bar.maximum():
            self._load_history_batch()

    def _load_history_batch(self):
        """Fetches the next chunk of history and appends it to the list."""
        if getattr(self, 'is_loading_history', False):
            return # Prevent spamming the database if already loading
            
        self.is_loading_history = True

        from PyQt6.QtWidgets import QListWidgetItem, QWidget, QHBoxLayout, QLabel, QPushButton, QMenu
        from PyQt6.QtCore import Qt, QSize
        import qtawesome as qta
        
        # 1. Determine theme state
        is_dark = True
        main_win = self.window()
        if main_win and hasattr(main_win, '_is_dark_theme'):
            is_dark = main_win._is_dark_theme
        
        text_color = "#cdd6f4" if is_dark else "#1e293b"
        icon_color = "#6c7086" if is_dark else "#64748b"
        
        # 🔑 THE FIX: Pass both the limit AND the current offset to the DB
        # Note: If your DB doesn't support 'offset' yet, we'll need to update it!
        try:
            sessions = self.db.get_recent_sessions(limit=20, offset=self.history_offset)
        except TypeError:
            # Fallback just in case you haven't updated the DB manager yet
            sessions = self.db.get_recent_sessions(limit=20)
            if self.history_offset > 0:
                sessions = [] # Prevent infinite looping of the same 20 items

        if not sessions:
            self.is_loading_history = False
            return

        for session in sessions:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, session["id"])
            
            row_widget = QWidget()
            row_widget.setObjectName("HistoryRowWidget")
            row_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(15, 0, 10, 0)
            row_layout.setSpacing(10)
            
            title_lbl = QLabel(session["title"])
            title_lbl.setObjectName("HistoryRowTitle")
            title_lbl.setStyleSheet(f"color: {text_color}; background: transparent; border: none; font-size: 13px; font-weight: 500;")
            
            opts_btn = QPushButton()
            opts_btn.setObjectName("HistoryOptionsBtn")
            opts_btn.setFixedSize(28, 28)
            opts_btn.setIcon(qta.icon('fa5s.ellipsis-v', color=icon_color))
            opts_btn.setIconSize(QSize(16, 16))
            opts_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus) 
            opts_btn.setStyleSheet("QPushButton::menu-indicator { image: none; width: 0px; } QPushButton { border: none; background: transparent; }")
            
            menu = QMenu(opts_btn)
            if hasattr(self, '_apply_menu_theme'):
                 self._apply_menu_theme(menu, is_dark)

            rename_action = menu.addAction(qta.icon('fa5s.edit', color='#89b4fa'), "Rename Chat")
            rename_action.triggered.connect(lambda _, s_id=session["id"], old_t=session["title"]: self._trigger_rename_chat(s_id, old_t))
            menu.addSeparator()

            delete_action = menu.addAction(qta.icon('fa5s.trash-alt', color='#ef4444'), "Delete Chat")
            delete_action.triggered.connect(lambda _, s_id=session["id"]: self._trigger_delete_chat(s_id))
            opts_btn.setMenu(menu)
            
            row_layout.addWidget(title_lbl)
            row_layout.addStretch()
            row_layout.addWidget(opts_btn)
            
            item.setSizeHint(QSize(0, 45))
            self.history_list.addItem(item)
            self.history_list.setItemWidget(item, row_widget)

        # 🔑 Increment the offset so the next scroll fetches the NEXT 20
        self.history_offset += 20
        self.is_loading_history = False

    def _update_row_colors(self):
        """Forces text color changes since Qt CSS cannot pass :selected states to setItemWidget."""
        from PyQt6.QtWidgets import QLabel
        
        # 1. Safely Detect Theme
        is_dark = getattr(self.window(), '_is_dark_theme', True)
            
        # 2. 🔑 THE FIX: The Colors!
        # Unselected: Light gray in Dark Mode, Slate in Light Mode
        normal_color = "#cdd6f4" if is_dark else "#1e293b"
        # Selected: The background bubble is solid, so text should ALWAYS be White
        selected_color = "#ffffff" 

        # 3. Target whichever list is in this specific file
        target_list = getattr(self, 'doc_list', getattr(self, 'history_list', None))
        if not target_list: 
            return

        # 4. Loop through and forcefully apply the correct color
        for i in range(target_list.count()):
            item = target_list.item(i)
            widget = target_list.itemWidget(item)
            if widget:
                lbl = widget.findChild(QLabel) 
                if lbl:
                    color = selected_color if item.isSelected() else normal_color
                    lbl.setStyleSheet(f"color: {color}; background: transparent; border: none; font-size: 13px; font-weight: 500;")

    def _trigger_delete_chat(self, session_id):
        """Modern confirmation with full original safety logic."""
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        
        # 1. Use the Prestige UI instead of QMessageBox
        dlg = PrestigeDialog(
            self, 
            "Delete Conversation", 
            "Are you sure you want to permanently delete this chat? This cannot be undone.", 
            is_dark
        )
        
        if dlg.exec():
            # 2. Keep your original Database Guardrail
            if hasattr(self.db, 'delete_session'):
                self.db.delete_session(session_id)
            else:
                logger.error(f"CRITICAL: DB Manager missing 'delete_session' method. Cannot remove {session_id}.")
                return

            # 3. Keep your original UI State Management
            if getattr(self, 'active_session_id', None) == session_id:
                # If they deleted the active chat, reset the view
                self._start_new_chat()
            else:
                # Otherwise, just update the sidebar
                self._refresh_history_list()

    def _trigger_rename_chat(self, session_id, old_title):
        """Modern input with full original validation logic."""
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        
        # 1. Use Prestige UI instead of QInputDialog
        dlg = PrestigeDialog(
            self, 
            "Rename Conversation", 
            "Enter a new title for this chat:", 
            is_dark, 
            is_input=True, 
            default_text=old_title
        )
        
        # 2. Keep your 'ok' and 'strip' validation
        if dlg.exec() and dlg.result_text and dlg.result_text.strip():
            new_title = dlg.result_text.strip()
            
            # 3. Keep your original Database Guardrail
            if hasattr(self.db, 'rename_session'):
                self.db.rename_session(session_id, new_title)
                self._refresh_history_list()
            else:
                logger.error("CRITICAL: DB Manager missing 'rename_session' method.")

    def _start_new_chat(self):
        self.active_session_id = self.db.create_session("New Conversation")
        self._clear_transcript()

        self.placeholder_lbl = QLabel("New chat started. Type or speak a message!")
        self.placeholder_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_lbl.setStyleSheet(
            "color: #6c7086; font-size: 16px; margin-top: 50px; font-weight: bold;"
        )
        self.transcript_layout.addWidget(self.placeholder_lbl)

        self._is_agent_typing = False
        self._refresh_history_list()

    def _load_selected_chat(self, item):
        from PyQt6.QtCore import Qt
        session_id = item.data(Qt.ItemDataRole.UserRole)
        self.active_session_id = session_id

        self._clear_transcript()
        self._is_agent_typing = False

        history = self.db.get_session_history(session_id)
        if not history:
            self.placeholder_lbl = QLabel("Empty conversation.")
            self.placeholder_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.placeholder_lbl.setStyleSheet(
                "color: #6c7086; font-size: 16px; margin-top: 50px; font-weight: bold;"
            )
            self.transcript_layout.addWidget(self.placeholder_lbl)
            return

        for msg in history:
            if msg["role"] == "user":
                self.log_user_message(msg["content"])
            elif msg["role"] == "assistant":
                self.log_agent_token(
                    msg["content"],
                    citation_sources=msg.get("sources"),
                )
                self._is_agent_typing = False

    def _apply_menu_theme(self, menu, is_dark: bool):
        """Standardizes the menu appearance to match the Prestige theme."""
        from PyQt6.QtGui import QPalette, QColor
        # THIS IS THE MAGIC LINE TO KILL THE GHOST SQUARE
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        palette = QPalette()
        if is_dark:
            bg, fg, sel_bg, sel_fg = "#1e1e2e", "#cdd6f4", "#313244", "#cdd6f4"
            border, hover = "rgba(255, 255, 255, 0.1)", "#313244"
        else:
            bg, fg, sel_bg, sel_fg = "#ffffff", "#1e293b", "#f1f5f9", "#0f172a"
            border, hover = "#cbd5e1", "#f1f5f9"

        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
            palette.setColor(role, QColor(bg))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(fg))
        palette.setColor(QPalette.ColorRole.Text, QColor(fg))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(sel_bg))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(sel_fg))

        menu.setPalette(palette)
        menu.setStyleSheet(f"""
            QMenu {{ 
                background-color: {bg}; 
                color: {fg}; 
                border: 1px solid {border}; 
                border-radius: 12px; 
                padding: 5px; 
            }}
            QMenu::item {{ 
                background-color: transparent; 
                padding: 8px 25px; 
                border-radius: 8px; 
            }}
            QMenu::item:selected {{ 
                background-color: {hover}; 
                color: {sel_fg}; 
            }}
        """)

    def refresh_menu_themes(self, is_dark: bool):
        """Updates all existing kebab menus in the history list."""
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            widget = self.history_list.itemWidget(item)
            if widget:
                # Find the button and its menu
                btn = widget.findChild(QPushButton, "HistoryOptionsBtn")
                if btn and btn.menu():
                    self._apply_menu_theme(btn.menu(), is_dark)

    def refresh_button_themes(self, is_dark: bool):
        """Dynamically updates the colors of the New Chat and Send buttons."""
        import qtawesome as qta
        
        # Icon color: Catppuccin Purple in Dark Mode, Deep Slate in Light Mode
        icon_color = "#8b5cf6" if is_dark else "#1e293b"
        
        # Subtle hover background: faint white wash for Dark, faint black wash for Light
        hover_bg = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.05)"
        
        # 1. Update New Chat Button (+)
        if hasattr(self, 'new_chat_btn'):
            self.new_chat_btn.setIcon(qta.icon('fa5s.plus', color=icon_color))
            self.new_chat_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; border-radius: 6px; padding: 6px; }}
                QPushButton:hover {{ background-color: {hover_bg}; }}
            """)
            
        # 2. Update Send Button (Paper Plane)
        if hasattr(self, 'send_btn'):
            self.send_btn.setIcon(qta.icon('fa5s.paper-plane', color=icon_color))
            self.send_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; border-radius: 6px; padding: 6px; }}
                QPushButton:hover {{ background-color: {hover_bg}; }}
            """)
    
    def eventFilter(self, obj, event):
        """Native resize handling without fighting Qt's geometry engine."""
        from PyQt6.QtCore import QEvent, QTimer
        
        if obj == self.scroll_area and event.type() == QEvent.Type.Resize:
            if hasattr(self, 'transcript_layout'):
                max_w = int(self.scroll_area.viewport().width() * 0.8)
                
                # Because we removed wrappers, the items ARE the bubbles!
                for i in range(self.transcript_layout.count()):
                    widget = self.transcript_layout.itemAt(i).widget()
                    if widget and widget.objectName() in ["UserBubble", "AgentBubble"]:
                        widget.setMaximumWidth(max_w)
            
            # Defer scroll to bottom so Qt's native layout math finishes
            QTimer.singleShot(0, self._scroll_to_bottom)
            
        return super().eventFilter(obj, event)

    def _scroll_to_bottom(self):
        """Native deferred scrolling ensuring layout pass completion."""
        def _execute_scroll():
            bar = self.scroll_area.verticalScrollBar()
            bar.setValue(bar.maximum())
            
        from PyQt6.QtCore import QTimer
        # Wait for geometry calculation, THEN wait for layout application
        QTimer.singleShot(0, lambda: QTimer.singleShot(0, _execute_scroll))

    def _attach_pending_citation_sources(self, label: ChatLabel) -> None:
        """Apply sources from the tool phase to this bubble (or [] if none pending)."""
        pending = getattr(self, "_pending_citation_sources", None)
        if pending is not None:
            label._citation_sources = _snapshot_citation_sources(pending)
            self._pending_citation_sources = None
        else:
            label._citation_sources = []

    def on_sources_found(self, sources):
        """Receive tool sources for inline citation links (no separate chip UI)."""
        self._pending_citation_sources = _snapshot_citation_sources(sources)
        cur = getattr(self, "current_agent_msg", None)
        if cur is not None and getattr(cur, "_assistant_turn_id", None) == getattr(
            self, "_user_turn_id", -1
        ):
            cur._citation_sources = _snapshot_citation_sources(sources)

    def _resolve_citation_link_for_label(self, label: ChatLabel, link_text: str):
        """Resolve href from this bubble's isolated _citation_sources (label-bound, not sender())."""
        raw = unquote((link_text or "").strip())
        sources = getattr(label, "_citation_sources", None) or []

        def _resolve_and_open(source_id: str) -> bool:
            wanted = _normalize_citation_id((source_id or "").strip())
            if not wanted:
                return True
            for src in sources:
                if wanted in _source_citation_match_keys(src):
                    self.open_source_preview(src)
                    return True
            ids_debug = [
                sorted(_source_citation_match_keys(s)) if isinstance(s, dict) else repr(type(s))
                for s in sources[:5]
            ]
            logger.warning(
                "Citation id %r (normalized %r) not found on this message (%d sources); sample ids %s",
                source_id,
                wanted,
                len(sources),
                ids_debug,
            )
            return True

        if raw.startswith(CITATION_HREF_PREFIX):
            tail = raw[len(CITATION_HREF_PREFIX) :].split("?")[0].split("#")[0].rstrip("/").strip()
            _resolve_and_open(tail)
            return

        if raw.startswith("qube://cite/"):
            tid = raw[len("qube://cite/") :].split("?")[0].split("#")[0].strip()
            _resolve_and_open(tid)
            return

        if raw.startswith("source_"):
            _resolve_and_open(raw.replace("source_", "", 1))
            return

        if raw.startswith("http://") or raw.startswith("https://"):
            import webbrowser

            webbrowser.open(raw)
            return

    def open_source_preview(self, source_dict):
        """Opens the Prestige-styled SourcePreviewer (see ui/components/prestige_dialog.py)."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        viewer = SourcePreviewer(
            source_dict.get("filename", "Source"),
            source_dict.get("content", ""),
            self,
            is_dark=is_dark,
        )
        viewer.show()

    def clear_stale_agent_pointer(self) -> None:
        """Drop the live assistant label handle (e.g. interrupt) without advancing _user_turn_id."""
        self.current_agent_msg = None

    def set_input_enabled(self, enabled: bool):
        """Locks the text input bar and resets its placeholder."""
        if hasattr(self, 'text_input') and hasattr(self, 'send_btn'):
            self.text_input.setEnabled(enabled)
            self.send_btn.setEnabled(enabled)
            
            if enabled:
                # 🔑 RESET: Back to normal
                self.text_input.setPlaceholderText("Type a message to Qube...")
                self.text_input.setFocus()
            else:
                # 🔑 DEFAULT LOCK: We will update this dynamically in a millisecond
                self.text_input.setPlaceholderText("Qube is thinking...")

    # 🔑 NEW: A dynamic receiver to update the text box live
    def update_action_placeholder(self, status: str):
        """Dynamically updates the text box placeholder based on worker status."""
        if not self.text_input.isEnabled() and status != "Idle":
            # Just clean up the string to sound natural (add an ellipsis if missing)
            display_text = status if "..." in status else f"{status}..."
            self.text_input.setPlaceholderText(display_text)