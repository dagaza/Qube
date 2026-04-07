from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
    QLineEdit, QPushButton, QListWidget, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
import qtawesome as qta
import logging

logger = logging.getLogger("Qube.UI.Conversations")

class ConversationsView(QWidget):
    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        self.llm = workers.get("llm")
        self.tts = workers.get("tts")
        
        self._setup_ui()
        self._start_new_chat() 

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1) 

        self.history_pane = self._build_history_pane()
        layout.addWidget(self.history_pane)

        self.chat_stage = self._build_chat_stage()
        layout.addWidget(self.chat_stage, stretch=1) 

    # --------------------------------------------------------- #
    #  PANEL BUILDERS                                           #
    # --------------------------------------------------------- #

    def _build_history_pane(self) -> QFrame:
        frame = QFrame()
        frame.setFixedWidth(250)
        frame.setObjectName("HistorySidebar")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)

        header_layout = QHBoxLayout()
        title = QLabel("CONVERSATIONS")
        title.setProperty("class", "SidebarTitle")
        
        self.new_chat_btn = QPushButton()
        self.new_chat_btn.setIcon(qta.icon('fa5s.plus'))
        self.new_chat_btn.setProperty("class", "IconButton")
        
        header_layout.addWidget(title)
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

        self.transcript_container = QWidget()
        self.transcript_container.setObjectName("ChatTranscriptContainer")
        self.transcript_layout = QVBoxLayout(self.transcript_container)
        self.transcript_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.transcript_layout.setContentsMargins(20, 20, 20, 20)
        self.transcript_layout.setSpacing(20)

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
        """Creates a standalone right-aligned bubble with guaranteed background rendering.

        Root cause: Qt's global QSS cascade does not reliably reach widgets that are
        created dynamically and inserted into a nested QScrollArea after boot. The OS
        native theme engine intercepts the paint event before our #UserBubble rule can
        apply. Fix: set the style inline via setStyleSheet(), which writes directly into
        the widget's C++ memory and cannot be overridden by the cascade or the OS theme.
        WA_StyledBackground is still required — it tells Qt that this QWidget (which is
        normally fully transparent) *does* own a painted background.
        """
        self._clear_placeholders()

        # Row widget: transparent wrapper that right-aligns the bubble
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addStretch()

        # Bubble container: inline style bypasses the broken global cascade
        bubble_container = QWidget()
        bubble_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bubble_container.setStyleSheet(
            "background-color: #89b4fa;"   # Catppuccin Blue
            "border-radius: 18px;"
        )
        bubble_container.setMaximumWidth(int(self.scroll_area.width() * 0.75))

        bubble_layout = QVBoxLayout(bubble_container)
        bubble_layout.setContentsMargins(16, 12, 16, 12)

        bubble_text = QLabel(text)
        bubble_text.setWordWrap(True)
        bubble_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # Colour lives here too — child QLabel inherits the bubble's inline scope,
        # so the global "#UserBubble QLabel" rule is also unreachable.
        bubble_text.setStyleSheet(
            "background-color: transparent;"
            "border: none;"
            "font-size: 14px;"
            "color: #11111b;"
        )

        bubble_layout.addWidget(bubble_text)
        row_layout.addWidget(bubble_container)
        self.transcript_layout.addWidget(row_widget)

        self._is_agent_typing = False
        self._scroll_to_bottom()

    def log_agent_token(self, token: str) -> None:
        """Streams agent tokens natively using Qt's Markdown renderer."""
        self._clear_placeholders()

        if not getattr(self, '_is_agent_typing', False):
            header = QLabel("QUBE")
            # Agent header has no background, so the global QSS colour rule is
            # enough — but we inline it too for consistency and safety.
            header.setStyleSheet(
                "color: #cba6f7;"
                "font-weight: bold;"
                "font-size: 11px;"
                "letter-spacing: 1px;"
                "margin-top: 15px;"
                "background-color: transparent;"
            )
            self.transcript_layout.addWidget(header)

            self.current_agent_msg = QLabel()
            self.current_agent_msg.setWordWrap(True)
            self.current_agent_msg.setTextFormat(Qt.TextFormat.MarkdownText)
            self.current_agent_msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
            self.current_agent_msg.setOpenExternalLinks(True)
            self.current_agent_msg.setStyleSheet(
                "color: #cdd6f4;"
                "font-size: 14px;"
                "line-height: 1.5;"
                "background-color: transparent;"
            )

            self.transcript_layout.addWidget(self.current_agent_msg)

            self._agent_text_buffer = ""
            self._is_agent_typing = True

        self._agent_text_buffer += token
        self.current_agent_msg.setText(self._agent_text_buffer)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """Forces the scroll area to the bottom, delaying slightly to let the layout math catch up."""
        QTimer.singleShot(10, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

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

    # --------------------------------------------------------- #
    #  INTERACTION & LOGIC                                      #
    # --------------------------------------------------------- #

    def _handle_text_submit(self):
        text = self.text_input.text().strip()
        if not text: return
        self.text_input.clear()
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
        """Pulls recent sessions from SQLite and populates the sidebar with custom widgets."""
        self.history_list.clear()
        from PyQt6.QtWidgets import QListWidgetItem, QWidget, QHBoxLayout, QLabel, QPushButton, QMenu
        from PyQt6.QtCore import Qt, QSize
        import qtawesome as qta
        
        sessions = self.db.get_recent_sessions(limit=20)
        for session in sessions:
            # 1. Create the invisible list item to hold the data
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, session["id"])
            
            # 2. Create the custom row widget
            row_widget = QWidget()
            row_widget.setObjectName("HistoryRowWidget")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(10, 5, 5, 5)
            row_layout.setSpacing(10)
            
            # 3. The Chat Title
            title_lbl = QLabel(session["title"])
            title_lbl.setStyleSheet("background: transparent; border: none; color: inherit;")
            
            # 4. The 3-Dot Options Button
            opts_btn = QPushButton()
            opts_btn.setIcon(qta.icon('fa5s.ellipsis-v', color='#6c7086'))
            opts_btn.setIconSize(QSize(14, 14))
            opts_btn.setFixedSize(24, 24)
            opts_btn.setObjectName("HistoryOptionsBtn")
            # Stop the button click from selecting the row underneath it
            opts_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus) 
            
            # 5. Build the Dropdown Menu
            menu = QMenu(opts_btn)
            
            # Determine theme for the menu palette (using our trusty prestige logic)
            is_dark = True
            main_win = self.window()
            if main_win and hasattr(main_win, '_is_dark_theme'):
                is_dark = main_win._is_dark_theme
            elif hasattr(self, '_is_dark_theme'):
                is_dark = self._is_dark_theme
            
            # Apply your theme fix to the menu if you brought that helper into this file
            if hasattr(self, '_apply_menu_theme'):
                 self._apply_menu_theme(menu, is_dark)
            else:
                 # Fallback basic styling if the helper isn't in this file
                 menu.setStyleSheet("QMenu { border-radius: 4px; padding: 4px; }")

            delete_action = menu.addAction(qta.icon('fa5s.trash-alt', color='#ef4444'), "Delete Chat")
            
            # Use a lambda with a default argument to "freeze" the session ID for this specific row
            delete_action.triggered.connect(lambda checked, s_id=session["id"]: self._trigger_delete_chat(s_id))
            
            opts_btn.setMenu(menu)
            
            # Assemble the row
            row_layout.addWidget(title_lbl)
            row_layout.addStretch()
            row_layout.addWidget(opts_btn)
            
            # 6. Crucial: Tell the QListWidgetItem exactly how big this custom widget is
            item.setSizeHint(row_widget.sizeHint())
            
            self.history_list.addItem(item)
            self.history_list.setItemWidget(item, row_widget)

    def _trigger_delete_chat(self, session_id):
        """Spawns a confirmation dialog and deletes the chat if confirmed."""
        from PyQt6.QtWidgets import QMessageBox
        
        # 1. The Safety Net
        reply = QMessageBox.question(
            self, 
            'Delete Conversation',
            "Are you sure you want to permanently delete this chat? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # 2. Delete from Database
            if hasattr(self.db, 'delete_session'):
                self.db.delete_session(session_id)
            else:
                logger.error("WARNING: Your db_manager does not have a 'delete_session' method!")
                return

            # 3. Handle the UI State
            if getattr(self, 'active_session_id', None) == session_id:
                # If they deleted the chat they are currently staring at, nuke the screen
                self._start_new_chat()
            else:
                # Otherwise, just quietly refresh the sidebar
                self._refresh_history_list()

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
                self.log_agent_token(msg["content"])
                self._is_agent_typing = False