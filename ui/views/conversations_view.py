from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QLineEdit, QPushButton, QListWidget, QScrollArea, QSizePolicy, QDialog
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QEvent
import qtawesome as qta
import logging

logger = logging.getLogger("Qube.UI.Conversations")

class ChatLabel(QLabel):
    """A QLabel that remembers its unwrapped width to force native layout expansion."""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.setMinimumWidth(0) # CRITICAL: Allows Qt to shrink the label gracefully
        
        self._cached_text = ""
        self._cached_ideal_width = 0

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
class PrestigeDialog(QDialog):
    def __init__(self, parent, title, message, is_dark=True, is_input=False, default_text=""):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # --- NEW: ENLARGED DIMENSIONS ---
        self.setMinimumWidth(450) # Increased from default
        
        self.result_text = None
        bg, fg = ("#1e1e2e", "#cdd6f4") if is_dark else ("#ffffff", "#1e293b")
        accent = "#f38ba8" if "Delete" in title else "#89b4fa"
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10) # Outer shadow/glow area
        
        self.container = QFrame()
        self.container.setObjectName("DialogContainer")
        self.container.setStyleSheet(f"""
            QFrame#DialogContainer {{ 
                background: {bg}; 
                border: 2px solid {accent}; 
                border-radius: 20px; 
            }}
            QLabel {{ color: {fg}; border: none; background: transparent; }}
        """)
        
        # Increased internal spacing and margins
        c_layout = QVBoxLayout(self.container)
        c_layout.setContentsMargins(30, 30, 30, 25) 
        c_layout.setSpacing(20) 
        
        # Header/Title
        t_lbl = QLabel(title.upper())
        t_lbl.setStyleSheet(f"color: {accent}; font-weight: bold; font-size: 12px; letter-spacing: 2px;")
        
        # Message (Increased font size)
        m_lbl = QLabel(message)
        m_lbl.setWordWrap(True)
        m_lbl.setStyleSheet(f"color: {fg}; font-size: 15px; line-height: 1.4;")
        
        c_layout.addWidget(t_lbl)
        c_layout.addWidget(m_lbl)
        
        # Input Field (Enlarged and Spaced)
        self.field = None
        if is_input:
            self.field = QLineEdit(default_text)
            self.field.setMinimumHeight(45) # Taller input field
            self.field.setStyleSheet(f"""
                QLineEdit {{ 
                    background: {'#313244' if is_dark else '#f8fafc'}; 
                    color: {fg}; 
                    border-radius: 10px; 
                    padding: 10px 15px; 
                    border: 1px solid {accent};
                    font-size: 14px;
                }}
            """)
            c_layout.addWidget(self.field)
            self.field.setFocus()

       # --- ENHANCED BUTTON STYLING ---
        btns = QHBoxLayout()
        btns.setSpacing(15)
        
        cancel_btn = QPushButton("CANCEL")
        con_b = QPushButton("CONFIRM")
        
        # Increased vertical padding (15px) and added min-height (45px)
        btn_style = f"""
            QPushButton {{ 
                padding: 15px 15px; 
                min-height: 30px;
                border-radius: 12px; 
                font-weight: bold; 
                font-size: 12px;
                letter-spacing: 1px;
            }}
        """
        
        cancel_btn.setStyleSheet(btn_style + f"""
            QPushButton {{ 
                color: {fg}; 
                border: 1px solid {border}; 
                background: transparent; 
            }}
            QPushButton:hover {{
                background: rgba(255, 255, 255, 0.05);
            }}
        """)
        
        con_b.setStyleSheet(btn_style + f"""
            QPushButton {{ 
                background: {accent}; 
                color: #11111b; 
                border: none; 
            }}
            QPushButton:hover {{
                background: {accent}; /* You could add a slightly brighter hex here if desired */
                opacity: 0.9;
            }}
        """)
        
        cancel_btn.clicked.connect(self.reject)
        con_b.clicked.connect(self.accept)
        
        btns.addStretch()
        btns.addWidget(cancel_btn)
        btns.addWidget(con_b)
        c_layout.addLayout(btns)
        
        layout.addWidget(self.container)

    def exec(self):
        """Returns the input text if Accepted and is_input=True, otherwise True/None."""
        if super().exec():
            return self.field.text().strip() if self.field else True
        return None

    def accept_action(self):
        if self.input_field:
            self.result_text = self.input_field.text()
        self.accept()
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

        #Container widget
        self.transcript_container = QWidget()
        self.transcript_container.setObjectName("ChatTranscriptContainer")
        
        # 🔑 FIX 5: Ensure scroll area truly allows horizontal expansion
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

    def log_agent_token(self, token: str) -> None:
        self._clear_placeholders()

        is_dark = True
        if self.window() and hasattr(self.window(), '_is_dark_theme'):
            is_dark = self.window()._is_dark_theme
            
        text_color = "#cdd6f4" if is_dark else "#1e293b"
        header_color = "#cba6f7" if is_dark else "#8839ef" 

        if not getattr(self, '_is_agent_typing', False):
            header = QLabel("QUBE")
            header.setStyleSheet(f"color: {header_color}; font-weight: bold; font-size: 11px; margin-top: 15px; background: transparent;")
            self.transcript_layout.addWidget(header)

            self.agent_msg_container = QFrame()
            # 🔑 FIX 2: Allow agent bubble to expand horizontally
            self.agent_msg_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            
            container_layout = QVBoxLayout(self.agent_msg_container)
            container_layout.setContentsMargins(0, 0, 0, 20)

            self.current_agent_msg = ChatLabel()
            self.current_agent_msg.setTextFormat(Qt.TextFormat.MarkdownText)
            self.current_agent_msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
            self.current_agent_msg.setOpenExternalLinks(True)
            
            from PyQt6.QtGui import QPalette, QColor
            palette = self.current_agent_msg.palette()
            palette.setColor(QPalette.ColorRole.WindowText, QColor(text_color))
            palette.setColor(QPalette.ColorRole.Text, QColor(text_color))
            self.current_agent_msg.setPalette(palette)
            self.current_agent_msg.setStyleSheet("font-size: 14px; background: transparent; border: none;")

            container_layout.addWidget(self.current_agent_msg)

            wrapper = MessageWrapper(self.agent_msg_container, is_user=False)
            self.transcript_layout.addWidget(wrapper)
            
            wrapper._update_width()

            self._agent_text_buffer = ""
            self._is_agent_typing = True

        self._agent_text_buffer += token
        self.current_agent_msg.setText(self._agent_text_buffer)
        
        # 🔑 MARKDOWN FIX: Force geometry update immediately after setting new Markdown text
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

        # --- THE FIX: SILENT GARBAGE COLLECTION ---
        # Before we count or draw the sessions, wipe out any abandoned empty ones!
        current_active = getattr(self, 'active_session_id', None)
        if hasattr(self.db, 'cleanup_empty_sessions'):
            self.db.cleanup_empty_sessions(current_active)

        count = self.db.get_session_count()
        display_count = "999+" if count > 999 else str(count)
        
        # Only update the text if the UI has finished building the label
        if hasattr(self, 'list_title'):
            self.list_title.setText(f"CONVERSATIONS ({display_count})")
        from PyQt6.QtWidgets import QListWidgetItem, QWidget, QHBoxLayout, QLabel, QPushButton, QMenu
        from PyQt6.QtCore import Qt, QSize
        import qtawesome as qta
        
        # 1. Determine theme state
        is_dark = True
        main_win = self.window()
        if main_win and hasattr(main_win, '_is_dark_theme'):
            is_dark = main_win._is_dark_theme
        
        # Define high-contrast colors for the text and icons
        text_color = "#cdd6f4" if is_dark else "#1e293b"
        # --- THE FIX: Use muted Catppuccin grays for the kebab icon ---
        icon_color = "#6c7086" if is_dark else "#64748b"
        
        sessions = self.db.get_recent_sessions(limit=20)
        for session in sessions:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, session["id"])
            
            row_widget = QWidget()
            row_widget.setObjectName("HistoryRowWidget")
            row_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(15, 0, 10, 0)
            row_layout.setSpacing(10)
            
            # 2. The Title - FIXED: Forcing the color here to bypass CSS inheritance bugs
            title_lbl = QLabel(session["title"])
            title_lbl.setObjectName("HistoryRowTitle")
            title_lbl.setStyleSheet(f"color: {text_color}; background: transparent; border: none; font-size: 13px; font-weight: 500;")
            
            # 3. The 3-Dot Button - FIXED: Hiding the chevron (menu-indicator)
            opts_btn = QPushButton()
            opts_btn.setObjectName("HistoryOptionsBtn")
            opts_btn.setFixedSize(28, 28)
            opts_btn.setIcon(qta.icon('fa5s.ellipsis-v', color=icon_color))
            opts_btn.setIconSize(QSize(16, 16))
            opts_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus) 
            
            # This specific line kills the chevron/arrow
            opts_btn.setStyleSheet("QPushButton::menu-indicator { image: none; width: 0px; } QPushButton { border: none; background: transparent; }")
            
            # 4. The Menu
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

    def _update_row_colors(self):
        """Forces text color changes since Qt CSS cannot pass :selected states to setItemWidget."""
        from PyQt6.QtWidgets import QLabel
        from PyQt6.QtWidgets import QApplication
        
        # 1. Detect Theme
        is_dark = True
        if self.window() and hasattr(self.window(), '_is_dark_theme'):
            is_dark = self.window()._is_dark_theme
        elif "light.qss" in QApplication.instance().styleSheet().lower():
            is_dark = False
            
        # 2. Define our exact Palette
        normal_color = "#cdd6f4" if is_dark else "#1e293b"
        selected_color = "#11111b" if is_dark else "#ffffff"

        # 3. Target whichever list is in this specific file
        target_list = getattr(self, 'doc_list', getattr(self, 'history_list', None))
        if not target_list: 
            return

        # 4. Loop through and forcefully apply the correct color
        for i in range(target_list.count()):
            item = target_list.item(i)
            widget = target_list.itemWidget(item)
            if widget:
                lbl = widget.findChild(QLabel) # Automatically grabs your title label
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
                self.log_agent_token(msg["content"])
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
        icon_color = "#cba6f7" if is_dark else "#1e293b"
        
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