from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
    QLineEdit, QPushButton, QListWidget, QTextEdit, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
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
        self._start_new_chat() # <--- ADD THIS: Ensures a valid DB session exists on boot!

    def _setup_ui(self):
        # Main Layout: 3 Columns (History | Chat Stage | Tools)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1) # 1px spacing creates a subtle border effect between panes

        # --- COLUMN 1: History Sub-Sidebar ---
        self.history_pane = self._build_history_pane()
        layout.addWidget(self.history_pane)

        # --- COLUMN 2: Main Chat Stage ---
        self.chat_stage = self._build_chat_stage()
        layout.addWidget(self.chat_stage, stretch=1) # Expands to fill available space

    # --------------------------------------------------------- #
    #  PANEL BUILDERS                                           #
    # --------------------------------------------------------- #

    def _build_history_pane(self) -> QFrame:
        frame = QFrame()
        frame.setFixedWidth(250)
        frame.setStyleSheet("background-color: #181825; border-right: 1px solid #313244;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("CONVERSATIONS")
        title.setStyleSheet("color: #a6adc8; font-weight: bold; letter-spacing: 1px; font-size: 11px; border: none;")
        
        self.new_chat_btn = QPushButton()
        self.new_chat_btn.setIcon(qta.icon('fa5s.plus', color='#a6e3a1'))
        self.new_chat_btn.setStyleSheet("border: none; background: transparent; padding: 5px;")
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.new_chat_btn)
        layout.addLayout(header_layout)

        # Search Bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search history...")
        self.search_bar.setStyleSheet("""
            QLineEdit { background-color: #313244; color: #cdd6f4; border-radius: 5px; padding: 8px; border: none; }
        """)
        layout.addWidget(self.search_bar)

        # History List
        self.history_list = QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; color: #bac2de; outline: none; }
            QListWidget::item { padding: 10px; border-radius: 5px; margin-bottom: 2px; }
            QListWidget::item:hover { background-color: #313244; }
            QListWidget::item:selected { background-color: #45475a; color: #cdd6f4; font-weight: bold; }
        """)
        layout.addWidget(self.history_list)

        # --- NEW: Wire the History UI ---
        self.new_chat_btn.clicked.connect(self._start_new_chat)
        self.history_list.itemClicked.connect(self._load_selected_chat)

        return frame

    def _build_chat_stage(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #1e1e2e; border: none;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        # 1. Transcript Area
        self.transcript_box = QTextEdit()
        self.transcript_box.setReadOnly(True)
        self.transcript_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.transcript_box.setStyleSheet("""
            QTextEdit { background-color: transparent; border: none; color: #cdd6f4; font-size: 14px; }
        """)
        
        # Placeholder for empty state
        self.transcript_box.setHtml("<h3 style='color:#6c7086; text-align:center; margin-top:50px;'>Select a conversation or start a new chat.</h3>")
        layout.addWidget(self.transcript_box)

        # 2. Input Bar Area
        input_container = QFrame()
        input_container.setStyleSheet("background-color: #313244; border-radius: 10px; border: 1px solid #45475a;")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(10, 5, 5, 5)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type a message to Qube...")
        self.text_input.setStyleSheet("background: transparent; border: none; color: #cdd6f4; font-size: 14px;")
        
        self.send_btn = QPushButton()
        self.send_btn.setIcon(qta.icon('fa5s.paper-plane', color='#89b4fa'))
        self.send_btn.setFixedSize(35, 35)
        self.send_btn.setStyleSheet("QPushButton { background-color: #1e1e2e; border-radius: 8px; } QPushButton:hover { background-color: #45475a; }")

        input_layout.addWidget(self.text_input)
        input_layout.addWidget(self.send_btn)
        layout.addWidget(input_container)

        # 3. Latency Metrics Footer
        latency_layout = QHBoxLayout()
        latency_style = "color: #6c7086; font-size: 10px; font-family: 'Consolas', monospace; border: none;"
        
        self.stt_latency_lbl = QLabel("STT: -- ms")
        self.ttft_latency_lbl = QLabel("TTFT: -- ms")
        self.tts_latency_lbl = QLabel("TTS: -- ms")
        
        for lbl in [self.stt_latency_lbl, self.ttft_latency_lbl, self.tts_latency_lbl]:
            lbl.setStyleSheet(latency_style)
            latency_layout.addWidget(lbl)
            
        latency_layout.addStretch()
        layout.addLayout(latency_layout)

        # Wire the text input to our internal handler
        self.send_btn.clicked.connect(self._handle_text_submit)
        self.text_input.returnPressed.connect(self._handle_text_submit)
        
        return frame
    
    # --- UI UPDATE RECEIVERS ---
    def log_user_message(self, text: str) -> None:
        """Renders user input as a distinct bubble."""
        # Clear the 'empty state' placeholder if it's there
        if "Select a conversation" in self.transcript_box.toHtml():
            self.transcript_box.clear()

        user_html = f"""
            <div style="margin-bottom: 20px;">
                <table width="100%" cellpadding="8" cellspacing="0">
                    <tr>
                        <td bgcolor="#313244" style="border-radius: 10px; color: #a6e3a1; font-family: 'Segoe UI';">
                            <span style="font-size: 10px; color: #9399b2; font-weight: bold;">USER</span><br/>
                            <span style="color: #cdd6f4; font-size: 14px;">{text}</span>
                        </td>
                    </tr>
                </table>
            </div>
        """
        self.transcript_box.append(user_html)
        self._is_agent_typing = False
        self.transcript_box.verticalScrollBar().setValue(self.transcript_box.verticalScrollBar().maximum())

    def log_agent_token(self, token: str) -> None:
        """Streams agent tokens into a clean typography block."""
        if not getattr(self, '_is_agent_typing', False):
            agent_header = f"""
                <div style="margin-top: 10px;">
                    <span style="font-size: 10px; color: #cba6f7; font-weight: bold;">QUBE</span><br/>
                </div>
            """
            self.transcript_box.append(agent_header)
            self._is_agent_typing = True
        
        cursor = self.transcript_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self.transcript_box.setTextCursor(cursor)
        self.transcript_box.verticalScrollBar().setValue(self.transcript_box.verticalScrollBar().maximum())

    def _handle_text_submit(self):
        """Captures text input and sends it to the LLM."""
        text = self.text_input.text().strip()
        if not text:
            return

        # 1. Clear the input box immediately
        self.text_input.clear()
        
        # 2. Visually log the user's message to the UI
        self.log_user_message(text)

        # 3. Ensure we have an active session ID for the Database
        if not hasattr(self, 'active_session_id'):
            recent_sessions = self.db.get_recent_sessions(limit=1)
            if recent_sessions:
                self.active_session_id = recent_sessions[0]['id']
            else:
                self.active_session_id = self.db.create_session("Text Conversation")

        # 4. Trigger the LLM Worker
        if self.llm:
            self.llm.generate_response(text, self.active_session_id)

    def update_stt_latency(self, ms: float) -> None:
        self.stt_latency_lbl.setText(f"STT: {ms:.0f} ms")

    def update_ttft_latency(self, ms: float) -> None:
        self.ttft_latency_lbl.setText(f"TTFT: {ms:.0f} ms")

    def update_tts_latency(self, ms: float) -> None:
        self.tts_latency_lbl.setText(f"TTS: {ms:.0f} ms")

    # --------------------------------------------------------- #
    #  HISTORY MANAGEMENT                                       #
    # --------------------------------------------------------- #

    def _refresh_history_list(self):
        """Pulls recent sessions from SQLite and populates the sidebar."""
        self.history_list.clear()
        from PyQt6.QtWidgets import QListWidgetItem
        from PyQt6.QtCore import Qt
        
        sessions = self.db.get_recent_sessions(limit=20)
        for session in sessions:
            # Create the list item using the DB title
            item = QListWidgetItem(session["title"])
            # Secretly store the UUID inside the item so we know what to load when clicked
            item.setData(Qt.ItemDataRole.UserRole, session["id"])
            self.history_list.addItem(item)

    def _start_new_chat(self):
        """Creates a fresh session and clears the UI."""
        self.active_session_id = self.db.create_session("New Conversation")
        
        self.transcript_box.clear()
        self.transcript_box.setHtml("<h3 style='color:#6c7086; text-align:center; margin-top:50px;'>New chat started. Type or speak a message!</h3>")
        self._is_agent_typing = False
        
        self._refresh_history_list()

    def _load_selected_chat(self, item):
        """Loads a historical conversation from the database into the UI."""
        from PyQt6.QtCore import Qt
        session_id = item.data(Qt.ItemDataRole.UserRole)
        self.active_session_id = session_id
        
        self.transcript_box.clear()
        self._is_agent_typing = False

        history = self.db.get_session_history(session_id)
        if not history:
            self.transcript_box.setHtml("<h3 style='color:#6c7086; text-align:center; margin-top:50px;'>Empty conversation.</h3>")
            return

        # Re-draw the entire conversation
        for msg in history:
            if msg["role"] == "user":
                self.log_user_message(msg["content"])
            elif msg["role"] == "assistant":
                self.log_agent_token(msg["content"])
                self._is_agent_typing = False # Reset so the next message gets a new header