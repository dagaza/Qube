from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
    QLineEdit, QPushButton, QListWidget, QTextEdit, 
    QDoubleSpinBox, QSpinBox, QCheckBox, QSizePolicy
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

        # --- COLUMN 3: Context & Tools Sidebar ---
        self.tools_pane = self._build_tools_pane()
        layout.addWidget(self.tools_pane)

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

        return frame

    def _build_tools_pane(self) -> QFrame:
        frame = QFrame()
        frame.setFixedWidth(260)
        frame.setStyleSheet("background-color: #181825; border-left: 1px solid #313244;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(25)

        # Reusable section header style
        header_style = "color: #a6adc8; font-weight: bold; font-size: 10px; letter-spacing: 1px; border: none;"
        control_style = "color: #cdd6f4; border: none;"

        # --- Voice Toggle ---
        voice_layout = QVBoxLayout()
        voice_layout.setSpacing(5)
        v_title = QLabel("AUDIO OUTPUT")
        v_title.setStyleSheet(header_style)
        self.voice_bypass_cb = QCheckBox(" Enable TTS Voice")
        self.voice_bypass_cb.setChecked(True)
        self.voice_bypass_cb.setStyleSheet(control_style)
        voice_layout.addWidget(v_title)
        voice_layout.addWidget(self.voice_bypass_cb)
        layout.addLayout(voice_layout)

        # --- Generation Parameters ---
        param_layout = QVBoxLayout()
        param_layout.setSpacing(10)
        p_title = QLabel("GENERATION PARAMETERS")
        p_title.setStyleSheet(header_style)
        param_layout.addWidget(p_title)

        # Temperature
        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temperature:", styleSheet=control_style))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.7)
        self.temp_spin.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 3px; padding: 3px;")
        temp_row.addWidget(self.temp_spin)
        param_layout.addLayout(temp_row)

        # Context Window
        ctx_row = QHBoxLayout()
        ctx_row.addWidget(QLabel("Context Limit:", styleSheet=control_style))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(1024, 128000)
        self.ctx_spin.setSingleStep(1024)
        self.ctx_spin.setValue(4096)
        self.ctx_spin.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 3px; padding: 3px;")
        ctx_row.addWidget(self.ctx_spin)
        param_layout.addLayout(ctx_row)

        self.update_ctx_btn = QPushButton(" Apply Context Update")
        self.update_ctx_btn.setStyleSheet("""
            QPushButton { background-color: #45475a; color: #cdd6f4; border-radius: 5px; padding: 6px; }
            QPushButton:hover { background-color: #585b70; }
        """)
        param_layout.addWidget(self.update_ctx_btn)
        layout.addLayout(param_layout)

        # --- MCP Tools ---
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(8)
        t_title = QLabel("MCP TOOLS (AGENTIC)")
        t_title.setStyleSheet(header_style)
        tools_layout.addWidget(t_title)

        self.tool_rag_cb = QCheckBox(" Local Knowledge Base")
        self.tool_rag_cb.setChecked(True)
        self.tool_rag_cb.setStyleSheet(control_style)
        
        self.tool_internet_cb = QCheckBox(" Internet Search")
        self.tool_internet_cb.setChecked(False)
        self.tool_internet_cb.setStyleSheet(control_style)

        tools_layout.addWidget(self.tool_rag_cb)
        tools_layout.addWidget(self.tool_internet_cb)
        layout.addLayout(tools_layout)

        layout.addStretch()
        return frame