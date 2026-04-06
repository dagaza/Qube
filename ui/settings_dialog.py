import qtawesome as qta
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QFrame, QPushButton, QVBoxLayout, QHBoxLayout,
    QProgressBar, QLabel, QCheckBox, QLineEdit, QDoubleSpinBox, QSpinBox, 
    QWidget, QScrollArea
)
from PyQt6.QtCore import Qt, QSize
from pathlib import Path

class SettingsDialog(QDialog):
    def __init__(self, parent_window, llm_worker, embedder, store, audio_worker):
        super().__init__(parent_window)
        
        # 1. References
        self.llm_worker = llm_worker
        self.embedder = embedder
        self.store = store
        self.audio_worker = audio_worker
        self.parent_window = parent_window

        # 2. Window Setup
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(500, 750) # Increased height to fit all features comfortably
        
        self._old_pos = None
        self._setup_ui()

    def _setup_ui(self):
        # Main background container
        self.main_container = QFrame(self)
        self.main_container.setObjectName("SettingsContainer")
        self.main_container.setStyleSheet("""
            #SettingsContainer {
                background-color: #1e1e2e;
                border: 1px solid #45475a;
                border-radius: 15px;
            }
            QLabel { color: #cdd6f4; font-size: 13px; }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 6px;
                color: #cdd6f4;
            }
            QCheckBox { color: #cdd6f4; spacing: 8px; }
        """)
        
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.main_container)

        container_layout = QVBoxLayout(self.main_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # 3. Title Bar
        self.title_bar = self._build_title_bar()
        container_layout.addWidget(self.title_bar)

        # 4. Content Area (Using a ScrollArea so it works on smaller screens)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(25, 10, 25, 25)
        content_layout.setSpacing(18)

        # --- SECTION 1: HARDWARE & AUDIO ---
        content_layout.addWidget(self._build_section_header("fa5s.microchip", "AUDIO & HARDWARE"))
        
        hw_form = QFormLayout()
        hw_form.setSpacing(12)
        hw_form.addRow("🎙️ Audio Input", self.parent_window.mic_selector)
        hw_form.addRow("🔊 Audio Output", self.parent_window.device_selector)
        
        # Silence Timeout
        self.timeout_spinner = QDoubleSpinBox()
        self.timeout_spinner.setRange(0.5, 5.0)
        self.timeout_spinner.setSingleStep(0.1)
        self.timeout_spinner.setValue(self.audio_worker.silence_timeout)
        self.timeout_spinner.setSuffix(" sec")
        self.timeout_spinner.valueChanged.connect(self.audio_worker.set_silence_timeout)
        hw_form.addRow("⏱️ Silence Cutoff", self.timeout_spinner)

        # Mic Sensitivity (Threshold) - RESTORED
        self.threshold_spinner = QSpinBox()
        self.threshold_spinner.setRange(1, 5000)
        self.threshold_spinner.setSingleStep(50)
        self.threshold_spinner.setValue(int(self.audio_worker.speech_threshold))
        self.threshold_spinner.valueChanged.connect(self.audio_worker.set_speech_threshold)
        hw_form.addRow("🎚️ Mic Sensitivity", self.threshold_spinner)

        content_layout.addLayout(hw_form)

        # --- SECTION 2: AI ROUTING ---
        content_layout.addWidget(self._build_section_header("fa5s.network-wired", "AI MODELS & ROUTING"))
        
        ai_form = QFormLayout()
        ai_form.setSpacing(12)
        ai_form.addRow("🗣️ Active Voice", self.parent_window.voice_selector)
        ai_form.addRow("🤖 AI Provider", self.parent_window.provider_selector)
        
        # TTS Model Load Button - RESTORED with icon
        self.parent_window.load_model_btn.setIcon(qta.icon("fa5s.download", color="#11111b"))
        self.parent_window.load_model_btn.setStyleSheet("""
            QPushButton { background-color: #fab387; color: #11111b; font-weight: bold; border-radius: 5px; padding: 5px; }
            QPushButton:hover { background-color: #f9e2af; }
        """)
        ai_form.addRow("🧠 TTS Engine", self.parent_window.load_model_btn)
        
        content_layout.addLayout(ai_form)

        # --- SECTION 3: KNOWLEDGE BASE (RAG) ---
        content_layout.addWidget(self._build_section_header("fa5s.database", "KNOWLEDGE BASE"))
        
        self.add_docs_btn = QPushButton("  Ingest Documents")
        self.add_docs_btn.setIcon(qta.icon("fa5s.file-medical", color="#11111b"))
        self.add_docs_btn.setStyleSheet("""
            QPushButton { background-color: #89b4fa; color: #11111b; font-weight: bold; border-radius: 8px; padding: 10px; }
            QPushButton:hover { background-color: #b4befe; }
        """)
        self.add_docs_btn.clicked.connect(self.browse_for_documents)
        content_layout.addWidget(self.add_docs_btn)

        self.ingest_progress = QProgressBar()
        self.ingest_progress.setFixedHeight(8)
        self.ingest_progress.setStyleSheet("""
            QProgressBar { background-color: #313244; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #a6e3a1; border-radius: 4px; }
        """)
        content_layout.addWidget(self.ingest_progress)
        
        self.ingest_status = QLabel("Ready.")
        self.ingest_status.setStyleSheet("color: #6c7086; font-size: 11px;")
        content_layout.addWidget(self.ingest_status)

        # --- SECTION 4: EXPERIMENTAL & NLP ---
        content_layout.addWidget(self._build_section_header("fa5s.flask", "EXPERIMENTAL FEATURES"))
        
        # Auto-fallback
        self.auto_fallback_cb = QCheckBox(" Enable Auto-Fallback (Search every prompt)")
        self.auto_fallback_cb.setChecked(False)
        self.auto_fallback_cb.toggled.connect(self.llm_worker.set_auto_fallback)
        content_layout.addWidget(self.auto_fallback_cb)

        # NLP Trigger - RESTORED
        self.nlp_cb = QCheckBox(" Enable NLP Trigger")
        self.nlp_cb.setStyleSheet("color: #fab387; font-weight: bold;")
        self.nlp_cb.toggled.connect(self.toggle_nlp_fields)
        content_layout.addWidget(self.nlp_cb)

        self.nlp_input = QLineEdit("check my notes, in my documents, what did the file say")
        self.nlp_input.setEnabled(False)
        self.nlp_input.textChanged.connect(self.llm_worker.set_nlp_keywords)
        content_layout.addWidget(QLabel("Trigger Keywords:"))
        content_layout.addWidget(self.nlp_input)

        scroll.setWidget(scroll_content)
        container_layout.addWidget(scroll)

    def _build_title_bar(self):
        bar = QFrame()
        bar.setFixedHeight(50)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 15, 0)
        title = QLabel("SYSTEM SETTINGS")
        title.setStyleSheet("font-weight: bold; letter-spacing: 1px; color: #89b4fa; font-size: 12px;")
        close_btn = QPushButton()
        close_btn.setFixedSize(30, 30)
        close_btn.setIcon(qta.icon("fa5s.times", color="#f38ba8"))
        close_btn.setStyleSheet("QPushButton { border: none; } QPushButton:hover { background-color: #313244; border-radius: 15px; }")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(close_btn)
        return bar

    def _build_section_header(self, icon_name, title_text):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 10, 0, 5)
        icon_label = QLabel()
        icon_label.setPixmap(qta.icon(icon_name, color="#fab387").pixmap(QSize(16, 16)))
        text_label = QLabel(title_text)
        text_label.setStyleSheet("font-weight: bold; color: #fab387; font-size: 11px;")
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch()
        return container

    def toggle_nlp_fields(self, checked):
        """Enables/Disables the keyword input based on checkbox state."""
        self.nlp_input.setEnabled(checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.title_bar.underMouse():
            self._old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self._old_pos is not None:
            delta = event.globalPosition().toPoint() - self._old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self._old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self._old_pos = None

    def browse_for_documents(self):
        from PyQt6.QtWidgets import QFileDialog
        from workers.ingestion_worker import IngestionWorker
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Documents", "", "Supported Files (*.pdf *.epub *.txt *.md)")
        if file_paths:
            paths = [Path(p) for p in file_paths]
            self.ingest_worker = IngestionWorker(paths, self.embedder, self.store)
            self.ingest_worker.progress_update.connect(self.ingest_progress.setValue)
            self.ingest_worker.file_done.connect(lambda msg: self.ingest_status.setText(msg))
            self.ingest_worker.ingestion_complete.connect(lambda total: self.ingest_status.setText(f"Done! {total} chunks added."))
            self.add_docs_btn.setEnabled(False)
            self.ingest_worker.finished.connect(lambda: self.add_docs_btn.setEnabled(True))
            self.ingest_worker.start()