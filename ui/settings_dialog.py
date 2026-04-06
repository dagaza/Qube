from PyQt6.QtWidgets import (QDialog, QFormLayout, QFrame, QPushButton, 
                             QProgressBar, QLabel, QCheckBox, QLineEdit, QFileDialog, QDoubleSpinBox, QSpinBox)
from workers import IngestionWorker
from workers import llm_worker
from workers import audio_worker

class SettingsDialog(QDialog):
    def __init__(self, parent_window, llm_worker, embedder, store, audio_worker): # Add them here
        super().__init__(parent_window)
        self.llm_worker = llm_worker
        self.embedder = embedder # Store locally
        self.store = store       # Store locally
        self.audio_worker = audio_worker # Store locally
        self.parent_window = parent_window
        
        self.setWindowTitle("⚙️ Qube Hardware & Routing Settings")
        self.setFixedSize(450, 520) 
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        self.layout = QFormLayout(self)
        
        # --- 1. HARDWARE ROUTING ---
        # Note: These widgets are still owned by MainWindow, which is fine!
        self.layout.addRow("🎙️ Audio Input:", parent_window.mic_selector)
        self.layout.addRow("🔊 Audio Output:", parent_window.device_selector)
        
        # NEW: The Silence Timeout Spinner
        self.timeout_spinner = QDoubleSpinBox()
        self.timeout_spinner.setRange(0.5, 5.0)       # Min 0.5s, Max 5.0s
        self.timeout_spinner.setSingleStep(0.1)       # Go up/down by 0.1s
        self.timeout_spinner.setValue(self.audio_worker.silence_timeout) # Sync to current value
        self.timeout_spinner.setSuffix(" sec")        # Adds "sec" to the visual text
        self.timeout_spinner.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 4px; border-radius: 3px;")
        self.timeout_spinner.valueChanged.connect(self.audio_worker.set_silence_timeout)
        self.layout.addRow("⏱️ Silence Cutoff:", self.timeout_spinner)

        # AUDIO UX: Mic Sensitivity Floor
        self.threshold_spinner = QSpinBox()
        self.threshold_spinner.setRange(1, 5000) # Lowered min range to 10 for quiet mics
        self.threshold_spinner.setSingleStep(50)
        self.threshold_spinner.setValue(int(self.audio_worker.speech_threshold)) 
        self.threshold_spinner.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 4px; border-radius: 3px;")
        self.threshold_spinner.valueChanged.connect(self.audio_worker.set_speech_threshold)
        self.layout.addRow("🎚️ Mic Sensitivity Floor:", self.threshold_spinner)

        self.layout.addRow("🗣️ Active Voice:", parent_window.voice_selector)
        self.layout.addRow("🤖 AI Provider:", parent_window.provider_selector) 
        self.layout.addRow("🧠 TTS Model:", parent_window.load_model_btn)

        # --- 2. RAG INGESTION UI ---
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #45475a;")
        self.layout.addRow(line)
        
        self.add_docs_btn = QPushButton("📄 Add Documents to Knowledge Base")
        self.add_docs_btn.setStyleSheet("background-color: #89b4fa; color: #11111b; font-weight: bold; padding: 5px;")
        self.add_docs_btn.clicked.connect(self.browse_for_documents)
        self.layout.addRow(self.add_docs_btn)

        self.ingest_progress = QProgressBar()
        self.ingest_progress.setRange(0, 100)
        self.ingest_progress.setValue(0)
        self.ingest_progress.setStyleSheet("QProgressBar::chunk { background-color: #f9e2af; }")
        self.layout.addRow(self.ingest_progress)
        
        self.ingest_status = QLabel("Ready.")
        self.ingest_status.setStyleSheet("font-size: 11px; color: #bac2de;")
        self.layout.addRow(self.ingest_status)

        # --- 3. RAG LOGIC: AUTO-FALLBACK ---
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet("color: #45475a;")
        self.layout.addRow(line2)
        
        self.auto_fallback_cb = QCheckBox("Enable Auto-Fallback (Search every prompt)")
        self.auto_fallback_cb.setStyleSheet("color: #cdd6f4;")
        self.auto_fallback_cb.setChecked(False) 
        
        # FIXED: Connect directly to our injected llm_worker
        self.auto_fallback_cb.toggled.connect(self.llm_worker.set_auto_fallback)
        self.layout.addRow(self.auto_fallback_cb)

        # --- 4. RAG LOGIC: NLP TRIGGER ---
        self.nlp_cb = QCheckBox("Enable NLP Trigger (Experimental)")
        self.nlp_cb.setStyleSheet("color: #fab387;") 
        self.nlp_cb.setChecked(False) 
        
        self.nlp_input = QLineEdit("check my notes, in my documents, what did the file say")
        self.nlp_input.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 5px; border-radius: 3px;")
        self.nlp_input.setEnabled(False) 
        
        # Connect NLP signals locally and to the worker
        self.nlp_cb.toggled.connect(self.toggle_nlp_fields)
        
        # FIXED: Connect directly to our injected llm_worker
        self.nlp_input.textChanged.connect(self.llm_worker.set_nlp_keywords)
        
        self.layout.addRow(self.nlp_cb)
        self.layout.addRow("Trigger Keywords:", self.nlp_input)

    # --- HELPER METHODS FOR SETTINGS ---
    
    def toggle_nlp_fields(self, checked):
        """Enables/Disables the keyword input based on checkbox state."""
        self.nlp_input.setEnabled(checked)

    def browse_for_documents(self):
        """Handles the file picker and starts the background ingestion thread."""
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path
        from workers.ingestion_worker import IngestionWorker

        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Documents", "", 
            "Supported Files (*.pdf *.epub *.txt *.md)"
        )
        
        if file_paths:
            paths = [Path(p) for p in file_paths]
            
            # Start the background ingestion worker
            # We use 'self.embedder' and 'self.store' which are now injected
            self.ingest_worker = IngestionWorker(
                paths, 
                self.embedder, 
                self.store
            )
            
            # Wire up the progress signals to the UI
            self.ingest_worker.progress_update.connect(self.ingest_progress.setValue)
            self.ingest_worker.file_done.connect(lambda msg: self.ingest_status.setText(msg))
            self.ingest_worker.error_occurred.connect(lambda msg: self.ingest_status.setText(f"Error: {msg}"))
            self.ingest_worker.ingestion_complete.connect(
                lambda total: self.ingest_status.setText(f"Done! Added {total} chunks.")
            )
            
            # Lock the button while processing, unlock when finished
            self.add_docs_btn.setEnabled(False)
            self.ingest_worker.finished.connect(lambda: self.add_docs_btn.setEnabled(True))
            
            self.ingest_worker.start()