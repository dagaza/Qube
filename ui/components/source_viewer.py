# ui/components/source_viewer.py
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QLabel, QPushButton
from PyQt6.QtCore import Qt

class SourcePreviewer(QDialog):
    def __init__(self, filename, content, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Source: {filename}")
        self.setMinimumSize(600, 500)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        
        # Apply Qube Dark Styling
        self.setStyleSheet("""
            QDialog { background-color: #1E1E1E; border: 1px solid #333; }
            QLabel { color: #4A90E2; font-size: 18px; font-weight: bold; }
            QTextEdit { 
                background-color: #252525; 
                color: #DDD; 
                border-radius: 5px; 
                padding: 15px; 
                font-family: 'Inter', sans-serif;
                font-size: 14px;
                line-height: 1.6;
            }
        """)

        layout = QVBoxLayout(self)
        
        title_lbl = QLabel(f"📄 {filename}")
        layout.addWidget(title_lbl)
        
        self.viewer = QTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setPlainText(content)
        layout.addWidget(self.viewer)
        
        close_btn = QPushButton("Close Preview")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)