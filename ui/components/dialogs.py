# ui/components/dialogs.py
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt

class QubeMessageBox(QDialog):
    def __init__(self, title: str, message: str, parent=None, is_error=False):
        super().__init__(parent)
        
        # 1. Remove the ugly OS window borders
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.setMinimumWidth(400)
        
        # 2. Base layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 3. Apply modern styling to match Qube
        bg_color = "#2D2D2D" if not is_error else "#3A1C1C" # Dark red for errors
        accent_color = "#4A90E2" if not is_error else "#E74C3C"
        
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_color};
                border-radius: 10px;
                border: 1px solid #444;
            }}
            QLabel#Title {{
                color: #FFFFFF;
                font-size: 16px;
                font-weight: bold;
            }}
            QLabel#Message {{
                color: #CCCCCC;
                font-size: 13px;
            }}
            QPushButton {{
                background-color: {accent_color};
                color: white;
                border-radius: 5px;
                padding: 8px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {accent_color}DD;
            }}
        """)
        
        # 4. Add Widgets
        title_lbl = QLabel(title)
        title_lbl.setObjectName("Title")
        
        msg_lbl = QLabel(message)
        msg_lbl.setObjectName("Message")
        msg_lbl.setWordWrap(True)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("Acknowledge")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        
        layout.addWidget(title_lbl)
        layout.addSpacing(10)
        layout.addWidget(msg_lbl)
        layout.addSpacing(20)
        layout.addLayout(btn_layout)