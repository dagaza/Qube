from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal
import json


class MemoryDebugPanel(QWidget):
    """
    Qube Memory Debug Dashboard (v5.2)

    ✔ Live retrieval tracing
    ✔ Ranking transparency
    ✔ No blocking / no inference overhead
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("MemoryDebugPanel")

        layout = QVBoxLayout(self)

        # ============================================================
        # HEADER
        # ============================================================
        self.title = QLabel("🧠 Memory Debug Dashboard")
        self.title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.title)

        # ============================================================
        # LAST QUERY DISPLAY
        # ============================================================
        self.last_query = QLabel("Last Query: —")
        layout.addWidget(self.last_query)

        # ============================================================
        # TRACE TABLE
        # ============================================================
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Content", "Distance", "Confidence", "Strength", "Score"
        ])

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        layout.addWidget(self.table)

        # ============================================================
        # RAW TRACE OUTPUT (debug fallback)
        # ============================================================
        self.raw_trace = QTextEdit()
        self.raw_trace.setReadOnly(True)
        self.raw_trace.setPlaceholderText("Raw memory trace logs...")
        layout.addWidget(self.raw_trace)

        # Stats
        self.stats_label = QLabel("Stats: —")
        layout.addWidget(self.stats_label)

    # ============================================================
    # PUBLIC API (CALLED BY TELEMETRY CONTROLLER)
    # ============================================================

    def set_query(self, query: str):
        self.last_query.setText(f"Last Query: {query}")

    def update_trace(self, trace_items: list):
        """
        trace_items = [
            {
                content,
                distance,
                confidence,
                strength,
                score
            }
        ]
        """

        self.table.setRowCount(len(trace_items))

        for i, item in enumerate(trace_items):
            self.table.setItem(i, 0, QTableWidgetItem(item.get("content", "")[:60]))
            self.table.setItem(i, 1, QTableWidgetItem(f"{item.get('distance', 0):.3f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{item.get('confidence', 0):.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(str(item.get("strength", 1))))
            self.table.setItem(i, 4, QTableWidgetItem(f"{item.get('score', 0):.3f}"))

    def append_raw_trace(self, text: str):
        self.raw_trace.append(text)

    def update_stats(self, stats: dict):
        self.stats_label.setText(
            "Stats: " + json.dumps(stats, indent=2)
        )