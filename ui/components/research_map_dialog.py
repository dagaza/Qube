"""Research map dialog — lightweight session knowledge graph viewer."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.components.prestige_dialog import _resolve_is_dark_from_parent


def _kind_label(kind: str) -> str:
    mapping = {
        "query": "Query",
        "source": "Source",
        "entity": "Entity",
        "about": "About",
        "supports": "Supports",
        "mentions": "Mentions",
        "conflicts": "Conflicts",
    }
    return mapping.get(kind or "", (kind or "Link").title())


class ResearchMapDialog(QDialog):
    """Frameless dialog showing nodes and edges from a session knowledge graph."""

    def __init__(
        self,
        graph: dict,
        parent=None,
        *,
        is_dark: bool | None = None,
        title: str = "Research map",
    ) -> None:
        super().__init__(parent)
        if is_dark is None:
            is_dark = _resolve_is_dark_from_parent(parent)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(520, 420)
        self.resize(640, 520)

        bg, fg = ("#1e1e2e", "#cdd6f4") if is_dark else ("#ffffff", "#1e293b")
        accent = "#89b4fa"
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"
        muted = "#a6adc8" if is_dark else "#64748b"
        surface = "#313244" if is_dark else "#f8fafc"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("ResearchMapContainer")
        container.setStyleSheet(
            f"""
            QFrame#ResearchMapContainer {{
                background: {bg};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
        """
        )
        inner = QVBoxLayout(container)
        inner.setContentsMargins(28, 26, 28, 22)
        inner.setSpacing(14)

        header = QLabel(title.upper())
        header.setStyleSheet(
            f"color: {accent}; font-weight: bold; font-size: 11px; letter-spacing: 2px;"
        )
        nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
        edges = [e for e in (graph.get("edges") or []) if isinstance(e, dict)]
        subtitle = QLabel(
            f"{len(nodes)} nodes · {len(edges)} links in this view"
            if nodes
            else "No research map data for this answer yet."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {muted}; font-size: 13px;")
        inner.addWidget(header)
        inner.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        node_labels = {
            str(n.get("id") or ""): str(n.get("label") or n.get("id") or "?")
            for n in nodes
        }

        if nodes:
            nodes_hdr = QLabel("NODES")
            nodes_hdr.setStyleSheet(
                f"color: {accent}; font-weight: bold; font-size: 10px; letter-spacing: 1.5px;"
            )
            layout.addWidget(nodes_hdr)
            for node in nodes:
                kind = str(node.get("kind") or "node")
                label = str(node.get("label") or node.get("id") or "?")
                row = QLabel(f"{_kind_label(kind)}: {label}")
                row.setWordWrap(True)
                row.setStyleSheet(
                    f"color: {fg}; background: {surface}; border: 1px solid {border}; "
                    f"border-radius: 10px; padding: 10px 12px; font-size: 13px;"
                )
                layout.addWidget(row)

        if edges:
            edges_hdr = QLabel("LINKS")
            edges_hdr.setStyleSheet(
                f"color: {accent}; font-weight: bold; font-size: 10px; letter-spacing: 1.5px; margin-top: 8px;"
            )
            layout.addWidget(edges_hdr)
            for edge in edges:
                from_id = str(edge.get("from") or "")
                to_id = str(edge.get("to") or "")
                kind = str(edge.get("kind") or "link")
                from_label = node_labels.get(from_id, from_id)
                to_label = node_labels.get(to_id, to_id)
                row = QLabel(f"{from_label} —{_kind_label(kind)}→ {to_label}")
                row.setWordWrap(True)
                row.setStyleSheet(f"color: {muted}; font-size: 12px; padding: 4px 2px;")
                layout.addWidget(row)

        layout.addStretch(1)
        scroll.setWidget(host)
        inner.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet(
            f"""
            QPushButton {{
                padding: 12px 22px;
                min-height: 32px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 1px;
                color: {fg};
                border: 1px solid {border};
                background: transparent;
            }}
            QPushButton:hover {{
                background: rgba(255, 255, 255, 0.05);
            }}
        """
        )
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        inner.addLayout(btn_row)

        outer.addWidget(container)
