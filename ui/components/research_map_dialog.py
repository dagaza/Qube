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

from core.theme.accessors import theme_for
from core.theme.color_utils import with_alpha
from core.theme.widget_styles import (
    PRESTIGE_ACCENT_LABEL,
    PRESTIGE_BODY_LABEL,
    PRESTIGE_GHOST_BUTTON,
    PRESTIGE_MUTED_LABEL,
    PRESTIGE_SOURCE_CONTAINER,
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
        theme = theme_for(is_dark=is_dark)
        border = theme.border_subtle if theme.is_dark else theme.border
        surface = theme.surface_elevated if theme.is_dark else theme.surface
        hover_bg = with_alpha(theme.text_primary, 0.05)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(520, 420)
        self.resize(640, 520)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("ResearchMapContainer")
        container.setStyleSheet(
            theme.style(
                PRESTIGE_SOURCE_CONTAINER,
                accent=theme.link,
                object_name="ResearchMapContainer",
            )
        )
        inner = QVBoxLayout(container)
        inner.setContentsMargins(28, 26, 28, 22)
        inner.setSpacing(14)

        header = QLabel(title.upper())
        header.setStyleSheet(
            theme.style(PRESTIGE_ACCENT_LABEL, accent=theme.link, font_size="11px")
        )
        nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
        edges = [e for e in (graph.get("edges") or []) if isinstance(e, dict)]
        subtitle = QLabel(
            f"{len(nodes)} nodes · {len(edges)} links in this view"
            if nodes
            else "No research map data for this answer yet."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(theme.style(PRESTIGE_MUTED_LABEL, font_size="13px"))
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

        section_hdr_style = theme.style(
            PRESTIGE_ACCENT_LABEL,
            accent=theme.link,
            font_size="10px",
            letter_spacing="1.5px",
        )
        node_row_style = (
            theme.style(PRESTIGE_BODY_LABEL, font_size="13px", font_weight="400")
            + f"""
            background: {surface};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 10px 12px;
            """
        )
        edge_row_style = theme.style(PRESTIGE_MUTED_LABEL, font_size="12px") + "padding: 4px 2px;"

        if nodes:
            nodes_hdr = QLabel("NODES")
            nodes_hdr.setStyleSheet(section_hdr_style)
            layout.addWidget(nodes_hdr)
            for node in nodes:
                kind = str(node.get("kind") or "node")
                label = str(node.get("label") or node.get("id") or "?")
                row = QLabel(f"{_kind_label(kind)}: {label}")
                row.setWordWrap(True)
                row.setStyleSheet(node_row_style)
                layout.addWidget(row)

        if edges:
            edges_hdr = QLabel("LINKS")
            edges_hdr.setStyleSheet(section_hdr_style + " margin-top: 8px;")
            layout.addWidget(edges_hdr)
            for edge in edges:
                from_id = str(edge.get("from") or "")
                to_id = str(edge.get("to") or "")
                kind = str(edge.get("kind") or "link")
                from_label = node_labels.get(from_id, from_id)
                to_label = node_labels.get(to_id, to_id)
                row = QLabel(f"{from_label} —{_kind_label(kind)}→ {to_label}")
                row.setWordWrap(True)
                row.setStyleSheet(edge_row_style)
                layout.addWidget(row)

        layout.addStretch(1)
        scroll.setWidget(host)
        inner.addWidget(scroll, stretch=1)

        from core.knowledge.retrieval_trace_reader import (
            format_retrieval_trace_summary,
            read_last_retrieval_trace,
        )

        trace = read_last_retrieval_trace()
        if trace:
            trace_hdr = QLabel("HOW THIS WAS RETRIEVED")
            trace_hdr.setStyleSheet(section_hdr_style)
            trace_body = QLabel(format_retrieval_trace_summary(trace))
            trace_body.setWordWrap(True)
            trace_body.setStyleSheet(
                theme.style(PRESTIGE_MUTED_LABEL, font_size="12px")
                + f"""
                background: {surface};
                border: 1px solid {border};
                border-radius: 10px;
                padding: 10px 12px;
                """
            )
            inner.addWidget(trace_hdr)
            inner.addWidget(trace_body)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet(
            theme.style(PRESTIGE_GHOST_BUTTON)
            + f"""
            QPushButton:hover {{
                background: {hover_bg};
            }}
            """
        )
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        inner.addLayout(btn_row)

        outer.addWidget(container)
