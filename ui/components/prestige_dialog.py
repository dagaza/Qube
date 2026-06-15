"""Frameless Prestige-styled dialogs shared across Qube (dark/light aware)."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QScrollArea,
    QWidget,
)
from PyQt6.QtCore import Qt


def _resolve_is_dark_from_parent(parent) -> bool:
    w = parent.window() if parent else None
    return getattr(w, "_is_dark_theme", True) if w else True


class PrestigeDialog(QDialog):
    def __init__(
        self,
        parent,
        title,
        message,
        is_dark=True,
        is_input=False,
        default_text="",
        *,
        tone: str = "default",
        min_width: int = 450,
        dialog_width: int | None = None,
        confirm_text: str = "CONFIRM",
        cancel_text: str = "CANCEL",
        show_cancel: bool = True,
    ):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        fixed_w = int(dialog_width) if dialog_width is not None else None
        if fixed_w is not None:
            fixed_w = max(300, fixed_w)
        else:
            self.setMinimumWidth(max(280, int(min_width)))

        self.result_text = None
        bg, fg = ("#1e1e2e", "#cdd6f4") if is_dark else ("#ffffff", "#1e293b")
        tone_key = str(tone or "default").lower().strip()
        if tone_key == "danger":
            accent = "#dc2626"
            confirm_fg = "#f8fafc"
        elif "Delete" in title:
            accent = "#f38ba8"
            confirm_fg = "#11111b"
        else:
            accent = "#89b4fa"
            confirm_fg = "#11111b"
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)

        self.container = QFrame()
        self.container.setObjectName("DialogContainer")
        self.container.setStyleSheet(
            f"""
            QFrame#DialogContainer {{
                background: {bg};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
            QLabel {{ color: {fg}; border: none; background: transparent; }}
        """
        )

        c_layout = QVBoxLayout(self.container)
        c_layout.setContentsMargins(30, 30, 30, 25)
        c_layout.setSpacing(20)

        if fixed_w is not None:
            # Outer layout margins (10px) + container side padding (30px each).
            self.container.setFixedWidth(fixed_w - 20)
            message_max_w = fixed_w - 80

        t_lbl = QLabel(title.upper())
        t_lbl.setStyleSheet(f"color: {accent}; font-weight: bold; font-size: 12px; letter-spacing: 2px;")

        m_lbl = QLabel(message)
        m_lbl.setWordWrap(True)
        m_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        m_lbl.setMinimumWidth(0)
        if fixed_w is not None:
            m_lbl.setMaximumWidth(message_max_w)
        m_lbl.setStyleSheet(f"color: {fg}; font-size: 15px; line-height: 1.4;")

        c_layout.addWidget(t_lbl)
        c_layout.addWidget(m_lbl)

        self.field = None
        if is_input:
            self.field = QLineEdit(default_text)
            self.field.setMinimumHeight(45)
            self.field.setStyleSheet(
                f"""
                QLineEdit {{
                    background: {'#313244' if is_dark else '#f8fafc'};
                    color: {fg};
                    border-radius: 10px;
                    padding: 10px 15px;
                    border: 1px solid {accent};
                    font-size: 14px;
                }}
            """
            )
            c_layout.addWidget(self.field)
            self.field.setFocus()

        btns = QHBoxLayout()
        btns.setSpacing(15)

        cancel_btn = QPushButton(cancel_text)
        con_b = QPushButton(confirm_text)

        btn_style = """
            QPushButton {
                padding: 15px 15px;
                min-height: 30px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 1px;
            }
        """

        cancel_btn.setStyleSheet(
            btn_style
            + f"""
            QPushButton {{
                color: {fg};
                border: 1px solid {border};
                background: transparent;
            }}
            QPushButton:hover {{
                background: rgba(255, 255, 255, 0.05);
            }}
        """
        )

        con_b.setStyleSheet(
            btn_style
            + f"""
            QPushButton {{
                background: {accent};
                color: {confirm_fg};
                border: none;
            }}
            QPushButton:hover {{
                background: {accent};
                opacity: 0.9;
            }}
        """
        )

        cancel_btn.clicked.connect(self.reject)
        con_b.clicked.connect(self.accept)

        btns.addStretch()
        if show_cancel:
            btns.addWidget(cancel_btn)
        btns.addWidget(con_b)
        c_layout.addLayout(btns)

        layout.addWidget(self.container)

    def exec(self):
        """Returns the input text if Accepted and is_input=True, otherwise True/None."""
        if super().exec():
            if self.field:
                self.result_text = self.field.text().strip()
                return self.result_text
            return True
        return None

    def accept_action(self):
        if getattr(self, "field", None):
            self.result_text = self.field.text()
        self.accept()


class SourcePreviewer(QDialog):
    """
    Read-only document viewer for citation sources: Prestige frameless chrome, theme-aware.
    """

    def __init__(self, filename: str, content: str, parent=None, *, is_dark: bool | None = None):
        super().__init__(parent)
        if is_dark is None:
            is_dark = _resolve_is_dark_from_parent(parent)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle(f"Source — {filename}")
        self.setMinimumSize(600, 500)
        self.resize(720, 560)

        bg, fg = ("#1e1e2e", "#cdd6f4") if is_dark else ("#ffffff", "#1e293b")
        accent = "#89b4fa"
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"
        surface = "#313244" if is_dark else "#f8fafc"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("SourcePreviewContainer")
        container.setStyleSheet(
            f"""
            QFrame#SourcePreviewContainer {{
                background: {bg};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
        """
        )

        inner = QVBoxLayout(container)
        inner.setContentsMargins(28, 26, 28, 22)
        inner.setSpacing(14)

        header = QLabel("SOURCE PREVIEW")
        header.setStyleSheet(
            f"color: {accent}; font-weight: bold; font-size: 11px; letter-spacing: 2px;"
        )
        title = QLabel(filename)
        title.setWordWrap(True)
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        title.setStyleSheet(f"color: {fg}; font-size: 16px; font-weight: bold;")

        inner.addWidget(header)
        inner.addWidget(title)

        self.viewer = QTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setPlainText(content)
        self.viewer.setMinimumHeight(280)
        self.viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viewer.setStyleSheet(
            f"""
            QTextEdit {{
                background: {surface};
                color: {fg};
                border: 1px solid {border};
                border-radius: 12px;
                padding: 14px 16px;
                font-size: 14px;
                line-height: 1.55;
            }}
        """
        )
        inner.addWidget(self.viewer, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("CLOSE")
        btn_style = f"""
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
        close_btn.setStyleSheet(btn_style)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        inner.addLayout(btn_row)

        outer.addWidget(container)


def _source_sort_key(src: dict) -> tuple:
    sid = src.get("id")
    if isinstance(sid, int):
        return (0, sid)
    if isinstance(sid, str) and sid.strip().isdigit():
        return (0, int(sid.strip()))
    if str(sid or "").strip().upper() == "W":
        return (0, 0)
    return (1, str(sid or ""))


def _source_type_label(src: dict) -> str:
    st = str(src.get("type") or "").strip().lower()
    if st == "web":
        return "Web"
    if st == "memory":
        return "Memory"
    if st == "rag":
        return "Document"
    return st.title() if st else "Source"


def _source_cite_label(src: dict) -> str:
    sid = src.get("id")
    if str(sid or "").strip().upper() == "W":
        return "[W]"
    if sid is not None and str(sid).strip():
        return f"[{sid}]"
    return "[?]"


class _CitationSourceRow(QFrame):
    """Single clickable source card inside ``CitationSourcesDialog``."""

    def __init__(self, parent=None, *, on_click=None):
        super().__init__(parent)
        self._on_click = on_click

    def mouseReleaseEvent(self, event):  # noqa: N802 — Qt API
        if (
            event.button() == Qt.MouseButton.LeftButton
            and callable(self._on_click)
        ):
            self._on_click()
        super().mouseReleaseEvent(event)


class CitationSourcesDialog(QDialog):
    """
    Prestige-styled list of citation sources attached to one assistant answer.
    """

    def __init__(
        self,
        sources: list,
        parent=None,
        *,
        is_dark: bool | None = None,
        on_open_source=None,
    ):
        super().__init__(parent)
        if is_dark is None:
            is_dark = _resolve_is_dark_from_parent(parent)

        self._on_open_source = on_open_source
        src_list = [s for s in (sources or []) if isinstance(s, dict)]
        src_list = sorted(src_list, key=_source_sort_key)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(520, 360)
        self.resize(620, 480)

        bg, fg = ("#1e1e2e", "#cdd6f4") if is_dark else ("#ffffff", "#1e293b")
        accent = "#89b4fa"
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"
        surface = "#313244" if is_dark else "#f8fafc"
        muted = "#a6adc8" if is_dark else "#64748b"
        row_hover = "rgba(255, 255, 255, 0.06)" if is_dark else "rgba(0, 0, 0, 0.04)"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("CitationSourcesContainer")
        container.setStyleSheet(
            f"""
            QFrame#CitationSourcesContainer {{
                background: {bg};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
        """
        )

        inner = QVBoxLayout(container)
        inner.setContentsMargins(28, 26, 28, 22)
        inner.setSpacing(14)

        header = QLabel("SOURCES")
        header.setStyleSheet(
            f"color: {accent}; font-weight: bold; font-size: 11px; letter-spacing: 2px;"
        )
        count = len(src_list)
        subtitle = QLabel(
            f"{count} source{'s' if count != 1 else ''} cited in this answer"
            if count
            else "No sources were attached to this answer."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {muted}; font-size: 13px;")

        inner.addWidget(header)
        inner.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        list_host = QWidget()
        list_layout = QVBoxLayout(list_host)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(10)

        for src in src_list:
            row = _CitationSourceRow(
                on_click=(
                    (lambda payload=src: on_open_source(payload))
                    if callable(on_open_source)
                    else None
                )
            )
            row.setObjectName("CitationSourceRow")
            row.setCursor(Qt.CursorShape.PointingHandCursor)
            row.setStyleSheet(
                f"""
                QFrame#CitationSourceRow {{
                    background: {surface};
                    border: 1px solid {border};
                    border-radius: 12px;
                }}
                QFrame#CitationSourceRow:hover {{
                    background: {row_hover};
                }}
                QLabel {{ background: transparent; border: none; }}
            """
            )
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(14, 12, 14, 12)
            row_layout.setSpacing(6)

            title_row = QHBoxLayout()
            cite_lbl = QLabel(_source_cite_label(src))
            cite_lbl.setStyleSheet(
                f"color: {accent}; font-weight: bold; font-size: 13px;"
            )
            name = str(src.get("filename") or "Untitled source").strip()
            title_lbl = QLabel(name)
            title_lbl.setWordWrap(True)
            title_lbl.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
            title_lbl.setStyleSheet(
                f"color: {fg}; font-size: 14px; font-weight: 600;"
            )
            type_lbl = QLabel(_source_type_label(src))
            type_lbl.setStyleSheet(
                f"color: {muted}; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;"
            )
            title_row.addWidget(cite_lbl, 0)
            title_row.addWidget(title_lbl, 1)
            title_row.addWidget(type_lbl, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            row_layout.addLayout(title_row)

            snippet = str(src.get("content") or "").strip()
            url = str(src.get("url") or "").strip()
            if snippet:
                preview = snippet if len(snippet) <= 220 else snippet[:217].rstrip() + "…"
                body_lbl = QLabel(preview)
                body_lbl.setWordWrap(True)
                body_lbl.setStyleSheet(
                    f"color: {muted}; font-size: 13px; line-height: 1.45;"
                )
                row_layout.addWidget(body_lbl)
            elif url:
                url_lbl = QLabel(url)
                url_lbl.setWordWrap(True)
                url_lbl.setStyleSheet(
                    f"color: {accent}; font-size: 12px;"
                )
                row_layout.addWidget(url_lbl)

            list_layout.addWidget(row)

        list_layout.addStretch(1)
        scroll.setWidget(list_host)
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
