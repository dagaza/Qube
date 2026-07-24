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
from PyQt6.QtGui import QShowEvent

from core.theme.accessors import theme_for
from core.theme.view_theme import view_resolved_theme
from core.theme.widget_styles import (
    PRESTIGE_ACCENT_LABEL,
    PRESTIGE_BODY_LABEL,
    PRESTIGE_CITATION_ROW,
    PRESTIGE_CITATIONS_CONTAINER,
    PRESTIGE_DIALOG_CANCEL,
    PRESTIGE_DIALOG_CONFIRM,
    PRESTIGE_DIALOG_CONTAINER,
    PRESTIGE_DIALOG_INPUT,
    PRESTIGE_DIALOG_MESSAGE,
    PRESTIGE_DIALOG_TITLE,
    PRESTIGE_GHOST_BUTTON,
    PRESTIGE_LINK_LABEL,
    PRESTIGE_MUTED_LABEL,
    PRESTIGE_SOURCE_CONTAINER,
    PRESTIGE_TEXT_VIEW,
    prestige_accent_colors,
)

_PRESTIGE_BTN_BASE = """
    QPushButton {
        padding: 15px 15px;
        min-height: 30px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 12px;
        letter-spacing: 1px;
    }
"""


def _resolve_is_dark_from_parent(parent) -> bool:
    w = parent.window() if parent else None
    return getattr(w, "_is_dark_theme", True) if w else True


def _dialog_theme(parent, is_dark: bool | None):
    if is_dark is None:
        is_dark = _resolve_is_dark_from_parent(parent)
    return view_resolved_theme(parent, is_dark=is_dark)


def _resolve_host_window(parent):
    if parent is None:
        return None
    if hasattr(parent, "frameGeometry") and hasattr(parent, "isWindow"):
        return parent
    return parent.window()


def _center_dialog_on_host(dialog: QDialog) -> None:
    from PyQt6.QtWidgets import QApplication

    host = _resolve_host_window(dialog.parent())
    dialog.adjustSize()
    frame = dialog.frameGeometry()
    if (
        host is not None
        and host.isVisible()
        and host.width() > 0
        and host.height() > 0
    ):
        frame.moveCenter(host.frameGeometry().center())
        dialog.move(frame.topLeft())
        return
    screen = QApplication.primaryScreen()
    if screen is None:
        return
    frame.moveCenter(screen.availableGeometry().center())
    dialog.move(frame.topLeft())


_CITATION_SOURCES_MIN_W = 680
_CITATION_SOURCES_MIN_H = 420
_CITATION_SOURCES_DEFAULT_W = 880
_CITATION_SOURCES_DEFAULT_H = 580
_CITATION_SOURCES_SNIPPET_MAX = 320
_CITATION_SOURCES_HOST_MARGIN_PX = 28


def _frame_dialog_within_host(
    dialog: QDialog,
    *,
    preferred_width: int,
    preferred_height: int,
    min_width: int,
    min_height: int,
    host_margin: int = _CITATION_SOURCES_HOST_MARGIN_PX,
    max_width_fraction: float = 0.94,
    max_height_fraction: float = 0.86,
) -> None:
    """Size and center a dialog so it stays inside the parent app window."""
    from PyQt6.QtWidgets import QApplication

    host = _resolve_host_window(dialog.parent())
    if (
        host is None
        or not host.isVisible()
        or host.width() <= 0
        or host.height() <= 0
    ):
        dialog.resize(preferred_width, preferred_height)
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        frame = dialog.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        dialog.move(frame.topLeft())
        return

    host_rect = host.frameGeometry()
    inset = max(0, int(host_margin))
    avail_w = max(1, host_rect.width() - 2 * inset)
    avail_h = max(1, host_rect.height() - 2 * inset)
    max_w = min(avail_w, int(host_rect.width() * max_width_fraction) - 2 * inset)
    max_h = min(avail_h, int(host_rect.height() * max_height_fraction) - 2 * inset)
    max_w = max(1, max_w)
    max_h = max(1, max_h)

    width = min(preferred_width, max_w)
    height = min(preferred_height, max_h)
    width = max(min(min_width, max_w), width)
    height = max(min(min_height, max_h), height)

    dialog.resize(width, height)

    w = dialog.width()
    h = dialog.height()
    x = host_rect.center().x() - w // 2
    y = host_rect.center().y() - h // 2

    min_x = host_rect.left() + inset
    min_y = host_rect.top() + inset
    max_x = host_rect.right() - inset - w + 1
    max_y = host_rect.bottom() - inset - h + 1
    if max_x < min_x:
        x = min_x
    else:
        x = max(min_x, min(x, max_x))
    if max_y < min_y:
        y = min_y
    else:
        y = max(min_y, min(y, max_y))

    dialog.move(x, y)


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
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        fixed_w = int(dialog_width) if dialog_width is not None else None
        if fixed_w is not None:
            fixed_w = max(300, fixed_w)
        else:
            self.setMinimumWidth(max(280, int(min_width)))

        self.result_text = None
        theme = _dialog_theme(parent, is_dark)
        accent, confirm_fg = prestige_accent_colors(theme, tone=tone, title=str(title))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)

        self.container = QFrame()
        self.container.setObjectName("DialogContainer")
        self.container.setStyleSheet(
            theme.style(PRESTIGE_DIALOG_CONTAINER, accent=accent, object_name="DialogContainer")
        )

        c_layout = QVBoxLayout(self.container)
        c_layout.setContentsMargins(30, 30, 30, 25)
        c_layout.setSpacing(20)

        if fixed_w is not None:
            self.container.setFixedWidth(fixed_w - 20)
            message_max_w = fixed_w - 80

        t_lbl = QLabel(title.upper())
        t_lbl.setStyleSheet(theme.style(PRESTIGE_DIALOG_TITLE, accent=accent))

        m_lbl = QLabel(message)
        m_lbl.setWordWrap(True)
        m_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        m_lbl.setMinimumWidth(0)
        if fixed_w is not None:
            m_lbl.setMaximumWidth(message_max_w)
        m_lbl.setStyleSheet(theme.style(PRESTIGE_DIALOG_MESSAGE))

        c_layout.addWidget(t_lbl)
        c_layout.addWidget(m_lbl)

        self.field = None
        if is_input:
            self.field = QLineEdit(default_text)
            self.field.setMinimumHeight(45)
            self.field.setStyleSheet(theme.style(PRESTIGE_DIALOG_INPUT, accent=accent))
            c_layout.addWidget(self.field)
            self.field.setFocus()

        btns = QHBoxLayout()
        btns.setSpacing(15)

        cancel_btn = QPushButton(cancel_text)
        con_b = QPushButton(confirm_text)

        cancel_btn.setStyleSheet(
            theme.style(PRESTIGE_DIALOG_CANCEL, btn_base=_PRESTIGE_BTN_BASE)
        )
        con_b.setStyleSheet(
            theme.style(
                PRESTIGE_DIALOG_CONFIRM,
                btn_base=_PRESTIGE_BTN_BASE,
                accent=accent,
                confirm_fg=confirm_fg,
            )
        )

        cancel_btn.clicked.connect(self.reject)
        con_b.clicked.connect(self.accept)

        btns.addStretch()
        if show_cancel:
            btns.addWidget(cancel_btn)
        btns.addWidget(con_b)
        c_layout.addLayout(btns)

        layout.addWidget(self.container)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        _center_dialog_on_host(self)
        self.raise_()
        self.activateWindow()

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
        theme = _dialog_theme(parent, is_dark)
        accent = theme.link

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle(f"Source — {filename}")
        self.setMinimumSize(600, 500)
        self.resize(720, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("SourcePreviewContainer")
        container.setStyleSheet(
            theme.style(PRESTIGE_SOURCE_CONTAINER, accent=accent, object_name="SourcePreviewContainer")
        )

        inner = QVBoxLayout(container)
        inner.setContentsMargins(28, 26, 28, 22)
        inner.setSpacing(14)

        header = QLabel("SOURCE PREVIEW")
        header.setStyleSheet(
            theme.style(PRESTIGE_ACCENT_LABEL, accent=accent, font_size="11px", letter_spacing="2px")
        )
        title = QLabel(filename)
        title.setWordWrap(True)
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        title.setStyleSheet(
            theme.style(PRESTIGE_BODY_LABEL, font_size="16px", font_weight="bold")
        )

        inner.addWidget(header)
        inner.addWidget(title)

        self.viewer = QTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setPlainText(content)
        self.viewer.setMinimumHeight(280)
        self.viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viewer.setStyleSheet(theme.style(PRESTIGE_TEXT_VIEW))
        inner.addWidget(self.viewer, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet(theme.style(PRESTIGE_GHOST_BUTTON))
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
    adapter = str(src.get("source_adapter") or "").strip()
    if adapter:
        return adapter.replace("_", " ").title()
    if st == "web":
        return "Web"
    if st == "memory":
        return "Memory"
    if st == "rag":
        return "Document"
    return st.title() if st else "Source"


def _source_metadata_line(src: dict) -> str:
    parts: list[str] = []
    venue = str(src.get("venue") or "").strip()
    if venue:
        parts.append(venue)
    pub = str(src.get("publication_date") or "").strip()
    if pub:
        parts.append(pub)
    fetch = str(src.get("fetch_status") or "").strip()
    if fetch and fetch != "snippet_only":
        parts.append(fetch.replace("_", " "))
    doi = str(src.get("doi") or "").strip()
    if doi:
        parts.append(f"DOI {doi}")
    rel = src.get("relevance_score")
    auth = src.get("authority_score")
    if rel is not None and auth is not None:
        try:
            parts.append(f"rel {float(rel):.2f} · auth {float(auth):.2f}")
        except (TypeError, ValueError):
            pass
    if src.get("preprint"):
        parts.append("preprint")
    return " · ".join(parts)


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
        transparency: dict | None = None,
        research_map_graph: dict | None = None,
        on_open_research_map=None,
        retrieval_bundle_id: str | None = None,
        retrieval_db=None,
    ):
        super().__init__(parent)
        theme = _dialog_theme(parent, is_dark)
        accent = theme.link

        self._on_open_source = on_open_source
        self._research_map_graph = research_map_graph
        self._on_open_research_map = on_open_research_map
        self._retrieval_bundle_id = retrieval_bundle_id
        self._retrieval_db = retrieval_db
        src_list = [s for s in (sources or []) if isinstance(s, dict)]
        src_list = sorted(src_list, key=_source_sort_key)
        self._src_list = src_list
        transparency = dict(transparency or {})

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(_CITATION_SOURCES_MIN_W, _CITATION_SOURCES_MIN_H)
        self.resize(_CITATION_SOURCES_DEFAULT_W, _CITATION_SOURCES_DEFAULT_H)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("CitationSourcesContainer")
        container.setStyleSheet(
            theme.style(
                PRESTIGE_CITATIONS_CONTAINER,
                accent=accent,
                object_name="CitationSourcesContainer",
            )
        )

        inner = QVBoxLayout(container)
        inner.setContentsMargins(28, 26, 28, 22)
        inner.setSpacing(14)

        header = QLabel("SOURCES")
        header.setStyleSheet(
            theme.style(PRESTIGE_ACCENT_LABEL, accent=accent, font_size="11px", letter_spacing="2px")
        )
        count = len(src_list)
        subtitle = QLabel(
            f"{count} source{'s' if count != 1 else ''} cited in this answer"
            if count
            else "No sources were attached to this answer."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(theme.style(PRESTIGE_MUTED_LABEL, font_size="13px"))

        inner.addWidget(header)
        inner.addWidget(subtitle)

        why_summary = str(transparency.get("why_summary") or "").strip()
        if why_summary:
            why_hdr = QLabel("WHY THESE SOURCES")
            why_hdr.setStyleSheet(
                theme.style(
                    PRESTIGE_ACCENT_LABEL,
                    accent=accent,
                    font_size="10px",
                    letter_spacing="1.5px",
                )
            )
            why_body = QLabel(why_summary.replace("\n", "\n"))
            why_body.setWordWrap(True)
            why_body.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            why_body.setStyleSheet(
                theme.style(PRESTIGE_BODY_LABEL, font_size="12px", font_weight="400")
                + " line-height: 1.45; padding: 8px 0;"
            )
            inner.addWidget(why_hdr)
            inner.addWidget(why_body)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        list_host = QWidget()
        list_layout = QVBoxLayout(list_host)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(12)

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
            row.setStyleSheet(theme.style(PRESTIGE_CITATION_ROW))
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(16, 14, 16, 14)
            row_layout.setSpacing(8)

            title_row = QHBoxLayout()
            title_row.setSpacing(10)
            cite_lbl = QLabel(_source_cite_label(src))
            cite_lbl.setStyleSheet(
                theme.style(PRESTIGE_ACCENT_LABEL, accent=accent, font_size="13px", letter_spacing="0px")
            )
            name = str(src.get("filename") or "Untitled source").strip()
            title_lbl = QLabel(name)
            title_lbl.setWordWrap(True)
            title_lbl.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
            title_lbl.setStyleSheet(theme.style(PRESTIGE_BODY_LABEL, font_size="14px"))
            title_row.addWidget(cite_lbl, 0)
            title_row.addWidget(title_lbl, 1)
            row_layout.addLayout(title_row)

            type_lbl = QLabel(_source_type_label(src))
            type_lbl.setWordWrap(True)
            type_lbl.setStyleSheet(
                theme.style(PRESTIGE_MUTED_LABEL, font_size="11px", font_weight="600")
                + " letter-spacing: 0.5px;"
            )
            row_layout.addWidget(type_lbl)

            meta_line = _source_metadata_line(src)
            if meta_line:
                meta_lbl = QLabel(meta_line)
                meta_lbl.setWordWrap(True)
                meta_lbl.setStyleSheet(
                    theme.style(PRESTIGE_MUTED_LABEL, font_size="11px", font_weight="600")
                )
                row_layout.addWidget(meta_lbl)

            snippet = str(src.get("content") or "").strip()
            url = str(src.get("url") or "").strip()
            if snippet:
                cap = _CITATION_SOURCES_SNIPPET_MAX
                preview = (
                    snippet
                    if len(snippet) <= cap
                    else snippet[: cap - 1].rstrip() + "…"
                )
                body_lbl = QLabel(preview)
                body_lbl.setWordWrap(True)
                body_lbl.setStyleSheet(
                    theme.style(PRESTIGE_MUTED_LABEL, font_size="13px")
                    + " line-height: 1.45;"
                )
                row_layout.addWidget(body_lbl)
            elif url:
                url_lbl = QLabel(url)
                url_lbl.setWordWrap(True)
                url_lbl.setStyleSheet(theme.style(PRESTIGE_LINK_LABEL))
                row_layout.addWidget(url_lbl)

            list_layout.addWidget(row)

        list_layout.addStretch(1)
        scroll.setWidget(list_host)
        inner.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        export_style = theme.style(PRESTIGE_GHOST_BUTTON, compact=True)
        if src_list:
            from PyQt6.QtWidgets import QApplication

            from core.knowledge.evidence_citations import sources_to_apa, sources_to_bibtex

            bibtex_btn = QPushButton("COPY BIBTEX")
            apa_btn = QPushButton("COPY APA")
            bibtex_btn.setStyleSheet(export_style)
            apa_btn.setStyleSheet(export_style)
            bibtex_btn.clicked.connect(
                lambda: QApplication.clipboard().setText(
                    sources_to_bibtex(src_list)
                )
            )
            apa_btn.clicked.connect(
                lambda: QApplication.clipboard().setText(sources_to_apa(src_list))
            )
            btn_row.addWidget(bibtex_btn)
            btn_row.addWidget(apa_btn)
        if research_map_graph and callable(on_open_research_map):
            map_btn = QPushButton("RESEARCH MAP")
            map_btn.setStyleSheet(export_style)
            map_btn.clicked.connect(on_open_research_map)
            btn_row.addWidget(map_btn)
        if retrieval_bundle_id and retrieval_db is not None:
            inspect_btn = QPushButton("INSPECT RETRIEVAL")
            inspect_btn.setStyleSheet(export_style)

            def _open_inspector() -> None:
                from core.knowledge.retrieval_trace_reader import read_last_retrieval_trace
                from ui.components.retrieval_inspector import open_retrieval_inspector_dialog

                record = retrieval_db.get_retrieval_record(bundle_id=retrieval_bundle_id)
                trace = read_last_retrieval_trace()
                preset_id = None
                if record:
                    preset_id = record.get("preset_id")
                elif trace:
                    preset_id = trace.get("preset_id")
                open_retrieval_inspector_dialog(
                    self,
                    is_dark=theme.is_dark,
                    trace=trace,
                    record=record,
                    preset_id=preset_id,
                    db=retrieval_db,
                )

            inspect_btn.clicked.connect(_open_inspector)
            btn_row.addWidget(inspect_btn)
        btn_row.addStretch()
        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet(theme.style(PRESTIGE_GHOST_BUTTON))
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        inner.addLayout(btn_row)

        outer.addWidget(container)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        _frame_dialog_within_host(
            self,
            preferred_width=_CITATION_SOURCES_DEFAULT_W,
            preferred_height=_CITATION_SOURCES_DEFAULT_H,
            min_width=_CITATION_SOURCES_MIN_W,
            min_height=_CITATION_SOURCES_MIN_H,
        )
        self.raise_()
        self.activateWindow()
