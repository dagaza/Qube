import os
import sys

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QListWidget,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QTextBrowser,
    QMenu,
    QGraphicsOpacityEffect,
)
from PyQt6.QtGui import (
    QAction,
    QTextOption,
    QTextBlockFormat,
    QTextCursor,
    QIcon,
    QColor,
    QPalette,
    QPixmap,
    QPainter,
    QFont,
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QEvent, QCoreApplication, QUrl, QSize
import math
import qtawesome as qta
import logging
from urllib.parse import unquote
import copy
import unicodedata
import weakref
import re
import re as _re_cite

from core.richtext_styles import markdown_document_stylesheet as _markdown_ui_stylesheet
from ui.components.prestige_dialog import PrestigeDialog
from ui.components.readability_toolbar_styles import readability_font_pair_stylesheet
from ui.components.sidebar_list_qss import apply_sidebar_row_title_colors
from ui.components.source_viewer import SourcePreviewer
from core.app_settings import get_engine_mode, set_native_reasoning_display_enabled

logger = logging.getLogger("Qube.UI.Conversations")

# In-text citation links must survive Qt's Markdown → anchor step. Qt's importer is flaky for
# custom schemes with numeric path segments (qube://cite/1); https + .invalid is linkified reliably.
CITATION_HREF_PREFIX = "https://qube.invalid/cite/"

# log_agent_token(..., citation_sources=...) — distinguish "not passed" (live stream) from explicit None (DB row with no sources)
_UNSET_SOURCES = object()

# --------------- Chat layout modes --------------- #
LAYOUT_FULL_WIDTH = "full_width"
LAYOUT_CENTERED_COLUMN = "centered_column"
_CENTERED_COLUMN_MAX_WIDTH = 800
_QWIDGETSIZE_MAX = (1 << 24) - 1
_LAYOUT_ICON_WIDE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "icons", "layout-wide.svg")
)
_LAYOUT_ICON_NARROW = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "icons", "layout-narrow.svg")
)
_LINE_SPACING_ICON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "icons", "line-spacing.svg")
)

# Chat utility toolbar: uniform icon / hit-target sizes
_CHAT_UTILITY_BTN = 30
_CHAT_UTILITY_ICON_PX = 18

# Readability (transcript-local; no persistence yet)
_BASE_CHAT_FONT_PT = 10.0
_FONT_SCALE_MIN = 0.85
_FONT_SCALE_MAX = 1.3
_FONT_SCALE_STEP = 0.05
_FONT_SCALE_STEP_COARSE = 0.1
_LINE_HEIGHT_COMPACT = "compact"
_LINE_HEIGHT_COMFORTABLE = "comfortable"
_LINE_HEIGHT_RELAXED = "relaxed"
_LINE_HEIGHT_CSS = {
    _LINE_HEIGHT_COMPACT: "1.25",
    _LINE_HEIGHT_COMFORTABLE: "1.45",
    _LINE_HEIGHT_RELAXED: "1.65",
}

ALIGN_LEFT = "align_left"
ALIGN_JUSTIFY = "align_justify"

# Smart auto-scroll: only follow new tokens if the scrollbar was already at (or near) the bottom.
_STICKY_SCROLL_TOLERANCE_PX = 24
def _parent_conversations_view(widget: QWidget):
    """Find ConversationsView ancestor so context menus can use _apply_menu_theme (Prestige styling)."""
    p = widget.parentWidget()
    while p is not None:
        if hasattr(p, "_apply_menu_theme"):
            return p
        p = p.parentWidget()
    return None


def _normalize_citation_id(value) -> str:
    """Single canonical form for matching cite tokens across JSON (int/float/str), Qt URLs, and LLM [W]."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).strip()
    s = str(value).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except ValueError:
        return s


def _source_citation_match_keys(src: dict) -> set[str]:
    """Normalized id values to compare against a clicked cite token (handles alternate keys)."""
    out: set[str] = set()
    if not isinstance(src, dict):
        return out
    for key in ("id", "cite_id", "source_id"):
        if key not in src:
            continue
        n = _normalize_citation_id(src.get(key))
        if n:
            out.add(n)
    return out


def _normalize_stored_source_id(src: dict) -> None:
    """Ensure citation ids are JSON-stable scalars; web citations use the string 'W' (not list/tuple)."""
    if not isinstance(src, dict):
        return
    rid = src.get("id")
    if isinstance(rid, (list, tuple)) and len(rid) == 1:
        rid = rid[0]
        src["id"] = rid
    if isinstance(rid, str) and rid.strip().upper() == "W":
        src["id"] = "W"
        return
    st = str(src.get("type", "")).lower()
    if st == "web" and rid in (None, ""):
        src["id"] = "W"


def _snapshot_citation_sources(sources) -> list:
    """Deep copy so each bubble owns an isolated list/dict graph (no cross-bubble mutation)."""
    if not sources:
        return []
    out = copy.deepcopy(list(sources))
    for src in out:
        _normalize_stored_source_id(src)
    return out


def _prepare_stream_for_qt_citation_links(raw: str) -> str:
    """
    Normalize citations before QLabel Markdown linkify. Later turns often emit
    [1](url), [[1]], or `[1]` which would otherwise stack with our link syntax and break Qt.
    """
    if not raw:
        return raw
    s = unicodedata.normalize("NFKC", raw)
    s = s.replace("\uff3b", "[").replace("\uff3d", "]")
    # Model-authored markdown links for citations → plain [n] / [W]
    s = _re_cite.sub(
        r"\[(\d+|[wW])\]\([^\)]*\)",
        lambda m: "[W]" if m.group(1).lower() == "w" else f"[{m.group(1)}]",
        s,
    )
    # Double-bracket wrappers e.g. [[1]] → [1] (repeat — models sometimes nest)
    for _ in range(4):
        ns = _re_cite.sub(
            r"\[\[(\d+|[wW])\]\]",
            lambda m: "[W]" if m.group(1).lower() == "w" else f"[{m.group(1)}]",
            s,
        )
        if ns == s:
            break
        s = ns

    def _unwrap_bt(m):
        inner = m.group(1)
        return "[W]" if inner.lower() == "[w]" else inner

    s = _re_cite.sub(r"`(\[\d+\]|\[[wW]\])`", _unwrap_bt, s)
    return s


def _markdown_cite_link_replacement(match) -> str:
    token = match.group(1)
    key = "W" if str(token).lower() == "w" else str(token)
    return f"[[{key}]](<{CITATION_HREF_PREFIX}{key}>)"


# Qt's setMarkdown() uses a GFM-ish parser: un-fenced lines with | / + / - can be parsed as tables.
# ASCII schematics (box-drawing with "+---+|") often produce a malformed table and break parsing for
# the *rest* of the document. Fence those blocks as literal code so later real tables still parse.


def _line_looks_like_box_drawing(line: str) -> bool:
    """Heuristic: schematic / box art line (not a normal prose table row)."""
    t = line.rstrip()
    if not t:
        return False
    # Markdown table separator: | --- | --- | — keep as markdown
    if re.match(r"^\s*\|(\s*:?-+:?\s*\|)+\s*$", t):
        return False
    # Strong signal: corner joints + edges
    if "+" in t and re.search(r"\+[-+|.\s]{2,}\+", t):
        return True
    if "+" in t and "|" in t and "-" in t and re.match(r"^[\s|+.\-=_/\\`:]+$", t):
        return True
    # Heavy structural characters, few letters (ASCII maps)
    pipe = t.count("|")
    letters = sum(1 for c in t if c.isalpha())
    if pipe >= 2 and letters <= max(2, len(t) // 10):
        if any(c in t for c in "+-|"):
            return True
    return False


def _fence_box_drawing_for_qt(text: str) -> str:
    """Wrap detected ASCII/box-drawing runs in fenced code blocks (outside existing ``` fences)."""
    if not text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    in_fence = False
    n = len(lines)

    while i < n:
        line = lines[i]
        st = line.strip()
        if st.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue

        if _line_looks_like_box_drawing(line):
            j = i
            buf: list[str] = []
            while j < n:
                ln = lines[j]
                lst = ln.strip()
                if lst.startswith("```"):
                    break
                if not ln.strip():
                    if buf and j + 1 < n and _line_looks_like_box_drawing(lines[j + 1]):
                        buf.append(ln)
                        j += 1
                        continue
                    if buf:
                        break
                    j += 1
                    continue
                if _line_looks_like_box_drawing(ln):
                    buf.append(ln)
                    j += 1
                    continue
                break
            if len(buf) >= 2:
                out.append("```")
                out.extend(buf)
                out.append("```")
                i = j
                continue

        out.append(line)
        i += 1

    return "\n".join(out)


def _qt_safe_markdown(markdown: str) -> str:
    """Sanitize LLM markdown before QTextDocument.setMarkdown (box art, future rules)."""
    return _fence_box_drawing_for_qt(markdown or "")


def _maybe_dump_markdown_html_pipeline(raw_md: str, font, is_dark: bool) -> None:
    """Set env QUBE_DUMP_MARKDOWN_HTML=1 to print to stderr (debug Qt swallow vs QLabel limits)."""
    if not os.environ.get("QUBE_DUMP_MARKDOWN_HTML"):
        return
    from PyQt6.QtGui import QTextDocument

    safe = _qt_safe_markdown(raw_md)
    doc = QTextDocument()
    doc.setDefaultFont(font)
    doc.setDefaultStyleSheet(_markdown_ui_stylesheet(is_dark))
    doc.setMarkdown(safe)
    html = doc.toHtml()
    sys.stderr.write(
        f"\n--- QUBE_DUMP_MARKDOWN_HTML len(raw)={len(raw_md)} len(safe)={len(safe)} len(html)={len(html)} ---\n"
    )
    cap = 250_000
    sys.stderr.write(html if len(html) <= cap else html[:cap] + "\n...[truncated]...\n")


class ChatLabel(QLabel):
    """User bubble: QLabel with width hint. Assistant replies use AgentMessageLabel instead."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._cached_text = ""
        self._cached_ideal_width = 0

    def cleanup_before_destruction(self) -> None:
        self._cached_text = ""
        self._cached_ideal_width = 0
        try:
            self.clear()
        except RuntimeError:
            pass

    def sizeHint(self):
        from PyQt6.QtGui import QTextDocument
        from PyQt6.QtCore import QSize, Qt

        hint = super().sizeHint()
        layout_key = self.text()
        if layout_key != self._cached_text:
            doc = QTextDocument()
            doc.setDefaultFont(self.font())
            if self.textFormat() == Qt.TextFormat.MarkdownText:
                doc.setMarkdown(layout_key)
            else:
                doc.setPlainText(layout_key)
            self._cached_ideal_width = int(doc.idealWidth()) + 15
            self._cached_text = layout_key
        return QSize(max(self._cached_ideal_width, hint.width()), hint.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateGeometry()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setObjectName("PrestigeMenu")
        view = _parent_conversations_view(self)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        if view is not None:
            view._apply_menu_theme(menu, is_dark)

        def _copy():
            if self.hasSelectedText():
                QApplication.clipboard().setText(self.selectedText())
            elif self.text():
                QApplication.clipboard().setText(self.text())

        copy_act = QAction("Copy", self)
        copy_act.triggered.connect(_copy)
        copy_act.setEnabled(bool(self.text()))
        menu.addAction(copy_act)
        menu.exec(event.globalPos())


class AgentMessageLabel(QTextBrowser):
    """
    Assistant bubble: read-only QTextBrowser shares QTextDocument with Qt's Markdown importer
    but applies full-document CSS and sets text width on resize ( QLabel + toHtml() could clip
    complex layouts). Pipe/box ASCII is pre-sanitized via _qt_safe_markdown().
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.setTabChangesFocus(False)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.document().setDocumentMargin(4)
        self.viewport().setAutoFillBackground(False)

        self._citation_sources: list = []
        self._conversations_view_ref = None
        self._citation_anchor_connected = False
        self._md_layout_source = ""
        self._agent_is_dark = True
        self._doc_layout_connected = False
        self._fixed_h = 0
        self._syncing_height = False

        doc_layout = self.document().documentLayout()
        if doc_layout is not None and hasattr(doc_layout, "documentSizeChanged"):
            doc_layout.documentSizeChanged.connect(self._on_document_size_changed)
            self._doc_layout_connected = True

    def _apply_document_paragraph_formats(
        self, doc, pct: int, justify_transcript: bool
    ) -> None:
        """Line height + horizontal alignment in one merge per block."""
        fmt = QTextBlockFormat()
        fmt.setLineHeight(float(pct), 1)
        fmt.setAlignment(
            Qt.AlignmentFlag.AlignJustify
            if justify_transcript
            else Qt.AlignmentFlag.AlignLeft
        )
        cur = QTextCursor(doc)
        cur.beginEditBlock()
        block = doc.firstBlock()
        while block.isValid():
            cur.setPosition(block.position())
            cur.mergeBlockFormat(fmt)
            block = block.next()
        cur.endEditBlock()

    def set_agent_markdown(
        self,
        markdown: str,
        *,
        is_dark: bool,
        document_stylesheet: str | None = None,
        line_height_percent: int | None = None,
        justify_transcript: bool = False,
    ) -> None:
        self._agent_is_dark = is_dark
        self._md_layout_source = markdown or ""
        safe = _qt_safe_markdown(self._md_layout_source)
        doc = self.document()
        doc.setDefaultFont(self.font())
        doc.setDefaultStyleSheet(
            document_stylesheet
            if document_stylesheet is not None
            else _markdown_ui_stylesheet(is_dark)
        )
        doc.setTextWidth(self.viewport().width())
        doc.setMarkdown(safe)
        pct = (
            line_height_percent
            if line_height_percent is not None
            else int(round(float(_LINE_HEIGHT_CSS[_LINE_HEIGHT_COMFORTABLE]) * 100))
        )
        self._apply_document_paragraph_formats(doc, pct, justify_transcript)
        _maybe_dump_markdown_html_pipeline(self._md_layout_source, self.font(), is_dark)
        self._sync_fixed_height()
        self.updateGeometry()

    def attach_citation_handling(self, conversations_view):
        self._conversations_view_ref = (
            weakref.ref(conversations_view) if conversations_view is not None else None
        )
        if not self._citation_anchor_connected:
            self.anchorClicked.connect(self._on_anchor_clicked)
            self._citation_anchor_connected = True

    def _on_anchor_clicked(self, url: QUrl):
        ref = self._conversations_view_ref
        view = ref() if ref is not None else None
        if view is not None and hasattr(view, "_resolve_citation_link_for_label"):
            view._resolve_citation_link_for_label(self, url.toString() if url.isValid() else "")

    def cleanup_before_destruction(self) -> None:
        if self._citation_anchor_connected:
            try:
                self.anchorClicked.disconnect(self._on_anchor_clicked)
            except TypeError:
                pass
            self._citation_anchor_connected = False
        self._conversations_view_ref = None
        self._citation_sources = []
        self._md_layout_source = ""
        if self._doc_layout_connected:
            try:
                self.document().documentLayout().documentSizeChanged.disconnect(
                    self._on_document_size_changed
                )
            except TypeError:
                pass
            self._doc_layout_connected = False
        try:
            self.clear()
        except RuntimeError:
            pass

    def sizeHint(self):
        return super().sizeHint()

    def heightForWidth(self, w: int) -> int:
        if w <= 0:
            return super().heightForWidth(w)
        return self._compute_doc_height(w)

    def hasHeightForWidth(self) -> bool:
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.document().setTextWidth(self.viewport().width())
        self._sync_fixed_height()
        self.updateGeometry()

    def _compute_doc_height(self, width: int) -> int:
        doc = self.document()
        content_w = self._content_width_from_outer_width(width)
        doc.setTextWidth(max(content_w, 1))
        doc_layout = doc.documentLayout()
        doc_h = doc_layout.documentSize().height() if doc_layout is not None else doc.size().height()
        return int(math.ceil(doc_h + self._non_document_vertical_space()))

    def _content_width_from_outer_width(self, outer_width: int) -> int:
        cm = self.contentsMargins()
        vm = self.viewportMargins()
        frame = self.frameWidth() * 2
        available = (
            int(outer_width)
            - frame
            - cm.left()
            - cm.right()
            - vm.left()
            - vm.right()
        )
        return max(1, available)

    def _non_document_vertical_space(self) -> int:
        cm = self.contentsMargins()
        vm = self.viewportMargins()
        frame = self.frameWidth() * 2
        return (
            frame
            + cm.top()
            + cm.bottom()
            + vm.top()
            + vm.bottom()
            + 2  # conservative safety pad for final line descenders
        )

    def _on_document_size_changed(self, _size) -> None:
        self._sync_fixed_height()

    def _sync_fixed_height(self) -> None:
        if self._syncing_height:
            return
        self._syncing_height = True
        try:
            w = max(self.width(), 1)
            h = self._compute_doc_height(w)
            if h != self._fixed_h:
                self._fixed_h = h
                self.setFixedHeight(h)
        finally:
            self._syncing_height = False

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setObjectName("PrestigeMenu")
        view = _parent_conversations_view(self)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        if view is not None:
            view._apply_menu_theme(menu, is_dark)

        tc = self.textCursor()
        copy_act = QAction("Copy", self)
        copy_act.setEnabled(tc.hasSelection())
        copy_act.triggered.connect(self.copy)
        menu.addAction(copy_act)

        sel_act = QAction("Select All", self)
        sel_act.triggered.connect(self.selectAll)
        menu.addAction(sel_act)

        menu.exec(event.globalPos())


class MessageWrapper(QWidget):
    """An autonomous layout row that takes full width and safely manages bubble expansion."""
    def __init__(self, bubble: QWidget, is_user: bool, parent=None):
        super().__init__(parent)
        self.bubble = bubble
        self.is_user = is_user  # 🔑 Save this state to use during resizing
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if self.is_user:
            layout.addStretch(1)
            layout.addWidget(bubble, 0, Qt.AlignmentFlag.AlignRight)
        else:
            layout.addWidget(bubble, 1)

    def cleanup_before_destruction(self) -> None:
        """Break references held by this row before Qt tears down the widget tree."""
        for lbl in self.findChildren(ChatLabel):
            lbl.cleanup_before_destruction()
        for w in self.findChildren(AgentMessageLabel):
            w.cleanup_before_destruction()
        self.bubble = None


class _ComposerRowHost(QWidget):
    """Keeps composer controls at width min(available, max_w), centered — matches transcript column cap."""

    def __init__(self, inner: QWidget, max_w: int, parent=None):
        super().__init__(parent)
        self._inner = inner
        self._max_w = max_w
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addStretch(1)
        layout.addWidget(inner, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = min(self._max_w, max(1, self.width()))
        if self._inner.width() != w:
            self._inner.setFixedWidth(w)


class ConversationsView(QWidget):
    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        self.llm = workers.get("llm")
        self.tts = workers.get("tts")
        self._pending_citation_sources = None
        self._user_turn_id = 0
        self._stop_requested_callback = None
        self._llm_in_progress = False
        self._awaiting_tts_end = False
        self._tts_playing = False
        self._layout_mode: str = LAYOUT_CENTERED_COLUMN
        self._font_scale: float = 1.0
        self._line_height_mode: str = _LINE_HEIGHT_COMFORTABLE
        self._focus_mode_enabled: bool = False
        self._high_contrast_enabled: bool = False
        self._reader_hover_wrapper: MessageWrapper | None = None
        self._transcript_alignment: str = ALIGN_JUSTIFY

        self._setup_ui()
        self._start_new_chat()

    def _notify_llm_active_session_changed(self) -> None:
        """Tell the LLM worker the focused thread changed so the local server can drop stale KV/prompt cache."""
        llm = getattr(self, "llm", None)
        if llm is not None and hasattr(llm, "notify_active_session_changed"):
            llm.notify_active_session_changed(getattr(self, "active_session_id", None))

    # --------------------------------------------------------- #
    #  LAYOUT MODE (container-level only)                        #
    # --------------------------------------------------------- #

    @property
    def layout_mode(self) -> str:
        return self._layout_mode

    def set_layout_mode(self, mode: str) -> None:
        """Switch the chat transcript between FULL_WIDTH and CENTERED_COLUMN layout.

        This only reconfigures container-level constraints and scroll-area
        alignment — individual message widgets and QTextDocument rendering
        are never touched.
        """
        if mode not in (LAYOUT_FULL_WIDTH, LAYOUT_CENTERED_COLUMN):
            return
        if mode == self._layout_mode:
            self._refresh_layout_mode_button()
            return
        self._layout_mode = mode
        self._apply_layout_mode()

    def _apply_layout_mode(self) -> None:
        if self._layout_mode == LAYOUT_CENTERED_COLUMN:
            self.transcript_container.setMaximumWidth(_CENTERED_COLUMN_MAX_WIDTH)
            self.scroll_area.setAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
            )
        else:
            self.transcript_container.setMaximumWidth(_QWIDGETSIZE_MAX)
            self.scroll_area.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )
        self.transcript_layout.invalidate()
        self.transcript_container.updateGeometry()
        self._refresh_layout_mode_button()

    def _make_tinted_svg_icon(self, svg_path: str, color_hex: str, size: int = 18) -> QIcon:
        pixmap = QPixmap(svg_path)
        if pixmap.isNull():
            return QIcon(svg_path)
        target_size = QSize(size, size)
        pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        tinted = QPixmap(pixmap.size())
        tinted.fill(Qt.GlobalColor.transparent)
        painter = QPainter(tinted)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), QColor(color_hex))
        painter.end()
        return QIcon(tinted)

    def _refresh_layout_mode_button(self, is_dark: bool | None = None) -> None:
        btn = getattr(self, "layout_mode_btn", None)
        if btn is None:
            return
        if is_dark is None:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
        icon_color = "#8b5cf6" if is_dark else "#1e293b"
        hover_bg = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.05)"
        if self.layout_mode == LAYOUT_CENTERED_COLUMN:
            btn.setIcon(
                self._make_tinted_svg_icon(
                    _LAYOUT_ICON_NARROW, icon_color, size=_CHAT_UTILITY_ICON_PX
                )
            )
            btn.setToolTip("Layout mode: Centered column")
        else:
            btn.setIcon(
                self._make_tinted_svg_icon(
                    _LAYOUT_ICON_WIDE, icon_color, size=_CHAT_UTILITY_ICON_PX
                )
            )
            btn.setToolTip("Layout mode: Full width")
        btn.setIconSize(QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX))
        btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        btn.setStyleSheet(
            f"""
            QPushButton {{ background: transparent; border: none; border-radius: 6px; padding: 6px; }}
            QPushButton:hover {{ background-color: {hover_bg}; }}
            """
        )

    def _toggle_layout_mode(self) -> None:
        next_mode = (
            LAYOUT_CENTERED_COLUMN
            if self.layout_mode == LAYOUT_FULL_WIDTH
            else LAYOUT_FULL_WIDTH
        )
        self.set_layout_mode(next_mode)

    # --------------------------------------------------------- #
    #  READABILITY / ACCESSIBILITY (transcript-local)            #
    # --------------------------------------------------------- #

    def _scaled_chat_font_pt(self) -> float:
        return max(8.0, min(28.0, _BASE_CHAT_FONT_PT * self._font_scale))

    def _line_height_css_value(self) -> str:
        return _LINE_HEIGHT_CSS.get(
            self._line_height_mode, _LINE_HEIGHT_CSS[_LINE_HEIGHT_COMFORTABLE]
        )

    def _line_height_proportional_percent(self) -> int:
        """Proportional line height percent for QTextBlockFormat (e.g. 145 → 1.45× natural)."""
        try:
            return int(round(float(self._line_height_css_value()) * 100))
        except ValueError:
            return 145

    def _high_contrast_markdown_css(self, is_dark: bool) -> str:
        if not self._high_contrast_enabled:
            return ""
        fg, _ = self._user_bubble_label_colors(is_dark)
        code_bg = self._user_bubble_frame_bg(is_dark)
        if is_dark:
            return (
                f"body, p, span, div, li, ul, ol, dd, dt, "
                f"table, thead, tbody, tr, th, td, "
                f"blockquote, "
                f"h1, h2, h3, h4, h5, h6, strong, em {{ color: {fg}; }}"
                f"a:link, a {{ color: #93c5fd; text-decoration: none; }}"
                f"a:visited {{ color: #c4b5fd; text-decoration: none; }}"
                f"code, pre {{ background-color: {code_bg}; color: {fg}; }}"
                f"table {{ border-color: #94a3b8; }}"
                f"th, td {{ border-color: #94a3b8; border-width: 1px; border-style: solid; }}"
                f"hr {{ border-color: #94a3b8; color: #94a3b8; }}"
            )
        return (
            f"body, p, span, div, li, ul, ol, dd, dt, "
            f"table, thead, tbody, tr, th, td, "
            f"blockquote, "
            f"h1, h2, h3, h4, h5, h6, strong, em {{ color: {fg}; }}"
            f"a:link, a {{ color: #1d4ed8; text-decoration: none; }}"
            f"a:visited {{ color: #6d28d9; text-decoration: none; }}"
            f"code, pre {{ background-color: {code_bg}; color: {fg}; }}"
            f"table {{ border-color: #475569; }}"
            f"th, td {{ border-color: #475569; border-width: 1px; border-style: solid; }}"
            f"hr {{ border-color: #475569; color: #475569; }}"
        )

    def _agent_markdown_stylesheet(self, is_dark: bool) -> str:
        base = _markdown_ui_stylesheet(is_dark)
        parts = [base, self._high_contrast_markdown_css(is_dark)]
        return "".join(parts)

    def _user_bubble_label_colors(self, is_dark: bool) -> tuple[str, str]:
        """(text_color, optional extra label style fragment)."""
        if self._high_contrast_enabled:
            if is_dark:
                return "#020617", ""
            return "#000000", ""
        return "#11111b", ""

    def _user_bubble_frame_bg(self, is_dark: bool) -> str:
        if self._high_contrast_enabled:
            if is_dark:
                return "#7dd3fc"
            return "#38bdf8"
        return "#89b4fa"

    def _qube_response_header_color(self, is_dark: bool) -> str:
        """Assistant turn 'QUBE' label — unchanged by high-contrast transcript mode."""
        return "#8b5cf6" if is_dark else "#8839ef"

    def _placeholder_muted_color(self, is_dark: bool) -> str:
        if self._high_contrast_enabled:
            return "#cbd5e1" if is_dark else "#475569"
        return "#6c7086"

    def _style_user_bubble(self, bubble: QFrame, lbl: ChatLabel) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        pt = self._scaled_chat_font_pt()
        fg, _ = self._user_bubble_label_colors(is_dark)
        bg = self._user_bubble_frame_bg(is_dark)
        f = lbl.font()
        f.setPointSizeF(pt)
        lbl.setFont(f)
        lbl._cached_text = ""
        lbl._cached_ideal_width = 0
        if self._transcript_alignment == ALIGN_JUSTIFY:
            lbl.setAlignment(
                Qt.AlignmentFlag.AlignJustify | Qt.AlignmentFlag.AlignTop
            )
        else:
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        lbl.setStyleSheet(
            f"background: transparent; border: none; font-size: {pt:.1f}pt; color: {fg};"
        )
        bubble.setStyleSheet(
            f"background-color: {bg}; border-radius: 18px;"
        )

    def _style_agent_message_shell(self, agent: AgentMessageLabel) -> None:
        pt = self._scaled_chat_font_pt()
        f = agent.font()
        f.setPointSizeF(pt)
        agent.setFont(f)
        agent.setStyleSheet(
            f"font-size: {pt:.1f}pt; background: transparent; border: none;"
        )

    def _iter_transcript_widgets(self):
        if not hasattr(self, "transcript_layout"):
            return
        for i in range(self.transcript_layout.count()):
            it = self.transcript_layout.itemAt(i)
            if it is None:
                continue
            w = it.widget()
            if w is not None:
                yield w

    def _find_latest_message_wrapper(self) -> MessageWrapper | None:
        last = None
        for w in self._iter_transcript_widgets():
            if isinstance(w, MessageWrapper):
                last = w
        return last

    def _register_reader_focus_tracking(self, wrapper: MessageWrapper) -> None:
        wrapper.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        wrapper.installEventFilter(self)

    def _apply_reader_focus_opacity(self) -> None:
        if not self._focus_mode_enabled:
            self._clear_reader_focus_effects()
            return
        target = self._reader_hover_wrapper or self._find_latest_message_wrapper()
        dim = 0.58
        pl = getattr(self, "placeholder_lbl", None)
        for i in range(self.transcript_layout.count()):
            it = self.transcript_layout.itemAt(i)
            if it is None:
                continue
            w = it.widget()
            if w is None:
                continue
            if w is pl:
                self._set_widget_opacity(w, dim)
                continue
            if isinstance(w, MessageWrapper):
                self._set_widget_opacity(w, 1.0 if w is target else dim)
                continue
            if isinstance(w, QLabel):
                nxt = None
                if i + 1 < self.transcript_layout.count():
                    nxt = self.transcript_layout.itemAt(i + 1).widget()
                partner = target is not None and nxt is target
                self._set_widget_opacity(w, 1.0 if partner else dim)
                continue
            self._set_widget_opacity(w, dim)

    def _set_widget_opacity(self, w: QWidget, opacity: float) -> None:
        if opacity >= 0.999:
            w.setGraphicsEffect(None)
            return
        eff = w.graphicsEffect()
        if not isinstance(eff, QGraphicsOpacityEffect):
            eff = QGraphicsOpacityEffect(w)
            w.setGraphicsEffect(eff)
        eff.setOpacity(opacity)

    def _clear_reader_focus_effects(self) -> None:
        for w in self._iter_transcript_widgets():
            w.setGraphicsEffect(None)

    def _refresh_ancillary_transcript_labels(self) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        pt = self._scaled_chat_font_pt()
        muted = self._placeholder_muted_color(is_dark)
        for w in self._iter_transcript_widgets():
            if isinstance(w, QLabel) and w is not getattr(self, "placeholder_lbl", None):
                qube_hdr = self._qube_response_header_color(is_dark)
                w.setStyleSheet(
                    f"color: {qube_hdr}; font-weight: bold; font-size: {pt:.1f}pt; margin-top: 15px; background: transparent;"
                )
        pl = getattr(self, "placeholder_lbl", None)
        if pl is not None:
            pl.setStyleSheet(
                f"color: {muted}; font-size: {pt:.1f}pt; margin-top: 50px; font-weight: bold;"
            )

    def _refresh_all_readability(self) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        sheet = self._agent_markdown_stylesheet(is_dark)
        for w in self._iter_transcript_widgets():
            if isinstance(w, MessageWrapper):
                if w.is_user and w.bubble is not None:
                    lbl = w.bubble.findChild(ChatLabel)
                    if lbl is not None:
                        self._style_user_bubble(w.bubble, lbl)
                else:
                    for agent in w.findChildren(AgentMessageLabel):
                        self._style_agent_message_shell(agent)
                        if agent._md_layout_source:
                            agent.set_agent_markdown(
                                agent._md_layout_source,
                                is_dark=is_dark,
                                document_stylesheet=sheet,
                                line_height_percent=self._line_height_proportional_percent(),
                                justify_transcript=(
                                    self._transcript_alignment == ALIGN_JUSTIFY
                                ),
                            )
        self._refresh_ancillary_transcript_labels()
        self._refresh_readability_toolbar()
        if self._focus_mode_enabled:
            self._apply_reader_focus_opacity()
        else:
            self._clear_reader_focus_effects()

    def _nudge_font_scale(self, delta: float) -> None:
        new_v = round(self._font_scale + delta, 4)
        new_v = max(_FONT_SCALE_MIN, min(_FONT_SCALE_MAX, new_v))
        if new_v == self._font_scale:
            return
        self._font_scale = new_v
        self._refresh_all_readability()

    def _font_scale_step_for_click(self) -> float:
        mods = QApplication.keyboardModifiers()
        if bool(mods & Qt.KeyboardModifier.ShiftModifier):
            return _FONT_SCALE_STEP_COARSE
        return _FONT_SCALE_STEP

    def _on_font_minus_clicked(self) -> None:
        self._nudge_font_scale(-self._font_scale_step_for_click())

    def _on_font_plus_clicked(self) -> None:
        self._nudge_font_scale(self._font_scale_step_for_click())

    def _cycle_line_height_mode(self) -> None:
        order = (
            _LINE_HEIGHT_COMPACT,
            _LINE_HEIGHT_COMFORTABLE,
            _LINE_HEIGHT_RELAXED,
        )
        try:
            i = order.index(self._line_height_mode)
        except ValueError:
            i = 1
        self._line_height_mode = order[(i + 1) % len(order)]
        self._refresh_all_readability()

    def _on_reader_focus_toggled(self, checked: bool) -> None:
        self._focus_mode_enabled = bool(checked)
        if not self._focus_mode_enabled:
            self._reader_hover_wrapper = None
        self._refresh_readability_toolbar()
        self._apply_reader_focus_opacity()

    def _on_high_contrast_toggled(self, checked: bool) -> None:
        self._high_contrast_enabled = bool(checked)
        self._refresh_all_readability()

    def _cycle_transcript_alignment(self) -> None:
        self._transcript_alignment = (
            ALIGN_JUSTIFY
            if self._transcript_alignment == ALIGN_LEFT
            else ALIGN_LEFT
        )
        self._refresh_all_readability()

    def _refresh_readability_toolbar(self, is_dark: bool | None = None) -> None:
        if not hasattr(self, "line_height_btn"):
            return
        if is_dark is None:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
        self.font_minus_btn.setEnabled(self._font_scale > _FONT_SCALE_MIN + 1e-6)
        self.font_plus_btn.setEnabled(self._font_scale < _FONT_SCALE_MAX - 1e-6)
        mode_labels = {
            _LINE_HEIGHT_COMPACT: "Compact line spacing",
            _LINE_HEIGHT_COMFORTABLE: "Comfortable line spacing",
            _LINE_HEIGHT_RELAXED: "Relaxed line spacing",
        }
        self.line_height_btn.setToolTip(
            mode_labels.get(self._line_height_mode, "Line spacing")
        )
        self.reader_focus_btn.blockSignals(True)
        self.high_contrast_btn.blockSignals(True)
        try:
            self.reader_focus_btn.setChecked(self._focus_mode_enabled)
            self.high_contrast_btn.setChecked(self._high_contrast_enabled)
        finally:
            self.reader_focus_btn.blockSignals(False)
            self.high_contrast_btn.blockSignals(False)
        hover_bg = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.05)"
        icon_muted = "#8b5cf6" if is_dark else "#1e293b"
        icon_active = "#c4b5fd" if is_dark else "#2563eb"
        is_justify = self._transcript_alignment == ALIGN_JUSTIFY
        self.text_align_btn.setToolTip(
            "Text alignment: Justified (click for left)"
            if is_justify
            else "Text alignment: Left (click for justified)"
        )
        self.text_align_btn.setIcon(
            qta.icon(
                "fa5s.align-justify" if is_justify else "fa5s.align-left",
                color=icon_muted,
            )
        )
        self.text_align_btn.setIconSize(
            QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX)
        )
        self.text_align_btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        lh_icon_color = icon_muted
        self.line_height_btn.setIcon(
            self._make_tinted_svg_icon(_LINE_SPACING_ICON, lh_icon_color, size=_CHAT_UTILITY_ICON_PX)
        )
        self.line_height_btn.setIconSize(
            QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX)
        )
        self.line_height_btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        self.reader_focus_btn.setIcon(
            qta.icon(
                "fa5s.crosshairs",
                color=icon_active if self._focus_mode_enabled else icon_muted,
            )
        )
        self.reader_focus_btn.setIconSize(
            QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX)
        )
        self.reader_focus_btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        self.high_contrast_btn.setIcon(
            qta.icon(
                "fa5s.adjust",
                color=icon_active if self._high_contrast_enabled else icon_muted,
            )
        )
        self.high_contrast_btn.setIconSize(
            QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX)
        )
        self.high_contrast_btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        for btn in (
            self.line_height_btn,
            self.text_align_btn,
            self.reader_focus_btn,
            self.high_contrast_btn,
        ):
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    background: transparent;
                    border: none;
                    border-radius: 6px;
                    padding: 4px;
                }}
                QPushButton:hover {{
                    background-color: {hover_bg};
                }}
                """
            )

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1) 

        self.history_pane = self._build_history_pane()
        layout.addWidget(self.history_pane)

        self.chat_stage = self._build_chat_stage()

        # Let chat stage expand naturally with main viewport.
        self.chat_stage.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout.addWidget(self.chat_stage, stretch=1) 

        # --- ADD THIS LINE AT THE BOTTOM ---
        # Forces the buttons to load with the default Dark Mode purple on startup
        self.refresh_button_themes(is_dark=True)
        self.refresh_think_toggle()

    # --------------------------------------------------------- #
    #  PANEL BUILDERS                                           #
    # --------------------------------------------------------- #

    def _build_history_pane(self) -> QFrame:
        frame = QFrame()
        frame.setFixedWidth(280)
        frame.setObjectName("HistorySidebar")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)

        header_layout = QHBoxLayout()
        
        # --- THE FIX: Change 'title' to 'self.list_title' ---
        self.list_title = QLabel("Conversations")
        self.list_title.setObjectName("ViewTitle")
        
        self.new_chat_btn = QPushButton()
        self.new_chat_btn.setIcon(qta.icon('fa5s.plus'))
        self.new_chat_btn.setProperty("class", "IconButton")
        
        # --- THE FIX: Make sure you add 'self.list_title' to the layout here ---
        header_layout.addWidget(self.list_title)
        
        header_layout.addStretch()
        header_layout.addWidget(self.new_chat_btn)
        layout.addLayout(header_layout)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search titles or messages…")
        self.search_bar.setObjectName("HistorySearch")
        layout.addWidget(self.search_bar)
        self._history_search_timer = QTimer(self)
        self._history_search_timer.setSingleShot(True)
        self._history_search_timer.timeout.connect(self._reload_history_sidebar)
        self.search_bar.textChanged.connect(self._on_history_search_changed)

        self.history_list = QListWidget()
        self.history_list.setObjectName("HistoryList")
        layout.addWidget(self.history_list)

        self.new_chat_btn.clicked.connect(self._start_new_chat)
        self.history_list.itemClicked.connect(self._load_selected_chat)
        self.history_list.itemSelectionChanged.connect(self._update_row_colors)

        # 🔑 NEW: Wire up the scrollbar for infinite scrolling
        self.history_offset = 0
        self.is_loading_history = False
        self.history_list.verticalScrollBar().valueChanged.connect(self._on_history_scroll)

        return frame

    def _build_chat_stage(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("ChatStage")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        utility_toolbar = QFrame()
        utility_toolbar.setObjectName("ChatUtilityToolbar")
        utility_toolbar.setFixedHeight(40)
        utility_toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        utility_layout = QHBoxLayout(utility_toolbar)
        utility_layout.setContentsMargins(0, 0, 0, 0)
        utility_layout.setSpacing(8)

        readability_host = QWidget()
        readability_host.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        read_row = QHBoxLayout(readability_host)
        read_row.setContentsMargins(0, 0, 0, 0)
        read_row.setSpacing(6)

        self.font_minus_btn = QPushButton("A−")
        self.font_minus_btn.setObjectName("ReadabilityFontMinus")
        self.font_minus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.font_minus_btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        self.font_minus_btn.setToolTip(
            "Decrease chat font (Shift+click: larger step)"
        )
        self.font_minus_btn.clicked.connect(self._on_font_minus_clicked)

        self.font_plus_btn = QPushButton("A+")
        self.font_plus_btn.setObjectName("ReadabilityFontPlus")
        self.font_plus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.font_plus_btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        self.font_plus_btn.setToolTip(
            "Increase chat font (Shift+click: larger step)"
        )
        self.font_plus_btn.clicked.connect(self._on_font_plus_clicked)

        self.line_height_btn = QPushButton()
        self.line_height_btn.setObjectName("ReadabilityLineHeight")
        self.line_height_btn.setProperty("class", "IconButton")
        self.line_height_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.line_height_btn.clicked.connect(self._cycle_line_height_mode)

        self.text_align_btn = QPushButton()
        self.text_align_btn.setObjectName("ReadabilityTextAlign")
        self.text_align_btn.setProperty("class", "IconButton")
        self.text_align_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.text_align_btn.clicked.connect(self._cycle_transcript_alignment)

        self.reader_focus_btn = QPushButton()
        self.reader_focus_btn.setObjectName("ReadabilityReaderFocus")
        self.reader_focus_btn.setProperty("class", "IconButton")
        self.reader_focus_btn.setCheckable(True)
        self.reader_focus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reader_focus_btn.setToolTip("Reader focus: dim other messages")
        self.reader_focus_btn.toggled.connect(self._on_reader_focus_toggled)

        self.high_contrast_btn = QPushButton()
        self.high_contrast_btn.setObjectName("ReadabilityHighContrast")
        self.high_contrast_btn.setProperty("class", "IconButton")
        self.high_contrast_btn.setCheckable(True)
        self.high_contrast_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.high_contrast_btn.setToolTip("High contrast (chat transcript only)")
        self.high_contrast_btn.toggled.connect(self._on_high_contrast_toggled)

        self.layout_mode_btn = QPushButton()
        self.layout_mode_btn.setObjectName("LayoutModeButton")
        self.layout_mode_btn.setProperty("class", "IconButton")
        self.layout_mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.layout_mode_btn.clicked.connect(self._toggle_layout_mode)

        read_row.addWidget(self.font_minus_btn)
        read_row.addWidget(self.font_plus_btn)
        read_row.addWidget(self.line_height_btn)
        read_row.addWidget(self.text_align_btn)
        read_row.addWidget(self.reader_focus_btn)
        read_row.addWidget(self.high_contrast_btn)
        read_row.addWidget(self.layout_mode_btn)

        utility_layout.addWidget(readability_host, 0, Qt.AlignmentFlag.AlignLeft)
        utility_layout.addStretch(1)
        layout.addWidget(utility_toolbar)

        # 1. The New Architecture: A Scroll Area containing a vertical list of message widgets
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("ChatScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.installEventFilter(self)

        # Container widget
        self.transcript_container = QWidget()
        self.transcript_container.setObjectName("ChatTranscriptContainer")
        
        # 🔑 The line that was crashing is perfectly safe now
        self.transcript_container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.transcript_layout = QVBoxLayout(self.transcript_container)
        self.transcript_layout.setContentsMargins(0, 0, 0, 0)
        self.transcript_layout.setSpacing(12)
        self.transcript_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.transcript_container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.viewport().setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._refresh_readability_toolbar(is_dark=True)
        self._apply_layout_mode()
        layout.addWidget(self.scroll_area)

        # Bottom stack: fixed cap = centered transcript column width; not tied to layout toggle.
        self.chat_bottom_container = QWidget()
        self.chat_bottom_container.setObjectName("ChatBottomContainer")
        bottom_stack_layout = QVBoxLayout(self.chat_bottom_container)
        bottom_stack_layout.setContentsMargins(0, 0, 0, 0)
        bottom_stack_layout.setSpacing(layout.spacing())

        # 2. Per-message action bar
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        action_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.web_btn = QPushButton("Web")
        self.web_btn.setCheckable(True)
        self.web_btn.setProperty("class", "ThinkToggleButton")
        self.web_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.web_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.web_btn.toggled.connect(self._apply_action_toggle_styles)

        self.think_btn = QPushButton("Think")
        self.think_btn.setCheckable(True)
        self.think_btn.setProperty("class", "ThinkToggleButton")
        self.think_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.think_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.think_btn.toggled.connect(self._on_think_toggled)

        action_layout.addWidget(self.web_btn)
        action_layout.addWidget(self.think_btn)
        bottom_stack_layout.addLayout(action_layout)

        # 3. Input Bar Area
        input_container = QFrame()
        input_container.setObjectName("ChatInputContainer")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(10, 5, 5, 5)
        input_layout.setSpacing(8)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type a message to Qube...")
        self.text_input.setObjectName("ChatTextInput")
        
        self.send_btn = QPushButton()
        self.send_btn.setIcon(qta.icon('fa5s.paper-plane'))
        self.send_btn.setFixedSize(35, 35)
        self.send_btn.setProperty("class", "SendButton")

        input_layout.addWidget(self.text_input, stretch=1)
        input_layout.addWidget(self.send_btn)
        bottom_stack_layout.addWidget(input_container)

        # 4. Latency Metrics Footer
        latency_layout = QHBoxLayout()
        self.stt_latency_lbl = QLabel("STT: -- ms")
        self.ttft_latency_lbl = QLabel("TTFT: -- ms")
        self.tts_latency_lbl = QLabel("TTS: -- ms")
        
        for lbl in [self.stt_latency_lbl, self.ttft_latency_lbl, self.tts_latency_lbl]:
            lbl.setProperty("class", "MiniLatencyLabel")
            latency_layout.addWidget(lbl)
            
        latency_layout.addStretch()
        bottom_stack_layout.addLayout(latency_layout)

        self.chat_bottom_container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self._composer_row_host = _ComposerRowHost(
            self.chat_bottom_container, _CENTERED_COLUMN_MAX_WIDTH
        )
        self._composer_row_host.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        layout.addWidget(self._composer_row_host)

        self.send_btn.clicked.connect(self._handle_send_or_stop)
        self.text_input.returnPressed.connect(self._handle_text_submit)
        
        return frame

    # --------------------------------------------------------- #
    #  UI UPDATE RECEIVERS (The Magic Happens Here)             #
    # --------------------------------------------------------- #

    def log_user_message(self, text: str) -> None:
        self._clear_placeholders()
        # New user turn: drop stale assistant pointer so Turn N+1 tools cannot overwrite Turn N bubbles.
        self._user_turn_id += 1
        self.current_agent_msg = None

        bubble = QFrame()
        # 🔑 FIX 2: Allow bubble to expand horizontally, minimum vertically
        bubble.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        bubble.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(16, 12, 16, 12)

        lbl = ChatLabel(text)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._style_user_bubble(bubble, lbl)

        bubble_layout.addWidget(lbl)

        wrapper = MessageWrapper(bubble, is_user=True)
        self._register_reader_focus_tracking(wrapper)
        self.transcript_layout.addWidget(wrapper)
        
        self._is_agent_typing = False
        self._scroll_to_bottom()
        if self._focus_mode_enabled:
            self._apply_reader_focus_opacity()

    def log_agent_token(self, token: str, *, citation_sources=_UNSET_SOURCES) -> None:
        self._clear_placeholders()

        is_dark = True
        if self.window() and hasattr(self.window(), '_is_dark_theme'):
            is_dark = self.window()._is_dark_theme
            
        header_color = self._qube_response_header_color(is_dark)
        hdr_pt = self._scaled_chat_font_pt()

        if not getattr(self, '_is_agent_typing', False):
            header = QLabel("QUBE")
            header.setStyleSheet(
                f"color: {header_color}; font-weight: bold; font-size: {hdr_pt:.1f}pt; margin-top: 6px; background: transparent;"
            )
            self.transcript_layout.addWidget(header)

            self.agent_msg_container = QFrame()
            self.agent_msg_container.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            
            container_layout = QVBoxLayout(self.agent_msg_container)
            container_layout.setContentsMargins(0, 0, 0, 2)

            self.current_agent_msg = AgentMessageLabel()
            self.current_agent_msg.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            self.current_agent_msg._assistant_turn_id = self._user_turn_id
            self.current_agent_msg.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextBrowserInteraction |
                Qt.TextInteractionFlag.LinksAccessibleByMouse
            )
            self.current_agent_msg.attach_citation_handling(self)
            self._style_agent_message_shell(self.current_agent_msg)

            # Per-bubble citation context (survives new turns and session reloads)
            if citation_sources is not _UNSET_SOURCES:
                self.current_agent_msg._citation_sources = _snapshot_citation_sources(citation_sources)
            else:
                self._attach_pending_citation_sources(self.current_agent_msg)

            container_layout.addWidget(self.current_agent_msg)

            wrapper = MessageWrapper(self.agent_msg_container, is_user=False)
            self._register_reader_focus_tracking(wrapper)
            self.transcript_layout.addWidget(wrapper)
            
            self._agent_text_buffer = ""
            self._is_agent_typing = True

        self._agent_text_buffer += token

        # Strip model markdown around cites, then linkify plain [n]/[W] for Qt MarkdownText.
        prepared = _prepare_stream_for_qt_citation_links(self._agent_text_buffer)
        rich_text = _re_cite.sub(
            r"\[\s*(\d+|[wW])\s*\]",
            _markdown_cite_link_replacement,
            prepared,
        )

        # Sticky scroll: capture "at bottom" before content grows, then scroll only if user was following the stream.
        follow_stream_tail = self._is_transcript_scrolled_to_bottom()
        self.current_agent_msg.set_agent_markdown(
            rich_text,
            is_dark=is_dark,
            document_stylesheet=self._agent_markdown_stylesheet(is_dark),
            line_height_percent=self._line_height_proportional_percent(),
            justify_transcript=(self._transcript_alignment == ALIGN_JUSTIFY),
        )

        self.current_agent_msg.updateGeometry()
        if follow_stream_tail:
            self._scroll_to_bottom()
        if self._focus_mode_enabled:
            self._apply_reader_focus_opacity()

    def _clear_placeholders(self):
        if hasattr(self, 'placeholder_lbl') and self.placeholder_lbl:
            self.placeholder_lbl.hide()
            self.placeholder_lbl.deleteLater()
            self.placeholder_lbl = None

    def _teardown_transcript_row(self, row: QWidget) -> None:
        """Disconnect citations and clear label-owned data while the widget tree is still valid."""
        if isinstance(row, MessageWrapper):
            row.cleanup_before_destruction()
        else:
            for lbl in row.findChildren(ChatLabel):
                lbl.cleanup_before_destruction()
            for w in row.findChildren(AgentMessageLabel):
                w.cleanup_before_destruction()

    def _clear_transcript(self):
        """Destroys all message widgets to prepare for a new chat.

        deleteLater() alone can leave PyQt wrappers and citation payloads alive until GC if
        Python still holds strong references (signal slots, view pointers, source snapshots).
        We clear those explicitly, then flush DeferredDelete so QObject teardown runs promptly.
        """
        self._reader_hover_wrapper = None
        self._clear_reader_focus_effects()
        self.placeholder_lbl = None
        self.current_agent_msg = None
        self._pending_citation_sources = None
        self._agent_text_buffer = ""

        while self.transcript_layout.count():
            item = self.transcript_layout.takeAt(0)
            w = item.widget()
            if w is None:
                continue
            self._teardown_transcript_row(w)
            w.deleteLater()

        etype = getattr(QEvent.Type, "DeferredDelete", None)
        if etype is not None:
            try:
                QCoreApplication.sendPostedEvents(None, int(etype))
            except RuntimeError:
                pass

    # --------------------------------------------------------- #
    #  INTERACTION & LOGIC                                      #
    # --------------------------------------------------------- #

    def _handle_text_submit(self):
        if self._is_stop_mode():
            self._request_stop()
            return
        text = self.text_input.text().strip()
        if not text: return
        self.text_input.clear()
        self._llm_in_progress = True
        self._awaiting_tts_end = False
        self._tts_playing = False
        self.set_input_enabled(False)
        self._refresh_send_stop_button()

        self.log_user_message(text)

        if not hasattr(self, 'active_session_id'):
            recent_sessions = self.db.get_recent_sessions(limit=1)
            if recent_sessions:
                self.active_session_id = recent_sessions[0]['id']
            else:
                self.active_session_id = self.db.create_session("Text Conversation")

        if self.llm:
            force_web = bool(hasattr(self, "web_btn") and self.web_btn.isChecked())
            if hasattr(self.llm, "set_force_web_next_turn"):
                self.llm.set_force_web_next_turn(force_web)
            if force_web:
                # Applies to upcoming query only.
                self.web_btn.setChecked(False)
            self.llm.generate_response(text, self.active_session_id)

    def update_stt_latency(self, ms: float) -> None:
        self.stt_latency_lbl.setText(f"STT: {ms:.0f} ms")

    def update_ttft_latency(self, ms: float) -> None:
        self.ttft_latency_lbl.setText(f"TTFT: {ms:.0f} ms")

    def update_tts_latency(self, ms: float) -> None:
        self.tts_latency_lbl.setText(f"TTS: {ms:.0f} ms")

    def _on_history_search_changed(self, _text: str) -> None:
        self._history_search_timer.stop()
        self._history_search_timer.start(280)

    def _reload_history_sidebar(self) -> None:
        """Rebuild sidebar: full-text search when the search box is non-empty, else paged recent list."""
        self.history_list.clear()
        self.history_offset = 0
        self.is_loading_history = False
        q = self.search_bar.text().strip() if getattr(self, "search_bar", None) else ""
        if q:
            try:
                sessions = self.db.get_sessions_for_sidebar_search(q, limit=200)
            except Exception as e:
                logger.exception("Sidebar history search failed: %s", e)
                sessions = []
            for session in sessions:
                self._append_history_session_row(session)
            self.history_offset = 10**9
            self.is_loading_history = False
        else:
            self._load_history_batch()
        self._update_row_colors()

    def _append_history_session_row(self, session: dict) -> None:
        from PyQt6.QtWidgets import QListWidgetItem, QWidget, QHBoxLayout, QLabel, QPushButton, QMenu
        from PyQt6.QtCore import QSize
        import qtawesome as qta

        is_dark = True
        main_win = self.window()
        if main_win and hasattr(main_win, "_is_dark_theme"):
            is_dark = main_win._is_dark_theme

        icon_color = "#6c7086" if is_dark else "#64748b"

        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, session["id"])

        row_widget = QWidget()
        row_widget.setObjectName("HistoryRowWidget")
        row_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(10, 0, 10, 0)
        row_layout.setSpacing(10)

        title_lbl = QLabel(session["title"])
        title_lbl.setObjectName("HistoryRowTitle")

        opts_btn = QPushButton()
        opts_btn.setObjectName("HistoryOptionsBtn")
        opts_btn.setFixedSize(28, 28)
        opts_btn.setIcon(qta.icon("fa5s.ellipsis-v", color=icon_color))
        opts_btn.setIconSize(QSize(16, 16))
        opts_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        opts_btn.setStyleSheet(
            "QPushButton::menu-indicator { image: none; width: 0px; } "
            "QPushButton { border: none; background: transparent; padding: 0px; }"
        )

        menu = QMenu(opts_btn)
        if hasattr(self, "_apply_menu_theme"):
            self._apply_menu_theme(menu, is_dark)

        rename_action = menu.addAction(
            qta.icon("fa5s.edit", color="#89b4fa"), "Rename Chat"
        )
        rename_action.triggered.connect(
            lambda _, s_id=session["id"], old_t=session["title"]: self._trigger_rename_chat(
                s_id, old_t
            )
        )
        menu.addSeparator()

        delete_action = menu.addAction(
            qta.icon("fa5s.trash-alt", color="#ef4444"), "Delete Chat"
        )
        delete_action.triggered.connect(
            lambda _, s_id=session["id"]: self._trigger_delete_chat(s_id)
        )
        opts_btn.setMenu(menu)

        row_layout.addWidget(title_lbl)
        row_layout.addStretch()
        row_layout.addWidget(opts_btn)

        item.setSizeHint(QSize(0, 45))
        self.history_list.addItem(item)
        self.history_list.setItemWidget(item, row_widget)

    def _refresh_history_list(self):
        """Resets offset, runs cleanup, updates count, rebuilds list (respects search box)."""
        self.history_offset = 0
        self.is_loading_history = False

        # Silent garbage collection for empty sessions
        current_active = getattr(self, "active_session_id", None)
        if hasattr(self.db, "cleanup_empty_sessions"):
            self.db.cleanup_empty_sessions(current_active)

        # Session total (sidebar title stays "CONVERSATIONS"; count reserved for telemetry)
        self._session_count = self.db.get_session_count()

        self._reload_history_sidebar()

    def _on_history_scroll(self, value):
        """Triggered every time the user scrolls."""
        bar = self.history_list.verticalScrollBar()
        # If the scrollbar hits the absolute maximum, load the next batch!
        if value == bar.maximum():
            self._load_history_batch()

    def _load_history_batch(self):
        """Fetches the next chunk of history and appends it to the list."""
        if getattr(self, "search_bar", None) and self.search_bar.text().strip():
            self.is_loading_history = False
            return
        if getattr(self, "is_loading_history", False):
            return

        self.is_loading_history = True
        
        # 🔑 THE FIX: Pass both the limit AND the current offset to the DB
        # Note: If your DB doesn't support 'offset' yet, we'll need to update it!
        try:
            sessions = self.db.get_recent_sessions(limit=20, offset=self.history_offset)
        except TypeError:
            # Fallback just in case you haven't updated the DB manager yet
            sessions = self.db.get_recent_sessions(limit=20)
            if self.history_offset > 0:
                sessions = [] # Prevent infinite looping of the same 20 items

        if not sessions:
            self.is_loading_history = False
            return

        for session in sessions:
            self._append_history_session_row(session)

        self.history_offset += 20
        self.is_loading_history = False

    def _update_row_colors(self):
        """Row title colors: QSS cannot target setItemWidget children via ::item; apply explicitly."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        target_list = getattr(self, "doc_list", getattr(self, "history_list", None))
        apply_sidebar_row_title_colors(target_list, is_dark=is_dark)

    def _trigger_delete_chat(self, session_id):
        """Modern confirmation with full original safety logic."""
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        
        # 1. Use the Prestige UI instead of QMessageBox
        dlg = PrestigeDialog(
            self, 
            "Delete Conversation", 
            "Are you sure you want to permanently delete this chat? This cannot be undone.", 
            is_dark
        )
        
        if dlg.exec():
            # 2. Keep your original Database Guardrail
            if hasattr(self.db, 'delete_session'):
                self.db.delete_session(session_id)
            else:
                logger.error(f"CRITICAL: DB Manager missing 'delete_session' method. Cannot remove {session_id}.")
                return

            # 3. Keep your original UI State Management
            if getattr(self, 'active_session_id', None) == session_id:
                # If they deleted the active chat, reset the view
                self._start_new_chat()
            else:
                # Otherwise, just update the sidebar
                self._refresh_history_list()

    def _trigger_rename_chat(self, session_id, old_title):
        """Modern input with full original validation logic."""
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        
        # 1. Use Prestige UI instead of QInputDialog
        dlg = PrestigeDialog(
            self, 
            "Rename Conversation", 
            "Enter a new title for this chat:", 
            is_dark, 
            is_input=True, 
            default_text=old_title
        )
        
        # 2. Keep your 'ok' and 'strip' validation
        if dlg.exec() and dlg.result_text and dlg.result_text.strip():
            new_title = dlg.result_text.strip()
            
            # 3. Keep your original Database Guardrail
            if hasattr(self.db, 'rename_session'):
                self.db.rename_session(session_id, new_title)
                self._refresh_history_list()
            else:
                logger.error("CRITICAL: DB Manager missing 'rename_session' method.")

    def _start_new_chat(self):
        self.active_session_id = self.db.create_session("New Conversation")
        self._notify_llm_active_session_changed()
        self._clear_transcript()

        self.placeholder_lbl = QLabel("New chat started. Type or speak a message after saying your wake word!")
        self.placeholder_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.transcript_layout.addWidget(self.placeholder_lbl)
        self._refresh_ancillary_transcript_labels()

        self._is_agent_typing = False
        self._refresh_history_list()
        self._scroll_to_bottom()

    def _load_selected_chat(self, item):
        from PyQt6.QtCore import Qt
        session_id = item.data(Qt.ItemDataRole.UserRole)
        self.active_session_id = session_id
        self._notify_llm_active_session_changed()

        self._clear_transcript()
        self._is_agent_typing = False

        history = self.db.get_session_history(session_id)
        if not history:
            self.placeholder_lbl = QLabel("Empty conversation.")
            self.placeholder_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.transcript_layout.addWidget(self.placeholder_lbl)
            self._refresh_ancillary_transcript_labels()
            self._scroll_to_bottom()
            return

        for msg in history:
            if msg["role"] == "user":
                self.log_user_message(msg["content"])
            elif msg["role"] == "assistant":
                self.log_agent_token(
                    msg["content"],
                    citation_sources=msg.get("sources"),
                )
                self._is_agent_typing = False

        self._refresh_all_readability()
        self._scroll_to_bottom()

    def _apply_menu_theme(self, menu, is_dark: bool):
        """Standardizes the menu appearance to match the Prestige theme."""
        from PyQt6.QtGui import QPalette, QColor
        # THIS IS THE MAGIC LINE TO KILL THE GHOST SQUARE
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        palette = QPalette()
        if is_dark:
            bg, fg, sel_bg, sel_fg = "#1e1e2e", "#cdd6f4", "#313244", "#cdd6f4"
            border, hover = "rgba(255, 255, 255, 0.1)", "#313244"
        else:
            bg, fg, sel_bg, sel_fg = "#ffffff", "#1e293b", "#f1f5f9", "#0f172a"
            border, hover = "#cbd5e1", "#f1f5f9"

        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
            palette.setColor(role, QColor(bg))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(fg))
        palette.setColor(QPalette.ColorRole.Text, QColor(fg))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(sel_bg))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(sel_fg))

        menu.setPalette(palette)
        menu.setStyleSheet(f"""
            QMenu {{ 
                background-color: {bg}; 
                color: {fg}; 
                border: 1px solid {border}; 
                border-radius: 12px; 
                padding: 5px; 
            }}
            QMenu::item {{ 
                background-color: transparent; 
                padding: 8px 25px; 
                border-radius: 8px; 
            }}
            QMenu::item:selected {{ 
                background-color: {hover}; 
                color: {sel_fg}; 
            }}
        """)

    def refresh_menu_themes(self, is_dark: bool):
        """Updates all existing kebab menus in the history list."""
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            widget = self.history_list.itemWidget(item)
            if widget:
                # Find the button and its menu
                btn = widget.findChild(QPushButton, "HistoryOptionsBtn")
                if btn and btn.menu():
                    self._apply_menu_theme(btn.menu(), is_dark)

    def _on_think_toggled(self, checked: bool) -> None:
        """Persist user preference; Think state re-syncs from ExecutionPolicy via refresh_think_toggle."""
        if not hasattr(self, "think_btn") or not self.think_btn.isEnabled():
            return
        set_native_reasoning_display_enabled(bool(checked))
        self.refresh_think_toggle()

    def refresh_think_toggle(self) -> None:
        """Sync Think button from native engine telemetry (ExecutionPolicy projection only)."""
        if not hasattr(self, "think_btn"):
            return
        eng = self.workers.get("native_engine") if self.workers else None
        mode = get_engine_mode()
        snap = eng.get_model_reasoning_telemetry() if eng else None
        capable = (
            mode == "internal"
            and snap is not None
            and snap.get("loaded")
            and bool(snap.get("supports_thinking_tokens"))
        )
        eff_on = bool((snap or {}).get("ui_display_thinking", False))
        self.think_btn.blockSignals(True)
        try:
            self.think_btn.setVisible(capable)
            self.think_btn.setEnabled(capable)
            if capable:
                self.think_btn.setChecked(bool(eff_on))
            else:
                self.think_btn.setChecked(False)
        finally:
            self.think_btn.blockSignals(False)
        self._apply_action_toggle_styles()

    def _apply_action_toggle_styles(self) -> None:
        """Render Web/Think toggle buttons with active/inactive styles."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_toggle_button_style(
            self.web_btn if hasattr(self, "web_btn") else None,
            is_dark=is_dark,
            active_bg=("#89b4fa" if is_dark else "#1d4ed8"),
        )
        self._apply_toggle_button_style(
            self.think_btn if hasattr(self, "think_btn") else None,
            is_dark=is_dark,
            active_bg=("#a6e3a1" if is_dark else "#40a02b"),
        )

    def _apply_toggle_button_style(self, btn, *, is_dark: bool, active_bg: str) -> None:
        if btn is None:
            return
        if btn.isChecked():
            fg = "#11111b" if is_dark else "#ffffff"
            bg = active_bg
            border = bg
        else:
            fg = "#cdd6f4" if is_dark else "#1e293b"
            bg = "rgba(255, 255, 255, 0.05)" if is_dark else "rgba(0, 0, 0, 0.04)"
            border = "#6c7086" if is_dark else "#cbd5e1"
        hover = "rgba(255, 255, 255, 0.1)" if is_dark else "rgba(0, 0, 0, 0.08)"
        btn.setStyleSheet(
            f"""
            QPushButton {{
                color: {fg};
                background: {bg};
                border: 1px solid {border};
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
            """
        )

    def refresh_button_themes(self, is_dark: bool):
        """Dynamically updates the colors of the New Chat and Send buttons."""
        import qtawesome as qta
        
        # Base icon color: Catppuccin Purple in Dark Mode, Deep Slate in Light Mode
        base_icon_color = "#8b5cf6" if is_dark else "#1e293b"
        
        # Subtle hover background: faint white wash for Dark, faint black wash for Light
        hover_bg = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.05)"
        
        # 1. Update New Chat Button (+)
        if hasattr(self, 'new_chat_btn'):
            self.new_chat_btn.setIcon(qta.icon('fa5s.plus', color=base_icon_color))
            self.new_chat_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; border-radius: 6px; padding: 6px; }}
                QPushButton:hover {{ background-color: {hover_bg}; }}
            """)
            
        # 2. Update Send / Stop button icon + style
        if hasattr(self, 'send_btn'):
            icon_name = 'fa5s.stop' if self._is_stop_mode() else 'fa5s.paper-plane'
            send_icon_color = "#f38ba8" if self._is_stop_mode() and is_dark else base_icon_color
            if self._is_stop_mode() and not is_dark:
                send_icon_color = "#dc2626"
            self.send_btn.setIcon(qta.icon(icon_name, color=send_icon_color))
            self.send_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; border-radius: 6px; padding: 6px; }}
                QPushButton:hover {{ background-color: {hover_bg}; }}
            """)
        self._refresh_readability_toolbar(is_dark=is_dark)
        self._refresh_layout_mode_button(is_dark=is_dark)
        if hasattr(self, "font_minus_btn"):
            font_btn_style = readability_font_pair_stylesheet(
                is_dark=is_dark, button_px=_CHAT_UTILITY_BTN
            )
            self.font_minus_btn.setStyleSheet(font_btn_style)
            self.font_plus_btn.setStyleSheet(font_btn_style)
        self._apply_action_toggle_styles()
        self._apply_history_list_surface(is_dark)

    def _apply_history_list_surface(self, is_dark: bool) -> None:
        """Sidebar list tint: QListWidget paints in an internal viewport — set palette on list + viewport."""
        bg = QColor("#232337" if is_dark else "#E9EFF5")
        if hasattr(self, "history_pane"):
            p = self.history_pane
            p.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            p.setAutoFillBackground(True)
            pa = p.palette()
            pa.setColor(QPalette.ColorRole.Window, bg)
            p.setPalette(pa)
        if not hasattr(self, "history_list"):
            return
        w = self.history_list
        w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        w.setAutoFillBackground(True)
        pal = w.palette()
        pal.setColor(QPalette.ColorRole.Window, bg)
        w.setPalette(pal)
        vp = w.viewport()
        if vp is not None:
            vp.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            vp.setAutoFillBackground(True)
            vpal = vp.palette()
            vpal.setColor(QPalette.ColorRole.Window, bg)
            vpal.setColor(QPalette.ColorRole.Base, bg)
            vp.setPalette(vpal)

    def showEvent(self, event: QEvent) -> None:
        """Re-sync Think toggle when returning to Conversations (e.g. model loaded on another screen)."""
        super().showEvent(event)
        self.refresh_think_toggle()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_history_list_surface(is_dark)
    
    def eventFilter(self, obj, event):
        """Native resize handling without fighting Qt's geometry engine."""
        if self._focus_mode_enabled and isinstance(obj, MessageWrapper):
            et = event.type()
            if et == QEvent.Type.HoverEnter:
                self._reader_hover_wrapper = obj
                self._apply_reader_focus_opacity()
            elif et == QEvent.Type.HoverLeave:
                if self._reader_hover_wrapper is obj:
                    self._reader_hover_wrapper = None
                    self._apply_reader_focus_opacity()
        return super().eventFilter(obj, event)

    def _is_transcript_scrolled_to_bottom(self) -> bool:
        """True if the chat viewport is at (or within tolerance of) the bottom — user is 'following' new text."""
        if not hasattr(self, "scroll_area"):
            return True
        bar = self.scroll_area.verticalScrollBar()
        mx = bar.maximum()
        if mx <= 0:
            return True
        return (mx - bar.value()) <= _STICKY_SCROLL_TOLERANCE_PX

    def _scroll_to_bottom_after_resize(self, was_at_bottom: bool) -> None:
        if was_at_bottom:
            self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """Deferred scroll to the absolute bottom (new message, new chat, or sticky stream follow)."""
        def _execute_scroll():
            bar = self.scroll_area.verticalScrollBar()
            bar.setValue(bar.maximum())
            
        from PyQt6.QtCore import QTimer
        # Wait for geometry calculation, THEN wait for layout application
        QTimer.singleShot(0, lambda: QTimer.singleShot(0, _execute_scroll))

    def _attach_pending_citation_sources(self, label: AgentMessageLabel) -> None:
        """Apply sources from the tool phase to this bubble (or [] if none pending)."""
        pending = getattr(self, "_pending_citation_sources", None)
        if pending is not None:
            label._citation_sources = _snapshot_citation_sources(pending)
            self._pending_citation_sources = None
        else:
            label._citation_sources = []

    def on_sources_found(self, sources):
        """Receive tool sources for inline citation links (no separate chip UI)."""
        self._pending_citation_sources = _snapshot_citation_sources(sources)
        cur = getattr(self, "current_agent_msg", None)
        if cur is not None and getattr(cur, "_assistant_turn_id", None) == getattr(
            self, "_user_turn_id", -1
        ):
            cur._citation_sources = _snapshot_citation_sources(sources)

    def _resolve_citation_link_for_label(self, label: AgentMessageLabel, link_text: str):
        """Resolve href from this bubble's isolated _citation_sources (label-bound, not sender())."""
        raw = unquote((link_text or "").strip())
        sources = getattr(label, "_citation_sources", None) or []

        def _resolve_and_open(source_id: str) -> bool:
            wanted = _normalize_citation_id((source_id or "").strip())
            if not wanted:
                return True
            for src in sources:
                if wanted in _source_citation_match_keys(src):
                    self.open_source_preview(src)
                    return True
            ids_debug = [
                sorted(_source_citation_match_keys(s)) if isinstance(s, dict) else repr(type(s))
                for s in sources[:5]
            ]
            logger.warning(
                "Citation id %r (normalized %r) not found on this message (%d sources); sample ids %s",
                source_id,
                wanted,
                len(sources),
                ids_debug,
            )
            return True

        if raw.startswith(CITATION_HREF_PREFIX):
            tail = raw[len(CITATION_HREF_PREFIX) :].split("?")[0].split("#")[0].rstrip("/").strip()
            _resolve_and_open(tail)
            return

        if raw.startswith("qube://cite/"):
            tid = raw[len("qube://cite/") :].split("?")[0].split("#")[0].strip()
            _resolve_and_open(tid)
            return

        if raw.startswith("source_"):
            _resolve_and_open(raw.replace("source_", "", 1))
            return

        if raw.startswith("http://") or raw.startswith("https://"):
            import webbrowser

            webbrowser.open(raw)
            return

    def open_source_preview(self, source_dict):
        """Opens the Prestige-styled SourcePreviewer (see ui/components/prestige_dialog.py)."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        viewer = SourcePreviewer(
            source_dict.get("filename", "Source"),
            source_dict.get("content", ""),
            self,
            is_dark=is_dark,
        )
        viewer.show()

    def clear_stale_agent_pointer(self) -> None:
        """Drop the live assistant label handle (e.g. interrupt) without advancing _user_turn_id."""
        self.current_agent_msg = None

    def set_input_enabled(self, enabled: bool):
        """Locks the text input bar and resets its placeholder."""
        if hasattr(self, 'text_input') and hasattr(self, 'send_btn'):
            self.text_input.setEnabled(enabled)
            # Keep stop clickable while text input is disabled.
            self.send_btn.setEnabled(True)
            
            if enabled:
                # 🔑 RESET: Back to normal
                self.text_input.setPlaceholderText("Type a message to Qube...")
                self.text_input.setFocus()
            else:
                # 🔑 DEFAULT LOCK: We will update this dynamically in a millisecond
                self.text_input.setPlaceholderText("Qube is thinking...")
        self._refresh_send_stop_button()

    # 🔑 NEW: A dynamic receiver to update the text box live
    def update_action_placeholder(self, status: str):
        """Dynamically updates the text box placeholder based on worker status."""
        if not self.text_input.isEnabled() and status != "Idle":
            # Just clean up the string to sound natural (add an ellipsis if missing)
            display_text = status if "..." in status else f"{status}..."
            self.text_input.setPlaceholderText(display_text)

    def set_stop_requested_callback(self, callback) -> None:
        self._stop_requested_callback = callback

    def on_llm_response_finished(self) -> None:
        self._llm_in_progress = False
        tts_enabled = bool(self.tts and not getattr(self.tts, "is_muted", False))
        self._awaiting_tts_end = tts_enabled
        logger.info(
            "[ChatUI] LLM finished; stop_mode transitions to await_tts=%s.",
            tts_enabled,
        )
        if not tts_enabled:
            self.set_input_enabled(True)
        self._refresh_send_stop_button()

    def on_tts_playback_started(self, _session_id: str = "") -> None:
        self._tts_playing = True
        self._awaiting_tts_end = True
        logger.info("[ChatUI] TTS playback started; keep Stop button active.")
        self._refresh_send_stop_button()

    def on_tts_playback_finished(self) -> None:
        self._tts_playing = False
        self._awaiting_tts_end = False
        logger.info("[ChatUI] TTS playback finished; restoring send mode if LLM is idle.")
        if not self._llm_in_progress:
            self.set_input_enabled(True)
        self._refresh_send_stop_button()

    def on_generation_stopped(self) -> None:
        logger.info("[ChatUI] Stop acknowledged; clearing active generation/audio state.")
        self._llm_in_progress = False
        self._awaiting_tts_end = False
        self._tts_playing = False
        self.set_input_enabled(True)
        self.clear_stale_agent_pointer()
        self._refresh_send_stop_button()

    def _is_stop_mode(self) -> bool:
        return self._llm_in_progress or self._awaiting_tts_end or self._tts_playing

    def _refresh_send_stop_button(self) -> None:
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        self.refresh_button_themes(is_dark)

    def _handle_send_or_stop(self) -> None:
        if self._is_stop_mode():
            self._request_stop()
            return
        self._handle_text_submit()

    def _request_stop(self) -> None:
        logger.info("[ChatUI] Stop button pressed by user.")
        cb = self._stop_requested_callback
        if callable(cb):
            cb()
        else:
            if self.llm and self.llm.isRunning():
                self.llm.cancel_generation()
            if self.tts and self.tts.isRunning():
                self.tts.stop_playback()
            self.on_generation_stopped()