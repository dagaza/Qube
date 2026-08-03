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
    QPlainTextEdit,
    QPushButton,
    QListWidget,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QTextBrowser,
    QMenu,
    QGraphicsOpacityEffect,
    QFileDialog,
)
from PyQt6.QtGui import (
    QAction,
    QTextDocument,
    QTextOption,
    QTextBlockFormat,
    QTextCursor,
    QIcon,
    QColor,
    QPalette,
    QPixmap,
    QPainter,
    QFont,
    QKeyEvent,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtCore import (
    Qt,
    QTimer,
    QPropertyAnimation,
    QEasingCurve,
    QEvent,
    QCoreApplication,
    QUrl,
    QSize,
    pyqtSignal,
)
import json
import math
import uuid
import qtawesome as qta
import logging
from urllib.parse import unquote
import copy
import unicodedata
import weakref
import re
from dataclasses import dataclass
import re as _re_cite
from pathlib import Path

from core.citation_normalize import (
    markdown_for_external_clipboard,
    normalize_labeled_citation_tokens,
)
from core.citation_integrity import (
    CITATION_TOKEN_RE,
    analyze_citations,
    normalize_citation_id as _normalize_citation_id,
    source_citation_match_keys as _source_citation_match_keys,
    valid_source_ids,
)
from core.citation_integrity_telemetry import log_citation_integrity
from core.app_settings import get_citation_integrity_ui_linkify
from core.richtext_styles import markdown_document_stylesheet as _markdown_ui_stylesheet
from core.theme.svg_icons import tinted_svg_icon, themed_fa_icon, themed_fa_pixmap
from core.theme.view_theme import view_resolved_theme
from ui.components.ghost_icon_button import apply_ghost_icon_button_style
from ui.shell_theme import accent_icon_color
from core.surface_fill.constants import SURFACE_CHAT_TRANSCRIPT
from ui.surface_fill.transcript_host import (
    TranscriptWallpaperHost,
    bind_transcript_wallpaper_readability,
)
from core.theme.widget_styles import (
    ACCENT_ICON,
    ACCENT_ICON_ACTIVE,
    AGENT_COPY_BUTTON,
    AGENT_MESSAGE_FRAME,
    AGENT_MESSAGE_SHELL,
    COMPOSER_SIDE_BUTTON,
    COMPOSER_SIDE_DIVIDER,
    DANGER_ICON,
    GHOST_ICON_BUTTON,
    HELP_ACTION_CHIP,
    HIGH_CONTRAST_MARKDOWN,
    LIST_SURFACE,
    LINK_ICON,
    MUTED_ICON,
    PLACEHOLDER_MUTED,
    QUBE_RESPONSE_HEADER,
    SETTINGS_LINE_EDIT,
    TELEMETRY_LABEL,
    TOGGLE_BUTTON,
    USER_BUBBLE_FRAME,
    USER_BUBBLE_LABEL,
    UTILITY_ICON_BUTTON,
)
from core.composer_discoverability import (
    COMPOSER_IDLE_PLACEHOLDER,
    EMPTY_SESSION_TRANSCRIPT_HINT,
    NEW_CHAT_TRANSCRIPT_HINT,
    RecentMention,
    composer_hint_entries,
    record_recent_attachment,
    record_recent_skill,
    resolve_recent_mention,
)
from core.composer_attachments import resolve_attachment_routing, validate_file_token
from core.composer_draft import (
    ComposerDraft,
    ROUTING_REJECT_ONE_SOURCE,
    add_routing_attachment,
    add_skill,
    composer_one_source_limit_request,
    composer_capability_unavailable_request,
    composer_prompt_required_request,
    deep_research_unavailable_request,
    deep_research_pro_downgrade_request,
    draft_from_text,
    merge_drafts,
    remove_routing_at,
    remove_skill_at,
    serialize_draft,
)
from core.composer_skills import ComposerSkillMention, parse_composer_input, strip_all_composer_tokens_for_display
from core.help_action_blocks import HelpActionChip, parse_help_action_blocks
from core.conversation_export import (
    export_conversation_markdown,
    export_folder_zip,
    format_conversation_markdown,
    sanitize_export_filename,
)
from core.assistant_message_export import (
    format_assistant_message_for_export,
    has_exportable_assistant_content,
    suggested_assistant_export_stem,
    write_assistant_message_markdown,
)
from ui.export.research_report_pdf import write_markdown_pdf
from core.composer_commands import execute_composer_command
from core.composer_mention_search import ComposerPaletteView
from core.composer_mention_trigger import (
    escape_strip_index,
    is_valid_mention_anchor,
    mention_query_suffix,
    resolve_mention_release,
)
from core.app_settings import (
    get_composer_at_mention_discovered,
    get_engine_mode,
    get_ui_assistant_message_background,
    set_composer_at_mention_discovered,
    set_native_reasoning_display_enabled,
)
from core.knowledge.deep_research_ui import deep_research_progress_percent
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE
from ui.sidebar_dimensions import LEFT_NAV_LIST_SIDEBAR_WIDTH
from ui.components.page_tour_help_button import PageTourHelpButton
from ui.components.prestige_menu_qss import apply_prestige_kebab_menu_theme
from ui.components.prestige_dialog import PrestigeDialog, CitationSourcesDialog
from ui.components.research_map_dialog import ResearchMapDialog
from ui.components.readability_toolbar_styles import readability_font_pair_stylesheet
from ui.components.sidebar_list_qss import apply_sidebar_row_theme
from ui.shell_theme import sidebar_row_action_icon_color
from ui.components.sidebar_folder_list import (
    FOLDER_ROW_MARGIN_LEFT,
    ROW_KIND_SESSION,
    SIDEBAR_ROW_KIND_ROLE,
    SIDEBAR_ROW_PAYLOAD_ROLE,
    SidebarFolderListController,
    add_new_folder_header_button,
    create_sidebar_header_actions_row,
)
from ui.components.source_viewer import SourcePreviewer
from ui.components.text_document_height import (
    font_descender_inset,
    measure_markdown_body_height,
    measure_wrapped_body_height,
    text_edit_chrome_vertical_px,
)
from ui.components.stream_markdown_split import (
    compose_streaming_markdown,
    normalize_inline_markdown_structure,
    split_stream_markdown_buffer,
)
from ui.components.composer_mention_popup import ComposerMentionPopup
from ui.components.composer_context_chips import ComposerContextChipStrip
from ui.components.composer_recent_mentions import ComposerRecentMentionsRow
from ui.components.hidden_feature_discovery import present_composer_at_mention_discovery
from ui.components.ingest_progress_row import IngestProgressRow
from ui.components.typing_indicator import TypingIndicatorWidget, TypingIndicatorMode
from ui.components.transcript_timeline_rail import (
    TRANSCRIPT_TIMELINE_RAIL_WIDTH_PX,
    TranscriptTimelineRail,
    TranscriptWaypointEntry,
    compute_active_waypoint_index,
    compute_scroll_target_for_waypoint_y,
    transcript_timeline_should_show,
    truncate_waypoint_label,
)

logger = logging.getLogger("Qube.UI.Conversations")

# In-text citation links must survive Qt's Markdown → anchor step. Qt's importer is flaky for
# custom schemes with numeric path segments (qube://cite/1); https + .invalid is linkified reliably.
CITATION_HREF_PREFIX = "https://qube.invalid/cite/"

# log_agent_token(..., citation_sources=...) — distinguish "not passed" (live stream) from explicit None (DB row with no sources)
_UNSET_SOURCES = object()

# --------------- Chat layout modes --------------- #
LAYOUT_FULL_WIDTH = "full_width"
LAYOUT_CENTERED_COLUMN = "centered_column"
_TRANSCRIPT_TIMELINE_SCROLL_MS = 220
_CENTERED_COLUMN_MAX_WIDTH = 800
_FULL_WIDTH_COLUMN_MAX_WIDTH = 1200
_QWIDGETSIZE_MAX = (1 << 24) - 1


def _user_bubble_max_width_for_wrapper(
    wrapper_width: int, *, transcript_column_max: int
) -> int:
    """Cap user bubble width: min(fraction of row, transcript column minus gutter)."""
    if wrapper_width <= 0:
        return 160
    return max(
        160,
        int(
            min(
                float(wrapper_width) * 0.88,
                float(transcript_column_max) - 24.0,
            )
        ),
    )
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
_AGENT_MESSAGE_CARD_MARGINS = (14, 10, 14, 8)
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
    s = normalize_labeled_citation_tokens(s)
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


def _markdown_cite_link_replacement(match, *, valid_ids: set[str] | None = None) -> str:
    token = match.group(1)
    key = "W" if str(token).lower() == "w" else str(token)
    if get_citation_integrity_ui_linkify() and valid_ids is not None:
        if _normalize_citation_id(key) not in valid_ids:
            return "[W]" if key == "W" else f"[{key}]"
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


class ChatUserBubble(QPlainTextEdit):
    """Read-only user message body: same wrap rules as the composer (long tokens break mid-word)."""

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.setObjectName("ChatUserBubble")
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setTabChangesFocus(False)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.document().setDocumentMargin(2)
        self.viewport().setAutoFillBackground(False)
        opt = QTextOption()
        opt.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        opt.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.document().setDefaultTextOption(opt)
        self._height_timer = QTimer(self)
        self._height_timer.setSingleShot(True)
        self._height_timer.timeout.connect(self._sync_document_height)
        self.document().contentsChanged.connect(self._schedule_height_sync)
        self.setPlainText(text or "")
        self._schedule_height_sync()

    def cleanup_before_destruction(self) -> None:
        try:
            self.clear()
        except RuntimeError:
            pass

    def _schedule_height_sync(self) -> None:
        self._height_timer.start(0)

    def _effective_wrap_width(self) -> int:
        vw = int(self.viewport().width())
        if vw > 4:
            return vw
        p = self.parentWidget()
        if isinstance(p, QWidget) and p.width() > 8:
            lay = p.layout()
            if isinstance(lay, QVBoxLayout):
                m = lay.contentsMargins()
                return max(40, p.width() - m.left() - m.right())
            return max(40, p.width() - 8)
        return 280

    def natural_body_width(self) -> int:
        """Unwrapped document width (longest line); used to shrink short user bubbles horizontally."""
        fm = self.fontMetrics()
        text = self.toPlainText()
        line_adv = 0
        for line in text.split("\n"):
            line_adv = max(line_adv, fm.horizontalAdvance(line))

        doc = self.document()
        old_tw = float(doc.textWidth())
        doc.setTextWidth(-1.0)
        dl = doc.documentLayout()
        if dl is not None and hasattr(dl, "invalidate"):
            try:
                dl.invalidate()
            except (RuntimeError, AttributeError):
                pass
        ideal = float(doc.idealWidth())
        restore = old_tw if old_tw > 0 else float(max(1, self.viewport().width()))
        doc.setTextWidth(restore)
        if dl is not None and hasattr(dl, "invalidate"):
            try:
                dl.invalidate()
            except (RuntimeError, AttributeError):
                pass
        dm = int(math.ceil(float(doc.documentMargin()) * 2.0))
        # QTextDocument.idealWidth() can be a few px under the painted run; prefer font metrics.
        core = max(float(line_adv), ideal)
        return max(40, int(math.ceil(core)) + dm + 8)

    def _block_stack_bottom_px(self) -> float:
        """QPlainTextEdit block geometry (can differ slightly from layout.blockBoundingRect)."""
        bottom = 0.0
        block = self.document().firstBlock()
        while block.isValid():
            geom = self.blockBoundingGeometry(block)
            if geom.isValid():
                bottom = max(bottom, float(geom.bottom()))
            block = block.next()
        return bottom

    def _compute_content_height(self, wrap_w: int) -> int:
        """Pixel height for plain text at wrap_w (viewport not always ready when layout runs)."""
        doc = self.document()
        fm = self.fontMetrics()
        body = measure_wrapped_body_height(
            doc,
            wrap_w,
            min_body_px=float(fm.lineSpacing()),
            block_bottom_px=self._block_stack_bottom_px(),
        )
        return int(math.ceil(body)) + font_descender_inset(fm)

    def minimumSizeHint(self) -> QSize:
        eff = max(40, self._effective_wrap_width())
        nat = self.natural_body_width()
        ww = min(eff, max(40, nat))
        h = self._compute_content_height(ww)
        return QSize(ww, min(h, 32000))

    def sizeHint(self) -> QSize:
        return self.minimumSizeHint()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_height_sync()

    def _sync_document_height(self) -> None:
        w = max(1, self._effective_wrap_width())
        self.document().setTextWidth(float(w))
        lay = self.document().documentLayout()
        if lay is not None and hasattr(lay, "invalidate"):
            try:
                lay.invalidate()
            except (RuntimeError, AttributeError):
                pass
        h = self._compute_content_height(w)
        if self.height() != h:
            self.setFixedHeight(h)
        self.updateGeometry()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setObjectName("PrestigeMenu")
        view = _parent_conversations_view(self)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        if view is not None:
            view._apply_menu_theme(menu, is_dark)

        def _copy():
            cur = self.textCursor()
            if cur.hasSelection():
                QApplication.clipboard().setText(cur.selectedText())
            elif self.toPlainText():
                QApplication.clipboard().setText(self.toPlainText())

        copy_act = QAction("Copy", self)
        copy_act.triggered.connect(_copy)
        copy_act.setEnabled(bool(self.toPlainText()))
        menu.addAction(copy_act)
        menu.exec(event.globalPos())


class UserBubbleFrame(QFrame):
    """Pins user bubble body width: up to the row cap, but shrinks to unwrapped text when shorter."""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        lbl = self.findChild(ChatUserBubble)
        if lbl is None:
            return
        lay = self.layout()
        ml = mr = 16
        if isinstance(lay, QVBoxLayout):
            m = lay.contentsMargins()
            ml, mr = m.left(), m.right()
        pw = self.parentWidget()
        cap_w = int(self.maximumWidth())
        view = _parent_conversations_view(lbl)
        col = (
            view.transcript_column_max_width()
            if view is not None
            else _CENTERED_COLUMN_MAX_WIDTH
        )
        if cap_w >= _QWIDGETSIZE_MAX - 4096:
            ww = (
                pw.width()
                if isinstance(pw, MessageWrapper) and pw.width() > 0
                else max(1, self.width())
            )
            cap_w = _user_bubble_max_width_for_wrapper(
                ww, transcript_column_max=col
            )
        inner_max = max(40, cap_w - ml - mr)
        natural = lbl.natural_body_width()
        body_w = min(inner_max, max(40, natural))
        frame_w = body_w + ml + mr
        if self.width() != frame_w:
            self.setFixedWidth(frame_w)
        if lbl.width() != body_w or lbl.maximumWidth() != body_w:
            lbl.setFixedWidth(body_w)
            lbl.setMaximumWidth(body_w)
            lbl._schedule_height_sync()
            lbl.updateGeometry()
            self.updateGeometry()


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
        # ClickFocus so mouse selection + Ctrl+C copies from this bubble, not the composer.
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.document().setDocumentMargin(2)
        self.viewport().setAutoFillBackground(False)

        self._citation_sources: list = []
        self._conversations_view_ref = None
        self._citation_anchor_connected = False
        self._md_layout_source = ""
        self._agent_is_dark = True
        self._doc_layout_connected = False
        self._fixed_h = 0
        self._syncing_height = False
        self._computing_doc_height = False
        self._streaming_markdown = False
        self._suppress_doc_size_sync = False
        self._stream_peak_h = 0
        self._height_coalesce = QTimer(self)
        self._height_coalesce.setSingleShot(True)
        self._height_coalesce.timeout.connect(self._sync_fixed_height)

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
        last_block = doc.lastBlock()
        while block.isValid():
            cur.setPosition(block.position())
            cur.mergeBlockFormat(fmt)
            if block == last_block:
                tail = QTextBlockFormat()
                tail.setBottomMargin(0.0)
                cur.mergeBlockFormat(tail)
            block = block.next()
        cur.endEditBlock()

    def _effective_content_width(self) -> int:
        """Wrap width for layout; fall back to transcript column when viewport is not ready."""
        vw = int(self.viewport().width())
        if vw > 4:
            return vw
        outer = int(self.width())
        if outer > 8:
            return max(1, self._content_width_from_outer_width(outer))
        view = _parent_conversations_view(self)
        col = (
            view.transcript_column_max_width()
            if view is not None
            else _CENTERED_COLUMN_MAX_WIDTH
        )
        return max(40, int(col) - 32)

    def _ensure_document_text_width(self) -> None:
        self.document().setTextWidth(float(max(1, self._effective_content_width())))

    def _reset_document_viewport_top(self) -> None:
        """QTextBrowser auto-scrolls to the caret on append; pin document origin at y=0."""
        bar = self.verticalScrollBar()
        if bar is not None:
            bar.setValue(0)
        cur = self.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.Start)
        self.setTextCursor(cur)

    def _apply_agent_document_content(
        self,
        doc,
        safe_markdown: str,
        *,
        line_height_percent: int | None,
        justify_transcript: bool,
    ) -> None:
        doc.setMarkdown(safe_markdown)
        pct = (
            line_height_percent
            if line_height_percent is not None
            else int(round(float(_LINE_HEIGHT_CSS[_LINE_HEIGHT_COMFORTABLE]) * 100))
        )
        self._apply_document_paragraph_formats(doc, pct, justify_transcript)
        self._reset_document_viewport_top()

    def set_agent_markdown(
        self,
        markdown: str,
        *,
        is_dark: bool,
        document_stylesheet: str | None = None,
        line_height_percent: int | None = None,
        justify_transcript: bool = False,
        streaming: bool = False,
    ) -> None:
        self._agent_is_dark = is_dark
        self._streaming_markdown = streaming
        if streaming:
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Minimum,
            )
        else:
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            self._stream_peak_h = 0
        self._md_layout_source = markdown or ""
        safe = _qt_safe_markdown(self._md_layout_source)
        doc = self.document()
        doc.setDefaultFont(self.font())
        doc.setDefaultStyleSheet(
            document_stylesheet
            if document_stylesheet is not None
            else _markdown_ui_stylesheet(is_dark)
        )
        self._ensure_document_text_width()
        self._suppress_doc_size_sync = True
        try:
            self._apply_agent_document_content(
                doc,
                safe,
                line_height_percent=line_height_percent,
                justify_transcript=justify_transcript,
            )
        finally:
            self._suppress_doc_size_sync = False
        if not streaming:
            _maybe_dump_markdown_html_pipeline(self._md_layout_source, self.font(), is_dark)
        self._sync_fixed_height()
        if streaming:
            self._schedule_height_sync()
        self.updateGeometry()
        parent = self.parentWidget()
        if parent is not None:
            parent.updateGeometry()

    def refresh_theme_styles(
        self,
        *,
        is_dark: bool,
        document_stylesheet: str,
        line_height_percent: int | None = None,
        justify_transcript: bool = False,
    ) -> None:
        """Re-apply theme colors without re-parsing markdown (theme-toggle path)."""
        self._agent_is_dark = is_dark
        doc = self.document()
        doc.setDefaultFont(self.font())
        doc.setDefaultStyleSheet(document_stylesheet)
        pct = (
            line_height_percent
            if line_height_percent is not None
            else int(round(float(_LINE_HEIGHT_CSS[_LINE_HEIGHT_COMFORTABLE]) * 100))
        )
        self._apply_document_paragraph_formats(doc, pct, justify_transcript)
        doc.markContentsDirty(0, doc.characterCount())
        self._ensure_document_text_width()
        self._sync_fixed_height()
        self.update()
        parent = self.parentWidget()
        if parent is not None:
            parent.updateGeometry()

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
        self._stream_peak_h = 0
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
        if self._streaming_markdown and self._fixed_h > 0:
            return QSize(max(self.width(), 1), self._fixed_h)
        return super().sizeHint()

    def heightForWidth(self, w: int) -> int:
        if self._streaming_markdown:
            if self._fixed_h > 0:
                return self._fixed_h
            if w <= 0:
                return super().heightForWidth(w)
        if w <= 0:
            return super().heightForWidth(w)
        if self._computing_doc_height:
            if self._fixed_h > 0:
                return self._fixed_h
            return max(int(self.fontMetrics().lineSpacing()) + 8, 24)
        return self._compute_doc_height(w)

    def hasHeightForWidth(self) -> bool:
        return not self._streaming_markdown

    def _schedule_height_sync(self) -> None:
        self._height_coalesce.start(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._ensure_document_text_width()
        if self._streaming_markdown:
            self._reset_document_viewport_top()
        self._schedule_height_sync()
        self.updateGeometry()

    def _compute_doc_height(self, width: int) -> int:
        if self._computing_doc_height:
            if self._fixed_h > 0:
                return self._fixed_h
            return max(int(self.fontMetrics().lineSpacing()) + 8, 24)
        self._computing_doc_height = True
        try:
            doc = self.document()
            content_w = self._effective_content_width()
            if content_w <= 1 and width > 1:
                content_w = self._content_width_from_outer_width(width)
            fm = self.fontMetrics()
            body = measure_markdown_body_height(
                doc,
                content_w,
                min_body_px=float(fm.lineSpacing()),
                bottom_inset_px=font_descender_inset(
                    fm, safety_px=2 if self._streaming_markdown else 1
                ),
                streaming=self._streaming_markdown,
            )
            return int(math.ceil(body)) + self._viewport_chrome_vertical_px()
        finally:
            self._computing_doc_height = False

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

    def _viewport_chrome_vertical_px(self) -> int:
        """Widget chrome only — descender inset lives in the markdown body measurement."""
        cm = self.contentsMargins()
        vm = self.viewportMargins()
        return text_edit_chrome_vertical_px(
            frame_width=self.frameWidth(),
            contents_top=cm.top(),
            contents_bottom=cm.bottom(),
            viewport_top=vm.top(),
            viewport_bottom=vm.bottom(),
        )

    def _on_document_size_changed(self, _size) -> None:
        if self._suppress_doc_size_sync or self._streaming_markdown:
            return
        self._schedule_height_sync()

    def _sync_fixed_height(self) -> None:
        if self._syncing_height:
            return
        self._syncing_height = True
        try:
            w = max(self.width(), 1)
            h = self._compute_doc_height(w)
            if self._streaming_markdown:
                h = max(h, self._fixed_h, self._stream_peak_h)
                self._stream_peak_h = h
                self._fixed_h = h
                self.setMinimumHeight(h)
                self.setMaximumHeight(_QWIDGETSIZE_MAX)
                self._reset_document_viewport_top()
            else:
                self.setMinimumHeight(0)
                self.setMaximumHeight(_QWIDGETSIZE_MAX)
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

    def plain_text_for_clipboard(self) -> str:
        """Rendered plain text for one-click copy (falls back to markdown source)."""
        text = self.toPlainText().strip()
        if text:
            return text
        return markdown_for_external_clipboard(self._md_layout_source or "")

    def markdown_for_clipboard(self) -> str:
        """Original markdown syntax for export (Obsidian, etc.), without Qt cite links."""
        src = (self._md_layout_source or "").strip()
        if src:
            return markdown_for_external_clipboard(src)
        return self.plain_text_for_clipboard()


class AgentMessageContainer(QFrame):
    """Assistant turn body: markdown bubble + per-message actions share one width."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AgentMessageContainer")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._agent: AgentMessageLabel | None = None
        self._actions_bar: QWidget | None = None

    def attach_agent(self, agent: AgentMessageLabel) -> None:
        self._agent = agent
        self._layout.addWidget(agent, 0)
        agent.installEventFilter(self)

    def attach_actions_bar(self, actions_bar: QWidget) -> None:
        self._actions_bar = actions_bar
        actions_bar.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self._layout.addWidget(actions_bar, 0)
        self._sync_actions_bar_width()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._agent and event.type() == QEvent.Type.Resize:
            self._sync_actions_bar_width()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_actions_bar_width()

    def _sync_actions_bar_width(self) -> None:
        agent = self._agent
        bar = self._actions_bar
        if agent is None or bar is None:
            return
        w = max(1, agent.width())
        if bar.width() != w:
            bar.setFixedWidth(w)


@dataclass
class _TranscriptWaypointRecord:
    wrapper: "MessageWrapper"
    label: str


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
            layout.addWidget(bubble, 0)
        else:
            layout.addWidget(bubble, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.is_user and self.bubble is not None and self.width() > 0:
            view = _parent_conversations_view(self)
            col = (
                view.transcript_column_max_width()
                if view is not None
                else _CENTERED_COLUMN_MAX_WIDTH
            )
            cap = _user_bubble_max_width_for_wrapper(
                self.width(), transcript_column_max=col
            )
            self.bubble.setMaximumWidth(cap)

    def cleanup_before_destruction(self) -> None:
        """Break references held by this row before Qt tears down the widget tree."""
        for lbl in self.findChildren(ChatUserBubble):
            lbl.cleanup_before_destruction()
        for w in self.findChildren(AgentMessageLabel):
            w.cleanup_before_destruction()
        for ind in self.findChildren(TypingIndicatorWidget):
            ind.stop()
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
        layout.addWidget(inner, 0)
        layout.addStretch(1)

    def set_column_max_width(self, max_w: int) -> None:
        self._max_w = max(1, int(max_w))
        w = min(self._max_w, max(1, self.width()))
        if self._inner.width() != w:
            self._inner.setFixedWidth(w)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = min(self._max_w, max(1, self.width()))
        if self._inner.width() != w:
            self._inner.setFixedWidth(w)


class _TranscriptColumnHost(QWidget):
    """Centers the capped transcript column with the turn-index rail flush to its right."""

    _RAIL_GAP_PX = 6

    def __init__(
        self,
        scroll_area: QScrollArea,
        rail: TranscriptTimelineRail,
        *,
        nominal_cap_provider,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("ChatTranscriptColumnHost")
        self._scroll_area = scroll_area
        self._rail = rail
        self._nominal_cap_provider = nominal_cap_provider

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addStretch(1)

        self._block = QWidget()
        self._block.setObjectName("ChatTranscriptColumnBlock")
        block_layout = QHBoxLayout(self._block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(self._RAIL_GAP_PX)
        block_layout.addWidget(scroll_area, 1)
        self._rail.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )
        block_layout.addWidget(rail, 0)

        outer.addWidget(self._block, 0)
        outer.addStretch(1)

    def sync_geometry(self) -> None:
        nominal = int(self._nominal_cap_provider())
        bar = self._scroll_area.verticalScrollBar()
        scrollbar_w = 0
        if bar is not None:
            if bar.isVisible():
                scrollbar_w = int(bar.width())
            elif self._rail.isVisible():
                scrollbar_w = int(bar.sizeHint().width())

        rail_w = TRANSCRIPT_TIMELINE_RAIL_WIDTH_PX if self._rail.isVisible() else 0
        gap = self._RAIL_GAP_PX if rail_w else 0
        block_w = nominal + scrollbar_w + gap + rail_w
        available = max(1, int(self.width()))
        block_w = min(block_w, available)

        if self._block.width() != block_w:
            self._block.setFixedWidth(block_w)

        scroll_w = max(1, block_w - gap - rail_w)
        if self._scroll_area.width() != scroll_w:
            self._scroll_area.setFixedWidth(scroll_w)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.sync_geometry()


_MENTION_QUERY_RE = re.compile(r"@([^\s@\[]*)$")


class ChatComposerEdit(QPlainTextEdit):
    """Chat composer: Enter sends, Shift+Enter inserts a newline; @ opens mention picker."""

    submit_requested = pyqtSignal()
    _MAX_VISIBLE_LINES = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setWordWrapMode(
            QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
        )
        self.setTabChangesFocus(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.document().setDocumentMargin(4)
        self._height_coalesce = QTimer(self)
        self._height_coalesce.setSingleShot(True)
        self._height_coalesce.timeout.connect(self._sync_height_to_content)
        self.textChanged.connect(self._schedule_height_sync)
        self.textChanged.connect(self._on_text_changed_mention)
        self._schedule_height_sync()
        self._mention_host = None
        self._mention_popup: ComposerMentionPopup | None = None
        self._mention_start_pos = -1
        self._mention_session_active = False
        self._mention_armed = False
        self._mention_arm_at = -1
        self._mention_arm_count = 0
        self._mention_arm_modifiers = Qt.KeyboardModifier.NoModifier
        self._mention_arm_trigger_key = Qt.Key.Key_unknown

    def bind_mention_host(self, host) -> None:
        """Attach ConversationsView (or compatible) for db/session/store context."""
        self._mention_host = host
        if self._mention_popup is None:
            self._mention_popup = ComposerMentionPopup(self)
            self._mention_popup.item_selected.connect(self._insert_mention_token)
            self._mention_popup.skill_selected.connect(self._insert_skill_token)
            self._mention_popup.command_selected.connect(self._run_composer_command)
            self._mention_popup.dismissed.connect(self._on_mention_dismissed)
        self._sync_mention_context()
        win = self.window()
        is_dark = getattr(win, "_is_dark_theme", True) if win else True
        self._mention_popup.apply_theme(is_dark)

    def _sync_mention_context(self) -> None:
        if not self._mention_popup or not self._mention_host:
            return
        db = getattr(self._mention_host, "db", None)
        store = None
        llm = getattr(self._mention_host, "llm", None)
        if llm is not None:
            store = getattr(llm, "store", None)
        sid = getattr(self._mention_host, "active_session_id", None)
        self._mention_popup.set_context(db=db, store=store, active_session_id=sid)

    def apply_mention_theme(self, is_dark: bool) -> None:
        if self._mention_popup:
            self._mention_popup.apply_theme(is_dark)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        vw = max(1, int(self.viewport().width()))
        self.document().setTextWidth(float(vw))
        if event.oldSize().width() != event.size().width():
            self._schedule_height_sync()

    def _schedule_height_sync(self) -> None:
        # Document layout updates after the current event; measure on the next tick
        # so documentSize() reflects new lines and wrapping at the current width.
        self._height_coalesce.start(0)

    def _mention_global_pos(self) -> object:
        rect = self.cursorRect()
        return self.mapToGlobal(rect.bottomLeft())

    def _active_mention_query(self) -> tuple[int, str] | None:
        cursor = self.textCursor()
        pos = cursor.position()
        text = self.toPlainText()[:pos]
        match = _MENTION_QUERY_RE.search(text)
        if not match:
            return None
        start = match.start()
        if start > 0 and not text[start - 1].isspace():
            return None
        return start, match.group(1)

    def _remove_composer_range(self, start: int, end: int) -> None:
        cursor = self.textCursor()
        cursor.beginEditBlock()
        cursor.setPosition(max(0, start))
        cursor.setPosition(min(end, len(self.toPlainText())), QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.endEditBlock()
        self.setTextCursor(cursor)

    def _disarm_mention_trigger(self) -> None:
        self._mention_armed = False
        self._mention_arm_at = -1
        self._mention_arm_count = 0
        self._mention_arm_modifiers = Qt.KeyboardModifier.NoModifier
        self._mention_arm_trigger_key = Qt.Key.Key_unknown

    @staticmethod
    def _modifier_flag_for_key(key: int) -> Qt.KeyboardModifier | None:
        if key in (
            Qt.Key.Key_Shift,
            getattr(Qt.Key, "Key_ShiftRight", Qt.Key.Key_unknown),
        ):
            return Qt.KeyboardModifier.ShiftModifier
        if key in (
            Qt.Key.Key_Control,
            getattr(Qt.Key, "Key_ControlRight", Qt.Key.Key_unknown),
        ):
            return Qt.KeyboardModifier.ControlModifier
        if key in (
            Qt.Key.Key_Alt,
            getattr(Qt.Key, "Key_AltGr", Qt.Key.Key_unknown),
        ):
            return Qt.KeyboardModifier.AltModifier
        if key in (
            Qt.Key.Key_Meta,
            getattr(Qt.Key, "Key_MetaRight", Qt.Key.Key_unknown),
        ):
            return Qt.KeyboardModifier.MetaModifier
        return None

    def _should_complete_mention_arm(self, event: QKeyEvent) -> bool:
        if not self._mention_armed:
            return False
        key = event.key()
        if self._mention_arm_modifiers == Qt.KeyboardModifier.NoModifier:
            return key == self._mention_arm_trigger_key
        mod = self._modifier_flag_for_key(key)
        return bool(mod and (self._mention_arm_modifiers & mod))

    def _open_mention_menu(self, at_pos: int, query: str | None = None) -> None:
        popup = self._mention_popup
        if popup is None:
            return
        text = self.toPlainText()
        if query is None:
            query = mention_query_suffix(text, at_pos)
        if at_pos < 0 or at_pos >= len(text) or text[at_pos] != "@":
            return
        self._sync_mention_context()
        self._mention_start_pos = at_pos
        self._mention_session_active = True
        gpos = self._mention_global_pos()
        popup.show_root(gpos)
        popup.set_composer_query(query or "", global_pos=gpos)
        self._maybe_show_at_discovery(popup)
        self.setFocus()

    def _complete_mention_arm(self) -> None:
        if not self._mention_armed:
            return
        text = self.toPlainText()
        at_pos = self._mention_arm_at
        arm_count = self._mention_arm_count
        self._disarm_mention_trigger()
        action = resolve_mention_release(arm_count)
        if action == "invalid":
            return
        if action == "escape":
            strip_idx = escape_strip_index(text, at_pos)
            if strip_idx >= 0:
                self._remove_composer_range(strip_idx, strip_idx + 1)
            return
        query = mention_query_suffix(text, at_pos)
        self._open_mention_menu(at_pos, query)

    def _maybe_show_at_discovery(self, popup: ComposerMentionPopup) -> None:
        if get_composer_at_mention_discovered():
            return
        set_composer_at_mention_discovered(True)
        win = self.window()
        if win is None or popup is None:
            return
        QTimer.singleShot(
            80,
            lambda: present_composer_at_mention_discovery(
                win,
                popup,
                on_finished=lambda: self.setFocus(),
            ),
        )

    def _on_text_changed_mention(self) -> None:
        if not self._mention_popup:
            return
        popup = self._mention_popup
        active = self._active_mention_query()
        if active is None:
            if self._mention_session_active or popup.isVisible():
                self._mention_session_active = False
                popup.close_mention()
                self._mention_start_pos = -1
            return
        start, query = active
        self._mention_start_pos = start
        self._mention_session_active = True
        self._sync_mention_context()
        gpos = self._mention_global_pos()
        if not popup.isVisible():
            popup.show_root(gpos)
        popup.set_composer_query(query, global_pos=gpos)

    def _insert_mention_token(self, attachment) -> None:
        host = self._mention_host
        if host is not None and hasattr(host, "add_composer_attachment"):
            host.add_composer_attachment(attachment)
        self._clear_mention_trigger_after_pick()

    def _insert_skill_token(self, mention) -> None:
        host = self._mention_host
        if host is not None and hasattr(host, "add_composer_skill"):
            host.add_composer_skill(mention)
        self._clear_mention_trigger_after_pick()

    def _clear_mention_trigger_after_pick(self) -> None:
        cursor = self.textCursor()
        text = self.toPlainText()
        if self._mention_start_pos >= 0:
            start = self._mention_start_pos
            end = min(cursor.position(), len(text))
            while end < len(text) and text[end] not in (" ", "\n"):
                end += 1
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            self.setTextCursor(cursor)
        self._mention_session_active = False
        self._mention_start_pos = -1
        self._disarm_mention_trigger()
        if self._mention_popup:
            self._mention_popup.hide()
        self.setFocus()

    def open_mention_palette(self, global_pos=None) -> None:
        """Open the @ composer palette (keyboard @ or attach button)."""
        popup = self._mention_popup
        if popup is None:
            return
        active = self._active_mention_query()
        if active is None:
            cursor = self.textCursor()
            start = cursor.position()
            cursor.insertText("@")
            self.setTextCursor(cursor)
            query = ""
        else:
            start, query = active
        self._sync_mention_context()
        self._mention_start_pos = start
        self._mention_session_active = True
        gpos = global_pos or self._mention_global_pos()
        popup.show_root(gpos)
        popup.set_composer_query(query, global_pos=gpos)
        self._maybe_show_at_discovery(popup)
        self.setFocus()

    def _clear_mention_trigger(self) -> None:
        cursor = self.textCursor()
        text = self.toPlainText()
        if self._mention_start_pos >= 0:
            start = self._mention_start_pos
            end = min(cursor.position(), len(text))
            while end < len(text) and text[end] not in (" ", "\n"):
                end += 1
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            self.setTextCursor(cursor)
        self._mention_session_active = False
        self._mention_start_pos = -1
        self._disarm_mention_trigger()
        if self._mention_popup:
            self._mention_popup.hide()

    def _run_composer_command(self, command) -> None:
        self._clear_mention_trigger()
        win = self.window()
        if win is None:
            return
        is_dark = getattr(win, "_is_dark_theme", True)

        if command.requires_confirmation:
            confirmed = PrestigeDialog(
                win,
                command.confirmation_title or "Confirm",
                command.confirmation_message or "Continue?",
                is_dark=is_dark,
                confirm_text="Confirm",
                cancel_text="Cancel",
            ).exec()
            if not confirmed:
                self.setFocus()
                return

        result = execute_composer_command(command.id, window=win)
        if result.dialog_message:
            PrestigeDialog(
                win,
                result.dialog_title or ("Command complete" if result.ok else "Command failed"),
                result.dialog_message,
                is_dark=is_dark,
                confirm_text="OK",
                show_cancel=False,
            ).exec()
        if result.ok and result.notification and hasattr(win, "show_app_notification"):
            win.show_app_notification(result.notification)
        elif not result.ok and not result.dialog_message:
            PrestigeDialog(
                win,
                result.dialog_title or "Command failed",
                "The command could not be completed.",
                is_dark=is_dark,
                confirm_text="OK",
                show_cancel=False,
            ).exec()
        else:
            self.setFocus()

    def _on_mention_dismissed(self) -> None:
        self._mention_session_active = False
        self._mention_start_pos = -1
        self._disarm_mention_trigger()
        self.setFocus()

    def _mention_palette_active(self) -> bool:
        popup = self._mention_popup
        return bool(
            self._mention_session_active
            and popup is not None
            and popup.isVisible()
        )

    def keyPressEvent(self, event):
        win = self.window()
        nc = getattr(win, "notification_center", None)
        if nc is not None and nc.handle_key(event):
            return

        mention_active = self._mention_palette_active()
        if mention_active and self._mention_popup is not None:
            key = event.key()
            nav_keys = (
                Qt.Key.Key_Up,
                Qt.Key.Key_Down,
                Qt.Key.Key_Tab,
                Qt.Key.Key_Backspace,
                Qt.Key.Key_Escape,
                Qt.Key.Key_Return,
                Qt.Key.Key_Enter,
            )
            if key in nav_keys or (
                self._mention_popup._view_mode == ComposerPaletteView.BROWSE
                and (
                    Qt.Key.Key_1 <= key <= Qt.Key.Key_5
                    or (
                        getattr(Qt.Key, "Keypad1", None) is not None
                        and Qt.Key.Keypad1 <= key <= Qt.Key.Keypad5
                    )
                )
            ):
                if self._mention_popup.handle_navigation_key(event):
                    return

        if event.text() == "@":
            before = self.toPlainText()[: self.textCursor().position()]
            if is_valid_mention_anchor(before):
                super().keyPressEvent(event)
                if self._mention_armed:
                    self._mention_arm_count += 1
                else:
                    self._mention_armed = True
                    self._mention_arm_at = self.textCursor().position() - 1
                    self._mention_arm_count = 1
                    self._mention_arm_modifiers = event.modifiers()
                    self._mention_arm_trigger_key = event.key()
                event.accept()
                return
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                super().keyPressEvent(event)
                return
            if mention_active:
                return
            event.accept()
            self.submit_requested.emit()
            return
        if mention_active:
            super().keyPressEvent(event)
            return
        if key == Qt.Key.Key_Up and self._at_top_visual_line():
            cursor = self.textCursor()
            cursor.movePosition(
                QTextCursor.MoveOperation.Start,
                QTextCursor.MoveMode.MoveAnchor,
            )
            self.setTextCursor(cursor)
            event.accept()
            return
        if key == Qt.Key.Key_Down and self._at_bottom_visual_line():
            cursor = self.textCursor()
            cursor.movePosition(
                QTextCursor.MoveOperation.End,
                QTextCursor.MoveMode.MoveAnchor,
            )
            self.setTextCursor(cursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self._should_complete_mention_arm(event):
            self._complete_mention_arm()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _at_top_visual_line(self) -> bool:
        cursor = self.textCursor()
        probe = QTextCursor(cursor)
        probe.movePosition(
            QTextCursor.MoveOperation.Up,
            QTextCursor.MoveMode.KeepAnchor,
        )
        return probe.position() == cursor.position()

    def _at_bottom_visual_line(self) -> bool:
        cursor = self.textCursor()
        probe = QTextCursor(cursor)
        probe.movePosition(
            QTextCursor.MoveOperation.Down,
            QTextCursor.MoveMode.KeepAnchor,
        )
        return probe.position() == cursor.position()

    def _line_step_px(self) -> int:
        """One text line in px (prefer font height so ~7 lines match visible rows)."""
        fm = self.fontMetrics()
        return max(1, fm.height(), int(round(fm.lineSpacing())))

    def _chrome_height(self) -> int:
        m = self.contentsMargins()
        dm = float(self.document().documentMargin())
        return int(
            math.ceil(
                m.top()
                + m.bottom()
                + self.frameWidth() * 2
                + 2.0 * dm
                + 6
            )
        )

    def _min_height(self) -> int:
        return self._line_step_px() + self._chrome_height()

    def _max_height(self) -> int:
        return self._line_step_px() * self._MAX_VISIBLE_LINES + self._chrome_height()

    def _block_stack_bottom(self, vw: int) -> float:
        doc = self.document()
        doc.setTextWidth(float(vw))
        bottom = 0.0
        block = doc.firstBlock()
        while block.isValid():
            rect = self.blockBoundingGeometry(block)
            bottom = max(bottom, float(rect.bottom()))
            block = block.next()
        return bottom

    def _measured_body_height(self, vw: int) -> float:
        """Body height independent of current widget height (avoids layout/height feedback loops)."""
        doc = self.document()
        doc.setTextWidth(float(vw))
        layout = doc.documentLayout()
        lay_h = float(layout.documentSize().height()) if layout is not None else 0.0
        ds = doc.size()
        ds_h = float(ds.height()) if ds.height() > 0 else 0.0
        block_h = self._block_stack_bottom(vw)
        text = self.toPlainText()
        step = float(self._line_step_px())
        explicit_lines = max(1, text.count("\n") + 1) if text else 1
        explicit_h = float(explicit_lines) * step
        return max(step, lay_h, ds_h, block_h, explicit_h)

    def _sync_height_to_content(self) -> None:
        if self.document().documentLayout() is None:
            return
        vw = int(self.viewport().width())
        if vw <= 0:
            outer = self.width() - 2 * self.frameWidth()
            vw = outer
        if vw <= 0:
            h0 = self._min_height()
            if self.height() != h0:
                self.setFixedHeight(h0)
                self.updateGeometry()
            return
        doc_h = self._measured_body_height(vw)
        want = int(math.ceil(doc_h)) + self._chrome_height()
        lo, hi = self._min_height(), self._max_height()
        h = max(lo, min(want, hi))
        if self.height() == h:
            return
        self.setFixedHeight(h)
        self.updateGeometry()
        vw2 = max(1, int(self.viewport().width()))
        self.document().setTextWidth(float(vw2))


class ConversationsView(QWidget):
    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        self.llm = workers.get("llm")
        self.tts = workers.get("tts")
        self._pending_citation_sources = None
        self._pending_evidence_transparency = None
        self._pending_stream_tokens_by_session: dict[str, str] = {}
        self._pending_stream_sources_by_session: dict[str, list] = {}
        self._pending_stream_transparency_by_session: dict[str, dict] = {}
        self._user_turn_id = 0
        self._stt_ms_for_turn: int | None = None
        self._stt_ms_value: float | None = None
        self._pending_ttft_ms: float | None = None
        self._stop_requested_callback = None
        self._before_send_callback = None
        self._manual_voice_callback = None
        self._llm_in_progress = False
        self._awaiting_tts_end = False
        self._tts_playing = False
        self._voice_capture_active = False
        self._voice_turn_active = False
        self._layout_mode: str = LAYOUT_CENTERED_COLUMN
        self._font_scale: float = 1.0
        self._line_height_mode: str = _LINE_HEIGHT_COMFORTABLE
        self._focus_mode_enabled: bool = False
        self._high_contrast_enabled: bool = False
        self._assistant_message_background_enabled: bool = (
            get_ui_assistant_message_background()
        )
        self._reader_hover_wrapper: MessageWrapper | None = None
        self._transcript_alignment: str = ALIGN_JUSTIFY
        self._agent_typing_wrapper: MessageWrapper | None = None
        self._agent_md_coalesce_timer = QTimer(self)
        self._agent_md_coalesce_timer.setSingleShot(True)
        self._agent_md_coalesce_timer.timeout.connect(self._flush_coalesced_agent_markdown)
        self._transcript_waypoints: list[_TranscriptWaypointRecord] = []
        self._transcript_timeline_refresh_timer = QTimer(self)
        self._transcript_timeline_refresh_timer.setSingleShot(True)
        self._transcript_timeline_refresh_timer.timeout.connect(
            self._refresh_transcript_timeline_rail
        )
        self._transcript_timeline_scroll_anim: QPropertyAnimation | None = None
        self._transcript_timeline_prev_sc: QShortcut | None = None
        self._transcript_timeline_next_sc: QShortcut | None = None

        self._active_folder_id: str | None = None
        self._folder_controller: SidebarFolderListController | None = None
        self._composer_draft = ComposerDraft()
        self._active_deep_research_request_id: str | None = None
        self._deep_research_session_id: str | None = None
        self._deep_research_in_progress = False

        self._setup_ui()
        self._start_new_chat()

    def focus_composer_if_ready(self) -> None:
        """Give the chat composer keyboard focus when the view is ready."""
        if not hasattr(self, "text_input") or not self.text_input.isEnabled():
            return
        win = self.window()
        if win is not None:
            tour = getattr(win, "_active_tour", None)
            if tour is not None and getattr(tour, "is_active", False):
                return
            if getattr(win, "_composer_at_mention_discovery", None) is not None:
                return
            if not win.isActiveWindow():
                win.activateWindow()
        self.text_input.setFocus(Qt.FocusReason.OtherFocusReason)

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

    def transcript_column_nominal_width(self) -> int:
        """Layout-mode column cap (800 narrow / 1200 wide), independent of current viewport size."""
        return (
            _FULL_WIDTH_COLUMN_MAX_WIDTH
            if self._layout_mode == LAYOUT_FULL_WIDTH
            else _CENTERED_COLUMN_MAX_WIDTH
        )

    def transcript_column_max_width(self) -> int:
        """Effective transcript column cap: nominal mode width, clamped to the scroll viewport."""
        nominal = self.transcript_column_nominal_width()
        if hasattr(self, "scroll_area"):
            vp = self.scroll_area.viewport()
            if vp is not None:
                vw = int(vp.width())
                if vw > 0:
                    # Never exceed the viewport — a 160px floor here caused the transcript
                    # container to outgrow narrow layouts and bleed under the history sidebar
                    # when the scroll area centered the oversized widget.
                    return min(nominal, vw)
        return nominal

    def set_layout_mode(self, mode: str) -> None:
        """Switch transcript column between 800px (centered) and 1200px (wide).

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
        self._sync_transcript_column_width_cap()
        composer_host = getattr(self, "_composer_row_host", None)
        if composer_host is not None:
            composer_host.set_column_max_width(self.transcript_column_nominal_width())
        self.transcript_layout.invalidate()
        self.transcript_container.updateGeometry()
        self._refresh_layout_mode_button()
        self._schedule_transcript_timeline_refresh()

    def _sync_transcript_scroll_alignment(self) -> None:
        """Pin transcript content to the left edge of the capped scroll column."""
        if not hasattr(self, "scroll_area"):
            return
        self.scroll_area.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )

    def _sync_transcript_column_width_cap(self) -> None:
        if not hasattr(self, "transcript_container") or not hasattr(self, "scroll_area"):
            return
        self._sync_transcript_scroll_alignment()
        column_host = getattr(self, "_transcript_column_host", None)
        if column_host is not None:
            column_host.sync_geometry()
        cap = self.transcript_column_max_width()
        if self.transcript_container.maximumWidth() != cap:
            self.transcript_container.setMaximumWidth(cap)
        self.transcript_layout.invalidate()
        self.transcript_container.updateGeometry()
        self._sync_agent_actions_bar_widths()

    def _theme(self, is_dark: bool | None = None):
        return view_resolved_theme(self, is_dark=is_dark)

    def _refresh_layout_mode_button(self, is_dark: bool | None = None) -> None:
        btn = getattr(self, "layout_mode_btn", None)
        if btn is None:
            return
        if is_dark is None:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        icon_color = accent_icon_color(theme)
        if self.layout_mode == LAYOUT_CENTERED_COLUMN:
            btn.setIcon(
                tinted_svg_icon(
                    _LAYOUT_ICON_NARROW, icon_color, size=_CHAT_UTILITY_ICON_PX
                )
            )
            btn.setToolTip(f"Layout mode: Narrow column ({_CENTERED_COLUMN_MAX_WIDTH}px)")
        else:
            btn.setIcon(
                tinted_svg_icon(
                    _LAYOUT_ICON_WIDE, icon_color, size=_CHAT_UTILITY_ICON_PX
                )
            )
            btn.setToolTip(f"Layout mode: Wide column ({_FULL_WIDTH_COLUMN_MAX_WIDTH}px)")
        btn.setIconSize(QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX))
        btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        btn.setStyleSheet(theme.style(GHOST_ICON_BUTTON))

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

    def _reading_font_family(self) -> str:
        from core.app_settings import get_ui_reading_font
        from core.reading_fonts import reading_font_qt_family

        return reading_font_qt_family(get_ui_reading_font())

    def refresh_reading_font(self) -> None:
        self._refresh_all_readability()

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
        theme = self._theme(is_dark)
        return theme.style(
            HIGH_CONTRAST_MARKDOWN,
            enabled=True,
            high_contrast=self._high_contrast_enabled,
        )

    def _agent_markdown_stylesheet(self, is_dark: bool) -> str:
        base = _markdown_ui_stylesheet(is_dark, theme=self._theme(is_dark))
        parts = [base, self._high_contrast_markdown_css(is_dark)]
        return "".join(parts)

    def _user_bubble_label_colors(self, is_dark: bool) -> tuple[str, str]:
        """(text_color, optional extra label style fragment)."""
        from core.theme.widget_styles import _user_bubble_text

        fg = _user_bubble_text(self._theme(is_dark), high_contrast=self._high_contrast_enabled)
        return fg, ""

    def _user_bubble_frame_bg(self, is_dark: bool) -> str:
        from core.theme.widget_styles import _user_bubble_frame

        return _user_bubble_frame(self._theme(is_dark), high_contrast=self._high_contrast_enabled)

    def _qube_response_header_color(self, is_dark: bool) -> str:
        """Assistant turn 'QUBE' label — unchanged by high-contrast transcript mode."""
        return self._theme(is_dark).color(QUBE_RESPONSE_HEADER)

    def _placeholder_muted_color(self, is_dark: bool) -> str:
        return self._theme(is_dark).color(PLACEHOLDER_MUTED)

    def _make_transcript_placeholder_label(self, text: str) -> QLabel:
        """Empty-state label: wraps within the transcript column and never forces horizontal growth."""
        lbl = QLabel(text)
        lbl.setObjectName("TranscriptPlaceholderLabel")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        lbl.setMinimumWidth(0)
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._style_transcript_placeholder_label(lbl)
        return lbl

    def _style_transcript_placeholder_label(self, lbl: QLabel) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        pt = self._scaled_chat_font_pt()
        muted = self._placeholder_muted_color(is_dark)
        lbl.setStyleSheet(
            f"color: {muted}; font-size: {pt:.1f}pt; margin-top: 50px; font-weight: bold;"
            f" background: transparent; border: none;"
        )

    def _style_user_bubble(self, bubble: QFrame, lbl: ChatUserBubble) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        pt = self._scaled_chat_font_pt()
        family = self._reading_font_family()
        f = lbl.font()
        f.setPointSizeF(pt)
        f.setFamily(family)
        lbl.setFont(f)
        lbl.document().setDefaultFont(f)
        opt = QTextOption()
        opt.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        opt.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        lbl.document().setDefaultTextOption(opt)
        lbl.setStyleSheet(
            theme.style(
                USER_BUBBLE_LABEL,
                high_contrast=self._high_contrast_enabled,
                font_pt=pt,
                font_family=family,
            )
        )
        bubble.setStyleSheet(
            theme.style(USER_BUBBLE_FRAME, high_contrast=self._high_contrast_enabled)
        )
        lbl.document().setTextWidth(float(max(1, lbl._effective_wrap_width())))
        lbl._schedule_height_sync()
        lbl.updateGeometry()

    def _style_agent_message_shell(self, agent: AgentMessageLabel) -> None:
        pt = self._scaled_chat_font_pt()
        family = self._reading_font_family()
        f = agent.font()
        f.setPointSizeF(pt)
        f.setFamily(family)
        agent.setFont(f)
        theme = self._theme(getattr(self.window(), "_is_dark_theme", True))
        agent.setStyleSheet(
            theme.style(AGENT_MESSAGE_SHELL, font_pt=pt, font_family=family)
        )
        fg = theme.qcolor(theme.text_primary)
        palette = agent.palette()
        palette.setColor(QPalette.ColorRole.Text, fg)
        palette.setColor(QPalette.ColorRole.WindowText, fg)
        agent.setPalette(palette)
        viewport = agent.viewport()
        if viewport is not None:
            vpal = viewport.palette()
            vpal.setColor(QPalette.ColorRole.Text, fg)
            vpal.setColor(QPalette.ColorRole.WindowText, fg)
            viewport.setPalette(vpal)

    def _style_agent_message_container(self, container: AgentMessageContainer) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        enabled = self._assistant_message_background_enabled
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, enabled)
        container.setStyleSheet(
            theme.style(
                AGENT_MESSAGE_FRAME,
                enabled=enabled,
                high_contrast=self._high_contrast_enabled,
            )
        )
        left, top, right, bottom = (
            _AGENT_MESSAGE_CARD_MARGINS if enabled else (0, 0, 0, 0)
        )
        container._layout.setContentsMargins(left, top, right, bottom)

    def refresh_assistant_message_background(self) -> None:
        self._assistant_message_background_enabled = (
            get_ui_assistant_message_background()
        )
        self._refresh_all_readability()

    def _style_agent_copy_button(self, btn: QPushButton, is_dark: bool) -> None:
        theme = self._theme(is_dark)
        btn.setIcon(themed_fa_icon("fa5s.copy", theme.color(MUTED_ICON), 16))
        btn.setIconSize(QSize(14, 14))
        btn.setStyleSheet(theme.style(AGENT_COPY_BUTTON))

    def _style_agent_sources_button(self, btn: QPushButton, is_dark: bool) -> None:
        theme = self._theme(is_dark)
        btn.setIcon(themed_fa_icon("fa5s.globe", theme.color(MUTED_ICON), 16))
        btn.setIconSize(QSize(14, 14))
        btn.setStyleSheet(theme.style(AGENT_COPY_BUTTON))

    def _style_agent_sources_button(self, btn: QPushButton, is_dark: bool) -> None:
        theme = self._theme(is_dark)
        btn.setIcon(themed_fa_icon("fa5s.globe", theme.color(MUTED_ICON), 16))
        btn.setIconSize(QSize(14, 14))
        btn.setStyleSheet(theme.style(AGENT_COPY_BUTTON))

    def _style_agent_export_button(self, btn: QPushButton, is_dark: bool) -> None:
        theme = self._theme(is_dark)
        btn.setIcon(themed_fa_icon("fa5s.file-export", theme.color(MUTED_ICON), 16))
        btn.setIconSize(QSize(14, 14))
        btn.setStyleSheet(theme.style(AGENT_COPY_BUTTON))

    def _sync_agent_export_button(
        self, agent: AgentMessageLabel, btn: QPushButton | None = None
    ) -> None:
        btn = btn or getattr(agent, "_export_action_btn", None)
        if btn is None:
            return
        markdown = (getattr(agent, "_md_layout_source", None) or "").strip()
        if not markdown:
            markdown = agent.markdown_for_clipboard()
        visible = has_exportable_assistant_content(markdown)
        btn.setVisible(visible)
        btn.setEnabled(visible)
        if visible:
            btn.setToolTip("Export answer as Markdown or PDF")

    def _export_agent_message(
        self,
        agent: AgentMessageLabel,
        *,
        as_pdf: bool = False,
    ) -> None:
        markdown = agent.markdown_for_clipboard()
        if not has_exportable_assistant_content(markdown):
            return
        body = format_assistant_message_for_export(markdown)
        stem = suggested_assistant_export_stem(markdown)
        if as_pdf:
            default_name = f"{stem}.pdf"
            file_filter = "PDF (*.pdf)"
        else:
            default_name = f"{stem}.md"
            file_filter = "Markdown (*.md)"
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Export Answer",
            default_name,
            file_filter,
        )
        if not dest:
            return
        try:
            if as_pdf:
                is_dark = getattr(self.window(), "_is_dark_theme", True)
                write_markdown_pdf(
                    body,
                    Path(dest),
                    document_stylesheet=self._agent_markdown_stylesheet(is_dark),
                )
            else:
                write_assistant_message_markdown(markdown, Path(dest))
            logger.info("Exported assistant answer to %s", dest)
        except OSError as exc:
            logger.exception("Failed to export assistant answer: %s", exc)

    def _build_agent_export_menu(
        self, agent: AgentMessageLabel, is_dark: bool
    ) -> QMenu:
        menu = QMenu()
        menu.setObjectName("PrestigeMenu")
        self._apply_menu_theme(menu, is_dark)
        md_act = menu.addAction("Save as Markdown")
        md_act.triggered.connect(
            lambda _checked=False, lbl=agent: self._export_agent_message(
                lbl, as_pdf=False
            )
        )
        pdf_act = menu.addAction("Save as PDF")
        pdf_act.triggered.connect(
            lambda _checked=False, lbl=agent: self._export_agent_message(
                lbl, as_pdf=True
            )
        )
        return menu

    def _sync_agent_sources_button(
        self, agent: AgentMessageLabel, btn: QPushButton | None = None
    ) -> None:
        btn = btn or getattr(agent, "_sources_action_btn", None)
        if btn is None:
            return
        sources = getattr(agent, "_citation_sources", None) or []
        count = len(sources)
        visible = count > 0
        btn.setVisible(visible)
        btn.setEnabled(visible)
        if visible:
            btn.setToolTip(
                f"View {count} source{'s' if count != 1 else ''} used in this answer"
            )
        else:
            btn.setToolTip("")

    def _show_agent_citation_sources(self, agent: AgentMessageLabel) -> None:
        sources = getattr(agent, "_citation_sources", None) or []
        if not sources:
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        transparency = getattr(agent, "_evidence_transparency", None)
        research_map_graph = None
        on_open_research_map = None

        if getattr(self, "active_session_id", None):
            from core.knowledge.graph.build import subgraph_for_bundle

            session_graph = self.db.get_session_knowledge_graph(
                str(self.active_session_id)
            )
            if session_graph:
                bundle_id = str((transparency or {}).get("bundle_id") or "")
                research_map_graph = (
                    subgraph_for_bundle(session_graph, bundle_id)
                    if bundle_id
                    else session_graph
                )
                if research_map_graph.get("nodes"):

                    def _open_map() -> None:
                        ResearchMapDialog(
                            research_map_graph,
                            self,
                            is_dark=is_dark,
                        ).exec()

                    on_open_research_map = _open_map
        bundle_id = str((transparency or {}).get("bundle_id") or "")
        dlg = CitationSourcesDialog(
            sources,
            self,
            is_dark=is_dark,
            on_open_source=self.open_source_preview,
            transparency=transparency,
            research_map_graph=research_map_graph,
            on_open_research_map=on_open_research_map,
            retrieval_bundle_id=bundle_id or None,
            retrieval_db=self.db,
        )
        dlg.exec()

    def _copy_agent_message_to_clipboard(
        self, agent: AgentMessageLabel, *, as_markdown: bool = False
    ) -> None:
        text = (
            agent.markdown_for_clipboard()
            if as_markdown
            else agent.plain_text_for_clipboard()
        )
        if not text:
            return
        QApplication.clipboard().setText(text)

    def _build_agent_copy_menu(
        self, agent: AgentMessageLabel, is_dark: bool
    ) -> QMenu:
        menu = QMenu()
        menu.setObjectName("PrestigeMenu")
        self._apply_menu_theme(menu, is_dark)
        plain_act = menu.addAction("Copy as plain text")
        plain_act.triggered.connect(
            lambda _checked=False, lbl=agent: self._copy_agent_message_to_clipboard(
                lbl, as_markdown=False
            )
        )
        md_act = menu.addAction("Copy as Markdown")
        md_act.triggered.connect(
            lambda _checked=False, lbl=agent: self._copy_agent_message_to_clipboard(
                lbl, as_markdown=True
            )
        )
        return menu

    def _create_agent_telemetry_label(self, text: str, is_dark: bool) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("class", "AgentMessageTelemetryLabel")
        lbl.setFixedHeight(28)
        lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self._style_agent_telemetry_label(lbl, is_dark)
        return lbl

    def _style_agent_telemetry_label(self, lbl: QLabel, is_dark: bool) -> None:
        lbl.setStyleSheet(self._theme(is_dark).style(TELEMETRY_LABEL))

    def _default_agent_telemetry_labels(self, is_dark: bool) -> dict[str, QLabel]:
        return {
            "stt": self._create_agent_telemetry_label("STT: --", is_dark),
            "ttft": self._create_agent_telemetry_label("TTFT: --", is_dark),
            "tts": self._create_agent_telemetry_label("TTS: --", is_dark),
            "tps": self._create_agent_telemetry_label("TPS: --", is_dark),
        }

    def _format_agent_telemetry_text(self, metric: str, value: float | None) -> str:
        if metric in ("stt", "ttft", "tts"):
            prefix = metric.upper()
            if value is None:
                return f"{prefix}: --"
            return f"{prefix}: {float(value) / 1000.0:.1f} seconds"
        if metric == "tps":
            if value is not None and value > 0:
                return f"TPS: {value:.1f} tok/s"
            return "TPS: --"
        return ""

    def _set_agent_telemetry_metric(
        self, agent: AgentMessageLabel, metric: str, value: float | None
    ) -> None:
        labels = getattr(agent, "_telemetry_labels", None)
        if not labels or metric not in labels:
            return
        stored = getattr(agent, "_telemetry_values", None)
        if stored is None:
            stored = {}
            agent._telemetry_values = stored
        stored[metric] = value
        labels[metric].setText(self._format_agent_telemetry_text(metric, value))

    def _apply_pending_stt_to_agent(self, agent: AgentMessageLabel) -> None:
        turn_id = getattr(agent, "_assistant_turn_id", None)
        if (
            turn_id is not None
            and self._stt_ms_for_turn == turn_id
            and self._stt_ms_value is not None
        ):
            self._set_agent_telemetry_metric(agent, "stt", self._stt_ms_value)
            self._stt_ms_for_turn = None
            self._stt_ms_value = None

    def _apply_pending_ttft_to_agent(self, agent: AgentMessageLabel) -> None:
        if self._pending_ttft_ms is not None:
            self._set_agent_telemetry_metric(agent, "ttft", self._pending_ttft_ms)
            self._pending_ttft_ms = None

    def _sync_agent_actions_bar_widths(self) -> None:
        for w in self._iter_transcript_widgets():
            if not isinstance(w, MessageWrapper) or w.is_user:
                continue
            container = w.bubble
            if isinstance(container, AgentMessageContainer):
                container._sync_actions_bar_width()

    def _add_agent_copy_button(
        self, container: AgentMessageContainer, agent: AgentMessageLabel
    ) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        actions_bar = QWidget()
        actions_bar.setObjectName("AgentMessageActionsBar")
        copy_row = QHBoxLayout(actions_bar)
        copy_row.setContentsMargins(0, 0, 0, 0)
        copy_row.setSpacing(2)

        copy_btn = QPushButton()
        copy_btn.setObjectName("AgentMessageCopyBtn")
        copy_btn.setFixedSize(28, 28)
        copy_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.setToolTip("Copy as plain text or Markdown")
        self._style_agent_copy_button(copy_btn, is_dark)
        copy_menu = self._build_agent_copy_menu(agent, is_dark)
        copy_btn.setMenu(copy_menu)

        sources_btn = QPushButton()
        sources_btn.setObjectName("AgentMessageSourcesBtn")
        sources_btn.setFixedSize(28, 28)
        sources_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sources_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._style_agent_sources_button(sources_btn, is_dark)
        sources_btn.clicked.connect(
            lambda _checked=False, lbl=agent: self._show_agent_citation_sources(lbl)
        )
        agent._sources_action_btn = sources_btn
        self._sync_agent_sources_button(agent, sources_btn)

        export_btn = QPushButton()
        export_btn.setObjectName("AgentMessageExportBtn")
        export_btn.setFixedSize(28, 28)
        export_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._style_agent_export_button(export_btn, is_dark)
        export_btn.setMenu(self._build_agent_export_menu(agent, is_dark))
        agent._export_action_btn = export_btn
        self._sync_agent_export_button(agent, export_btn)

        agent._telemetry_labels = self._default_agent_telemetry_labels(is_dark)
        agent._telemetry_values = {}

        copy_row.addWidget(copy_btn, 0, Qt.AlignmentFlag.AlignLeft)
        copy_row.addWidget(sources_btn, 0, Qt.AlignmentFlag.AlignLeft)
        copy_row.addWidget(export_btn, 0, Qt.AlignmentFlag.AlignLeft)
        copy_row.addStretch(1)
        copy_row.addSpacing(12)
        for key in ("stt", "ttft", "tts", "tps"):
            copy_row.addWidget(
                agent._telemetry_labels[key],
                0,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
        container.attach_actions_bar(actions_bar)
        self._apply_pending_stt_to_agent(agent)
        self._apply_pending_ttft_to_agent(agent)
        agent._help_action_chips = []
        agent._help_action_buttons = []

    def _style_help_action_chip(self, btn: QPushButton, is_dark: bool) -> None:
        btn.setStyleSheet(self._theme(is_dark).style(HELP_ACTION_CHIP))

    def _on_help_action_chip_clicked(self, settings_section: str) -> None:
        win = self.window()
        if win is not None and hasattr(win, "_open_settings_section"):
            win._open_settings_section(settings_section)
            return
        from ui.onboarding.tour_helpers import open_settings_section

        if win is not None:
            open_settings_section(win, settings_section)

    def _sync_help_action_chips(self, agent: AgentMessageLabel) -> None:
        container = agent.parentWidget()
        if not isinstance(container, AgentMessageContainer):
            return
        actions_bar = getattr(container, "_actions_bar", None)
        if actions_bar is None:
            return
        row = actions_bar.layout()
        if row is None:
            return

        for btn in list(getattr(agent, "_help_action_buttons", []) or []):
            row.removeWidget(btn)
            btn.deleteLater()
        agent._help_action_buttons = []

        chips: list[HelpActionChip] = list(getattr(agent, "_help_action_chips", []) or [])
        if not chips:
            return

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        insert_at = 3  # after copy, sources, and export buttons
        for chip in chips:
            btn = QPushButton(chip.label)
            btn.setObjectName("HelpActionChipBtn")
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(chip.label)
            self._style_help_action_chip(btn, is_dark)
            btn.clicked.connect(
                lambda _checked=False, section=chip.settings_section: self._on_help_action_chip_clicked(
                    section
                )
            )
            row.insertWidget(insert_at, btn, 0, Qt.AlignmentFlag.AlignLeft)
            insert_at += 1
            agent._help_action_buttons.append(btn)
        container._sync_actions_bar_width()

    def _refresh_agent_copy_buttons(self, is_dark: bool) -> None:
        for w in self._iter_transcript_widgets():
            if not isinstance(w, MessageWrapper) or w.is_user:
                continue
            copy_btn = w.findChild(QPushButton, "AgentMessageCopyBtn")
            if copy_btn is not None:
                self._style_agent_copy_button(copy_btn, is_dark)
                menu = copy_btn.menu()
                if menu is not None:
                    self._apply_menu_theme(menu, is_dark)
            sources_btn = w.findChild(QPushButton, "AgentMessageSourcesBtn")
            if sources_btn is not None:
                self._style_agent_sources_button(sources_btn, is_dark)
            export_btn = w.findChild(QPushButton, "AgentMessageExportBtn")
            if export_btn is not None:
                self._style_agent_export_button(export_btn, is_dark)
                menu = export_btn.menu()
                if menu is not None:
                    self._apply_menu_theme(menu, is_dark)
            for agent in w.findChildren(AgentMessageLabel):
                self._sync_agent_sources_button(agent, sources_btn)
                self._sync_agent_export_button(agent, export_btn)
                labels = getattr(agent, "_telemetry_labels", None)
                if labels:
                    for lbl in labels.values():
                        self._style_agent_telemetry_label(lbl, is_dark)

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
        for w in self._iter_transcript_widgets():
            if isinstance(w, QLabel) and w is not getattr(self, "placeholder_lbl", None):
                qube_hdr = self._qube_response_header_color(is_dark)
                w.setStyleSheet(
                    f"color: {qube_hdr}; font-weight: bold; font-size: {pt:.1f}pt; margin-top: 15px; background: transparent;"
                )
        pl = getattr(self, "placeholder_lbl", None)
        if pl is not None:
            self._style_transcript_placeholder_label(pl)

    def _refresh_transcript_theme(self) -> None:
        """Update transcript chrome + markdown colors without re-parsing every bubble."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        sheet = self._agent_markdown_stylesheet(is_dark)
        line_height = self._line_height_proportional_percent()
        justify = self._transcript_alignment == ALIGN_JUSTIFY
        for w in self._iter_transcript_widgets():
            if isinstance(w, MessageWrapper):
                if w.is_user and w.bubble is not None:
                    lbl = w.bubble.findChild(ChatUserBubble)
                    if lbl is not None:
                        self._style_user_bubble(w.bubble, lbl)
                else:
                    container = w.bubble
                    if isinstance(container, AgentMessageContainer):
                        self._style_agent_message_container(container)
                    for agent in w.findChildren(AgentMessageLabel):
                        self._style_agent_message_shell(agent)
                        if agent._md_layout_source:
                            agent.refresh_theme_styles(
                                is_dark=is_dark,
                                document_stylesheet=sheet,
                                line_height_percent=line_height,
                                justify_transcript=justify,
                            )
        self._refresh_ancillary_transcript_labels()
        self._sync_agent_actions_bar_widths()

    def _refresh_all_readability(self) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        sheet = self._agent_markdown_stylesheet(is_dark)
        for w in self._iter_transcript_widgets():
            if isinstance(w, MessageWrapper):
                if w.is_user and w.bubble is not None:
                    lbl = w.bubble.findChild(ChatUserBubble)
                    if lbl is not None:
                        self._style_user_bubble(w.bubble, lbl)
                else:
                    container = w.bubble
                    if isinstance(container, AgentMessageContainer):
                        self._style_agent_message_container(container)
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
        self._sync_agent_actions_bar_widths()
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

    def _refresh_transcript_wallpaper(self) -> None:
        bind_transcript_wallpaper_readability(
            getattr(self, "_chat_transcript_wallpaper_host", None),
            high_contrast=self._high_contrast_enabled,
            reader_focus=self._focus_mode_enabled,
        )

    def _on_reader_focus_toggled(self, checked: bool) -> None:
        self._focus_mode_enabled = bool(checked)
        if not self._focus_mode_enabled:
            self._reader_hover_wrapper = None
        self._refresh_readability_toolbar()
        self._apply_reader_focus_opacity()
        self._refresh_transcript_wallpaper()

    def _on_high_contrast_toggled(self, checked: bool) -> None:
        self._high_contrast_enabled = bool(checked)
        self._refresh_all_readability()
        self._refresh_transcript_wallpaper()

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
        theme = self._theme(is_dark)
        icon_muted = accent_icon_color(theme)
        icon_active = theme.color(ACCENT_ICON_ACTIVE)
        utility_icon_style = theme.style(UTILITY_ICON_BUTTON)
        is_justify = self._transcript_alignment == ALIGN_JUSTIFY
        self.text_align_btn.setToolTip(
            "Text alignment: Justified (click for left)"
            if is_justify
            else "Text alignment: Left (click for justified)"
        )
        self.text_align_btn.setIcon(
            themed_fa_icon(
                "fa5s.align-justify" if is_justify else "fa5s.align-left",
                icon_muted,
                _CHAT_UTILITY_ICON_PX,
            )
        )
        self.text_align_btn.setIconSize(
            QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX)
        )
        self.text_align_btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        lh_icon_color = icon_muted
        self.line_height_btn.setIcon(
            tinted_svg_icon(_LINE_SPACING_ICON, lh_icon_color, size=_CHAT_UTILITY_ICON_PX)
        )
        self.line_height_btn.setIconSize(
            QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX)
        )
        self.line_height_btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        self.reader_focus_btn.setIcon(
            themed_fa_icon(
                "fa5s.crosshairs",
                icon_active if self._focus_mode_enabled else icon_muted,
                _CHAT_UTILITY_ICON_PX,
            )
        )
        self.reader_focus_btn.setIconSize(
            QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX)
        )
        self.reader_focus_btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        self.high_contrast_btn.setIcon(
            themed_fa_icon(
                "fa5s.adjust",
                icon_active if self._high_contrast_enabled else icon_muted,
                _CHAT_UTILITY_ICON_PX,
            )
        )
        self.high_contrast_btn.setIconSize(
            QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX)
        )
        self.high_contrast_btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        if hasattr(self, "conversation_download_btn"):
            self.conversation_download_btn.setIcon(
                themed_fa_icon("fa5s.download", icon_muted, _CHAT_UTILITY_ICON_PX)
            )
            self.conversation_download_btn.setIconSize(
                QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX)
            )
            self.conversation_download_btn.setFixedSize(
                _CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN
            )
        if hasattr(self, "conversation_copy_btn"):
            self.conversation_copy_btn.setIcon(
                themed_fa_icon("fa5s.copy", icon_muted, _CHAT_UTILITY_ICON_PX)
            )
            self.conversation_copy_btn.setIconSize(
                QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX)
            )
            self.conversation_copy_btn.setFixedSize(
                _CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN
            )
        for btn in (
            self.line_height_btn,
            self.text_align_btn,
            self.reader_focus_btn,
            self.high_contrast_btn,
            getattr(self, "conversation_download_btn", None),
            getattr(self, "conversation_copy_btn", None),
        ):
            if btn is not None:
                btn.setStyleSheet(utility_icon_style)

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1) 

        self.history_pane = self._build_history_pane()
        layout.addWidget(self.history_pane)

        self.chat_stage = self._build_chat_stage()

        # Ignored horizontal policy: transcript placeholders must not widen the stage past
        # the allocated cell (matches Library preview_stage — prevents bleed under sidebars).
        self.chat_stage.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )

        layout.addWidget(self.chat_stage, stretch=1) 

        self.refresh_button_themes(getattr(self.window(), "_is_dark_theme", True))
        self.refresh_think_toggle()

        from ui.components.type_to_search import install_type_to_focus

        install_type_to_focus(
            self,
            self.text_input,
            extra_block=self._composer_type_to_focus_blocked,
        )

    def _composer_type_to_focus_blocked(self) -> bool:
        win = self.window()
        return win is not None and getattr(win, "_composer_at_mention_discovery", None) is not None

    # --------------------------------------------------------- #
    #  PANEL BUILDERS                                           #
    # --------------------------------------------------------- #

    def _build_history_pane(self) -> QFrame:
        frame = QFrame()
        frame.setFixedWidth(LEFT_NAV_LIST_SIDEBAR_WIDTH)
        frame.setObjectName("HistorySidebar")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)

        header_layout = QHBoxLayout()
        
        # --- THE FIX: Change 'title' to 'self.list_title' ---
        self.list_title = QLabel("Conversations")
        self.list_title.setObjectName("ViewTitle")
        self.list_title.setProperty("class", "PageTitle")
        
        # --- THE FIX: Make sure you add 'self.list_title' to the layout here ---
        header_layout.addWidget(self.list_title)
        self.page_tour_help_btn = PageTourHelpButton(
            "conversations",
            area_display_name="Conversations",
            parent=frame,
        )
        header_layout.addWidget(self.page_tour_help_btn)
        
        header_layout.addStretch()
        (
            _actions_host,
            actions_outer,
            actions_cluster,
        ) = create_sidebar_header_actions_row()
        header_layout.addWidget(_actions_host)

        self.new_chat_btn = QPushButton()
        self.new_chat_btn.setIcon(
            themed_fa_icon("fa5s.plus", accent_icon_color(self._theme()), 16)
        )
        self.new_chat_btn.setIconSize(QSize(16, 16))
        self.new_chat_btn.setToolTip("New conversation")
        apply_ghost_icon_button_style(self.new_chat_btn, self._theme())
        actions_outer.addWidget(self.new_chat_btn)

        self.new_folder_btn = add_new_folder_header_button(
            actions_cluster,
            on_new_folder=lambda: self._folder_controller.prompt_create_folder()
            if self._folder_controller
            else None,
            theme_host=self,
        )
        layout.addLayout(header_layout)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search titles or messages…")
        self.search_bar.setObjectName("HistorySearch")
        self.search_bar.setToolTip("Search by conversation title or message text")
        layout.addWidget(self.search_bar)
        self._history_search_timer = QTimer(self)
        self._history_search_timer.setSingleShot(True)
        self._history_search_timer.timeout.connect(self._reload_history_sidebar)
        self.search_bar.textChanged.connect(self._on_history_search_changed)

        self.history_list = QListWidget()
        self.history_list.setObjectName("HistoryList")
        layout.addWidget(self.history_list)

        self.new_chat_btn.clicked.connect(self._start_new_chat)
        self.history_list.itemClicked.connect(self._on_history_item_clicked)
        self.history_list.itemSelectionChanged.connect(self._update_row_colors)

        self._active_folder_id = self.db.get_main_conversation_folder_id()
        self._folder_controller = SidebarFolderListController(
            scope="conversation",
            list_widget=self.history_list,
            db=self.db,
            parent=self,
            append_item_row=self._append_history_session_row,
            apply_menu_theme=self._apply_menu_theme,
            get_is_dark=lambda: getattr(self.window(), "_is_dark_theme", True),
            on_reload=self._reload_history_sidebar,
            on_active_folder_changed=self._set_active_folder_id,
            on_export_folder=self._trigger_export_folder,
        )
        self.sort_btn = self._folder_controller.setup_sort_header_button(
            actions_cluster
        )

        self.history_list.itemDoubleClicked.connect(self._on_history_item_double_clicked)

        return frame

    def _build_chat_stage(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("ChatStage")
        frame.setStyleSheet(
            "QFrame#ChatStage { background: transparent; border: none; }"
        )
        outer_layout = QVBoxLayout(frame)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        mainstage_content = QWidget()
        mainstage_content.setObjectName("ChatMainstageContent")
        layout = QVBoxLayout(mainstage_content)
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

        conversation_actions_host = QWidget()
        conversation_actions_host.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        conversation_actions_layout = QHBoxLayout(conversation_actions_host)
        conversation_actions_layout.setContentsMargins(0, 0, 0, 0)
        conversation_actions_layout.setSpacing(6)

        self.conversation_download_btn = QPushButton()
        self.conversation_download_btn.setObjectName("ConversationDownloadButton")
        self.conversation_download_btn.setProperty("class", "IconButton")
        self.conversation_download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.conversation_download_btn.setToolTip("Download conversation as Markdown")
        self.conversation_download_btn.clicked.connect(self._export_active_conversation)

        self.conversation_copy_btn = QPushButton()
        self.conversation_copy_btn.setObjectName("ConversationCopyButton")
        self.conversation_copy_btn.setProperty("class", "IconButton")
        self.conversation_copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.conversation_copy_btn.setToolTip("Copy conversation to clipboard")
        self.conversation_copy_btn.clicked.connect(
            self._copy_active_conversation_to_clipboard
        )
        self.conversation_download_btn.setEnabled(False)
        self.conversation_copy_btn.setEnabled(False)

        conversation_actions_layout.addWidget(self.conversation_download_btn)
        conversation_actions_layout.addWidget(self.conversation_copy_btn)

        utility_layout.addWidget(readability_host, 0, Qt.AlignmentFlag.AlignLeft)
        utility_layout.addStretch(1)
        utility_layout.addWidget(
            conversation_actions_host, 0, Qt.AlignmentFlag.AlignRight
        )
        layout.addWidget(utility_toolbar)

        # Transcript column: scroll area + turn-index rail appended to the right of the bubbles.
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("ChatScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.installEventFilter(self)
        self.scroll_area.viewport().installEventFilter(self)
        self.scroll_area.verticalScrollBar().valueChanged.connect(
            self._on_transcript_scroll_changed
        )

        def _on_transcript_scrollbar_geometry_changed(*_args) -> None:
            column_host = getattr(self, "_transcript_column_host", None)
            if column_host is not None:
                column_host.sync_geometry()

        self.scroll_area.verticalScrollBar().rangeChanged.connect(
            _on_transcript_scrollbar_geometry_changed
        )

        # Container widget
        self.transcript_container = QWidget()
        self.transcript_container.setObjectName("ChatTranscriptContainer")
        self.transcript_container.setMinimumWidth(0)

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
        self.scroll_area.setStyleSheet(
            "QScrollArea#ChatScrollArea { background: transparent; border: none; }"
        )

        self._transcript_timeline_rail = TranscriptTimelineRail()
        self._transcript_timeline_rail.waypoint_clicked.connect(
            self._on_transcript_timeline_waypoint_clicked
        )
        self._transcript_timeline_rail.hide()
        self._transcript_timeline_rail.apply_theme(self._theme().is_dark)

        self._transcript_column_host = _TranscriptColumnHost(
            self.scroll_area,
            self._transcript_timeline_rail,
            nominal_cap_provider=self.transcript_column_nominal_width,
        )
        self._transcript_column_host.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout.addWidget(self._transcript_column_host, stretch=1)
        self._setup_transcript_timeline_shortcuts()

        # Bottom stack: composer stays at 800px max (independent of transcript layout toggle).
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
        self.web_btn.setToolTip(
            "Search the web for every message in this chat until you turn it off"
        )
        self.web_btn.toggled.connect(self._on_web_toggled)

        self.think_btn = QPushButton("Think")
        self.think_btn.setCheckable(True)
        self.think_btn.setProperty("class", "ThinkToggleButton")
        self.think_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.think_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.think_btn.setToolTip(
            "Show the model's reasoning process in responses (internal engine only)"
        )
        self.think_btn.toggled.connect(self._on_think_toggled)

        action_layout.addWidget(self.web_btn)
        action_layout.addWidget(self.think_btn)
        bottom_stack_layout.addLayout(action_layout)

        self.composer_chip_strip = ComposerContextChipStrip()
        self.composer_chip_strip.routing_removed.connect(self._on_composer_routing_removed)
        self.composer_chip_strip.skill_removed.connect(self._on_composer_skill_removed)
        bottom_stack_layout.addWidget(self.composer_chip_strip)

        self.composer_recent_row = ComposerRecentMentionsRow()
        self.composer_recent_row.mention_clicked.connect(self._on_composer_recent_mention_clicked)
        bottom_stack_layout.addWidget(self.composer_recent_row)

        self.deep_research_progress_row = IngestProgressRow()
        self.deep_research_progress_row.hide()
        bottom_stack_layout.addWidget(self.deep_research_progress_row)

        # 3. Input Bar Area
        input_container = QFrame()
        input_container.setObjectName("ChatInputContainer")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(10, 5, 5, 5)
        input_layout.setSpacing(8)

        self.composer_attach_btn = QPushButton()
        self.composer_attach_btn.setObjectName("ComposerAttachButton")
        self.composer_attach_btn.setFixedSize(32, 32)
        self.composer_attach_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.composer_attach_btn.setToolTip(
            "Add file, tool, skill, or conversation (@)"
        )

        self.composer_voice_btn = QPushButton()
        self.composer_voice_btn.setObjectName("ComposerVoiceButton")
        self.composer_voice_btn.setFixedSize(32, 32)
        self.composer_voice_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.composer_voice_btn.setToolTip(
            "Speak your message (push-to-talk)"
        )

        self.composer_side_divider = QFrame()
        self.composer_side_divider.setObjectName("ComposerSideDivider")
        self.composer_side_divider.setFrameShape(QFrame.Shape.NoFrame)
        self.composer_side_divider.setFixedSize(16, 1)

        composer_side_col = QWidget()
        composer_side_layout = QVBoxLayout(composer_side_col)
        composer_side_layout.setContentsMargins(0, 0, 0, 0)
        composer_side_layout.setSpacing(2)
        composer_side_layout.addWidget(self.composer_attach_btn, 0, Qt.AlignmentFlag.AlignLeft)
        composer_side_layout.addWidget(
            self.composer_side_divider, 0, Qt.AlignmentFlag.AlignHCenter
        )
        composer_side_layout.addWidget(self.composer_voice_btn, 0, Qt.AlignmentFlag.AlignLeft)
        composer_side_layout.addStretch(1)

        self.text_input = ChatComposerEdit()
        self.text_input.setPlaceholderText(COMPOSER_IDLE_PLACEHOLDER)
        self.text_input.setObjectName("ChatTextInput")
        self.text_input.setToolTip(
            "Enter to send. Shift+Enter adds a line break. Type @ to attach tools, files, and skills."
        )
        
        self.send_btn = QPushButton()
        _init_theme = self._theme()
        self.send_btn.setIcon(
            themed_fa_icon("fa5s.paper-plane", accent_icon_color(_init_theme), 18)
        )
        self.send_btn.setFixedSize(35, 35)
        self.send_btn.setProperty("class", "SendButton")
        self.send_btn.setToolTip("Send message")

        input_layout.addWidget(composer_side_col)
        input_layout.addWidget(self.text_input, stretch=1)
        input_layout.addWidget(self.send_btn)
        bottom_stack_layout.addWidget(input_container)

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

        self.composer_attach_btn.clicked.connect(self._open_composer_palette_from_button)
        self.composer_voice_btn.clicked.connect(self._handle_manual_voice_input)
        self.send_btn.clicked.connect(self._handle_send_or_stop)
        self.text_input.submit_requested.connect(self._handle_text_submit)
        self.text_input.textChanged.connect(self._on_composer_body_changed)
        self.text_input.bind_mention_host(self)
        _composer_is_dark = self._theme().is_dark
        self._style_composer_side_buttons(_composer_is_dark)
        self._refresh_composer_chip_strip()

        self._chat_transcript_wallpaper_host = TranscriptWallpaperHost(
            SURFACE_CHAT_TRANSCRIPT,
            mainstage_content,
            parent=frame,
        )
        self._refresh_readability_toolbar(_composer_is_dark)
        self._apply_layout_mode()
        self._refresh_transcript_wallpaper()
        outer_layout.addWidget(self._chat_transcript_wallpaper_host, stretch=1)

        return frame

    # --------------------------------------------------------- #
    #  UI UPDATE RECEIVERS (The Magic Happens Here)             #
    # --------------------------------------------------------- #

    def log_user_message(self, text: str, *, pending_assistant: bool = False) -> None:
        self._clear_placeholders()
        self._flush_agent_markdown_coalesce_immediate(finalize=True)
        # New user turn: drop stale assistant pointer so Turn N+1 tools cannot overwrite Turn N bubbles.
        self._user_turn_id += 1
        self.current_agent_msg = None
        self._pending_ttft_ms = None

        parsed = draft_from_text(text)
        display_body = (parsed.body or "").strip()
        if not display_body:
            display_body = strip_all_composer_tokens_for_display(text)

        bubble = UserBubbleFrame()
        bubble.setObjectName("UserBubble")
        # Preferred width: short prompts shrink to content; cap still from MessageWrapper.
        bubble.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        bubble.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(16, 12, 16, 12)
        bubble_layout.setSpacing(8)

        if parsed.routing or parsed.skills:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            chip_row = ComposerContextChipStrip(bubble)
            chip_row.set_draft(parsed, editable=False, compact=True)
            chip_row.apply_theme(is_dark)
            bubble_layout.addWidget(chip_row)

        if display_body:
            lbl = ChatUserBubble(display_body)
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._style_user_bubble(bubble, lbl)
            bubble_layout.addWidget(lbl)
        elif parsed.routing or parsed.skills:
            bubble.setMinimumHeight(36)
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            bg = self._user_bubble_frame_bg(is_dark)
            bubble.setStyleSheet(f"background-color: {bg}; border-radius: 18px;")

        wrapper = MessageWrapper(bubble, is_user=True)
        self._register_reader_focus_tracking(wrapper)
        self.transcript_layout.addWidget(wrapper)
        self._transcript_waypoints.append(
            _TranscriptWaypointRecord(
                wrapper=wrapper,
                label=truncate_waypoint_label(display_body or text),
            )
        )
        self._schedule_transcript_timeline_refresh()
        
        self._is_agent_typing = False
        self._scroll_to_bottom()
        if self._focus_mode_enabled:
            self._apply_reader_focus_opacity()

        if pending_assistant:
            self._show_agent_typing_row()
        self._refresh_conversation_action_buttons()

    def _show_agent_typing_row(self) -> None:
        """Assistant row with animated dots until the first streamed token arrives."""
        self._hide_agent_typing_row()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        bubble = QFrame()
        bubble.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        bubble.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        bl = QVBoxLayout(bubble)
        bl.setContentsMargins(8, 10, 16, 8)
        bl.setSpacing(0)
        indicator = TypingIndicatorWidget(
            mode=TypingIndicatorMode.FADE,
            dot_count=3,
            fixed_width=52,
            fixed_height=24,
        )
        indicator.set_dark_theme(is_dark)
        bl.addWidget(indicator, 0, Qt.AlignmentFlag.AlignLeft)
        indicator.start()
        wrap = MessageWrapper(bubble, is_user=False)
        self._register_reader_focus_tracking(wrap)
        self._agent_typing_wrapper = wrap
        self.transcript_layout.addWidget(wrap)
        self._scroll_to_bottom()

    def _hide_agent_typing_row(self) -> None:
        w = self._agent_typing_wrapper
        if w is None:
            return
        self._agent_typing_wrapper = None
        for ind in w.findChildren(TypingIndicatorWidget):
            ind.stop()
        self.transcript_layout.removeWidget(w)
        w.deleteLater()

    def _flush_agent_markdown_coalesce_immediate(self, *, finalize: bool = False) -> None:
        if getattr(self, "_agent_md_coalesce_timer", None) is not None:
            self._agent_md_coalesce_timer.stop()
        self._flush_coalesced_agent_markdown(finalize=finalize)

    def _schedule_coalesced_agent_markdown(self) -> None:
        self._agent_md_coalesce_timer.start(48)

    def _prepare_agent_markdown_source(
        self,
        buf: str,
        *,
        finalize: bool,
        valid_ids: set[str] | None = None,
    ) -> str:
        prepared = _prepare_stream_for_qt_citation_links(buf)

        def _repl(match: re.Match[str]) -> str:
            return _markdown_cite_link_replacement(match, valid_ids=valid_ids)

        rich_text = normalize_inline_markdown_structure(
            CITATION_TOKEN_RE.sub(_repl, prepared)
        )
        if finalize or not rich_text:
            return rich_text
        stable, tail = split_stream_markdown_buffer(rich_text)
        if not tail:
            return rich_text
        return compose_streaming_markdown(stable, tail)

    def _sanitize_agent_stream_text(self, text: str) -> str:
        from core.output_artifact_strip import strip_output_artifacts

        return strip_output_artifacts(
            text or "",
            harmony_active=self._harmony_output_cleanup_active(),
            reasoning_family=self._reasoning_family_harmony_leak_strip_active(),
        )

    def _flush_coalesced_agent_markdown(self, *, finalize: bool = False) -> None:
        cur = getattr(self, "current_agent_msg", None)
        if cur is None:
            return
        raw = getattr(self, "_agent_text_buffer", "") or ""
        buf = self._sanitize_agent_stream_text(raw)
        if finalize:
            buf, help_actions = parse_help_action_blocks(buf)
            if cur is not None:
                cur._help_action_chips = help_actions
                self._sync_help_action_chips(cur)
            self._agent_text_buffer = buf
        elif buf != raw:
            self._agent_text_buffer = buf
        is_dark = True
        if self.window() and hasattr(self.window(), "_is_dark_theme"):
            is_dark = self.window()._is_dark_theme
        cite_sources = getattr(cur, "_citation_sources", None) or []
        valid_ids = valid_source_ids(cite_sources)
        rich_text = self._prepare_agent_markdown_source(
            buf,
            finalize=finalize,
            valid_ids=valid_ids,
        )
        follow_stream_tail = self._is_transcript_scrolled_to_bottom()
        streaming = bool(getattr(self, "_llm_in_progress", False)) and not finalize
        try:
            cur.set_agent_markdown(
                rich_text,
                is_dark=is_dark,
                document_stylesheet=self._agent_markdown_stylesheet(is_dark),
                line_height_percent=self._line_height_proportional_percent(),
                justify_transcript=(self._transcript_alignment == ALIGN_JUSTIFY),
                streaming=streaming,
            )
        except RuntimeError:
            return
        cur.updateGeometry()
        parent = cur.parentWidget()
        if isinstance(parent, AgentMessageContainer):
            parent._sync_actions_bar_width()
        if follow_stream_tail:
            self._scroll_to_bottom()
        self._schedule_transcript_timeline_refresh()
        if finalize:
            self._sync_agent_export_button(cur)
        if self._focus_mode_enabled:
            self._apply_reader_focus_opacity()

    def log_agent_token(
        self,
        token: str,
        *,
        citation_sources=_UNSET_SOURCES,
        evidence_transparency=None,
    ) -> None:
        token = self._sanitize_agent_stream_text(str(token or ""))
        if not token:
            return
        self._hide_agent_typing_row()
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

            self.agent_msg_container = AgentMessageContainer()

            self.current_agent_msg = AgentMessageLabel()
            self.current_agent_msg.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred,
            )
            self.current_agent_msg._assistant_turn_id = self._user_turn_id
            self.current_agent_msg.attach_citation_handling(self)
            self._style_agent_message_shell(self.current_agent_msg)

            # Per-bubble citation context (survives new turns and session reloads)
            if citation_sources is not _UNSET_SOURCES:
                self.current_agent_msg._citation_sources = _snapshot_citation_sources(citation_sources)
            else:
                self._attach_pending_citation_sources(self.current_agent_msg)
            if evidence_transparency:
                self.current_agent_msg._evidence_transparency = copy.deepcopy(
                    evidence_transparency
                )
            else:
                self._attach_pending_evidence_transparency(self.current_agent_msg)

            self.agent_msg_container.attach_agent(self.current_agent_msg)
            self._add_agent_copy_button(self.agent_msg_container, self.current_agent_msg)
            self._style_agent_message_container(self.agent_msg_container)

            wrapper = MessageWrapper(self.agent_msg_container, is_user=False)
            self._register_reader_focus_tracking(wrapper)
            self.transcript_layout.addWidget(wrapper)
            self._sync_agent_actions_bar_widths()
            
            self._agent_text_buffer = ""
            self._is_agent_typing = True

        self._agent_text_buffer += token

        # Hybrid streaming markdown: stable prefix is parsed as Markdown; the live tail is
        # escaped so partial ** / list / table syntax renders literally until complete.
        self._schedule_coalesced_agent_markdown()
        self._refresh_conversation_action_buttons()

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
            for lbl in row.findChildren(ChatUserBubble):
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
        if hasattr(self, "_agent_md_coalesce_timer") and self._agent_md_coalesce_timer is not None:
            self._agent_md_coalesce_timer.stop()
        self._flush_coalesced_agent_markdown()
        self.current_agent_msg = None
        self._pending_citation_sources = None
        self._agent_text_buffer = ""
        self._hide_agent_typing_row()
        self._transcript_waypoints.clear()

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
        self._schedule_transcript_timeline_refresh()
        self._refresh_conversation_action_buttons()

    # --------------------------------------------------------- #
    #  INTERACTION & LOGIC                                      #
    # --------------------------------------------------------- #

    def _web_toggle_active(self) -> bool:
        return bool(getattr(self, "web_btn", None) and self.web_btn.isChecked())

    def _composer_action_icon_color(self, is_dark: bool) -> str:
        return self._theme(is_dark).color(ACCENT_ICON)

    def _composer_action_hover_bg(self, is_dark: bool) -> str:
        return self._theme(is_dark).surface_hover

    def _composer_side_divider_color(self, is_dark: bool) -> str:
        theme = self._theme(is_dark)
        return theme.border_subtle if theme.is_dark else theme.border

    def _style_composer_side_buttons(self, is_dark: bool) -> None:
        theme = self._theme(is_dark)
        icon_color = accent_icon_color(theme)
        button_qss = theme.style(COMPOSER_SIDE_BUTTON)
        if hasattr(self, "composer_attach_btn"):
            self.composer_attach_btn.setIcon(themed_fa_icon("fa5s.at", icon_color, 16))
            self.composer_attach_btn.setIconSize(QSize(16, 16))
            self.composer_attach_btn.setStyleSheet(button_qss)
        if hasattr(self, "composer_voice_btn"):
            self.composer_voice_btn.setIcon(
                themed_fa_icon("fa5s.microphone", icon_color, 16)
            )
            self.composer_voice_btn.setIconSize(QSize(16, 16))
            self.composer_voice_btn.setStyleSheet(button_qss)
        if hasattr(self, "composer_side_divider"):
            self.composer_side_divider.setStyleSheet(theme.style(COMPOSER_SIDE_DIVIDER))

    def _style_composer_attach_button(self, is_dark: bool) -> None:
        self._style_composer_side_buttons(is_dark)

    def _notify_composer_one_source_limit(self) -> None:
        """Show a short in-app toast; bypasses notification policy so it appears while typing."""
        self._notify_composer_toast(composer_one_source_limit_request())

    def _notify_composer_toast(self, request) -> None:
        nc = getattr(self.window(), "notification_center", None)
        if nc is None:
            return
        nc.show_notification(request)

    def _refresh_composer_chip_strip(self) -> None:
        if not hasattr(self, "composer_chip_strip"):
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self.composer_chip_strip.set_draft(self._composer_draft, editable=True)
        self.composer_chip_strip.apply_theme(is_dark)
        self._refresh_composer_recent_mentions()

    def _reset_composer_draft(self) -> None:
        self._composer_draft = ComposerDraft()
        if hasattr(self, "text_input"):
            self.text_input.blockSignals(True)
            self.text_input.clear()
            self.text_input.blockSignals(False)
            if hasattr(self.text_input, "_schedule_height_sync"):
                self.text_input._schedule_height_sync()
        self._refresh_composer_chip_strip()

    def _apply_composer_prefill(self, text: str) -> None:
        prefill = (text or "").strip()
        if not prefill:
            self._reset_composer_draft()
            return
        lifted = draft_from_text(prefill)
        self._composer_draft = ComposerDraft(body=lifted.body, skills=list(lifted.skills))
        reject_reason = None
        for att in lifted.routing:
            updated, _added, reason = add_routing_attachment(self._composer_draft, att)
            if reason == ROUTING_REJECT_ONE_SOURCE:
                reject_reason = reason
                break
            self._composer_draft = updated
        if reject_reason == ROUTING_REJECT_ONE_SOURCE:
            self._notify_composer_one_source_limit()
        if hasattr(self, "text_input"):
            self.text_input.blockSignals(True)
            self.text_input.setPlainText(lifted.body)
            self.text_input.blockSignals(False)
            if hasattr(self.text_input, "_schedule_height_sync"):
                self.text_input._schedule_height_sync()
        self._refresh_composer_chip_strip()

    def add_composer_attachment(self, attachment) -> None:
        if attachment.kind == "file" and not validate_file_token(attachment.id):
            return
        updated, _added, reject_reason = add_routing_attachment(
            self._composer_draft,
            attachment,
            skip_internet_when_web_active=self._web_toggle_active(),
        )
        if reject_reason == ROUTING_REJECT_ONE_SOURCE:
            self._notify_composer_one_source_limit()
            return
        self._composer_draft = updated
        record_recent_attachment(attachment)
        self._refresh_composer_chip_strip()
        if hasattr(self, "text_input"):
            self.text_input.setFocus()

    def add_composer_skill(self, mention) -> None:
        updated, _added = add_skill(self._composer_draft, mention)
        self._composer_draft = updated
        record_recent_skill(mention)
        self._refresh_composer_chip_strip()
        if hasattr(self, "text_input"):
            self.text_input.setFocus()

    def _refresh_composer_recent_mentions(self) -> None:
        row = getattr(self, "composer_recent_row", None)
        if row is None:
            return
        if not self._composer_draft.is_empty():
            row.hide()
            return
        entries, using_defaults = composer_hint_entries()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        row.set_entries(entries, using_defaults=using_defaults)
        row.apply_theme(is_dark)

    def _on_composer_recent_mention_clicked(self, mention: RecentMention) -> None:
        resolved = resolve_recent_mention(mention)
        if isinstance(resolved, ComposerSkillMention):
            self.add_composer_skill(resolved)
            return
        if resolved is not None:
            self.add_composer_attachment(resolved)

    def _on_composer_routing_removed(self, index: int) -> None:
        self._composer_draft = remove_routing_at(self._composer_draft, index)
        self._refresh_composer_chip_strip()

    def _on_composer_skill_removed(self, index: int) -> None:
        self._composer_draft = remove_skill_at(self._composer_draft, index)
        self._refresh_composer_chip_strip()

    def _on_composer_body_changed(self) -> None:
        text = self.text_input.toPlainText()
        if "@[" not in text:
            self._composer_draft.body = text
            self._refresh_composer_recent_mentions()
            return
        lifted = draft_from_text(text)
        merged, reject_reason = merge_drafts(
            self._composer_draft,
            lifted,
            skip_internet_when_web_active=self._web_toggle_active(),
        )
        self._composer_draft = merged
        if reject_reason == ROUTING_REJECT_ONE_SOURCE:
            self._notify_composer_one_source_limit()
        clean = lifted.body
        if clean != text:
            self.text_input.blockSignals(True)
            self.text_input.setPlainText(clean)
            cursor = self.text_input.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.text_input.setTextCursor(cursor)
            self.text_input.blockSignals(False)
            if hasattr(self.text_input, "_schedule_height_sync"):
                self.text_input._schedule_height_sync()
        self._composer_draft.body = clean
        self._refresh_composer_chip_strip()

    def _open_composer_palette_from_button(self) -> None:
        if not hasattr(self, "text_input"):
            return
        pos = self.composer_attach_btn.mapToGlobal(
            self.composer_attach_btn.rect().bottomLeft()
        )
        self.text_input.open_mention_palette(pos)
        self.text_input.setFocus()

    def _sync_telemetry_session_egress(self, session_id: str | None) -> None:
        win = self.window()
        if win is None or not hasattr(win, "peek_telemetry_view"):
            return
        tv = win.peek_telemetry_view()
        if tv is not None:
            tv.set_active_session_id(session_id)

    def _handle_text_submit(self):
        if self._is_stop_mode():
            self._request_stop()
            return
        self._composer_draft.body = self.text_input.toPlainText()
        if self._composer_draft.is_empty():
            return
        if self._composer_draft.routing_requires_body():
            self._notify_composer_toast(composer_prompt_required_request())
            return
        before_send = self._before_send_callback
        if callable(before_send) and not before_send():
            return
        for att in self._composer_draft.routing:
            record_recent_attachment(att)
        for skill in self._composer_draft.skills:
            record_recent_skill(skill)
        raw = serialize_draft(self._composer_draft)
        clean, attachments, enforced_skills = parse_composer_input(raw)
        routing = resolve_attachment_routing(attachments)
        if routing and routing.get("route") == "capability":
            from core.integrations.capability_invoke import parse_composer_capability_urn
            from core.integrations.capability_availability import resolve_capability_availability

            cap_urn = parse_composer_capability_urn(
                str(routing.get("capability_urn") or "")
            )
            if cap_urn is not None:
                availability = resolve_capability_availability(cap_urn)
                if not availability.available:
                    self._notify_composer_toast(
                        composer_capability_unavailable_request(availability.user_message)
                    )
                    return
            session_for_gate = self._ensure_active_session_for_send()
            next_turn_id = str(
                int(getattr(self.llm, "_routing_debug_turn_seq", 0)) + 1
            )
            from core.integrations.composer_capability_gate import (
                format_step_approval_message,
                pending_step_approvals,
            )
            from core.integrations.step_approval import step_approval_store

            pending_caps = pending_step_approvals(
                session_for_gate, next_turn_id, attachments
            )
            if pending_caps:
                win = self.window()
                is_dark = getattr(win, "_is_dark_theme", True)
                confirmed = PrestigeDialog(
                    win,
                    "Approve integration actions",
                    format_step_approval_message(pending_caps),
                    is_dark=is_dark,
                    confirm_text="Run for this message",
                    cancel_text="Cancel",
                ).exec()
                if not confirmed:
                    self.setFocus()
                    return
                step_approval_store.grant_many(
                    session_for_gate,
                    next_turn_id,
                    [item.urn for item in pending_caps],
                )
        if routing and routing.get("route") == "deep_research":
            self._reset_composer_draft()
            self._submit_deep_research(
                raw=raw,
                query=clean,
                enforced_skills=enforced_skills,
                routing=routing,
            )
            return
        self._reset_composer_draft()
        self._llm_in_progress = True
        self._awaiting_tts_end = False
        self._tts_playing = False
        self.set_input_enabled(False)
        self._refresh_send_stop_button()

        self.log_user_message(raw, pending_assistant=True)

        if not hasattr(self, 'active_session_id'):
            recent_sessions = self.db.get_recent_sessions(limit=1)
            if recent_sessions:
                self.active_session_id = recent_sessions[0]['id']
            else:
                folder_id = self._active_folder_id or self.db.get_main_conversation_folder_id()
                self.active_session_id = self.db.create_session(
                    "Text Conversation", folder_id=folder_id
                )

        if self.llm:
            from core.input_source import INPUT_SOURCE_TEXT

            prompt = clean if clean else raw
            self.llm.generate_response(
                prompt,
                self.active_session_id,
                attachments=attachments,
                enforced_skills=enforced_skills,
                persist_content=raw,
                input_source=INPUT_SOURCE_TEXT,
            )
            self._sync_telemetry_session_egress(str(self.active_session_id))

    def _ensure_active_session_for_send(self) -> str:
        if not hasattr(self, "active_session_id") or not self.active_session_id:
            recent_sessions = self.db.get_recent_sessions(limit=1)
            if recent_sessions:
                self.active_session_id = recent_sessions[0]["id"]
            else:
                folder_id = self._active_folder_id or self.db.get_main_conversation_folder_id()
                self.active_session_id = self.db.create_session(
                    "Text Conversation", folder_id=folder_id
                )
        return str(self.active_session_id)

    def _submit_deep_research(
        self,
        *,
        raw: str,
        query: str,
        enforced_skills: tuple[str, ...],
        routing: dict | None = None,
    ) -> None:
        _ = enforced_skills
        routing = routing or {}
        session_id = self._ensure_active_session_for_send()
        self.db.add_message(session_id, "user", raw)
        self.log_user_message(raw, pending_assistant=False)

        worker = (self.workers or {}).get("deep_research")
        if worker is None:
            self._notify_composer_toast(deep_research_unavailable_request())
            return

        force_thorough = bool(routing.get("deep_research_force_thorough"))
        if force_thorough:
            from core.deep_research_pro_features import user_has_pro_thorough

            if not user_has_pro_thorough():
                self._notify_composer_toast(deep_research_pro_downgrade_request())

        from core.deep_research_pro_features import resolve_deep_research_profile

        resolved_profile = resolve_deep_research_profile(force_thorough=force_thorough)
        profile_label = resolved_profile.spec.label

        request_id = str(uuid.uuid4())
        self._active_deep_research_request_id = request_id
        self._deep_research_session_id = session_id
        self._deep_research_in_progress = True
        self._begin_deep_research_progress(f"Starting {profile_label.lower()} deep research…")
        self._refresh_send_stop_button()
        worker.enqueue(
            {
                "request_id": request_id,
                "session_id": session_id,
                "query": query,
                "knowledge_service": SERVICE_SCIENTIFIC_EVIDENCE,
                "deep_research_force_thorough": force_thorough,
            }
        )

    def _begin_deep_research_progress(self, detail: str = "") -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self.deep_research_progress_row.apply_theme(is_dark)
        self.deep_research_progress_row.begin(detail=detail)
        self.deep_research_progress_row.show()

    def _hide_deep_research_progress(self) -> None:
        if hasattr(self, "deep_research_progress_row"):
            self.deep_research_progress_row.finish()
            self.deep_research_progress_row.hide()

    def _deep_research_progress_detail(self, payload: dict) -> str:
        message = str(payload.get("message") or "").strip()
        phase = str(payload.get("phase") or "")
        sources_found = int(payload.get("sources_found") or 0)
        if phase == "retrieving":
            idx = max(0, int(payload.get("sub_query_index") or 0))
            total = max(1, int(payload.get("sub_query_total") or 1))
            if message:
                detail = message
            else:
                detail = f"Retrieving evidence ({idx}/{total})…"
            if sources_found > 0:
                detail = f"{detail} · {sources_found} source(s) found"
            return detail
        if message:
            return message
        labels = {
            "decomposing": "Planning sub-queries…",
            "merging": "Merging and de-duplicating sources…",
            "reporting": "Building bibliography…",
            "synthesizing": "Synthesizing findings from evidence…",
        }
        return labels.get(phase, "Deep research in progress…")

    def on_deep_research_progress(self, payload: dict) -> None:
        request_id = str(payload.get("request_id") or "")
        if request_id != getattr(self, "_active_deep_research_request_id", None):
            return
        tracked_session = str(getattr(self, "_deep_research_session_id", "") or "")
        active = str(getattr(self, "active_session_id", "") or "")
        if tracked_session and active and tracked_session != active:
            return
        detail = self._deep_research_progress_detail(payload)
        percent = deep_research_progress_percent(payload)
        if not self.deep_research_progress_row.isVisible():
            self._begin_deep_research_progress(detail=detail)
        self.deep_research_progress_row.update_progress(percent, detail=detail)

    def on_deep_research_finished(self, payload: dict) -> None:
        request_id = str(payload.get("request_id") or "")
        session_id = str(payload.get("session_id") or "")
        if request_id == getattr(self, "_active_deep_research_request_id", None):
            self._hide_deep_research_progress()
            self._active_deep_research_request_id = None
            self._deep_research_session_id = None
            self._deep_research_in_progress = False
            self._refresh_send_stop_button()

        status = str(payload.get("status") or "")
        if status == "cancelled":
            return
        report = str(payload.get("report_markdown") or "").strip()
        if status == "error":
            report = f"**Deep research failed:** {payload.get('error', 'unknown error')}"
        elif status == "no_results" and not report:
            report = "Deep research completed but found no matching sources."

        sources = list(payload.get("sources") or [])
        transparency = dict(payload.get("evidence_transparency") or {})
        bundle_id = payload.get("bundle_id")
        bundle_id_str = str(bundle_id) if bundle_id else None

        if session_id and report:
            from core.knowledge.ui_sources_payload import encode_sources_payload

            src_payload = encode_sources_payload(
                sources,
                transparency=transparency or None,
            )
            self.db.add_message(
                session_id,
                "assistant",
                report,
                sources_json=src_payload,
                evidence_bundle_id=bundle_id_str,
            )
            bundle_dict = payload.get("bundle_dict")
            if isinstance(bundle_dict, dict) and bundle_dict:
                from core.knowledge.graph.bundle_codec import bundle_from_dict
                from core.knowledge.graph.service import record_bundle_in_session_graph

                bundle = bundle_from_dict(bundle_dict)
                record_bundle_in_session_graph(
                    self.db,
                    session_id=session_id,
                    bundle=bundle,
                )

        active = str(getattr(self, "active_session_id", "") or "")
        if active and session_id == active and report:
            self.log_agent_token(
                report,
                citation_sources=sources,
                evidence_transparency=transparency or None,
            )
            self._flush_agent_markdown_coalesce_immediate(finalize=True)
            self._is_agent_typing = False
            self.current_agent_msg = None

    def update_stt_latency(self, ms: float) -> None:
        self._stt_ms_for_turn = self._user_turn_id + 1
        self._stt_ms_value = float(ms)
        agent = getattr(self, "current_agent_msg", None)
        if agent is not None:
            self._apply_pending_stt_to_agent(agent)

    def update_ttft_latency(self, ms: float) -> None:
        self._pending_ttft_ms = float(ms)
        agent = getattr(self, "current_agent_msg", None)
        if agent is not None:
            self._apply_pending_ttft_to_agent(agent)

    def update_tts_latency(self, ms: float) -> None:
        agent = getattr(self, "current_agent_msg", None)
        if agent is not None:
            self._set_agent_telemetry_metric(agent, "tts", float(ms))

    def update_tps(self, tps: float) -> None:
        agent = getattr(self, "current_agent_msg", None)
        if agent is not None:
            value = float(tps) if tps and tps > 0 else None
            self._set_agent_telemetry_metric(agent, "tps", value)

    def _on_history_search_changed(self, _text: str) -> None:
        self._history_search_timer.stop()
        self._history_search_timer.start(280)

    def _set_active_folder_id(self, folder_id: str) -> None:
        self._active_folder_id = folder_id
        self._update_row_colors()

    def _sidebar_active_folder_id(self) -> str | None:
        if not hasattr(self, "history_list"):
            return None
        return getattr(self, "_active_folder_id", None) or self.db.get_main_conversation_folder_id()

    def _update_row_colors(self):
        """Row title colors + action icons (QSS cannot target setItemWidget children)."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        target_list = getattr(self, "history_list", None)
        apply_sidebar_row_theme(
            target_list,
            is_dark=is_dark,
            theme=theme,
            active_folder_id=self._sidebar_active_folder_id(),
        )

    def _on_history_item_clicked(self, item) -> None:
        if self._folder_controller and self._folder_controller.handle_item_clicked(item):
            return
        self._load_selected_chat(item)

    def _on_history_item_double_clicked(self, item) -> None:
        if self._folder_controller:
            self._folder_controller.handle_item_double_clicked(item)

    def _reload_history_sidebar(self) -> None:
        """Rebuild sidebar: search → flat list; else folder-grouped browse."""
        if not self._folder_controller:
            return
        q = self.search_bar.text().strip() if getattr(self, "search_bar", None) else ""
        if q:
            try:
                sessions = self.db.get_sessions_for_sidebar_search(q, limit=200)
            except Exception as e:
                logger.exception("Sidebar history search failed: %s", e)
                sessions = []
            self._folder_controller.reload_search_mode(sessions)
        else:
            self._folder_controller.reload_browse_mode()
        self._update_row_colors()

    def _append_history_session_row(self, session: dict, indent_left: int = FOLDER_ROW_MARGIN_LEFT) -> None:
        from PyQt6.QtWidgets import QListWidgetItem, QWidget, QHBoxLayout, QLabel, QPushButton, QMenu
        from PyQt6.QtCore import QSize
        import qtawesome as qta

        is_dark = True
        main_win = self.window()
        if main_win and hasattr(main_win, "_is_dark_theme"):
            is_dark = main_win._is_dark_theme

        theme = self._theme(is_dark)
        icon_color = sidebar_row_action_icon_color(theme)

        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, session["id"])
        item.setData(SIDEBAR_ROW_KIND_ROLE, ROW_KIND_SESSION)
        item.setData(SIDEBAR_ROW_PAYLOAD_ROLE, session)

        row_widget = QWidget()
        row_widget.setObjectName("HistoryRowWidget")
        row_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(indent_left, 0, 10, 0)
        row_layout.setSpacing(10)

        title_lbl = QLabel(session["title"])
        title_lbl.setObjectName("HistoryRowTitle")

        opts_btn = QPushButton()
        opts_btn.setObjectName("HistoryOptionsBtn")
        opts_btn.setFixedSize(28, 28)
        opts_btn.setIcon(themed_fa_icon("fa5s.ellipsis-v", icon_color, 16))
        opts_btn.setIconSize(QSize(16, 16))
        opts_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        opts_btn.setStyleSheet(
            "QPushButton::menu-indicator { image: none; width: 0px; } "
            "QPushButton { border: none; background: transparent; padding: 0px; }"
        )
        opts_btn.setToolTip("Conversation actions")

        menu = QMenu(opts_btn)
        if hasattr(self, "_apply_menu_theme"):
            self._apply_menu_theme(menu, is_dark)
        if self._folder_controller:
            self._folder_controller.register_menu(menu)

        rename_action = menu.addAction(
            themed_fa_icon("fa5s.edit", theme.color(LINK_ICON), 16), "Rename Chat"
        )
        rename_action.triggered.connect(
            lambda _, s_id=session["id"], old_t=session["title"]: self._trigger_rename_chat(
                s_id, old_t
            )
        )

        if self._folder_controller:
            session_folder_id = session.get("folder_id") or self.db.get_main_conversation_folder_id()
            self._folder_controller.build_move_submenu_for_item(
                menu,
                session_folder_id,
                lambda folder_id, s_id=session["id"]: self._move_session_to_folder(
                    s_id, folder_id
                ),
            )

        export_action = menu.addAction(
            themed_fa_icon("fa5s.file-export", theme.color(LINK_ICON), 16), "Export"
        )
        export_action.triggered.connect(
            lambda _, s_id=session["id"], title=session["title"]: self._trigger_export_chat(
                s_id, title
            )
        )

        menu.addSeparator()

        delete_action = menu.addAction(
            themed_fa_icon("fa5s.trash-alt", theme.color(DANGER_ICON), 16), "Delete Chat"
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

    def _move_session_to_folder(self, session_id: str, folder_id: str) -> None:
        if self.db.move_session_to_folder(session_id, folder_id):
            self._refresh_history_list()

    def _transcript_has_messages(self) -> bool:
        for w in self._iter_transcript_widgets():
            if isinstance(w, MessageWrapper):
                return True
        return False

    def _refresh_conversation_action_buttons(self) -> None:
        enabled = self._transcript_has_messages()
        for attr in ("conversation_download_btn", "conversation_copy_btn"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(enabled)

    def _active_conversation_export_target(self) -> tuple[str, str] | None:
        session_id = getattr(self, "active_session_id", None)
        if not session_id:
            return None
        session = self.db.get_session(session_id)
        if session is None:
            return None
        return session_id, str(session.get("title") or "Untitled")

    def _export_active_conversation(self) -> None:
        if not self._transcript_has_messages():
            return
        target = self._active_conversation_export_target()
        if not target:
            return
        session_id, title = target
        self._trigger_export_chat(session_id, title)

    def _copy_active_conversation_to_clipboard(self) -> None:
        if not self._transcript_has_messages():
            return
        target = self._active_conversation_export_target()
        if not target:
            return
        session_id, title = target
        messages = self.db.get_session_history(session_id)
        body = format_conversation_markdown(title, messages)
        QApplication.clipboard().setText(body)

    def _trigger_export_chat(self, session_id: str, title: str) -> None:
        default_name = f"{sanitize_export_filename(title)}.md"
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Export Conversation",
            default_name,
            "Markdown (*.md)",
        )
        if not dest:
            return
        try:
            if export_conversation_markdown(self.db, session_id, Path(dest)):
                logger.info("Exported conversation %s to %s", session_id, dest)
            else:
                logger.warning("Export failed: session %s not found", session_id)
        except OSError as e:
            logger.exception("Failed to export conversation %s: %s", session_id, e)

    def _trigger_export_folder(self, folder_id: str, folder_name: str) -> None:
        default_name = f"{sanitize_export_filename(folder_name)}.zip"
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Export Folder",
            default_name,
            "ZIP archive (*.zip)",
        )
        if not dest:
            return
        try:
            count = export_folder_zip(self.db, folder_id, Path(dest))
            logger.info("Exported %d conversation(s) from folder %s to %s", count, folder_id, dest)
        except OSError as e:
            logger.exception("Failed to export folder %s: %s", folder_id, e)

    def _refresh_history_list(self):
        """Runs cleanup, updates count, rebuilds list (respects search box)."""
        current_active = getattr(self, "active_session_id", None)
        if hasattr(self.db, "cleanup_empty_sessions"):
            self.db.cleanup_empty_sessions(current_active)

        self._session_count = self.db.get_session_count()
        self._reload_history_sidebar()

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
        folder_id = self._active_folder_id or self.db.get_main_conversation_folder_id()
        self.active_session_id = self.db.create_session(
            "New Conversation", folder_id=folder_id
        )
        self._reset_web_toggle()
        self._reset_composer_draft()
        self._notify_llm_active_session_changed()
        self._clear_transcript()

        self.placeholder_lbl = self._make_transcript_placeholder_label(
            NEW_CHAT_TRANSCRIPT_HINT
        )
        self.transcript_layout.addWidget(self.placeholder_lbl)
        self._refresh_ancillary_transcript_labels()

        self._is_agent_typing = False
        self._refresh_history_list()
        self._scroll_to_bottom()

    def start_new_chat_with_composer_prefill(self, text: str) -> None:
        """Open a fresh session and seed the composer (e.g. Library → chat with document)."""
        self._start_new_chat()
        if not hasattr(self, "text_input"):
            return
        self._apply_composer_prefill(text)
        self.text_input.setFocus()

    def _load_selected_chat(self, item):
        from PyQt6.QtCore import Qt
        session_id = item.data(Qt.ItemDataRole.UserRole)
        self.active_session_id = session_id
        self._sync_telemetry_session_egress(str(session_id))
        self._reset_composer_draft()
        self._notify_llm_active_session_changed()
        if hasattr(self, "text_input") and hasattr(self.text_input, "_sync_mention_context"):
            self.text_input._sync_mention_context()

        self._clear_transcript()
        self._is_agent_typing = False

        history = self.db.get_session_history(session_id)
        if not history:
            self.placeholder_lbl = self._make_transcript_placeholder_label(
                EMPTY_SESSION_TRANSCRIPT_HINT
            )
            self.transcript_layout.addWidget(self.placeholder_lbl)
            self._flush_pending_stream_for_active_session()
            self._refresh_ancillary_transcript_labels()
            self._scroll_to_bottom()
            self._refresh_conversation_action_buttons()
            return

        for msg in history:
            if msg["role"] == "user":
                self.log_user_message(msg["content"])
            elif msg["role"] == "assistant":
                self.log_agent_token(
                    msg["content"],
                    citation_sources=msg.get("sources"),
                    evidence_transparency=msg.get("evidence_transparency"),
                )
                self._is_agent_typing = False

        # Reconcile stream chunks received while this session was not visible.
        self._flush_pending_stream_for_active_session()
        self._flush_agent_markdown_coalesce_immediate(finalize=True)

        self._refresh_all_readability()
        self._scroll_to_bottom()
        self._schedule_transcript_timeline_refresh()
        self._refresh_conversation_action_buttons()

    def _apply_menu_theme(self, menu, is_dark: bool):
        """Standardizes the menu appearance to match the Prestige theme."""
        apply_prestige_kebab_menu_theme(menu, is_dark)

    def refresh_menu_themes(self, is_dark: bool):
        """Updates all existing kebab menus in the history list."""
        if self._folder_controller:
            self._folder_controller.refresh_menu_themes(is_dark)
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            widget = self.history_list.itemWidget(item)
            if widget:
                btn = widget.findChild(QPushButton, "HistoryOptionsBtn")
                if btn and btn.menu():
                    self._apply_menu_theme(btn.menu(), is_dark)

    def _on_think_toggled(self, checked: bool) -> None:
        """Persist user preference; Think state re-syncs from ExecutionPolicy via refresh_think_toggle."""
        if not hasattr(self, "think_btn") or not self.think_btn.isEnabled():
            return
        set_native_reasoning_display_enabled(bool(checked))
        self.refresh_think_toggle()

    def _harmony_output_cleanup_active(self) -> bool:
        eng = self.workers.get("native_engine") if self.workers else None
        if eng is None:
            return False
        try:
            snap = eng.get_model_reasoning_telemetry() or {}
            return bool(snap.get("harmony_model_active"))
        except Exception:
            return False

    def _reasoning_family_harmony_leak_strip_active(self) -> bool:
        eng = self.workers.get("native_engine") if self.workers else None
        if eng is None:
            return False
        try:
            from core.qwen3_thinking_policy import (
                is_reasoning_family_harmony_leak_strip_candidate,
            )

            snap = eng.get_model_reasoning_telemetry() or {}
            if not bool(snap.get("loaded")):
                return False
            name = str(snap.get("model_name", "") or "")
            base = str(snap.get("model_basename", "") or "")
            path = str(getattr(eng, "_model_path", "") or "")
            ident_name = f"{name} {base}".strip()
            if is_reasoning_family_harmony_leak_strip_candidate(
                model_path=path,
                model_name=ident_name,
            ):
                return True
            return bool(snap.get("supports_thinking_tokens"))
        except Exception:
            return False

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

    def _on_web_toggled(self, checked: bool) -> None:
        if self.llm and hasattr(self.llm, "set_force_web_enabled"):
            self.llm.set_force_web_enabled(checked)
        self._apply_action_toggle_styles()
        win = self.window()
        if win is not None and hasattr(win, "refresh_web_indicator"):
            win.refresh_web_indicator()

    def _reset_web_toggle(self) -> None:
        """Turn off sticky web search when starting a fresh chat session."""
        if not hasattr(self, "web_btn"):
            return
        self.web_btn.blockSignals(True)
        try:
            self.web_btn.setChecked(False)
        finally:
            self.web_btn.blockSignals(False)
        if self.llm and hasattr(self.llm, "set_force_web_enabled"):
            self.llm.set_force_web_enabled(False)
        self._apply_action_toggle_styles()
        win = self.window()
        if win is not None and hasattr(win, "refresh_web_indicator"):
            win.refresh_web_indicator()

    def _apply_action_toggle_styles(self) -> None:
        """Render Web/Think toggle buttons with active/inactive styles."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        self._apply_toggle_button_style(
            self.web_btn if hasattr(self, "web_btn") else None,
            theme=theme,
            active_bg=theme.link,
        )
        self._apply_toggle_button_style(
            self.think_btn if hasattr(self, "think_btn") else None,
            theme=theme,
            active_bg=theme.success,
        )

    def _apply_toggle_button_style(self, btn, *, theme, active_bg: str) -> None:
        if btn is None:
            return
        btn.setStyleSheet(
            theme.style(TOGGLE_BUTTON, checked=btn.isChecked(), active_bg=active_bg)
        )

    def refresh_button_themes(self, is_dark: bool):
        """Dynamically updates the colors of the New Chat and Send buttons."""
        if hasattr(self, "text_input") and hasattr(self.text_input, "apply_mention_theme"):
            self.text_input.apply_mention_theme(is_dark)
        theme = self._theme(is_dark)
        editable_field = theme.style(SETTINGS_LINE_EDIT)
        if hasattr(self, "text_input"):
            self.text_input.setStyleSheet(editable_field)
        if hasattr(self, "search_bar"):
            self.search_bar.setStyleSheet(editable_field)
        if hasattr(self, "composer_chip_strip"):
            self.composer_chip_strip.apply_theme(is_dark)
        if hasattr(self, "deep_research_progress_row"):
            self.deep_research_progress_row.apply_theme(is_dark)
        self._style_composer_side_buttons(is_dark)
        base_icon_color = accent_icon_color(theme)

        if hasattr(self, "new_chat_btn"):
            self.new_chat_btn.setIcon(themed_fa_icon("fa5s.plus", base_icon_color, 16))
            apply_ghost_icon_button_style(self.new_chat_btn, theme)
        if hasattr(self, "new_folder_btn"):
            self.new_folder_btn.setIcon(
                themed_fa_icon("fa5s.folder-plus", base_icon_color, 16)
            )
            apply_ghost_icon_button_style(self.new_folder_btn, theme)
        if hasattr(self, "sort_btn"):
            self.sort_btn.setIcon(themed_fa_icon("fa5s.sort", base_icon_color, 16))
            apply_ghost_icon_button_style(self.sort_btn, theme, hide_menu_indicator=True)

        if hasattr(self, "send_btn"):
            icon_name = "fa5s.stop" if self._is_stop_mode() else "fa5s.paper-plane"
            send_icon_color = (
                theme.color(DANGER_ICON)
                if self._is_stop_mode()
                else base_icon_color
            )
            self.send_btn.setIcon(themed_fa_icon(icon_name, send_icon_color, 18))
            self.send_btn.setToolTip(self._stop_button_tooltip())
            apply_ghost_icon_button_style(self.send_btn, theme, fixed_size=None)
        self._refresh_readability_toolbar(is_dark=is_dark)
        self._refresh_layout_mode_button(is_dark=is_dark)
        if hasattr(self, "_transcript_timeline_rail"):
            self._transcript_timeline_rail.apply_theme(is_dark)
            self._schedule_transcript_timeline_refresh()
        if hasattr(self, "font_minus_btn"):
            font_btn_style = readability_font_pair_stylesheet(
                is_dark=is_dark, theme=theme, button_px=_CHAT_UTILITY_BTN
            )
            self.font_minus_btn.setStyleSheet(font_btn_style)
            self.font_plus_btn.setStyleSheet(font_btn_style)
        self._apply_action_toggle_styles()
        self._apply_history_list_surface(is_dark)
        self._refresh_ancillary_transcript_labels()
        tw = self._agent_typing_wrapper
        if tw is not None:
            for ind in tw.findChildren(TypingIndicatorWidget):
                ind.set_dark_theme(is_dark)
        self._refresh_agent_copy_buttons(is_dark)
        self._refresh_transcript_wallpaper()

    def _apply_history_list_surface(self, is_dark: bool) -> None:
        """Sidebar list tint: QListWidget paints in an internal viewport — set palette on list + viewport."""
        bg = self._theme(is_dark).qcolor_role(LIST_SURFACE)
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
        QTimer.singleShot(0, self.focus_composer_if_ready)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_transcript_column_width_cap()
        self._schedule_transcript_timeline_refresh()

    def eventFilter(self, obj, event):
        """Native resize handling without fighting Qt's geometry engine."""
        if hasattr(self, "scroll_area") and event.type() == QEvent.Type.Resize:
            if obj is self.scroll_area or obj is self.scroll_area.viewport():
                self._sync_transcript_column_width_cap()
                self._schedule_transcript_timeline_refresh()
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

    def _schedule_transcript_timeline_refresh(self) -> None:
        timer = getattr(self, "_transcript_timeline_refresh_timer", None)
        if timer is None:
            return
        timer.start(0)

    def _transcript_container_height(self) -> int:
        container = getattr(self, "transcript_container", None)
        if container is None:
            return 0
        height = int(container.sizeHint().height())
        if height <= 0:
            height = int(container.height())
        return max(0, height)

    def _collect_transcript_timeline_entries(self) -> tuple[list[TranscriptWaypointEntry], list[int]]:
        container = getattr(self, "transcript_container", None)
        if container is None:
            return [], []
        entries: list[TranscriptWaypointEntry] = []
        waypoint_ys: list[int] = []
        for record in getattr(self, "_transcript_waypoints", ()):
            wrapper = record.wrapper
            if wrapper is None:
                continue
            try:
                y = wrapper.mapTo(container, wrapper.rect().topLeft()).y()
            except RuntimeError:
                continue
            entries.append(TranscriptWaypointEntry(y=int(y), label=record.label))
            waypoint_ys.append(int(y))
        return entries, waypoint_ys

    def _refresh_transcript_timeline_rail(self) -> None:
        rail = getattr(self, "_transcript_timeline_rail", None)
        scroll = getattr(self, "scroll_area", None)
        if rail is None or scroll is None:
            return

        container_h = self._transcript_container_height()
        viewport = scroll.viewport()
        viewport_h = int(viewport.height()) if viewport is not None else 0
        entries, waypoint_ys = self._collect_transcript_timeline_entries()
        show = transcript_timeline_should_show(
            container_h,
            viewport_h,
            waypoint_count=len(entries),
        )
        scroll_top = scroll.verticalScrollBar().value()
        active = compute_active_waypoint_index(scroll_top, waypoint_ys)

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        rail.apply_theme(is_dark)
        rail.set_geometry_from_container(
            entries,
            container_height=container_h,
            show=show,
        )
        rail.set_active_index(active)
        column_host = getattr(self, "_transcript_column_host", None)
        if column_host is not None:
            column_host.sync_geometry()
        self._sync_transcript_timeline_shortcuts_enabled()

    def _setup_transcript_timeline_shortcuts(self) -> None:
        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
        prev_sc = QShortcut(QKeySequence("Ctrl+Up"), self)
        prev_sc.setContext(ctx)
        prev_sc.setAutoRepeat(False)
        prev_sc.activated.connect(lambda: self._jump_transcript_waypoint(-1))
        next_sc = QShortcut(QKeySequence("Ctrl+Down"), self)
        next_sc.setContext(ctx)
        next_sc.setAutoRepeat(False)
        next_sc.activated.connect(lambda: self._jump_transcript_waypoint(1))
        self._transcript_timeline_prev_sc = prev_sc
        self._transcript_timeline_next_sc = next_sc
        self._sync_transcript_timeline_shortcuts_enabled()

    def _sync_transcript_timeline_shortcuts_enabled(self) -> None:
        rail = getattr(self, "_transcript_timeline_rail", None)
        enabled = rail is not None and rail.isVisible() and bool(
            getattr(self, "_transcript_waypoints", ())
        )
        for sc in (
            getattr(self, "_transcript_timeline_prev_sc", None),
            getattr(self, "_transcript_timeline_next_sc", None),
        ):
            if sc is not None:
                sc.setEnabled(enabled)

    def _active_transcript_waypoint_index(self) -> int:
        records = getattr(self, "_transcript_waypoints", [])
        if not records:
            return 0
        scroll = getattr(self, "scroll_area", None)
        if scroll is None:
            return 0
        _entries, waypoint_ys = self._collect_transcript_timeline_entries()
        if not waypoint_ys:
            return 0
        return compute_active_waypoint_index(
            scroll.verticalScrollBar().value(),
            waypoint_ys,
        )

    def _jump_transcript_waypoint(self, delta: int) -> None:
        records = getattr(self, "_transcript_waypoints", [])
        rail = getattr(self, "_transcript_timeline_rail", None)
        if not records or rail is None or not rail.isVisible() or delta == 0:
            return
        current = self._active_transcript_waypoint_index()
        target = max(0, min(len(records) - 1, current + int(delta)))
        self._scroll_to_transcript_waypoint_index(target, animated=True)

    def _scroll_target_for_waypoint_index(self, index: int) -> int | None:
        records = getattr(self, "_transcript_waypoints", [])
        scroll = getattr(self, "scroll_area", None)
        container = getattr(self, "transcript_container", None)
        if (
            index < 0
            or index >= len(records)
            or scroll is None
            or container is None
        ):
            return None
        wrapper = records[index].wrapper
        if wrapper is None:
            return None
        try:
            y = wrapper.mapTo(container, wrapper.rect().topLeft()).y()
        except RuntimeError:
            return None
        bar = scroll.verticalScrollBar()
        return compute_scroll_target_for_waypoint_y(
            int(y),
            margin=24,
            scroll_min=bar.minimum(),
            scroll_max=bar.maximum(),
        )

    def _scroll_to_transcript_waypoint_index(
        self,
        index: int,
        *,
        animated: bool = True,
    ) -> None:
        scroll = getattr(self, "scroll_area", None)
        if scroll is None:
            return
        target = self._scroll_target_for_waypoint_index(index)
        if target is None:
            return
        bar = scroll.verticalScrollBar()
        if not animated or bar.value() == target:
            self._stop_transcript_timeline_scroll_anim()
            bar.setValue(target)
        else:
            self._animate_transcript_scroll_to(target)
        rail = getattr(self, "_transcript_timeline_rail", None)
        if rail is not None:
            rail.set_active_index(index)

    def _stop_transcript_timeline_scroll_anim(self) -> None:
        anim = getattr(self, "_transcript_timeline_scroll_anim", None)
        if anim is not None:
            anim.stop()
            self._transcript_timeline_scroll_anim = None

    def _animate_transcript_scroll_to(self, target: int) -> None:
        scroll = getattr(self, "scroll_area", None)
        if scroll is None:
            return
        bar = scroll.verticalScrollBar()
        self._stop_transcript_timeline_scroll_anim()
        anim = QPropertyAnimation(bar, b"value", self)
        anim.setDuration(_TRANSCRIPT_TIMELINE_SCROLL_MS)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(bar.value())
        anim.setEndValue(int(target))
        self._transcript_timeline_scroll_anim = anim
        anim.start()

    def _on_transcript_scroll_changed(self, _value: int) -> None:
        rail = getattr(self, "_transcript_timeline_rail", None)
        scroll = getattr(self, "scroll_area", None)
        if rail is None or scroll is None or not rail.isVisible():
            return
        _entries, waypoint_ys = self._collect_transcript_timeline_entries()
        if not waypoint_ys:
            return
        active = compute_active_waypoint_index(
            scroll.verticalScrollBar().value(),
            waypoint_ys,
        )
        rail.set_active_index(active)

    def _on_transcript_timeline_waypoint_clicked(self, index: int) -> None:
        self._scroll_to_transcript_waypoint_index(index, animated=True)

    def _attach_pending_citation_sources(self, label: AgentMessageLabel) -> None:
        """Apply sources from the tool phase to this bubble (or [] if none pending)."""
        pending = getattr(self, "_pending_citation_sources", None)
        if pending is not None:
            label._citation_sources = _snapshot_citation_sources(pending)
            self._pending_citation_sources = None
        else:
            label._citation_sources = []

    def _attach_pending_evidence_transparency(self, label: AgentMessageLabel) -> None:
        pending = getattr(self, "_pending_evidence_transparency", None)
        if pending:
            label._evidence_transparency = copy.deepcopy(pending)
            self._pending_evidence_transparency = None

    def _flush_pending_stream_for_active_session(self) -> None:
        """Render stream chunks that arrived while this session was off-screen."""
        sid = str(getattr(self, "active_session_id", "") or "")
        if not sid:
            return
        buffered = str(self._pending_stream_tokens_by_session.pop(sid, "") or "")
        if not buffered:
            return
        src = self._pending_stream_sources_by_session.pop(sid, None)
        if src is not None:
            self._pending_citation_sources = _snapshot_citation_sources(src)
        transparency = self._pending_stream_transparency_by_session.pop(sid, None)
        if transparency:
            self._pending_evidence_transparency = copy.deepcopy(transparency)
        self.log_agent_token(buffered)

    def on_llm_token_streamed(self, session_id: str, token: str) -> None:
        """Only render stream chunks for the currently active chat session."""
        active = str(getattr(self, "active_session_id", "") or "")
        sid = str(session_id or "")
        if not active or sid != active:
            if sid:
                prev = self._pending_stream_tokens_by_session.get(sid, "")
                self._pending_stream_tokens_by_session[sid] = prev + str(token or "")
            return
        self.log_agent_token(str(token or ""))

    def on_llm_stream_replaced(self, session_id: str, text: str) -> None:
        active = str(getattr(self, "active_session_id", "") or "")
        sid = str(session_id or "")
        if not active or sid != active:
            return
        cleaned = self._sanitize_agent_stream_text(text or "")
        if not cleaned:
            return
        self._agent_text_buffer = cleaned
        self._flush_agent_markdown_coalesce_immediate(finalize=True)

    def on_sources_found(self, session_id: str, sources):
        """Receive tool sources for inline citation links (no separate chip UI)."""
        active = str(getattr(self, "active_session_id", "") or "")
        sid = str(session_id or "")
        if not active or sid != active:
            if sid:
                self._pending_stream_sources_by_session[sid] = _snapshot_citation_sources(sources)
            return
        self._pending_citation_sources = _snapshot_citation_sources(sources)
        cur = getattr(self, "current_agent_msg", None)
        if cur is not None and getattr(cur, "_assistant_turn_id", None) == getattr(
            self, "_user_turn_id", -1
        ):
            cur._citation_sources = _snapshot_citation_sources(sources)
            self._sync_agent_sources_button(cur)
            if (getattr(self, "_agent_text_buffer", "") or "").strip():
                self._schedule_coalesced_agent_markdown()

    def on_evidence_transparency_found(self, session_id: str, transparency: dict) -> None:
        """Attach retrieval transparency during foreground streaming (@evidence / @trusted)."""
        active = str(getattr(self, "active_session_id", "") or "")
        sid = str(session_id or "")
        if not transparency:
            return
        snapshot = copy.deepcopy(transparency)
        if not active or sid != active:
            if sid:
                self._pending_stream_transparency_by_session[sid] = snapshot
            return
        cur = getattr(self, "current_agent_msg", None)
        if cur is not None and getattr(cur, "_assistant_turn_id", None) == getattr(
            self, "_user_turn_id", -1
        ):
            cur._evidence_transparency = snapshot
            self._sync_agent_sources_button(cur)
        else:
            self._pending_evidence_transparency = snapshot

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
        """Opens web URLs in the browser; other sources use the in-app preview dialog."""
        url = str((source_dict or {}).get("url") or "").strip()
        if url.startswith(("http://", "https://")):
            import webbrowser

            webbrowser.open(url)
            return
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
        if getattr(self, "_deep_research_in_progress", False) and not self._llm_in_progress:
            enabled = True
        if hasattr(self, 'text_input') and hasattr(self, 'send_btn'):
            self.text_input.setEnabled(enabled)
            if hasattr(self, "composer_attach_btn"):
                self.composer_attach_btn.setEnabled(enabled)
            if hasattr(self, "composer_voice_btn"):
                self.composer_voice_btn.setEnabled(enabled)
            # Keep stop clickable while text input is disabled.
            self.send_btn.setEnabled(True)
            if enabled:
                self._voice_turn_active = False

            if enabled:
                if not getattr(self, "_deep_research_in_progress", False):
                    self.text_input.setPlaceholderText(COMPOSER_IDLE_PLACEHOLDER)
                    self.text_input.setFocus()
            elif not self._is_stop_mode():
                self.text_input.setPlaceholderText("Qube is working...")
        self._refresh_send_stop_button()

    def apply_presence_label(self, presence_label: str) -> None:
        """Composer placeholder — always driven from canonical presence_label."""
        from core.assistant_activity import composer_placeholder_text

        if not hasattr(self, "text_input"):
            return
        placeholder = composer_placeholder_text(
            presence_label,
            stop_mode=self._is_stop_mode(),
        )
        if placeholder is None:
            return
        self.text_input.setPlaceholderText(placeholder)

    def update_action_placeholder(self, status: str):
        """Backward-compatible alias; prefer apply_presence_label with presence_label."""
        self.apply_presence_label(status)

    def set_stop_requested_callback(self, callback) -> None:
        self._stop_requested_callback = callback

    def set_before_send_callback(self, callback) -> None:
        self._before_send_callback = callback

    def set_manual_voice_callback(self, callback) -> None:
        self._manual_voice_callback = callback

    def _handle_manual_voice_input(self) -> None:
        if self._voice_capture_active:
            return
        cb = self._manual_voice_callback
        if not callable(cb):
            return
        cb()

    def _will_play_tts_after_response(self) -> bool:
        """Match main.py: voice output must be unmuted and the toolbar toggle on."""
        if not self.tts or getattr(self.tts, "is_muted", False):
            return False
        win = self.window()
        toggle = getattr(win, "voice_bypass_toggle", None)
        if toggle is not None:
            return bool(toggle.isChecked())
        return False

    def _restore_send_mode_if_idle(self) -> None:
        if self._llm_in_progress:
            return
        self._awaiting_tts_end = False
        self._tts_playing = False
        self.set_input_enabled(True)
        self._refresh_send_stop_button()

    def on_voice_capture_started(self) -> None:
        """Wakeword listening window — expose Stop so false triggers can be dismissed."""
        self._voice_capture_active = True
        self._voice_turn_active = True
        if hasattr(self, "text_input") and hasattr(self, "send_btn"):
            self.text_input.setEnabled(False)
            if hasattr(self, "composer_attach_btn"):
                self.composer_attach_btn.setEnabled(False)
            if hasattr(self, "composer_voice_btn"):
                self.composer_voice_btn.setEnabled(False)
            self.send_btn.setEnabled(True)
        self._refresh_send_stop_button()

    def on_voice_capture_processing(self) -> None:
        """Mic gate closed; STT/LLM pipeline still running — keep Stop available."""
        if not self._voice_capture_active and not self._voice_turn_active:
            return
        self._voice_capture_active = False
        self._voice_turn_active = True
        if hasattr(self, "text_input") and hasattr(self, "send_btn"):
            self.text_input.setEnabled(False)
            if hasattr(self, "composer_attach_btn"):
                self.composer_attach_btn.setEnabled(False)
            if hasattr(self, "composer_voice_btn"):
                self.composer_voice_btn.setEnabled(False)
            self.send_btn.setEnabled(True)
        self._refresh_send_stop_button()

    def on_voice_capture_ended(self) -> None:
        """Listening aborted without entering the STT/LLM voice pipeline."""
        if not self._voice_capture_active:
            return
        self._voice_capture_active = False
        self._voice_turn_active = False
        self._refresh_send_stop_button()

    def on_voice_capture_stopped(self) -> None:
        """User cancelled a mistaken wakeword before utterance capture finished."""
        self._voice_capture_active = False
        self._voice_turn_active = False
        self.set_input_enabled(True)
        self._refresh_send_stop_button()

    def on_turn_complete_idle(self) -> None:
        """Status bubble returned to idle — release stop mode if generation is done."""
        self._voice_capture_active = False
        self._voice_turn_active = False
        self._restore_send_mode_if_idle()

    def on_llm_response_finished(self, session_id: str, final_text: str = "") -> None:
        sid = str(session_id or "")
        if sid:
            self._pending_stream_tokens_by_session.pop(sid, None)
            self._pending_stream_sources_by_session.pop(sid, None)
            self._pending_stream_transparency_by_session.pop(sid, None)
        active = str(getattr(self, "active_session_id", "") or "")
        if active and sid == active:
            cleaned = self._sanitize_agent_stream_text(final_text or "")
            if cleaned:
                cur = getattr(self, "current_agent_msg", None)
                if cur is None:
                    self.log_agent_token(cleaned)
                else:
                    # The finished worker text is the sanitized source of truth.
                    # Replace the active bubble instead of appending/reconciling around leaked prefix text.
                    self._agent_text_buffer = cleaned
                    self._schedule_coalesced_agent_markdown()
                try:
                    cite_sources = []
                    if cur is not None:
                        cite_sources = getattr(cur, "_citation_sources", None) or []
                    report = analyze_citations(cleaned, cite_sources)
                    log_citation_integrity(
                        report,
                        phase="ui_finalize",
                        session_id=sid,
                    )
                except Exception:
                    logger.debug("[CitationIntegrity] ui_finalize telemetry failed", exc_info=True)
            self._flush_agent_markdown_coalesce_immediate(finalize=True)
            self._hide_agent_typing_row()
        elif self._llm_in_progress:
            logger.warning(
                "[ChatUI] LLM finished for session %s but active is %s; releasing UI anyway.",
                sid,
                active or "(none)",
            )
            self._flush_agent_markdown_coalesce_immediate(finalize=True)
            self._hide_agent_typing_row()

        self._llm_in_progress = False
        tts_expected = self._will_play_tts_after_response()
        self._awaiting_tts_end = tts_expected
        logger.info(
            "[ChatUI] LLM finished; stop_mode transitions to await_tts=%s.",
            tts_expected,
        )
        if not tts_expected:
            self._voice_turn_active = False
            self.set_input_enabled(True)
        self._refresh_send_stop_button()

    def on_tts_playback_started(self, _session_id: str = "") -> None:
        self._tts_playing = True
        self._awaiting_tts_end = True
        logger.info("[ChatUI] TTS playback started; keep Stop button active.")
        self._refresh_send_stop_button()

    def on_tts_playback_finished(self) -> None:
        self._tts_playing = False
        logger.info("[ChatUI] TTS playback finished; restoring send mode if LLM is idle.")
        self._restore_send_mode_if_idle()

    def on_tts_turn_settled(self) -> None:
        """End-of-turn sentinel processed (even when no audio was output)."""
        self._voice_turn_active = False
        self._awaiting_tts_end = False
        self._restore_send_mode_if_idle()

    def on_generation_stopped(self) -> None:
        logger.info("[ChatUI] Stop acknowledged; clearing active generation/audio state.")
        self._flush_agent_markdown_coalesce_immediate(finalize=True)
        self._hide_agent_typing_row()
        self._llm_in_progress = False
        self._awaiting_tts_end = False
        self._tts_playing = False
        self._voice_capture_active = False
        self._voice_turn_active = False
        self.set_input_enabled(True)
        self.clear_stale_agent_pointer()
        self._refresh_send_stop_button()

    def _is_stop_mode(self) -> bool:
        return (
            self._voice_capture_active
            or self._voice_turn_active
            or self._llm_in_progress
            or self._awaiting_tts_end
            or self._tts_playing
            or getattr(self, "_deep_research_in_progress", False)
        )

    def _stop_button_tooltip(self) -> str:
        if self._voice_capture_active:
            return "Stop listening"
        if getattr(self, "_deep_research_in_progress", False):
            return "Stop deep research"
        if self._is_stop_mode():
            return "Stop response"
        return "Send message"

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

    def interrupt_active_response(self) -> None:
        """
        Public interrupt hook for non-chat UI actions (e.g. model switching).
        Reuses the same stop path as the chat Stop button.
        """
        if self._is_stop_mode():
            self._request_stop()