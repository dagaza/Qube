import os

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QPushButton,
    QListWidget,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
    QProgressBar,
    QLineEdit,
    QApplication,
    QGraphicsOpacityEffect,
)
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QIcon,
    QColor,
    QPixmap,
    QPainter,
    QTextBlockFormat,
    QTextCursor,
)
import qtawesome as qta
from pathlib import Path
from ui.components.prestige_dialog import PrestigeDialog
import logging

logger = logging.getLogger("Qube.UI.Library")

# Match ConversationsView utility toolbar / layout (library preview).
LAYOUT_FULL_WIDTH = "full_width"
LAYOUT_CENTERED_COLUMN = "centered_column"
_CENTERED_COLUMN_MAX_WIDTH = 900
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
_CHAT_UTILITY_BTN = 30
_CHAT_UTILITY_ICON_PX = 18
_BASE_PREVIEW_FONT_PT = 10.0
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
_PREVIEW_READER_FOCUS_DIM = 0.58


class _LibraryTranscriptWidthHost(QWidget):
    """Centers transcript; inner width = min(available, cap). QTextEdit stays inside inner only."""

    def __init__(self, inner: QWidget, max_w: int, parent=None):
        super().__init__(parent)
        self._inner = inner
        self._max_w = max(1, int(max_w))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addStretch(1)
        # Default alignment (0): inner fills cell height. AlignCenter would vertically
        # center at sizeHint and collapse the transcript; side stretches still center the column.
        lay.addWidget(inner, 0)
        lay.addStretch(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_max_width_cap(self, max_w: int) -> None:
        self._max_w = max(1, int(max_w))
        self._sync_inner_width()

    def _sync_inner_width(self) -> None:
        w = min(self._max_w, max(1, self.width()))
        if self._inner.width() != w:
            self._inner.setFixedWidth(w)
        self._inner.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_inner_width()


class LibraryView(QWidget):
    ingest_requested = pyqtSignal(list)

    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        # We need the vector store for reconstruction and deep deletion
        # (Assuming 'store' was added to the workers dictionary in main.py)
        self.store = workers.get("store") 

        # 🔑 THE FIX: Declare the flag here so it exists on boot!
        self._had_ingestion_error = False
        
        self.active_filename = None
        self._font_scale: float = 1.0
        self._line_height_mode: str = _LINE_HEIGHT_COMFORTABLE
        self._focus_mode_enabled: bool = False
        self._high_contrast_enabled: bool = False
        self._layout_mode: str = LAYOUT_FULL_WIDTH

        self._setup_ui()
        self.refresh_library_list()
        # Rows are built before this view is parented under MainWindow; QSS + selection need a pass once attached.
        QTimer.singleShot(0, self._update_row_colors)

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        # --- COLUMN 1: Document List Sidebar ---
        self.list_pane = self._build_list_pane()
        layout.addWidget(self.list_pane)

        # --- COLUMN 2: Document Preview Stage ---
        self.preview_stage = self._build_preview_stage()

        # 🔑 THE FIX: Force the layout to ignore the internal text's size demands
        self.preview_stage.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        layout.addWidget(self.preview_stage, stretch=1)

        # Forces the button to load with the default Dark Mode purple on startup
        self.refresh_button_themes(is_dark=True)

    def _build_list_pane(self) -> QFrame:
        frame = QFrame()
        frame.setFixedWidth(280)
        frame.setObjectName("LibrarySidebar") 
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)

        # --- HEADER AREA ---
        header_layout = QHBoxLayout()
        self.list_title = QLabel("KNOWLEDGE BASE")
        self.list_title.setProperty("class", "SidebarTitle")
        
        self.add_btn = QPushButton()
        self.add_btn.setIcon(qta.icon('fa5s.plus')) 
        self.add_btn.setProperty("class", "IconButton") 
        self.add_btn.setToolTip("Ingest New Document")
        self.add_btn.clicked.connect(self._browse_for_document) 
        
        header_layout.addWidget(self.list_title)
        header_layout.addStretch()
        header_layout.addWidget(self.add_btn)
        
        layout.addLayout(header_layout)

        # --- PROGRESS BAR (Moved here, right under the header!) ---
        self.ingest_progress = QProgressBar()
        self.ingest_progress.setObjectName("IngestProgressBar")
        self.ingest_progress.setRange(0, 100)
        self.ingest_progress.setFixedHeight(4) # Made it slightly thinner for a sleeker look
        self.ingest_progress.setTextVisible(False)
        self.ingest_progress.hide() 
        layout.addWidget(self.ingest_progress)

        # The Search Bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search documents...")
        self.search_bar.setObjectName("LibrarySearchBar")
        self.search_bar.textChanged.connect(self._filter_list)
        layout.addWidget(self.search_bar)

        # Document List
        self.doc_list = QListWidget()
        self.doc_list.setObjectName("LibraryDocList")
        self.doc_list.itemClicked.connect(self._on_document_selected)
        self.doc_list.itemSelectionChanged.connect(self._update_row_colors)
        
        # 🔑 NEW: Wire up the scrollbar for infinite scrolling
        self.library_offset = 0
        self.is_loading_library = False
        self.doc_list.verticalScrollBar().valueChanged.connect(self._on_library_scroll)

        layout.addWidget(self.doc_list)

        return frame
    
    def _build_preview_stage(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("LibraryPreviewStage")

        # 🔑 FIX 1: Strip any global card styling from the main frame
        frame.setStyleSheet("background: transparent; border: none;")

        layout = QVBoxLayout(frame)
        # Match ConversationsView._build_chat_stage so the toolbar row lines up with chat.
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
        self.font_minus_btn.setToolTip("Decrease preview font (Shift+click: larger step)")
        self.font_minus_btn.clicked.connect(self._on_font_minus_clicked)

        self.font_plus_btn = QPushButton("A+")
        self.font_plus_btn.setObjectName("ReadabilityFontPlus")
        self.font_plus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.font_plus_btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        self.font_plus_btn.setToolTip("Increase preview font (Shift+click: larger step)")
        self.font_plus_btn.clicked.connect(self._on_font_plus_clicked)

        self.line_height_btn = QPushButton()
        self.line_height_btn.setObjectName("ReadabilityLineHeight")
        self.line_height_btn.setProperty("class", "IconButton")
        self.line_height_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.line_height_btn.clicked.connect(self._cycle_line_height_mode)

        self.reader_focus_btn = QPushButton()
        self.reader_focus_btn.setObjectName("ReadabilityReaderFocus")
        self.reader_focus_btn.setProperty("class", "IconButton")
        self.reader_focus_btn.setCheckable(True)
        self.reader_focus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reader_focus_btn.setToolTip("Reader focus: dim document header")
        self.reader_focus_btn.toggled.connect(self._on_reader_focus_toggled)

        self.high_contrast_btn = QPushButton()
        self.high_contrast_btn.setObjectName("ReadabilityHighContrast")
        self.high_contrast_btn.setProperty("class", "IconButton")
        self.high_contrast_btn.setCheckable(True)
        self.high_contrast_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.high_contrast_btn.setToolTip("High contrast (document preview)")
        self.high_contrast_btn.toggled.connect(self._on_high_contrast_toggled)

        self.layout_mode_btn = QPushButton()
        self.layout_mode_btn.setObjectName("LayoutModeButton")
        self.layout_mode_btn.setProperty("class", "IconButton")
        self.layout_mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.layout_mode_btn.clicked.connect(self._toggle_layout_mode)

        read_row.addWidget(self.font_minus_btn)
        read_row.addWidget(self.font_plus_btn)
        read_row.addWidget(self.line_height_btn)
        read_row.addWidget(self.reader_focus_btn)
        read_row.addWidget(self.high_contrast_btn)
        read_row.addWidget(self.layout_mode_btn)

        utility_layout.addWidget(readability_host, 0, Qt.AlignmentFlag.AlignLeft)
        utility_layout.addStretch(1)
        layout.addWidget(utility_toolbar)

        # Header Area for Preview
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.doc_title = QLabel("No Document Selected")
        self.doc_title.setObjectName("PreviewDocTitle")
        
        # 🔑 FIX 1: Stop the title from forcing the window wider on long filenames
        self.doc_title.setWordWrap(True)
        self.doc_title.setMinimumWidth(0)
        self.doc_title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        self.doc_stats = QLabel("")
        self.doc_stats.setObjectName("PreviewStatsText")
        
        # 🔑 FIX 2: Stop the stats from stretching the window
        self.doc_stats.setWordWrap(True)
        self.doc_stats.setMinimumWidth(0)
        self.doc_stats.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        title_host = QWidget()
        title_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        title_host.setMinimumWidth(0)
        self._preview_header_host = title_host
        title_vbox = QVBoxLayout(title_host)
        title_vbox.setContentsMargins(0, 0, 0, 0)
        title_vbox.setSpacing(4)
        title_vbox.addWidget(self.doc_title)
        title_vbox.addWidget(self.doc_stats)
        
        # Give the title block full header width so wrapping only occurs at real stage boundaries.
        header_layout.addWidget(title_host, 1)
        layout.addLayout(header_layout)

        # Reconstructed Text Area
        self.text_preview = QTextEdit()
        self.text_preview.setObjectName("DocumentPreviewArea")
        self.text_preview.setReadOnly(True)
        
        # --- THE FIX: Aggressive Wrapping & Shrink Allowance ---
        from PyQt6.QtGui import QTextOption
        
        # 1. Allow the widget to shrink freely when the user resizes the app
        self.text_preview.setMinimumWidth(0) 
        self.text_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # 2. Force wrapping strictly at the widget's edge
        self.text_preview.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        
        # 3. CRITICAL: Break long unbreakable strings (like PDF hashes or long titles) 
        # instead of stretching the parent window.
        self.text_preview.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        
        # 4. Strip the default PyQt sunken card box for that clean look we discussed
        self.text_preview.setStyleSheet("background: transparent; border: none;")

        self.text_preview.setPlaceholderText("Select a document from the left to view its contents.")

        transcript_inner = QWidget()
        transcript_inner.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        transcript_inner.setMinimumWidth(0)
        ti_layout = QVBoxLayout(transcript_inner)
        ti_layout.setContentsMargins(0, 0, 0, 0)
        ti_layout.setSpacing(0)
        ti_layout.addWidget(self.text_preview, 1)

        self._transcript_width_host = _LibraryTranscriptWidthHost(
            transcript_inner, _QWIDGETSIZE_MAX
        )
        layout.addWidget(self._transcript_width_host, stretch=1)

        self._apply_library_layout_mode()
        self._refresh_readability_toolbar(is_dark=True)
        self._apply_library_preview_readability()

        return frame

    # --------------------------------------------------------- #
    #  Preview utility toolbar + transcript column width (Conversations parity)
    # --------------------------------------------------------- #

    @property
    def layout_mode(self) -> str:
        return self._layout_mode

    def set_layout_mode(self, mode: str) -> None:
        if mode not in (LAYOUT_FULL_WIDTH, LAYOUT_CENTERED_COLUMN):
            return
        if mode == self._layout_mode:
            self._refresh_layout_mode_button()
            return
        self._layout_mode = mode
        self._apply_library_layout_mode()

    def _apply_library_layout_mode(self) -> None:
        host = getattr(self, "_transcript_width_host", None)
        if host is None:
            return
        cap = (
            _CENTERED_COLUMN_MAX_WIDTH
            if self._layout_mode == LAYOUT_CENTERED_COLUMN
            else _QWIDGETSIZE_MAX
        )
        host.set_max_width_cap(cap)
        host.updateGeometry()
        self._refresh_layout_mode_button()

    def _toggle_layout_mode(self) -> None:
        next_mode = (
            LAYOUT_CENTERED_COLUMN
            if self._layout_mode == LAYOUT_FULL_WIDTH
            else LAYOUT_FULL_WIDTH
        )
        self.set_layout_mode(next_mode)

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

    def _scaled_preview_font_pt(self) -> float:
        return max(8.0, min(28.0, _BASE_PREVIEW_FONT_PT * self._font_scale))

    def _line_height_css_value(self) -> str:
        return _LINE_HEIGHT_CSS.get(
            self._line_height_mode, _LINE_HEIGHT_CSS[_LINE_HEIGHT_COMFORTABLE]
        )

    def _line_height_proportional_percent(self) -> int:
        try:
            return int(round(float(self._line_height_css_value()) * 100))
        except ValueError:
            return 145

    def _preview_body_color(self, is_dark: bool) -> str:
        if self._high_contrast_enabled:
            return "#f8fafc" if is_dark else "#0f172a"
        return "#cdd6f4" if is_dark else "#1e293b"

    def _apply_preview_line_height(self, doc) -> None:
        fmt = QTextBlockFormat()
        fmt.setLineHeight(float(self._line_height_proportional_percent()), 1)
        cur = QTextCursor(doc)
        cur.beginEditBlock()
        block = doc.firstBlock()
        while block.isValid():
            cur.setPosition(block.position())
            cur.mergeBlockFormat(fmt)
            block = block.next()
        cur.endEditBlock()

    def _apply_library_preview_readability(self) -> None:
        if not hasattr(self, "text_preview"):
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        pt = self._scaled_preview_font_pt()
        f = self.text_preview.font()
        f.setPointSizeF(pt)
        self.text_preview.setFont(f)
        doc = self.text_preview.document()
        doc.setDefaultFont(f)
        self._apply_preview_line_height(doc)
        fg = self._preview_body_color(is_dark)
        self.text_preview.setStyleSheet(
            f"background: transparent; border: none; color: {fg}; font-size: {pt:.1f}pt;"
        )
        self._apply_preview_reader_focus_opacity()

    def _nudge_font_scale(self, delta: float) -> None:
        new_v = round(self._font_scale + delta, 4)
        new_v = max(_FONT_SCALE_MIN, min(_FONT_SCALE_MAX, new_v))
        if new_v == self._font_scale:
            return
        self._font_scale = new_v
        self._apply_library_preview_readability()
        self._refresh_readability_toolbar()

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
        self._apply_library_preview_readability()
        self._refresh_readability_toolbar()

    def _on_reader_focus_toggled(self, checked: bool) -> None:
        self._focus_mode_enabled = bool(checked)
        self._refresh_readability_toolbar()
        self._apply_preview_reader_focus_opacity()

    def _on_high_contrast_toggled(self, checked: bool) -> None:
        self._high_contrast_enabled = bool(checked)
        self._apply_library_preview_readability()
        self._refresh_readability_toolbar()

    def _set_header_opacity(self, w: QWidget | None, opacity: float) -> None:
        if w is None:
            return
        if opacity >= 0.999:
            w.setGraphicsEffect(None)
            return
        eff = w.graphicsEffect()
        if not isinstance(eff, QGraphicsOpacityEffect):
            eff = QGraphicsOpacityEffect(w)
            w.setGraphicsEffect(eff)
        eff.setOpacity(opacity)

    def _apply_preview_reader_focus_opacity(self) -> None:
        host = getattr(self, "_preview_header_host", None)
        if not self._focus_mode_enabled:
            self._set_header_opacity(host, 1.0)
            return
        self._set_header_opacity(host, _PREVIEW_READER_FOCUS_DIM)

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
        self.line_height_btn.setIcon(
            self._make_tinted_svg_icon(_LINE_SPACING_ICON, icon_muted, size=_CHAT_UTILITY_ICON_PX)
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
        for btn in (self.line_height_btn, self.reader_focus_btn, self.high_contrast_btn):
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
        self._refresh_layout_mode_button(is_dark=is_dark)

    # --------------------------------------------------------- #
    #  LOGIC WIRING                                             #
    # --------------------------------------------------------- #

    def _apply_menu_theme(self, menu, is_dark: bool):
        """Standardizes the menu appearance with Prestige rounding and colors."""
        
        # THIS IS THE MAGIC LINE TO KILL THE GHOST SQUARE
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        bg, fg, hover = ("#1e1e2e", "#cdd6f4", "#313244") if is_dark else ("#ffffff", "#1e293b", "#f1f5f9")
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"

        menu.setStyleSheet(f"""
            QMenu {{ background-color: {bg}; color: {fg}; border: 1px solid {border}; border-radius: 12px; padding: 5px; }}
            QMenu::item {{ background-color: transparent; padding: 8px 25px; border-radius: 8px; }}
            QMenu::item:selected {{ background-color: {hover}; color: {fg}; }}
        """)

    def refresh_library_list(self):
        """Clears the list, resets the offset, and pulls the first batch."""
        self.doc_list.clear()
        self.library_offset = 0
        self.is_loading_library = False

        count = self.db.get_document_count()
        display_count = "999+" if count > 999 else str(count)
        
        if hasattr(self, 'list_title'):
            self.list_title.setText(f"KNOWLEDGE BASE ({display_count})")
            
        # Load the initial batch!
        self._load_library_batch()

    def _on_library_scroll(self, value):
        """Triggered every time the user scrolls."""
        bar = self.doc_list.verticalScrollBar()
        # If the scrollbar hits the absolute maximum, load the next batch
        if value == bar.maximum():
            self._load_library_batch()

    def _load_library_batch(self):
        """Fetches the next chunk of documents and appends it to the list."""
        if getattr(self, 'is_loading_library', False):
            return 
            
        self.is_loading_library = True

        from PyQt6.QtWidgets import QListWidgetItem, QWidget, QHBoxLayout, QLabel, QPushButton, QMenu
        from PyQt6.QtCore import Qt, QSize
        import qtawesome as qta
        
        # Match ConversationsView._load_history_batch: window may be None until after MainWindow adds this widget.
        is_dark = True
        main_win = self.window()
        if main_win and hasattr(main_win, "_is_dark_theme"):
            is_dark = main_win._is_dark_theme

        t_color = "#cdd6f4" if is_dark else "#1e293b"
        icon_color = "#6c7086" if is_dark else "#64748b" 
        
        # 🔑 THE FIX: Pass both limit and offset to the DB
        try:
            docs = self.db.get_library_documents(limit=20, offset=self.library_offset)
        except TypeError:
            # Fallback if DB isn't updated yet
            docs = self.db.get_library_documents()
            if self.library_offset > 0:
                docs = [] 

        if not docs:
            self.is_loading_library = False
            return

        for doc in docs:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, doc)
            
            row = QWidget()
            row.setObjectName("HistoryRowWidget")
            row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            
            lay = QHBoxLayout(row)
            lay.setContentsMargins(15, 0, 10, 0)
            lay.setSpacing(10)
            
            # 1. The Title
            item_text = f"{doc['filename']} ({doc['file_size_kb']} KB)"
            lbl = QLabel(item_text)
            lbl.setObjectName("HistoryRowTitle")
            lbl.setStyleSheet(f"color: {t_color}; background: transparent; border: none; font-size: 13px; font-weight: 500;")
            
            # 2. The Kebab Button
            btn = QPushButton()
            btn.setObjectName("HistoryOptionsBtn")
            btn.setFixedSize(28, 28)
            btn.setIcon(qta.icon('fa5s.ellipsis-v', color=icon_color))
            btn.setIconSize(QSize(16, 16)) 
            btn.setStyleSheet("QPushButton::menu-indicator { image: none; width: 0px; } QPushButton { border: none; background: transparent; }")
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            
            # 3. The Menu
            menu = QMenu(btn)
            if hasattr(self, '_apply_menu_theme'):
                self._apply_menu_theme(menu, is_dark)
            
            rename_action = menu.addAction(qta.icon('fa5s.edit', color='#89b4fa'), "Rename Document")
            rename_action.triggered.connect(lambda _, fname=doc['filename']: self._trigger_rename_document(fname))
            
            menu.addSeparator()

            delete_action = menu.addAction(qta.icon('fa5s.trash-alt', color='#ef4444'), "Delete Document")
            delete_action.triggered.connect(lambda _, fname=doc['filename']: self._trigger_delete_document(fname))
            
            btn.setMenu(menu)
            
            lay.addWidget(lbl)
            lay.addStretch()
            lay.addWidget(btn)
            
            item.setSizeHint(QSize(0, 45))
            self.doc_list.addItem(item)
            self.doc_list.setItemWidget(item, row)

        # 🔑 Increment the offset for the next scroll
        self.library_offset += 20
        self.is_loading_library = False

    def _update_row_colors(self):
        """Forces text color changes since Qt CSS cannot pass :selected states to setItemWidget."""
        from PyQt6.QtWidgets import QLabel
        
        # 1. Safely Detect Theme
        is_dark = getattr(self.window(), '_is_dark_theme', True)
            
        # 2. 🔑 THE FIX: The Colors!
        # Unselected: Light gray in Dark Mode, Slate in Light Mode
        normal_color = "#cdd6f4" if is_dark else "#1e293b"
        # Selected: The background bubble is solid, so text should ALWAYS be White
        selected_color = "#ffffff" 

        # 3. Target whichever list is in this specific file
        target_list = getattr(self, 'doc_list', getattr(self, 'history_list', None))
        if not target_list: 
            return

        # 4. Loop through and forcefully apply the correct color
        for i in range(target_list.count()):
            item = target_list.item(i)
            widget = target_list.itemWidget(item)
            if widget:
                lbl = widget.findChild(QLabel, "HistoryRowTitle")
                if lbl:
                    color = selected_color if item.isSelected() else normal_color
                    lbl.setStyleSheet(f"color: {color}; background: transparent; border: none; font-size: 13px; font-weight: 500;")

    def _trigger_rename_document(self, old_filename):
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        dlg = PrestigeDialog(self, "Rename Document", f"Enter a new name for '{old_filename}':", is_dark, is_input=True, default_text=old_filename)
        
        if dlg.exec() and dlg.result_text and dlg.result_text.strip():
            new_name = dlg.result_text.strip()
            
            # 1. Update SQLite
            self.db.rename_document_metadata(old_filename, new_name)
            
            # 2. Update Vector Store (CRITICAL: You must implement this in your LanceDB class!)
            if self.store and hasattr(self.store, 'rename_document'):
                self.store.rename_document(old_filename, new_name)
            elif self.store:
                logger.warning(f"Renamed {old_filename} in SQLite, but 'rename_document' is missing in Vector Store!")

            # 3. Update UI if they renamed the currently open document
            if self.active_filename == old_filename:
                self.active_filename = new_name
                self.doc_title.setText(new_name)
                
            self.refresh_library_list()

    def _trigger_delete_document(self, filename):
        """Spawns the Prestige dialog and coordinates deletion from both DBs."""
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        dlg = PrestigeDialog(self, "Delete Document", f"Are you sure you want to permanently delete and un-index '{filename}'?", is_dark)
        
        if dlg.exec():
            logger.info(f"User initiated deletion of {filename}")
            
            if self.store:
                self.store.delete_document(filename)
            
            self.db.delete_document_metadata(filename)
            
            # If they deleted the document they are currently looking at, clear the preview
            if self.active_filename == filename:
                self.active_filename = None
                self.doc_title.setText("No Document Selected")
                self.doc_stats.setText("")
                self.text_preview.setHtml("<center><h3>Document deleted.</h3></center>")
                self._apply_library_preview_readability()

            self.refresh_library_list()

    def _filter_list(self, text: str):
        """Hides/Shows list items based on the search bar text."""
        search_term = text.lower()
        for i in range(self.doc_list.count()):
            item = self.doc_list.item(i)
            # Retrieve the filename from the stored metadata
            doc_data = item.data(Qt.ItemDataRole.UserRole)
            if doc_data and search_term in doc_data['filename'].lower():
                item.setHidden(False)
            else:
                item.setHidden(True)

    def _on_document_selected(self, item):
        doc_data = item.data(Qt.ItemDataRole.UserRole)
        self.active_filename = doc_data['filename']
        
        self.doc_title.setText(self.active_filename)
        self.doc_stats.setText(f"Size: {doc_data['file_size_kb']} KB | Chunks Indexed: {doc_data['chunk_count']}")

        self.text_preview.setHtml("<center><h3>Reconstructing document from vector space...</h3></center>")
        self._apply_library_preview_readability()

        # Pull chunks from LanceDB and stitch them together
        if self.store:
            content = self.store.reconstruct_document(self.active_filename)
            self.text_preview.setPlainText(content)
        else:
            self.text_preview.setPlainText("Error: Vector store not connected.")
        self._apply_library_preview_readability()

    def _browse_for_document(self):
        """Opens a file dialog, checks for duplicates, and handles overwrites."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Documents to Ingest", "", "Documents (*.txt *.md *.pdf *.epub)"
        )
        if not files:
            return

        paths = [Path(f) for f in files]
        
        # 1. Check if any selected files already exist in our SQLite registry
        existing_files = []
        current_docs = [doc['filename'] for doc in self.db.get_library_documents()]
        
        for p in paths:
            if p.name in current_docs:
                existing_files.append(p.name)

        # 2. Prompt the user if duplicates are found
        if existing_files:
            msg = (f"The following {len(existing_files)} file(s) already exist in your Knowledge Base:\n\n"
                   f"{', '.join(existing_files[:5])}" + ("..." if len(existing_files) > 5 else "") +
                   "\n\nDo you want to overwrite them?")
            
            # 🔑 Use PrestigeDialog for the Yes/No check
            is_dark = getattr(self.window(), '_is_dark_theme', True)
            dialog = PrestigeDialog(self, "Overwrite Files?", msg, is_dark)

            # .exec() returns truthy if they clicked the primary confirmation button
            if dialog.exec():
                logger.info("User chose to overwrite existing files. Purging old data...")
                for name in existing_files:
                    if self.store: self.store.delete_document(name)
                    self.db.delete_document_metadata(name)
            else:
                paths = [p for p in paths if p.name not in existing_files]
                if not paths:
                    logger.info("Ingestion cancelled; all selected files were duplicates and user declined overwrite.")
                    return

        # 3. Proceed with the standard ingestion UI updates
        self.ingest_progress.setValue(0)
        self.ingest_progress.show()
        self.add_btn.setEnabled(False)
        # REMOVED: self.add_btn.setText(...)
        
        logger.info(f"Emitting {len(paths)} files to main pipeline for ingestion.")
        # 🔑 Reset the flag right before starting a new job
        self._had_ingestion_error = False
        self.ingest_requested.emit(paths)

    # --- UI Receivers for Worker Progress ---
    def update_ingestion_progress(self, percent: int):
        self.ingest_progress.setValue(percent)

    def show_error(self, error_msg: str):
        """Displays ingestion errors to the user and resets the UI."""
        self._had_ingestion_error = True 
        
        self.ingest_progress.hide()
        self.ingest_progress.setValue(0)
        self.add_btn.setEnabled(True)

        # 🔑 Use the superior PrestigeDialog
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        dialog = PrestigeDialog(self, "Ingestion Failed", str(error_msg), is_dark)
        dialog.exec()

    def complete_ingestion(self, total_chunks: int):
        self.ingest_progress.hide()
        self.ingest_progress.setValue(0)
        self.add_btn.setEnabled(True)
        
        if self._had_ingestion_error:
            return 
            
        self.refresh_library_list()
        
        if total_chunks == 0:
            is_dark = getattr(self.window(), '_is_dark_theme', True)
            msg = "Process finished, but 0 chunks were added. This usually means the file was already in the database, or it is a scanned PDF with no readable text."
            dialog = PrestigeDialog(self, "No Data Added", msg, is_dark)
            dialog.exec()

    def refresh_button_themes(self, is_dark: bool):
        """Dynamically updates the color of the Add Document button."""
        import qtawesome as qta
        
        # Icon color: Catppuccin Purple in Dark Mode, Deep Slate in Light Mode
        icon_color = "#8b5cf6" if is_dark else "#1e293b"
        
        # Subtle hover background
        hover_bg = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.05)"
        
        if hasattr(self, 'add_btn'):
            self.add_btn.setIcon(qta.icon('fa5s.plus', color=icon_color))
            self.add_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; border-radius: 6px; padding: 6px; }}
                QPushButton:hover {{ background-color: {hover_bg}; }}
            """)

        self._refresh_readability_toolbar(is_dark=is_dark)
        if hasattr(self, "font_minus_btn"):
            dis = "#6c7086" if is_dark else "#94a3b8"
            font_btn_style = f"""
                QPushButton {{
                    background: transparent;
                    border: none;
                    border-radius: 6px;
                    padding: 2px 4px;
                    color: {icon_color};
                    font-weight: 700;
                    font-size: 13px;
                    min-width: {_CHAT_UTILITY_BTN}px;
                    max-width: {_CHAT_UTILITY_BTN}px;
                    min-height: {_CHAT_UTILITY_BTN}px;
                    max-height: {_CHAT_UTILITY_BTN}px;
                }}
                QPushButton:hover {{ background-color: {hover_bg}; }}
                QPushButton:disabled {{ color: {dis}; }}
            """
            self.font_minus_btn.setStyleSheet(font_btn_style)
            self.font_plus_btn.setStyleSheet(font_btn_style)

        if hasattr(self, "text_preview"):
            self._apply_library_preview_readability()