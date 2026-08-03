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
    QLineEdit,
    QApplication,
    QGraphicsOpacityEffect,
)
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal, QEvent
from PyQt6.QtGui import (
    QIcon,
    QColor,
    QPalette,
    QPixmap,
    QPainter,
    QTextBlockFormat,
    QTextCursor,
)
import qtawesome as qta
from pathlib import Path
from ui.sidebar_dimensions import LEFT_NAV_LIST_SIDEBAR_WIDTH
from ui.components.page_tour_help_button import PageTourHelpButton
from ui.components.prestige_menu_qss import apply_prestige_kebab_menu_theme
from ui.components.prestige_dialog import PrestigeDialog
from core.composer_attachments import validate_file_token
from ui.components.readability_toolbar_styles import readability_font_pair_stylesheet
from ui.components.sidebar_list_qss import apply_sidebar_row_theme
from ui.shell_theme import sidebar_row_action_icon_color
from ui.components.sidebar_folder_list import (
    FOLDER_ROW_MARGIN_LEFT,
    ROW_KIND_DOCUMENT,
    SIDEBAR_ROW_KIND_ROLE,
    SIDEBAR_ROW_PAYLOAD_ROLE,
    SidebarFolderListController,
    add_new_folder_header_button,
    create_sidebar_header_actions_row,
)
from ui.components.ingest_progress_row import IngestProgressRow
from ui.components.library_ingest_mode_dialog import LibraryIngestModeDialog
from ui.components.pro_gem_badge import (
    apply_pro_gem_badge_theme,
    make_pro_gem_badge,
)
from core.library_ingest_modes import is_precision_ingest_mode
from core.app_settings import get_ui_library_transcript_background
from core.theme.view_theme import view_resolved_theme
from core.theme.svg_icons import tinted_svg_icon, themed_fa_icon, themed_fa_pixmap
from ui.components.ghost_icon_button import apply_ghost_icon_button_style
from ui.shell_theme import accent_icon_color
from core.surface_fill.constants import SURFACE_LIBRARY_PREVIEW
from ui.surface_fill.transcript_host import (
    TranscriptWallpaperHost,
    bind_transcript_wallpaper_readability,
)
from core.theme.widget_styles import (
    ACCENT_ICON,
    ACCENT_ICON_ACTIVE,
    AGENT_MESSAGE_FRAME,
    CHAT_WITH_DOC_FAB,
    DANGER_ICON,
    GHOST_ICON_BUTTON,
    LINK_ICON,
    LIST_SURFACE,
    MUTED_ICON,
    TRANSPARENT_FRAME,
    TRANSPARENT_TEXT_PREVIEW,
    UTILITY_ICON_BUTTON,
)
import logging

logger = logging.getLogger("Qube.UI.Library")

# Match ConversationsView utility toolbar / layout (library preview).
LAYOUT_FULL_WIDTH = "full_width"
LAYOUT_CENTERED_COLUMN = "centered_column"
_CENTERED_COLUMN_MAX_WIDTH = 800
_FULL_WIDTH_COLUMN_MAX_WIDTH = 1200
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
_LIBRARY_TRANSCRIPT_CARD_MARGINS = (14, 10, 14, 8)
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

ALIGN_LEFT = "align_left"
ALIGN_JUSTIFY = "align_justify"

# QLabel word-wrap breaks at spaces only; long underscore_joined filenames stay one token.
_PREVIEW_TITLE_SOFT_BREAK = "\u200b"
_CHAT_WITH_DOC_FAB_SIZE = 52
_CHAT_WITH_DOC_FAB_MARGIN = 24


def _filename_title_for_label(text: str) -> str:
    """Insert zero-width break opportunities after underscores so titles wrap in QLabel."""
    if not text or "_" not in text:
        return text
    return text.replace("_", f"_{_PREVIEW_TITLE_SOFT_BREAK}")


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
    ingest_requested = pyqtSignal(list, str, str)

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
        self._library_transcript_background_enabled: bool = (
            get_ui_library_transcript_background()
        )
        self._layout_mode: str = LAYOUT_CENTERED_COLUMN
        self._transcript_alignment: str = ALIGN_JUSTIFY

        self._active_folder_id: str | None = None
        self._folder_controller: SidebarFolderListController | None = None

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

        self._setup_chat_with_doc_fab()
        self.preview_stage.installEventFilter(self)

        self.refresh_button_themes(self._theme().is_dark)

        from ui.components.type_to_search import install_type_to_search

        install_type_to_search(self, self.search_bar)

    def _build_list_pane(self) -> QFrame:
        frame = QFrame()
        frame.setFixedWidth(LEFT_NAV_LIST_SIDEBAR_WIDTH)
        frame.setObjectName("LibrarySidebar") 
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(15)

        # --- HEADER AREA ---
        header_layout = QHBoxLayout()
        self.list_title = QLabel("Library")
        self.list_title.setObjectName("ViewTitle")
        self.list_title.setProperty("class", "PageTitle")
        
        header_layout.addWidget(self.list_title)
        self.page_tour_help_btn = PageTourHelpButton(
            "library",
            area_display_name="Library",
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

        self.add_btn = QPushButton()
        self.add_btn.setIcon(
            themed_fa_icon("fa5s.plus", accent_icon_color(self._theme()), 16)
        )
        self.add_btn.setIconSize(QSize(16, 16))
        self.add_btn.setToolTip("Ingest New Document")
        apply_ghost_icon_button_style(self.add_btn, self._theme())
        self.add_btn.clicked.connect(self._browse_for_document)
        actions_outer.addWidget(self.add_btn)

        self.new_folder_btn = add_new_folder_header_button(
            actions_cluster,
            on_new_folder=lambda: self._folder_controller.prompt_create_folder()
            if self._folder_controller
            else None,
            theme_host=self,
        )
        
        layout.addLayout(header_layout)

        self.ingest_progress_row = IngestProgressRow()
        self.ingest_progress_row.hide()
        layout.addWidget(self.ingest_progress_row)

        # The Search Bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search titles or indexed text…")
        self.search_bar.setObjectName("LibrarySearchBar")
        self.search_bar.setToolTip("Search by document title or indexed text")
        layout.addWidget(self.search_bar)
        self._library_search_timer = QTimer(self)
        self._library_search_timer.setSingleShot(True)
        self._library_search_timer.timeout.connect(self._reload_library_sidebar)
        self.search_bar.textChanged.connect(self._on_library_search_changed)

        # Document List
        self.doc_list = QListWidget()
        self.doc_list.setObjectName("LibraryDocList")
        self.doc_list.itemClicked.connect(self._on_library_item_clicked)
        self.doc_list.itemSelectionChanged.connect(self._update_row_colors)

        self._active_folder_id = self.db.get_main_library_folder_id()
        self._sync_ingest_button_for_active_folder()
        self._folder_controller = SidebarFolderListController(
            scope="library",
            list_widget=self.doc_list,
            db=self.db,
            parent=self,
            append_item_row=self._append_library_doc_row,
            apply_menu_theme=self._apply_menu_theme,
            get_is_dark=lambda: getattr(self.window(), "_is_dark_theme", True),
            on_reload=self._reload_library_sidebar,
            on_active_folder_changed=self._set_active_folder_id,
            on_after_folder_delete=self._purge_deleted_library_files,
            sort_mode="name",
        )
        self.sort_btn = self._folder_controller.setup_sort_header_button(
            actions_cluster
        )

        self.doc_list.itemDoubleClicked.connect(self._on_library_item_double_clicked)

        layout.addWidget(self.doc_list)

        return frame

    def _setup_chat_with_doc_fab(self) -> None:
        """Floating action button: chat with the currently open library document."""
        self._chat_with_doc_btn = QPushButton(self.preview_stage)
        self._chat_with_doc_btn.setObjectName("LibraryChatWithDocFab")
        self._chat_with_doc_btn.setFixedSize(_CHAT_WITH_DOC_FAB_SIZE, _CHAT_WITH_DOC_FAB_SIZE)
        self._chat_with_doc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._chat_with_doc_btn.setToolTip("Chat with document")
        theme = self._theme(True)
        self._chat_with_doc_btn.setIcon(
            themed_fa_icon(
                "fa5s.comment-alt",
                theme.brand_fg,
                _CHAT_UTILITY_ICON_PX + 2,
            )
        )
        self._chat_with_doc_btn.setIconSize(QSize(_CHAT_UTILITY_ICON_PX + 2, _CHAT_UTILITY_ICON_PX + 2))
        self._chat_with_doc_btn.setStyleSheet(
            theme.style(
                CHAT_WITH_DOC_FAB,
                radius=_CHAT_WITH_DOC_FAB_SIZE // 2,
                object_name="LibraryChatWithDocFab",
            )
        )
        self._chat_with_doc_btn.clicked.connect(self._on_chat_with_document_clicked)
        self._chat_with_doc_btn.hide()
        self._chat_with_doc_btn.raise_()

    def _reposition_chat_with_doc_fab(self) -> None:
        btn = getattr(self, "_chat_with_doc_btn", None)
        host = getattr(self, "preview_stage", None)
        if btn is None or host is None:
            return
        x = max(0, host.width() - btn.width() - _CHAT_WITH_DOC_FAB_MARGIN)
        y = max(0, host.height() - btn.height() - _CHAT_WITH_DOC_FAB_MARGIN)
        btn.move(x, y)

    def _sync_chat_with_doc_fab_visibility(self) -> None:
        btn = getattr(self, "_chat_with_doc_btn", None)
        if btn is None:
            return
        if getattr(self, "_tour_chat_fab_preview_active", False):
            show = True
        else:
            show = bool(self.active_filename and validate_file_token(self.active_filename))
        btn.setVisible(show)
        if show:
            btn.raise_()
            self._reposition_chat_with_doc_fab()

    def _on_chat_with_document_clicked(self) -> None:
        filename = (self.active_filename or "").strip()
        if not filename or not validate_file_token(filename):
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self.window(),
            "Chat with document",
            f'Start a new conversation about "{filename}"?',
            is_dark=is_dark,
        )
        if not dlg.exec():
            return
        main_win = self.window()
        if main_win is not None and hasattr(main_win, "open_chat_with_library_document"):
            main_win.open_chat_with_library_document(filename)

    def eventFilter(self, obj, event) -> bool:
        if obj is getattr(self, "preview_stage", None) and event.type() == QEvent.Type.Resize:
            self._reposition_chat_with_doc_fab()
        return super().eventFilter(obj, event)
    
    def _build_preview_stage(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("LibraryPreviewStage")

        # Strip card styling; wallpaper host paints the mainstage background.
        frame.setStyleSheet("background: transparent; border: none;")

        outer_layout = QVBoxLayout(frame)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        mainstage_content = QWidget()
        mainstage_content.setObjectName("LibraryMainstageContent")
        layout = QVBoxLayout(mainstage_content)
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
        read_row.addWidget(self.text_align_btn)
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

        _preview_hdr_align = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        self.doc_title.setAlignment(_preview_hdr_align)
        self.doc_stats.setAlignment(_preview_hdr_align)

        title_host = QWidget()
        title_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        title_host.setMinimumWidth(0)
        self._preview_header_host = title_host
        title_vbox = QVBoxLayout(title_host)
        title_vbox.setContentsMargins(0, 0, 0, 0)
        title_vbox.setSpacing(4)
        title_vbox.addWidget(self.doc_title)
        title_vbox.addWidget(self.doc_stats)

        # Use the same width host as transcript content so header width tracks available space
        # but never exceeds 800px; this prevents narrow sizeHint collapse.
        self._preview_header_width_host = _LibraryTranscriptWidthHost(
            title_host, _CENTERED_COLUMN_MAX_WIDTH
        )
        header_layout.addWidget(self._preview_header_width_host, 1)
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

        transcript_inner = QFrame()
        transcript_inner.setObjectName("LibraryTranscriptCard")
        transcript_inner.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        transcript_inner.setMinimumWidth(0)
        self._library_transcript_card = transcript_inner
        ti_layout = QVBoxLayout(transcript_inner)
        ti_layout.setContentsMargins(0, 0, 0, 0)
        ti_layout.setSpacing(0)
        ti_layout.addWidget(self.text_preview, 1)

        self._transcript_width_host = _LibraryTranscriptWidthHost(
            transcript_inner, _CENTERED_COLUMN_MAX_WIDTH
        )
        layout.addWidget(self._transcript_width_host, stretch=1)

        self._library_transcript_wallpaper_host = TranscriptWallpaperHost(
            SURFACE_LIBRARY_PREVIEW,
            mainstage_content,
            parent=frame,
        )
        self._apply_library_layout_mode()
        self._refresh_readability_toolbar(self._theme().is_dark)
        self._apply_library_preview_readability()
        self._refresh_transcript_wallpaper()
        outer_layout.addWidget(self._library_transcript_wallpaper_host, stretch=1)

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

    def _preview_column_max_width(self) -> int:
        """Match ConversationsView.transcript_column_max_width nominal caps (800 / 1200)."""
        return (
            _FULL_WIDTH_COLUMN_MAX_WIDTH
            if self._layout_mode == LAYOUT_FULL_WIDTH
            else _CENTERED_COLUMN_MAX_WIDTH
        )

    def _apply_library_layout_mode(self) -> None:
        cap = self._preview_column_max_width()
        for host_attr in ("_transcript_width_host", "_preview_header_width_host"):
            host = getattr(self, host_attr, None)
            if host is not None:
                host.set_max_width_cap(cap)
                host.updateGeometry()
        self._refresh_layout_mode_button()

    def _set_preview_doc_title(self, title: str) -> None:
        """Show a filename in the preview header with wrap-friendly break points."""
        if not hasattr(self, "doc_title"):
            return
        plain = (title or "").strip()
        if plain and plain != "No Document Selected":
            self.doc_title.setToolTip(plain)
        else:
            self.doc_title.setToolTip("")
        self.doc_title.setText(_filename_title_for_label(plain or title))

    def _toggle_layout_mode(self) -> None:
        next_mode = (
            LAYOUT_CENTERED_COLUMN
            if self._layout_mode == LAYOUT_FULL_WIDTH
            else LAYOUT_FULL_WIDTH
        )
        self.set_layout_mode(next_mode)

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
            btn.setToolTip(
                f"Layout mode: Narrow column ({_CENTERED_COLUMN_MAX_WIDTH}px)"
            )
        else:
            btn.setIcon(
                tinted_svg_icon(
                    _LAYOUT_ICON_WIDE, icon_color, size=_CHAT_UTILITY_ICON_PX
                )
            )
            btn.setToolTip(
                f"Layout mode: Wide column ({_FULL_WIDTH_COLUMN_MAX_WIDTH}px)"
            )
        btn.setIconSize(QSize(_CHAT_UTILITY_ICON_PX, _CHAT_UTILITY_ICON_PX))
        btn.setFixedSize(_CHAT_UTILITY_BTN, _CHAT_UTILITY_BTN)
        btn.setStyleSheet(theme.style(GHOST_ICON_BUTTON))

    def _scaled_preview_font_pt(self) -> float:
        return max(8.0, min(28.0, _BASE_PREVIEW_FONT_PT * self._font_scale))

    def _reading_font_family(self) -> str:
        from core.app_settings import get_ui_reading_font
        from core.reading_fonts import reading_font_qt_family

        return reading_font_qt_family(get_ui_reading_font())

    def refresh_reading_font(self) -> None:
        self._apply_library_preview_readability()

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
        theme = self._theme(is_dark)
        if self._high_contrast_enabled:
            return theme.brand_fg if theme.is_dark else theme.text_primary
        return theme.text_primary

    def _apply_preview_paragraph_formats(self, doc) -> None:
        """Line height + transcript alignment in one merge per block (avoids format clobbering)."""
        ha = (
            Qt.AlignmentFlag.AlignJustify
            if self._transcript_alignment == ALIGN_JUSTIFY
            else Qt.AlignmentFlag.AlignLeft
        )
        fmt = QTextBlockFormat()
        fmt.setLineHeight(float(self._line_height_proportional_percent()), 1)
        fmt.setAlignment(ha)
        cur = QTextCursor(doc)
        cur.beginEditBlock()
        block = doc.firstBlock()
        while block.isValid():
            cur.setPosition(block.position())
            cur.mergeBlockFormat(fmt)
            block = block.next()
        cur.endEditBlock()

    def _style_library_transcript_card(self) -> None:
        card = getattr(self, "_library_transcript_card", None)
        if card is None:
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        enabled = self._library_transcript_background_enabled
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, enabled)
        card.setStyleSheet(
            theme.style(
                AGENT_MESSAGE_FRAME,
                enabled=enabled,
                high_contrast=self._high_contrast_enabled,
                object_name="LibraryTranscriptCard",
            )
        )
        layout = card.layout()
        if layout is not None:
            left, top, right, bottom = (
                _LIBRARY_TRANSCRIPT_CARD_MARGINS if enabled else (0, 0, 0, 0)
            )
            layout.setContentsMargins(left, top, right, bottom)

    def refresh_library_transcript_background(self) -> None:
        self._library_transcript_background_enabled = (
            get_ui_library_transcript_background()
        )
        self._apply_library_preview_readability()

    def _apply_library_preview_readability(self) -> None:
        if not hasattr(self, "text_preview"):
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        pt = self._scaled_preview_font_pt()
        family = self._reading_font_family()
        f = self.text_preview.font()
        f.setPointSizeF(pt)
        f.setFamily(family)
        self.text_preview.setFont(f)
        doc = self.text_preview.document()
        doc.setDefaultFont(f)
        self._apply_preview_paragraph_formats(doc)
        fg = self._preview_body_color(is_dark)
        self.text_preview.setStyleSheet(
            theme.style(
                TRANSPARENT_TEXT_PREVIEW,
                color=fg,
                font_pt=pt,
                font_family=family,
            )
        )
        self._style_library_transcript_card()
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

    def _cycle_transcript_alignment(self) -> None:
        self._transcript_alignment = (
            ALIGN_JUSTIFY
            if self._transcript_alignment == ALIGN_LEFT
            else ALIGN_LEFT
        )
        self._apply_library_preview_readability()
        self._refresh_readability_toolbar()

    def _refresh_transcript_wallpaper(self) -> None:
        bind_transcript_wallpaper_readability(
            getattr(self, "_library_transcript_wallpaper_host", None),
            high_contrast=self._high_contrast_enabled,
            reader_focus=self._focus_mode_enabled,
        )

    def _on_reader_focus_toggled(self, checked: bool) -> None:
        self._focus_mode_enabled = bool(checked)
        self._refresh_readability_toolbar()
        self._apply_preview_reader_focus_opacity()
        self._refresh_transcript_wallpaper()

    def _on_high_contrast_toggled(self, checked: bool) -> None:
        self._high_contrast_enabled = bool(checked)
        self._apply_library_preview_readability()
        self._refresh_readability_toolbar()
        self._refresh_transcript_wallpaper()

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
        self.line_height_btn.setIcon(
            tinted_svg_icon(_LINE_SPACING_ICON, icon_muted, size=_CHAT_UTILITY_ICON_PX)
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
        for btn in (
            self.line_height_btn,
            self.text_align_btn,
            self.reader_focus_btn,
            self.high_contrast_btn,
        ):
            btn.setStyleSheet(utility_icon_style)
        self._refresh_layout_mode_button(is_dark=is_dark)

    # --------------------------------------------------------- #
    #  LOGIC WIRING                                             #
    # --------------------------------------------------------- #

    def show_qube_documentation_folder(self) -> None:
        """Select the reserved Qube folder and show built-in help docs."""
        folder_id = self.db.get_qube_library_folder_id()
        folders = self.db.list_library_folders()
        qube = next((row for row in folders if row.get("id") == folder_id), None)
        if qube and qube.get("is_collapsed"):
            self.db.set_library_folder_collapsed(folder_id, False)
        self._set_active_folder_id(folder_id)
        self.refresh_library_list()

    def _apply_menu_theme(self, menu, is_dark: bool):
        """Standardizes the menu appearance with Prestige rounding and colors."""
        apply_prestige_kebab_menu_theme(menu, is_dark)

    def _set_active_folder_id(self, folder_id: str) -> None:
        self._active_folder_id = folder_id
        self._sync_ingest_button_for_active_folder()
        self._update_row_colors()

    def _sync_ingest_button_for_active_folder(self) -> None:
        folder_id = self._active_folder_id or self.db.get_main_library_folder_id()
        allowed = self.db.library_folder_allows_user_ingest(folder_id)
        self.add_btn.setEnabled(allowed)
        if allowed:
            self.add_btn.setToolTip("Ingest New Document")
        else:
            self.add_btn.setToolTip(
                "The Qube folder is reserved for app-generated knowledge. "
                "Select Main or another folder to add documents."
            )

    def _purge_deleted_library_files(self, filenames: list[str]) -> None:
        if not self.store:
            return
        for name in filenames:
            try:
                self.store.delete_document(name)
            except Exception as e:
                logger.exception("Failed to purge LanceDB document %s: %s", name, e)
        if self.active_filename in filenames:
            self.active_filename = None
            self.doc_title.setText("No Document Selected")
            self.doc_title.setToolTip("")
            self.doc_stats.setText("")
            self.text_preview.setHtml(
                "<center><h3>Document deleted.</h3></center>"
            )
            self._apply_library_preview_readability()
            self._sync_chat_with_doc_fab_visibility()

    def refresh_library_list(self):
        """Rebuild the list (respects search box)."""
        self._document_count = self.db.get_document_count()
        self._reload_library_sidebar()

    def _on_library_item_clicked(self, item) -> None:
        if self._folder_controller and self._folder_controller.handle_item_clicked(item):
            return
        self._on_document_selected(item)

    def _on_library_item_double_clicked(self, item) -> None:
        if self._folder_controller:
            self._folder_controller.handle_item_double_clicked(item)

    def _on_library_search_changed(self, _text: str) -> None:
        self._library_search_timer.stop()
        self._library_search_timer.start(280)

    def _reload_library_sidebar(self) -> None:
        """Rebuild sidebar: search → flat list; else folder-grouped browse."""
        if not self._folder_controller:
            return
        q = self.search_bar.text().strip() if getattr(self, "search_bar", None) else ""
        if q:
            content_hits: set[str] = set()
            if self.store and hasattr(self.store, "find_sources_matching_text"):
                try:
                    content_hits = self.store.find_sources_matching_text(q)
                except Exception as e:
                    logger.exception("Library sidebar content search failed: %s", e)
            try:
                docs = self.db.get_library_documents_for_sidebar_search(
                    q, list(content_hits), limit=200
                )
            except Exception as e:
                logger.exception("Library sidebar DB search failed: %s", e)
                docs = []
            self._folder_controller.reload_search_mode(docs)
        else:
            self._folder_controller.reload_browse_mode()
        self._update_row_colors()

    def _append_library_doc_row(self, doc: dict, indent_left: int = FOLDER_ROW_MARGIN_LEFT) -> None:
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
        item.setData(Qt.ItemDataRole.UserRole, doc["filename"])
        item.setData(SIDEBAR_ROW_KIND_ROLE, ROW_KIND_DOCUMENT)
        item.setData(SIDEBAR_ROW_PAYLOAD_ROLE, doc)

        row = QWidget()
        row.setObjectName("HistoryRowWidget")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        lay = QHBoxLayout(row)
        lay.setContentsMargins(indent_left, 0, 10, 0)
        lay.setSpacing(10)

        item_text = f"{doc['filename']} ({doc['file_size_kb']} KB)"
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)

        if is_precision_ingest_mode(doc.get("ingest_mode")):
            title_row.addWidget(make_pro_gem_badge(row))

        lbl = QLabel(item_text)
        lbl.setObjectName("HistoryRowTitle")
        blurb = (doc.get("summary_blurb") or "").strip()
        if blurb:
            lbl.setToolTip(blurb)
        title_row.addWidget(lbl, stretch=1)

        title_host = QWidget()
        title_host.setLayout(title_row)

        btn = QPushButton()
        btn.setObjectName("HistoryOptionsBtn")
        btn.setFixedSize(28, 28)
        btn.setIcon(themed_fa_icon("fa5s.ellipsis-v", icon_color, 16))
        btn.setIconSize(QSize(16, 16))
        btn.setStyleSheet(
            "QPushButton::menu-indicator { image: none; width: 0px; } "
            "QPushButton { border: none; background: transparent; padding: 0px; }"
        )
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setToolTip("Document actions")

        menu = QMenu(btn)
        if hasattr(self, "_apply_menu_theme"):
            self._apply_menu_theme(menu, is_dark)
        if self._folder_controller:
            self._folder_controller.register_menu(menu)

        rename_action = menu.addAction(
            themed_fa_icon("fa5s.edit", theme.color(LINK_ICON), 16), "Rename Document"
        )
        rename_action.triggered.connect(
            lambda _, fname=doc["filename"]: self._trigger_rename_document(fname)
        )

        if self._folder_controller:
            doc_folder_id = doc.get("folder_id") or self.db.get_main_library_folder_id()
            self._folder_controller.build_move_submenu_for_item(
                menu,
                doc_folder_id,
                lambda folder_id, fname=doc["filename"]: self._move_document_to_folder(
                    fname, folder_id
                ),
            )

        menu.addSeparator()

        delete_action = menu.addAction(
            themed_fa_icon("fa5s.trash-alt", theme.color(DANGER_ICON), 16), "Delete Document"
        )
        delete_action.triggered.connect(
            lambda _, fname=doc["filename"]: self._trigger_delete_document(fname)
        )

        btn.setMenu(menu)

        lay.addWidget(title_host)
        lay.addStretch()
        lay.addWidget(btn)

        item.setSizeHint(QSize(0, 45))
        self.doc_list.addItem(item)
        self.doc_list.setItemWidget(item, row)

    def _move_document_to_folder(self, filename: str, folder_id: str) -> None:
        if self.db.move_document_to_folder(filename, folder_id):
            self.refresh_library_list()

    def _update_row_colors(self):
        """Row title colors + action icons (QSS cannot target setItemWidget children)."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        target_list = getattr(self, "doc_list", getattr(self, "history_list", None))
        active_folder_id = None
        if target_list is getattr(self, "doc_list", None):
            active_folder_id = self._active_folder_id or self.db.get_main_library_folder_id()
        apply_sidebar_row_theme(
            target_list,
            is_dark=is_dark,
            theme=theme,
            active_folder_id=active_folder_id,
        )
        if target_list is not None:
            for badge in target_list.findChildren(QLabel):
                if badge.objectName() == "ProGemBadge":
                    apply_pro_gem_badge_theme(badge, parent=target_list)

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
                self._set_preview_doc_title(new_name)
                
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
                self.doc_title.setToolTip("")
                self.doc_stats.setText("")
                self.text_preview.setHtml("<center><h3>Document deleted.</h3></center>")
                self._apply_library_preview_readability()
                self._sync_chat_with_doc_fab_visibility()

            self.refresh_library_list()

    def _on_document_selected(self, item):
        doc_data = item.data(SIDEBAR_ROW_PAYLOAD_ROLE)
        if not isinstance(doc_data, dict):
            doc_data = {"filename": item.data(Qt.ItemDataRole.UserRole)}
        self.active_filename = doc_data['filename']
        
        self._set_preview_doc_title(self.active_filename)
        stats = (
            f"Size: {doc_data['file_size_kb']} KB | "
            f"Chunks Indexed: {doc_data['chunk_count']}"
        )
        if is_precision_ingest_mode(doc_data.get("ingest_mode")):
            stats = f"{stats} | Precision ingest"
        blurb = (doc_data.get("summary_blurb") or "").strip()
        if blurb:
            stats = f"{stats} | {blurb}"
        self.doc_stats.setText(stats)

        self.text_preview.setHtml("<center><h3>Reconstructing document from vector space...</h3></center>")
        self._apply_library_preview_readability()

        self._render_document_preview(self.active_filename)
        self._apply_library_preview_readability()
        self._sync_chat_with_doc_fab_visibility()

    def _render_document_preview(self, filename: str) -> None:
        """Load stitched preview from LanceDB (HTML when chunk metadata has breadcrumbs)."""
        if not self.store:
            self.text_preview.setPlainText("Error: Vector store not connected.")
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        content, is_html = self.store.reconstruct_document_for_preview(
            filename,
            breadcrumb_color=theme.text_secondary,
            body_color=self._preview_body_color(is_dark),
            font_pt=self._scaled_preview_font_pt(),
        )
        if is_html:
            self.text_preview.setHtml(content)
        else:
            self.text_preview.setPlainText(content)

    def _browse_for_document(self):
        """Choose indexing mode, open file dialog, check duplicates, and ingest."""
        folder_id = self._active_folder_id or self.db.get_main_library_folder_id()
        if not self.db.library_folder_allows_user_ingest(folder_id):
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self,
                "Cannot Add Here",
                "The Qube folder is reserved for knowledge Qube creates automatically. "
                "Select the Main folder or another folder to add your own documents.",
                is_dark,
            ).exec()
            return

        from ui.bootstrap_feature_prompts import ensure_search_models_for_feature

        if not ensure_search_models_for_feature(
            self.window(),
            feature_label="Library document upload",
        ):
            return

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        mode_dialog = LibraryIngestModeDialog(
            self,
            is_dark=is_dark,
        )
        if not mode_dialog.exec():
            logger.info("Ingestion cancelled at indexing mode chooser.")
            return
        ingest_mode = mode_dialog.selected_mode()
        if not ingest_mode:
            return

        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Documents to Ingest", "", "Documents (*.txt *.md *.pdf *.epub)"
        )
        if not files:
            return

        paths = [Path(f) for f in files]
        
        # 1. Check if any selected files already exist in our SQLite registry
        existing_files = []
        current_docs = self.db.get_all_library_document_filenames()

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
        self.begin_ingest_progress_ui()
        self.add_btn.setEnabled(False)
        
        logger.info(
            "Emitting %d files to main pipeline for ingestion (mode=%s).",
            len(paths),
            ingest_mode,
        )
        self._had_ingestion_error = False
        folder_id = self._active_folder_id or self.db.get_main_library_folder_id()
        self.ingest_requested.emit(paths, folder_id, ingest_mode)

    # --- UI Receivers for Worker Progress ---
    def begin_ingest_progress_ui(self, *, detail: str = "") -> None:
        """Show ingest progress row with a spinner until the bar advances."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self.ingest_progress_row.apply_theme(is_dark)
        self.ingest_progress_row.begin(detail=detail)
        self.ingest_progress_row.show()

    def _hide_ingest_progress_ui(self) -> None:
        self.ingest_progress_row.finish()
        self.ingest_progress_row.hide()

    def update_ingestion_progress(self, percent: int, *, detail: str | None = None):
        if not self.ingest_progress_row.isVisible():
            self.begin_ingest_progress_ui(detail=detail or "")
        self.ingest_progress_row.update_progress(percent, detail=detail)

    def set_ingest_progress_detail(self, detail: str) -> None:
        if not self.ingest_progress_row.isVisible():
            self.begin_ingest_progress_ui(detail=detail)
        else:
            self.ingest_progress_row.set_detail(detail)

    def show_error(self, error_msg: str, *, title: str = "Ingestion Failed"):
        """Displays ingestion errors to the user and resets the UI."""
        self._had_ingestion_error = True 

        self._hide_ingest_progress_ui()
        self._sync_ingest_button_for_active_folder()

        is_dark = getattr(self.window(), '_is_dark_theme', True)
        dialog = PrestigeDialog(self.window(), title, str(error_msg), is_dark)
        dialog.exec()

    def finish_reindex_ui(self) -> None:
        """Hide re-embed progress after a mode switch without ingestion dialogs."""
        self._hide_ingest_progress_ui()
        self._sync_ingest_button_for_active_folder()
        self.refresh_library_list()

    def complete_ingestion(self, total_chunks: int, *, warn_if_empty: bool = True):
        self._hide_ingest_progress_ui()
        self._sync_ingest_button_for_active_folder()
        
        if self._had_ingestion_error:
            return 
            
        self.refresh_library_list()
        
        if total_chunks == 0 and warn_if_empty:
            is_dark = getattr(self.window(), '_is_dark_theme', True)
            msg = (
                "Process finished, but 0 chunks were added. This usually means the "
                "file was already in the database, or it is a scanned PDF with no "
                "readable text."
            )
            dialog = PrestigeDialog(self.window(), "No Data Added", msg, is_dark)
            dialog.exec()

    def refresh_menu_themes(self, is_dark: bool) -> None:
        if self._folder_controller:
            self._folder_controller.refresh_menu_themes(is_dark)
        for i in range(self.doc_list.count()):
            item = self.doc_list.item(i)
            widget = self.doc_list.itemWidget(item)
            if widget:
                btn = widget.findChild(QPushButton, "HistoryOptionsBtn")
                if btn and btn.menu():
                    self._apply_menu_theme(btn.menu(), is_dark)

    def refresh_button_themes(self, is_dark: bool):
        """Dynamically updates the color of the Add Document button."""
        if hasattr(self, "ingest_progress_row"):
            self.ingest_progress_row.apply_theme(is_dark)

        theme = self._theme(is_dark)
        base_icon_color = accent_icon_color(theme)

        if hasattr(self, "add_btn"):
            self.add_btn.setIcon(themed_fa_icon("fa5s.plus", base_icon_color, 16))
            apply_ghost_icon_button_style(self.add_btn, theme)
        if hasattr(self, "new_folder_btn"):
            self.new_folder_btn.setIcon(
                themed_fa_icon("fa5s.folder-plus", base_icon_color, 16)
            )
            apply_ghost_icon_button_style(self.new_folder_btn, theme)
        if hasattr(self, "sort_btn"):
            self.sort_btn.setIcon(themed_fa_icon("fa5s.sort", base_icon_color, 16))
            apply_ghost_icon_button_style(self.sort_btn, theme, hide_menu_indicator=True)

        self._refresh_readability_toolbar(is_dark=is_dark)
        if hasattr(self, "font_minus_btn"):
            font_btn_style = readability_font_pair_stylesheet(
                is_dark=is_dark, theme=theme, button_px=_CHAT_UTILITY_BTN
            )
            self.font_minus_btn.setStyleSheet(font_btn_style)
            self.font_plus_btn.setStyleSheet(font_btn_style)

        if hasattr(self, "text_preview"):
            if self.active_filename and self.store:
                self._render_document_preview(self.active_filename)
            self._apply_library_preview_readability()

        self._apply_library_list_surface(is_dark)
        self._refresh_transcript_wallpaper()

    def _apply_library_list_surface(self, is_dark: bool) -> None:
        """Sidebar list tint: QListWidget paints in an internal viewport — set palette on list + viewport."""
        bg = self._theme(is_dark).qcolor_role(LIST_SURFACE)
        if hasattr(self, "list_pane"):
            p = self.list_pane
            p.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            p.setAutoFillBackground(True)
            pa = p.palette()
            pa.setColor(QPalette.ColorRole.Window, bg)
            p.setPalette(pa)
        if not hasattr(self, "doc_list"):
            return
        w = self.doc_list
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
        super().showEvent(event)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_library_list_surface(is_dark)
        self._reposition_chat_with_doc_fab()