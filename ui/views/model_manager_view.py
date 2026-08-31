"""Model Manager: Hub "app store" browser, README, quant selection, and downloads."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("Qube.ModelManager")

import qtawesome as qta
from PyQt6.QtGui import QColor, QFont, QPalette, QTextDocument, QPainter, QIcon, QDesktopServices
from PyQt6.QtCore import QEvent, Qt, QThread, QSize, QTimer, QUrl, QRect, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QLayout,
    QLayoutItem,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QPushButton,
    QLineEdit,
    QProgressBar,
    QListWidget,
    QListView,
    QListWidgetItem,
    QComboBox,
    QTextBrowser,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from core.app_settings import (
    get_llm_models_dir,
    get_model_manager_hardware_suggestions,
    is_secondary_gguf_shard,
    missing_gguf_shards,
    parse_gguf_shard_info,
    resolve_internal_model_path,
    set_internal_model_path,
)
from core.hf_hub_errors import HubErrorInfo, coerce_hub_error
from core.qube_verified_models import branding_for_entry, load_qube_verified_models
from core.catalog_hardware_recommendation import (
    CatalogFitLevel,
    build_catalog_recommendation_plan,
    sort_entries_by_hardware_fit,
)
from core.hardware_capability_profile import format_tier_detail
from core.gguf_quant import parse_quant_from_gguf_path
from core.memory_budget_profile import detect_memory_budget_profile, experience_for_download
from core.quant_recommendation import (
    QuantFileRecommendation,
    QuantRecommendationContext,
    QuantRecommendationPlan,
    QuantBadgeKind,
    build_context_from_hub_meta,
    recommend_quants,
)
from core.hub_readme_html import hf_readme_markdown_to_safe_html, strip_hub_readme_preamble
from core.hf_publisher_branding import HuggingFaceBrandingResolver, owner_from_repo_id
from core.model_capability_service import ModelCapabilityService
from core.publisher_guidance_service import PublisherGuidanceService
from core.richtext_styles import markdown_document_stylesheet
from core.theme.accessors import theme_for
from core.theme.svg_icons import themed_fa_icon
from core.theme.view_theme import view_resolved_theme
from core.theme.widget_styles import (
    ACCENT_CHIP,
    ACCENT_ICON,
    CAPABILITY_CHIP,
    COMBO_POPUP_LIST,
    COMBO_POPUP_SHELL,
    COMBO_POPUP_VIEWPORT,
    CONNECTIVITY_ERROR_BANNER,
    DIVIDER_ACCENT,
    LINK_ICON,
    LIST_SURFACE,
    META_HINT,
    META_LABEL,
    MODEL_HUB_OFFICIAL_BADGE,
    MUTED_ICON,
    MUTED_STATUS,
    QUANT_BADGE_PRIMARY,
    QUANT_BADGE_SECONDARY,
    STAGE_SURFACE,
    SUCCESS_STATUS,
    WARNING_STATUS,
    HUB_MUTED_HINT,
    HUB_MUTED_ROW,
)
from ui.sidebar_dimensions import LEFT_NAV_LIST_SIDEBAR_WIDTH
from ui.components.page_tour_help_button import PageTourHelpButton
from ui.components.prestige_dialog import PrestigeDialog
from ui.components.hub_error_dialog import HubErrorDialog
from ui.components.brand_buttons import (
    apply_brand_primary,
    apply_brand_success,
    apply_brand_danger,
)
from workers.hf_connectivity_probe_worker import HfConnectivityProbeWorker
from workers.hf_model_search_worker import HfModelSearchWorker
from workers.hf_model_meta_worker import HfModelMetaWorker
from workers.hf_readme_worker import HfReadmeWorker
from workers.hf_repo_files_worker import HfRepoFilesWorker
from workers.model_download_worker import HuggingFaceGgufDownloadWorker
from core.paths import resource_path

# Extra display data on Hub .gguf combo rows (file size, right-aligned in popup).
HUB_FILE_COMBO_SIZE_ROLE = int(Qt.ItemDataRole.UserRole) + 42
HUB_FILE_COMBO_BYTES_ROLE = int(Qt.ItemDataRole.UserRole) + 43
HUB_FILE_COMBO_SHARD_ENTRIES_ROLE = int(Qt.ItemDataRole.UserRole) + 44
HUB_FILE_COMBO_REC_ROLE = int(Qt.ItemDataRole.UserRole) + 45
MODEL_MANAGER_CONTENT_WIDTH_SCALE = 1.2
HUB_ROW_REPO_ROLE = int(Qt.ItemDataRole.UserRole)
HUB_ROW_TITLE_ROLE = int(Qt.ItemDataRole.UserRole) + 1
HUB_ROW_DESC_ROLE = int(Qt.ItemDataRole.UserRole) + 2
HUB_ROW_CAPS_ROLE = int(Qt.ItemDataRole.UserRole) + 3
HUB_ROW_UPDATED_ROLE = int(Qt.ItemDataRole.UserRole) + 4
HUB_ROW_VERIFIED_ROLE = int(Qt.ItemDataRole.UserRole) + 5
HUB_ROW_BRANDING_ROLE = int(Qt.ItemDataRole.UserRole) + 6
HUB_ROW_DOWNLOAD_REPO_ROLE = int(Qt.ItemDataRole.UserRole) + 7
HUB_ROW_CATALOG_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 8
HUB_ROW_GGUF_REPOS_ROLE = int(Qt.ItemDataRole.UserRole) + 9
HUB_ROW_IS_CATALOG_ROLE = int(Qt.ItemDataRole.UserRole) + 10
HUB_ROW_HARDWARE_FIT_ROLE = int(Qt.ItemDataRole.UserRole) + 11
HUB_SEARCH_PAGE_SIZE = 20


def _resolve_bundled_asset_url(asset_url: str) -> Path | None:
    """Resolve ``/assets/...`` (or absolute path) to a bundled read-only file."""
    raw = str(asset_url or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return path
    rel = raw.lstrip("/")
    if not rel:
        return None
    candidate = resource_path(*rel.split("/"))
    return candidate if candidate.is_file() else None


def _resolve_hub_brand_logo(logo_url: str) -> Path | None:
    """Resolve branding logo path (/assets/..., absolute, or cached avatar file)."""
    return _resolve_bundled_asset_url(logo_url)


def _default_hub_fallback_logo() -> Path | None:
    """Generic HF avatar when publisher logo is missing."""
    for parts in (
        ("assets", "logos", "hf-logo.svg"),
        ("assets", "icons", "hf-logo.svg"),
    ):
        candidate = resource_path(*parts)
        if candidate.is_file():
            return candidate
    return None


def _hub_file_combo_list_qss(theme) -> str:
    """Widget-local QSS for the combo's QAbstractItemView (app QSS misses detached popups on many styles)."""
    return theme.style(COMBO_POPUP_LIST)


def _hub_file_combo_viewport_qss(theme) -> str:
    return theme.style(COMBO_POPUP_VIEWPORT)


class HubFileComboDelegate(QStyledItemDelegate):
    """Popup rows: format chip, model/quant chips, size, and chevron."""

    _GAP = 10
    _SIZE_MAX = 160

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        raw = index.data(int(HUB_FILE_COMBO_SIZE_ROLE))
        if raw is None:
            super().paint(painter, option, index)
            return
        size_label = str(raw).strip()
        if not size_label:
            super().paint(painter, option, index)
            return

        path = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(path, str) or not path.lower().endswith(".gguf"):
            path = index.data(Qt.ItemDataRole.DisplayRole)
        path_s = path if isinstance(path, str) else (str(path) if path is not None else "")
        painter.save()

        rect = option.rect
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(rect, option.palette.highlight())
            pen = option.palette.color(QPalette.ColorRole.HighlightedText)
        else:
            painter.fillRect(rect, option.palette.base())
            pen = option.palette.color(QPalette.ColorRole.Text)
        painter.setPen(pen)

        fm = option.fontMetrics
        chevron_w = 18
        size_text_w = fm.horizontalAdvance(size_label) + 14
        size_w = max(72, min(size_text_w, self._SIZE_MAX, max(rect.width() // 3, 84)))
        right_edge = rect.right() - 6
        chev_r = QRect(right_edge - chevron_w + 1, rect.top(), chevron_w, rect.height())
        size_r = QRect(chev_r.left() - size_w - 8, rect.top(), size_w, rect.height())

        left = rect.left() + 8
        chip_h = max(18, rect.height() - 12)
        chip_y = rect.top() + (rect.height() - chip_h) // 2

        def draw_chip(text: str, width_pad: int, role: QPalette.ColorRole) -> int:
            nonlocal left
            tw = fm.horizontalAdvance(text) + width_pad
            cw = max(48, tw)
            cr = QRect(left, chip_y, cw, chip_h)
            c = option.palette.color(role)
            c.setAlpha(55)
            painter.setBrush(c)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(cr, 9, 9)
            painter.setPen(option.palette.color(QPalette.ColorRole.Text))
            painter.drawText(cr, int(Qt.AlignmentFlag.AlignCenter), text)
            left += cw + 8
            return cw

        draw_chip("GGUF", 18, QPalette.ColorRole.Highlight)
        parsed = parse_quant_from_gguf_path(path_s)
        quant = parsed.normalized if parsed is not None else "AUTO"
        rec_raw = index.data(int(HUB_FILE_COMBO_REC_ROLE))
        rec = rec_raw if isinstance(rec_raw, QuantFileRecommendation) else None
        badge_text = ""
        badge_primary = False
        if rec is not None and rec.badge != QuantBadgeKind.NONE and rec.badge_text:
            badge_text = rec.badge_text
            badge_primary = rec.badge == QuantBadgeKind.RECOMMENDED
        badge_w = 0
        if badge_text:
            badge_w = max(72, fm.horizontalAdvance(badge_text) + 24) + 8
        quant_w = max(48, fm.horizontalAdvance(quant) + 20)
        model_r = QRect(
            left,
            rect.top(),
            max(20, size_r.left() - left - quant_w - badge_w - 16),
            rect.height(),
        )
        model_name = Path(path_s).name
        mono_font = painter.font()
        mono_font.setFamilies(["Consolas", "Monospace"])
        painter.setFont(mono_font)
        elided = fm.elidedText(model_name, Qt.TextElideMode.ElideMiddle, model_r.width())
        painter.drawText(model_r, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), elided)
        left = model_r.right() + 8
        draw_chip(quant, 20, QPalette.ColorRole.Mid)
        if badge_text:
            host = option.widget
            win = host.window() if host is not None else None
            is_dark = bool(getattr(win, "_is_dark_theme", True))
            theme = view_resolved_theme(host, is_dark=is_dark)
            if badge_primary:
                bg = theme.qcolor(theme.accent)
                bg.setAlpha(90)
                fg = theme.qcolor(theme.text_on_accent)
            else:
                bg = theme.qcolor(theme.text_muted)
                bg.setAlpha(70)
                fg = theme.qcolor(theme.text_primary)
            tw = fm.horizontalAdvance(badge_text) + 22
            cw = max(72, tw)
            cr = QRect(left, chip_y, cw, chip_h)
            painter.setBrush(bg)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(cr, 9, 9)
            painter.setPen(fg)
            painter.drawText(cr, int(Qt.AlignmentFlag.AlignCenter), badge_text)
            left += cw + 8
        painter.drawText(size_r, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), size_label)
        painter.drawText(chev_r, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter), "▾")
        painter.restore()


class HubFileComboBox(QComboBox):
    """Repaints popup after open — Qt reparents the list; shell stays white without a deferred polish."""

    def __init__(self, manager: "ModelManagerView", parent: QWidget | None = None):
        super().__init__(parent)
        self._hub_manager = manager

    def showPopup(self) -> None:
        super().showPopup()
        m = self._hub_manager
        if m is not None:
            QTimer.singleShot(0, m._on_hub_file_combo_popup_opened)

    def wheelEvent(self, event) -> None:
        # Let the detail scroll area handle the wheel — do not change quant selection.
        event.ignore()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        m = self._hub_manager
        if m is None:
            return
        pm = getattr(m, "_hub_combo_chevron_pixmap", None)
        if pm is None:
            return
        r = self.rect()
        x = r.right() - 16 - pm.width() // 2
        y = r.center().y() - pm.height() // 2
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawPixmap(x, y, pm)
        p.end()


class FlowLayout(QLayout):
    """Minimal flow layout for wrapping chip rows on narrow widths."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientations:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize(0, 0)
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        m = self.contentsMargins()
        start_x = rect.x() + m.left()
        x = start_x
        y = rect.y() + m.top()
        max_right = rect.right() - m.right()

        line_items: list[tuple[QLayoutItem, QSize]] = []
        line_h = 0

        def flush_line() -> None:
            nonlocal x, y, line_h, line_items
            if not line_items:
                return
            cx = start_x
            for it, sz in line_items:
                if not test_only:
                    dy = (line_h - sz.height()) // 2
                    it.setGeometry(QRect(cx, y + dy, sz.width(), sz.height()))
                cx += sz.width() + self.spacing()
            y += line_h + self.spacing()
            x = start_x
            line_h = 0
            line_items = []

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width()
            if line_items and next_x > max_right:
                flush_line()
                next_x = x + hint.width()
            line_items.append((item, hint))
            x = next_x + self.spacing()
            line_h = max(line_h, hint.height())

        # Last line should not include extra trailing spacing in measured height.
        if line_items:
            if not test_only:
                cx = start_x
                for it, sz in line_items:
                    dy = (line_h - sz.height()) // 2
                    it.setGeometry(QRect(cx, y + dy, sz.width(), sz.height()))
                    cx += sz.width() + self.spacing()
            y += line_h

        return (y - rect.y()) + m.bottom()


class ModelManagerView(QWidget):
    """Hub browser: search, README, file list, and GGUF downloads."""

    native_library_changed = pyqtSignal()
    download_succeeded = pyqtSignal(str)  # basename of saved model file

    _HUB_LIST_MAX_LINES = 3

    @staticmethod
    def _scaled_content_width(px: int) -> int:
        return max(1, int(round(px * MODEL_MANAGER_CONTENT_WIDTH_SCALE)))

    @staticmethod
    def _configure_wrapping_label(lbl: QLabel) -> None:
        """Let labels grow vertically with word wrap instead of clipping in tight layouts."""
        lbl.setWordWrap(True)
        pol = lbl.sizePolicy()
        pol.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        pol.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        lbl.setSizePolicy(pol)
        lbl.setMinimumWidth(0)

    def _theme(self, is_dark: bool | None = None):
        return view_resolved_theme(self, is_dark=is_dark)

    def _refresh_download_options_card_geometry(self) -> None:
        if not hasattr(self, "download_options_card"):
            return
        for w in (
            getattr(self, "hub_quant_hint_lbl", None),
            getattr(self, "system_chip_lbl", None),
            getattr(self, "hub_quant_rationale_row", None),
            getattr(self, "hub_quant_rationale_lbl", None),
            getattr(self, "download_status", None),
        ):
            if w is not None:
                w.updateGeometry()
        self.download_options_card.updateGeometry()
        self.download_options_card.adjustSize()
        if hasattr(self, "detail_cards_content"):
            self.detail_cards_content.updateGeometry()
        if hasattr(self, "detail_scroll"):
            self.detail_scroll.updateGeometry()

    @staticmethod
    def _elide_single_line(text: str, lbl: QLabel) -> str:
        fm = lbl.fontMetrics()
        width = max(8, lbl.width())
        return fm.elidedText(str(text or ""), Qt.TextElideMode.ElideRight, width)

    @staticmethod
    def _capability_chip_spec(cap: str) -> tuple[str, str]:
        c = str(cap or "").strip()
        low = c.lower()
        if "audio" in low:
            return ("fa5s.volume-up", "capability audio")
        if "tts" in low:
            return ("fa5s.volume-up", "capability tts")
        if "stt" in low or "asr" in low:
            return ("fa5s.microphone", "capability stt")
        if "vision" in low:
            return ("fa5s.eye", "capability vision")
        if "tool" in low:
            return ("fa5s.wrench", "capability tools")
        if "reason" in low:
            return ("fa5s.brain", "capability reasoning")
        if "code" in low or "coder" in low:
            return ("fa5s.code", "capability coding")
        if "chat" in low:
            return ("fa5s.comments", "capability chat")
        return ("fa5s.star", "capability general")

    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        self._llm = workers.get("llm")
        self._download_worker: HuggingFaceGgufDownloadWorker | None = None
        self._list_worker: HfRepoFilesWorker | None = None
        self._search_worker: HfModelSearchWorker | None = None
        self._readme_worker: HfReadmeWorker | None = None
        self._meta_worker: HfModelMetaWorker | None = None
        self._curated_meta_worker: HfModelMetaWorker | None = None
        self._curated_meta_queue: list[str] = []
        self._capability_service = ModelCapabilityService()
        self._publisher_guidance_service = PublisherGuidanceService()
        self._branding_resolver = HuggingFaceBrandingResolver()
        self._download_ui_cancel_mode = False
        self._download_ui_load_mode = False
        self._download_queue_paths: list[tuple[str, int | None]] = []
        self._download_queue_index: int = 0
        self._download_completed_paths: list[str] = []
        self._download_failed_path: str | None = None
        self._download_total_bytes: int = 0
        self._download_completed_bytes: int = 0
        self._download_current_bytes_total: int = 0
        self._download_current_path: str = ""
        self._search_seq = 0
        self._detail_seq = 0
        self._current_repo_id = ""
        self._catalog_gguf_repos: tuple[str, ...] = ()
        self._catalog_gguf_repo_index: int = 0
        self._last_readme_markdown: str | None = None
        # Strong refs to QThread instances that are still running after we replace them — never
        # drop the last reference while isRunning() or Qt aborts with "Destroyed while still running".
        self._retired_hf_threads: list[QThread] = []
        self._current_meta_capabilities: list[str] = []
        self._quant_rec_context: QuantRecommendationContext | None = None
        self._quant_rec_plan: QuantRecommendationPlan | None = None
        self._hub_meta_snapshot: dict | None = None
        self._search_models_cache: list[dict] = []
        self._search_visible_count: int = 0
        self._catalog_hardware_plan = None
        self._hardware_suggestions_enabled = get_model_manager_hardware_suggestions()
        self._hardware_suggestions_dirty = False
        self._hub_probe_worker: HfConnectivityProbeWorker | None = None
        self._hub_reachable: bool | None = None
        self._hub_status_detail: str = ""
        self._pending_download_retry: bool = False
        self._pending_hub_redownload: tuple[str, str] | None = None
        self._tour_load_more_preview_active: bool = False

        os.makedirs(get_llm_models_dir(), exist_ok=True)
        self._setup_ui()
        self._populate_editors_picks()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_hub_muted_labels(is_dark)
        self._apply_hub_metadata_styles(is_dark)
        self.refresh_button_themes(is_dark)
        self._apply_hub_list_surface(is_dark)
        self._apply_hub_file_combo_popup_theme(is_dark)
        self._apply_hub_combo_chevron(is_dark)
        self._apply_hub_connectivity_banner(is_dark)
        self._update_hub_row_colors()
        QTimer.singleShot(0, self._refresh_hub_row_heights)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if getattr(self, "_hardware_suggestions_dirty", False):
            self._hardware_suggestions_dirty = False
            if not self.hub_search_edit.text().strip():
                self._populate_editors_picks()
        else:
            enabled = get_model_manager_hardware_suggestions()
            if enabled != self._hardware_suggestions_enabled:
                self._hardware_suggestions_enabled = enabled
                if not self.hub_search_edit.text().strip():
                    self._populate_editors_picks()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_hub_list_surface(is_dark)
        self._apply_hub_metadata_styles(is_dark)
        self._apply_hub_combo_chevron(is_dark)
        self._apply_hub_connectivity_banner(is_dark)
        self._start_hub_connectivity_probe()
        QTimer.singleShot(0, self._refresh_hub_row_heights)

    def eventFilter(self, obj, event) -> bool:
        if (
            hasattr(self, "hub_model_list")
            and obj is self.hub_model_list.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self._refresh_hub_row_heights()
        return super().eventFilter(obj, event)

    def _hub_viewport_width_for_rows(self) -> int:
        """Inner width for wrapped Hub labels (viewport may be 0 before first layout)."""
        if not hasattr(self, "hub_model_list"):
            return 248
        vw = self.hub_model_list.viewport().width()
        if vw < 48:
            outer = self.hub_model_list.width()
            if outer >= 48:
                vw = max(48, outer - 8)
        if vw < 48:
            vw = 248
        return vw

    def _style_hub_title_label(self, lbl: QLabel, item: QListWidgetItem) -> None:
        """Explicit font + foreground (Library row pattern): QSS alone drifts when the app sheet is replaced on toggle."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        if item.isSelected():
            fg = theme.text_on_accent if theme.is_dark else theme.text_primary
        else:
            fg = theme.text_primary
        lbl.setStyleSheet(
            f"color: {fg}; background: transparent; border: none; "
            'font-size: 13px; font-weight: 500; font-family: "Inter"; '
            "padding: 0px; margin: 0px;"
        )

    @staticmethod
    def _hub_uniform_label_content_height(font: QFont, width_px: int) -> float:
        """Height for exactly `_HUB_LIST_MAX_LINES` lines using the same QTextDocument layout as elision."""
        doc = QTextDocument()
        doc.setDocumentMargin(0.0)
        doc.setDefaultFont(font)
        n = ModelManagerView._HUB_LIST_MAX_LINES
        doc.setPlainText("\n".join(["x"] * n))
        w = max(1, int(width_px))
        doc.setTextWidth(w)
        return float(doc.size().height())

    @staticmethod
    def _hub_elide_plain_text_for_height(
        full: str, font: QFont, width_px: int, max_height_px: float
    ) -> str:
        """Shrink text with an ellipsis so QTextDocument height stays within max_height_px."""
        if not full:
            return ""
        w = max(1, int(width_px))
        doc = QTextDocument()
        doc.setDocumentMargin(0.0)
        doc.setDefaultFont(font)
        doc.setPlainText(full)
        doc.setTextWidth(w)
        if doc.size().height() <= max_height_px + 0.5:
            return full

        ell = "…"
        lo, hi = 0, len(full)
        best = ell
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = full[:mid].rstrip()
            if mid < len(full):
                cand = cand + ell
            doc.setPlainText(cand)
            doc.setTextWidth(w)
            if doc.size().height() <= max_height_px + 0.5:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _disconnect_hf_worker_signals(self, worker: QThread | None) -> None:
        """Disconnect Model Manager slots so retired workers cannot touch the UI."""
        if worker is None:
            return
        for name in (
            "finished_ok",
            "failed",
            "progress_pct",
            "status_message",
            "insufficient_space_error",
            "download_cancelled",
        ):
            sig = getattr(worker, name, None)
            if sig is not None:
                try:
                    sig.disconnect()
                except TypeError:
                    pass
        try:
            worker.finished.disconnect()
        except TypeError:
            pass

    def _finalize_retired_hf_thread(self, worker: QThread) -> None:
        try:
            self._retired_hf_threads.remove(worker)
        except ValueError:
            pass
        worker.deleteLater()

    def _retire_hf_thread(self, worker: QThread | None) -> None:
        """Keep a reference until QThread finishes, then deleteLater (safe replacement)."""
        if worker is None:
            return
        self._disconnect_hf_worker_signals(worker)
        if worker.isRunning():
            worker.finished.connect(
                lambda w=worker: self._finalize_retired_hf_thread(w)
            )
            self._retired_hf_threads.append(worker)
        else:
            worker.deleteLater()

    def shutdown_hf_workers(self) -> None:
        """Cancel/stop Hugging Face QThreads so application exit is not blocked."""
        if hasattr(self, "_search_timer"):
            self._search_timer.stop()

        dw = getattr(self, "_download_worker", None)
        if dw is not None and dw.isRunning():
            self._disconnect_hf_worker_signals(dw)
            dw.cancel()
            if not dw.wait(10_000):
                logger.warning("Download worker did not finish within 10s during shutdown.")

        for attr in ("_search_worker", "_readme_worker", "_list_worker", "_meta_worker", "_curated_meta_worker", "_hub_probe_worker"):
            w = getattr(self, attr, None)
            if w is None or not w.isRunning():
                continue
            self._disconnect_hf_worker_signals(w)
            w.requestInterruption()
            if not w.wait(5000):
                logger.warning("%s did not finish within 5s during shutdown.", attr)

        for w in list(self._retired_hf_threads):
            if w is None or not w.isRunning():
                continue
            self._disconnect_hf_worker_signals(w)
            if hasattr(w, "cancel"):
                w.cancel()
            w.requestInterruption()
            w.wait(3000)

    def _apply_hub_row_size_hint(self, item: QListWidgetItem, row: QWidget) -> None:
        """High-density card sizing + one-line elision for title/description."""
        lay = row.layout()
        if not lay:
            return
        vw = self._hub_viewport_width_for_rows()
        row.setFixedWidth(vw)
        row.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        item.setSizeHint(QSize(vw, 72))

        title_lbl = row.findChild(QLabel, "HubModelRowTitle")
        desc_lbl = row.findChild(QLabel, "HubModelRowDescription")
        ts_lbl = row.findChild(QLabel, "HubModelRowTimestamp")
        if not title_lbl or not desc_lbl or not ts_lbl:
            return

        self._style_hub_title_label(title_lbl, item)

        title_lbl.setText(self._elide_single_line(item.data(HUB_ROW_TITLE_ROLE), title_lbl))
        desc_lbl.setText(self._elide_single_line(item.data(HUB_ROW_DESC_ROLE), desc_lbl))
        ts_lbl.setText(self._elide_single_line(item.data(HUB_ROW_UPDATED_ROLE), ts_lbl))

        title_lbl.updateGeometry()
        desc_lbl.updateGeometry()
        ts_lbl.updateGeometry()
        row.updateGeometry()

    def _apply_hub_row_surface(self, row: QWidget) -> None:
        row.setStyleSheet(
            "background-color: transparent; border: none;"
        )
        row.update()

    def _official_badge_stylesheet(self, *, is_dark: bool | None = None) -> str:
        theme = self._theme(is_dark)
        fg = theme.color(MODEL_HUB_OFFICIAL_BADGE)
        return (
            f"QLabel {{ color: {fg}; font-size: 10px; font-weight: 700; "
            f"background: transparent; }}"
        )

    def _apply_hub_official_badge_theme(
        self, row: QWidget, *, is_dark: bool | None = None
    ) -> None:
        badge = row.findChild(QLabel, "HubModelRowOfficialBadge")
        if badge is None:
            return
        badge.setStyleSheet(self._official_badge_stylesheet(is_dark=is_dark))

    def _refresh_hub_row_heights(self) -> None:
        """Recompute after resize, first show, or when viewport width was unknown during populate."""
        if not hasattr(self, "hub_model_list"):
            return
        for i in range(self.hub_model_list.count()):
            it = self.hub_model_list.item(i)
            row = self.hub_model_list.itemWidget(it)
            if it is not None and row is not None:
                self._apply_hub_row_size_hint(it, row)
        self.hub_model_list.doItemsLayout()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        # Keep right breathing room, but let the sidebar reach top and bottom like Conversations.
        main_layout.setContentsMargins(0, 0, 40, 0)
        main_layout.setSpacing(16)
        # --- Hub "app store" row (fixed sidebar + expanding detail; no splitter handle) ---
        hub_container = QWidget()
        hub_h = QHBoxLayout(hub_container)
        hub_h.setContentsMargins(0, 0, 0, 0)
        hub_h.setSpacing(0)

        # Left: same sidebar shell as Conversations / Library (QSS + HistoryRowWidget pattern)
        left = QFrame()
        left.setFixedWidth(LEFT_NAV_LIST_SIDEBAR_WIDTH)
        left.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        left.setObjectName("ModelManagerSidebar")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(15, 20, 15, 20)
        left_l.setSpacing(15)
        self.hub_sidebar = left

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        title = QLabel("Model Manager")
        title.setObjectName("ViewTitle")
        title.setProperty("class", "PageTitle")
        title_row.addWidget(title)
        self.page_tour_help_btn = PageTourHelpButton(
            "model_manager",
            area_display_name="Model Manager",
            parent=left,
        )
        title_row.addWidget(self.page_tour_help_btn)
        title_row.addStretch(1)
        left_l.addLayout(title_row)
        left_l.addWidget(self._section_header("fa5s.th-large", "HUGGING FACE REPOSITORIES"))

        self.hub_search_edit = QLineEdit()
        self.hub_search_edit.setObjectName("HubModelSearchBar")
        self.hub_search_edit.setPlaceholderText("Search GGUF models on the Hub…")
        self.hub_search_edit.setToolTip(
            "Search Hugging Face for GGUF models, or filter the Qube Verified list."
        )
        self.hub_search_edit.textChanged.connect(self._schedule_hub_search)
        left_l.addWidget(self.hub_search_edit)

        self.hub_status_banner = QFrame()
        self.hub_status_banner.setObjectName("ModelManagerHubStatusBanner")
        self.hub_status_banner.setVisible(False)
        banner_l = QHBoxLayout(self.hub_status_banner)
        banner_l.setContentsMargins(10, 8, 10, 8)
        banner_l.setSpacing(8)
        self.hub_status_banner_icon = QLabel()
        self.hub_status_banner_icon.setFixedSize(16, 16)
        self.hub_status_banner_text = QLabel("")
        self.hub_status_banner_text.setWordWrap(True)
        banner_l.addWidget(self.hub_status_banner_icon, alignment=Qt.AlignmentFlag.AlignTop)
        banner_l.addWidget(self.hub_status_banner_text, stretch=1)
        left_l.addWidget(self.hub_status_banner)

        self.hub_list_hint = QLabel("Qube Verified — curated GGUF models")
        self.hub_list_hint.setWordWrap(True)
        self.hub_list_hint.setToolTip(
            "Curated models tested for Qube. Clear the search box to browse this list."
        )
        left_l.addWidget(self.hub_list_hint)

        self.hub_search_retry_btn = QPushButton("Retry search")
        self.hub_search_retry_btn.setObjectName("ModelManagerHubSearchRetry")
        self.hub_search_retry_btn.setVisible(False)
        apply_brand_primary(self.hub_search_retry_btn, icon_name="fa5s.redo")
        self.hub_search_retry_btn.clicked.connect(self._run_hub_search)
        left_l.addWidget(self.hub_search_retry_btn)

        self.hub_model_list = QListWidget()
        self.hub_model_list.setObjectName("ModelHubList")
        # Parent frame matches LEFT_NAV_LIST_SIDEBAR_WIDTH; min width matching that here forced horizontal
        # overflow and list items bleeding under the right panel.
        self.hub_model_list.setMinimumWidth(0)
        self.hub_model_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.hub_model_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.hub_model_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.hub_model_list.setToolTip(
            "Select a model repository to view details and download options."
        )
        self.hub_model_list.currentItemChanged.connect(self._on_hub_selection_changed)
        self.hub_model_list.itemSelectionChanged.connect(self._update_hub_row_colors)
        left_l.addWidget(self.hub_model_list, stretch=1)
        self.hub_model_list.viewport().installEventFilter(self)
        self.hub_load_more_btn = QPushButton("Load More")
        apply_brand_primary(self.hub_load_more_btn)
        self.hub_load_more_btn.setToolTip("Load more models from Hugging Face")
        self._set_hub_load_more_visible(False)
        self.hub_load_more_btn.clicked.connect(self._load_more_hub_search_results)
        left_l.addWidget(self.hub_load_more_btn)

        # Right: detail
        right = QWidget()
        right.setMaximumWidth(900)
        right.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        right_l = QVBoxLayout(right)
        # Match transcript-like vertical start while keeping bottom breathing room.
        right_l.setContentsMargins(8, 75, 0, 40)
        right_l.setSpacing(10)

        detail_header_row = QWidget(parent=right)
        detail_header_l = QHBoxLayout(detail_header_row)
        detail_header_l.setContentsMargins(0, 0, 0, 0)
        detail_header_l.setSpacing(8)
        title_info_row = QWidget(parent=detail_header_row)
        title_info_row.setObjectName("ModelManagerDetailTitleRow")
        title_info_l = QHBoxLayout(title_info_row)
        title_info_l.setContentsMargins(0, 0, 0, 0)
        title_info_l.setSpacing(4)
        self.detail_title = QLabel("Select a model", parent=title_info_row)
        self.detail_title.setWordWrap(True)
        f = self.detail_title.font()
        f.setBold(True)
        f.setPointSize(18)
        self.detail_title.setFont(f)
        title_info_l.addWidget(self.detail_title, stretch=0)
        self.detail_info_btn = QPushButton(parent=title_info_row)
        self.detail_info_btn.setObjectName("ModelManagerDetailInfoButton")
        self.detail_info_btn.setProperty("class", "IconButton")
        self.detail_info_btn.setFixedSize(24, 24)
        _boot_theme = theme_for(is_dark=True)
        self.detail_info_btn.setIcon(
            themed_fa_icon("fa5s.info-circle", _boot_theme.color(MUTED_STATUS), 16)
        )
        self.detail_info_btn.setIconSize(QSize(16, 16))
        self.detail_info_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.detail_info_btn.setVisible(False)
        self.detail_info_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        title_info_l.addWidget(
            self.detail_info_btn,
            stretch=0,
            alignment=Qt.AlignmentFlag.AlignTop,
        )
        detail_header_l.addWidget(
            title_info_row,
            stretch=0,
            alignment=Qt.AlignmentFlag.AlignTop,
        )
        detail_header_l.addStretch(1)
        self.detail_source_btn = QPushButton(parent=detail_header_row)
        self.detail_source_btn.setObjectName("ModelManagerSourceButton")
        self.detail_source_btn.setProperty("class", "IconButton")
        self.detail_source_btn.setToolTip("Open source repository on Hugging Face")
        self.detail_source_btn.setIcon(
            qta.icon("fa5s.external-link-alt", color=_boot_theme.color(ACCENT_ICON))
        )
        self.detail_source_btn.setIconSize(QSize(14, 14))
        self.detail_source_btn.setVisible(False)
        self.detail_source_btn.clicked.connect(self._open_current_repo_source)
        detail_header_l.addWidget(self.detail_source_btn, stretch=0, alignment=Qt.AlignmentFlag.AlignTop)

        self.detail_branding_row = QWidget(parent=right)
        detail_branding_outer = QVBoxLayout(self.detail_branding_row)
        detail_branding_outer.setContentsMargins(0, 0, 0, 0)
        detail_branding_outer.setSpacing(4)

        self.detail_official_row = QWidget(parent=self.detail_branding_row)
        detail_official_l = QHBoxLayout(self.detail_official_row)
        detail_official_l.setContentsMargins(0, 0, 0, 0)
        detail_official_l.setSpacing(6)
        self.detail_brand_logo = QLabel(parent=self.detail_official_row)
        self.detail_brand_logo.setObjectName("ModelManagerDetailBrandLogo")
        self.detail_brand_logo.setFixedSize(16, 16)
        self.detail_brand_text = QLabel("", parent=self.detail_official_row)
        self.detail_brand_text.setObjectName("ModelManagerDetailBrandText")
        detail_official_l.addWidget(self.detail_brand_logo, stretch=0)
        detail_official_l.addWidget(self.detail_brand_text, stretch=0)
        detail_official_l.addStretch(1)

        self.detail_variant_row = QWidget(parent=self.detail_branding_row)
        detail_variant_l = QHBoxLayout(self.detail_variant_row)
        detail_variant_l.setContentsMargins(0, 0, 0, 0)
        detail_variant_l.setSpacing(6)
        self.detail_variant_logo = QLabel(parent=self.detail_variant_row)
        self.detail_variant_logo.setObjectName("ModelManagerDetailVariantLogo")
        self.detail_variant_logo.setFixedSize(16, 16)
        self.detail_variant_text = QLabel("", parent=self.detail_variant_row)
        self.detail_variant_text.setObjectName("ModelManagerDetailVariantText")
        detail_variant_l.addWidget(self.detail_variant_logo, stretch=0)
        detail_variant_l.addWidget(self.detail_variant_text, stretch=0)
        detail_variant_l.addStretch(1)
        self.detail_variant_row.hide()

        detail_branding_outer.addWidget(self.detail_official_row)
        detail_branding_outer.addWidget(self.detail_variant_row)
        self.detail_official_row.hide()
        self.detail_branding_row.hide()

        # All detail cards (meta, download, readme) share one outer scroll — no inner card scroll.
        self.detail_scroll = QScrollArea(parent=right)
        self.detail_scroll.setObjectName("ModelManagerDetailScrollArea")
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _detail_scroll_pol = QSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.detail_scroll.setSizePolicy(_detail_scroll_pol)
        self.detail_cards_content = QWidget()
        self.detail_cards_content.setObjectName("ModelManagerDetailScrollContent")
        detail_cards_l = QVBoxLayout(self.detail_cards_content)
        detail_cards_l.setContentsMargins(0, 0, 0, 0)
        detail_cards_l.setSpacing(10)
        detail_cards_l.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        _detail_content_pol = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        self.detail_cards_content.setSizePolicy(_detail_content_pol)
        self.detail_scroll.setWidget(self.detail_cards_content)

        # Metadata panel (chip-heavy layout; actual colors come from QSS semantic classes).
        self.meta_panel = QFrame(parent=self.detail_cards_content)
        self.meta_panel.setProperty("class", "MetaPanelCard")
        self.meta_panel.setObjectName("ModelManagerMetaCard")
        meta_panel_l = QVBoxLayout(self.meta_panel)
        meta_panel_l.setContentsMargins(12, 12, 12, 12)
        meta_panel_l.setSpacing(8)
        meta_panel_l.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        _meta_pol = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.meta_panel.setSizePolicy(_meta_pol)

        self.meta_row_1 = QWidget(parent=self.meta_panel)
        meta_row_1_l = FlowLayout(self.meta_row_1, spacing=8)
        meta_row_1_l.setContentsMargins(0, 0, 0, 0)

        self.meta_params_title_lbl = QLabel("Params:")
        self.meta_params_title_lbl.setProperty("class", "MetaLabel")
        self.meta_params_chip = QLabel("--")
        self.meta_params_chip.setProperty("class", "Chip primary")
        self.meta_arch_title_lbl = QLabel("Arch:")
        self.meta_arch_title_lbl.setProperty("class", "MetaLabel")
        self.meta_arch_chip = QLabel("--")
        self.meta_arch_chip.setProperty("class", "Chip primary")
        self.meta_domain_title_lbl = QLabel("Domain:")
        self.meta_domain_title_lbl.setProperty("class", "MetaLabel")
        self.meta_domain_chip = QLabel("--")
        self.meta_domain_chip.setProperty("class", "Chip primary")
        self.meta_format_title_lbl = QLabel("Format:")
        self.meta_format_title_lbl.setProperty("class", "MetaLabel")
        self.meta_format_chip = QLabel("--")
        self.meta_format_chip.setProperty("class", "Chip accent")

        for w in (
            self.meta_params_title_lbl,
            self.meta_params_chip,
            self.meta_arch_title_lbl,
            self.meta_arch_chip,
            self.meta_domain_title_lbl,
            self.meta_domain_chip,
            self.meta_format_title_lbl,
            self.meta_format_chip,
        ):
            w.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            meta_row_1_l.addWidget(w)

        self.meta_row_2 = QWidget(parent=self.meta_panel)
        meta_row_2_l = QHBoxLayout(self.meta_row_2)
        meta_row_2_l.setContentsMargins(0, 0, 0, 0)
        meta_row_2_l.setSpacing(8)
        self.meta_caps_title_lbl = QLabel("Capabilities:")
        self.meta_caps_title_lbl.setProperty("class", "MetaLabel")
        self.meta_caps_wrap = QWidget(parent=self.meta_row_2)
        self.meta_caps_wrap_l = FlowLayout(self.meta_caps_wrap, spacing=6)
        self.meta_caps_wrap_l.setContentsMargins(0, 0, 0, 0)
        meta_row_2_l.addWidget(self.meta_caps_title_lbl)
        meta_row_2_l.addWidget(self.meta_caps_wrap, stretch=1)

        self.meta_rows_divider = QFrame(parent=self.meta_panel)
        self.meta_rows_divider.setFrameShape(QFrame.Shape.HLine)
        self.meta_rows_divider.setFrameShadow(QFrame.Shadow.Plain)
        self.meta_rows_divider.setStyleSheet(_boot_theme.style(DIVIDER_ACCENT))
        divider_host = QWidget(parent=self.meta_panel)
        divider_l = QHBoxLayout(divider_host)
        divider_l.setContentsMargins(8, 2, 8, 2)
        divider_l.setSpacing(0)
        divider_l.addWidget(self.meta_rows_divider)

        self.meta_hint_lbl = QLabel("", parent=self.meta_panel)
        self.meta_hint_lbl.setWordWrap(True)
        self.meta_hint_lbl.hide()

        meta_panel_l.addWidget(self.meta_row_1)
        meta_panel_l.addWidget(divider_host)
        meta_panel_l.addWidget(self.meta_row_2)
        meta_panel_l.addWidget(self.meta_hint_lbl)

        # Download options card
        self.download_options_card = QFrame(parent=self.detail_cards_content)
        self.download_options_card.setProperty("class", "DownloadOptionsCard")
        self.download_options_card.setObjectName("ModelManagerDownloadCard")
        dl_card_l = QVBoxLayout(self.download_options_card)
        dl_card_l.setContentsMargins(12, 12, 12, 12)
        dl_card_l.setSpacing(8)
        dl_card_l.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        _dl_card_pol = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.download_options_card.setSizePolicy(_dl_card_pol)

        dl_header_row = QWidget(parent=self.download_options_card)
        dl_header_l = QHBoxLayout(dl_header_row)
        dl_header_l.setContentsMargins(0, 0, 0, 0)
        dl_header_l.setSpacing(8)
        dl_header_icon = QLabel()
        self._download_options_icon_label = dl_header_icon
        dl_header_icon.setProperty("icon_name", "fa5s.box-open")
        dl_header_title = QLabel("Download Options")
        dl_header_title.setProperty("class", "SectionHeaderLabel")
        dl_header_l.addWidget(dl_header_icon)
        dl_header_l.addWidget(dl_header_title)
        dl_header_l.addStretch(1)

        q_lab = QLabel("Loading available files…")
        self.hub_quant_hint_lbl = q_lab
        q_lab.setProperty("class", "ToolsPaneControl")
        self._configure_wrapping_label(self.hub_quant_hint_lbl)

        files_row_host = QWidget(parent=self.download_options_card)
        files_row = QHBoxLayout(files_row_host)
        files_row.setContentsMargins(0, 0, 0, 0)
        files_row.setSpacing(8)
        self.hf_file_combo = HubFileComboBox(self, parent=files_row_host)
        self.hf_file_combo.setObjectName("HubFileComboBox")
        self.hf_file_combo.setToolTip(
            "Choose a GGUF quantization variant to download or load."
        )
        self.hf_file_combo.setMinimumWidth(self._scaled_content_width(160))
        self.hf_file_combo.setMinimumHeight(34)
        _combo_pol = self.hf_file_combo.sizePolicy()
        _combo_pol.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        self.hf_file_combo.setSizePolicy(_combo_pol)
        self.hf_file_combo.currentIndexChanged.connect(self._on_hf_file_combo_changed)
        # Popup QListView is reparented to a separate window; style it by objectName + palette
        # (same pattern as MainWindow._apply_menu_theme / PrestigeMenuList).
        _hub_combo_view = self.hf_file_combo.view()
        _hub_combo_view.setObjectName("HubFileComboDropdownView")
        self._hub_file_combo_delegate = HubFileComboDelegate(_hub_combo_view)
        _hub_combo_view.setItemDelegate(self._hub_file_combo_delegate)
        _hub_combo_view.setMinimumWidth(self._scaled_content_width(280))
        _hub_combo_view.setAutoFillBackground(True)
        _vp = _hub_combo_view.viewport()
        if _vp is not None:
            _vp.setAutoFillBackground(True)
        files_row.addWidget(self.hf_file_combo, stretch=1)

        self.download_btn = QPushButton("Download")
        apply_brand_primary(self.download_btn, icon_name="fa5s.download")
        self.download_btn.setToolTip("Download the selected model file")
        self.download_btn.clicked.connect(self._start_download)
        self.download_btn.setMinimumWidth(self._scaled_content_width(108))
        self.download_btn.setMinimumHeight(34)
        _dl_btn_pol = self.download_btn.sizePolicy()
        _dl_btn_pol.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
        _dl_btn_pol.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        self.download_btn.setSizePolicy(_dl_btn_pol)

        self.download_status = QLabel("")
        self._configure_wrapping_label(self.download_status)
        self.download_status.setVisible(False)

        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setTextVisible(True)
        _dp_pol = self.download_progress.sizePolicy()
        _dp_pol.setRetainSizeWhenHidden(True)
        self.download_progress.setSizePolicy(_dp_pol)
        self.download_progress.hide()

        system_row_host = QWidget(parent=self.download_options_card)
        system_row = QVBoxLayout(system_row_host)
        system_row.setContentsMargins(0, 2, 0, 0)
        system_row.setSpacing(0)
        self.system_chip_lbl = QLabel("System: --", parent=system_row_host)
        self.system_chip_lbl.setProperty("class", "Chip outlined")
        self.system_chip_lbl.setToolTip(
            "Whether this model variant fits your GPU memory and CPU configuration."
        )
        self._configure_wrapping_label(self.system_chip_lbl)
        self._set_system_match_style("unknown")
        system_row.addWidget(self.system_chip_lbl)

        download_row_host = QWidget(parent=self.download_options_card)
        download_row = QHBoxLayout(download_row_host)
        download_row.setContentsMargins(0, 4, 0, 0)
        download_row.setSpacing(8)
        download_row.addStretch(1)
        download_row.addWidget(self.download_btn, stretch=0, alignment=Qt.AlignmentFlag.AlignRight)

        self.hub_quant_rationale_row = QWidget(parent=self.download_options_card)
        rationale_outer = QVBoxLayout(self.hub_quant_rationale_row)
        rationale_outer.setContentsMargins(0, 4, 0, 0)
        rationale_outer.setSpacing(6)
        rationale_badge_row = QHBoxLayout()
        rationale_badge_row.setContentsMargins(0, 0, 0, 0)
        rationale_badge_row.setSpacing(0)
        self.hub_quant_rationale_badge = QLabel("", parent=self.hub_quant_rationale_row)
        self.hub_quant_rationale_badge.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        _badge_pol = self.hub_quant_rationale_badge.sizePolicy()
        _badge_pol.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
        _badge_pol.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        self.hub_quant_rationale_badge.setSizePolicy(_badge_pol)
        rationale_badge_row.addWidget(
            self.hub_quant_rationale_badge,
            stretch=0,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        rationale_badge_row.addStretch(1)
        rationale_outer.addLayout(rationale_badge_row)
        self.hub_quant_rationale_lbl = QLabel("", parent=self.hub_quant_rationale_row)
        self.hub_quant_rationale_lbl.setProperty("class", "ToolsPaneControl")
        self._configure_wrapping_label(self.hub_quant_rationale_lbl)
        rationale_outer.addWidget(self.hub_quant_rationale_lbl)
        _rationale_pol = self.hub_quant_rationale_row.sizePolicy()
        _rationale_pol.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        self.hub_quant_rationale_row.setSizePolicy(_rationale_pol)
        self.hub_quant_rationale_row.hide()

        dl_card_l.addWidget(dl_header_row)
        dl_card_l.addWidget(q_lab)
        dl_card_l.addWidget(files_row_host)
        dl_card_l.addWidget(system_row_host)
        dl_card_l.addWidget(self.hub_quant_rationale_row)
        dl_card_l.addWidget(download_row_host)
        dl_card_l.addWidget(self.download_status)
        dl_card_l.addWidget(self.download_progress)

        self.readme_browser = QTextBrowser(parent=self.detail_cards_content)
        self.readme_browser.setObjectName("ModelManagerReadmeCard")
        self.readme_browser.setMinimumHeight(180)
        self.readme_browser.setOpenExternalLinks(True)
        self.readme_browser.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)
        self.readme_browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.readme_browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _readme_pol = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.readme_browser.setSizePolicy(_readme_pol)

        detail_cards_l.addWidget(self.meta_panel)
        detail_cards_l.addWidget(self.download_options_card)
        detail_cards_l.addWidget(self.readme_browser)

        right_l.addWidget(detail_header_row)
        right_l.addWidget(self.detail_branding_row)
        right_l.addWidget(self.detail_scroll, stretch=1)

        right_host = QWidget()
        right_host_l = QHBoxLayout(right_host)
        # Keep a fixed gap from the sidebar while preserving left pinning behavior on resize.
        right_host_l.setContentsMargins(10, 0, 0, 0)
        right_host_l.setSpacing(0)
        right_host_l.addWidget(right, 1)

        hub_h.addWidget(left)
        hub_h.addWidget(right_host, stretch=1)

        main_layout.addWidget(hub_container, stretch=1)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._run_hub_search)

        from ui.components.type_to_search import install_type_to_search

        install_type_to_search(self, self.hub_search_edit)

    def _set_hub_load_more_visible(self, visible: bool) -> None:
        if getattr(self, "_tour_load_more_preview_active", False):
            self.hub_load_more_btn.setVisible(True)
            return
        self.hub_load_more_btn.setVisible(bool(visible))

    def begin_load_more_tutorial_preview(self) -> None:
        """Show Load More during the Model Manager guided tour."""
        self._tour_load_more_preview_active = True
        self._set_hub_load_more_visible(True)

    def end_load_more_tutorial_preview(self) -> None:
        """Hide tour-only Load More preview and restore normal visibility."""
        if not getattr(self, "_tour_load_more_preview_active", False):
            return
        self._tour_load_more_preview_active = False
        self._set_hub_load_more_visible(False)

    def _meta_style(self, label: QLabel, *, is_dark: bool, strong: bool = False) -> None:
        theme = self._theme(is_dark)
        label.setStyleSheet(
            theme.style(META_LABEL, strong=strong)
            + " font-size: 12px;"
            + (" font-weight: 600;" if strong else " font-weight: 500;")
        )

    def _reset_hub_metadata_labels(self) -> None:
        if hasattr(self, "meta_params_chip"):
            self.meta_params_chip.setText("--")
            self.meta_arch_chip.setText("--")
            self.meta_domain_chip.setText("--")
            self.meta_format_chip.setText("--")
        self._render_capability_chips([])
        if hasattr(self, "system_chip_lbl"):
            self.system_chip_lbl.setText("System: --")
            self._set_system_match_style("unknown")
        if hasattr(self, "download_btn"):
            self.download_btn.setText("Download")
        if hasattr(self, "meta_hint_lbl"):
            self.meta_hint_lbl.hide()
        self._set_download_status_text("")

    def _set_meta_hint(self, text: str | None) -> None:
        if not hasattr(self, "meta_hint_lbl"):
            return
        msg = str(text or "").strip()
        self.meta_hint_lbl.setText(msg)
        self.meta_hint_lbl.setVisible(bool(msg))

    def _apply_hub_metadata(self, meta: dict | None) -> None:
        m = meta or {}
        self.meta_params_chip.setText(str(m.get("params", "Unknown")))
        self.meta_arch_chip.setText(str(m.get("arch", "Unknown")))
        self.meta_domain_chip.setText(str(m.get("domain", "Unknown")))
        self.meta_format_chip.setText(str(m.get("format", "Unknown")))
        caps = m.get("capabilities") or []
        if isinstance(caps, list):
            clean_caps = [str(c).strip() for c in caps if str(c).strip()]
        else:
            clean_caps = []
        self._render_capability_chips(clean_caps)
        self._set_meta_hint(None)

    def _apply_hub_metadata_styles(self, is_dark: bool) -> None:
        if not hasattr(self, "meta_params_chip"):
            return
        for lbl in (
            self.meta_params_title_lbl,
            self.meta_arch_title_lbl,
            self.meta_domain_title_lbl,
            self.meta_format_title_lbl,
            self.meta_caps_title_lbl,
        ):
            self._meta_style(lbl, is_dark=is_dark, strong=True)
        for chip in (
            self.meta_params_chip,
            self.meta_arch_chip,
            self.meta_domain_chip,
            self.meta_format_chip,
        ):
            chip.style().unpolish(chip)
            chip.style().polish(chip)
            chip.update()
        # Capability chips are built dynamically with widget-local styles, so rebuild
        # them on theme refresh instead of relying on app-level QSS to override them.
        self._render_capability_chips(list(getattr(self, "_current_meta_capabilities", [])))
        theme = self._theme(is_dark)
        self.meta_hint_lbl.setStyleSheet(
            theme.style(HUB_MUTED_HINT) + " font-weight: 500; background: transparent;"
        )
        self._set_system_match_style("unknown")
        self._refresh_download_options_header_icon()

    def _refresh_download_options_header_icon(self) -> None:
        if not hasattr(self, "_download_options_icon_label"):
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        self._download_options_icon_label.setPixmap(
            qta.icon("fa5s.box-open", color=theme.color(ACCENT_ICON)).pixmap(QSize(14, 14))
        )

    def _capability_icon_name(self, cap: str) -> str:
        c = cap.lower()
        if "audio" in c:
            return "fa5s.volume-up"
        if "tts" in c:
            return "fa5s.volume-up"
        if "stt" in c or "asr" in c:
            return "fa5s.microphone"
        if "vision" in c:
            return "fa5s.eye"
        if "tool" in c:
            return "fa5s.wrench"
        if "reason" in c:
            return "fa5s.brain"
        if "code" in c or "coder" in c:
            return "fa5s.code"
        if "multi" in c:
            return "fa5s.globe"
        return "fa5s.star"

    def _render_capability_chips(self, caps: list[str]) -> None:
        if not hasattr(self, "meta_caps_wrap_l"):
            return
        self._current_meta_capabilities = [str(cap).strip() for cap in (caps or []) if str(cap).strip()]
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        chip_style = theme.style(CAPABILITY_CHIP)
        icon_color = theme.text_primary
        while self.meta_caps_wrap_l.count():
            it = self.meta_caps_wrap_l.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        if not self._current_meta_capabilities:
            empty_chip = QLabel("Unknown")
            empty_chip.setProperty("class", "Chip muted")
            self.meta_caps_wrap_l.addWidget(empty_chip)
            return
        for cap in self._current_meta_capabilities:
            chip = QFrame()
            chip.setProperty("class", "Chip capability")
            chip.setStyleSheet(chip_style)
            chip_l = QHBoxLayout(chip)
            chip_l.setContentsMargins(8, 3, 8, 3)
            chip_l.setSpacing(6)
            icon_lbl = QLabel()
            icon_lbl.setProperty("class", "ChipIcon")
            icon_lbl.setPixmap(qta.icon(self._capability_icon_name(cap), color=icon_color).pixmap(QSize(11, 11)))
            txt_lbl = QLabel(cap)
            txt_lbl.setProperty("class", "ChipLabel")
            chip_l.addWidget(icon_lbl)
            chip_l.addWidget(txt_lbl)
            self.meta_caps_wrap_l.addWidget(chip)

    @staticmethod
    def _fmt_gib(n: int) -> str:
        return f"{(max(0, int(n)) / (1024**3)):.2f} GB"

    def _update_gpu_fit_status(self, _index: int | None = None) -> None:
        if not hasattr(self, "hf_file_combo"):
            return
        idx = self.hf_file_combo.currentIndex()
        if idx < 0:
            if hasattr(self, "system_chip_lbl"):
                self.system_chip_lbl.setText("System: --")
                self._set_system_match_style("unknown")
            return
        size_b = self.hf_file_combo.itemData(idx, int(HUB_FILE_COMBO_BYTES_ROLE))
        try:
            q_bytes = int(size_b) if size_b is not None else 0
        except (TypeError, ValueError):
            q_bytes = 0
        profile = detect_memory_budget_profile()
        exp = experience_for_download(q_bytes, profile)
        if hasattr(self, "system_chip_lbl"):
            self.system_chip_lbl.setText(exp.short_label)
            style_map = {
                "best": "fit",
                "caution": "caution",
                "neutral": "unknown",
                "unknown": "unknown",
            }
            self._set_system_match_style(style_map.get(exp.style, "unknown"))
        if hasattr(self, "hf_file_combo") and exp.detail:
            tip = exp.detail
            rec_raw = self.hf_file_combo.itemData(idx, int(HUB_FILE_COMBO_REC_ROLE))
            if isinstance(rec_raw, QuantFileRecommendation) and rec_raw.rationale:
                tip = f"{rec_raw.rationale}\n\n{exp.detail}"
            self.hf_file_combo.setToolTip(tip)
        self._refresh_download_options_card_geometry()

    def _on_hf_file_combo_changed(self, _index: int) -> None:
        self._update_download_button_label()
        self._update_download_selection_hint()
        self._update_quant_rationale_label()
        self._update_gpu_fit_status(_index)
        self._sync_download_action_state()

    def _download_queue_active(self) -> bool:
        return len(self._download_queue_paths) > 0

    def _reset_download_queue_state(self) -> None:
        self._download_queue_paths = []
        self._download_queue_index = 0
        self._download_completed_paths = []
        self._download_failed_path = None
        self._download_total_bytes = 0
        self._download_completed_bytes = 0
        self._download_current_bytes_total = 0
        self._download_current_path = ""

    def _update_download_button_label(self) -> None:
        if not hasattr(self, "download_btn") or not hasattr(self, "hf_file_combo"):
            return
        if getattr(self, "_download_ui_load_mode", False):
            return
        idx = self.hf_file_combo.currentIndex()
        if idx < 0:
            self.download_btn.setText("Download")
            return
        entries = self._selected_hf_repo_files_for_download()
        if len(entries) > 1:
            raw = self.hf_file_combo.itemData(idx, int(HUB_FILE_COMBO_BYTES_ROLE))
            try:
                sz = int(raw) if raw is not None else 0
            except (TypeError, ValueError):
                sz = 0
            if sz > 0:
                self.download_btn.setText(f"Download {len(entries)} files ({self._fmt_bytes(sz)})")
            else:
                self.download_btn.setText(f"Download {len(entries)} files")
            return
        raw = self.hf_file_combo.itemData(idx, int(HUB_FILE_COMBO_BYTES_ROLE))
        try:
            sz = int(raw) if raw is not None else 0
        except (TypeError, ValueError):
            sz = 0
        if sz > 0:
            self.download_btn.setText(f"Download ({self._fmt_bytes(sz)})")
        else:
            self.download_btn.setText("Download")

    def _build_quant_recommendation_context(self) -> QuantRecommendationContext:
        repo = str(getattr(self, "_current_repo_id", "") or "").strip()
        current = self.hub_model_list.currentItem() if hasattr(self, "hub_model_list") else None
        title = str(current.data(HUB_ROW_TITLE_ROLE) or self.detail_title.text() or repo) if current else str(self.detail_title.text() or repo)
        desc = str(current.data(HUB_ROW_DESC_ROLE) or "") if current else ""
        meta = dict(self._hub_meta_snapshot or {})
        cap_summary = None
        if repo:
            try:
                caps = self._capability_service.get_capabilities(repo)
                if caps is not None:
                    cap_summary = self._capability_service.summarize_for_ui(caps)
            except Exception:
                cap_summary = None
        return build_context_from_hub_meta(
            repo_id=repo,
            title=title,
            description=desc,
            meta=meta,
            capability_summary=cap_summary,
        )

    def _apply_quant_recommendations_to_combo(self, normalized: list[tuple[str, int | None]]) -> None:
        if not normalized:
            self._quant_rec_plan = None
            return
        ctx = self._quant_rec_context
        if ctx is None:
            ctx = self._build_quant_recommendation_context()
            self._quant_rec_context = ctx
        plan = recommend_quants(ctx, normalized)
        self._quant_rec_plan = plan
        rec_by_path = {r.path: r for r in plan.files}
        for i in range(1, self.hf_file_combo.count()):
            path = self.hf_file_combo.itemData(i)
            if not isinstance(path, str):
                continue
            rec = rec_by_path.get(path)
            if rec is not None:
                self.hf_file_combo.setItemData(i, rec, int(HUB_FILE_COMBO_REC_ROLE))
        if hasattr(self, "hub_quant_hint_lbl") and plan.summary_hint:
            self.hub_quant_hint_lbl.setText(plan.summary_hint)
        if plan.default_index is not None and 0 <= plan.default_index < self.hf_file_combo.count():
            self.hf_file_combo.setCurrentIndex(plan.default_index)
        tip_parts = [
            f"Suggested: {plan.primary_quant}",
            f"Alternative: {plan.secondary_quant}",
        ]
        if plan.plan_confidence.value != "high":
            tip_parts.append(f"Confidence: {plan.plan_confidence.value}")
        self.hf_file_combo.setToolTip(" · ".join(tip_parts))
        self._refresh_download_options_card_geometry()

    def _refresh_quant_recommendations(self) -> None:
        normalized: list[tuple[str, int | None]] = []
        for i in range(1, self.hf_file_combo.count()):
            path = self.hf_file_combo.itemData(i)
            if not isinstance(path, str):
                continue
            raw_b = self.hf_file_combo.itemData(i, int(HUB_FILE_COMBO_BYTES_ROLE))
            try:
                sz = int(raw_b) if raw_b is not None else None
            except (TypeError, ValueError):
                sz = None
            normalized.append((path, sz))
        if not normalized:
            return
        self._quant_rec_context = self._build_quant_recommendation_context()
        self.hf_file_combo.blockSignals(True)
        cur = self.hf_file_combo.currentIndex()
        self._apply_quant_recommendations_to_combo(normalized)
        if cur >= 0:
            self.hf_file_combo.setCurrentIndex(cur)
        self.hf_file_combo.blockSignals(False)
        self._update_quant_rationale_label()
        self._update_gpu_fit_status()

    def _style_quant_rationale_badge(self, badge_lbl: QLabel, *, is_primary: bool, is_dark: bool) -> None:
        theme = self._theme(is_dark)
        role = QUANT_BADGE_PRIMARY if is_primary else QUANT_BADGE_SECONDARY
        badge_lbl.setStyleSheet(theme.style(role))

    def _update_quant_rationale_label(self) -> None:
        if not hasattr(self, "hub_quant_rationale_lbl") or not hasattr(self, "hf_file_combo"):
            return
        row = getattr(self, "hub_quant_rationale_row", None)
        badge_lbl = getattr(self, "hub_quant_rationale_badge", None)
        text_lbl = self.hub_quant_rationale_lbl
        idx = self.hf_file_combo.currentIndex()

        def _hide_rationale() -> None:
            if row is not None:
                row.hide()
            text_lbl.setText("")
            if badge_lbl is not None:
                badge_lbl.setText("")

        if idx <= 0:
            _hide_rationale()
            self._refresh_download_options_card_geometry()
            return
        rec_raw = self.hf_file_combo.itemData(idx, int(HUB_FILE_COMBO_REC_ROLE))
        if not isinstance(rec_raw, QuantFileRecommendation):
            _hide_rationale()
            self._refresh_download_options_card_geometry()
            return
        text = str(rec_raw.rationale or "").strip()
        if not text:
            _hide_rationale()
            self._refresh_download_options_card_geometry()
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        text_lbl.setStyleSheet(theme.style(META_LABEL) + " font-size: 12px;")
        text_lbl.setText(text)
        if badge_lbl is not None:
            if rec_raw.badge != QuantBadgeKind.NONE and rec_raw.badge_text:
                badge_lbl.setText(rec_raw.badge_text)
                self._style_quant_rationale_badge(
                    badge_lbl,
                    is_primary=rec_raw.badge == QuantBadgeKind.RECOMMENDED,
                    is_dark=is_dark,
                )
                badge_lbl.show()
            else:
                badge_lbl.setText("Alternative")
                self._style_quant_rationale_badge(badge_lbl, is_primary=False, is_dark=is_dark)
                badge_lbl.show()
        if row is not None:
            row.show()
        self._refresh_download_options_card_geometry()

    def _update_download_selection_hint(self) -> None:
        if not hasattr(self, "hub_quant_hint_lbl") or not hasattr(self, "hf_file_combo"):
            return
        idx = self.hf_file_combo.currentIndex()
        if idx <= 0:
            return
        entries = self._selected_hf_repo_files_for_download()
        if len(entries) <= 1:
            return
        names = [Path(p).name for p, _ in entries]
        if len(names) <= 3:
            suffix = ", ".join(names)
        else:
            suffix = ", ".join(names[:3]) + f", +{len(names) - 3} more"
        self.hub_quant_hint_lbl.setText(
            f"Bundle selected: downloading all {len(entries)} shard files automatically ({suffix})."
        )

    def _set_download_progress_text(self, prefix: str = "Downloading model") -> None:
        if not hasattr(self, "download_progress"):
            return
        self.download_progress.setFormat(f"{prefix} (%p%)")

    def _on_download_progress_pct(self, pct: int) -> None:
        pct_i = max(0, min(100, int(pct)))
        if self._download_queue_active():
            n = len(self._download_queue_paths)
            current_idx = max(0, min(self._download_queue_index + 1, n))
            agg_pct = pct_i
            if self._download_total_bytes > 0 and self._download_current_bytes_total > 0:
                done = self._download_completed_bytes + (
                    self._download_current_bytes_total * (pct_i / 100.0)
                )
                agg_pct = int(min(100, max(0, (done * 100.0) / self._download_total_bytes)))
            elif n > 0:
                agg_pct = int(min(100, ((self._download_queue_index + (pct_i / 100.0)) * 100.0) / n))
            self.download_progress.setValue(agg_pct)
            self._set_download_progress_text(f"Downloading shard {current_idx}/{n}")
            return
        self.download_progress.setValue(pct_i)
        self._set_download_progress_text()

    def _on_download_status_message(self, msg: str) -> None:
        # Keep status label quiet during active download; progress text carries the state.
        # Preserve "Downloading model" text per UX request.
        if self._download_worker and self._download_worker.isRunning():
            if self._download_queue_active():
                n = len(self._download_queue_paths)
                idx = max(1, min(self._download_queue_index + 1, n))
                self._set_download_progress_text(f"Downloading shard {idx}/{n}")
            else:
                self._set_download_progress_text()

    def _set_download_status_text(self, text: str) -> None:
        if not hasattr(self, "download_status"):
            return
        msg = str(text or "").strip()
        self.download_status.setText(msg)
        self.download_status.setVisible(bool(msg))

    def _selected_local_model_path(self) -> Path | None:
        sel = self._selected_hf_repo_file()
        if not sel:
            return None
        raw = Path(get_llm_models_dir()) / Path(sel).name
        return Path(resolve_internal_model_path(str(raw)))

    def _is_selected_model_downloaded(self) -> bool:
        p = self._selected_local_model_path()
        return bool(p is not None and p.is_file())

    def _is_selected_model_loaded(self) -> bool:
        p = self._selected_local_model_path()
        if p is None:
            return False
        eng = self.workers.get("native_engine") if self.workers else None
        snap = eng.get_model_reasoning_telemetry() if eng else None
        if not snap or not snap.get("loaded"):
            return False
        return str(snap.get("model_basename") or "").strip() == p.name

    def _sync_download_button_tooltip(self) -> None:
        if getattr(self, "_download_ui_cancel_mode", False):
            tip = "Stop the current download"
        elif getattr(self, "_download_ui_load_mode", False):
            tip = "Load the selected model into the native engine"
        else:
            tip = "Download the selected model file from Hugging Face"
        self.download_btn.setToolTip(tip)

    def _set_download_button_download_mode(self) -> None:
        try:
            self.download_btn.clicked.disconnect()
        except TypeError:
            pass
        self._download_ui_cancel_mode = False
        self._download_ui_load_mode = False
        self.download_btn.setEnabled(True)
        self._apply_download_action_button_style(mode="download")
        self.download_btn.clicked.connect(self._start_download)
        self._update_download_button_label()
        self._sync_download_button_tooltip()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self.refresh_button_themes(is_dark)

    def _set_download_button_load_mode(self, enabled: bool) -> None:
        try:
            self.download_btn.clicked.disconnect()
        except TypeError:
            pass
        self._download_ui_cancel_mode = False
        self._download_ui_load_mode = True
        self.download_btn.setText("Load Model")
        self._apply_download_action_button_style(mode="load")
        self.download_btn.setEnabled(bool(enabled))
        self.download_btn.clicked.connect(self._load_selected_model)
        self._sync_download_button_tooltip()

    def _apply_download_action_button_style(self, mode: str) -> None:
        """Apply the brand style + matching icon for the current mode.

        Uses the shared `brand_buttons` helper so widget-level QSS drives the
        final render (highest specificity in Qt), avoiding the light-theme
        leak-through where the generic `QPushButton { background-color: #ffffff; }`
        rule beats app-level `QPushButton[class~="..."]` rules on some styles.
        The helper also tints the mode-specific icon to the variant's
        foreground color (`BRAND_FG_COLOR`) so the icon never blends into its
        own background (regression guard: the default `fa5s.download` icon
        used to render in brand-purple on the brand-purple button).
        """
        if mode == "load":
            apply_brand_success(self.download_btn, icon_name="fa5s.play")
        elif mode == "cancel":
            apply_brand_danger(self.download_btn, icon_name="fa5s.times")
        else:
            apply_brand_primary(self.download_btn, icon_name="fa5s.download")

    def _sync_download_action_state(self) -> None:
        if getattr(self, "_download_ui_cancel_mode", False):
            return
        if self._is_selected_model_downloaded():
            loaded = self._is_selected_model_loaded()
            self._set_download_button_load_mode(enabled=not loaded)
            p = self._selected_local_model_path()
            if loaded:
                self._set_download_status_text(f"Loaded: {p.name if p else 'model'}")
            else:
                self._set_download_status_text(f"Saved: {p.name if p else 'model'}")
            self.download_progress.hide()
            self._update_gpu_fit_status()
            return
        self._set_download_button_download_mode()
        if not (self._download_worker and self._download_worker.isRunning()):
            self._set_download_status_text("")
        self._update_gpu_fit_status()

    def _load_selected_model(self) -> None:
        p = self._selected_local_model_path()
        if p is None or not p.is_file():
            self._show_error("Model not found", "Selected file is not available locally.")
            self._sync_download_action_state()
            return
        missing = missing_gguf_shards(str(p))
        if missing:
            preview = "\n".join(f"- {name}" for name in missing[:8])
            extra = ""
            if len(missing) > 8:
                extra = f"\n- ... and {len(missing) - 8} more"
            self._show_error(
                "Missing model shards",
                "This model is split across multiple GGUF shard files, but some parts are missing.\n\n"
                "Download the missing files before loading:\n"
                f"{preview}{extra}",
            )
            self._set_download_status_text("Missing shards - download all parts first.")
            self._sync_download_action_state()
            return
        set_internal_model_path(str(p))
        repo = str(getattr(self, "_current_repo_id", "") or "").strip()
        if repo:
            self._publisher_guidance_service.record_provenance(str(p), repo)
        if self._llm:
            cv = getattr(self.window(), "conversations_view", None)
            if cv is not None and hasattr(cv, "interrupt_active_response"):
                cv.interrupt_active_response()
            self._llm.refresh_native_model_from_settings()
        self.native_library_changed.emit()
        self._set_download_status_text(f"Loaded: {p.name}")
        self._set_download_button_load_mode(enabled=False)

    def _apply_hub_list_surface(self, is_dark: bool) -> None:
        """Match Conversations sidebar/list background palette on both themes."""
        theme = self._theme(is_dark)
        bg_hex = theme.color(LIST_SURFACE)
        bg = QColor(bg_hex)
        stage_bg_hex = theme.color(STAGE_SURFACE)
        stage_bg = QColor(stage_bg_hex)
        border = theme.border_subtle if theme.is_dark else theme.border
        if hasattr(self, "hub_sidebar"):
            p = self.hub_sidebar
            p.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            p.setAutoFillBackground(True)
            pa = p.palette()
            pa.setColor(QPalette.ColorRole.Window, bg)
            p.setPalette(pa)
        if not hasattr(self, "hub_model_list"):
            return
        w = self.hub_model_list
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

        if hasattr(self, "meta_panel"):
            self.meta_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.meta_panel.setStyleSheet(
                f"#ModelManagerMetaCard {{ background-color: {bg_hex}; border: 1px solid {border}; border-radius: 10px; }}"
            )
        if hasattr(self, "download_options_card"):
            self.download_options_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.download_options_card.setStyleSheet(
                f"#ModelManagerDownloadCard {{ background-color: {bg_hex}; border: 1px solid {border}; border-radius: 10px; }}"
            )
        if hasattr(self, "detail_scroll"):
            self.detail_scroll.setStyleSheet(
                "#ModelManagerDetailScrollArea { background: transparent; border: none; }"
                f"#ModelManagerDetailScrollContent {{ background-color: {stage_bg_hex}; }}"
            )
            vp = self.detail_scroll.viewport()
            if vp is not None:
                vp.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
                vp.setAutoFillBackground(True)
                vpal = vp.palette()
                vpal.setColor(QPalette.ColorRole.Window, stage_bg)
                vpal.setColor(QPalette.ColorRole.Base, stage_bg)
                vp.setPalette(vpal)
            if hasattr(self, "detail_cards_content"):
                self.detail_cards_content.setAttribute(
                    Qt.WidgetAttribute.WA_StyledBackground, True
                )
                self.detail_cards_content.setAutoFillBackground(False)
        if hasattr(self, "readme_browser"):
            self.readme_browser.setStyleSheet(
                f"#ModelManagerReadmeCard {{ background-color: {bg_hex}; border: 1px solid {border}; border-radius: 10px; }}"
            )
            self.readme_browser.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.readme_browser.setAutoFillBackground(True)
            rp = self.readme_browser.palette()
            rp.setColor(QPalette.ColorRole.Base, bg)
            rp.setColor(QPalette.ColorRole.Window, bg)
            self.readme_browser.setPalette(rp)
            rv = self.readme_browser.viewport()
            if rv is not None:
                rv.setAutoFillBackground(True)
                vp2 = rv.palette()
                vp2.setColor(QPalette.ColorRole.Base, bg)
                vp2.setColor(QPalette.ColorRole.Window, bg)
                rv.setPalette(vp2)

    def _apply_hub_combo_chevron(self, is_dark: bool) -> None:
        """QSS border triangles render as lines on some styles; use SVG via file URL."""
        if not hasattr(self, "hf_file_combo"):
            return
        name = (
            "hub_combo_chevron_dark.svg"
            if is_dark
            else "hub_combo_chevron_light.svg"
        )
        svg = resource_path("assets", "icons", name)
        if not svg.is_file():
            return
        self._hub_combo_chevron_pixmap = QIcon(str(svg)).pixmap(QSize(12, 12))
        url = QUrl.fromLocalFile(str(svg)).toString()
        self.hf_file_combo.setStyleSheet(
            "#HubFileComboBox { padding-right: 28px; } "
            "#HubFileComboBox::drop-down { "
            "subcontrol-origin: padding; subcontrol-position: top right; width: 24px; border: none; "
            "} "
            "#HubFileComboBox::down-arrow { "
            f'image: url("{url}"); width: 12px; height: 12px; '
            "}"
        )

    def _apply_hub_file_combo_popup_theme(self, is_dark: bool) -> None:
        """Palette + widget-local QSS on the list view (matches MainWindow prestige menus)."""
        if not hasattr(self, "hf_file_combo"):
            return
        theme = self._theme(is_dark)
        v = self.hf_file_combo.view()
        palette = QPalette()
        bg = theme.qcolor(theme.background)
        fg = theme.qcolor(theme.text_primary)
        sel_bg = theme.qcolor(theme.surface_elevated)
        sel_fg = theme.qcolor(theme.text_primary)
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
            palette.setColor(role, bg)
        palette.setColor(QPalette.ColorRole.WindowText, fg)
        palette.setColor(QPalette.ColorRole.Text, fg)
        palette.setColor(QPalette.ColorRole.Highlight, sel_bg)
        palette.setColor(QPalette.ColorRole.HighlightedText, sel_fg)
        v.setPalette(palette)
        v.setStyleSheet(_hub_file_combo_list_qss(theme))
        vp = v.viewport()
        if vp is not None:
            vp.setPalette(palette)
            vp.setStyleSheet(_hub_file_combo_viewport_qss(theme))

    def _on_hub_file_combo_popup_opened(self) -> None:
        """After the popup is shown: detached window + parents often stay system-colored until styled here."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        v = self.hf_file_combo.view()
        if v is not None and getattr(self, "_hub_file_combo_delegate", None) is not None:
            self._hub_file_combo_delegate.setParent(v)
            v.setItemDelegate(self._hub_file_combo_delegate)
        self._apply_hub_file_combo_popup_theme(is_dark)
        self._polish_hub_file_combo_popup_shell(is_dark)

    def _polish_hub_file_combo_popup_shell(self, is_dark: bool) -> None:
        """Paint the popup container / scroll chrome (not a child of HubFileComboBox for QSS)."""
        theme = self._theme(is_dark)
        combo = self.hf_file_combo
        v = combo.view()
        if v is None:
            return
        shell_qss = theme.style(COMBO_POPUP_SHELL)
        bg_qss = theme.style(COMBO_POPUP_VIEWPORT)
        main_win = self.window()

        outer = v.window()
        if outer is not None and outer is not main_win:
            outer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            outer.setStyleSheet(shell_qss)

        p = v.parentWidget()
        depth = 0
        while p is not None and p is not combo and depth < 10:
            p.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            if p is not outer:
                p.setStyleSheet(bg_qss)
            p = p.parentWidget()
            depth += 1

    def _apply_hub_muted_labels(self, is_dark: bool) -> None:
        """Secondary text — matches muted sidebar copy in Chat / Library."""
        theme = self._theme(is_dark)
        hint_style = theme.style(HUB_MUTED_HINT)
        if hasattr(self, "hub_list_hint"):
            self.hub_list_hint.setStyleSheet(hint_style)
        if hasattr(self, "detail_info_btn"):
            self.detail_info_btn.setIcon(
                themed_fa_icon("fa5s.info-circle", theme.color(MUTED_STATUS), 16)
            )
        if hasattr(self, "hub_model_list"):
            row_style = theme.style(HUB_MUTED_ROW)
            for i in range(self.hub_model_list.count()):
                item = self.hub_model_list.item(i)
                row = self.hub_model_list.itemWidget(item)
                if not row:
                    continue
                desc_lbl = row.findChild(QLabel, "HubModelRowDescription")
                ts_lbl = row.findChild(QLabel, "HubModelRowTimestamp")
                if desc_lbl is not None:
                    desc_lbl.setStyleSheet(row_style)
                if ts_lbl is not None:
                    ts_lbl.setStyleSheet(row_style)

    def _append_hub_model_row(
        self,
        title: str,
        repo_id: str,
        *,
        description: str = "",
        capabilities: list[str] | None = None,
        updated_at: str = "",
        verified: bool = False,
        branding: dict | None = None,
        catalog_id: str = "",
        gguf_repos: list[str] | None = None,
        is_catalog: bool = False,
        catalog_publisher: str = "",
        hardware_fit: str = "",
    ) -> None:
        """One Hub row as a dense card with avatar, chips, and metadata."""
        download_repo = str(repo_id or "").strip()
        repos_list = [str(r).strip() for r in (gguf_repos or []) if str(r).strip()]
        if download_repo and download_repo not in repos_list:
            repos_list.insert(0, download_repo)
        if not repos_list and download_repo:
            repos_list = [download_repo]
        item = QListWidgetItem()
        item.setData(HUB_ROW_REPO_ROLE, download_repo)
        item.setData(HUB_ROW_DOWNLOAD_REPO_ROLE, download_repo)
        item.setData(HUB_ROW_TITLE_ROLE, title)
        item.setData(HUB_ROW_DESC_ROLE, description or download_repo)
        item.setData(HUB_ROW_CAPS_ROLE, list(capabilities or []))
        item.setData(HUB_ROW_UPDATED_ROLE, updated_at or "")
        item.setData(HUB_ROW_VERIFIED_ROLE, bool(verified))
        branding_payload = dict(branding or {})
        pub_key = str(catalog_publisher or "").strip().lower()
        if pub_key:
            branding_payload["catalog_publisher"] = pub_key
        item.setData(HUB_ROW_BRANDING_ROLE, branding_payload)
        item.setData(HUB_ROW_CATALOG_ID_ROLE, str(catalog_id or "").strip())
        item.setData(HUB_ROW_GGUF_REPOS_ROLE, repos_list)
        item.setData(HUB_ROW_IS_CATALOG_ROLE, bool(is_catalog))
        item.setData(HUB_ROW_HARDWARE_FIT_ROLE, str(hardware_fit or "").strip())

        row = QWidget()
        row.setObjectName("HistoryRowWidget")
        self._apply_hub_row_surface(row)
        base_l = QHBoxLayout(row)
        base_l.setContentsMargins(12, 8, 10, 8)
        base_l.setSpacing(10)

        avatar = QLabel()
        avatar.setObjectName("HubModelRowAvatar")
        avatar.setFixedSize(24, 24)
        branding_data = dict(branding or {})
        publisher_name = str(branding_data.get("name", "") or "").strip()
        branding_logo = str(branding_data.get("logo", "") or "").strip()
        logo_path = self._resolve_hub_brand_logo(branding_logo)
        is_official = bool(branding_data.get("official", False))
        if logo_path is None:
            logo_path = _default_hub_fallback_logo()
        if logo_path is not None and logo_path.is_file():
            avatar.setPixmap(QIcon(str(logo_path)).pixmap(QSize(22, 22)))
        if is_official and publisher_name:
            avatar.setToolTip(f"Official model by {publisher_name}")
        base_l.addWidget(avatar, stretch=0, alignment=Qt.AlignmentFlag.AlignVCenter)

        right_col = QWidget()
        right_l = QVBoxLayout(right_col)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(4)

        top_row = QWidget()
        top_l = QHBoxLayout(top_row)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setSpacing(6)

        branding_data = dict(branding or {})
        publisher_name = str(branding_data.get("name", "") or "").strip()
        branding_logo = str(branding_data.get("logo", "") or "").strip()
        logo_path = self._resolve_hub_brand_logo(branding_logo)
        is_official = bool(branding_data.get("official", False))
        title_lbl = QLabel(str(title or repo_id))
        title_lbl.setObjectName("HubModelRowTitle")
        title_lbl.setProperty("class", "ModelCardTitle")
        title_lbl.setWordWrap(False)
        if logo_path is not None and publisher_name:
            title_lbl.setToolTip(f"Official model by {publisher_name}")
        top_l.addWidget(title_lbl, stretch=0)

        verified_lbl = QLabel()
        verified_lbl.setObjectName("HubModelRowVerifiedIcon")
        verified_lbl.setPixmap(
            qta.icon("fa5s.award", color=self._theme().accent).pixmap(QSize(12, 12))
        )
        verified_lbl.setVisible(bool(verified))
        top_l.addWidget(verified_lbl, stretch=0)

        if logo_path is not None and is_official:
            official_lbl = QLabel("Official")
            official_lbl.setObjectName("HubModelRowOfficialBadge")
            official_lbl.setStyleSheet(self._official_badge_stylesheet())
            official_lbl.setToolTip(f"Official model by {publisher_name}" if publisher_name else "Official model")
            top_l.addWidget(official_lbl, stretch=0)

        fit_label = str(hardware_fit or "").strip()
        if fit_label:
            fit_lbl = QLabel(fit_label)
            fit_lbl.setObjectName("HubModelRowHardwareFitBadge")
            theme = self._theme()
            if fit_label == "Good fit":
                fit_fg = theme.success
            else:
                fit_fg = theme.warning
            fit_lbl.setStyleSheet(
                f"QLabel {{ color: {fit_fg}; font-size: 10px; font-weight: 700; background: transparent; }}"
            )
            plan = getattr(self, "_catalog_hardware_plan", None)
            tip = fit_label
            if plan is not None:
                for assessment in plan.assessments:
                    if assessment.title == title and assessment.rationale:
                        tip = assessment.rationale
                        break
            fit_lbl.setToolTip(tip)
            top_l.addWidget(fit_lbl, stretch=0)

        top_l.addStretch(1)

        caps_wrap = QWidget()
        caps_wrap_l = QHBoxLayout(caps_wrap)
        caps_wrap_l.setContentsMargins(0, 0, 0, 0)
        caps_wrap_l.setSpacing(4)
        caps_wrap.setObjectName("HubModelRowCapabilities")
        self._populate_hub_capability_chips(caps_wrap, capabilities or [])
        top_l.addWidget(caps_wrap, stretch=0)

        bottom_row = QWidget()
        bottom_l = QHBoxLayout(bottom_row)
        bottom_l.setContentsMargins(0, 0, 0, 0)
        bottom_l.setSpacing(6)

        desc_lbl = QLabel(str(description or repo_id))
        desc_lbl.setObjectName("HubModelRowDescription")
        desc_lbl.setWordWrap(False)
        desc_lbl.setProperty("class", "muted")
        bottom_l.addWidget(desc_lbl, stretch=1)
        bottom_l.addStretch(1)

        ts_lbl = QLabel(str(updated_at or ""))
        ts_lbl.setObjectName("HubModelRowTimestamp")
        ts_lbl.setProperty("class", "muted")
        bottom_l.addWidget(ts_lbl, stretch=0)

        right_l.addWidget(top_row)
        right_l.addWidget(bottom_row)
        base_l.addWidget(right_col, stretch=1)

        self.hub_model_list.addItem(item)
        self.hub_model_list.setItemWidget(item, row)
        self._apply_hub_row_size_hint(item, row)

    def _resolve_hub_brand_logo(self, logo_url: str) -> Path | None:
        return _resolve_hub_brand_logo(logo_url)

    def _apply_detail_branding(self, branding: dict | None) -> None:
        if not hasattr(self, "detail_branding_row"):
            return
        data = dict(branding or {})
        publisher_name = str(data.get("name", "") or "").strip()
        logo_url = str(data.get("logo", "") or "").strip()
        is_official = bool(data.get("official", False))
        logo_path = self._resolve_hub_brand_logo(logo_url)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        official_fg = theme.color(MODEL_HUB_OFFICIAL_BADGE)
        variant_fg = theme.text_muted

        show_official = bool(publisher_name and is_official and logo_path is not None)
        if show_official:
            self.detail_brand_logo.setPixmap(QIcon(str(logo_path)).pixmap(QSize(16, 16)))
            official_text = f"Official model by {publisher_name}"
            self.detail_brand_text.setText(official_text)
            self.detail_brand_text.setStyleSheet(
                f"color: {official_fg}; font-size: 12px; font-weight: 600; background: transparent;"
            )
            self.detail_brand_logo.setToolTip(official_text)
            self.detail_brand_text.setToolTip(official_text)
            self.detail_official_row.show()
        else:
            self.detail_brand_logo.clear()
            self.detail_brand_text.setText("")
            self.detail_brand_logo.setToolTip("")
            self.detail_brand_text.setToolTip("")
            self.detail_official_row.hide()

        catalog_publisher = str(data.get("catalog_publisher", "") or "").strip().lower()
        gguf_repo = str(getattr(self, "_current_repo_id", "") or "").strip()
        gguf_owner = owner_from_repo_id(gguf_repo)
        show_variant = False
        if show_official and gguf_repo and gguf_owner and catalog_publisher and gguf_owner != catalog_publisher:
            variant = self._branding_resolver.resolve_variant_branding(gguf_repo)
            if isinstance(variant, dict) and variant.get("name"):
                variant_name = str(variant.get("name", "") or "").strip()
                variant_logo = self._resolve_hub_brand_logo(str(variant.get("logo", "") or ""))
                if variant_logo is None:
                    variant_logo = _default_hub_fallback_logo()
                if variant_name and variant_logo is not None:
                    self.detail_variant_logo.setPixmap(
                        QIcon(str(variant_logo)).pixmap(QSize(16, 16))
                    )
                    variant_text = f"Modified by {variant_name}"
                    self.detail_variant_text.setText(variant_text)
                    self.detail_variant_text.setStyleSheet(
                        f"color: {variant_fg}; font-size: 12px; font-weight: 600; background: transparent;"
                    )
                    self.detail_variant_logo.setToolTip(variant_text)
                    self.detail_variant_text.setToolTip(variant_text)
                    self.detail_variant_row.show()
                    show_variant = True
        if not show_variant:
            self.detail_variant_logo.clear()
            self.detail_variant_text.setText("")
            self.detail_variant_logo.setToolTip("")
            self.detail_variant_text.setToolTip("")
            self.detail_variant_row.hide()

        if show_official or show_variant:
            self.detail_branding_row.show()
        else:
            self.detail_branding_row.hide()

    def _populate_hub_capability_chips(self, caps_wrap: QWidget, capabilities: list[str]) -> None:
        lay = caps_wrap.layout()
        if lay is None:
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        icon_color = theme.accent
        chip_border = theme.accent
        while lay.count():
            it = lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for cap in (capabilities or [])[:3]:
            icon_name, cap_class = self._capability_chip_spec(cap)
            chip = QFrame()
            chip.setProperty("class", f"Chip outlined {cap_class}")
            chip.setToolTip(str(cap))
            chip.setStyleSheet(
                f"QFrame {{ border: 1px solid {chip_border}; border-radius: 10px; background-color: transparent; }}"
            )
            chip_l = QHBoxLayout(chip)
            chip_l.setContentsMargins(6, 3, 6, 3)
            chip_l.setSpacing(0)
            chip_icon = QLabel()
            chip_icon.setStyleSheet("QLabel { border: none; background: transparent; padding: 0px; margin: 0px; }")
            chip_icon.setPixmap(qta.icon(icon_name, color=icon_color).pixmap(QSize(10, 10)))
            chip_l.addWidget(chip_icon)
            lay.addWidget(chip)

    def _resolve_row_capabilities(
        self,
        *,
        repo_id: str,
        title: str,
        description: str,
        hf_pipeline_tag: str = "",
        hf_tags: list[str] | None = None,
        readme: str | None = None,
        fallback_caps: list[str] | None = None,
    ) -> list[str]:
        model_payload = {
            "id": str(repo_id or "").strip(),
            "name": str(title or "").strip(),
            "description": str(description or "").strip(),
            "readme": str(readme or ""),
            "huggingface": {
                "pipeline_tag": str(hf_pipeline_tag or "").strip(),
                "tags": list(hf_tags or []),
            },
        }
        if fallback_caps:
            # Use fallback inferred capabilities as additional weak hint tags.
            model_payload["huggingface"]["tags"] = list(model_payload["huggingface"]["tags"]) + [
                str(c).lower() for c in fallback_caps if str(c).strip()
            ]
        # Force refresh so new pattern/heuristic updates (e.g., coding) appear immediately in UI.
        caps = self._capability_service.get_or_detect(model_payload, force_refresh=True)
        summary = self._capability_service.summarize_for_ui(caps)
        labels: list[str] = []
        label_map = {
            "reasoning": "Reasoning",
            "tool_use": "Tool Use",
            "vision": "Vision",
            "coding": "Coding",
            "tts": "TTS",
            "stt": "STT",
            "audio": "Audio",
        }
        for key in ("reasoning", "tool_use", "vision", "coding", "audio", "tts", "stt"):
            entry = summary.get(key) or {}
            if bool(entry.get("value", False)):
                labels.append(label_map[key])
        return labels[:4]

    def _update_hub_row_colors(self) -> None:
        """Re-apply hub title fg for selection + theme, then re-layout row heights."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        for i in range(self.hub_model_list.count()):
            it = self.hub_model_list.item(i)
            row = self.hub_model_list.itemWidget(it)
            if row is not None:
                self._apply_hub_row_surface(row)
                self._apply_hub_official_badge_theme(row, is_dark=is_dark)
        self.hub_model_list.viewport().update()
        self._refresh_hub_row_heights()

    def refresh_button_themes(self, is_dark: bool) -> None:
        """Re-apply the brand style + mode-matched icon for the Download action.

        `_apply_download_action_button_style` already routes to the correct
        brand variant AND tints the correct icon through the shared
        `brand_buttons` helper, so theme toggles only need to re-invoke it
        for whichever mode is currently active. The `is_dark` argument is
        kept for parity with `LibraryView.refresh_button_themes` (all brand
        variants intentionally use the same icon color in both themes).
        """
        if not hasattr(self, "download_btn"):
            return
        if getattr(self, "_download_ui_cancel_mode", False):
            self._apply_download_action_button_style(mode="cancel")
            return
        if getattr(self, "_download_ui_load_mode", False):
            self._apply_download_action_button_style(mode="load")
            return
        self._apply_download_action_button_style(mode="download")
        if hasattr(self, "hub_search_retry_btn") and self.hub_search_retry_btn.isVisible():
            apply_brand_primary(self.hub_search_retry_btn, icon_name="fa5s.redo")

    def _set_system_match_style(self, state: str) -> None:
        """Render System label as plain text (no chip card), colored by memory experience."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        if state == "fit":
            color = theme.color(SUCCESS_STATUS)
        elif state in ("caution", "no_fit"):
            color = theme.color(WARNING_STATUS)
        else:
            color = theme.color(MUTED_STATUS)
        if hasattr(self, "system_chip_lbl"):
            self.system_chip_lbl.setStyleSheet(
                f"background: transparent; border: none; padding: 0px; color: {color}; font-size: 12px; font-weight: 600;"
            )

    def _open_current_repo_source(self) -> None:
        repo = str(getattr(self, "_current_repo_id", "") or "").strip()
        if not repo:
            return
        url = QUrl(f"https://huggingface.co/{repo}")
        if url.isValid():
            QDesktopServices.openUrl(url)

    def _schedule_hub_search(self, _text: str = "") -> None:
        self._search_timer.stop()
        self._search_timer.start(500)

    def _hardware_fit_label_for_level(self, fit_level: CatalogFitLevel) -> str:
        if fit_level == CatalogFitLevel.EXCELLENT:
            return "Good fit"
        if fit_level == CatalogFitLevel.GOOD:
            return "Good fit"
        if fit_level == CatalogFitLevel.MARGINAL:
            return "May run slow"
        return ""

    def refresh_hardware_suggestions(self) -> None:
        """Re-apply verified-list ordering when the Settings toggle changes."""
        self._hardware_suggestions_enabled = get_model_manager_hardware_suggestions()
        if not self.isVisible():
            self._hardware_suggestions_dirty = True
            return
        if not self.hub_search_edit.text().strip():
            self._populate_editors_picks()

    def _populate_editors_picks(self) -> None:
        self._search_models_cache = []
        self._search_visible_count = 0
        if hasattr(self, "hub_load_more_btn"):
            self._set_hub_load_more_visible(False)
        verified_models = load_qube_verified_models()
        self._verified_models = verified_models
        suggestions_enabled = get_model_manager_hardware_suggestions()
        self._hardware_suggestions_enabled = suggestions_enabled
        fit_by_id: dict[str, CatalogFitLevel] = {}
        if suggestions_enabled:
            self._catalog_hardware_plan = build_catalog_recommendation_plan(verified_models)
            fit_by_id = {a.catalog_id: a.fit_level for a in self._catalog_hardware_plan.assessments}
            sorted_models = sort_entries_by_hardware_fit(verified_models, self._catalog_hardware_plan)
        else:
            self._catalog_hardware_plan = None
            sorted_models = verified_models
        self.hub_model_list.blockSignals(True)
        self.hub_model_list.clear()
        for entry in sorted_models:
            download_repo = entry.gguf_repo
            title = entry.title
            description = entry.description or download_repo
            branding = branding_for_entry(entry, resolver=self._branding_resolver)
            resolved_caps = self._resolve_row_capabilities(
                repo_id=download_repo,
                title=title,
                description=description,
            )
            fit_level = fit_by_id.get(entry.catalog_id, CatalogFitLevel.UNKNOWN)
            self._append_hub_model_row(
                title,
                download_repo,
                description=description,
                capabilities=resolved_caps,
                updated_at="",
                verified=True,
                branding=branding,
                catalog_id=entry.catalog_id,
                gguf_repos=list(entry.gguf_repos),
                is_catalog=entry.is_catalog_card,
                catalog_publisher=entry.publisher,
                hardware_fit=self._hardware_fit_label_for_level(fit_level) if suggestions_enabled else "",
            )
        self.hub_model_list.blockSignals(False)
        if suggestions_enabled and self._catalog_hardware_plan is not None:
            plan = self._catalog_hardware_plan
            self.hub_list_hint.setText(plan.banner_text)
            hint_tip = f"{plan.detail_text}\n\n{format_tier_detail(plan.profile)}"
            self.hub_list_hint.setToolTip(hint_tip)
        else:
            self.hub_list_hint.setText("Qube Verified — curated GGUF models")
            self.hub_list_hint.setToolTip("")
        self._start_curated_metadata_refresh()
        self._apply_hub_muted_labels(getattr(self.window(), "_is_dark_theme", True))
        self._update_hub_row_colors()
        self._select_first_hub_item()
        QTimer.singleShot(0, self._refresh_hub_row_heights)

    def _start_curated_metadata_refresh(self) -> None:
        verified = getattr(self, "_verified_models", None) or load_qube_verified_models()
        self._curated_meta_queue = [e.gguf_repo for e in verified if e.gguf_repo]
        self._start_next_curated_meta_worker()

    def _start_next_curated_meta_worker(self) -> None:
        if self._curated_meta_worker is not None and self._curated_meta_worker.isRunning():
            return
        if not self._curated_meta_queue:
            return
        repo = self._curated_meta_queue.pop(0)
        self._retire_hf_thread(self._curated_meta_worker)
        self._curated_meta_worker = HfModelMetaWorker(repo)
        self._curated_meta_worker.finished_ok.connect(self._on_curated_meta_finished)
        self._curated_meta_worker.failed.connect(self._on_curated_meta_failed)
        self._curated_meta_worker.finished.connect(self._on_curated_meta_thread_finished)
        self._curated_meta_worker.start()

    def _on_curated_meta_finished(self, repo: str, meta: dict) -> None:
        self._note_hub_success()
        self._apply_curated_card_metadata(repo, meta)

    def _on_curated_meta_failed(self, _repo: str, err: object) -> None:
        info = coerce_hub_error(err)
        self._note_hub_failure(info)

    def _on_curated_meta_thread_finished(self) -> None:
        self._retire_hf_thread(self._curated_meta_worker)
        self._curated_meta_worker = None
        self._start_next_curated_meta_worker()

    def _apply_curated_card_metadata(self, repo: str, meta: dict) -> None:
        if not hasattr(self, "hub_model_list"):
            return
        repo_s = str(repo or "").strip()
        if not repo_s:
            return
        updated = str((meta or {}).get("updated_at", "") or "").strip()
        raw_caps = [str(c).strip() for c in list((meta or {}).get("capabilities") or []) if str(c).strip()]
        for i in range(self.hub_model_list.count()):
            item = self.hub_model_list.item(i)
            if item is None:
                continue
            if str(item.data(HUB_ROW_REPO_ROLE) or "").strip() != repo_s:
                continue
            title = str(item.data(HUB_ROW_TITLE_ROLE) or repo_s)
            desc = str(item.data(HUB_ROW_DESC_ROLE) or repo_s)
            caps = self._resolve_row_capabilities(
                repo_id=repo_s,
                title=title,
                description=desc,
                fallback_caps=raw_caps,
            )
            if updated:
                item.setData(HUB_ROW_UPDATED_ROLE, updated)
            if caps:
                item.setData(HUB_ROW_CAPS_ROLE, caps[:3])
            row = self.hub_model_list.itemWidget(item)
            if row is not None:
                ts_lbl = row.findChild(QLabel, "HubModelRowTimestamp")
                if ts_lbl is not None:
                    ts_lbl.setText(updated)
                caps_wrap = row.findChild(QWidget, "HubModelRowCapabilities")
                if caps_wrap is not None and caps:
                    self._populate_hub_capability_chips(caps_wrap, caps[:3])
                self._apply_hub_row_size_hint(item, row)
            break

    def _select_first_hub_item(self) -> None:
        if self.hub_model_list.count() > 0:
            self.hub_model_list.setCurrentRow(0)

    def _run_hub_search(self) -> None:
        q = self.hub_search_edit.text().strip()
        if not q:
            self._search_seq += 1
            self._set_hub_search_retry_visible(False)
            self._populate_editors_picks()
            return

        self._search_seq += 1
        seq = self._search_seq
        self._search_models_cache = []
        self._search_visible_count = 0
        self._set_hub_search_retry_visible(False)
        self.hub_list_hint.setText("Searching Hugging Face (GGUF-tagged models)…")
        if hasattr(self, "hub_load_more_btn"):
            self._set_hub_load_more_visible(False)

        self._retire_hf_thread(self._search_worker)
        self._search_worker = HfModelSearchWorker(q, seq, limit=200)
        self._search_worker.finished_ok.connect(self._apply_hub_search_results)
        self._search_worker.failed.connect(self._on_hub_search_failed)
        self._search_worker.start()

    def _append_hub_search_result_row(self, m: dict) -> None:
        rid = m.get("repo_id", "")
        title = m.get("title", rid)
        if not rid:
            return
        description = str(m.get("description", "") or rid)
        resolved_caps = self._resolve_row_capabilities(
            repo_id=str(rid),
            title=str(title),
            description=description,
            hf_pipeline_tag=str(m.get("hf_pipeline_tag", "") or ""),
            hf_tags=[str(t) for t in list(m.get("hf_tags") or []) if str(t).strip()],
            fallback_caps=list(m.get("capabilities") or []),
        )
        self._append_hub_model_row(
            title,
            rid,
            description=description,
            capabilities=resolved_caps,
            updated_at=str(m.get("updated_at", "") or ""),
            verified=False,
            branding=dict(m.get("branding") or {}),
        )

    def _render_hub_search_page(self) -> None:
        total = len(self._search_models_cache)
        visible = max(0, min(self._search_visible_count, total))
        prev_repo = ""
        cur = self.hub_model_list.currentItem() if hasattr(self, "hub_model_list") else None
        if cur is not None:
            prev_repo = str(cur.data(HUB_ROW_REPO_ROLE) or "")
        self.hub_model_list.blockSignals(True)
        self.hub_model_list.clear()
        for m in self._search_models_cache[:visible]:
            self._append_hub_search_result_row(m)
        self.hub_model_list.blockSignals(False)
        self._apply_hub_muted_labels(getattr(self.window(), "_is_dark_theme", True))
        self._update_hub_row_colors()
        if total == 0:
            self.hub_list_hint.setText("No GGUF-related models found for this query.")
        elif visible < total:
            self.hub_list_hint.setText(
                f"Showing {visible} of {total} GGUF-related model(s). Click Load More to view more results."
            )
        else:
            self.hub_list_hint.setText(f"Showing all {total} GGUF-related model(s).")
        if hasattr(self, "hub_load_more_btn"):
            self._set_hub_load_more_visible(visible < total)
        if self.hub_model_list.count() > 0:
            restored = False
            if prev_repo:
                for i in range(self.hub_model_list.count()):
                    it = self.hub_model_list.item(i)
                    if str(it.data(HUB_ROW_REPO_ROLE) or "") == prev_repo:
                        self.hub_model_list.setCurrentRow(i)
                        restored = True
                        break
            if not restored:
                self.hub_model_list.setCurrentRow(0)
        QTimer.singleShot(0, self._refresh_hub_row_heights)

    def _load_more_hub_search_results(self) -> None:
        total = len(self._search_models_cache)
        if total <= 0:
            if hasattr(self, "hub_load_more_btn"):
                self._set_hub_load_more_visible(False)
            return
        self._search_visible_count = min(total, self._search_visible_count + HUB_SEARCH_PAGE_SIZE)
        self._render_hub_search_page()

    def _apply_hub_search_results(self, models: list, seq: int) -> None:
        if seq != self._search_seq:
            return
        self._note_hub_success()
        self._set_hub_search_retry_visible(False)
        self._search_models_cache = list(models or [])
        self._search_visible_count = min(HUB_SEARCH_PAGE_SIZE, len(self._search_models_cache))
        self._render_hub_search_page()

    def _on_hub_search_failed(self, err: object, seq: int) -> None:
        if seq != self._search_seq:
            return
        info = coerce_hub_error(err)
        self._note_hub_failure(info)
        self._search_models_cache = []
        self._search_visible_count = 0
        if hasattr(self, "hub_load_more_btn"):
            self._set_hub_load_more_visible(False)
        if info.inline_only:
            self.hub_list_hint.setText(
                "Can't reach Hugging Face — check your connection, then tap Retry search. "
                "Your previous list is still shown below."
            )
            self._set_hub_search_retry_visible(True)
            return
        self.hub_list_hint.setText("Search failed — try different keywords.")
        self._set_hub_search_retry_visible(info.retryable)
        self._show_hub_error_dialog(info, on_retry=self._run_hub_search)

    def _model_description_tooltip_text(
        self,
        description: str,
        download_repo: str,
        *,
        is_catalog: bool,
    ) -> str:
        desc = str(description or "").strip()
        repo_s = str(download_repo or "").strip()
        if desc and desc == repo_s:
            desc = ""
        if is_catalog and desc and repo_s:
            return f"{desc}\n\nGGUF source: {repo_s}"
        if desc:
            return desc
        if repo_s:
            return f"GGUF source: {repo_s}"
        return ""

    def _apply_detail_description_info(
        self,
        description: str,
        download_repo: str,
        *,
        is_catalog: bool,
    ) -> None:
        if not hasattr(self, "detail_info_btn"):
            return
        tip = self._model_description_tooltip_text(
            description, download_repo, is_catalog=is_catalog
        )
        if tip:
            self.detail_info_btn.setToolTip(tip)
            self.detail_info_btn.setVisible(True)
        else:
            self.detail_info_btn.setToolTip("")
            self.detail_info_btn.setVisible(False)

    def _sync_catalog_gguf_repos_from_item(self, current: QListWidgetItem) -> None:
        raw = current.data(HUB_ROW_GGUF_REPOS_ROLE)
        repos: list[str] = []
        if isinstance(raw, (list, tuple)):
            repos = [str(r).strip() for r in raw if str(r).strip()]
        download = str(
            current.data(HUB_ROW_DOWNLOAD_REPO_ROLE) or current.data(HUB_ROW_REPO_ROLE) or ""
        ).strip()
        if download and download not in repos:
            repos.insert(0, download)
        if not repos and download:
            repos = [download]
        self._catalog_gguf_repos = tuple(repos)
        self._catalog_gguf_repo_index = 0

    def _try_next_catalog_gguf_repo(self, seq: int) -> bool:
        """When primary gguf_repo has no files, try the next entry in gguf_repos."""
        repos = getattr(self, "_catalog_gguf_repos", ())
        idx = getattr(self, "_catalog_gguf_repo_index", 0)
        if idx + 1 >= len(repos):
            return False
        next_idx = idx + 1
        next_repo = str(repos[next_idx]).strip()
        if not next_repo:
            return False
        self._catalog_gguf_repo_index = next_idx
        self._current_repo_id = next_repo
        current = self.hub_model_list.currentItem() if hasattr(self, "hub_model_list") else None
        if current is not None:
            current.setData(HUB_ROW_REPO_ROLE, next_repo)
            current.setData(HUB_ROW_DOWNLOAD_REPO_ROLE, next_repo)
            desc = str(current.data(HUB_ROW_DESC_ROLE) or "")
            self._apply_detail_description_info(
                desc,
                next_repo,
                is_catalog=bool(current.data(HUB_ROW_IS_CATALOG_ROLE)),
            )
            self._apply_detail_branding(dict(current.data(HUB_ROW_BRANDING_ROLE) or {}))
        logger.info("Catalog GGUF fallback: trying alternate repo %s", next_repo)
        self._reload_hub_detail_workers(next_repo, seq)
        return True

    def _reload_hub_detail_workers(self, repo: str, seq: int) -> None:
        """Restart README, metadata, and file-list workers for ``repo``."""
        self.readme_browser.clear()
        self.readme_browser.setPlainText("Loading README…")
        self._last_readme_markdown = None
        self.hf_file_combo.blockSignals(True)
        self.hf_file_combo.clear()
        self.hf_file_combo.addItem("Loading file list…")
        self.hf_file_combo.blockSignals(False)
        if hasattr(self, "hub_quant_hint_lbl"):
            self.hub_quant_hint_lbl.setText("Fetching .gguf file list…")
        self._retire_hf_thread(self._readme_worker)
        self._readme_worker = None
        self._retire_hf_thread(self._list_worker)
        self._list_worker = None
        self._retire_hf_thread(self._meta_worker)
        self._meta_worker = None
        self._readme_worker = HfReadmeWorker(str(repo))
        self._readme_worker.finished_ok.connect(
            lambda r, t, s=seq: self._apply_readme_if_current(r, t, s)
        )
        self._readme_worker.failed.connect(
            lambda r, err, s=seq: self._apply_readme_failed_if_current(r, err, s)
        )
        self._readme_worker.start()
        self._reset_hub_metadata_labels()
        self._set_meta_hint("Loading model metadata…")
        self._meta_worker = HfModelMetaWorker(str(repo))
        self._meta_worker.finished_ok.connect(
            lambda r, meta, s=seq: self._apply_meta_if_current(r, meta, s)
        )
        self._meta_worker.failed.connect(
            lambda r, err, s=seq: self._apply_meta_failed_if_current(r, err, s)
        )
        self._meta_worker.start()
        self._start_list_worker_for_repo(str(repo), seq)

    def _on_hub_selection_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if not current:
            self._clear_detail_pane()
            return
        repo = current.data(HUB_ROW_DOWNLOAD_REPO_ROLE) or current.data(HUB_ROW_REPO_ROLE)
        title = current.data(HUB_ROW_TITLE_ROLE) or repo
        if not repo:
            return
        self._scroll_detail_to_top()
        self._detail_seq += 1
        seq = self._detail_seq
        download_repo = str(repo)
        self._current_repo_id = download_repo
        self._sync_catalog_gguf_repos_from_item(current)
        self.detail_title.setText(str(title))
        is_catalog = bool(current.data(HUB_ROW_IS_CATALOG_ROLE))
        desc = str(current.data(HUB_ROW_DESC_ROLE) or "")
        self._apply_detail_description_info(
            desc,
            download_repo,
            is_catalog=is_catalog,
        )
        if hasattr(self, "detail_source_btn"):
            self.detail_source_btn.setVisible(True)
        self._apply_detail_branding(dict(current.data(HUB_ROW_BRANDING_ROLE) or {}))
        self._set_download_status_text("")
        self._reload_hub_detail_workers(download_repo, seq)

    def _clear_detail_pane(self) -> None:
        self._retire_hf_thread(self._readme_worker)
        self._readme_worker = None
        self._retire_hf_thread(self._list_worker)
        self._list_worker = None
        self._retire_hf_thread(self._meta_worker)
        self._meta_worker = None
        self._current_repo_id = ""
        self._catalog_gguf_repos = ()
        self._catalog_gguf_repo_index = 0
        self._last_readme_markdown = None
        self.detail_title.setText("Select a model")
        self._apply_detail_description_info("", "", is_catalog=False)
        if hasattr(self, "detail_source_btn"):
            self.detail_source_btn.setVisible(False)
        self._apply_detail_branding(None)
        if hasattr(self, "detail_variant_row"):
            self.detail_variant_row.hide()
        self.readme_browser.clear()
        self.hf_file_combo.blockSignals(True)
        self.hf_file_combo.clear()
        self.hf_file_combo.addItem("-- Select a model from the list --")
        self.hf_file_combo.blockSignals(False)
        self._reset_hub_metadata_labels()
        self._hub_meta_snapshot = None
        self._quant_rec_context = None
        self._quant_rec_plan = None
        if hasattr(self, "hub_quant_hint_lbl"):
            self.hub_quant_hint_lbl.setText("Select a model to view available quantizations.")
        if hasattr(self, "hub_quant_rationale_row"):
            self.hub_quant_rationale_row.hide()
        if hasattr(self, "hub_quant_rationale_lbl"):
            self.hub_quant_rationale_lbl.setText("")
        if hasattr(self, "hub_quant_rationale_badge"):
            self.hub_quant_rationale_badge.setText("")
        self._update_download_button_label()
        self._update_gpu_fit_status()
        self._sync_download_action_state()

    def _apply_meta_if_current(self, repo: str, meta: dict, seq: int) -> None:
        if seq != self._detail_seq:
            return
        self._note_hub_success()
        current = self.hub_model_list.currentItem() if hasattr(self, "hub_model_list") else None
        row_title = str(current.data(HUB_ROW_TITLE_ROLE) or self.detail_title.text() or repo) if current else str(self.detail_title.text() or repo)
        row_desc = str(current.data(HUB_ROW_DESC_ROLE) or repo) if current else str(repo)
        hf_tags = [str(t) for t in list((meta or {}).get("hf_tags") or []) if str(t).strip()]
        hf_pipeline_tag = str((meta or {}).get("hf_pipeline_tag") or "")
        resolved_caps = self._resolve_row_capabilities(
            repo_id=str(repo),
            title=row_title,
            description=row_desc,
            hf_pipeline_tag=hf_pipeline_tag,
            hf_tags=hf_tags,
            fallback_caps=list((meta or {}).get("capabilities") or []),
        )
        meta = dict(meta or {})
        meta["capabilities"] = resolved_caps
        self._hub_meta_snapshot = meta
        self._quant_rec_context = self._build_quant_recommendation_context()
        self._apply_hub_metadata(meta)
        if self.hf_file_combo.count() > 1:
            self._refresh_quant_recommendations()

    def _apply_meta_failed_if_current(self, repo: str, err: object, seq: int) -> None:
        if seq != self._detail_seq:
            return
        info = coerce_hub_error(err)
        self._note_hub_failure(info)
        self._reset_hub_metadata_labels()
        if info.is_platform_outage:
            self._set_meta_hint(
                "Hugging Face is unreachable — metadata is unavailable. "
                "Quantization listing and README may also fail until connectivity returns."
            )
        else:
            self._set_meta_hint(
                "Model metadata unavailable for this repository. "
                "Quantization and README are still available."
            )

    def _sync_readme_panel_height(self) -> None:
        """Size README to its content so the outer detail scroll owns vertical scrolling."""
        if not hasattr(self, "readme_browser"):
            return
        browser = self.readme_browser
        vp_w = browser.viewport().width()
        width = max(200, vp_w if vp_w > 0 else browser.width())
        doc = browser.document()
        doc.setTextWidth(float(width))
        layout = doc.documentLayout()
        if layout is not None:
            h = int(layout.documentSize().height()) + 24
        else:
            h = 180
        h = max(180, min(h, 16000))
        browser.setFixedHeight(h)
        if hasattr(self, "detail_cards_content"):
            self.detail_cards_content.updateGeometry()
        if hasattr(self, "detail_scroll"):
            self.detail_scroll.updateGeometry()

    def _scroll_detail_to_top(self) -> None:
        if hasattr(self, "detail_scroll"):
            bar = self.detail_scroll.verticalScrollBar()
            if bar is not None:
                bar.setValue(bar.minimum())

    def _render_readme_with_fallback(self, is_dark: bool) -> None:
        """Python-Markdown + setHtml first (GFM tables/lists); then Qt setMarkdown; then plain text."""
        text = self._last_readme_markdown
        if not text:
            return
        prepared = strip_hub_readme_preamble(text)
        if not prepared:
            self.readme_browser.setHtml('<div class="hub-readme"></div>')
            QTimer.singleShot(0, self._sync_readme_panel_height)
            return
        doc = self.readme_browser.document()
        doc.setDefaultFont(self.readme_browser.font())
        doc.setDefaultStyleSheet(markdown_document_stylesheet(is_dark))
        html = hf_readme_markdown_to_safe_html(text)
        if html is not None:
            self.readme_browser.setHtml(html)
        else:
            try:
                self.readme_browser.setMarkdown(prepared)
            except Exception:
                self.readme_browser.setPlainText(prepared)
        QTimer.singleShot(0, self._sync_readme_panel_height)

    def _apply_readme_if_current(self, repo: str, text: str, seq: int) -> None:
        if seq != self._detail_seq:
            return
        self._note_hub_success()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._last_readme_markdown = text
        self._render_readme_with_fallback(is_dark)
        guidance = self._publisher_guidance_service.extract_and_store(repo, text)
        self._set_meta_hint(self._publisher_guidance_service.summarize_for_ui(guidance))
        current = self.hub_model_list.currentItem() if hasattr(self, "hub_model_list") else None
        if current is None:
            return
        repo_id = str(current.data(HUB_ROW_REPO_ROLE) or repo)
        title = str(current.data(HUB_ROW_TITLE_ROLE) or repo_id)
        desc = str(current.data(HUB_ROW_DESC_ROLE) or repo_id)
        refreshed_caps = self._resolve_row_capabilities(
            repo_id=repo_id,
            title=title,
            description=desc,
            readme=text,
            fallback_caps=list(current.data(HUB_ROW_CAPS_ROLE) or []),
        )
        current.setData(HUB_ROW_CAPS_ROLE, refreshed_caps)
        row = self.hub_model_list.itemWidget(current)
        if row is not None:
            caps_wrap = row.findChild(QWidget, "HubModelRowCapabilities")
            if caps_wrap is not None:
                self._populate_hub_capability_chips(caps_wrap, refreshed_caps)
            self._apply_hub_row_size_hint(current, row)
        self._render_capability_chips(refreshed_caps)

    def _apply_readme_failed_if_current(self, repo: str, err: object, seq: int) -> None:
        if seq != self._detail_seq:
            return
        info = coerce_hub_error(err)
        self._note_hub_failure(info)
        self._last_readme_markdown = None
        self.readme_browser.setPlainText(
            f"Could not load README for `{repo}`.\n\n{info.message}\n\n"
            "You can still pick a .gguf file below if the repo file list loads successfully."
        )

    def _start_list_worker_for_repo(self, repo: str, seq: int) -> None:
        self.download_btn.setEnabled(False)
        if hasattr(self, "hub_quant_hint_lbl"):
            self.hub_quant_hint_lbl.setText("Fetching .gguf file list…")

        self._retire_hf_thread(self._list_worker)
        self._list_worker = None
        self._list_worker = HfRepoFilesWorker(repo)
        self._list_worker.finished_ok.connect(
            lambda paths, s=seq: self._on_hf_list_finished(paths, s)
        )
        self._list_worker.failed.connect(lambda err, s=seq: self._on_hf_list_failed(err, s))
        self._list_worker.finished.connect(self._on_hf_list_thread_finished)
        self._list_worker.start()

    def _on_hf_list_thread_finished(self) -> None:
        dl_busy = self._download_worker and self._download_worker.isRunning()
        if not dl_busy:
            self.download_btn.setEnabled(True)

    def _on_hf_list_finished(self, entries: list, seq: int) -> None:
        if seq != self._detail_seq:
            return
        self._note_hub_success()
        self.hf_file_combo.blockSignals(True)
        self.hf_file_combo.clear()
        normalized: list[tuple[str, int | None]] = []
        for e in entries:
            if isinstance(e, str):
                normalized.append((e, None))
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                raw_sz = e[1]
                sz: int | None = None
                if raw_sz is not None:
                    try:
                        sz = int(raw_sz)
                    except (TypeError, ValueError):
                        sz = None
                normalized.append((str(e[0]), sz))
            elif isinstance(e, (list, tuple)) and len(e) == 1:
                normalized.append((str(e[0]), None))
        # Hide secondary shard fragments (N-of-M where N>1) from the picker and
        # collapse shard sets into one logical selection row.
        hidden_secondary = 0
        filtered: list[tuple[str, int | None]] = []
        shard_totals: dict[str, int] = {}
        shard_group_entries: dict[str, list[tuple[int, str, int | None]]] = {}
        for path, size_b in normalized:
            s_info = parse_gguf_shard_info(path)
            if s_info is not None:
                key = str(s_info["prefix"]).lower()
                try:
                    shard_totals[key] = max(shard_totals.get(key, 0), int(s_info["total"]))
                except (TypeError, ValueError):
                    pass
                try:
                    part_i = int(s_info["part"])
                except (TypeError, ValueError):
                    part_i = 1
                shard_group_entries.setdefault(key, []).append((part_i, path, size_b))
            if is_secondary_gguf_shard(path):
                hidden_secondary += 1
                continue
            filtered.append((path, size_b))
        normalized = filtered

        # Smallest-to-largest by known size; unknown sizes are sent to the end.
        normalized.sort(
            key=lambda it: (
                it[1] is None,
                int(it[1]) if it[1] is not None else 0,
                str(it[0]).lower(),
            )
        )
        if not normalized:
            if self._try_next_catalog_gguf_repo(seq):
                return
            self.hf_file_combo.addItem("(No .gguf files in this repository)")
            if hasattr(self, "hub_quant_hint_lbl"):
                self.hub_quant_hint_lbl.setText("No .gguf files found for this repository.")
        else:
            self.hf_file_combo.addItem("-- Select a .gguf file --")
            for path, size_b in normalized:
                label = path
                shard_entries: list[tuple[str, int | None]] = [(path, size_b)]
                agg_size_b = size_b if isinstance(size_b, int) else None
                s_info = parse_gguf_shard_info(path)
                if s_info is not None:
                    key = str(s_info["prefix"]).lower()
                    total = shard_totals.get(key, int(s_info["total"]))
                    bundle_name = f"{Path(str(s_info['prefix'])).name}.gguf"
                    label = f"{bundle_name} ({total} shards)"
                    grp = shard_group_entries.get(key) or []
                    if grp:
                        grp_sorted = sorted(grp, key=lambda it: (int(it[0]), str(it[1]).lower()))
                        shard_entries = [(p, sz) for _, p, sz in grp_sorted]
                        known_sizes = [int(sz) for _, _, sz in grp_sorted if isinstance(sz, int)]
                        if len(known_sizes) == len(grp_sorted):
                            agg_size_b = sum(known_sizes)
                self.hf_file_combo.addItem(label, path)
                idx = self.hf_file_combo.count() - 1
                self.hf_file_combo.setItemData(
                    idx,
                    shard_entries,
                    int(HUB_FILE_COMBO_SHARD_ENTRIES_ROLE),
                )
                if agg_size_b is not None:
                    self.hf_file_combo.setItemData(
                        idx,
                        self._fmt_bytes(agg_size_b),
                        int(HUB_FILE_COMBO_SIZE_ROLE),
                    )
                    self.hf_file_combo.setItemData(
                        idx,
                        int(agg_size_b),
                        int(HUB_FILE_COMBO_BYTES_ROLE),
                    )
            self._quant_rec_context = self._build_quant_recommendation_context()
            self._apply_quant_recommendations_to_combo(normalized)
            if hasattr(self, "hub_quant_hint_lbl") and not (self._quant_rec_plan and self._quant_rec_plan.summary_hint):
                msg = f"{len(normalized)} file(s) available. Choose a quantization, then Download."
                if hidden_secondary > 0:
                    msg += f" ({hidden_secondary} shard fragment file(s) hidden.)"
                self.hub_quant_hint_lbl.setText(msg)
        self.hf_file_combo.blockSignals(False)
        self._update_download_button_label()
        self._update_download_selection_hint()
        self._update_quant_rationale_label()
        self._update_gpu_fit_status()
        self._sync_download_action_state()
        self._refresh_download_options_card_geometry()
        self._try_complete_pending_hub_redownload(seq)

    def request_hub_redownload(self, repo_id: str, filename: str) -> None:
        """Open a Hub repo and start downloading ``filename`` when the file list is ready."""
        repo = str(repo_id or "").strip()
        fname = Path(str(filename or "").strip()).name
        if not repo or not fname:
            return
        self._pending_hub_redownload = (repo, fname)
        if not self._select_hub_repo_by_id(repo):
            self._detail_seq += 1
            seq = self._detail_seq
            self._current_repo_id = repo
            if hasattr(self, "detail_title"):
                self.detail_title.setText(repo)
            if hasattr(self, "detail_source_btn"):
                self.detail_source_btn.setVisible(True)
            self._reload_hub_detail_workers(repo, seq)

    def _select_hub_repo_by_id(self, repo_id: str) -> bool:
        repo = str(repo_id or "").strip()
        if not repo or not hasattr(self, "hub_model_list"):
            return False
        for row in range(self.hub_model_list.count()):
            item = self.hub_model_list.item(row)
            if item is None:
                continue
            candidate = str(
                item.data(HUB_ROW_DOWNLOAD_REPO_ROLE) or item.data(HUB_ROW_REPO_ROLE) or ""
            ).strip()
            if candidate == repo:
                self.hub_model_list.setCurrentItem(item)
                return True
        return False

    def _try_complete_pending_hub_redownload(self, seq: int) -> None:
        pending = getattr(self, "_pending_hub_redownload", None)
        if not pending:
            return
        repo_id, filename = pending
        if seq != self._detail_seq or self._current_repo_id.strip() != repo_id:
            return
        target = filename.lower()
        if not hasattr(self, "hf_file_combo"):
            self._pending_hub_redownload = None
            return
        for i in range(1, self.hf_file_combo.count()):
            path = self.hf_file_combo.itemData(i)
            if path and Path(str(path)).name.lower() == target:
                self.hf_file_combo.blockSignals(True)
                self.hf_file_combo.setCurrentIndex(i)
                self.hf_file_combo.blockSignals(False)
                self._pending_hub_redownload = None
                self._update_download_selection_hint()
                self._update_quant_rationale_label()
                self._sync_download_action_state()
                from PyQt6.QtCore import QTimer

                QTimer.singleShot(0, self._start_download)
                return
        self._pending_hub_redownload = None

    def _on_hf_list_failed(self, err: object, seq: int) -> None:
        if seq != self._detail_seq:
            return
        info = coerce_hub_error(err)
        self._note_hub_failure(info)
        if self._try_next_catalog_gguf_repo(seq):
            return
        self.hf_file_combo.blockSignals(True)
        self.hf_file_combo.clear()
        self.hf_file_combo.addItem("-- Could not list files --")
        self.hf_file_combo.blockSignals(False)
        if hasattr(self, "hub_quant_hint_lbl"):
            self.hub_quant_hint_lbl.setText("Could not list .gguf files for this repository.")
        self._update_download_button_label()
        self._update_gpu_fit_status()
        self._sync_download_action_state()

        def _retry_list() -> None:
            repo = str(getattr(self, "_current_repo_id", "") or "").strip()
            if repo:
                self._start_list_worker_for_repo(repo, seq)

        self._show_hub_error_dialog(info, on_retry=_retry_list if info.retryable else None)

    def _section_header(self, icon_name: str, text: str) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = self._theme(is_dark)
        icon_color = theme.accent if theme.is_dark else theme.text_secondary
        ic = QLabel()
        ic.setProperty("icon_name", icon_name)
        ic.setPixmap(qta.icon(icon_name, color=icon_color).pixmap(QSize(18, 18)))
        self._hub_section_icon_label = ic
        lbl = QLabel(text)
        lbl.setProperty("class", "SectionHeaderLabel")
        h.addWidget(ic)
        h.addWidget(lbl)
        h.addStretch()
        return row

    def _divider(self) -> QFrame:
        line = QFrame()
        line.setObjectName("SettingsDivider")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    def _selected_hf_repo_file(self) -> str | None:
        if self.hf_file_combo.count() == 0:
            return None
        i = self.hf_file_combo.currentIndex()
        if i < 0:
            return None
        raw = self.hf_file_combo.itemData(i, int(Qt.ItemDataRole.UserRole))
        if isinstance(raw, str) and raw.lower().endswith(".gguf"):
            return raw
        t = self.hf_file_combo.itemText(i).strip()
        if t.startswith("--") or t.startswith("("):
            return None
        if not t.lower().endswith(".gguf"):
            return None
        return t

    def _selected_hf_repo_files_for_download(self) -> list[tuple[str, int | None]]:
        if self.hf_file_combo.count() == 0:
            return []
        i = self.hf_file_combo.currentIndex()
        if i < 0:
            return []
        raw_entries = self.hf_file_combo.itemData(i, int(HUB_FILE_COMBO_SHARD_ENTRIES_ROLE))
        out: list[tuple[str, int | None]] = []
        if isinstance(raw_entries, (list, tuple)):
            for ent in raw_entries:
                if not isinstance(ent, (list, tuple)) or len(ent) < 1:
                    continue
                path = str(ent[0] or "").strip()
                if not path.lower().endswith(".gguf"):
                    continue
                sz: int | None = None
                if len(ent) > 1 and ent[1] is not None:
                    try:
                        sz = int(ent[1])
                    except (TypeError, ValueError):
                        sz = None
                out.append((path, sz))
        if out:
            return out
        one = self._selected_hf_repo_file()
        if not one:
            return []
        raw_sz = self.hf_file_combo.itemData(i, int(HUB_FILE_COMBO_BYTES_ROLE))
        try:
            one_sz = int(raw_sz) if raw_sz is not None else None
        except (TypeError, ValueError):
            one_sz = None
        return [(one, one_sz)]

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        n = max(0, int(n))
        if n >= 1024**4:
            return f"{n / 1024**4:.2f} TB"
        if n >= 1024**3:
            return f"{n / 1024**3:.2f} GB"
        if n >= 1024**2:
            return f"{n / 1024**2:.0f} MB"
        return f"{n} bytes"

    def _set_download_button_cancel_mode(self, cancel_mode: bool) -> None:
        try:
            self.download_btn.clicked.disconnect()
        except TypeError:
            pass
        self._download_ui_cancel_mode = cancel_mode
        self._download_ui_load_mode = False
        if cancel_mode:
            self.download_btn.setText("Cancel")
            self._apply_download_action_button_style(mode="cancel")
            self.download_btn.clicked.connect(self._cancel_download)
        else:
            self._set_download_button_download_mode()
            return
        self._sync_download_button_tooltip()

    def _restore_download_idle_ui(self, *, clear_queue: bool = True) -> None:
        self._download_ui_cancel_mode = False
        self.download_progress.hide()
        self.download_progress.setValue(0)
        self.download_progress.setFormat("(%p%)")
        if clear_queue:
            self._reset_download_queue_state()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_hub_file_combo_popup_theme(is_dark)
        self._sync_download_action_state()

    def _cancel_download(self) -> None:
        if self._download_worker and self._download_worker.isRunning():
            self._set_download_status_text("Cancelling…")
            self._download_worker.cancel()

    def _start_download(self) -> None:
        if self._download_worker and self._download_worker.isRunning():
            self._show_error("Busy", "A download is already in progress.")
            return
        repo = self._current_repo_id.strip()
        entries = self._selected_hf_repo_files_for_download()
        if not repo:
            self._show_error("No model", "Select a model from the Hub list first.")
            return
        if not entries:
            self._show_error(
                "No file selected",
                "Wait for the file list to load, then choose a .gguf variant.",
            )
            return

        self.download_progress.setValue(0)
        self.download_progress.show()
        self._set_download_status_text("")
        self._set_download_progress_text()
        self._set_download_button_cancel_mode(True)
        self._download_queue_paths = list(entries)
        self._download_queue_index = 0
        self._download_completed_paths = []
        self._download_failed_path = None
        self._download_total_bytes = sum(int(sz) for _, sz in entries if isinstance(sz, int) and sz > 0)
        self._download_completed_bytes = 0
        self._download_current_bytes_total = 0
        self._download_current_path = ""
        self._start_next_shard_download(repo)

    def _start_next_shard_download(self, repo: str) -> None:
        n = len(self._download_queue_paths)
        if n == 0:
            return
        if self._download_queue_index >= n:
            # Queue complete.
            first = self._download_queue_paths[0][0]
            local_first = Path(get_llm_models_dir()) / Path(first).name
            resolved = resolve_internal_model_path(str(local_first))
            self._restore_download_idle_ui()
            self._set_download_status_text(f"Saved: {os.path.basename(resolved)} ({n}/{n} shards)")
            set_internal_model_path(resolved)
            repo = str(getattr(self, "_current_repo_id", "") or "").strip()
            if repo:
                self._publisher_guidance_service.record_provenance(resolved, repo)
            self.native_library_changed.emit()
            self._sync_download_action_state()
            self.download_succeeded.emit(os.path.basename(resolved))
            return

        fname, sz = self._download_queue_paths[self._download_queue_index]
        self._download_current_path = str(fname)
        self._download_current_bytes_total = int(sz) if isinstance(sz, int) and sz > 0 else 0
        step = self._download_queue_index + 1
        self._set_download_status_text(f"Downloading shard {step}/{n}: {Path(fname).name}")
        self._set_download_progress_text(f"Downloading shard {step}/{n}")

        self._retire_hf_thread(self._download_worker)
        self._download_worker = None
        self._download_worker = HuggingFaceGgufDownloadWorker(
            repo, fname, get_llm_models_dir()
        )
        self._download_worker.progress_pct.connect(self._on_download_progress_pct)
        self._download_worker.status_message.connect(self._on_download_status_message)
        self._download_worker.finished_ok.connect(self._on_download_finished)
        self._download_worker.failed.connect(self._on_download_failed)
        self._download_worker.insufficient_space_error.connect(
            self._on_insufficient_space
        )
        self._download_worker.download_cancelled.connect(self._on_download_cancelled)
        self._download_worker.start()

    def _on_download_finished(self, path: str) -> None:
        self._note_hub_success()
        if self._download_queue_active():
            self._download_completed_paths.append(str(path))
            if self._download_current_bytes_total > 0:
                self._download_completed_bytes += self._download_current_bytes_total
            self._download_queue_index += 1
            n = len(self._download_queue_paths)
            if self._download_queue_index < n:
                self._set_download_status_text(
                    f"Saved {self._download_queue_index}/{n} shards"
                )
                self._start_next_shard_download(self._current_repo_id.strip())
                return
            self._start_next_shard_download(self._current_repo_id.strip())
            return
        self._restore_download_idle_ui()
        resolved = resolve_internal_model_path(path)
        self._set_download_status_text(f"Saved: {os.path.basename(resolved)}")
        set_internal_model_path(resolved)
        # Do not auto-load after download; user can manually select/load later.
        self.native_library_changed.emit()
        self._sync_download_action_state()
        self.download_succeeded.emit(os.path.basename(resolved))

    def _on_download_failed(self, err: object) -> None:
        info = coerce_hub_error(err)
        self._note_hub_failure(info)
        failed_name = Path(self._download_current_path).name if self._download_current_path else "model"
        message = self._format_download_failure_message(info)
        dialog_info = HubErrorInfo(
            kind=info.kind,
            title=info.title if not self._download_queue_active() else "Download failed",
            message=message,
            technical_detail=info.technical_detail,
            retryable=info.retryable,
            show_status_link=info.show_status_link,
        )
        if self._download_queue_active():
            self._download_failed_path = failed_name
            n = len(self._download_queue_paths)
            done = self._download_queue_index
            self._pending_download_retry = info.retryable
            self._restore_download_idle_ui(clear_queue=not info.retryable)
            self._set_download_status_text(f"Download failed after {done}/{n} shards")
            self._show_hub_error_dialog(
                dialog_info,
                on_retry=self._retry_download_from_current_shard if info.retryable else None,
            )
            return
        self._restore_download_idle_ui()
        self._set_download_status_text("")
        self._show_hub_error_dialog(
            dialog_info,
            on_retry=self._start_download if info.retryable else None,
        )

    def _on_insufficient_space(self, required: int, available: int) -> None:
        failed_name = Path(self._download_current_path).name if self._download_current_path else "model"
        if self._download_queue_active():
            n = len(self._download_queue_paths)
            done = self._download_queue_index
            self._restore_download_idle_ui()
            self._set_download_status_text(f"Stopped at {done}/{n} shards (disk full)")
            self._show_error(
                "Not enough disk space",
                f"Stopped while downloading shard {done + 1}/{n}: {failed_name}\n\n"
                f"This download needs about {self._fmt_bytes(required)} free on the destination "
                f"drive (including a 500 MB safety margin). "
                f"You have about {self._fmt_bytes(available)} available.",
            )
            return
        self._restore_download_idle_ui()
        self._set_download_status_text("")
        self._show_error(
            "Not enough disk space",
            f"This download needs about {self._fmt_bytes(required)} free on the destination "
            f"drive (including a 500 MB safety margin). "
            f"You have about {self._fmt_bytes(available)} available.",
        )

    def _on_download_cancelled(self) -> None:
        if self._download_queue_active():
            n = len(self._download_queue_paths)
            done = self._download_queue_index
            self._restore_download_idle_ui()
            self._set_download_status_text(f"Download cancelled ({done}/{n} shards saved).")
            return
        self._restore_download_idle_ui()
        self._set_download_status_text("Download cancelled.")

    def _apply_hub_connectivity_banner(self, is_dark: bool) -> None:
        if not hasattr(self, "hub_status_banner"):
            return
        theme = self._theme(is_dark)
        self.hub_status_banner.setStyleSheet(
            theme.style(CONNECTIVITY_ERROR_BANNER, object_name="ModelManagerHubStatusBanner")
        )
        self.hub_status_banner_icon.setPixmap(
            qta.icon("fa5s.exclamation-triangle", color=theme.error).pixmap(QSize(16, 16))
        )
        if self._hub_reachable is False:
            detail = self._hub_status_detail.strip()
            text = (
                "Can't reach Hugging Face — Hub search and downloads are unavailable. "
                "Browse Qube Verified models below or retry when you're back online."
            )
            if detail:
                text = f"{text}\n{detail}"
            self.hub_status_banner_text.setText(text)
            self.hub_status_banner.setVisible(True)
        else:
            self.hub_status_banner.setVisible(False)

    def _set_hub_search_retry_visible(self, visible: bool) -> None:
        if hasattr(self, "hub_search_retry_btn"):
            self.hub_search_retry_btn.setVisible(bool(visible))

    def _note_hub_success(self) -> None:
        self._hub_reachable = True
        self._hub_status_detail = ""
        self._set_hub_search_retry_visible(False)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_hub_connectivity_banner(is_dark)

    def _note_hub_failure(self, info: HubErrorInfo) -> None:
        info = coerce_hub_error(info)
        if not info.is_platform_outage:
            return
        self._hub_reachable = False
        self._hub_status_detail = info.message
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_hub_connectivity_banner(is_dark)

    def _start_hub_connectivity_probe(self) -> None:
        if self._hub_probe_worker is not None and self._hub_probe_worker.isRunning():
            return
        self._retire_hf_thread(self._hub_probe_worker)
        self._hub_probe_worker = HfConnectivityProbeWorker()
        self._hub_probe_worker.finished_ok.connect(self._on_hub_probe_ok)
        self._hub_probe_worker.failed.connect(self._on_hub_probe_failed)
        self._hub_probe_worker.start()

    def _on_hub_probe_ok(self) -> None:
        self._note_hub_success()

    def _on_hub_probe_failed(self, info: object) -> None:
        self._note_hub_failure(coerce_hub_error(info))

    def _show_hub_error_dialog(
        self,
        info: HubErrorInfo,
        *,
        on_retry=None,
        force_modal: bool = False,
    ) -> None:
        info = coerce_hub_error(info)
        if info.inline_only and not force_modal:
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = HubErrorDialog(self.window(), info, is_dark=is_dark)
        if dlg.exec_retry() and on_retry is not None:
            on_retry()

    def _format_download_failure_message(self, info: HubErrorInfo) -> str:
        info = coerce_hub_error(info)
        lines = [info.message]
        if self._download_queue_active():
            n = len(self._download_queue_paths)
            done = self._download_queue_index
            failed_name = (
                Path(self._download_current_path).name if self._download_current_path else "model"
            )
            lines.insert(0, f"Failed while downloading shard {done + 1}/{n}: {failed_name}")
            saved = list(self._download_completed_paths)
            if saved:
                lines.append("")
                lines.append(
                    f"{len(saved)} shard(s) were saved locally before the failure."
                )
                preview = "\n".join(f"  • {Path(p).name}" for p in saved[:6])
                if preview:
                    lines.append(preview)
                if len(saved) > 6:
                    lines.append(f"  • … and {len(saved) - 6} more")
                if self._download_queue_paths:
                    first_path = self._download_queue_paths[0][0]
                    local_first = Path(get_llm_models_dir()) / Path(first_path).name
                    if local_first.is_file() or saved:
                        probe = str(local_first if local_first.is_file() else saved[0])
                        missing = missing_gguf_shards(probe)
                        if missing:
                            lines.append("")
                            lines.append(
                                "This model cannot load until every shard is present. "
                                "Retry the download to fetch the missing parts."
                            )
                            miss_preview = "\n".join(f"  • {name}" for name in missing[:6])
                            if miss_preview:
                                lines.append("Missing:")
                                lines.append(miss_preview)
                            if len(missing) > 6:
                                lines.append(f"  • … and {len(missing) - 6} more")
        if info.technical_detail and info.technical_detail not in "\n".join(lines):
            lines.append("")
            lines.append(f"({info.technical_detail})")
        return "\n".join(lines)

    def _retry_download_from_current_shard(self) -> None:
        if not self._download_queue_paths or not self._current_repo_id.strip():
            self._start_download()
            return
        self._pending_download_retry = False
        repo = self._current_repo_id.strip()
        self.download_progress.setValue(0)
        self.download_progress.show()
        self._set_download_button_cancel_mode(True)
        if self._download_completed_bytes > 0 and self._download_total_bytes > 0:
            pct = int(self._download_completed_bytes * 100 / self._download_total_bytes)
            self.download_progress.setValue(min(99, max(0, pct)))
        self._start_next_shard_download(repo)

    def _show_error(self, title: str, message: str) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(self.window(), title, message, is_dark=is_dark)
        dlg.exec()

    def refresh_after_theme_toggle(self) -> None:
        """Keep Hub chrome aligned with global light/dark (see MainWindow._toggle_theme)."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_hub_muted_labels(is_dark)
        self._apply_hub_metadata_styles(is_dark)
        self.refresh_button_themes(is_dark)
        self._apply_hub_list_surface(is_dark)
        self._apply_hub_file_combo_popup_theme(is_dark)
        self._apply_hub_combo_chevron(is_dark)
        if hasattr(self, "_hub_section_icon_label") and self._hub_section_icon_label:
            name = self._hub_section_icon_label.property("icon_name")
            if name:
                theme = self._theme(is_dark)
                c = theme.accent if theme.is_dark else theme.text_secondary
                self._hub_section_icon_label.setPixmap(
                    qta.icon(str(name), color=c).pixmap(QSize(18, 18))
                )
        self._update_hub_row_colors()
        self._refresh_hub_row_heights()
        current = self.hub_model_list.currentItem() if hasattr(self, "hub_model_list") else None
        self._apply_detail_branding(dict(current.data(HUB_ROW_BRANDING_ROLE) or {}) if current else None)
        self._apply_hub_connectivity_banner(is_dark)
        if self._last_readme_markdown:
            self._render_readme_with_fallback(is_dark)
        self._update_quant_rationale_label()
