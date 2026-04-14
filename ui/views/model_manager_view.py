"""Model Manager: Hub "app store" browser, README, quant selection, and downloads."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("Qube.ModelManager")

import qtawesome as qta
from PyQt6.QtGui import QColor, QFont, QPalette, QTextDocument, QPainter
from PyQt6.QtCore import QEvent, Qt, QThread, QSize, QTimer, QUrl, QRect, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
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
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from core.app_settings import (
    get_llm_models_dir,
    set_internal_model_path,
)
from core.gpu_layers_cap import detect_gpu_vram_bytes
from core.hub_readme_html import hf_readme_markdown_to_safe_html, strip_hub_readme_preamble
from core.richtext_styles import markdown_document_stylesheet
from ui.components.prestige_dialog import PrestigeDialog
from workers.hf_model_search_worker import HfModelSearchWorker
from workers.hf_model_meta_worker import HfModelMetaWorker
from workers.hf_readme_worker import HfReadmeWorker
from workers.hf_repo_files_worker import HfRepoFilesWorker
from workers.model_download_worker import HuggingFaceGgufDownloadWorker

# Extra display data on Hub .gguf combo rows (file size, right-aligned in popup).
HUB_FILE_COMBO_SIZE_ROLE = int(Qt.ItemDataRole.UserRole) + 42
HUB_FILE_COMBO_BYTES_ROLE = int(Qt.ItemDataRole.UserRole) + 43


def _model_manager_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


# Curated GGUF collections — safe starting points for new users.
QUBE_VERIFIED_MODELS: list[dict[str, str]] = [
    {
        "repo_id": "google/gemma-4-26B-A4B",
        "title": "Gemma 4 26B A4B",
    },
    {
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "title": "Llama 3.1 8B Instruct",
    },
    {
        "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "title": "Mistral 7B Instruct",
    },
    {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "title": "Phi-3 Mini 4K Instruct",
    },
    {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "title": "Qwen 2.5 7B Instruct",
    },
]


def _hub_file_combo_list_qss(is_dark: bool) -> str:
    """Widget-local QSS for the combo's QAbstractItemView (app QSS misses detached popups on many styles)."""
    if is_dark:
        return """
            QAbstractItemView {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: none;
                outline: none;
            }
            QAbstractItemView::item {
                min-height: 32px;
                padding: 8px 12px;
                color: #cdd6f4;
            }
            QAbstractItemView::item:selected {
                background-color: rgba(255, 255, 255, 0.1);
                color: #ffffff;
            }
            QAbstractItemView::item:hover {
                background-color: #313244;
                color: #cdd6f4;
            }
        """
    return """
        QAbstractItemView {
            background-color: #ffffff;
            color: #1e293b;
            border: none;
            outline: none;
        }
        QAbstractItemView::item {
            min-height: 32px;
            padding: 8px 12px;
            color: #1e293b;
        }
        QAbstractItemView::item:selected {
            background-color: #f1f5f9;
            color: #0f172a;
        }
        QAbstractItemView::item:hover {
            background-color: #f1f5f9;
            color: #0f172a;
        }
    """


def _hub_file_combo_viewport_qss(is_dark: bool) -> str:
    return (
        "background-color: #1e1e2e;"
        if is_dark
        else "background-color: #ffffff;"
    )


class HubFileComboDelegate(QStyledItemDelegate):
    """Popup rows: format chip, model/quant chips, size, and chevron."""

    _GAP = 10
    _SIZE_MAX = 160

    @staticmethod
    def _infer_quant_label(path_s: str) -> str:
        name = Path(path_s).name
        stem = name[:-5] if name.lower().endswith(".gguf") else name
        tokens = [t for t in stem.replace("_", "-").split("-") if t]
        for t in reversed(tokens):
            u = t.upper()
            if u.startswith(("Q", "IQ")) and any(ch.isdigit() for ch in u):
                return u
        return "AUTO"

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        raw = index.data(int(HUB_FILE_COMBO_SIZE_ROLE))
        if raw is None:
            super().paint(painter, option, index)
            return
        size_label = str(raw).strip()
        if not size_label:
            super().paint(painter, option, index)
            return

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
        quant = self._infer_quant_label(path_s)
        quant_w = max(48, fm.horizontalAdvance(quant) + 20)
        model_r = QRect(left, rect.top(), max(20, size_r.left() - left - quant_w - 16), rect.height())
        model_name = Path(path_s).name
        mono_font = painter.font()
        mono_font.setFamilies(["Consolas", "Monospace"])
        painter.setFont(mono_font)
        elided = fm.elidedText(model_name, Qt.TextElideMode.ElideMiddle, model_r.width())
        painter.drawText(model_r, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), elided)
        left = model_r.right() + 8
        draw_chip(quant, 20, QPalette.ColorRole.Mid)
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


class ModelManagerView(QWidget):
    """Hub browser: search, README, file list, and GGUF downloads."""

    native_library_changed = pyqtSignal()

    _HUB_LIST_MAX_LINES = 3

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
        self._download_ui_cancel_mode = False
        self._download_ui_load_mode = False
        self._search_seq = 0
        self._detail_seq = 0
        self._current_repo_id = ""
        self._last_readme_markdown: str | None = None
        # Strong refs to QThread instances that are still running after we replace them — never
        # drop the last reference while isRunning() or Qt aborts with "Destroyed while still running".
        self._retired_hf_threads: list[QThread] = []

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
        self._update_hub_row_colors()
        QTimer.singleShot(0, self._refresh_hub_row_heights)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_hub_list_surface(is_dark)
        self._apply_hub_metadata_styles(is_dark)
        self._apply_hub_combo_chevron(is_dark)
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
        fg = "#ffffff" if item.isSelected() else ("#cdd6f4" if is_dark else "#1e293b")
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

        for attr in ("_search_worker", "_readme_worker", "_list_worker", "_meta_worker"):
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
        """Uniform row height (N lines via QTextDocument, same as elision) + elide overflow."""
        lay = row.layout()
        if not lay:
            return
        m = lay.contentsMargins()
        vw = self._hub_viewport_width_for_rows()
        lbl = row.findChild(QLabel, "HubModelRowTitle")
        if not lbl:
            item.setSizeHint(QSize(vw, 52))
            return

        row.setFixedWidth(vw)
        row.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        text_w = max(1, vw - m.left() - m.right())
        self._style_hub_title_label(lbl, item)
        lbl.ensurePolished()
        font = lbl.font()
        content_h = self._hub_uniform_label_content_height(font, text_w)

        title = item.data(Qt.ItemDataRole.UserRole + 1)
        repo = item.data(Qt.ItemDataRole.UserRole)
        full = f"{title or ''}\n{repo or ''}"

        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.TextFormat.PlainText)
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        lbl.setMinimumWidth(0)
        lbl.setMaximumWidth(text_w)
        lbl.setFixedWidth(text_w)
        display = self._hub_elide_plain_text_for_height(full, font, text_w, content_h)
        lbl.setText(display)
        lbl.setFixedHeight(int(content_h))

        total_h = int(m.top() + m.bottom() + content_h)
        item.setSizeHint(QSize(vw, total_h))
        lbl.updateGeometry()
        row.updateGeometry()

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
        left.setFixedWidth(280)
        left.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        left.setObjectName("ModelManagerSidebar")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(15, 20, 15, 20)
        left_l.setSpacing(15)
        self.hub_sidebar = left

        title = QLabel("MODEL MANAGER")
        title.setObjectName("ViewTitle")
        left_l.addWidget(title)
        left_l.addWidget(self._section_header("fa5s.th-large", "HUGGING FACE REPOSITORIES"))

        self.hub_search_edit = QLineEdit()
        self.hub_search_edit.setObjectName("HubModelSearchBar")
        self.hub_search_edit.setPlaceholderText("Search GGUF models on the Hub…")
        self.hub_search_edit.textChanged.connect(self._schedule_hub_search)
        left_l.addWidget(self.hub_search_edit)

        self.hub_list_hint = QLabel("Qube Verified — curated GGUF models")
        self.hub_list_hint.setWordWrap(True)
        left_l.addWidget(self.hub_list_hint)

        self.hub_model_list = QListWidget()
        self.hub_model_list.setObjectName("ModelHubList")
        # Parent frame is 280px with horizontal margins; min width 280 here forced horizontal
        # overflow and list items bleeding under the right panel.
        self.hub_model_list.setMinimumWidth(0)
        self.hub_model_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.hub_model_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.hub_model_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.hub_model_list.currentItemChanged.connect(self._on_hub_selection_changed)
        self.hub_model_list.itemSelectionChanged.connect(self._update_hub_row_colors)
        left_l.addWidget(self.hub_model_list, stretch=1)
        self.hub_model_list.viewport().installEventFilter(self)

        # Right: detail
        right = QWidget()
        right.setMaximumWidth(900)
        right_l = QVBoxLayout(right)
        # Match transcript-like vertical start while keeping bottom breathing room.
        right_l.setContentsMargins(8, 75, 0, 40)
        right_l.setSpacing(10)

        self.detail_title = QLabel("Select a model")
        self.detail_title.setWordWrap(True)
        f = self.detail_title.font()
        f.setBold(True)
        f.setPointSize(18)
        self.detail_title.setFont(f)

        self.detail_subtitle = QLabel("", parent=right)
        self.detail_subtitle.setWordWrap(True)
        self.detail_subtitle.hide()

        # Metadata panel (chip-heavy layout; actual colors come from QSS semantic classes).
        self.meta_panel = QFrame(parent=right)
        self.meta_panel.setProperty("class", "MetaPanelCard")
        meta_panel_l = QVBoxLayout(self.meta_panel)
        meta_panel_l.setContentsMargins(12, 12, 12, 12)
        meta_panel_l.setSpacing(8)

        self.meta_row_1 = QWidget(parent=self.meta_panel)
        meta_row_1_l = QHBoxLayout(self.meta_row_1)
        meta_row_1_l.setContentsMargins(0, 0, 0, 0)
        meta_row_1_l.setSpacing(10)

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
            meta_row_1_l.addWidget(w)
        meta_row_1_l.addStretch(1)

        self.meta_row_2 = QWidget(parent=self.meta_panel)
        meta_row_2_l = QHBoxLayout(self.meta_row_2)
        meta_row_2_l.setContentsMargins(0, 0, 0, 0)
        meta_row_2_l.setSpacing(8)
        self.meta_caps_title_lbl = QLabel("Capabilities:")
        self.meta_caps_title_lbl.setProperty("class", "MetaLabel")
        self.meta_caps_wrap = QWidget(parent=self.meta_row_2)
        self.meta_caps_wrap_l = QHBoxLayout(self.meta_caps_wrap)
        self.meta_caps_wrap_l.setContentsMargins(0, 0, 0, 0)
        self.meta_caps_wrap_l.setSpacing(6)
        meta_row_2_l.addWidget(self.meta_caps_title_lbl)
        meta_row_2_l.addWidget(self.meta_caps_wrap, stretch=1)

        self.meta_hint_lbl = QLabel("", parent=self.meta_panel)
        self.meta_hint_lbl.setWordWrap(True)
        self.meta_hint_lbl.hide()

        meta_panel_l.addWidget(self.meta_row_1)
        meta_panel_l.addWidget(self.meta_row_2)
        meta_panel_l.addWidget(self.meta_hint_lbl)

        self.readme_browser = QTextBrowser()
        self.readme_browser.setMinimumHeight(220)
        self.readme_browser.setOpenExternalLinks(True)
        self.readme_browser.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)

        # Download options card
        self.download_options_card = QFrame(parent=right)
        self.download_options_card.setProperty("class", "DownloadOptionsCard")
        dl_card_l = QVBoxLayout(self.download_options_card)
        dl_card_l.setContentsMargins(12, 12, 12, 12)
        dl_card_l.setSpacing(10)

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

        files_row = QHBoxLayout()
        files_row.setContentsMargins(0, 0, 0, 0)
        files_row.setSpacing(8)
        self.hf_file_combo = HubFileComboBox(self)
        self.hf_file_combo.setObjectName("HubFileComboBox")
        self.hf_file_combo.setMinimumWidth(200)
        self.hf_file_combo.currentIndexChanged.connect(self._update_gpu_fit_status)
        self.hf_file_combo.currentIndexChanged.connect(self._on_hf_file_combo_changed)
        # Popup QListView is reparented to a separate window; style it by objectName + palette
        # (same pattern as MainWindow._apply_menu_theme / PrestigeMenuList).
        _hub_combo_view = self.hf_file_combo.view()
        _hub_combo_view.setObjectName("HubFileComboDropdownView")
        self._hub_file_combo_delegate = HubFileComboDelegate(_hub_combo_view)
        _hub_combo_view.setItemDelegate(self._hub_file_combo_delegate)
        _hub_combo_view.setMinimumWidth(420)
        _hub_combo_view.setAutoFillBackground(True)
        _vp = _hub_combo_view.viewport()
        if _vp is not None:
            _vp.setAutoFillBackground(True)
        files_row.addWidget(self.hf_file_combo, stretch=1)
        self.download_btn = QPushButton("Download")
        self.download_btn.setProperty("class", "PrimaryActionButton")
        self.download_btn.setIcon(qta.icon("fa5s.download", color="#89b4fa"))
        self.download_btn.clicked.connect(self._start_download)

        self.download_status = QLabel("")
        self.download_status.setWordWrap(True)
        self.download_status.setVisible(False)

        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setTextVisible(True)
        _dp_pol = self.download_progress.sizePolicy()
        _dp_pol.setRetainSizeWhenHidden(True)
        self.download_progress.setSizePolicy(_dp_pol)
        self.download_progress.hide()

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        self.system_chip_lbl = QLabel("System: --")
        self.system_chip_lbl.setProperty("class", "Chip outlined")
        self._set_system_match_style("unknown")
        action_row.addWidget(self.system_chip_lbl, stretch=0)
        action_row.addStretch(1)
        action_row.addWidget(self.download_btn, stretch=0)

        dl_card_l.addWidget(dl_header_row)
        dl_card_l.addWidget(q_lab)
        dl_card_l.addLayout(files_row)
        dl_card_l.addLayout(action_row)
        dl_card_l.addWidget(self.download_status)
        dl_card_l.addWidget(self.download_progress)

        right_l.addWidget(self.detail_title)
        right_l.addWidget(self.meta_panel)
        right_l.addWidget(self.download_options_card)
        right_l.addWidget(self.readme_browser, stretch=1)

        right_host = QWidget()
        right_host_l = QHBoxLayout(right_host)
        # Keep a fixed gap from the sidebar while preserving left pinning behavior on resize.
        right_host_l.setContentsMargins(10, 0, 0, 0)
        right_host_l.setSpacing(0)
        right_host_l.addWidget(right, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        right_host_l.addStretch(1)

        hub_h.addWidget(left)
        hub_h.addWidget(right_host, stretch=1)

        main_layout.addWidget(hub_container, stretch=1)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._run_hub_search)

    def _meta_style(self, label: QLabel, *, is_dark: bool, strong: bool = False) -> None:
        fg = "#cdd6f4" if is_dark else "#1e293b"
        muted = "#94a3b8" if is_dark else "#64748b"
        color = fg if strong else muted
        weight = 600 if strong else 500
        label.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: {weight}; background: transparent;"
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
        hint_color = "#6c7086" if is_dark else "#64748b"
        self.meta_hint_lbl.setStyleSheet(
            f"color: {hint_color}; font-size: 11px; font-weight: 500; background: transparent;"
        )
        self._set_system_match_style("unknown")
        self._refresh_download_options_header_icon()

    def _refresh_download_options_header_icon(self) -> None:
        if not hasattr(self, "_download_options_icon_label"):
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        icon_color = "#8b5cf6" if is_dark else "#4c4f69"
        self._download_options_icon_label.setPixmap(
            qta.icon("fa5s.box-open", color=icon_color).pixmap(QSize(14, 14))
        )

    def _capability_icon_name(self, cap: str) -> str:
        c = cap.lower()
        if "vision" in c:
            return "fa5s.eye"
        if "tool" in c:
            return "fa5s.wrench"
        if "reason" in c:
            return "fa5s.brain"
        if "code" in c:
            return "fa5s.code"
        if "multi" in c:
            return "fa5s.globe"
        return "fa5s.star"

    def _render_capability_chips(self, caps: list[str]) -> None:
        if not hasattr(self, "meta_caps_wrap_l"):
            return
        while self.meta_caps_wrap_l.count():
            it = self.meta_caps_wrap_l.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        if not caps:
            empty_chip = QLabel("Unknown")
            empty_chip.setProperty("class", "Chip muted")
            self.meta_caps_wrap_l.addWidget(empty_chip)
            self.meta_caps_wrap_l.addStretch(1)
            return
        for cap in caps:
            chip = QFrame()
            chip.setProperty("class", "Chip capability")
            chip_l = QHBoxLayout(chip)
            chip_l.setContentsMargins(8, 3, 8, 3)
            chip_l.setSpacing(6)
            icon_lbl = QLabel()
            icon_lbl.setProperty("class", "ChipIcon")
            icon_lbl.setPixmap(qta.icon(self._capability_icon_name(cap)).pixmap(QSize(11, 11)))
            txt_lbl = QLabel(cap)
            txt_lbl.setProperty("class", "ChipLabel")
            chip_l.addWidget(icon_lbl)
            chip_l.addWidget(txt_lbl)
            self.meta_caps_wrap_l.addWidget(chip)
        self.meta_caps_wrap_l.addStretch(1)

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
        if q_bytes <= 0:
            if hasattr(self, "system_chip_lbl"):
                self.system_chip_lbl.setText("System: Unknown")
                self._set_system_match_style("unknown")
            return
        vram_b = int(detect_gpu_vram_bytes() or 0)
        if vram_b <= 0:
            if hasattr(self, "system_chip_lbl"):
                self.system_chip_lbl.setText(f"System: Quant {self._fmt_gib(q_bytes)}")
                self._set_system_match_style("unknown")
            return
        if q_bytes <= vram_b:
            status = f"Fits ({self._fmt_gib(q_bytes)} / {self._fmt_gib(vram_b)})"
            if hasattr(self, "system_chip_lbl"):
                self.system_chip_lbl.setText(f"System: {status}")
                self._set_system_match_style("fit")
        else:
            status = f"Does not fit ({self._fmt_gib(q_bytes)} / {self._fmt_gib(vram_b)})"
            if hasattr(self, "system_chip_lbl"):
                self.system_chip_lbl.setText(f"System: {status}")
                self._set_system_match_style("no_fit")

    def _on_hf_file_combo_changed(self, _index: int) -> None:
        self._update_download_button_label()
        self._sync_download_action_state()

    def _update_download_button_label(self) -> None:
        if not hasattr(self, "download_btn") or not hasattr(self, "hf_file_combo"):
            return
        if getattr(self, "_download_ui_load_mode", False):
            return
        idx = self.hf_file_combo.currentIndex()
        if idx < 0:
            self.download_btn.setText("Download")
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

    def _set_download_progress_text(self, prefix: str = "Downloading model") -> None:
        if not hasattr(self, "download_progress"):
            return
        self.download_progress.setFormat(f"{prefix} (%p%)")

    def _on_download_progress_pct(self, pct: int) -> None:
        self.download_progress.setValue(int(pct))
        self._set_download_progress_text()

    def _on_download_status_message(self, msg: str) -> None:
        # Keep status label quiet during active download; progress text carries the state.
        # Preserve "Downloading model" text per UX request.
        if self._download_worker and self._download_worker.isRunning():
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
        return Path(get_llm_models_dir()) / Path(sel).name

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
        self.download_btn.setIcon(qta.icon("fa5s.play", color="#f8fafc"))
        self._apply_download_action_button_style(mode="load")
        self.download_btn.setEnabled(bool(enabled))
        self.download_btn.clicked.connect(self._load_selected_model)

    def _apply_download_action_button_style(self, mode: str) -> None:
        """Theme-safe button colors for Download/Load actions."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        if mode == "load":
            if is_dark:
                bg, fg, border = ("rgba(255, 255, 255, 0.08)", "#cdd6f4", "rgba(255, 255, 255, 0.16)")
            else:
                bg, fg, border = ("#ffffff", "#1e293b", "#cbd5e1")
        else:
            if is_dark:
                bg, fg, border = ("#8b5cf6", "#f8fafc", "#8b5cf6")
            else:
                bg, fg, border = ("#1e293b", "#f8fafc", "#1e293b")
        self.download_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {bg};
                color: {fg};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: 700;
            }}
            """
        )

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
            return
        self._set_download_button_download_mode()
        if not (self._download_worker and self._download_worker.isRunning()):
            self._set_download_status_text("")

    def _load_selected_model(self) -> None:
        p = self._selected_local_model_path()
        if p is None or not p.is_file():
            self._show_error("Model not found", "Selected file is not available locally.")
            self._sync_download_action_state()
            return
        set_internal_model_path(str(p))
        if self._llm:
            self._llm.refresh_native_model_from_settings()
        self.native_library_changed.emit()
        self._set_download_status_text(f"Loaded: {p.name}")
        self._set_download_button_load_mode(enabled=False)

    def _apply_hub_list_surface(self, is_dark: bool) -> None:
        """Match Conversations sidebar/list background palette on both themes."""
        bg = QColor("#232337" if is_dark else "#E9EFF5")
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

    def _apply_hub_combo_chevron(self, is_dark: bool) -> None:
        """QSS border triangles render as lines on some styles; use SVG via file URL."""
        if not hasattr(self, "hf_file_combo"):
            return
        name = (
            "hub_combo_chevron_dark.svg"
            if is_dark
            else "hub_combo_chevron_light.svg"
        )
        svg = _model_manager_project_root() / "assets" / "icons" / name
        if not svg.is_file():
            return
        url = QUrl.fromLocalFile(str(svg)).toString()
        self.hf_file_combo.setStyleSheet(
            "#HubFileComboBox::down-arrow { "
            f'image: url("{url}"); '
            "width: 12px; height: 12px; "
            "subcontrol-origin: padding; subcontrol-position: center; "
            "}"
        )

    def _apply_hub_file_combo_popup_theme(self, is_dark: bool) -> None:
        """Palette + widget-local QSS on the list view (matches MainWindow prestige menus)."""
        if not hasattr(self, "hf_file_combo"):
            return
        v = self.hf_file_combo.view()
        palette = QPalette()
        if is_dark:
            bg = QColor("#1e1e2e")
            fg = QColor("#cdd6f4")
            sel_bg = QColor("#313244")
            sel_fg = QColor("#cdd6f4")
        else:
            bg = QColor("#ffffff")
            fg = QColor("#1e293b")
            sel_bg = QColor("#f1f5f9")
            sel_fg = QColor("#0f172a")
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
            palette.setColor(role, bg)
        palette.setColor(QPalette.ColorRole.WindowText, fg)
        palette.setColor(QPalette.ColorRole.Text, fg)
        palette.setColor(QPalette.ColorRole.Highlight, sel_bg)
        palette.setColor(QPalette.ColorRole.HighlightedText, sel_fg)
        v.setPalette(palette)
        v.setStyleSheet(_hub_file_combo_list_qss(is_dark))
        vp = v.viewport()
        if vp is not None:
            vp.setPalette(palette)
            vp.setStyleSheet(_hub_file_combo_viewport_qss(is_dark))

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
        combo = self.hf_file_combo
        v = combo.view()
        if v is None:
            return
        bg = "#1e1e2e" if is_dark else "#ffffff"
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"
        main_win = self.window()

        outer = v.window()
        if outer is not None and outer is not main_win:
            outer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            outer.setStyleSheet(
                f"QWidget {{ background-color: {bg}; border: 1px solid {border}; border-radius: 8px; }}"
            )

        p = v.parentWidget()
        depth = 0
        while p is not None and p is not combo and depth < 10:
            p.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            if p is not outer:
                p.setStyleSheet(f"background-color: {bg}; border: none;")
            p = p.parentWidget()
            depth += 1

    def _apply_hub_muted_labels(self, is_dark: bool) -> None:
        """Secondary text — matches muted sidebar copy in Chat / Library."""
        muted = "#94a3b8" if is_dark else "#64748b"
        hint_style = f"color: {muted}; font-size: 11px;"
        sub_style = f"color: {muted}; font-size: 12px;"
        if hasattr(self, "hub_list_hint"):
            self.hub_list_hint.setStyleSheet(hint_style)
        if hasattr(self, "detail_subtitle"):
            self.detail_subtitle.setStyleSheet(sub_style)

    def _append_hub_model_row(self, title: str, repo_id: str) -> None:
        """One Hub row — same HistoryRowWidget / HistoryRowTitle pattern as Chat & Library."""
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, repo_id)
        item.setData(Qt.ItemDataRole.UserRole + 1, title)

        row = QWidget()
        row.setObjectName("HistoryRowWidget")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(15, 8, 10, 10)
        lay.setSpacing(8)

        lbl = QLabel()
        lbl.setObjectName("HubModelRowTitle")
        lbl.setWordWrap(True)
        # Title color + Inter 13px: _style_hub_title_label (same idea as Library history rows).
        lay.addWidget(lbl, stretch=0, alignment=Qt.AlignmentFlag.AlignTop)

        self.hub_model_list.addItem(item)
        self.hub_model_list.setItemWidget(item, row)
        self._apply_hub_row_size_hint(item, row)

    def _update_hub_row_colors(self) -> None:
        """Re-apply hub title fg for selection + theme, then re-layout row heights."""
        self.hub_model_list.viewport().update()
        self._refresh_hub_row_heights()

    def refresh_button_themes(self, is_dark: bool) -> None:
        """Icon accents for Hub actions — same pattern as LibraryView.refresh_button_themes."""
        filled_icon_color = "#f8fafc"
        if not hasattr(self, "download_btn"):
            return
        if getattr(self, "_download_ui_cancel_mode", False):
            return
        if getattr(self, "_download_ui_load_mode", False):
            self.download_btn.setIcon(qta.icon("fa5s.play", color=filled_icon_color))
            self._apply_download_action_button_style(mode="load")
            return
        self.download_btn.setIcon(qta.icon("fa5s.download", color=filled_icon_color))
        self._apply_download_action_button_style(mode="download")

    def _set_system_match_style(self, state: str) -> None:
        """Render System label as plain text (no chip card), colored by fit state."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        if state == "fit":
            color = "#10b981"
        elif state == "no_fit":
            color = "#f59e0b" if is_dark else "#b45309"
        else:
            color = "#94a3b8" if is_dark else "#64748b"
        if hasattr(self, "system_chip_lbl"):
            self.system_chip_lbl.setStyleSheet(
                f"background: transparent; border: none; padding: 0px; color: {color}; font-size: 12px; font-weight: 600;"
            )

    def _schedule_hub_search(self, _text: str = "") -> None:
        self._search_timer.stop()
        self._search_timer.start(500)

    def _populate_editors_picks(self) -> None:
        self.hub_model_list.blockSignals(True)
        self.hub_model_list.clear()
        for entry in QUBE_VERIFIED_MODELS:
            self._append_hub_model_row(entry["title"], entry["repo_id"])
        self.hub_model_list.blockSignals(False)
        self.hub_list_hint.setText("Qube Verified — curated GGUF models")
        self._update_hub_row_colors()
        self._select_first_hub_item()
        QTimer.singleShot(0, self._refresh_hub_row_heights)

    def _select_first_hub_item(self) -> None:
        if self.hub_model_list.count() > 0:
            self.hub_model_list.setCurrentRow(0)

    def _run_hub_search(self) -> None:
        q = self.hub_search_edit.text().strip()
        if not q:
            self._search_seq += 1
            self._populate_editors_picks()
            return

        self._search_seq += 1
        seq = self._search_seq
        self.hub_list_hint.setText("Searching Hugging Face (GGUF-tagged models)…")

        self._retire_hf_thread(self._search_worker)
        self._search_worker = HfModelSearchWorker(q, seq)
        self._search_worker.finished_ok.connect(self._apply_hub_search_results)
        self._search_worker.failed.connect(self._on_hub_search_failed)
        self._search_worker.start()

    def _apply_hub_search_results(self, models: list, seq: int) -> None:
        if seq != self._search_seq:
            return
        self.hub_model_list.blockSignals(True)
        self.hub_model_list.clear()
        for m in models:
            rid = m.get("repo_id", "")
            title = m.get("title", rid)
            if not rid:
                continue
            self._append_hub_model_row(title, rid)
        self.hub_model_list.blockSignals(False)
        self._update_hub_row_colors()
        n = self.hub_model_list.count()
        self.hub_list_hint.setText(
            f"{n} GGUF-related model(s) — refine your search if needed."
        )
        QTimer.singleShot(0, self._refresh_hub_row_heights)

    def _on_hub_search_failed(self, msg: str, seq: int) -> None:
        if seq != self._search_seq:
            return
        self.hub_list_hint.setText("Search failed — try different keywords.")
        self._show_error("Hub search failed", msg)

    def _on_hub_selection_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if not current:
            self._clear_detail_pane()
            return
        repo = current.data(Qt.ItemDataRole.UserRole)
        title = current.data(Qt.ItemDataRole.UserRole + 1) or repo
        if not repo:
            return
        self._detail_seq += 1
        seq = self._detail_seq
        self._current_repo_id = str(repo)
        self.detail_title.setText(str(title))
        self.detail_subtitle.setText(str(repo))
        self.readme_browser.clear()
        self.readme_browser.setPlainText("Loading README…")
        self._last_readme_markdown = None

        self.hf_file_combo.blockSignals(True)
        self.hf_file_combo.clear()
        self.hf_file_combo.addItem("Loading file list…")
        self.hf_file_combo.blockSignals(False)
        self._set_download_status_text("")
        if hasattr(self, "hub_quant_hint_lbl"):
            self.hub_quant_hint_lbl.setText("Loading available quantizations…")

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

    def _clear_detail_pane(self) -> None:
        self._retire_hf_thread(self._readme_worker)
        self._readme_worker = None
        self._retire_hf_thread(self._list_worker)
        self._list_worker = None
        self._retire_hf_thread(self._meta_worker)
        self._meta_worker = None
        self._current_repo_id = ""
        self._last_readme_markdown = None
        self.detail_title.setText("Select a model")
        self.detail_subtitle.setText("")
        self.readme_browser.clear()
        self.hf_file_combo.blockSignals(True)
        self.hf_file_combo.clear()
        self.hf_file_combo.addItem("-- Select a model from the list --")
        self.hf_file_combo.blockSignals(False)
        self._reset_hub_metadata_labels()
        if hasattr(self, "hub_quant_hint_lbl"):
            self.hub_quant_hint_lbl.setText("Select a model to view available quantizations.")
        self._update_download_button_label()
        self._update_gpu_fit_status()
        self._sync_download_action_state()

    def _apply_meta_if_current(self, repo: str, meta: dict, seq: int) -> None:
        if seq != self._detail_seq:
            return
        self._apply_hub_metadata(meta)

    def _apply_meta_failed_if_current(self, repo: str, err: str, seq: int) -> None:
        if seq != self._detail_seq:
            return
        self._reset_hub_metadata_labels()
        self._set_meta_hint(
            "Model metadata unavailable for this repository. Quantization and README are still available."
        )

    def _render_readme_with_fallback(self, is_dark: bool) -> None:
        """Python-Markdown + setHtml first (GFM tables/lists); then Qt setMarkdown; then plain text."""
        text = self._last_readme_markdown
        if not text:
            return
        prepared = strip_hub_readme_preamble(text)
        if not prepared:
            self.readme_browser.setHtml('<div class="hub-readme"></div>')
            return
        doc = self.readme_browser.document()
        doc.setDefaultFont(self.readme_browser.font())
        doc.setDefaultStyleSheet(markdown_document_stylesheet(is_dark))
        html = hf_readme_markdown_to_safe_html(text)
        if html is not None:
            self.readme_browser.setHtml(html)
            return
        try:
            self.readme_browser.setMarkdown(prepared)
        except Exception:
            self.readme_browser.setPlainText(prepared)

    def _apply_readme_if_current(self, repo: str, text: str, seq: int) -> None:
        if seq != self._detail_seq:
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._last_readme_markdown = text
        self._render_readme_with_fallback(is_dark)

    def _apply_readme_failed_if_current(self, repo: str, err: str, seq: int) -> None:
        if seq != self._detail_seq:
            return
        self._last_readme_markdown = None
        self.readme_browser.setPlainText(
            f"Could not load README for `{repo}`.\n\n{err}\n\n"
            "You can still pick a .gguf file below if the repo lists files on the Hub."
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
        # Smallest-to-largest by known size; unknown sizes are sent to the end.
        normalized.sort(
            key=lambda it: (
                it[1] is None,
                int(it[1]) if it[1] is not None else 0,
                str(it[0]).lower(),
            )
        )
        if not normalized:
            self.hf_file_combo.addItem("(No .gguf files in this repository)")
            if hasattr(self, "hub_quant_hint_lbl"):
                self.hub_quant_hint_lbl.setText("No .gguf files found for this repository.")
        else:
            self.hf_file_combo.addItem("-- Select a .gguf file --")
            for path, size_b in normalized:
                self.hf_file_combo.addItem(path)
                idx = self.hf_file_combo.count() - 1
                if size_b is not None:
                    self.hf_file_combo.setItemData(
                        idx,
                        self._fmt_bytes(size_b),
                        int(HUB_FILE_COMBO_SIZE_ROLE),
                    )
                    self.hf_file_combo.setItemData(
                        idx,
                        int(size_b),
                        int(HUB_FILE_COMBO_BYTES_ROLE),
                    )
            # Auto-select the smallest quantization entry.
            self.hf_file_combo.setCurrentIndex(1)
            if hasattr(self, "hub_quant_hint_lbl"):
                self.hub_quant_hint_lbl.setText(
                    f"{len(normalized)} file(s) available. Choose a quantization, then Download."
                )
        self.hf_file_combo.blockSignals(False)
        self._update_download_button_label()
        self._update_gpu_fit_status()
        self._sync_download_action_state()

    def _on_hf_list_failed(self, msg: str, seq: int) -> None:
        if seq != self._detail_seq:
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
        self._show_error("Could not list files", msg)

    def _section_header(self, icon_name: str, text: str) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        icon_color = "#8b5cf6" if is_dark else "#4c4f69"
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
        t = self.hf_file_combo.itemText(i).strip()
        if t.startswith("--") or t.startswith("("):
            return None
        if not t.lower().endswith(".gguf"):
            return None
        return t

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
            self.download_btn.setIcon(qta.icon("fa5s.times", color="#fef2f2"))
            self.download_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #dc2626;
                    border: 1px solid #b91c1c;
                    color: #fef2f2;
                    font-weight: bold;
                    border-radius: 6px;
                    padding: 8px 12px;
                }
                QPushButton:hover { background-color: #b91c1c; }
                """
            )
            self.download_btn.clicked.connect(self._cancel_download)
        else:
            self._set_download_button_download_mode()

    def _restore_download_idle_ui(self) -> None:
        self._download_ui_cancel_mode = False
        self.download_progress.hide()
        self.download_progress.setValue(0)
        self.download_progress.setFormat("(%p%)")
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
        fname = self._selected_hf_repo_file()
        if not repo:
            self._show_error("No model", "Select a model from the Hub list first.")
            return
        if not fname:
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
        self._restore_download_idle_ui()
        self._set_download_status_text(f"Saved: {os.path.basename(path)}")
        set_internal_model_path(path)
        # Do not auto-load after download; user can manually select/load later.
        self.native_library_changed.emit()
        self._sync_download_action_state()

    def _on_download_failed(self, msg: str) -> None:
        self._restore_download_idle_ui()
        self._set_download_status_text("")
        self._show_error("Download failed", msg)

    def _on_insufficient_space(self, required: int, available: int) -> None:
        self._restore_download_idle_ui()
        self._set_download_status_text("")
        self._show_error(
            "Not enough disk space",
            f"This download needs about {self._fmt_bytes(required)} free on the destination "
            f"drive (including a 500 MB safety margin). "
            f"You have about {self._fmt_bytes(available)} available.",
        )

    def _on_download_cancelled(self) -> None:
        self._restore_download_idle_ui()
        self._set_download_status_text("Download cancelled.")

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
                c = "#8b5cf6" if is_dark else "#4c4f69"
                self._hub_section_icon_label.setPixmap(
                    qta.icon(str(name), color=c).pixmap(QSize(18, 18))
                )
        self._update_hub_row_colors()
        self._refresh_hub_row_heights()
        if self._last_readme_markdown:
            self._render_readme_with_fallback(is_dark)
