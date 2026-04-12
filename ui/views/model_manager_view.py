"""Model Manager: Hub "app store" browser, README, quant selection, and downloads."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("Qube.ModelManager")

import qtawesome as qta
from PyQt6.QtGui import QColor, QFont, QPalette, QTextDocument
from PyQt6.QtCore import QEvent, Qt, QThread, QSize, QTimer, pyqtSignal
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
    QSplitter,
    QTextBrowser,
    QSizePolicy,
)

from core.app_settings import (
    get_llm_models_dir,
    set_internal_model_path,
)
from core.richtext_styles import markdown_document_stylesheet
from ui.components.prestige_dialog import PrestigeDialog
from workers.hf_model_search_worker import HfModelSearchWorker
from workers.hf_readme_worker import HfReadmeWorker
from workers.hf_repo_files_worker import HfRepoFilesWorker
from workers.model_download_worker import HuggingFaceGgufDownloadWorker

# Curated GGUF collections — safe starting points for new users.
QUBE_VERIFIED_MODELS: list[dict[str, str]] = [
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
        self._download_ui_cancel_mode = False
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
        self.refresh_button_themes(is_dark)
        self._apply_hub_file_combo_popup_theme(is_dark)
        self._update_hub_row_colors()
        QTimer.singleShot(0, self._refresh_hub_row_heights)

    def showEvent(self, event) -> None:
        super().showEvent(event)
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

        for attr in ("_search_worker", "_readme_worker", "_list_worker"):
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
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(16)

        title = QLabel("MODEL MANAGER")
        title.setObjectName("ViewTitle")
        main_layout.addWidget(title)

        # --- Hub "app store" split ---
        main_layout.addWidget(self._section_header("fa5s.th-large", "HUGGING FACE — BROWSE & DOWNLOAD"))
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)

        # Left: same sidebar shell as Conversations / Library (QSS + HistoryRowWidget pattern)
        left = QFrame()
        left.setFixedWidth(280)
        left.setObjectName("ModelManagerSidebar")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(15, 20, 15, 20)
        left_l.setSpacing(15)

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
        # overflow and list items bleeding under the splitter / right panel.
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
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 0, 0, 0)
        right_l.setSpacing(10)

        self.detail_title = QLabel("Select a model")
        self.detail_title.setWordWrap(True)
        f = self.detail_title.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        self.detail_title.setFont(f)

        self.detail_subtitle = QLabel("")
        self.detail_subtitle.setWordWrap(True)

        self.readme_browser = QTextBrowser()
        self.readme_browser.setMinimumHeight(220)
        self.readme_browser.setOpenExternalLinks(True)
        self.readme_browser.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)

        q_lab = QLabel("Quantization (.gguf file)")
        q_lab.setProperty("class", "ToolsPaneControl")

        files_row = QHBoxLayout()
        self.hf_file_combo = HubFileComboBox(self)
        self.hf_file_combo.setObjectName("HubFileComboBox")
        self.hf_file_combo.setMinimumWidth(200)
        # Popup QListView is reparented to a separate window; style it by objectName + palette
        # (same pattern as MainWindow._apply_menu_theme / PrestigeMenuList).
        _hub_combo_view = self.hf_file_combo.view()
        _hub_combo_view.setObjectName("HubFileComboDropdownView")
        _hub_combo_view.setAutoFillBackground(True)
        _vp = _hub_combo_view.viewport()
        if _vp is not None:
            _vp.setAutoFillBackground(True)
        files_row.addWidget(self.hf_file_combo, stretch=1)
        self.download_btn = QPushButton("Download")
        self.download_btn.setIcon(qta.icon("fa5s.download", color="#89b4fa"))
        self.download_btn.clicked.connect(self._start_download)
        files_row.addWidget(self.download_btn)

        self.download_status = QLabel("")
        self.download_status.setWordWrap(True)

        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setTextVisible(True)

        right_l.addWidget(self.detail_title)
        right_l.addWidget(self.detail_subtitle)
        right_l.addWidget(self.readme_browser, stretch=1)
        right_l.addWidget(q_lab)
        right_l.addLayout(files_row)
        right_l.addWidget(self.download_status)
        right_l.addWidget(self.download_progress)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 720])

        main_layout.addWidget(splitter, stretch=1)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._run_hub_search)

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
        icon_color = "#8b5cf6" if is_dark else "#1e293b"
        hover_bg = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.05)"
        if hasattr(self, "download_btn") and not getattr(self, "_download_ui_cancel_mode", False):
            self.download_btn.setIcon(qta.icon("fa5s.download", color=icon_color))
            self.download_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; border-radius: 6px; padding: 8px 12px; }}
                QPushButton:hover {{ background-color: {hover_bg}; }}
            """)

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
        QTimer.singleShot(0, self._refresh_hub_row_heights)

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
        self.download_status.setText("")

        self._retire_hf_thread(self._readme_worker)
        self._readme_worker = None
        self._retire_hf_thread(self._list_worker)
        self._list_worker = None

        self._readme_worker = HfReadmeWorker(str(repo))
        self._readme_worker.finished_ok.connect(
            lambda r, t, s=seq: self._apply_readme_if_current(r, t, s)
        )
        self._readme_worker.failed.connect(
            lambda r, err, s=seq: self._apply_readme_failed_if_current(r, err, s)
        )
        self._readme_worker.start()

        self._start_list_worker_for_repo(str(repo), seq)

    def _clear_detail_pane(self) -> None:
        self._retire_hf_thread(self._readme_worker)
        self._readme_worker = None
        self._retire_hf_thread(self._list_worker)
        self._list_worker = None
        self._current_repo_id = ""
        self._last_readme_markdown = None
        self.detail_title.setText("Select a model")
        self.detail_subtitle.setText("")
        self.readme_browser.clear()
        self.hf_file_combo.blockSignals(True)
        self.hf_file_combo.clear()
        self.hf_file_combo.addItem("-- Select a model from the list --")
        self.hf_file_combo.blockSignals(False)

    def _apply_readme_if_current(self, repo: str, text: str, seq: int) -> None:
        if seq != self._detail_seq:
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._last_readme_markdown = text
        doc = self.readme_browser.document()
        doc.setDefaultFont(self.readme_browser.font())
        doc.setDefaultStyleSheet(markdown_document_stylesheet(is_dark))
        try:
            self.readme_browser.setMarkdown(text)
        except Exception:
            self._last_readme_markdown = None
            self.readme_browser.setPlainText(text)

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
        self.download_status.setText("Fetching .gguf file list…")

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

    def _on_hf_list_finished(self, paths: list, seq: int) -> None:
        if seq != self._detail_seq:
            return
        self.hf_file_combo.blockSignals(True)
        self.hf_file_combo.clear()
        if not paths:
            self.hf_file_combo.addItem("(No .gguf files in this repository)")
            self.download_status.setText("No .gguf files found for this repo.")
        else:
            self.hf_file_combo.addItem("-- Select a .gguf file --")
            for p in paths:
                self.hf_file_combo.addItem(p)
            self.hf_file_combo.setCurrentIndex(0)
            self.download_status.setText(
                f"{len(paths)} file(s) available. Choose a quantization, then Download."
            )
        self.hf_file_combo.blockSignals(False)

    def _on_hf_list_failed(self, msg: str, seq: int) -> None:
        if seq != self._detail_seq:
            return
        self.hf_file_combo.blockSignals(True)
        self.hf_file_combo.clear()
        self.hf_file_combo.addItem("-- Could not list files --")
        self.hf_file_combo.blockSignals(False)
        self.download_status.setText("")
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
            self.download_btn.setText("Download")
            self.download_btn.setIcon(qta.icon("fa5s.download", color="#89b4fa"))
            self.download_btn.setStyleSheet("")
            self.download_btn.clicked.connect(self._start_download)

    def _restore_download_idle_ui(self) -> None:
        self._set_download_button_cancel_mode(False)
        self.download_btn.setEnabled(True)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self.refresh_button_themes(is_dark)
        self._apply_hub_file_combo_popup_theme(is_dark)

    def _cancel_download(self) -> None:
        if self._download_worker and self._download_worker.isRunning():
            self.download_status.setText("Cancelling…")
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
        self.download_status.setText("Starting…")
        self._set_download_button_cancel_mode(True)

        self._retire_hf_thread(self._download_worker)
        self._download_worker = None
        self._download_worker = HuggingFaceGgufDownloadWorker(
            repo, fname, get_llm_models_dir()
        )
        self._download_worker.progress_pct.connect(self.download_progress.setValue)
        self._download_worker.status_message.connect(self.download_status.setText)
        self._download_worker.finished_ok.connect(self._on_download_finished)
        self._download_worker.failed.connect(self._on_download_failed)
        self._download_worker.insufficient_space_error.connect(
            self._on_insufficient_space
        )
        self._download_worker.download_cancelled.connect(self._on_download_cancelled)
        self._download_worker.start()

    def _on_download_finished(self, path: str) -> None:
        self._restore_download_idle_ui()
        self.download_status.setText(f"Saved: {os.path.basename(path)}")
        set_internal_model_path(path)
        if self._llm:
            self._llm.refresh_native_model_from_settings()
        self.native_library_changed.emit()

    def _on_download_failed(self, msg: str) -> None:
        self._restore_download_idle_ui()
        self.download_progress.setValue(0)
        self.download_status.setText("")
        self._show_error("Download failed", msg)

    def _on_insufficient_space(self, required: int, available: int) -> None:
        self._restore_download_idle_ui()
        self.download_progress.setValue(0)
        self.download_status.setText("")
        self._show_error(
            "Not enough disk space",
            f"This download needs about {self._fmt_bytes(required)} free on the destination "
            f"drive (including a 500 MB safety margin). "
            f"You have about {self._fmt_bytes(available)} available.",
        )

    def _on_download_cancelled(self) -> None:
        self._restore_download_idle_ui()
        self.download_progress.setValue(0)
        self.download_status.setText("Download cancelled.")

    def _show_error(self, title: str, message: str) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(self.window(), title, message, is_dark=is_dark)
        dlg.exec()

    def refresh_after_theme_toggle(self) -> None:
        """Keep Hub chrome aligned with global light/dark (see MainWindow._toggle_theme)."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_hub_muted_labels(is_dark)
        self.refresh_button_themes(is_dark)
        self._apply_hub_file_combo_popup_theme(is_dark)
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
            doc = self.readme_browser.document()
            doc.setDefaultFont(self.readme_browser.font())
            doc.setDefaultStyleSheet(markdown_document_stylesheet(is_dark))
            try:
                self.readme_browser.setMarkdown(self._last_readme_markdown)
            except Exception:
                pass
