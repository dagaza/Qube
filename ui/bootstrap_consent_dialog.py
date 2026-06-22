"""First-run bootstrap consent dialog (Task #46 / AB#46)."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPixmap, QShowEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.bootstrap_feasibility import (
    BootstrapBlockReason,
    BootstrapModelFeasibility,
    BootstrapSessionAssessment,
    assess_model_feasibility,
    build_session_assessment,
    can_proceed_with_selection,
    feasible_recommended_selection,
    format_shell_install_warning_message,
    models_blocked_for_session,
)
from core.bootstrap_hf_metadata import (
    BootstrapSizeSource,
    ResolvedBootstrapSize,
    format_bootstrap_size_tag_tooltip,
)
from core.bootstrap_manifest import (
    ADVANCED_ORDER,
    BOOTSTRAP_MODELS,
    BootstrapHintLevel,
    BootstrapModelId,
    MAIN_LLM_GROUP,
    RECOMMENDED_ORDER,
    SIDECAR_GROUP,
    bootstrap_tier_tag,
    default_selection,
    format_bootstrap_tier_tag_tooltip,
    format_byte_size,
    locked_recommended_ids,
    normalize_selection,
)
from core.bootstrap_selection import (
    budget_headroom_bytes,
    format_available_disk,
    preflight_download,
    required_bytes_for,
    selection_within_budget,
    total_selected_bytes,
    save_bootstrap_selection,
)
from workers.bootstrap_metadata_worker import BootstrapMetadataWorker
from workers.model_download_worker import SAFETY_BUFFER_BYTES
from ui.app_icon import apply_window_branding, finalize_window_branding
from ui.components.prestige_dialog import PrestigeDialog
from ui.splash_widget import resolve_splash_logo_path

# Estimated/Verified download-size chips — sizing logic stays; UI hidden for now.
_SHOW_BOOTSTRAP_SIZE_CHIPS = False


class _BootstrapModelScrollArea(QScrollArea):
    """Model list scroll area — do not inherit full content height as minimum."""

    def minimumSizeHint(self) -> QSize:
        width = super().minimumSizeHint().width()
        return QSize(width, 300)

    def sizeHint(self) -> QSize:
        width = super().sizeHint().width()
        return QSize(width, 420)


class _BootstrapModelRow(QFrame):
    """Model option row; any click inside the row toggles selection."""

    def __init__(
        self,
        checkbox: QCheckBox,
        *,
        locked: bool,
        object_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._checkbox = checkbox
        self._locked = locked
        self._disk_blocked = False
        self._style_object_name = object_name
        self.setObjectName(object_name)
        self._apply_row_cursor()

    def set_disk_blocked(self, blocked: bool) -> None:
        self._disk_blocked = blocked
        if blocked and not self._checkbox.isChecked():
            self.setObjectName("BootstrapModelRowDiskBlocked")
        else:
            self.setObjectName(self._style_object_name)
        self.style().unpolish(self)
        self.style().polish(self)
        self._apply_row_cursor()

    def _apply_row_cursor(self) -> None:
        if self._locked:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif self._disk_blocked and not self._checkbox.isChecked():
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    @staticmethod
    def _pass_clicks_to_row(widget: QWidget) -> None:
        widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    @staticmethod
    def _transparent_label_background(widget: QWidget) -> None:
        """Prevent global app QSS/palette from painting opaque label boxes on row chrome."""
        widget.setAutoFillBackground(False)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._locked or not self._checkbox.isEnabled():
            super().mouseReleaseEvent(event)
            return
        if self._disk_blocked and not self._checkbox.isChecked():
            event.accept()
            return
        self._checkbox.toggle()
        event.accept()


class _BootstrapRowChip(QLabel):
    """Small row chip (tier / size) with hover tooltip; click toggles the row checkbox."""

    def __init__(
        self,
        checkbox: QCheckBox,
        row: "_BootstrapModelRow",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._checkbox = checkbox
        self._row = row
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._row._locked or not self._checkbox.isEnabled():
            super().mouseReleaseEvent(event)
            return
        if self._row._disk_blocked and not self._checkbox.isChecked():
            event.accept()
            return
        self._checkbox.toggle()
        event.accept()


class _CollapsibleHeader(QFrame):
    """Clickable disclosure header row."""

    clicked = pyqtSignal()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _CollapsiblePanel(QFrame):
    """Compact header with expandable detail body."""

    def __init__(
        self,
        title: str,
        *,
        collapsed: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._summary = ""
        self._collapsed = collapsed
        self.setObjectName("BootstrapCollapsiblePanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _CollapsibleHeader()
        self._header.setObjectName("BootstrapCollapsibleHeader")
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.clicked.connect(self._toggle_collapsed)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(10, 10, 12, 10)
        header_layout.setSpacing(8)
        self._arrow = QLabel()
        self._arrow.setObjectName("BootstrapCollapsibleArrow")
        self._arrow.setFixedWidth(14)
        self._arrow.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._header_text = QLabel()
        self._header_text.setObjectName("BootstrapCollapsibleHeaderText")
        self._header_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        header_layout.addWidget(self._arrow)
        header_layout.addWidget(self._header_text, 1)
        layout.addWidget(self._header)

        self._summary_label = QLabel()
        self._summary_label.setObjectName("BootstrapCollapsibleSummary")
        self._summary_label.setWordWrap(True)
        self._summary_label.hide()
        layout.addWidget(self._summary_label)

        self._body = QFrame()
        self._body.setObjectName("BootstrapCollapsibleBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(12, 8, 12, 10)
        self._body_layout.setSpacing(6)
        layout.addWidget(self._body)

        self._body.setVisible(not collapsed)
        self._refresh_header()

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def set_summary(self, summary: str) -> None:
        self._summary = summary.strip()
        self._refresh_header()

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self._body.setVisible(not collapsed)
        self._refresh_header()

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    def _toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def _refresh_header(self) -> None:
        self._arrow.setText("\u25B6" if self._collapsed else "\u25BC")
        self._header_text.setText(self._title)
        if self._collapsed and self._summary:
            self._summary_label.setText(self._summary)
            self._summary_label.show()
        else:
            self._summary_label.hide()


class BootstrapConsentPanel(QWidget):
    """Recommended/advanced model selection — embeddable in splash or standalone dialog."""

    selection_confirmed = pyqtSignal(set)

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
        embedded: bool = False,
        split_embedded: bool = False,
    ) -> None:
        super().__init__(parent)
        self._embedded = embedded or split_embedded
        self._split_embedded = split_embedded
        self._advanced = False
        self._selected: set[BootstrapModelId] | None = None
        self._checkboxes: dict[BootstrapModelId, QCheckBox] = {}
        self._rows: dict[BootstrapModelId, _BootstrapModelRow] = {}
        self._feasibility_notes: dict[BootstrapModelId, QLabel] = {}
        self._tier_tags: dict[BootstrapModelId, QLabel] = {}
        self._size_tags: dict[BootstrapModelId, QLabel] = {}
        self._block_tags: dict[BootstrapModelId, QLabel] = {}
        self._assessment = self._initial_assessment()
        self._selection_state = self._feasible_recommended_state()
        self._user_touched_selection = False
        self._metadata_worker: BootstrapMetadataWorker | None = None

        self.setObjectName(
            "BootstrapConsentPanelSplit"
            if split_embedded
            else "BootstrapConsentPanelEmbedded"
            if embedded
            else "BootstrapConsentPanel"
        )
        self.setProperty("qube_tooltip_clip", True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(False)
        if split_embedded:
            self.setMinimumWidth(620)
        else:
            self.setMinimumSize(640, 680)

        root = QVBoxLayout(self)
        if self._split_embedded:
            root.setContentsMargins(20, 16, 24, 16)
        else:
            root.setContentsMargins(28, 20, 28, 20)
        root.setSpacing(10)

        self._logo = QLabel()
        self._logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = resolve_splash_logo_path()
        if logo_path is not None:
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                self._logo.setPixmap(
                    pix.scaledToWidth(72, Qt.TransformationMode.SmoothTransformation)
                )
        root.addWidget(self._logo)

        self._brand_title = QLabel("Qube")
        self._brand_title.setObjectName("BootstrapBrandTitle")
        self._brand_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._brand_title)

        if self._split_embedded:
            self._logo.hide()
            self._brand_title.hide()

        self._title = QLabel()
        self._title.setObjectName("BootstrapTitle")
        root.addWidget(self._title)

        self._intro = QLabel()
        self._intro.setWordWrap(True)
        self._intro.setObjectName("BootstrapIntro")
        root.addWidget(self._intro)

        self._legend = QLabel()
        self._legend.setWordWrap(True)
        self._legend.setObjectName("BootstrapLegend")
        root.addWidget(self._legend)

        self._details_panel = _CollapsiblePanel("System and download details", collapsed=True)
        details_layout = self._details_panel.body_layout()
        self._hardware_summary = QLabel(self._format_hardware_summary())
        self._hardware_summary.setObjectName("BootstrapHardwareSummary")
        self._hardware_summary.setWordWrap(True)
        self._hf_status = QLabel("Verifying download sizes with Hugging Face...")
        self._hf_status.setObjectName("BootstrapHfStatus")
        self._hf_status.setWordWrap(True)
        self._disk_summary = QLabel()
        self._disk_summary.setObjectName("BootstrapDiskSummary")
        self._disk_summary.setWordWrap(True)
        self._disk_notice = QLabel()
        self._disk_notice.setObjectName("BootstrapDiskNotice")
        self._disk_notice.setWordWrap(True)
        self._disk_notice.hide()
        details_layout.addWidget(self._hardware_summary)
        details_layout.addWidget(self._hf_status)
        details_layout.addWidget(self._disk_summary)
        details_layout.addWidget(self._disk_notice)
        root.addWidget(self._details_panel)

        root.addSpacing(8)

        bulk_bar = QFrame()
        bulk_bar.setObjectName("BootstrapBulkBar")
        bulk_layout = QHBoxLayout(bulk_bar)
        bulk_layout.setContentsMargins(10, 7, 10, 7)
        bulk_layout.setSpacing(8)
        bulk_caption = QLabel("Quick select")
        bulk_caption.setObjectName("BootstrapBulkCaption")
        self._select_all_btn = QPushButton("All")
        self._select_all_btn.setObjectName("BootstrapBulkPill")
        self._select_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._select_all_btn.clicked.connect(self._select_all_visible)
        self._deselect_all_btn = QPushButton("None")
        self._deselect_all_btn.setObjectName("BootstrapBulkPill")
        self._deselect_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._deselect_all_btn.clicked.connect(self._deselect_all_visible)
        bulk_layout.addWidget(bulk_caption)
        bulk_layout.addWidget(self._select_all_btn)
        bulk_layout.addWidget(self._deselect_all_btn)
        bulk_layout.addStretch()
        root.addWidget(bulk_bar)

        self._list_section = QWidget()
        self._list_section.setObjectName("BootstrapListSection")
        self._list_section.setAutoFillBackground(False)
        list_section_layout = QVBoxLayout(self._list_section)
        list_section_layout.setContentsMargins(0, 0, 0, 0)
        list_section_layout.setSpacing(6)

        self._scroll = _BootstrapModelScrollArea()
        self._scroll.setObjectName("BootstrapScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._scroll.viewport().setObjectName("BootstrapScrollViewport")
        self._scroll.viewport().setAutoFillBackground(False)
        self._scroll_host = QWidget()
        self._scroll_host.setObjectName("BootstrapScrollHost")
        self._scroll_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._scroll_host.setAutoFillBackground(False)
        self._scroll_host.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        self._scroll_layout = QVBoxLayout(self._scroll_host)
        self._scroll_layout.setContentsMargins(0, 0, 14, 0)
        self._scroll_layout.setSpacing(10)
        self._scroll.setWidget(self._scroll_host)
        self._scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        list_section_layout.addWidget(self._scroll, 1)

        self._total_label = QLabel()
        self._total_label.setObjectName("BootstrapTotalLabel")
        list_section_layout.addWidget(self._total_label, 0)

        root.addWidget(self._list_section, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self._back_btn = QPushButton("← Back")
        self._back_btn.clicked.connect(self._back_to_recommended)
        self._advanced_btn = QPushButton("Advanced Configuration")
        self._advanced_btn.clicked.connect(self._open_advanced)
        self._recommended_btn = QPushButton("Use Recommended Settings")
        self._recommended_btn.clicked.connect(self._use_recommended)
        self._download_btn = QPushButton("Download && Continue")
        self._download_btn.setDefault(True)
        self._download_btn.clicked.connect(self._accept_selection)
        btn_row.addWidget(self._back_btn)
        btn_row.addWidget(self._advanced_btn)
        btn_row.addWidget(self._recommended_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._download_btn)
        root.addLayout(btn_row)

        self._apply_styles()
        self._apply_scrollbar_style()
        self._sync_mode_ui()
        if not self._embedded and not self._split_embedded:
            apply_window_branding(self)

    def showEvent(self, event: QShowEvent | None) -> None:
        super().showEvent(event)
        if not self._embedded and not self._split_embedded:
            finalize_window_branding(self)
        QTimer.singleShot(0, self._refresh_list_section_layout)
        if self._metadata_worker is None:
            self._metadata_worker = BootstrapMetadataWorker(self)
            self._metadata_worker.finished_ok.connect(self._on_metadata_resolved)
            self._metadata_worker.failed.connect(self._on_metadata_failed)
            self._metadata_worker.start()

    def _refresh_list_section_layout(self) -> None:
        """Re-run layout after view switches (Recommended -> Advanced, etc.)."""
        layout = self.layout()
        if layout is not None:
            layout.activate()
        self._scroll_host.setMinimumHeight(0)
        self._scroll_host.updateGeometry()
        self._list_section.updateGeometry()

    @staticmethod
    def _initial_assessment() -> BootstrapSessionAssessment:
        resolved = {
            model_id: ResolvedBootstrapSize(
                model_id=model_id,
                size_bytes=BOOTSTRAP_MODELS[model_id].size_bytes,
                source=BootstrapSizeSource.ESTIMATE,
                detail="Loading…",
            )
            for model_id in BootstrapModelId
        }
        return build_session_assessment(resolved=resolved)

    def _format_hardware_summary(self) -> str:
        return self._assessment.hardware_summary().replace(" \u00b7 ", " | ")

    def _format_hf_status(self, base: str) -> str:
        return base.strip()

    def _on_metadata_resolved(self, resolved: object) -> None:
        if not isinstance(resolved, dict):
            return
        self._assessment = build_session_assessment(resolved=resolved)
        self._hardware_summary.setText(self._format_hardware_summary())
        self._hf_status.setText(self._format_hf_status(self._assessment.size_source_summary()))
        if not self._advanced and not self._user_touched_selection:
            self._apply_recommended_preset(expand_details_if_skipped=True)
        else:
            self._refresh_model_labels()
            self._refresh_totals()

    def _on_metadata_failed(self, message: str) -> None:
        self._hf_status.setText(
            self._format_hf_status(
                f"Could not verify all sizes with Hugging Face ({message}). "
                "Using offline estimates for download totals."
            )
        )
        if not self._advanced and not self._user_touched_selection:
            self._apply_recommended_preset(expand_details_if_skipped=True)
        else:
            self._refresh_model_labels()
            self._refresh_totals()

    def _resolved_sizes(self) -> dict[BootstrapModelId, int]:
        return self._assessment.size_bytes

    def _model_checkbox_label(self, model_id: BootstrapModelId) -> str:
        spec = BOOTSTRAP_MODELS[model_id]
        size = self._resolved_sizes().get(model_id, spec.size_bytes)
        return f"{spec.label} ({format_byte_size(size)})"

    def _model_checkbox_tooltip(self, model_id: BootstrapModelId) -> str:
        entry = self._assessment.resolved_sizes.get(model_id)
        if entry is None:
            return ""
        return format_bootstrap_size_tag_tooltip(entry)

    def _size_tag_tooltip(self, model_id: BootstrapModelId) -> str:
        entry = self._assessment.resolved_sizes.get(model_id)
        if entry is None:
            spec = BOOTSTRAP_MODELS[model_id]
            entry = ResolvedBootstrapSize(
                model_id=model_id,
                size_bytes=spec.size_bytes,
                source=BootstrapSizeSource.ESTIMATE,
                detail="Offline catalogue estimate",
            )
        return format_bootstrap_size_tag_tooltip(entry)

    def _feasible_recommended_set(self) -> set[BootstrapModelId]:
        return feasible_recommended_selection(
            self._assessment,
            locked_ids=locked_recommended_ids(),
        )

    def _feasible_recommended_state(self) -> dict[BootstrapModelId, bool]:
        feasible = self._feasible_recommended_set()
        return {mid: mid in feasible for mid in BootstrapModelId}

    def _apply_recommended_preset(self, *, expand_details_if_skipped: bool = False) -> None:
        """Apply the feasible recommended preset (recommended view only)."""
        feasible = self._feasible_recommended_set()
        self._selection_state = {mid: mid in feasible for mid in BootstrapModelId}
        skipped = default_selection(advanced=False) - feasible
        self._rebuild_model_list(persist_first=False)
        self._scroll_to_model_list_top()
        if expand_details_if_skipped and skipped:
            self._details_panel.set_collapsed(False)
        self._refresh_list_section_layout()

    def _advanced_preset_set(self) -> set[BootstrapModelId]:
        return normalize_selection(default_selection(advanced=True))

    def _apply_advanced_preset(self) -> None:
        """Reset advanced view to spec defaults (core on, optionals off)."""
        preset = self._advanced_preset_set()
        self._selection_state = {mid: mid in preset for mid in BootstrapModelId}
        self._rebuild_model_list(persist_first=False)
        self._scroll_to_model_list_top()

    def _apply_advanced_defaults(self) -> None:
        """Open advanced mode with spec defaults (core on, optionals off)."""
        self._advanced = True
        self._selection_state = self._default_state(advanced=True)
        self._sync_mode_ui()

    def _scroll_to_model_list_top(self) -> None:
        self._scroll.verticalScrollBar().setValue(0)

    def _details_collapsed_summary(
        self,
        *,
        download_bytes: int,
        headroom: int,
        can_proceed: bool,
    ) -> str:
        profile = self._assessment.profile
        ram = f"{profile.total_ram_gb:.0f} GB RAM" if profile.total_ram_gb > 0 else "RAM unknown"
        parts = [ram, f"{format_available_disk()} free", f"{format_byte_size(download_bytes)} selected"]
        if headroom < 0:
            parts.append("over disk budget")
        elif not can_proceed:
            parts.append("requirements not met")
        else:
            parts.append("ready")
        return " | ".join(parts)

    def _size_tag_for(self, model_id: BootstrapModelId) -> tuple[str, str]:
        entry = self._assessment.resolved_sizes.get(model_id)
        if entry and entry.source is BootstrapSizeSource.HUGGINGFACE:
            return "Verified", "BootstrapSizeTagVerified"
        return "Estimated", "BootstrapSizeTagEstimate"

    def _tier_tag_tooltip(self, model_id: BootstrapModelId) -> str:
        return format_bootstrap_tier_tag_tooltip(model_id)

    def _apply_tier_tag(self, model_id: BootstrapModelId) -> None:
        tag = self._tier_tags.get(model_id)
        if tag is None:
            return
        text, style = bootstrap_tier_tag(model_id)
        tag.setText(text)
        tag.setToolTip(self._tier_tag_tooltip(model_id))
        tag.setObjectName(style)
        tag.style().unpolish(tag)
        tag.style().polish(tag)

    def _blocked_models_detail(
        self,
        blocked: dict[BootstrapModelId, BootstrapModelFeasibility],
    ) -> str:
        lines: list[str] = []
        for model_id in self._visible_model_ids():
            fit = blocked.get(model_id)
            if fit is None:
                continue
            lines.append(f"{BOOTSTRAP_MODELS[model_id].label}: {fit.message}")
        return "\n".join(lines)

    def _refresh_model_labels(self) -> None:
        for model_id, cb in self._checkboxes.items():
            cb.setText(self._model_checkbox_label(model_id))
            self._apply_tier_tag(model_id)
            self._apply_size_tag(model_id)
        self._update_feasibility_notes()

    def _apply_size_tag(self, model_id: BootstrapModelId) -> None:
        tag = self._size_tags.get(model_id)
        if tag is None:
            return
        text, style = self._size_tag_for(model_id)
        tag.setText(text)
        tag.setToolTip(self._size_tag_tooltip(model_id))
        tag.setObjectName(style)
        tag.style().unpolish(tag)
        tag.style().polish(tag)
        if not _SHOW_BOOTSTRAP_SIZE_CHIPS:
            tag.hide()

    def _update_row_badges(self) -> None:
        selected = self._effective_selection()
        visible = set(self._visible_model_ids())
        blocked = models_blocked_for_session(selected, visible, self._assessment)
        for model_id in self._tier_tags:
            self._apply_tier_tag(model_id)
        for model_id, size_tag in self._size_tags.items():
            self._apply_size_tag(model_id)
        for model_id, block_tag in self._block_tags.items():
            cb = self._checkboxes.get(model_id)
            fit = blocked.get(model_id)
            if cb is not None and fit is not None and not cb.isChecked():
                if fit.block_reason is BootstrapBlockReason.DISK:
                    block_tag.setText("Disk")
                    block_tag.setObjectName("BootstrapBlockTagDisk")
                elif fit.block_reason is BootstrapBlockReason.MEMORY:
                    block_tag.setText("Memory")
                    block_tag.setObjectName("BootstrapBlockTagMemory")
                else:
                    block_tag.hide()
                    continue
                block_tag.show()
                block_tag.style().unpolish(block_tag)
                block_tag.style().polish(block_tag)
            else:
                block_tag.hide()

    def _checkbox_tooltip(
        self,
        model_id: BootstrapModelId,
        *,
        fit: BootstrapModelFeasibility | None = None,
    ) -> str:
        parts = [self._model_checkbox_tooltip(model_id)]
        if fit is not None and fit.message:
            parts.append(fit.message)
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _feasibility_note_style(block_reason: BootstrapBlockReason) -> str:
        if block_reason is BootstrapBlockReason.MEMORY:
            return "BootstrapModelFeasibilityBlock"
        if block_reason is BootstrapBlockReason.DISK:
            return "BootstrapModelFeasibilityDisk"
        return "BootstrapModelFeasibilityNote"

    def _set_feasibility_note(
        self,
        model_id: BootstrapModelId,
        *,
        text: str,
        block_reason: BootstrapBlockReason,
        visible: bool,
    ) -> None:
        note = self._feasibility_notes.get(model_id)
        if note is None:
            return
        note.setText(text)
        note.setObjectName(self._feasibility_note_style(block_reason))
        note.setVisible(visible)
        if visible:
            note.style().unpolish(note)
            note.style().polish(note)

    def _update_feasibility_notes(self) -> None:
        selected = self._effective_selection()
        for model_id, note in self._feasibility_notes.items():
            fit = assess_model_feasibility(model_id, selected, self._assessment)
            cb = self._checkboxes.get(model_id)
            is_checked = cb.isChecked() if cb is not None else model_id in selected
            if is_checked and fit.message and fit.block_reason is BootstrapBlockReason.NONE:
                self._set_feasibility_note(
                    model_id,
                    text=fit.message,
                    block_reason=BootstrapBlockReason.NONE,
                    visible=True,
                )
            else:
                self._set_feasibility_note(
                    model_id,
                    text="",
                    block_reason=BootstrapBlockReason.NONE,
                    visible=False,
                )

    @staticmethod
    def _default_state(*, advanced: bool) -> dict[BootstrapModelId, bool]:
        selected = default_selection(advanced=advanced)
        return {mid: mid in selected for mid in BootstrapModelId}

    def _apply_styles(self) -> None:
        if self._split_embedded:
            root_style = """
            QWidget#BootstrapConsentPanelSplit {
                background: transparent;
            }
            """
        elif self._embedded:
            root_style = """
            QWidget#BootstrapConsentPanelEmbedded {
                background: #12151f;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 16px;
            }
            """
        else:
            root_style = """
            QWidget#BootstrapConsentPanel {
                background: #12151f;
            }
            """
        self.setStyleSheet(
            root_style
            + """
            QWidget#BootstrapConsentPanelSplit QLabel,
            QWidget#BootstrapConsentPanelEmbedded QLabel,
            QWidget#BootstrapConsentPanel QLabel,
            QWidget#BootstrapModelTitleRow {
                background: transparent;
            }
            QLabel#BootstrapBrandTitle {
                color: #f8fafc;
                font-size: 22px;
                font-weight: 800;
            }
            QLabel#BootstrapTitle {
                color: #f8fafc;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#BootstrapIntro,
            QLabel#BootstrapLegend {
                color: #94a3b8;
                font-size: 13px;
            }
            QFrame#BootstrapCollapsiblePanel {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 10px;
            }
            QFrame#BootstrapCollapsibleHeader {
                background: transparent;
                border: none;
            }
            QFrame#BootstrapCollapsibleHeader:hover QLabel#BootstrapCollapsibleHeaderText,
            QFrame#BootstrapCollapsibleHeader:hover QLabel#BootstrapCollapsibleArrow {
                color: #c4b5fd;
            }
            QLabel#BootstrapCollapsibleArrow {
                color: #94a3b8;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel#BootstrapCollapsibleHeaderText {
                color: #94a3b8;
                font-size: 11px;
                font-weight: 600;
            }
            QLabel#BootstrapCollapsibleSummary {
                color: #64748b;
                font-size: 10px;
                line-height: 1.35;
                padding: 0 12px 8px 32px;
            }
            QFrame#BootstrapCollapsibleBody {
                border-top: 1px solid rgba(255, 255, 255, 0.06);
            }
            QLabel#BootstrapDiskSummary {
                color: #cbd5e1;
                font-size: 12px;
            }
            QLabel#BootstrapDiskSummaryOver {
                color: #fca5a5;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#BootstrapDiskNotice {
                color: #fbbf24;
                font-size: 11px;
                line-height: 1.35;
            }
            QLabel#BootstrapSizeTagVerified {
                background: rgba(34, 197, 94, 0.14);
                color: #86efac;
                border: 1px solid rgba(34, 197, 94, 0.28);
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#BootstrapSizeTagEstimate {
                background: rgba(148, 163, 184, 0.1);
                color: #94a3b8;
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#BootstrapTierTagRequired {
                background: rgba(139, 92, 246, 0.16);
                color: #c4b5fd;
                border: 1px solid rgba(139, 92, 246, 0.35);
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#BootstrapTierTagRecommended {
                background: rgba(56, 189, 248, 0.14);
                color: #7dd3fc;
                border: 1px solid rgba(56, 189, 248, 0.28);
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#BootstrapTierTagOptional {
                background: rgba(148, 163, 184, 0.08);
                color: #94a3b8;
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#BootstrapBlockTagDisk {
                background: rgba(251, 146, 60, 0.14);
                color: #fb923c;
                border: 1px solid rgba(251, 146, 60, 0.3);
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#BootstrapBlockTagMemory {
                background: rgba(248, 113, 113, 0.14);
                color: #fca5a5;
                border: 1px solid rgba(248, 113, 113, 0.3);
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }
            QFrame#BootstrapBulkBar {
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 8px;
            }
            QLabel#BootstrapBulkCaption {
                color: #64748b;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }
            QPushButton#BootstrapBulkPill {
                background: rgba(139, 92, 246, 0.12);
                color: #c4b5fd;
                border: 1px solid rgba(139, 92, 246, 0.28);
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 14px;
                min-width: 48px;
            }
            QPushButton#BootstrapBulkPill:hover {
                background: rgba(139, 92, 246, 0.22);
                color: #ede9fe;
            }
            QPushButton#BootstrapBulkPill:disabled {
                color: rgba(196, 181, 253, 0.35);
                border-color: rgba(139, 92, 246, 0.12);
            }
            QLabel#BootstrapLegend {
                font-size: 11px;
                color: #64748b;
            }
            QLabel#BootstrapHardwareSummary {
                color: #e2e8f0;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#BootstrapHfStatus {
                color: #94a3b8;
                font-size: 11px;
            }
            QLabel#BootstrapModelFeasibilityNote {
                color: #fbbf24;
                font-size: 11px;
                margin-left: 22px;
            }
            QLabel#BootstrapModelFeasibilityDisk {
                color: #fb923c;
                font-size: 11px;
                margin-left: 22px;
            }
            QLabel#BootstrapModelFeasibilityBlock {
                color: #fca5a5;
                font-size: 11px;
                margin-left: 22px;
            }
            QLabel#BootstrapTotalLabel {
                color: #c4b5fd;
                font-size: 13px;
                font-weight: 600;
            }
            QScrollArea#BootstrapScroll,
            QWidget#BootstrapScrollViewport,
            QWidget#BootstrapScrollHost,
            QWidget#BootstrapListSection {
                background: transparent;
                border: none;
            }
            QLabel#BootstrapBulkSep {
                color: rgba(148, 163, 184, 0.45);
                font-size: 11px;
                padding: 0 2px;
            }
            QPushButton#BootstrapLinkBtn {
                background: transparent;
                border: none;
                color: #64748b;
                font-size: 11px;
                font-weight: 500;
                padding: 0 2px;
            }
            QPushButton#BootstrapLinkBtn:hover {
                color: #c4b5fd;
            }
            QFrame#BootstrapModelRowDiskBlocked {
                background: rgba(15, 23, 42, 0.55);
                border: 1px dashed rgba(148, 163, 184, 0.22);
                border-radius: 10px;
            }
            QFrame#BootstrapModelRow {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 10px;
            }
            QFrame#BootstrapModelRowLocked {
                background: rgba(139, 92, 246, 0.08);
                border: 1px solid rgba(139, 92, 246, 0.22);
                border-radius: 10px;
            }
            QFrame#BootstrapModelRowInfo {
                background: rgba(148, 163, 184, 0.06);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 10px;
            }
            QFrame#BootstrapModelRowCaution {
                background: rgba(251, 191, 36, 0.06);
                border: 1px solid rgba(251, 191, 36, 0.22);
                border-radius: 10px;
            }
            QFrame#BootstrapModelRowCoreWarning {
                background: rgba(251, 146, 60, 0.08);
                border: 1px solid rgba(251, 146, 60, 0.28);
                border-radius: 10px;
            }
            QLabel#BootstrapModelDesc {
                color: #64748b;
                font-size: 11px;
                margin-left: 22px;
            }
            QLabel#BootstrapModelDescInfo {
                color: #94a3b8;
                font-size: 11px;
                margin-left: 22px;
            }
            QLabel#BootstrapModelDescCaution {
                color: #fbbf24;
                font-size: 11px;
                margin-left: 22px;
            }
            QLabel#BootstrapModelDescCoreWarning {
                color: #fb923c;
                font-size: 11px;
                margin-left: 22px;
            }
            QCheckBox#BootstrapCheckLocked {
                color: #c4b5fd;
            }
            QCheckBox {
                color: #e2e8f0;
                font-size: 13px;
                background: transparent;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid rgba(255, 255, 255, 0.22);
                background: rgba(15, 23, 42, 0.65);
            }
            QCheckBox::indicator:checked {
                background: #8b5cf6;
                border-color: #a78bfa;
            }
            QCheckBox::indicator:disabled {
                background: rgba(148, 163, 184, 0.12);
                border-color: rgba(148, 163, 184, 0.2);
            }
            QCheckBox::indicator:checked:disabled {
                background: #6d28d9;
                border-color: #a78bfa;
            }
            QPushButton {
                padding: 10px 16px;
                border-radius: 8px;
                font-weight: 600;
            }
            QPushButton#BootstrapPrimaryBtn {
                background: #8b5cf6;
                color: #0f172a;
                border: none;
            }
            QPushButton#BootstrapPrimaryBtn:disabled {
                background: rgba(139, 92, 246, 0.28);
                color: rgba(226, 232, 240, 0.45);
            }
            QPushButton#BootstrapSecondaryBtn {
                background: transparent;
                color: #cbd5e1;
                border: 1px solid rgba(255, 255, 255, 0.15);
            }
            QPushButton#BootstrapSecondaryBtn:disabled {
                color: rgba(203, 213, 225, 0.35);
                border-color: rgba(255, 255, 255, 0.06);
            }
            """
        )
        self._download_btn.setObjectName("BootstrapPrimaryBtn")
        for btn in (self._back_btn, self._advanced_btn, self._recommended_btn):
            btn.setObjectName("BootstrapSecondaryBtn")

    def _apply_scrollbar_style(self) -> None:
        """Style the model-list scrollbar directly (dialog QSS does not reach it on Windows)."""
        bar = self._scroll.verticalScrollBar()
        bar.setObjectName("BootstrapScrollBar")
        bar.setStyleSheet(
            """
            QScrollBar#BootstrapScrollBar:vertical {
                border: none;
                background: transparent;
                width: 10px;
                margin: 0px;
            }
            QScrollBar#BootstrapScrollBar::handle:vertical {
                background-color: rgba(148, 163, 184, 0.38);
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar#BootstrapScrollBar::handle:vertical:hover {
                background-color: rgba(196, 181, 253, 0.62);
            }
            QScrollBar#BootstrapScrollBar::add-line:vertical,
            QScrollBar#BootstrapScrollBar::sub-line:vertical,
            QScrollBar#BootstrapScrollBar::add-page:vertical,
            QScrollBar#BootstrapScrollBar::sub-page:vertical {
                border: none;
                background: transparent;
                width: 0px;
                height: 0px;
            }
            """
        )

    @staticmethod
    def _row_object_name(spec, *, advanced: bool, locked: bool) -> str:
        if locked:
            return "BootstrapModelRowLocked"
        hint = spec.hint_for(advanced=advanced)
        if hint is BootstrapHintLevel.CORE_WARNING:
            return "BootstrapModelRowCoreWarning"
        if hint is BootstrapHintLevel.CAUTION:
            return "BootstrapModelRowCaution"
        if hint is BootstrapHintLevel.INFO:
            return "BootstrapModelRowInfo"
        return "BootstrapModelRow"

    @staticmethod
    def _desc_object_name(spec, *, advanced: bool) -> str:
        hint = spec.hint_for(advanced=advanced)
        if hint is BootstrapHintLevel.CORE_WARNING:
            return "BootstrapModelDescCoreWarning"
        if hint is BootstrapHintLevel.CAUTION:
            return "BootstrapModelDescCaution"
        if hint is BootstrapHintLevel.INFO:
            return "BootstrapModelDescInfo"
        return "BootstrapModelDesc"

    def _legend_text(self) -> str:
        return "Highlighted rows show trade-offs when changing core models."

    def _visible_model_ids(self) -> tuple[BootstrapModelId, ...]:
        return ADVANCED_ORDER if self._advanced else RECOMMENDED_ORDER

    def _is_locked_in_view(self, model_id: BootstrapModelId) -> bool:
        spec = BOOTSTRAP_MODELS[model_id]
        return spec.locked_in_recommended and not self._advanced

    def _effective_selection(self) -> set[BootstrapModelId]:
        selected = {mid for mid, checked in self._selection_state.items() if checked}
        return self._enforce_locked_recommended(normalize_selection(selected))

    def _normalize_and_apply_state(self) -> None:
        selected = normalize_selection(
            {mid for mid, checked in self._selection_state.items() if checked}
        )
        if not self._advanced:
            selected.update(locked_recommended_ids())
            selected = normalize_selection(selected)
        self._selection_state = {mid: mid in selected for mid in BootstrapModelId}
        self._sync_checkboxes_from_state()
        self._refresh_totals()

    def _sync_checkboxes_from_state(self) -> None:
        for mid, cb in self._checkboxes.items():
            checked = self._selection_state.get(mid, False)
            if self._is_locked_in_view(mid):
                checked = True
            if cb.isChecked() != checked:
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)

    def _select_all_visible(self) -> None:
        self._user_touched_selection = True
        self._persist_checkbox_state()
        for model_id in self._visible_model_ids():
            if not self._is_locked_in_view(model_id):
                self._selection_state[model_id] = True
        self._normalize_and_apply_state()

    def _deselect_all_visible(self) -> None:
        self._user_touched_selection = True
        self._persist_checkbox_state()
        for model_id in self._visible_model_ids():
            if not self._is_locked_in_view(model_id):
                self._selection_state[model_id] = False
        if not self._advanced:
            for model_id in locked_recommended_ids():
                self._selection_state[model_id] = True
        else:
            for model_id, spec in BOOTSTRAP_MODELS.items():
                if spec.default_advanced:
                    self._selection_state[model_id] = True
        self._sync_checkboxes_from_state()
        self._refresh_totals()

    def _rebuild_model_list(self, *, persist_first: bool = True) -> None:
        if persist_first:
            self._persist_checkbox_state()
        self._checkboxes.clear()
        self._rows.clear()
        self._feasibility_notes.clear()
        self._tier_tags.clear()
        self._size_tags.clear()
        self._block_tags.clear()
        while self._scroll_layout.count():
            item = self._scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        order = ADVANCED_ORDER if self._advanced else RECOMMENDED_ORDER
        for model_id in order:
            spec = BOOTSTRAP_MODELS[model_id]
            locked = spec.locked_in_recommended and not self._advanced
            cb = QCheckBox(self._model_checkbox_label(model_id))
            cb.setToolTip(self._model_checkbox_tooltip(model_id))
            cb.blockSignals(True)
            cb.setChecked(self._selection_state.get(model_id, False))
            if locked:
                cb.setChecked(True)
                cb.setEnabled(False)
                cb.setObjectName("BootstrapCheckLocked")
                self._selection_state[model_id] = True
            cb.blockSignals(False)
            cb.toggled.connect(lambda _checked, mid=model_id: self._on_checkbox_toggled(mid))
            self._checkboxes[model_id] = cb
            cb.setToolTip(self._model_checkbox_tooltip(model_id))
            _BootstrapModelRow._pass_clicks_to_row(cb)

            row = _BootstrapModelRow(
                cb,
                locked=locked,
                object_name=self._row_object_name(spec, advanced=self._advanced, locked=locked),
            )
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(12, 9, 12, 9)
            row_layout.setSpacing(3)

            title_row = QWidget()
            title_row.setObjectName("BootstrapModelTitleRow")
            _BootstrapModelRow._transparent_label_background(title_row)
            title_layout = QHBoxLayout(title_row)
            title_layout.setContentsMargins(0, 0, 0, 0)
            title_layout.setSpacing(4)
            title_layout.addWidget(cb, 1)
            tier_tag = _BootstrapRowChip(cb, row)
            tier_tag.setObjectName("BootstrapTierTagOptional")
            size_tag = _BootstrapRowChip(cb, row)
            size_tag.setObjectName("BootstrapSizeTagEstimate")
            block_tag = QLabel()
            block_tag.setObjectName("BootstrapBlockTagDisk")
            block_tag.hide()
            _BootstrapModelRow._pass_clicks_to_row(block_tag)
            title_layout.addWidget(tier_tag)
            title_layout.addWidget(size_tag)
            title_layout.addWidget(block_tag)
            self._tier_tags[model_id] = tier_tag
            self._size_tags[model_id] = size_tag
            self._block_tags[model_id] = block_tag
            _BootstrapModelRow._pass_clicks_to_row(title_row)

            desc = QLabel(spec.description_for(advanced=self._advanced))
            desc.setWordWrap(True)
            desc.setObjectName(self._desc_object_name(spec, advanced=self._advanced))
            _BootstrapModelRow._transparent_label_background(desc)
            _BootstrapModelRow._pass_clicks_to_row(desc)
            row_layout.addWidget(title_row)
            row_layout.addWidget(desc)

            if locked:
                note = QLabel("Required - included in download total")
                note.setObjectName("BootstrapModelDescInfo")
                _BootstrapModelRow._transparent_label_background(note)
                _BootstrapModelRow._pass_clicks_to_row(note)
                row_layout.addWidget(note)

            feasibility_note = QLabel()
            feasibility_note.setWordWrap(True)
            feasibility_note.setObjectName("BootstrapModelFeasibilityNote")
            feasibility_note.hide()
            _BootstrapModelRow._transparent_label_background(feasibility_note)
            _BootstrapModelRow._pass_clicks_to_row(feasibility_note)
            row_layout.addWidget(feasibility_note)
            self._feasibility_notes[model_id] = feasibility_note

            self._scroll_layout.addWidget(row)
            self._rows[model_id] = row

        self._update_row_badges()
        self._update_feasibility_notes()
        self._refresh_totals()
        QTimer.singleShot(0, self._refresh_list_section_layout)

    def _persist_checkbox_state(self) -> None:
        for mid, cb in self._checkboxes.items():
            self._selection_state[mid] = cb.isChecked()

    def _sync_mode_ui(self) -> None:
        if self._advanced:
            self._title.setText("Advanced configuration")
            self._intro.setText(
                "Fine-tune downloads or continue without models for a minimal shell install. "
                "Features you enable later can prompt you to download the models they need."
            )
            self._legend.setText(self._legend_text())
            self._legend.show()
            self._back_btn.show()
            self._recommended_btn.show()
            self._advanced_btn.hide()
        else:
            self._sync_recommended_title()
            self._intro.setText(
                "Pick optional models below or open Advanced Configuration. "
                "Required models stay selected and respect your disk and memory limits."
            )
            self._legend.hide()
            self._back_btn.hide()
            self._recommended_btn.show()
            self._advanced_btn.show()
        self._rebuild_model_list()
        self._scroll_to_model_list_top()
        if not self._advanced:
            skipped = default_selection(advanced=False) - self._feasible_recommended_set()
            if skipped:
                self._details_panel.set_collapsed(False)
        self._refresh_list_section_layout()

    def _enforce_locked_recommended(self, selected: set[BootstrapModelId]) -> set[BootstrapModelId]:
        if self._advanced:
            return selected
        out = set(selected)
        out.update(locked_recommended_ids())
        return normalize_selection(out)

    def _is_on_recommended_preset(self) -> bool:
        """True when the current selection matches the feasible Recommended preset."""
        selected = normalize_selection(self._current_selection())
        if not self._advanced:
            selected = self._enforce_locked_recommended(selected)
        return selected == self._feasible_recommended_set()

    def _sync_recommended_title(self) -> None:
        if self._advanced:
            return
        if self._is_on_recommended_preset():
            self._title.setText("Recommended configuration")
        else:
            self._title.setText("Custom configuration")

    def _current_selection(self) -> set[BootstrapModelId]:
        self._persist_checkbox_state()
        return self._effective_selection()

    def _update_disk_affordability(self) -> None:
        selected = self._effective_selection()
        visible = set(self._visible_model_ids())
        blocked = models_blocked_for_session(selected, visible, self._assessment)
        for model_id, row in self._rows.items():
            if self._is_locked_in_view(model_id):
                continue
            cb = self._checkboxes[model_id]
            fit = blocked.get(model_id)
            session_blocked = fit is not None and not cb.isChecked()
            row.set_disk_blocked(session_blocked)
            if session_blocked and fit is not None:
                cb.setEnabled(False)
                cb.setToolTip(self._checkbox_tooltip(model_id, fit=fit))
            else:
                cb.setEnabled(True)
                cb.setToolTip(self._checkbox_tooltip(model_id))
        self._update_row_badges()
        self._update_feasibility_notes()

    def _on_checkbox_toggled(self, model_id: BootstrapModelId) -> None:
        self._user_touched_selection = True
        self._persist_checkbox_state()
        if self._selection_state.get(model_id, False):
            if model_id in SIDECAR_GROUP:
                for other in SIDECAR_GROUP - {model_id}:
                    self._selection_state[other] = False
            if model_id in MAIN_LLM_GROUP:
                for other in MAIN_LLM_GROUP - {model_id}:
                    self._selection_state[other] = False
        selected = normalize_selection(
            {mid for mid, checked in self._selection_state.items() if checked}
        )
        if not self._advanced:
            selected.update(locked_recommended_ids())
            selected = normalize_selection(selected)
        sizes = self._resolved_sizes()
        if self._selection_state.get(model_id, False):
            base = normalize_selection(
                {mid for mid, checked in self._selection_state.items() if checked and mid != model_id}
            )
            if not self._advanced:
                base.update(locked_recommended_ids())
                base = normalize_selection(base)
            fit = assess_model_feasibility(model_id, base, self._assessment)
            if fit.block_reason is not BootstrapBlockReason.NONE:
                self._selection_state[model_id] = False
                selected = normalize_selection(
                    {mid for mid, checked in self._selection_state.items() if checked}
                )
                if not self._advanced:
                    selected.update(locked_recommended_ids())
                    selected = normalize_selection(selected)
                self._disk_notice.setText(fit.message)
                self._disk_notice.show()
                self._selection_state = {mid: mid in selected for mid in BootstrapModelId}
                self._sync_checkboxes_from_state()
                self._refresh_totals()
                return
        if not selection_within_budget(selected, sizes=sizes):
            self._selection_state[model_id] = False
            selected = normalize_selection(
                {mid for mid, checked in self._selection_state.items() if checked}
            )
            if not self._advanced:
                selected.update(locked_recommended_ids())
                selected = normalize_selection(selected)
            self._disk_notice.setText(
                f"Cannot add {BOOTSTRAP_MODELS[model_id].label} - not enough free disk space "
                f"for this selection. Deselect other models or free disk space."
            )
            self._disk_notice.show()
            self._selection_state = {mid: mid in selected for mid in BootstrapModelId}
            self._sync_checkboxes_from_state()
            self._refresh_totals()
            return
        self._selection_state = {mid: mid in selected for mid in BootstrapModelId}
        self._sync_checkboxes_from_state()
        self._refresh_totals()

    def _disk_status_line(
        self,
        *,
        download_bytes: int,
        headroom: int,
        over_budget: bool = False,
    ) -> str:
        if over_budget:
            return (
                f"Available: {format_available_disk()} | "
                f"Selected: {format_byte_size(download_bytes)} + "
                f"{format_byte_size(SAFETY_BUFFER_BYTES)} safety buffer | "
                f"Over budget by {format_byte_size(-headroom)}"
            )
        return (
            f"Available: {format_available_disk()} | "
            f"Selected: {format_byte_size(download_bytes)} + "
            f"{format_byte_size(SAFETY_BUFFER_BYTES)} safety buffer | "
            f"Headroom: {format_byte_size(headroom)}"
        )

    def _download_block_reason(
        self,
        *,
        can_proceed: bool,
        proceed_message: str,
        headroom: int,
    ) -> str:
        if can_proceed:
            return ""
        if headroom < 0:
            return (
                "Not enough free disk space for this selection. "
                "Deselect models or free disk space."
            )
        return proceed_message

    def _is_core_only_recommended_selection(self, selected: set[BootstrapModelId]) -> bool:
        if self._advanced:
            return False
        return normalize_selection(selected) == normalize_selection(set(locked_recommended_ids()))

    def _required_models_disk_notice(self, *, sizes: dict[BootstrapModelId, int]) -> str:
        required = locked_recommended_ids()
        need = required_bytes_for(required, sizes=sizes)
        download = total_selected_bytes(required, sizes=sizes)
        return (
            f"Required models need {format_byte_size(need)} "
            f"({format_byte_size(download)} downloads + "
            f"{format_byte_size(SAFETY_BUFFER_BYTES)} safety buffer); "
            f"only {format_available_disk()} available. Free disk space to continue."
        )

    def _refresh_totals(self) -> None:
        sizes = self._resolved_sizes()
        selected = self._effective_selection()
        download_bytes = total_selected_bytes(selected, sizes=sizes)
        headroom = budget_headroom_bytes(selected, sizes=sizes)
        allow_shell = self._advanced and not selected
        can_proceed, proceed_message = can_proceed_with_selection(
            selected,
            self._assessment,
            allow_empty=allow_shell,
        )

        if headroom < 0:
            self._disk_summary.setObjectName("BootstrapDiskSummaryOver")
            self._disk_summary.setText(
                self._disk_status_line(
                    download_bytes=download_bytes,
                    headroom=headroom,
                    over_budget=True,
                )
            )
            if self._is_core_only_recommended_selection(selected):
                self._disk_notice.setText(self._required_models_disk_notice(sizes=sizes))
            else:
                self._disk_notice.setText(
                    "Your selection exceeds available disk space. Deselect models or free disk "
                    "space before you can continue."
                )
            self._disk_notice.show()
        elif not can_proceed:
            self._disk_summary.setObjectName("BootstrapDiskSummary")
            self._disk_summary.setText(
                self._disk_status_line(download_bytes=download_bytes, headroom=headroom)
            )
            self._disk_notice.setText(proceed_message)
            self._disk_notice.show()
        else:
            self._disk_summary.setObjectName("BootstrapDiskSummary")
            self._disk_summary.setText(
                self._disk_status_line(download_bytes=download_bytes, headroom=headroom)
            )
            blocked = models_blocked_for_session(
                selected, set(self._visible_model_ids()), self._assessment
            )
            if blocked:
                self._disk_notice.setText(self._blocked_models_detail(blocked))
                self._disk_notice.show()
            else:
                self._disk_notice.hide()

        self._details_panel.set_summary(
            self._details_collapsed_summary(
                download_bytes=download_bytes,
                headroom=headroom,
                can_proceed=can_proceed,
            )
        )
        self._disk_summary.style().unpolish(self._disk_summary)
        self._disk_summary.style().polish(self._disk_summary)

        if not self._advanced and locked_recommended_ids() & selected:
            locked_bytes = total_selected_bytes(locked_recommended_ids() & selected, sizes=sizes)
            self._total_label.setText(
                f"Download total: {format_byte_size(download_bytes)} "
                f"(includes {format_byte_size(locked_bytes)} required)"
            )
        else:
            self._total_label.setText(f"Download total: {format_byte_size(download_bytes)}")
        self._sync_recommended_title()
        self._recommended_btn.setEnabled(not self._is_on_recommended_preset())
        if allow_shell:
            self._download_btn.setText("Continue without models")
        else:
            self._download_btn.setText("Download && Continue")
        self._download_btn.setEnabled(can_proceed)
        block_reason = self._download_block_reason(
            can_proceed=can_proceed,
            proceed_message=proceed_message,
            headroom=headroom,
        )
        if not can_proceed and self._is_core_only_recommended_selection(selected) and headroom < 0:
            block_reason = self._required_models_disk_notice(sizes=sizes)
        self._download_btn.setToolTip(block_reason)
        self._update_disk_affordability()
        if not can_proceed:
            self._details_panel.set_collapsed(False)

    def _open_advanced(self) -> None:
        self._user_touched_selection = False
        self._apply_advanced_defaults()

    def _back_to_recommended(self) -> None:
        self._user_touched_selection = False
        self._advanced = False
        feasible = self._feasible_recommended_set()
        self._selection_state = {mid: mid in feasible for mid in BootstrapModelId}
        skipped = default_selection(advanced=False) - feasible
        self._sync_mode_ui()
        if skipped:
            self._details_panel.set_collapsed(False)

    def _use_recommended(self) -> None:
        self._user_touched_selection = False
        self._apply_recommended_preset(expand_details_if_skipped=True)

    def _confirm_shell_install(self) -> bool:
        parent = self.window() if self._embedded else self
        dlg = PrestigeDialog(
            parent,
            "Continue without models?",
            format_shell_install_warning_message(),
            is_dark=True,
            tone="danger",
            dialog_width=480,
            confirm_text="Continue without models",
            cancel_text="Go back",
        )
        return bool(dlg.exec())

    def _accept_selection(self) -> None:
        selected = self._current_selection()
        allow_empty = self._advanced and not selected
        can_proceed, message = can_proceed_with_selection(
            selected,
            self._assessment,
            allow_empty=allow_empty,
        )
        if not can_proceed:
            if "disk space" in message.lower():
                QMessageBox.warning(self, "Insufficient disk space", message)
            else:
                QMessageBox.warning(self, "Models may not run on this system", message)
            return
        if allow_empty and not self._confirm_shell_install():
            return
        ok, message = preflight_download(selected, sizes=self._resolved_sizes())
        if not ok:
            QMessageBox.warning(self, "Insufficient disk space", message)
            return
        self._selected = selected
        self.selection_confirmed.emit(set(selected))

    def selected_models(self) -> set[BootstrapModelId] | None:
        return set(self._selected) if self._selected is not None else None


class BootstrapConsentDialog(QDialog):
    """Modal wrapper around :class:`BootstrapConsentPanel` for standalone use."""

    def __init__(self, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected: set[BootstrapModelId] | None = None
        self.setObjectName("BootstrapConsentDialog")
        self.setProperty("qube_tooltip_clip", True)
        self.setWindowTitle("Welcome to Qube")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            QDialog#BootstrapConsentDialog {
                background: #12151f;
                color: #cbd5e1;
            }
            """
        )
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setMinimumSize(640, 680)
        self.resize(700, 820)
        apply_window_branding(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._panel = BootstrapConsentPanel(parent=self, embedded=False)
        self._panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._panel.selection_confirmed.connect(self._on_selection_confirmed)
        layout.addWidget(self._panel)

    def showEvent(self, event: QShowEvent | None) -> None:
        super().showEvent(event)
        finalize_window_branding(self)

    def _on_selection_confirmed(self, selected: set[BootstrapModelId]) -> None:
        self._selected = set(selected)
        self.accept()

    def selected_models(self) -> set[BootstrapModelId] | None:
        return set(self._selected) if self._selected is not None else None


def run_bootstrap_consent(parent: QWidget | None = None) -> set[BootstrapModelId] | None:
    """Block until the user confirms model selection or cancels."""
    dlg = BootstrapConsentDialog(parent=parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return dlg.selected_models()


def run_bootstrap_consent_flow(parent: QWidget | None = None) -> set[BootstrapModelId] | None:
    """Show consent, persist selection, and return chosen model ids."""
    selected = run_bootstrap_consent(parent)
    if not selected:
        return None
    save_bootstrap_selection(selected)
    return selected
