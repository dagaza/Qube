"""First-connect and drift grant review dialog for MCP capabilities."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.integrations.capability_drift import CapabilityDriftDiff, format_drift_summary
from core.integrations.grant_review import (
    GrantReviewChange,
    GrantReviewRow,
    SuggestedCapabilityPreset,
    apply_grant_review_rows,
    build_grant_review_rows,
    save_suggested_capability_preset,
    suggest_capability_presets,
)
from core.integrations.mcp_discovery import McpDiscoveryResult, PROVIDER_ID
from core.theme.accessors import theme_for
from core.theme.widget_styles import (
    PRESTIGE_ACCENT_LABEL,
    PRESTIGE_DIALOG_CANCEL,
    PRESTIGE_DIALOG_CONFIRM,
    PRESTIGE_MUTED_LABEL,
    PRESTIGE_SOURCE_CONTAINER,
    TOGGLE_BUTTON,
)
from ui.components.prestige_dialog import _center_dialog_on_host, _resolve_is_dark_from_parent
from ui.components.toggle import PrestigeToggle

_DIALOG_WIDTH = 560
_DIALOG_MIN_HEIGHT = 420
_CONTENT_MARGIN = 28
_SCROLL_BODY_RIGHT_INSET = 18
_PROVIDER_ID = PROVIDER_ID

_TIER_LABELS = {
    "read": "Read",
    "write": "Write",
    "destructive": "Destructive",
}


class CapabilityGrantReviewDialog(QDialog):
    """Modal grant review after MCP discovery (first connect or drift)."""

    def __init__(
        self,
        parent,
        *,
        server_label: str,
        namespace: str,
        result: McpDiscoveryResult,
        is_dark: bool | None = None,
    ) -> None:
        super().__init__(parent)
        if is_dark is None:
            is_dark = _resolve_is_dark_from_parent(parent)
        theme = theme_for(is_dark=is_dark)
        self._is_dark = is_dark
        self._theme = theme

        self._saved = False
        self._server_label = server_label
        self._rows: list[GrantReviewRow] = build_grant_review_rows(
            _PROVIDER_ID,
            list(result.descriptors),
            namespace=namespace,
            first_connect=result.first_connect,
            drift=result.drift,
        )
        self._row_widgets: list[tuple[GrantReviewRow, PrestigeToggle]] = []
        self._preset_widgets: list[tuple[SuggestedCapabilityPreset, PrestigeToggle]] = []
        self._selected_preset: SuggestedCapabilityPreset | None = None
        self._presets: list[SuggestedCapabilityPreset] = suggest_capability_presets(
            namespace=namespace,
            server_label=server_label,
            descriptors=list(result.descriptors),
        )

        title = f"{server_label} — Review permissions"
        if result.drift and not result.first_connect:
            title = f"{server_label} updated"

        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(_DIALOG_WIDTH)
        self.setMinimumHeight(_DIALOG_MIN_HEIGHT)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("CapabilityGrantReviewContainer")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        container.setStyleSheet(
            theme.style(
                PRESTIGE_SOURCE_CONTAINER,
                accent=theme.link,
                object_name="CapabilityGrantReviewContainer",
            )
            + f"""
            QLabel {{
                color: {theme.text_primary};
                background: transparent;
                border: none;
            }}
            """
        )

        inner = QVBoxLayout(container)
        inner.setContentsMargins(_CONTENT_MARGIN, 26, _CONTENT_MARGIN, 22)
        inner.setSpacing(14)

        header = QLabel(title.upper())
        header.setStyleSheet(
            theme.style(PRESTIGE_ACCENT_LABEL, accent=theme.link, font_size="11px")
        )
        inner.addWidget(header)

        intro_lines = ["Capabilities discovered from this MCP server."]
        if result.drift:
            intro_lines.append(format_drift_summary(result.drift) + ".")
        intro_lines.append(
            "Read capabilities are suggested on; write and destructive stay off until you allow them."
        )
        intro = QLabel(" ".join(intro_lines))
        intro.setWordWrap(True)
        intro.setStyleSheet(theme.style(PRESTIGE_MUTED_LABEL, font_size="12px"))
        inner.addWidget(intro)

        if result.drift:
            inner.addWidget(self._build_drift_banner(result.drift, theme))

        if self._rows:
            inner.addWidget(self._build_capability_bulk_actions(theme))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        scroll.setStyleSheet(
            """
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            """
        )
        scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        scroll.viewport().setAutoFillBackground(False)
        scroll.viewport().setStyleSheet("background: transparent;")

        body = QWidget()
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, _SCROLL_BODY_RIGHT_INSET, 0)
        body_layout.setSpacing(6)

        for row in self._rows:
            body_layout.addWidget(self._build_capability_row(row, theme))
        body_layout.addStretch(1)
        scroll.setWidget(body)
        scroll.setMinimumHeight(220)
        inner.addWidget(scroll, stretch=1)
        self._sync_bulk_selection()

        if self._presets:
            preset_label = QLabel("Suggested presets — choose one to apply")
            preset_label.setStyleSheet(
                theme.style(PRESTIGE_ACCENT_LABEL, accent=theme.link, font_size="10px")
            )
            inner.addWidget(preset_label)
            for preset in self._presets:
                inner.addWidget(self._build_preset_row(preset, theme))

        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        cancel = QPushButton("LATER")
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setStyleSheet(theme.style(PRESTIGE_DIALOG_CANCEL))
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)

        save = QPushButton("SAVE PERMISSIONS")
        save.setCursor(Qt.CursorShape.PointingHandCursor)
        save.setStyleSheet(
            theme.style(
                PRESTIGE_DIALOG_CONFIRM,
                accent=theme.link,
                accent_text=theme.text_on_accent,
            )
        )
        save.clicked.connect(self._on_save)
        buttons.addWidget(save)
        inner.addLayout(buttons)

        outer.addWidget(container)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, lambda: _center_dialog_on_host(self))

    @property
    def saved(self) -> bool:
        return self._saved

    def _build_drift_banner(self, drift: CapabilityDriftDiff, theme) -> QWidget:
        row = QFrame()
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet(
            f"background: {theme.surface_elevated}; border-radius: 10px; padding: 8px;"
        )
        layout = QVBoxLayout(row)
        layout.setContentsMargins(12, 8, 12, 8)
        text = QLabel(format_drift_summary(drift).capitalize())
        text.setWordWrap(True)
        layout.addWidget(text)
        if drift.removed:
            removed = QLabel(
                "Removed: " + ", ".join(action.replace("-", " ") for action in drift.removed[:6])
            )
            removed.setWordWrap(True)
            removed.setStyleSheet(theme.style(PRESTIGE_MUTED_LABEL, font_size="11px"))
            layout.addWidget(removed)
        return row

    def _build_capability_bulk_actions(self, theme) -> QWidget:
        row = QWidget()
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(10)

        segment = QFrame()
        segment.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        border = theme.border_subtle if theme.is_dark else theme.border
        segment.setStyleSheet(
            f"QFrame {{ background: transparent; border: 1px solid {border}; border-radius: 8px; }}"
        )
        segment_layout = QHBoxLayout(segment)
        segment_layout.setContentsMargins(2, 2, 2, 2)
        segment_layout.setSpacing(2)

        self._bulk_all_btn = QPushButton("All")
        self._bulk_none_btn = QPushButton("None")
        for btn in (self._bulk_all_btn, self._bulk_none_btn):
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(28)
            btn.setMinimumWidth(52)

        self._bulk_group = QButtonGroup(row)
        self._bulk_group.setExclusive(True)
        self._bulk_group.addButton(self._bulk_all_btn)
        self._bulk_group.addButton(self._bulk_none_btn)
        self._bulk_all_btn.clicked.connect(self._on_bulk_all_clicked)
        self._bulk_none_btn.clicked.connect(self._on_bulk_none_clicked)

        segment_layout.addWidget(self._bulk_all_btn)
        segment_layout.addWidget(self._bulk_none_btn)
        layout.addWidget(segment, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)

        self._style_bulk_button(self._bulk_all_btn, checked=False)
        self._style_bulk_button(self._bulk_none_btn, checked=False)
        return row

    def _style_bulk_button(self, btn: QPushButton, *, checked: bool) -> None:
        btn.blockSignals(True)
        btn.setChecked(checked)
        btn.blockSignals(False)
        btn.setStyleSheet(
            self._theme.style(TOGGLE_BUTTON, checked=checked, active_bg=self._theme.link)
            + " QPushButton { border-radius: 6px; font-size: 11px; font-weight: 600; padding: 4px 10px; }"
        )

    def _enabled_capability_toggles(self) -> list[PrestigeToggle]:
        return [toggle for row, toggle in self._row_widgets if row.enabled]

    def _sync_bulk_selection(self) -> None:
        all_btn = getattr(self, "_bulk_all_btn", None)
        none_btn = getattr(self, "_bulk_none_btn", None)
        if all_btn is None or none_btn is None:
            return
        enabled = self._enabled_capability_toggles()
        if not enabled:
            all_btn.setEnabled(False)
            none_btn.setEnabled(False)
            self._style_bulk_button(all_btn, checked=False)
            self._style_bulk_button(none_btn, checked=False)
            return
        all_btn.setEnabled(True)
        none_btn.setEnabled(True)
        all_checked = all(t.isChecked() for t in enabled)
        none_checked = not any(t.isChecked() for t in enabled)
        self._style_bulk_button(all_btn, checked=all_checked)
        self._style_bulk_button(none_btn, checked=none_checked)

    def _on_bulk_all_clicked(self) -> None:
        self._set_all_capabilities(True)

    def _on_bulk_none_clicked(self) -> None:
        self._set_all_capabilities(False)

    def _set_all_capabilities(self, checked: bool) -> None:
        self._clear_preset_selection()
        for row, toggle in self._row_widgets:
            if not row.enabled:
                continue
            toggle.blockSignals(True)
            toggle.setChecked(checked)
            toggle.blockSignals(False)
        self._sync_bulk_selection()

    def _clear_preset_selection(self) -> None:
        for _preset, toggle in self._preset_widgets:
            toggle.blockSignals(True)
            toggle.setChecked(False)
            toggle.blockSignals(False)
        self._selected_preset = None

    def _on_capability_toggle_changed(self) -> None:
        self._clear_preset_selection()
        self._sync_bulk_selection()

    def _build_capability_row(self, row: GrantReviewRow, theme) -> QWidget:
        outer = QWidget()
        outer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(outer)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(10)

        toggle = PrestigeToggle()
        toggle.setEnabled(row.enabled)
        toggle.setChecked(row.checked)
        toggle.toggled.connect(lambda _checked: self._on_capability_toggle_changed())
        self._row_widgets.append((row, toggle))
        layout.addWidget(toggle, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        title = QLabel(row.descriptor.action.replace("-", " ").title())
        title.setWordWrap(True)
        text_col.addWidget(title)

        meta_parts: list[str] = []
        if row.change is GrantReviewChange.NEW:
            meta_parts.append("new")
        elif row.change is GrantReviewChange.CHANGED:
            meta_parts.append("changed")
        if not row.enabled:
            meta_parts.append("needs review")
        if meta_parts:
            meta = QLabel(" · ".join(meta_parts))
            meta.setWordWrap(True)
            meta.setStyleSheet(theme.style(PRESTIGE_MUTED_LABEL, font_size="11px"))
            text_col.addWidget(meta)
        layout.addLayout(text_col, stretch=1)

        tier = self._build_tier_hint(row.descriptor.tier.value)
        layout.addWidget(tier, alignment=Qt.AlignmentFlag.AlignTop)
        return outer

    def _build_tier_hint(self, tier_value: str) -> QLabel:
        muted = "#585b70" if self._is_dark else "#94a3b8"
        label = QLabel(_TIER_LABELS.get(tier_value, tier_value.title()))
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop
        )
        label.setStyleSheet(
            f"color: {muted}; font-size: 10px; background: transparent; border: none; padding-right: 4px;"
        )
        label.setMinimumWidth(68)
        return label

    def _build_preset_row(
        self,
        preset: SuggestedCapabilityPreset,
        theme,
    ) -> QWidget:
        row = QWidget()
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(10)

        toggle = PrestigeToggle()
        toggle.toggled.connect(
            lambda checked, p=preset, t=toggle: self._on_preset_toggled(p, t, checked)
        )
        self._preset_widgets.append((preset, toggle))
        layout.addWidget(toggle, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        title = QLabel(preset.label)
        title.setWordWrap(True)
        text_col.addWidget(title)
        meta = QLabel(preset.description)
        meta.setWordWrap(True)
        meta.setStyleSheet(theme.style(PRESTIGE_MUTED_LABEL, font_size="11px"))
        text_col.addWidget(meta)
        layout.addLayout(text_col, stretch=1)
        return row

    def _on_preset_toggled(
        self,
        preset: SuggestedCapabilityPreset,
        toggle: PrestigeToggle,
        checked: bool,
    ) -> None:
        if checked:
            for _other_preset, other_toggle in self._preset_widgets:
                if other_toggle is toggle:
                    continue
                other_toggle.blockSignals(True)
                other_toggle.setChecked(False)
                other_toggle.blockSignals(False)
            self._selected_preset = preset
            self._apply_preset_to_capability_toggles(preset)
            return
        if self._selected_preset is preset:
            self._selected_preset = None

    def _apply_preset_to_capability_toggles(
        self,
        preset: SuggestedCapabilityPreset,
    ) -> None:
        from core.integrations.grant_review import capability_in_preset

        for row, cap_toggle in self._row_widgets:
            if not row.enabled:
                continue
            cap_toggle.blockSignals(True)
            cap_toggle.setChecked(capability_in_preset(row.descriptor, preset))
            cap_toggle.blockSignals(False)
        self._sync_bulk_selection()

    def _on_save(self) -> None:
        updated: list[GrantReviewRow] = []
        for row, toggle in self._row_widgets:
            updated.append(
                GrantReviewRow(
                    descriptor=row.descriptor,
                    checked=toggle.isChecked(),
                    enabled=row.enabled,
                    change=row.change,
                    ui_state=row.ui_state,
                )
            )
        apply_grant_review_rows(_PROVIDER_ID, updated)
        if self._selected_preset is not None:
            save_suggested_capability_preset(
                self._selected_preset,
                server_label=self._server_label,
            )
        self._saved = True
        self.accept()


def open_capability_grant_review_dialog(
    parent,
    *,
    server_label: str,
    namespace: str,
    result: McpDiscoveryResult,
    is_dark: bool | None = None,
) -> bool:
    """Show grant review when first connecting or when drift is detected."""
    if result.error:
        return False
    if not result.first_connect and result.drift is None:
        return False
    dialog = CapabilityGrantReviewDialog(
        parent,
        server_label=server_label,
        namespace=namespace,
        result=result,
        is_dark=is_dark,
    )
    dialog.exec()
    return dialog.saved
