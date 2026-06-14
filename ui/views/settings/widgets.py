"""Reusable settings UI widgets."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.components.brand_buttons import apply_brand_caution
from ui.components.toggle import PrestigeToggle

SETTINGS_SECTION_RESET_BUTTON_TEXT = "Reset to default configuration"


def make_settings_page_action_button(
    text: str,
    *,
    caution: bool = False,
) -> QPushButton:
    btn = QPushButton(text)
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    if caution:
        apply_brand_caution(btn)
    return btn


def add_section_reset_footer(
    layout: QVBoxLayout | QFormLayout,
    host,
    section_id: str,
    *,
    is_dark: bool = True,
) -> QPushButton:
    """Divider, spacing, and centered reset button for a settings page."""
    divider = SettingsSectionDivider(is_dark=is_dark)
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.addStretch(1)
    btn = make_settings_page_action_button(
        SETTINGS_SECTION_RESET_BUTTON_TEXT,
        caution=True,
    )
    btn.setToolTip(
        "Restore every setting on this page to its default configuration."
    )
    btn.clicked.connect(lambda _checked=False, sid=section_id: host._on_reset_section_defaults(sid))
    row_layout.addWidget(btn)
    row_layout.addStretch(1)

    if isinstance(layout, QFormLayout):
        layout.addRow(divider)
        layout.addRow("", row)
    else:
        layout.addWidget(divider)
        layout.addWidget(row)
    return btn


class SettingsSectionDivider(QWidget):
    """Full-width horizontal rule; custom-painted so it stays visible in QFormLayout."""

    _MARGIN_TOP = 20
    _LINE_HEIGHT = 2

    def __init__(self, *, is_dark: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsSectionDivider")
        self.setFixedHeight(self._MARGIN_TOP + self._LINE_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._is_dark = is_dark

    def apply_theme(self, is_dark: bool) -> None:
        if is_dark != self._is_dark:
            self._is_dark = is_dark
            self.update()

    def paintEvent(self, event) -> None:
        del event
        color = QColor("#585b70" if self._is_dark else "#cbd5e1")
        painter = QPainter(self)
        painter.fillRect(0, self._MARGIN_TOP, self.width(), self._LINE_HEIGHT, color)


def make_subsection_label(text: str, *, anchor: str | None = None) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("SettingsSubsectionLabel")
    if anchor:
        lbl.setProperty("settings_anchor", anchor)
    return lbl


def add_subsection_to_form(
    form: QFormLayout, text: str, *, anchor: str | None = None
) -> QLabel:
    lbl = make_subsection_label(text, anchor=anchor)
    form.addRow(lbl)
    return lbl


def add_subsection_to_layout(
    layout: QVBoxLayout, text: str, *, anchor: str | None = None
) -> QLabel:
    lbl = make_subsection_label(text, anchor=anchor)
    layout.addWidget(lbl)
    return lbl


def add_section_divider_to_form(form: QFormLayout, *, is_dark: bool = True) -> SettingsSectionDivider:
    """Full-width horizontal rule separating major settings blocks."""
    divider = SettingsSectionDivider(is_dark=is_dark)
    form.addRow(divider)
    return divider


def wrap_subsection(content: QWidget, *, anchor: str | None = None) -> QWidget:
    """Wrap content in a container tagged for engine-mode visibility toggles."""
    wrapper = QWidget()
    if anchor:
        wrapper.setProperty("settings_anchor", anchor)
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(content)
    return wrapper


def make_disclosure_row(
    host,
    label_text: str,
    tooltip: str,
    *,
    panel_object_name: str = "SettingsDisclosurePanel",
) -> tuple[PrestigeToggle, QLabel, QWidget]:
    """Toggle + label row and an empty panel for progressive disclosure."""
    toggle = PrestigeToggle()
    toggle.setToolTip(tooltip)
    label = QLabel(label_text)
    label.setToolTip(tooltip)
    info_btn = host._make_settings_info_button(tooltip)
    label_cluster = QWidget()
    label_cluster_layout = QHBoxLayout(label_cluster)
    label_cluster_layout.setContentsMargins(0, 0, 0, 0)
    label_cluster_layout.setSpacing(6)
    label_cluster_layout.addWidget(label)
    label_cluster_layout.addWidget(info_btn)
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(8)
    row_layout.addWidget(toggle, alignment=Qt.AlignmentFlag.AlignLeft)
    row_layout.addWidget(label_cluster)
    row_layout.addStretch(1)
    panel = QWidget()
    panel.setObjectName(panel_object_name)
    panel_layout = QVBoxLayout(panel)
    panel_layout.setContentsMargins(0, 8, 0, 0)
    panel_layout.setSpacing(12)
    return toggle, row, panel


def register_theme_button(host, button) -> None:
    """Track a SelectorButton for theme refresh (also see ``collect_theme_buttons``)."""
    buttons = getattr(host, "_theme_buttons", None)
    if buttons is None:
        host._theme_buttons = []
    if button not in host._theme_buttons:
        host._theme_buttons.append(button)


def collect_theme_buttons(host) -> None:
    """Register all SelectorButton widgets under the settings view."""
    from ui.components.selector_button import SelectorButton

    host._theme_buttons = []
    for btn in host.findChildren(SelectorButton):
        host._theme_buttons.append(btn)


def track_internal_ai_label(host, label: QLabel) -> None:
    labels = getattr(host, "_ai_internal_subsection_labels", None)
    if labels is None:
        host._ai_internal_subsection_labels = []
    host._ai_internal_subsection_labels.append(label)


def make_external_engine_hint(host) -> QLabel:
    hint = QLabel(
        "Local model library, hardware tuning, chat template, and startup loading "
        "apply when the Internal engine is selected. Switch AI Engine to Internal "
        "to configure these options."
    )
    hint.setWordWrap(True)
    hint.setProperty("class", "SettingsHint")
    hint.setObjectName("SettingsExternalEngineHint")
    host._ai_external_engine_hint = hint
    return hint
