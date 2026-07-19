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
from ui.components.page_tour_help_button import PageTourHelpButton
from ui.components.selector_button import SelectorButton
from ui.components.toggle import PrestigeToggle

SETTINGS_SECTION_RESET_BUTTON_TEXT = "Reset to default configuration"
SETTINGS_SELECTOR_LABELS_PROP = "settings_selector_labels"
SETTINGS_SELECTOR_MIN_WIDTH_PROP = "settings_selector_min_width"


def fit_settings_selector_width(selector: SelectorButton, *labels: str) -> None:
    """Size a settings ``SelectorButton`` to its widest label without eliding.

    Uses the same ``elidedText`` probe as ``SelectorButton.paintEvent`` so width
    matches what is actually painted. Call again after ``setText`` or once the
    widget is shown if fonts were not final at construction time.
    """
    label_list = [label for label in labels if label]
    if not label_list:
        label_list = [selector.text()] if selector.text() else [""]

    inset = SelectorButton.PADDING_LEFT + SelectorButton.PADDING_RIGHT
    fm = selector.fontMetrics()

    def width_for(label: str) -> int:
        if not label:
            return inset + 40
        width = fm.horizontalAdvance(label) + inset
        limit = width + 48
        while width <= limit:
            if (
                fm.elidedText(label, Qt.TextElideMode.ElideRight, width - inset)
                == label
            ):
                return width
            width += 1
        return limit

    width = max(width_for(label) for label in label_list)
    min_width = selector.property(SETTINGS_SELECTOR_MIN_WIDTH_PROP)
    if isinstance(min_width, int) and min_width > 0:
        width = max(width, min_width)
    selector.setFixedWidth(width)
    policy = selector.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
    selector.setSizePolicy(policy)


def register_settings_selector_width(selector: SelectorButton, *labels: str) -> None:
    """Remember menu labels and size the button to the widest one."""
    selector.setProperty(SETTINGS_SELECTOR_LABELS_PROP, list(labels))
    fit_settings_selector_width(selector, *labels)


def refit_settings_selector_width(selector: SelectorButton) -> None:
    """Recompute width from registered menu labels (or current text)."""
    stored = selector.property(SETTINGS_SELECTOR_LABELS_PROP)
    if isinstance(stored, list) and stored:
        fit_settings_selector_width(selector, *stored)
    elif selector.text():
        fit_settings_selector_width(selector, selector.text())


def schedule_settings_selector_refit(selector: SelectorButton) -> None:
    """Refit after the event loop runs so app fonts/layout are applied."""
    from PyQt6.QtCore import QTimer

    QTimer.singleShot(0, lambda s=selector: refit_settings_selector_width(s))


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


def make_settings_section_header_row(
    parent: QWidget,
    *,
    initial_tour_id: str,
    initial_area_display_name: str | None = None,
    icon_size: int = 20,
) -> tuple[QLabel, PageTourHelpButton, QLabel, QWidget]:
    """Right-pane section icon + title + guided tour ? button."""
    row_host = QWidget(parent)
    layout = QHBoxLayout(row_host)
    layout.setContentsMargins(0, 0, 12, 0)
    layout.setSpacing(8)

    icon_lbl = QLabel()
    icon_lbl.setProperty("class", "SectionHeaderIcon")
    icon_lbl.setProperty("icon_size", icon_size)
    icon_lbl.setFixedSize(icon_size + 2, icon_size + 2)
    layout.addWidget(icon_lbl, alignment=Qt.AlignmentFlag.AlignVCenter)

    title_lbl = QLabel("")
    title_lbl.setObjectName("ViewTitle")
    title_lbl.setProperty("class", "PageTitle")
    layout.addWidget(title_lbl)

    tour_btn = PageTourHelpButton(
        initial_tour_id,
        area_display_name=initial_area_display_name,
        parent=parent,
    )
    layout.addWidget(tour_btn, alignment=Qt.AlignmentFlag.AlignTop)
    layout.addStretch(1)

    return title_lbl, tour_btn, icon_lbl, row_host


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


def add_section_divider_to_layout(
    layout: QVBoxLayout, *, is_dark: bool = True
) -> SettingsSectionDivider:
    """Full-width horizontal rule separating major settings blocks."""
    divider = SettingsSectionDivider(is_dark=is_dark)
    layout.addWidget(divider)
    return divider


def make_settings_hint(text: str) -> QLabel:
    """Muted body copy for settings sections (avoids bold #SettingsFormContainer QLabel)."""
    hint = QLabel(text)
    hint.setWordWrap(True)
    hint.setObjectName("SettingsHint")
    return hint


def make_settings_action_status_label() -> QLabel:
    """Transient feedback line below a settings action button."""
    lbl = QLabel("")
    lbl.setObjectName("SettingsActionStatus")
    lbl.setWordWrap(True)
    return lbl


def make_settings_action_row(button: QPushButton) -> QWidget:
    """Left-aligned action button with trailing stretch."""
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(0)
    row_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignLeft)
    row_layout.addStretch(1)
    return row


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
