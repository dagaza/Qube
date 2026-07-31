"""Reusable settings UI widgets."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
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
from ui.views.settings.settings_card_style import SETTINGS_CARD_FORM_HORIZONTAL_INSET
from ui.views.settings.settings_theme import resolve_settings_theme, settings_divider_color
from core.theme.color_utils import theme_qcolor

SETTINGS_SECTION_RESET_BUTTON_TEXT = "Reset to default configuration"


def _apply_settings_control_tooltip(widget: QWidget, tooltip: str) -> None:
    """Ensure QubeApplication tooltip routing receives hover events on ``widget``."""
    widget.setToolTip(tooltip)
    widget.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
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
) -> tuple[QLabel, PageTourHelpButton, QLabel, QWidget, QPushButton]:
    """Right-pane section icon + title + guided tour ? + collapse-all control."""
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

    collapse_all_btn = QPushButton()
    collapse_all_btn.setObjectName("SettingsSectionCollapseAllButton")
    collapse_all_btn.setFixedSize(30, 30)
    collapse_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    collapse_all_btn.setStyleSheet("background: transparent; border: none;")
    collapse_all_btn.setToolTip("Collapse all sections on this page")
    collapse_all_btn.hide()
    layout.addWidget(collapse_all_btn, alignment=Qt.AlignmentFlag.AlignVCenter)

    layout.addStretch(1)

    return title_lbl, tour_btn, icon_lbl, row_host, collapse_all_btn


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
        add_settings_full_width_row(layout, row)
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
        theme = resolve_settings_theme(is_dark=self._is_dark)
        color = theme_qcolor(settings_divider_color(theme))
        painter = QPainter(self)
        painter.fillRect(0, self._MARGIN_TOP, self.width(), self._LINE_HEIGHT, color)


def _prepare_settings_form_row_widget(widget: QWidget) -> None:
    policy = widget.sizePolicy()
    if policy.horizontalPolicy() != QSizePolicy.Policy.Fixed:
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, policy.verticalPolicy())
    widget.setMinimumWidth(0)


def _install_form_label_column_ruler(form: QFormLayout) -> None:
    """Reserve label-column width on hint-only forms (matches labeled settings cards)."""
    ruler = QLabel("Context limit")
    ruler.setObjectName("SettingsFormLabelColumnRuler")
    ruler.setFixedHeight(0)
    ruler.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    field = QWidget()
    field.setFixedHeight(0)
    field.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    form.addRow(ruler, field)


def make_settings_form() -> tuple[QWidget, QFormLayout]:
    """Standard settings QFormLayout host (label column + field column rhythm)."""
    form_host = QWidget()
    form_host.setMinimumWidth(0)
    form_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    form = QFormLayout(form_host)
    form.setSpacing(15)
    form.setHorizontalSpacing(12)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    inset = SETTINGS_CARD_FORM_HORIZONTAL_INSET
    form.setContentsMargins(inset, 0, inset, 0)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    _install_form_label_column_ruler(form)
    return form_host, form


def add_settings_full_width_row(form: QFormLayout, widget: QWidget) -> None:
    """Add a row in the field column that expands to the form's right margin."""
    _prepare_settings_form_row_widget(widget)
    form.addRow("", widget)


def add_settings_span_row(form: QFormLayout, widget: QWidget) -> None:
    """Span both form columns (nested sub-forms and inner mini-form descriptions)."""
    _prepare_settings_form_row_widget(widget)
    form.addRow(widget)


def add_settings_field_column_row(form: QFormLayout, widget: QWidget) -> None:
    """Add a control aligned with labeled field rows (toggle/disclosure rows)."""
    add_settings_full_width_row(form, widget)


def add_settings_field_row(
    form: QFormLayout, label: str | QWidget, field: QWidget
) -> None:
    """Add a classic label + field row (controls align to the shared field column)."""
    form.addRow(label, field)


def prepare_settings_card_form(
    card_layout: QVBoxLayout,
) -> tuple[QWidget, QFormLayout]:
    """Create a card form wired for collapse headers before ``addWidget(form_host)``."""
    from ui.views.settings.settings_card_style import resolve_collapsible_handle_for_layout

    form_host, form = make_settings_form()
    handle = resolve_collapsible_handle_for_layout(card_layout)
    if handle is not None:
        form._settings_collapsible_handle = handle  # type: ignore[attr-defined]
    return form_host, form


def add_settings_card_form(card_layout: QVBoxLayout) -> QFormLayout:
    """Attach a settings form to a section card and return its layout."""
    form_host, form = prepare_settings_card_form(card_layout)
    card_layout.addWidget(form_host)
    return form


def add_settings_card_row(card_layout: QVBoxLayout, widget: QWidget) -> None:
    """Add one full-width row to a card with the same inset as form field rows."""
    form_host, form = make_settings_form()
    add_settings_full_width_row(form, widget)
    card_layout.addWidget(form_host)


def settings_layout_row(layout: QHBoxLayout | QVBoxLayout) -> QWidget:
    """Wrap a row/column layout for use in a settings card form."""
    host = QWidget()
    host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    host.setLayout(layout)
    return host


def make_subsection_label(text: str, *, anchor: str | None = None) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("SettingsSubsectionLabel")
    if anchor:
        lbl.setProperty("settings_anchor", anchor)
    return lbl


def add_subsection_to_form(
    form: QFormLayout, text: str, *, anchor: str | None = None
) -> QLabel:
    from ui.views.settings.settings_card_style import (
        apply_collapsible_card_title,
        resolve_collapsible_handle_for_form,
    )

    handle = resolve_collapsible_handle_for_form(form)
    if handle is not None:
        return apply_collapsible_card_title(handle, text, anchor=anchor)
    lbl = make_subsection_label(text, anchor=anchor)
    form.addRow(lbl)
    return lbl


def add_subsection_to_layout(
    layout: QVBoxLayout, text: str, *, anchor: str | None = None
) -> QLabel:
    from ui.views.settings.settings_card_style import (
        apply_collapsible_card_title,
        resolve_collapsible_handle_for_layout,
    )

    handle = resolve_collapsible_handle_for_layout(layout)
    if handle is not None:
        return apply_collapsible_card_title(handle, text, anchor=anchor)
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
    hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    hint.setMinimumWidth(0)
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
    wrapper.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
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
    _apply_settings_control_tooltip(toggle, tooltip)
    label = QLabel(label_text)
    _apply_settings_control_tooltip(label, tooltip)
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


def make_pro_feature_toggle_row(
    host,
    *,
    label: str,
    tooltip: str,
    feature_id: str,
    checked: bool,
    on_toggled,
    info_attr: str | None = None,
) -> tuple[QWidget, QLabel]:
    """Toggle row for a Pro-gated Library feature."""
    from core.capabilities import has_feature

    toggle = PrestigeToggle()
    _apply_settings_control_tooltip(toggle, tooltip)
    label_widget = QLabel(label)
    _apply_settings_control_tooltip(label_widget, tooltip)
    info_btn = host._make_settings_info_button(tooltip)
    if info_attr:
        setattr(host, info_attr, info_btn)
    label_cluster = QWidget()
    _apply_settings_control_tooltip(label_cluster, tooltip)
    label_cluster_layout = QHBoxLayout(label_cluster)
    label_cluster_layout.setContentsMargins(0, 0, 0, 0)
    label_cluster_layout.setSpacing(6)
    label_cluster_layout.addWidget(label_widget)
    label_cluster_layout.addWidget(info_btn)
    row = QWidget()
    _apply_settings_control_tooltip(row, tooltip)
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(8)
    row_layout.addWidget(toggle, alignment=Qt.AlignmentFlag.AlignLeft)
    row_layout.addWidget(label_cluster)
    row_layout.addStretch(1)

    licensed = has_feature(feature_id)
    effective_checked = bool(checked and licensed)
    toggle.blockSignals(True)
    toggle.setChecked(effective_checked)
    toggle.blockSignals(False)
    toggle.toggled.connect(on_toggled)
    toggle.setProperty("pro_feature_id", feature_id)
    row.setProperty("pro_feature_id", feature_id)
    return row, label_widget


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


def track_internal_ai_label(host, label: QLabel | None) -> None:
    if label is None:
        return
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
