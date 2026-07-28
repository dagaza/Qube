"""General application settings section."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from core import app_settings as _general_settings
from core.ui_language import (
    UI_LANGUAGE_DESCRIPTIONS,
    UI_LANGUAGE_LABELS,
    UiLanguage,
    tr,
)
from ui.components.selector_button import SelectorButton
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_section_reset_footer,
    add_settings_card_form,
    add_subsection_to_form,
    add_settings_full_width_row,
    prepare_settings_card_form,
    register_settings_selector_width,
    schedule_settings_selector_refit,
)


def build_section(host, *, is_dark: bool) -> QWidget:
    general_widget = QWidget()
    general_widget.setObjectName("SettingsFormContainer")
    general_layout = QVBoxLayout(general_widget)
    general_layout.setContentsMargins(15, 0, 15, 10)
    general_layout.setSpacing(15)

    language_card, language_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    host.general_language_card = language_card
    language_form = add_settings_card_form(language_card_layout)
    language_heading = add_subsection_to_form(language_form, tr("Language"), anchor="language")
    language_heading.setToolTip(
        tr(
            "Choose British or American English spelling for labels, tooltips, "
            "and other on-screen text."
        )
    )

    language_row = QWidget()
    language_row_layout = QHBoxLayout(language_row)
    language_row_layout.setSpacing(16)
    host.ui_language_group = QButtonGroup(host)
    host.ui_language_group.setExclusive(True)
    current_language = _general_settings.get_ui_language()
    host.ui_language_cbs: dict[UiLanguage, QCheckBox] = {}
    for language_id in (UiLanguage.BRITISH, UiLanguage.AMERICAN):
        cb = QCheckBox(tr(UI_LANGUAGE_LABELS[language_id]))
        cb.setToolTip(tr(UI_LANGUAGE_DESCRIPTIONS[language_id]))
        cb.setProperty("ui_language_id", language_id.value)
        cb.setChecked(language_id == current_language)
        host.ui_language_group.addButton(cb)
        host.ui_language_cbs[language_id] = cb
        language_row_layout.addWidget(cb)
    host.ui_language_group.buttonToggled.connect(host._on_ui_language_toggled)
    language_row_layout.addStretch()
    add_settings_full_width_row(language_form, language_row)
    general_layout.addWidget(language_card)

    personal_card, personal_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    personal_form_host, personal_form = prepare_settings_card_form(personal_card_layout)
    add_subsection_to_form(personal_form, "Personalization", anchor="personalization")

    host.profile_units_selector = SelectorButton("Use inferred units", is_dark=is_dark)
    register_settings_selector_width(
        host.profile_units_selector,
        "Use inferred units",
        "Metric",
        "Imperial",
    )
    host.profile_units_selector.setMenu(QMenu(host.profile_units_selector))
    host.profile_units_selector.setToolTip(
        "Default measurement units for weather and other numeric answers. "
        "Unset lets Qube learn units from conversation."
    )
    profile_units_row = QWidget()
    profile_units_layout = QHBoxLayout(profile_units_row)
    profile_units_layout.setContentsMargins(0, 0, 0, 0)
    profile_units_lbl = QLabel("Default units")
    profile_units_lbl.setToolTip(host.profile_units_selector.toolTip())
    profile_units_layout.addWidget(profile_units_lbl)
    profile_units_layout.addWidget(host.profile_units_selector)
    profile_units_layout.addStretch(1)
    add_settings_full_width_row(personal_form, profile_units_row)
    personal_card_layout.addWidget(personal_form_host)
    general_layout.addWidget(personal_card)

    host._build_profile_units_menu()
    host._sync_profile_units_selector()
    schedule_settings_selector_refit(host.profile_units_selector)

    add_section_reset_footer(general_layout, host, "general", is_dark=is_dark)

    return general_widget
