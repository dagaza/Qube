"""General application settings section."""

from __future__ import annotations

from PyQt6.QtWidgets import QButtonGroup, QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from core import app_settings as _general_settings
from core.ui_language import (
    UI_LANGUAGE_DESCRIPTIONS,
    UI_LANGUAGE_LABELS,
    UiLanguage,
    tr,
)
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import add_section_reset_footer, add_subsection_to_layout


def build_section(host, *, is_dark: bool) -> QWidget:
    general_widget = QWidget()
    general_widget.setObjectName("SettingsFormContainer")
    general_layout = QVBoxLayout(general_widget)
    general_layout.setContentsMargins(15, 0, 15, 10)
    general_layout.setSpacing(15)

    language_card, language_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    host.general_language_card = language_card
    add_subsection_to_layout(language_card_layout, tr("Language"))

    language_lbl = QLabel(tr("Application language"))
    language_lbl.setObjectName("SettingsSubsectionLabel")
    language_lbl.setToolTip(
        tr(
            "Choose British or American English spelling for labels, tooltips, "
            "and other on-screen text."
        )
    )
    host._ui_language_lbl = language_lbl
    language_card_layout.addWidget(language_lbl)

    language_row = QHBoxLayout()
    language_row.setSpacing(16)
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
        language_row.addWidget(cb)
    host.ui_language_group.buttonToggled.connect(host._on_ui_language_toggled)
    language_row.addStretch()
    language_card_layout.addLayout(language_row)
    general_layout.addWidget(language_card)

    add_section_reset_footer(general_layout, host, "general", is_dark=is_dark)

    return general_widget
