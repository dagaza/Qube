"""Settings card form layout width regression tests."""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget

from ui.components.selector_button import SelectorButton
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_field_row,
    add_subsection_to_form,
    make_settings_nested_form,
    prepare_settings_card_form,
    register_settings_selector_width,
    settings_layout_row,
    wrap_subsection,
    add_settings_full_width_row,
)


@pytest.mark.ui
def test_card_form_host_fills_section_card(_qube_app):
    host = QWidget()
    host._current_settings_section_id = "ai.models"
    host._settings_section_cards = []
    host._settings_collapsible_cards_by_section = {}

    page = QWidget()
    page.resize(800, 400)
    page_layout = QVBoxLayout(page)

    wrapper, card_layout = begin_settings_section_card(host, is_dark=True)
    form_host, form = prepare_settings_card_form(card_layout)
    add_subsection_to_form(form, "Engine & routing")
    selector = SelectorButton("Internal Engine (native)", is_dark=True)
    register_settings_selector_width(
        selector,
        "Internal Engine (native)",
        "External Server (Port 11434)",
    )
    add_settings_field_row(form, "AI Engine", selector)
    card_layout.addWidget(form_host)
    page_layout.addWidget(wrapper)

    page.show()
    QApplication.processEvents()

    card = host._settings_section_cards[0]
    assert form_host.width() >= int(card.width() * 0.9)


@pytest.mark.ui
def test_nested_local_models_row_expands(_qube_app):
    host = QWidget()
    host._current_settings_section_id = "ai.models"
    host._settings_section_cards = []
    host._settings_collapsible_cards_by_section = {}

    page = QWidget()
    page.resize(800, 500)
    page_layout = QVBoxLayout(page)

    wrapper, card_layout = begin_settings_section_card(host, is_dark=True)
    form_host, form = prepare_settings_card_form(card_layout)

    local_models_inner, local_models_form = make_settings_nested_form()
    local_row = QHBoxLayout()
    list_host = QWidget()
    list_host.setMinimumWidth(120)
    local_row.addWidget(list_host, stretch=1)
    local_models_form.addRow("On this device", settings_layout_row(local_row))
    add_settings_full_width_row(form, wrap_subsection(local_models_inner))

    card_layout.addWidget(form_host)
    page_layout.addWidget(wrapper)

    page.show()
    QApplication.processEvents()

    assert local_models_inner.width() >= int(form_host.width() * 0.9)
