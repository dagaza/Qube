"""Settings card form layout width regression tests."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.components.selector_button import SelectorButton
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_field_row,
    add_subsection_to_form,
    make_settings_nested_form,
    prepare_settings_card_form,
    prepare_settings_wrapped_label,
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


@pytest.mark.ui
def test_advanced_json_card_stays_compact(_qube_app):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "advanced_section",
        "ui/views/settings/sections/advanced.py",
    )
    mod = importlib.util.module_from_spec(spec)
    import ui.views.settings.settings_card_style  # noqa: F401

    spec.loader.exec_module(mod)

    host = type("Host", (), {})()
    host._current_settings_section_id = "advanced"
    host._settings_section_cards = []
    host._settings_collapsible_cards_by_section = {}
    host._on_open_settings_json_clicked = lambda: None

    page = QWidget()
    page.resize(900, 700)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    content = mod.build_section(host, is_dark=True)
    scroll.setWidget(content)
    page_layout = QVBoxLayout(page)
    page_layout.addWidget(scroll)

    page.show()
    QApplication.processEvents()

    card = host._settings_section_cards[0]
    form_host = card.layout().itemAt(0).widget()
    hint = host.settings_json_hint_lbl

    assert hint.height() <= hint.sizeHint().height() + 4
    assert card.height() < 400


@pytest.mark.ui
def test_nested_model_storage_label_is_not_vertically_collapsed(_qube_app):
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
    models_dir_label = QLabel(
        "/Users/example/Library/Application Support/Qube/models/llm"
    )
    prepare_settings_wrapped_label(models_dir_label)
    add_settings_field_row(local_models_form, "Model storage", models_dir_label)
    add_settings_full_width_row(form, wrap_subsection(local_models_inner))

    card_layout.addWidget(form_host)
    page_layout.addWidget(wrapper)

    page.show()
    QApplication.processEvents()

    line_height = QFontMetrics(models_dir_label.font()).lineSpacing()
    assert models_dir_label.height() >= line_height
    assert models_dir_label.sizeHint().height() >= line_height


@pytest.mark.ui
def test_web_discovery_advanced_hint_spans_nested_form(_qube_app):
    from PyQt6.QtWidgets import QSpinBox

    from ui.views.settings.primitives.typography import make_settings_hint
    from ui.views.settings.widgets import add_settings_span_row

    page = QWidget()
    page.resize(900, 400)
    page_layout = QVBoxLayout(page)

    panel, form = make_settings_nested_form()
    form.setContentsMargins(16, 0, 0, 0)
    add_settings_field_row(form, "Session limit override", QSpinBox())
    burst_hint = make_settings_hint(
        "Burst limit is fixed at 6 live queries per 10 minutes. "
        "Lowering the session limit is always allowed; raising above the "
        "default (30) requires confirmation."
    )
    add_settings_span_row(form, burst_hint)
    page_layout.addWidget(panel)

    page.show()
    QApplication.processEvents()

    assert burst_hint.width() >= int(page.width() * 0.55)
    assert "requires confirmation." in burst_hint.text()


@pytest.mark.ui
def test_web_discovery_searxng_url_row_expands(_qube_app):
    from PyQt6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QSizePolicy

    page = QWidget()
    page.resize(900, 200)
    page_layout = QVBoxLayout(page)

    form_host, form = make_settings_nested_form()
    url_field = QLineEdit()
    url_field.setMinimumWidth(0)
    url_field.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Fixed,
    )
    row_layout = QHBoxLayout()
    row_layout.addWidget(url_field, stretch=1)
    row_layout.addWidget(QPushButton("Set up SearXNG…"))
    add_settings_field_row(form, "SearXNG base URL", settings_layout_row(row_layout))
    page_layout.addWidget(form_host)

    page.show()
    QApplication.processEvents()

    assert url_field.width() >= int(page.width() * 0.35)


@pytest.mark.ui
def test_backup_interval_selector_fits_widest_label(_qube_app):
    from ui.components.selector_button import SelectorButton
    from ui.views.settings.widgets import register_settings_selector_width

    selector = SelectorButton("Every 30 days", is_dark=True)
    labels = ["Every 7 days", "Every 14 days", "Every 30 days", "Every 90 days"]
    register_settings_selector_width(selector, *labels)

    page = QWidget()
    page.resize(900, 120)
    page_layout = QVBoxLayout(page)
    page_layout.addWidget(selector)
    page.show()
    QApplication.processEvents()

    fm = selector.fontMetrics()
    widest = max(fm.horizontalAdvance(label) for label in labels)
    assert selector.width() >= widest
    assert "Every 90 days" == fm.elidedText(
        "Every 90 days",
        Qt.TextElideMode.ElideRight,
        selector.width() - SelectorButton.PADDING_LEFT - SelectorButton.PADDING_RIGHT,
    )
