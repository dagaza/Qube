"""Custom knowledge sources settings section."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from core.knowledge.configured_sources import (
    ConfiguredSource,
    delete_configured_source,
    list_configured_sources,
    save_configured_source,
    test_configured_source,
)
from core.knowledge.connectors.base import list_connector_types
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE
from ui.components.brand_buttons import apply_brand_primary
from ui.components.selector_button import SelectorButton
from ui.views.settings.knowledge_list_table import (
    apply_borderless_list_table_theme,
    configure_borderless_list_table,
    populate_table_rows,
    selected_data_row,
)
from ui.views.settings.widgets import (
    add_subsection_to_layout,
    register_settings_selector_width,
    schedule_settings_selector_refit,
    wrap_subsection,
)

_CUSTOM_SOURCES_PLACEHOLDER = (
    "Custom sources you add will appear here."
)
_DEFAULT_CONNECTOR_ID = "rest_json"


def build_knowledge_custom_sources_section(host, *, is_dark: bool) -> QWidget:
    container = QWidget()
    container.setMinimumWidth(0)
    container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    add_subsection_to_layout(layout, "Custom sources", anchor="knowledge_custom_sources")

    inner = QWidget()
    outer = QVBoxLayout(inner)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(10)

    form_host = QWidget()
    form = QFormLayout(form_host)
    form.setSpacing(10)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

    host.custom_source_id_input = QLineEdit()
    host.custom_source_label_input = QLineEdit()
    connector_types = list_connector_types()
    default_connector = (
        _DEFAULT_CONNECTOR_ID
        if _DEFAULT_CONNECTOR_ID in connector_types
        else (connector_types[0] if connector_types else _DEFAULT_CONNECTOR_ID)
    )
    host._custom_source_connector_id = default_connector
    host.custom_source_connector_selector = SelectorButton(default_connector, is_dark=is_dark)
    host.custom_source_connector_selector.setMenu(
        QMenu(host.custom_source_connector_selector)
    )
    register_settings_selector_width(
        host.custom_source_connector_selector,
        *connector_types,
    )
    host.custom_source_base_url_input = QLineEdit()
    host.custom_source_search_path_input = QLineEdit()
    host.custom_source_search_path_input.setPlaceholderText("/api/search?q={query}")

    form.addRow("Source id", host.custom_source_id_input)
    form.addRow("Label", host.custom_source_label_input)
    form.addRow("Connector", host.custom_source_connector_selector)
    form.addRow("Base URL", host.custom_source_base_url_input)
    form.addRow("Search path", host.custom_source_search_path_input)
    outer.addWidget(form_host)

    btn_row = QHBoxLayout()
    host.custom_source_save_btn = QPushButton("Save source")
    host.custom_source_test_btn = QPushButton("Test")
    host.custom_source_delete_btn = QPushButton("Delete selected")
    apply_brand_primary(host.custom_source_save_btn)
    btn_row.addWidget(host.custom_source_save_btn)
    btn_row.addWidget(host.custom_source_test_btn)
    btn_row.addWidget(host.custom_source_delete_btn)
    btn_row.addStretch(1)
    outer.addLayout(btn_row)

    host.custom_source_status_label = QLabel()
    host.custom_source_status_label.setWordWrap(True)
    host.custom_source_status_label.setObjectName("SettingsActionStatus")
    outer.addWidget(host.custom_source_status_label)

    host.custom_sources_table = QTableWidget()
    configure_borderless_list_table(
        host.custom_sources_table,
        columns=("Label", "Source id", "Connector"),
        object_name="KnowledgeCustomSourcesTable",
    )
    apply_borderless_list_table_theme(host.custom_sources_table, is_dark=is_dark)
    outer.addWidget(host.custom_sources_table)

    host.custom_source_save_btn.clicked.connect(host._save_custom_source)
    host.custom_source_test_btn.clicked.connect(host._test_custom_source)
    host.custom_source_delete_btn.clicked.connect(host._delete_custom_source)

    host._build_custom_source_connector_menu()
    schedule_settings_selector_refit(host.custom_source_connector_selector)

    layout.addWidget(wrap_subsection(inner, anchor="knowledge_custom_sources"))
    _refresh_custom_sources_list(host, is_dark=is_dark)
    return container


def _refresh_custom_sources_list(host, *, is_dark: bool = True) -> None:
    rows = [
        (source.label, source.id, source.connector_type)
        for source in list_configured_sources()
    ]
    populate_table_rows(
        host.custom_sources_table,
        rows=rows,
        placeholder=_CUSTOM_SOURCES_PLACEHOLDER,
        is_dark=is_dark,
    )


def _source_from_host(host) -> ConfiguredSource:
    connector = getattr(host, "_custom_source_connector_id", _DEFAULT_CONNECTOR_ID)
    source_id = host.custom_source_id_input.text().strip().lower()
    return ConfiguredSource(
        id=source_id,
        label=host.custom_source_label_input.text().strip(),
        connector_type=str(connector or _DEFAULT_CONNECTOR_ID),
        knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
        config={
            "base_url": host.custom_source_base_url_input.text().strip(),
            "search_path": host.custom_source_search_path_input.text().strip() or "/?q={query}",
            "method": "GET",
            "adapter_id": source_id,
            "response_mapping": {
                "items_path": "$",
                "title": "$.title",
                "snippet": "$.description",
                "url": "$.url",
            },
        },
        auth={"type": "bearer", "credential_ref": source_id},
    )


def save_custom_source_from_host(host) -> None:
    source = _source_from_host(host)
    save_configured_source(source)
    host.custom_source_status_label.setText(f"Saved source {source.id}")
    is_dark = getattr(host.window(), "_is_dark_theme", True)
    _refresh_custom_sources_list(host, is_dark=is_dark)


def test_custom_source_from_host(host) -> None:
    source = _source_from_host(host)
    ok, message = test_configured_source(source)
    host.custom_source_status_label.setText(message if ok else f"Test failed: {message}")


def delete_custom_source_from_host(host) -> None:
    row = selected_data_row(host.custom_sources_table)
    sources = list_configured_sources()
    if row is None or row >= len(sources):
        return
    delete_configured_source(sources[row].id)
    is_dark = getattr(host.window(), "_is_dark_theme", True)
    _refresh_custom_sources_list(host, is_dark=is_dark)
