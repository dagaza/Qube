"""Custom knowledge sources settings section."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, QFileSystemWatcher
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
    clear_configured_source_search_cache,
    delete_configured_source,
    list_configured_sources,
    load_configured_source,
    save_configured_source,
    sources_dir,
    test_configured_source,
)
from core.knowledge.custom_source_editor import (
    build_configured_source_from_fields,
    configured_source_to_field_values,
)
from core.knowledge.connectors.base import list_connector_types
from ui.components.brand_buttons import apply_brand_primary
from ui.components.selector_button import SelectorButton
from ui.views.settings.knowledge_list_table import (
    apply_borderless_list_table_theme,
    configure_borderless_list_table,
    populate_table_rows,
    selected_data_row,
)
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_card_form,
    add_subsection_to_form,
    register_settings_selector_width,
    schedule_settings_selector_refit,
    wrap_subsection,
    add_settings_full_width_row,
    add_settings_span_row,
)

_CUSTOM_SOURCES_PLACEHOLDER = (
    "Custom sources you add will appear here."
)
_DEFAULT_CONNECTOR_ID = "rest_json"
_MCP_CONNECTOR_ID = "mcp"
_SOURCES_RELOAD_MS = 400
_MCP_PROVIDER_ID = "mcp"


def build_knowledge_custom_sources_section(host, *, is_dark: bool) -> QWidget:
    card, card_layout = begin_settings_section_card(host, is_dark=is_dark)
    card_form = add_settings_card_form(card_layout)
    add_subsection_to_form(card_form, "Custom sources", anchor="knowledge_custom_sources")

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
    host._custom_source_default_connector_id = default_connector
    host._custom_source_connector_id = default_connector
    host._custom_source_selected_id = ""
    host._custom_source_loaded: ConfiguredSource | None = None
    host.custom_source_connector_selector = SelectorButton(default_connector, is_dark=is_dark)
    host.custom_source_connector_selector.setMenu(
        QMenu(host.custom_source_connector_selector)
    )
    register_settings_selector_width(
        host.custom_source_connector_selector,
        *connector_types,
    )

    host.custom_source_rest_group = QWidget()
    rest_form = QFormLayout(host.custom_source_rest_group)
    rest_form.setContentsMargins(0, 0, 0, 0)
    rest_form.setSpacing(10)
    host.custom_source_base_url_input = QLineEdit()
    host.custom_source_search_path_input = QLineEdit()
    host.custom_source_search_path_input.setPlaceholderText("/api/search?q={query}")
    rest_form.addRow("Base URL", host.custom_source_base_url_input)
    rest_form.addRow("Search path", host.custom_source_search_path_input)

    host.custom_source_mcp_group = QWidget()
    mcp_form = QFormLayout(host.custom_source_mcp_group)
    mcp_form.setContentsMargins(0, 0, 0, 0)
    mcp_form.setSpacing(10)
    host.custom_source_mcp_command_input = QLineEdit()
    host.custom_source_mcp_command_input.setPlaceholderText(
        '["mcp-server-filesystem.cmd", "C:\\\\Data"]'
    )
    host.custom_source_mcp_namespace_input = QLineEdit()
    host.custom_source_mcp_namespace_input.setPlaceholderText("filesystem")
    host.custom_source_mcp_namespace_input.setToolTip(
        "Use namespace `filesystem` for the official MCP Filesystem server. "
        "Requires a Qube Pro license."
    )
    host.custom_source_mcp_command_input.setToolTip(
        "JSON array launching the MCP server, e.g. "
        '["mcp-server-filesystem", "/path/to/folder"]. '
        "Filesystem MCP requires a Qube Pro license."
    )
    host.custom_source_mcp_tool_input = QLineEdit()
    host.custom_source_mcp_tool_input.setPlaceholderText("search_files")
    mcp_form.addRow("Command", host.custom_source_mcp_command_input)
    mcp_form.addRow("Namespace", host.custom_source_mcp_namespace_input)
    mcp_form.addRow("Tool name", host.custom_source_mcp_tool_input)

    form.addRow("Source id", host.custom_source_id_input)
    form.addRow("Label", host.custom_source_label_input)
    form.addRow("Connector", host.custom_source_connector_selector)
    form.addRow(host.custom_source_rest_group)
    form.addRow(host.custom_source_mcp_group)
    outer.addWidget(form_host)

    btn_row = QHBoxLayout()
    host.custom_source_new_btn = QPushButton("New source")
    host.custom_source_save_btn = QPushButton("Save source")
    host.custom_source_test_btn = QPushButton("Test")
    host.custom_source_delete_btn = QPushButton("Delete selected")
    apply_brand_primary(host.custom_source_save_btn)
    btn_row.addWidget(host.custom_source_new_btn)
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

    host.custom_source_new_btn.clicked.connect(host._new_custom_source)
    host.custom_source_save_btn.clicked.connect(host._save_custom_source)
    host.custom_source_test_btn.clicked.connect(host._test_custom_source)
    host.custom_source_delete_btn.clicked.connect(host._delete_custom_source)
    host.custom_sources_table.itemSelectionChanged.connect(
        lambda: _on_custom_source_row_selected(host)
    )

    host._build_custom_source_connector_menu()
    schedule_settings_selector_refit(host.custom_source_connector_selector)
    sync_custom_source_connector_fields(host)

    add_settings_span_row(card_form, wrap_subsection(inner, anchor="knowledge_custom_sources"))
    setup_custom_sources_dir_watcher(host)
    _refresh_custom_sources_list(host, is_dark=is_dark)
    return card


def _mcp_namespace_for_source(source: ConfiguredSource) -> str:
    cfg = dict(source.config or {})
    return str(cfg.get("namespace") or cfg.get("adapter_id") or source.id).strip().lower()


def _refresh_integrations_consent_if_available(host) -> None:
    if not hasattr(host, "integrations_consent_layout"):
        return
    from ui.views.settings.sections.integrations import sync_integrations_consent_panel

    is_dark = getattr(host.window(), "_is_dark_theme", True)
    sync_integrations_consent_panel(host, is_dark=is_dark)


def _discover_mcp_source_if_applicable(host, source: ConfiguredSource) -> str | None:
    if source.connector_type != _MCP_CONNECTOR_ID:
        return None
    from core.integrations.mcp_discovery import discover_and_cache_mcp_source

    namespace = _mcp_namespace_for_source(source)
    result = discover_and_cache_mcp_source(dict(source.config or {}), namespace=namespace)
    if result.error:
        return result.error
    _maybe_open_grant_review_dialog(host, source, result)
    return None


def _maybe_open_grant_review_dialog(host, source: ConfiguredSource, result) -> None:
    if result.error:
        return
    if not result.first_connect and result.drift is None:
        return
    from ui.components.capability_grant_review_dialog import (
        open_capability_grant_review_dialog,
    )

    parent = host.window() if hasattr(host, "window") else None
    if parent is None:
        return
    is_dark = getattr(parent, "_is_dark_theme", True)
    namespace = _mcp_namespace_for_source(source)
    saved = open_capability_grant_review_dialog(
        parent,
        server_label=source.label or source.id,
        namespace=namespace,
        result=result,
        is_dark=is_dark,
    )
    if saved:
        _refresh_integrations_consent_if_available(host)


def setup_custom_sources_dir_watcher(host) -> None:
    """Watch ``knowledge/sources`` for files added or edited outside the app."""
    timer = QTimer(host)
    timer.setSingleShot(True)
    timer.setInterval(_SOURCES_RELOAD_MS)
    timer.timeout.connect(lambda: _reload_custom_sources_from_disk(host))
    host._custom_sources_reload_timer = timer

    watcher = QFileSystemWatcher(host)
    watcher.directoryChanged.connect(lambda _path: timer.start())
    host._custom_sources_watcher = watcher

    directory = str(sources_dir())
    if directory not in watcher.directories():
        watcher.addPath(directory)


def sync_custom_source_connector_fields(host) -> None:
    """Show connector-specific fields and hide the rest."""
    connector = getattr(host, "_custom_source_connector_id", _DEFAULT_CONNECTOR_ID)
    is_mcp = connector == _MCP_CONNECTOR_ID
    if hasattr(host, "custom_source_rest_group"):
        host.custom_source_rest_group.setVisible(not is_mcp)
    if hasattr(host, "custom_source_mcp_group"):
        host.custom_source_mcp_group.setVisible(is_mcp)


def reset_custom_source_form(host) -> None:
    """Clear the editor for creating a new source (does not delete on disk)."""
    host._custom_source_selected_id = ""
    host._custom_source_loaded = None
    default_connector = getattr(
        host,
        "_custom_source_default_connector_id",
        _DEFAULT_CONNECTOR_ID,
    )
    host._custom_source_connector_id = default_connector

    host.custom_source_id_input.clear()
    host.custom_source_label_input.clear()
    host.custom_source_base_url_input.clear()
    host.custom_source_search_path_input.clear()
    host.custom_source_mcp_command_input.clear()
    host.custom_source_mcp_namespace_input.clear()
    host.custom_source_mcp_tool_input.clear()

    if hasattr(host, "_sync_custom_source_connector_selector"):
        host._sync_custom_source_connector_selector()
    elif hasattr(host, "custom_source_connector_selector"):
        host.custom_source_connector_selector.setText(default_connector)
    sync_custom_source_connector_fields(host)

    host.custom_sources_table.blockSignals(True)
    host.custom_sources_table.clearSelection()
    host.custom_sources_table.blockSignals(False)


def new_custom_source_from_host(host) -> None:
    reset_custom_source_form(host)
    host.custom_source_status_label.setText("Enter details for a new source, then Save.")
    host.custom_source_id_input.setFocus()


def apply_configured_source_to_host(host, source: ConfiguredSource) -> None:
    """Populate the editor from a configured source loaded from disk."""
    host._custom_source_loaded = source
    host._custom_source_selected_id = source.id
    values = configured_source_to_field_values(source)
    host.custom_source_id_input.setText(values["id"])
    host.custom_source_label_input.setText(values["label"])
    host._custom_source_connector_id = values["connector_type"]
    if hasattr(host, "_sync_custom_source_connector_selector"):
        host._sync_custom_source_connector_selector()
    else:
        host.custom_source_connector_selector.setText(values["connector_type"])
    sync_custom_source_connector_fields(host)

    host.custom_source_base_url_input.setText(values["base_url"])
    host.custom_source_search_path_input.setText(values["search_path"])
    host.custom_source_mcp_command_input.setText(values["mcp_command"])
    host.custom_source_mcp_namespace_input.setText(values["mcp_namespace"])
    host.custom_source_mcp_tool_input.setText(values["mcp_tool_name"])


def configured_source_from_host(host) -> ConfiguredSource:
    """Build a :class:`ConfiguredSource` from the editor, preserving non-REST config."""
    loaded: ConfiguredSource | None = getattr(host, "_custom_source_loaded", None)
    source_id = host.custom_source_id_input.text().strip().lower()
    if loaded is not None and loaded.id != source_id:
        loaded = None
    return build_configured_source_from_fields(
        source_id=host.custom_source_id_input.text(),
        label=host.custom_source_label_input.text(),
        connector_type=getattr(host, "_custom_source_connector_id", _DEFAULT_CONNECTOR_ID),
        base_url=host.custom_source_base_url_input.text(),
        search_path=host.custom_source_search_path_input.text(),
        mcp_command=host.custom_source_mcp_command_input.text(),
        mcp_namespace=host.custom_source_mcp_namespace_input.text(),
        mcp_tool_name=host.custom_source_mcp_tool_input.text(),
        loaded=loaded,
    )


def _on_custom_source_row_selected(host) -> None:
    row = selected_data_row(host.custom_sources_table)
    sources = list_configured_sources()
    if row is None or row >= len(sources):
        return
    apply_configured_source_to_host(host, sources[row])


def _reload_custom_sources_from_disk(host) -> None:
    from core.integrations.descriptor_cache import reconcile_mcp_integration_state

    clear_configured_source_search_cache()
    reconcile_mcp_integration_state()
    is_dark = getattr(host.window(), "_is_dark_theme", True)
    selected_id = getattr(host, "_custom_source_selected_id", "")
    _refresh_custom_sources_list(host, is_dark=is_dark, preserve_selection=True)
    _refresh_integrations_consent_if_available(host)
    if not selected_id:
        return
    source = load_configured_source(selected_id)
    if source is None:
        host._custom_source_selected_id = ""
        host._custom_source_loaded = None
        return
    apply_configured_source_to_host(host, source)


def _refresh_custom_sources_list(
    host,
    *,
    is_dark: bool = True,
    preserve_selection: bool = True,
) -> None:
    selected_id = getattr(host, "_custom_source_selected_id", "") if preserve_selection else ""
    sources = list_configured_sources()
    rows = [
        (source.label, source.id, source.connector_type)
        for source in sources
    ]
    populate_table_rows(
        host.custom_sources_table,
        rows=rows,
        placeholder=_CUSTOM_SOURCES_PLACEHOLDER,
        is_dark=is_dark,
    )
    if not selected_id:
        return
    for idx, source in enumerate(sources):
        if source.id == selected_id:
            host.custom_sources_table.blockSignals(True)
            host.custom_sources_table.selectRow(idx)
            host.custom_sources_table.blockSignals(False)
            break


def save_custom_source_from_host(host) -> None:
    source = configured_source_from_host(host)
    from core.mcp_filesystem_pro_features import require_pro_mcp_filesystem_for_source

    require_pro_mcp_filesystem_for_source(source)
    save_configured_source(source)
    host._custom_source_selected_id = source.id
    host._custom_source_loaded = source
    discover_error = _discover_mcp_source_if_applicable(host, source)
    if discover_error:
        host.custom_source_status_label.setText(
            f"Saved source {source.id}, but MCP discovery failed: {discover_error}"
        )
    else:
        host.custom_source_status_label.setText(f"Saved source {source.id}")
    is_dark = getattr(host.window(), "_is_dark_theme", True)
    _refresh_custom_sources_list(host, is_dark=is_dark)
    _refresh_integrations_consent_if_available(host)


def test_custom_source_from_host(host) -> None:
    source = configured_source_from_host(host)
    from core.mcp_filesystem_pro_features import require_pro_mcp_filesystem_for_source

    require_pro_mcp_filesystem_for_source(source)
    if source.connector_type == _MCP_CONNECTOR_ID:
        from core.integrations.mcp_discovery import discover_and_cache_mcp_source

        namespace = _mcp_namespace_for_source(source)
        result = discover_and_cache_mcp_source(dict(source.config or {}), namespace=namespace)
        if result.error:
            host.custom_source_status_label.setText(f"Test failed: {result.error}")
            return
        host.custom_source_status_label.setText(
            f"OK — MCP server responded ({result.count} capabilities registered for Integrations)"
        )
        _maybe_open_grant_review_dialog(host, source, result)
        _refresh_integrations_consent_if_available(host)
        return

    ok, message = test_configured_source(source)
    host.custom_source_status_label.setText(message if ok else f"Test failed: {message}")
    if ok:
        _refresh_integrations_consent_if_available(host)


def delete_custom_source_from_host(host) -> None:
    row = selected_data_row(host.custom_sources_table)
    sources = list_configured_sources()
    if row is None or row >= len(sources):
        return
    deleted = sources[row]
    deleted_id = deleted.id
    delete_configured_source(deleted_id)
    from core.integrations.descriptor_cache import reconcile_mcp_integration_state

    reconcile_mcp_integration_state()
    if getattr(host, "_custom_source_selected_id", "") == deleted_id:
        reset_custom_source_form(host)
    is_dark = getattr(host.window(), "_is_dark_theme", True)
    _refresh_custom_sources_list(host, is_dark=is_dark)
    _refresh_integrations_consent_if_available(host)
