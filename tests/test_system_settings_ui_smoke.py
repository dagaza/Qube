"""UI smoke tests for Settings → System sections."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

from core.diagnostic_logs import iter_diagnostic_logs_by_category
from ui.main_window import MAIN_STAGE_SETTINGS, MAIN_STAGE_TELEMETRY


SYSTEM_SECTION_WIDGETS: dict[str, tuple[str, ...]] = {
    "privacy.data": (
        "privacy_data_overview_hint",
        "privacy_data_session_audit_hint",
        "privacy_data_open_telemetry_discovery_btn",
        "privacy_data_open_telemetry_integrations_btn",
        "privacy_data_privacy_tier_selector",
        "privacy_data_privacy_tier_description",
        "privacy_data_open_knowledge_discovery_btn",
        "privacy_data_internet_hybrid_toggle",
        "privacy_data_what_leaves_card",
    ),
    "diagnostics": (
        "diagnostic_logs_hint_lbl",
        "open_logs_folder_btn",
    ),
    "license": (
        "license_section_card",
        "license_status_banner",
        "license_status_banner_title",
        "license_status_banner_body",
        "license_hint_lbl",
        "license_status_lbl",
        "import_license_btn",
        "remove_license_btn",
    ),
    "advanced": (
        "settings_json_hint_lbl",
        "open_settings_json_btn",
        "settings_file_status_lbl",
    ),
}


def _open_settings(main_window, qtbot):
    main_window.show()
    main_window.resize(1400, 900)
    qtbot.waitExposed(main_window)
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None
    qtbot.waitExposed(settings)
    return settings


def _section_content(settings, section_id: str) -> QWidget:
    scroll = settings._section_scroll_by_id.get(section_id)
    assert scroll is not None, f"missing scroll for {section_id}"
    content = scroll.widget()
    assert content is not None, f"missing content widget for {section_id}"
    return content


def _is_descendant(widget: QWidget, ancestor: QWidget) -> bool:
    node: QWidget | None = widget
    while node is not None:
        if node is ancestor:
            return True
        node = node.parentWidget()
    return False


def _select_system_section(settings, qtbot, section_id: str) -> None:
    settings.select_settings_section(section_id)
    qtbot.wait(50)


def _assert_widget_in_active_section(settings, section_id: str, widget, label: str) -> None:
    assert widget is not None, f"{section_id} missing {label}"
    content = _section_content(settings, section_id)
    assert _is_descendant(widget, content), f"{label} not in {section_id} section body"
    assert widget.isVisibleTo(content), f"{section_id}.{label} should be visible"


@pytest.mark.ui
@pytest.mark.parametrize(
    ("section_id", "widget_attrs"),
    tuple(SYSTEM_SECTION_WIDGETS.items()),
    ids=list(SYSTEM_SECTION_WIDGETS.keys()),
)
def test_system_section_key_widgets_exist(
    fresh_main_window, qtbot, section_id, widget_attrs
):
    settings = _open_settings(fresh_main_window, qtbot)
    _select_system_section(settings, qtbot, section_id)

    for attr in widget_attrs:
        widget = getattr(settings, attr, None)
        _assert_widget_in_active_section(settings, section_id, widget, attr)


@pytest.mark.ui
def test_privacy_data_builds_audit_log_controls_only(fresh_main_window, qtbot):
    settings = _open_settings(fresh_main_window, qtbot)
    _select_system_section(settings, qtbot, "privacy.data")

    audit_ids = {spec.id for spec in iter_diagnostic_logs_by_category("audit")}
    technical_ids = {spec.id for spec in iter_diagnostic_logs_by_category("technical")}

    privacy_content = _section_content(settings, "privacy.data")
    for log_id in audit_ids:
        view_btn = settings.diagnostic_log_view_buttons.get(log_id)
        assert view_btn is not None
        assert _is_descendant(view_btn, privacy_content)
        assert log_id in settings.diagnostic_log_recording_toggles
    for log_id in technical_ids:
        view_btn = settings.diagnostic_log_view_buttons.get(log_id)
        if view_btn is not None:
            assert not _is_descendant(view_btn, privacy_content)

    for log_id in ("routing_debug", "web_search_audit"):
        toggle = settings.diagnostic_log_redaction_toggles.get(log_id)
        assert toggle is not None
        assert _is_descendant(toggle, privacy_content)

    open_logs_btn = getattr(settings, "open_logs_folder_btn", None)
    assert open_logs_btn is None or not _is_descendant(open_logs_btn, privacy_content)


@pytest.mark.ui
def test_diagnostics_builds_technical_log_controls_only(fresh_main_window, qtbot):
    settings = _open_settings(fresh_main_window, qtbot)
    _select_system_section(settings, qtbot, "diagnostics")

    audit_ids = {spec.id for spec in iter_diagnostic_logs_by_category("audit")}
    technical_ids = {spec.id for spec in iter_diagnostic_logs_by_category("technical")}

    diagnostics_content = _section_content(settings, "diagnostics")
    for log_id in technical_ids:
        view_btn = settings.diagnostic_log_view_buttons.get(log_id)
        assert view_btn is not None
        assert _is_descendant(view_btn, diagnostics_content)
        assert log_id in settings.diagnostic_log_recording_toggles
    for log_id in audit_ids:
        view_btn = settings.diagnostic_log_view_buttons.get(log_id)
        if view_btn is not None:
            assert not _is_descendant(view_btn, diagnostics_content)

    for toggle in settings.diagnostic_log_redaction_toggles.values():
        assert not _is_descendant(toggle, diagnostics_content)

    _assert_widget_in_active_section(
        settings,
        "diagnostics",
        settings.open_logs_folder_btn,
        "open_logs_folder_btn",
    )
    assert settings.open_logs_folder_btn.isEnabled()


@pytest.mark.ui
def test_license_section_shows_status_text(fresh_main_window, qtbot):
    settings = _open_settings(fresh_main_window, qtbot)
    _select_system_section(settings, qtbot, "license")

    status = settings.license_status_lbl.text().strip()
    assert status
    _assert_widget_in_active_section(
        settings, "license", settings.import_license_btn, "import_license_btn"
    )
    _assert_widget_in_active_section(
        settings, "license", settings.remove_license_btn, "remove_license_btn"
    )
    assert settings.import_license_btn.isEnabled()
    assert not settings.remove_license_btn.isEnabled()


@pytest.mark.ui
def test_privacy_data_telemetry_buttons_route_to_telemetry(fresh_main_window, qtbot):
    settings = _open_settings(fresh_main_window, qtbot)
    _select_system_section(settings, qtbot, "privacy.data")

    qtbot.mouseClick(
        settings.privacy_data_open_telemetry_discovery_btn,
        Qt.MouseButton.LeftButton,
    )
    qtbot.wait(150)

    assert MAIN_STAGE_TELEMETRY in fresh_main_window._main_stage_built
    assert (
        fresh_main_window.main_stage.currentWidget()
        is fresh_main_window.ensure_telemetry_view()
    )

    qtbot.mouseClick(fresh_main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = fresh_main_window.peek_settings_view()
    assert settings is not None
    _select_system_section(settings, qtbot, "privacy.data")

    qtbot.mouseClick(
        settings.privacy_data_open_telemetry_integrations_btn,
        Qt.MouseButton.LeftButton,
    )
    qtbot.wait(150)

    assert (
        fresh_main_window.main_stage.currentWidget()
        is fresh_main_window.ensure_telemetry_view()
    )


@pytest.mark.ui
def test_privacy_data_knowledge_button_selects_knowledge_section(
    fresh_main_window, qtbot
):
    settings = _open_settings(fresh_main_window, qtbot)
    _select_system_section(settings, qtbot, "privacy.data")

    qtbot.mouseClick(
        settings.privacy_data_open_knowledge_discovery_btn,
        Qt.MouseButton.LeftButton,
    )
    qtbot.wait(10)

    assert settings._settings_active_section_id == "knowledge"
    assert (
        settings.settings_section_list.currentRow()
        == settings._section_row_by_id["knowledge"]
    )
    assert MAIN_STAGE_SETTINGS in fresh_main_window._main_stage_built


@pytest.mark.ui
def test_open_telemetry_focus_navigates_from_main_window(fresh_main_window, qtbot):
    fresh_main_window.open_telemetry_focus("web_discovery")
    qtbot.wait(150)

    assert MAIN_STAGE_TELEMETRY in fresh_main_window._main_stage_built
    telemetry = fresh_main_window.ensure_telemetry_view()
    assert fresh_main_window.main_stage.currentWidget() is telemetry
    assert getattr(telemetry, "discovery_card", None) is not None

    fresh_main_window.open_telemetry_focus("session_integrations")
    qtbot.wait(150)

    assert getattr(telemetry, "session_egress_panel", None) is not None
