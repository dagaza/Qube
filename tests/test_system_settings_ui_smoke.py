"""UI smoke tests for Settings → System sections."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

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
    qtbot.mouseClick(main_window.nav_settings, Qt.MouseButton.LeftButton)
    settings = main_window.peek_settings_view()
    assert settings is not None
    return settings


def _select_system_section(settings, qtbot, section_id: str) -> None:
    settings.select_settings_section(section_id)
    qtbot.wait(10)


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
        assert widget is not None, f"{section_id} missing {attr}"
        assert widget.isVisible(), f"{section_id}.{attr} should be visible"


@pytest.mark.ui
def test_privacy_data_builds_audit_log_controls_only(fresh_main_window, qtbot):
    settings = _open_settings(fresh_main_window, qtbot)
    _select_system_section(settings, qtbot, "privacy.data")

    audit_ids = {spec.id for spec in iter_diagnostic_logs_by_category("audit")}
    technical_ids = {spec.id for spec in iter_diagnostic_logs_by_category("technical")}

    assert audit_ids <= set(settings.diagnostic_log_view_buttons)
    assert audit_ids <= set(settings.diagnostic_log_recording_toggles)
    assert technical_ids.isdisjoint(settings.diagnostic_log_view_buttons)
    assert technical_ids.isdisjoint(settings.diagnostic_log_recording_toggles)

    for log_id in ("routing_debug", "web_search_audit"):
        assert log_id in settings.diagnostic_log_redaction_toggles

    assert not hasattr(settings, "open_logs_folder_btn")


@pytest.mark.ui
def test_diagnostics_builds_technical_log_controls_only(fresh_main_window, qtbot):
    settings = _open_settings(fresh_main_window, qtbot)
    _select_system_section(settings, qtbot, "diagnostics")

    audit_ids = {spec.id for spec in iter_diagnostic_logs_by_category("audit")}
    technical_ids = {spec.id for spec in iter_diagnostic_logs_by_category("technical")}

    assert technical_ids <= set(settings.diagnostic_log_view_buttons)
    assert technical_ids <= set(settings.diagnostic_log_recording_toggles)
    assert audit_ids.isdisjoint(settings.diagnostic_log_view_buttons)
    assert audit_ids.isdisjoint(settings.diagnostic_log_recording_toggles)
    assert settings.diagnostic_log_redaction_toggles == {}

    assert settings.open_logs_folder_btn.isVisible()
    assert settings.open_logs_folder_btn.isEnabled()


@pytest.mark.ui
def test_license_section_shows_status_text(fresh_main_window, qtbot):
    settings = _open_settings(fresh_main_window, qtbot)
    _select_system_section(settings, qtbot, "license")

    status = settings.license_status_lbl.text().strip()
    assert status
    assert settings.import_license_btn.isEnabled()
    assert settings.remove_license_btn.isEnabled()


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
