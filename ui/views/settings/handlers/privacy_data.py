"""Settings handler mixin: Privacy & data controls."""

from __future__ import annotations

import logging

from core.app_settings import (
    KEY_DISCOVERY_API_FALLBACK_ENABLED,
    KEY_DISCOVERY_PRIVACY_TIER,
    KEY_MCP_INTERNET_HYBRID,
    KEY_ROUTING_DEBUG_REDACT_QUERY_ENABLED,
    KEY_WEB_SEARCH_AUDIT_REDACT_ENABLED,
    get_mcp_internet_hybrid_enabled,
    get_routing_debug_redact_query_enabled,
    get_web_search_audit_redact_enabled,
    set_mcp_internet_hybrid_enabled,
    set_routing_debug_redact_query_enabled,
    set_web_search_audit_redact_enabled,
)
from core.web_search_audit import web_search_audit_redact_env_override
from mcp.routing_debug import routing_debug_log_redact_query_env_override

logger = logging.getLogger("Qube.UI.SettingsPrivacyData")

_LAUNCH_OVERRIDE_NOTE = (
    "Redaction for this log is controlled by how Qube was launched. "
    "Use Settings here when no launch override is present."
)


class PrivacyDataHandlersMixin:
    def _diagnostic_log_redaction_launch_override(self, log_id: str) -> bool | None:
        if log_id == "web_search_audit":
            return web_search_audit_redact_env_override()
        if log_id == "routing_debug":
            return routing_debug_log_redact_query_env_override()
        return None

    def _diagnostic_log_redaction_enabled(self, log_id: str) -> bool:
        if log_id == "web_search_audit":
            from core.web_search_audit import web_search_audit_redact_enabled

            return web_search_audit_redact_enabled()
        if log_id == "routing_debug":
            from mcp.routing_debug import routing_debug_log_redact_query

            return routing_debug_log_redact_query()
        return False

    def _sync_diagnostic_log_redaction_toggle(self, log_id: str) -> None:
        toggles = getattr(self, "diagnostic_log_redaction_toggles", None)
        if not isinstance(toggles, dict):
            return
        toggle = toggles.get(log_id)
        if toggle is None:
            return

        env_override = self._diagnostic_log_redaction_launch_override(log_id)
        toggle.blockSignals(True)
        toggle.setChecked(self._diagnostic_log_redaction_enabled(log_id))
        toggle.blockSignals(False)

        note = getattr(self, "diagnostic_log_redaction_env_notes", {}).get(log_id)
        if env_override is None:
            toggle.setEnabled(True)
            if note is not None:
                note.hide()
        else:
            toggle.setEnabled(False)
            toggle.setChecked(bool(env_override))
            if note is not None:
                note.setText(_LAUNCH_OVERRIDE_NOTE)
                note.show()

    def _sync_all_diagnostic_log_redaction_toggles(self) -> None:
        from core.diagnostic_logs import iter_diagnostic_logs

        for spec in iter_diagnostic_logs():
            if spec.supports_redaction_toggle:
                self._sync_diagnostic_log_redaction_toggle(spec.id)

    def _on_diagnostic_log_redaction_toggled(self, log_id: str, enabled: bool) -> None:
        if self._diagnostic_log_redaction_launch_override(log_id) is not None:
            self._sync_diagnostic_log_redaction_toggle(log_id)
            return

        if log_id == "web_search_audit":
            if enabled == get_web_search_audit_redact_enabled():
                return
            set_web_search_audit_redact_enabled(enabled)
            message = (
                "Web search log redaction is now on — new entries hash queries and "
                "omit snippet bodies."
                if enabled
                else "Web search log redaction is now off."
            )
        elif log_id == "routing_debug":
            if enabled == get_routing_debug_redact_query_enabled():
                return
            set_routing_debug_redact_query_enabled(enabled)
            message = (
                "Routing debug log will hash user queries in new entries."
                if enabled
                else "Routing debug log will record full query text in new entries."
            )
        else:
            logger.warning("Unknown diagnostic log redaction id: %s", log_id)
            return

        self._sync_diagnostic_log_redaction_toggle(log_id)
        if hasattr(self, "_show_settings_file_status"):
            self._show_settings_file_status(message, persistent=True)

    def _on_privacy_data_open_telemetry_discovery_clicked(self) -> None:
        win = self.window()
        if win is not None and hasattr(win, "open_telemetry_focus"):
            win.open_telemetry_focus("web_discovery")

    def _on_privacy_data_open_telemetry_integrations_clicked(self) -> None:
        win = self.window()
        if win is not None and hasattr(win, "open_telemetry_focus"):
            win.open_telemetry_focus("session_integrations")

    def _apply_external_privacy_settings_changed(self, changed: set) -> None:
        privacy_keys = {
            KEY_DISCOVERY_PRIVACY_TIER,
            KEY_DISCOVERY_API_FALLBACK_ENABLED,
            KEY_MCP_INTERNET_HYBRID,
            KEY_WEB_SEARCH_AUDIT_REDACT_ENABLED,
            KEY_ROUTING_DEBUG_REDACT_QUERY_ENABLED,
        }
        if not (privacy_keys & changed):
            return
        if hasattr(self, "_sync_discovery_privacy_tier_selector"):
            self._sync_discovery_privacy_tier_selector()
        if getattr(self, "web_discovery_policy_section", None) is not None:
            from ui.views.settings.sections.knowledge_web_discovery import (
                sync_web_discovery_policy_section,
            )

            sync_web_discovery_policy_section(self)
        self._sync_privacy_data_section_ui()

    def _sync_privacy_data_internet_hybrid_toggle(self) -> None:
        toggle = getattr(self, "privacy_data_internet_hybrid_toggle", None)
        if toggle is None:
            return
        enabled = get_mcp_internet_hybrid_enabled()
        if toggle.isChecked() != enabled:
            toggle.blockSignals(True)
            toggle.setChecked(enabled)
            toggle.blockSignals(False)

    def _on_privacy_data_internet_hybrid_toggled(self, enabled: bool) -> None:
        if get_mcp_internet_hybrid_enabled() == bool(enabled):
            return
        set_mcp_internet_hybrid_enabled(enabled)

        win = self.window()
        if win is not None:
            llm_worker = getattr(win, "_llm_worker", None)
            if llm_worker is not None:
                llm_worker.set_mcp_internet_hybrid(enabled)
            toolbar_toggle = getattr(win, "tool_internet_hybrid_toggle", None)
            if toolbar_toggle is not None and toolbar_toggle.isChecked() != enabled:
                toolbar_toggle.blockSignals(True)
                toolbar_toggle.setChecked(enabled)
                toolbar_toggle.blockSignals(False)
            if hasattr(win, "_web_indicator_hybrid"):
                win._web_indicator_hybrid = bool(enabled)
            if hasattr(win, "_apply_web_indicator"):
                win._apply_web_indicator()

        self._sync_privacy_data_internet_hybrid_toggle()
        if hasattr(self, "_emit_external_settings_changed"):
            self._emit_external_settings_changed(KEY_MCP_INTERNET_HYBRID)
        if hasattr(self, "_show_settings_file_status"):
            message = (
                "Hybrid Internet Mode is now on — Qube may auto-route to web search "
                "when context warrants it."
                if enabled
                else "Hybrid Internet Mode is now off."
            )
            self._show_settings_file_status(message, persistent=True)
