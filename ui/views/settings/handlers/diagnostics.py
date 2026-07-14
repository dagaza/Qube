"""Settings handler mixin: diagnostic log viewers."""

from __future__ import annotations

import logging

from core.app_log_sink import app_log_env_override
from core.llm_debug_sink import llm_debug_log_env_override
from core.app_settings import (
    get_routing_debug_log_enabled,
    get_skills_debug_log_enabled,
    get_web_search_audit_log_enabled,
    set_routing_debug_log_enabled,
    set_skills_debug_log_enabled,
    set_web_search_audit_log_enabled,
)
from core.diagnostic_logs import (
    clear_diagnostic_log,
    describe_log_status,
    diagnostic_log_recording_enabled,
    get_diagnostic_log,
    iter_diagnostic_logs,
    open_logs_folder,
)
from core.logging_bootstrap import (
    init_skills_debug_logging,
    init_web_search_audit_logging,
    set_app_log_file_recording_enabled,
    set_llm_debug_log_file_recording_enabled,
)
from core.web_search_audit import web_search_audit_log_env_override
from mcp.routing_debug import routing_debug_log_env_override
from ui.components.diagnostic_log_viewer_dialog import DiagnosticLogViewerDialog
from ui.components.prestige_dialog import PrestigeDialog

logger = logging.getLogger("Qube.UI.SettingsDiagnostics")

_LAUNCH_OVERRIDE_NOTE = (
    "Recording for this log is controlled by how Qube was launched. "
    "Use Settings here when no launch override is present."
)


class DiagnosticsHandlersMixin:
    def _ensure_diagnostic_log_dialogs(self) -> dict[str, DiagnosticLogViewerDialog]:
        dialogs = getattr(self, "_diagnostic_log_dialogs", None)
        if dialogs is None:
            dialogs = {}
            self._diagnostic_log_dialogs = dialogs
        return dialogs

    def _diagnostic_log_launch_override(self, log_id: str) -> bool | None:
        if log_id == "routing_debug":
            return routing_debug_log_env_override()
        if log_id == "web_search_audit":
            return web_search_audit_log_env_override()
        if log_id == "app_log":
            return app_log_env_override()
        if log_id == "llm_debug":
            return llm_debug_log_env_override()
        return None

    def _sync_diagnostic_log_recording_toggle(self, log_id: str) -> None:
        toggles = getattr(self, "diagnostic_log_recording_toggles", None)
        if not isinstance(toggles, dict):
            return
        toggle = toggles.get(log_id)
        if toggle is None:
            return

        env_override = self._diagnostic_log_launch_override(log_id)
        toggle.blockSignals(True)
        toggle.setChecked(diagnostic_log_recording_enabled(log_id))
        toggle.blockSignals(False)

        if env_override is None:
            toggle.setEnabled(True)
            note = getattr(self, "diagnostic_log_recording_env_notes", {}).get(log_id)
            if note is not None:
                note.hide()
        else:
            toggle.setEnabled(False)
            toggle.setChecked(bool(env_override))
            note = getattr(self, "diagnostic_log_recording_env_notes", {}).get(log_id)
            if note is not None:
                note.setText(_LAUNCH_OVERRIDE_NOTE)
                note.show()

    def _sync_all_diagnostic_log_recording_toggles(self) -> None:
        for spec in iter_diagnostic_logs():
            if spec.supports_recording_toggle:
                self._sync_diagnostic_log_ui(spec.id)

    def _sync_routing_debug_log_ui(self) -> None:
        self._sync_diagnostic_log_ui("routing_debug")

    def _sync_skills_debug_log_ui(self) -> None:
        self._sync_diagnostic_log_ui("skills_debug")

    def _sync_diagnostic_log_ui(self, log_id: str) -> None:
        self._sync_diagnostic_log_recording_toggle(log_id)

        status_labels = getattr(self, "diagnostic_log_status_labels", None)
        if isinstance(status_labels, dict):
            spec = get_diagnostic_log(log_id)
            label = status_labels.get(log_id)
            if spec is not None and label is not None:
                label.setText(describe_log_status(spec))

        dialogs = getattr(self, "_diagnostic_log_dialogs", None)
        if isinstance(dialogs, dict):
            dialog = dialogs.get(log_id)
            if dialog is not None:
                dialog.sync_recording_toggle()
                dialog._refresh()

    def _on_open_logs_folder_clicked(self) -> None:
        if open_logs_folder():
            self._show_settings_file_status("Opened the logs folder in your file manager.")
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        PrestigeDialog(
            self,
            "Could not open logs folder",
            "Qube could not open the logs folder in your file manager.",
            is_dark=is_dark,
        ).exec()

    def _on_diagnostic_log_recording_toggled(self, log_id: str, enabled: bool) -> None:
        if self._diagnostic_log_launch_override(log_id) is not None:
            self._sync_diagnostic_log_ui(log_id)
            return

        if log_id == "routing_debug":
            if enabled == get_routing_debug_log_enabled():
                return
            set_routing_debug_log_enabled(enabled)
            self._sync_diagnostic_log_ui(log_id)
            if enabled:
                message = (
                    "Routing debug recording is now on. Send a chat message and refresh "
                    "this log to see new entries."
                )
            else:
                message = "Routing debug recording is now off. New chat turns will not be added."
            self._show_settings_file_status(message, persistent=True)
            return

        if log_id == "skills_debug":
            if enabled == get_skills_debug_log_enabled():
                return
            set_skills_debug_log_enabled(enabled)
            if enabled:
                init_skills_debug_logging()
            self._sync_diagnostic_log_ui(log_id)
            if enabled:
                message = (
                    "Skills debug recording is now on. Send a chat message and refresh "
                    "this log to see activation entries."
                )
            else:
                message = "Skills debug recording is now off. New chat turns will not be added."
            self._show_settings_file_status(message, persistent=True)
            return

        if log_id == "web_search_audit":
            if enabled == get_web_search_audit_log_enabled():
                return
            set_web_search_audit_log_enabled(enabled)
            if enabled:
                init_web_search_audit_logging()
            self._sync_diagnostic_log_ui(log_id)
            if enabled:
                message = (
                    "Web search log recording is now on. Trigger a web search and "
                    "refresh this log to see new entries."
                )
            else:
                message = (
                    "Web search log recording is now off. New searches will not be added."
                )
            self._show_settings_file_status(message, persistent=True)
            return

        if log_id == "app_log":
            set_app_log_file_recording_enabled(enabled)
            self._sync_diagnostic_log_ui(log_id)
            if enabled:
                message = "Application log recording is now on."
            else:
                message = (
                    "Application log recording is now off. Terminal output is unchanged."
                )
            self._show_settings_file_status(message, persistent=True)
            return

        if log_id == "llm_debug":
            set_llm_debug_log_file_recording_enabled(enabled)
            self._sync_diagnostic_log_ui(log_id)
            if enabled:
                message = (
                    "LLM debug log recording is now on. New introspection events will be "
                    "written on the next qualifying action."
                )
            else:
                message = "LLM debug log recording is now off. New file entries will not be added."
            self._show_settings_file_status(message, persistent=True)
            return

        logger.warning("Unknown diagnostic log id for recording toggle: %s", log_id)

    def _on_routing_debug_log_recording_toggled(self, enabled: bool) -> None:
        self._on_diagnostic_log_recording_toggled("routing_debug", enabled)

    def _on_skills_debug_log_recording_toggled(self, enabled: bool) -> None:
        self._on_diagnostic_log_recording_toggled("skills_debug", enabled)

    def _on_view_diagnostic_log_clicked(self, log_id: str) -> None:
        spec = get_diagnostic_log(log_id)
        if spec is None:
            logger.warning("Unknown diagnostic log id: %s", log_id)
            return

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dialogs = self._ensure_diagnostic_log_dialogs()
        dialog = dialogs.get(log_id)
        if dialog is None:
            on_recording_toggle = None
            if spec.supports_recording_toggle:
                on_recording_toggle = (
                    lambda enabled, lid=log_id: self._on_diagnostic_log_recording_toggled(
                        lid, enabled
                    )
                )
            dialog = DiagnosticLogViewerDialog(
                spec,
                self,
                is_dark=is_dark,
                on_recording_toggle=on_recording_toggle,
            )
            dialogs[log_id] = dialog
        else:
            dialog.refresh_theme(is_dark)
            if spec.supports_recording_toggle:
                dialog.sync_recording_toggle()

        dialog._refresh()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._show_settings_file_status(f"Viewing {spec.title}.")

    def _on_clear_diagnostic_log_clicked(self, log_id: str) -> None:
        spec = get_diagnostic_log(log_id)
        if spec is None:
            logger.warning("Unknown diagnostic log id: %s", log_id)
            return

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        path = spec.path_fn()
        dlg = PrestigeDialog(
            self.window(),
            f"Clear {spec.title}?",
            (
                f"This deletes all contents of:\n{path}\n\n"
                "Rotated backup files for this log are removed as well. "
                "The file is recreated automatically when Qube logs to it again."
            ),
            is_dark=is_dark,
            tone="danger",
            dialog_width=480,
            confirm_text="CLEAR LOG",
        )
        if not dlg.exec():
            return

        result = clear_diagnostic_log(spec)
        self._sync_diagnostic_log_ui(log_id)
        if result.success:
            self._show_settings_file_status(result.detail)
            return

        PrestigeDialog(
            self,
            "Could not clear log",
            result.detail,
            is_dark=is_dark,
        ).exec()
