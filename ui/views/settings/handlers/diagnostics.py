"""Settings handler mixin: diagnostic log viewers."""

from __future__ import annotations

import logging

from core.app_settings import (
    get_routing_debug_log_enabled,
    get_skills_debug_log_enabled,
    set_routing_debug_log_enabled,
    set_skills_debug_log_enabled,
)
from core.diagnostic_logs import (
    describe_log_status,
    get_diagnostic_log,
    open_logs_folder,
)
from core.logging_bootstrap import init_skills_debug_logging
from mcp.routing_debug import routing_debug_log_env_override
from ui.components.diagnostic_log_viewer_dialog import DiagnosticLogViewerDialog
from ui.components.prestige_dialog import PrestigeDialog

logger = logging.getLogger("Qube.UI.SettingsDiagnostics")


class DiagnosticsHandlersMixin:
    def _ensure_diagnostic_log_dialogs(self) -> dict[str, DiagnosticLogViewerDialog]:
        dialogs = getattr(self, "_diagnostic_log_dialogs", None)
        if dialogs is None:
            dialogs = {}
            self._diagnostic_log_dialogs = dialogs
        return dialogs

    def _sync_routing_debug_log_ui(self) -> None:
        self._sync_diagnostic_log_ui("routing_debug")

    def _sync_skills_debug_log_ui(self) -> None:
        self._sync_diagnostic_log_ui("skills_debug")

    def _sync_diagnostic_log_ui(self, log_id: str) -> None:
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

    def _on_routing_debug_log_recording_toggled(self, enabled: bool) -> None:
        if routing_debug_log_env_override() is not None:
            self._sync_routing_debug_log_ui()
            return
        if enabled == get_routing_debug_log_enabled():
            return

        set_routing_debug_log_enabled(enabled)
        self._sync_routing_debug_log_ui()
        if enabled:
            message = (
                "Routing debug recording is now on. Send a chat message and refresh "
                "this log to see new entries."
            )
        else:
            message = "Routing debug recording is now off. New chat turns will not be added."
        self._show_settings_file_status(message, persistent=True)

    def _on_skills_debug_log_recording_toggled(self, enabled: bool) -> None:
        if enabled == get_skills_debug_log_enabled():
            return

        set_skills_debug_log_enabled(enabled)
        if enabled:
            init_skills_debug_logging()
        self._sync_skills_debug_log_ui()
        if enabled:
            message = (
                "Skills debug recording is now on. Send a chat message and refresh "
                "this log to see activation entries."
            )
        else:
            message = "Skills debug recording is now off. New chat turns will not be added."
        self._show_settings_file_status(message, persistent=True)

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
                if log_id == "skills_debug":
                    on_recording_toggle = self._on_skills_debug_log_recording_toggled
                else:
                    on_recording_toggle = self._on_routing_debug_log_recording_toggled
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
