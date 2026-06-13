"""Settings handler mixin: diagnostic log viewers."""

from __future__ import annotations

import logging

from core.diagnostic_logs import get_diagnostic_log, open_logs_folder
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

    def _on_view_diagnostic_log_clicked(self, log_id: str) -> None:
        spec = get_diagnostic_log(log_id)
        if spec is None:
            logger.warning("Unknown diagnostic log id: %s", log_id)
            return

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dialogs = self._ensure_diagnostic_log_dialogs()
        dialog = dialogs.get(log_id)
        if dialog is None:
            dialog = DiagnosticLogViewerDialog(spec, self, is_dark=is_dark)
            dialogs[log_id] = dialog
        else:
            dialog.refresh_theme(is_dark)

        dialog._refresh()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._show_settings_file_status(f"Viewing {spec.title}.")
