"""Software update handlers for SettingsView."""

from __future__ import annotations

from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QDialog

from core.app_release_update import AppUpdateCheckResult, AppUpdateStatus
from core.support_feedback import open_external_url
from ui.components.prestige_dialog import PrestigeDialog


class UpdateHandlersMixin:
    def _set_update_check_busy(self, busy: bool) -> None:
        button = getattr(self, "check_for_updates_btn", None)
        if button is None:
            return
        button.setEnabled(not busy)
        button.setText("Checking for updates…" if busy else "Check for updates")

    def _on_check_for_updates_clicked(self) -> None:
        worker = getattr(self, "_app_update_check_worker", None)
        if worker is not None and worker.isRunning():
            return

        self._set_update_check_busy(True)

        from workers.app_update_check_worker import AppUpdateCheckWorker

        worker = AppUpdateCheckWorker()
        self._app_update_check_worker = worker
        worker.finished.connect(self._on_app_update_check_finished)
        worker.start()

    def _on_app_update_check_finished(self, result: AppUpdateCheckResult) -> None:
        self._set_update_check_busy(False)
        self._show_app_update_check_dialog(result)

    def _show_app_update_check_dialog(self, result: AppUpdateCheckResult) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        current = result.current_version

        if result.status == AppUpdateStatus.UP_TO_DATE:
            latest = result.latest_version or current
            PrestigeDialog(
                self.window(),
                "You're up to date",
                f"Qube {current} is the latest release on GitHub.\n\nLatest release: {latest}.",
                is_dark=is_dark,
                confirm_text="OK",
                show_cancel=False,
            ).exec()
            return

        if result.status == AppUpdateStatus.ERROR:
            message = result.error_message or "Could not check for updates."
            if result.release_page_url:
                message += f"\n\nYou can still browse releases manually:\n{result.release_page_url}"
            PrestigeDialog(
                self.window(),
                "Update check failed",
                message,
                is_dark=is_dark,
                confirm_text="OK",
                show_cancel=False,
            ).exec()
            return

        latest = result.latest_version or current
        lines = [
            f"A newer version is available: Qube {latest}.",
            f"You are running Qube {current}.",
            "",
            "Download the installer for your platform and install over your existing copy. "
            "Your models, Library, memory, and settings are kept.",
        ]
        if result.release_notes:
            lines.extend(["", result.release_notes])
        if result.download_url:
            lines.extend(["", f"Download:\n{result.download_url}"])
        elif result.release_page_url:
            lines.extend(["", f"Release page:\n{result.release_page_url}"])

        dialog = PrestigeDialog(
            self.window(),
            "Update available",
            "\n".join(lines),
            is_dark=is_dark,
            confirm_text="OPEN DOWNLOAD",
            cancel_text="NOT NOW",
            show_cancel=True,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        target = result.download_url or result.release_page_url
        if not target:
            return
        if not open_external_url(QUrl(target)):
            PrestigeDialog(
                self.window(),
                "Browser unavailable",
                f"Qube could not open your web browser.\n\nVisit this URL manually:\n{target}",
                is_dark=is_dark,
                confirm_text="OK",
                show_cancel=False,
            ).exec()
