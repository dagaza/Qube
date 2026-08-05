"""Handlers for Settings → Backup & restore."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QApplication, QFileDialog

from core import app_settings as _backup_settings
from core.paths import user_data_root
from core.state_backup.export import ExportResult, export_state_backup
from core.state_backup.import_backup import RestoreResult, restore_state_backup
from core.state_backup.manifest import BACKUP_EXTENSION, default_backup_filename
from core.state_backup.paths import default_backups_dir
from core.state_backup.scheduler import AutoBackupResult
from ui.components.prestige_dialog import PrestigeDialog


def _interval_label(days: int) -> str:
    return f"Every {days} days"


class _StateBackupExportWorker(QThread):
    finished_with_result = pyqtSignal(object)

    def __init__(self, destination: Path, *, include_wallpapers: bool = False):
        super().__init__()
        self._destination = destination
        self._include_wallpapers = include_wallpapers

    def run(self) -> None:
        result = export_state_backup(
            self._destination,
            include_wallpapers=self._include_wallpapers,
        )
        self.finished_with_result.emit(result)


class _StateBackupRestoreWorker(QThread):
    finished_with_result = pyqtSignal(object)

    def __init__(self, source: Path):
        super().__init__()
        self._source = source

    def run(self) -> None:
        result = restore_state_backup(self._source)
        self.finished_with_result.emit(result)


class BackupRestoreHandlersMixin:
    def _state_backup_is_dark(self) -> bool:
        return getattr(self.window(), "_is_dark_theme", True)

    def _refresh_state_backup_status_hint(self) -> None:
        hint = getattr(self, "state_backup_status_hint", None)
        if hint is None:
            return
        last_at = _backup_settings.get_backup_last_run_at()
        status = _backup_settings.get_backup_last_run_status()
        path = _backup_settings.get_backup_last_run_path()
        if not last_at:
            hint.setText(
                "No automatic backup has run yet. Enable automatic backup above to "
                "save local archives on startup when due."
            )
            return
        try:
            when = datetime.fromisoformat(last_at.replace("Z", "+00:00"))
            stamp = when.strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            stamp = last_at
        detail = status or "unknown"
        lines = [f"Last automatic backup: {stamp} ({detail})."]
        if path:
            lines.append(f"Archive: {path}")
        hint.setText("\n".join(lines))

    def _refresh_state_backup_storage_hint(self) -> None:
        hint = getattr(self, "state_backup_storage_hint", None)
        if hint is None:
            return
        from core.state_backup.storage_summary import format_storage_summary_text

        include_wallpapers = _backup_settings.get_backup_include_wallpapers()
        hint.setText(
            format_storage_summary_text(include_wallpapers=include_wallpapers)
        )

    def _refresh_state_backup_hints(self) -> None:
        self._refresh_state_backup_storage_hint()
        self._refresh_state_backup_status_hint()

    def _on_state_backup_auto_enabled_toggled(self, enabled: bool) -> None:
        _backup_settings.set_backup_auto_enabled(enabled)

    def _on_state_backup_interval_menu_requested(self) -> None:
        items = [
            (_interval_label(days), days)
            for days in _backup_settings.BACKUP_INTERVAL_DAYS_CHOICES
        ]

        def _apply(days: int) -> None:
            _backup_settings.set_backup_interval_days(days)
            selector = getattr(self, "state_backup_interval_selector", None)
            if selector is not None:
                selector.setText(_interval_label(days))

        self._build_prestige_menu(self.state_backup_interval_selector, items, _apply)

    def _on_state_backup_retention_changed(self, value: int) -> None:
        _backup_settings.set_backup_retention_count(value)

    def _on_state_backup_include_wallpapers_toggled(self, enabled: bool) -> None:
        _backup_settings.set_backup_include_wallpapers(enabled)
        self._refresh_state_backup_storage_hint()

    def notify_auto_state_backup_finished(self, result: AutoBackupResult) -> None:
        self._refresh_state_backup_hints()

    def _on_open_backup_restore_guide_clicked(self) -> None:
        from ui.onboarding.tour_helpers import open_qube_help_document

        window = self.window()
        if window is None:
            return
        if not open_qube_help_document(
            window,
            "workflows/backup-or-restore-qube-state.md",
        ):
            is_dark = self._state_backup_is_dark()
            PrestigeDialog(
                self.window(),
                "Help article unavailable",
                "Open Library → Qube and search for "
                "\"Back up or restore Qube state\".",
                is_dark,
            ).exec()

    def _on_open_backup_restore_settings_clicked(self) -> None:
        self.select_settings_section("system.backup", anchor="overview")

    def _set_state_backup_buttons_enabled(self, enabled: bool) -> None:
        for attr in (
            "state_backup_create_btn",
            "state_backup_restore_btn",
            "state_backup_open_backups_btn",
        ):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(enabled)

    def _on_state_backup_create_clicked(self) -> None:
        default_name = default_backup_filename()
        backups_dir = default_backups_dir(user_data_root())
        path, _ = QFileDialog.getSaveFileName(
            self.window(),
            "Export Qube state backup",
            str(backups_dir / default_name),
            f"Qube state backup (*{BACKUP_EXTENSION});;All files (*.*)",
        )
        if not path:
            return
        destination = Path(path)
        if not destination.name.endswith(BACKUP_EXTENSION):
            destination = destination.with_name(f"{destination.name}{BACKUP_EXTENSION}")

        self._set_state_backup_buttons_enabled(False)
        worker = _StateBackupExportWorker(destination)
        worker.finished_with_result.connect(
            lambda result, w=worker: self._on_state_backup_export_finished(result, w)
        )
        worker.start()
        self._state_backup_export_worker = worker

    def _on_state_backup_export_finished(self, result: ExportResult, worker: QThread) -> None:
        self._set_state_backup_buttons_enabled(True)
        worker.deleteLater()
        self._state_backup_export_worker = None
        is_dark = self._state_backup_is_dark()
        if not result.ok:
            PrestigeDialog(
                self.window(),
                "Backup failed",
                result.error or "Could not create the backup archive.",
                is_dark,
                tone="danger",
            ).exec()
            return
        PrestigeDialog(
            self.window(),
            "Backup saved",
            (
                f"Saved {result.file_count} file(s) "
                f"({result.total_bytes:,} bytes) to:\n{result.destination}"
            ),
            is_dark,
        ).exec()
        self._refresh_state_backup_hints()

    def _on_state_backup_restore_clicked(self) -> None:
        is_dark = self._state_backup_is_dark()
        backups_dir = default_backups_dir(user_data_root())
        path, _ = QFileDialog.getOpenFileName(
            self.window(),
            "Restore Qube state backup",
            str(backups_dir),
            f"Qube state backup (*{BACKUP_EXTENSION});;All files (*.*)",
        )
        if not path:
            return

        dlg = PrestigeDialog(
            self.window(),
            "Restore from backup?",
            (
                "This replaces conversations, library indexes, memory vectors, settings, "
                "and related local state with the contents of the selected backup.\n\n"
                "A safety snapshot of your current state is saved under "
                f"{default_backups_dir()} before anything is overwritten.\n\n"
                "Qube must be restarted after restore. Model files are not changed."
            ),
            is_dark,
            tone="danger",
            dialog_width=520,
            confirm_text="RESTORE",
        )
        if not dlg.exec():
            return

        self._set_state_backup_buttons_enabled(False)
        worker = _StateBackupRestoreWorker(Path(path))
        worker.finished_with_result.connect(
            lambda result, w=worker: self._on_state_backup_restore_finished(result, w)
        )
        worker.start()
        self._state_backup_restore_worker = worker

    def _on_state_backup_restore_finished(self, result: RestoreResult, worker: QThread) -> None:
        self._set_state_backup_buttons_enabled(True)
        worker.deleteLater()
        self._state_backup_restore_worker = None
        is_dark = self._state_backup_is_dark()
        if not result.ok:
            PrestigeDialog(
                self.window(),
                "Restore failed",
                result.error or "Could not restore from the backup archive.",
                is_dark,
                tone="danger",
            ).exec()
            return

        snapshot_line = ""
        if result.pre_restore_backup is not None:
            snapshot_line = f"\n\nPre-restore snapshot:\n{result.pre_restore_backup}"

        dlg = PrestigeDialog(
            self.window(),
            "Restore complete",
            (
                f"Restored {result.files_restored} file(s) from the backup."
                f"{snapshot_line}\n\nRestart Qube now to load the restored state."
            ),
            is_dark,
            tone="default",
            confirm_text="QUIT QUBE",
        )
        if dlg.exec():
            app = QApplication.instance()
            if app is not None:
                app.quit()

    def _on_state_backup_open_backups_clicked(self) -> None:
        target = default_backups_dir(user_data_root())
        target.mkdir(parents=True, exist_ok=True)
        url = QUrl.fromLocalFile(str(target))
        if not QDesktopServices.openUrl(url):
            is_dark = self._state_backup_is_dark()
            PrestigeDialog(
                self.window(),
                "Could not open folder",
                f"Open this path manually:\n{target}",
                is_dark,
            ).exec()
