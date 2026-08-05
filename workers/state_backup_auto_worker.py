"""Background worker for scheduled automatic state backups."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from core.state_backup.scheduler import AutoBackupResult, run_auto_backup_if_due


class StateBackupAutoWorker(QThread):
    finished_with_result = pyqtSignal(object)

    def run(self) -> None:
        self.finished_with_result.emit(run_auto_backup_if_due())
