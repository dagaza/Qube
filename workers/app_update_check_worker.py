"""Background worker for GitHub Releases update checks."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QThread, pyqtSignal

from core.app_release_update import AppUpdateCheckResult, check_for_app_update

logger = logging.getLogger("Qube.AppUpdateCheckWorker")


class AppUpdateCheckWorker(QThread):
    finished = pyqtSignal(object)

    def run(self) -> None:
        try:
            result = check_for_app_update()
        except Exception as exc:  # pragma: no cover - defensive UI path
            logger.exception("Unexpected update check failure: %s", exc)
            from core.__version__ import __version__
            from core.app_release_update import AppUpdateCheckResult, AppUpdateStatus

            result = AppUpdateCheckResult(
                status=AppUpdateStatus.ERROR,
                current_version=__version__,
                error_message="Could not check for updates.",
                release_page_url="https://github.com/dagaza/Qube/releases",
            )
        self.finished.emit(result)
