"""Background download of a single bootstrap model from Settings."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QThread, pyqtSignal

from core.bootstrap_download import run_bootstrap_model_download
from core.bootstrap_manifest import BootstrapModelId

logger = logging.getLogger("Qube.BootstrapDownloadWorker")


class BootstrapModelDownloadWorker(QThread):
    finished_ok = pyqtSignal(bool)  # used_mock
    failed = pyqtSignal(str)
    progress = pyqtSignal(str, str, int, str)

    def __init__(self, model_id: BootstrapModelId) -> None:
        super().__init__()
        self._model_id = model_id

    def run(self) -> None:
        try:

            def _on_progress(step_label: str, filename: str, percent: int, source: str) -> None:
                self.progress.emit(step_label, filename, percent, source)

            errors, used_mock = run_bootstrap_model_download(
                {self._model_id},
                on_progress=_on_progress,
            )
            if errors:
                self.failed.emit("; ".join(errors))
                return
            self.finished_ok.emit(used_mock)
        except Exception as exc:
            logger.exception("Bootstrap model download failed for %s", self._model_id)
            self.failed.emit(str(exc))
