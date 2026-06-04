"""Background worker for Settings companion commentary preview."""

from __future__ import annotations

import time

from PyQt6.QtCore import QThread, pyqtSignal

from core.sidecar_types import SidecarResult, SidecarTask

_SIDECAR_LOAD_WAIT_SEC = 45.0


class CompanionVerbalTestWorker(QThread):
    """Runs a blocking sidecar preview off the UI thread."""

    finished = pyqtSignal(object)

    def __init__(self, sidecar_client, payload: dict, parent=None) -> None:
        super().__init__(parent)
        self._client = sidecar_client
        self._payload = dict(payload or {})

    def run(self) -> None:
        if self._client is None:
            self.finished.emit(
                SidecarResult(
                    ok=False,
                    error="model_unavailable",
                    task=SidecarTask.companion_line,
                )
            )
            return
        deadline = time.monotonic() + _SIDECAR_LOAD_WAIT_SEC
        while not getattr(self._client, "available", False):
            if time.monotonic() >= deadline:
                self.finished.emit(
                    SidecarResult(
                        ok=False,
                        error="model_unavailable",
                        task=SidecarTask.companion_line,
                    )
                )
                return
            time.sleep(0.25)
        preview = getattr(self._client, "preview_companion_line", None)
        if not callable(preview):
            self.finished.emit(
                SidecarResult(
                    ok=False,
                    error="preview_unavailable",
                    task=SidecarTask.companion_line,
                )
            )
            return
        self.finished.emit(preview(**self._payload))
