"""Background Hugging Face size resolution for bootstrap consent."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from core.bootstrap_hf_metadata import resolve_all_bootstrap_sizes


class BootstrapMetadataWorker(QThread):
    """Fetch live Hugging Face file sizes for bootstrap models."""

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            resolved = resolve_all_bootstrap_sizes()
            self.finished_ok.emit(resolved)
        except Exception as exc:
            self.failed.emit(str(exc))
