"""Lightweight background probe for Hugging Face Hub reachability."""

from __future__ import annotations

import logging

import requests
from PyQt6.QtCore import QThread, pyqtSignal

from core.hf_hub_errors import HubErrorInfo, classify_hf_error

logger = logging.getLogger("Qube.HFConnectivity")

HF_PROBE_URL = "https://huggingface.co/api/models?limit=1"


class HfConnectivityProbeWorker(QThread):
    """HEAD/GET probe; emits reachability without blocking the UI thread."""

    finished_ok = pyqtSignal()
    failed = pyqtSignal(object)  # HubErrorInfo

    def __init__(self, timeout_s: float = 8.0):
        super().__init__()
        self._timeout_s = max(2.0, float(timeout_s))

    def run(self) -> None:
        if self.isInterruptionRequested():
            return
        try:
            resp = requests.get(
                HF_PROBE_URL,
                timeout=(min(5.0, self._timeout_s), self._timeout_s),
                headers={"User-Agent": "Qube-Desktop/1.0"},
            )
            if self.isInterruptionRequested():
                return
            if resp.status_code == 200:
                self.finished_ok.emit()
                return
            info = classify_hf_error(
                f"HTTP {resp.status_code}",
                http_status=resp.status_code,
                context="connectivity probe",
            )
            self.failed.emit(info)
        except Exception as e:
            if self.isInterruptionRequested():
                return
            logger.debug("HF connectivity probe failed: %s", e)
            self.failed.emit(classify_hf_error(e, context="connectivity probe"))
