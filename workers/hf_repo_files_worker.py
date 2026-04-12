"""Background fetch of .gguf filenames from a Hugging Face model repo (Hub API)."""

from __future__ import annotations

import logging
import re
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger("Qube.HFRepoFiles")


def _sanitize_repo_id(repo_id: str) -> str:
    s = (repo_id or "").strip()
    if not re.match(r"^[\w.\-/]+$", s):
        raise ValueError("Invalid repository id format.")
    return s


class HfRepoFilesWorker(QThread):
    """Lists repo files via huggingface_hub; filters to .gguf paths."""

    finished_ok = pyqtSignal(list)  # list[str] repo-relative paths
    failed = pyqtSignal(str)

    def __init__(self, repo_id: str, revision: Optional[str] = None):
        super().__init__()
        self._repo_id = repo_id
        self._revision = revision

    def run(self) -> None:
        try:
            repo = _sanitize_repo_id(self._repo_id)
        except ValueError as e:
            self.failed.emit(str(e))
            return

        try:
            if self.isInterruptionRequested():
                return
            from huggingface_hub import HfApi

            api = HfApi()
            files = api.list_repo_files(
                repo,
                repo_type="model",
                revision=self._revision,
            )
        except Exception as e:
            logger.exception("HfApi.list_repo_files failed: %s", e)
            self.failed.emit(str(e))
            return

        gguf = sorted(f for f in files if str(f).lower().endswith(".gguf"))
        self.finished_ok.emit(gguf)
