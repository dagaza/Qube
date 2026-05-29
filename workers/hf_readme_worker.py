"""Background fetch of model README from Hugging Face (raw markdown text)."""

from __future__ import annotations

import logging

import requests
from PyQt6.QtCore import QThread, pyqtSignal

from core.hf_hub_errors import HubErrorInfo, classify_hf_error, classify_hf_http_status

logger = logging.getLogger("Qube.HFReadme")

_RAW_URL = "https://huggingface.co/{repo}/raw/main/{name}"
_RESOLVE_URL = "https://huggingface.co/{repo}/resolve/main/{name}"

# Try common README filenames first.
_README_CANDIDATES = (
    "README.md",
    "Readme.md",
    "README.MD",
    "readme.md",
)


class HfReadmeWorker(QThread):
    """Fetches README markdown text for a model repo."""

    finished_ok = pyqtSignal(str, str)  # repo_id, markdown_text
    failed = pyqtSignal(str, object)  # repo_id, HubErrorInfo

    def __init__(self, repo_id: str):
        super().__init__()
        self._repo_id = (repo_id or "").strip()

    def run(self) -> None:
        repo = self._repo_id
        if not repo:
            self.failed.emit("", classify_hf_error("Empty repository id."))
            return

        headers = {"Accept": "text/plain, text/markdown, */*"}
        last_info: HubErrorInfo | None = None
        for name in _README_CANDIDATES:
            if self.isInterruptionRequested():
                return
            for tmpl in (_RAW_URL, _RESOLVE_URL):
                if self.isInterruptionRequested():
                    return
                url = tmpl.format(repo=repo, name=name)
                try:
                    r = requests.get(url, timeout=(15, 60), headers=headers)
                    if r.status_code == 200 and (r.text or "").strip():
                        text = r.text
                        # Strip pathological binary / huge payloads
                        if len(text) > 1_500_000:
                            text = text[:1_500_000] + "\n\n… *(README truncated for display)*"
                        self.finished_ok.emit(repo, text)
                        return
                    last_info = classify_hf_http_status(
                        r.status_code,
                        context="README fetch",
                    )
                except requests.RequestException as e:
                    last_info = classify_hf_error(e, context="README fetch")
                    logger.debug("README fetch attempt failed %s: %s", url, e)

        self.failed.emit(
            repo,
            last_info
            or classify_hf_error("No README found.", context="README fetch"),
        )
