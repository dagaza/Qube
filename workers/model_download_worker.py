"""Background download of a single .gguf file from Hugging Face (HTTP streaming + progress)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger("Qube.ModelDownload")

# Extra headroom on top of Content-Length (bytes) before starting the stream.
SAFETY_BUFFER_BYTES = 500 * 1024 * 1024


def _sanitize_repo_id(repo_id: str) -> str:
    s = (repo_id or "").strip()
    if not re.match(r"^[\w.\-/]+$", s):
        raise ValueError("Invalid repository id format.")
    return s


def _sanitize_repo_file_path(name: str) -> str:
    """Repo-relative path; allows subfolders (e.g. from Hub file list)."""
    n = (name or "").strip().strip("/")
    if not n.lower().endswith(".gguf"):
        raise ValueError("File must be a .gguf model.")
    if ".." in n or n.startswith("/"):
        raise ValueError("Invalid file path.")
    return n


def _unlink_quiet(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except OSError as e:
        logger.warning("Could not remove partial download %s: %s", path, e)


def _free_bytes_on_filesystem(path: Path) -> int:
    """Free space (bytes) on the filesystem that contains ``path``."""
    try:
        return int(shutil.disk_usage(str(path)).free)
    except OSError as e:
        logger.error("disk_usage failed for %s: %s", path, e)
        return 0


class HuggingFaceGgufDownloadWorker(QThread):
    progress_pct = pyqtSignal(int)  # 0–100
    status_message = pyqtSignal(str)
    finished_ok = pyqtSignal(str)  # absolute path saved
    failed = pyqtSignal(str)
    insufficient_space_error = pyqtSignal(int, int)  # required_bytes, available_bytes
    download_cancelled = pyqtSignal()

    def __init__(
        self,
        repo_id: str,
        filename: str,
        dest_dir: str,
        revision: str | None = None,
    ):
        super().__init__()
        self._repo_id = repo_id
        self._filename = filename
        self._dest_dir = dest_dir
        self._revision = revision
        self._cancel_event = threading.Event()

    @property
    def _is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        """Request cooperative cancellation; streaming loop observes ``_is_cancelled``."""
        self._cancel_event.set()

    def run(self) -> None:
        try:
            repo = _sanitize_repo_id(self._repo_id)
            fname = _sanitize_repo_file_path(self._filename)
        except ValueError as e:
            self.failed.emit(str(e))
            return

        dest_root = Path(self._dest_dir).resolve()
        dest_root.mkdir(parents=True, exist_ok=True)
        local_name = Path(fname).name
        out_path = dest_root / local_name
        tmp_path = str(out_path) + ".part"

        url = hf_hub_url(
            repo_id=repo,
            filename=fname,
            repo_type="model",
            revision=self._revision,
        )
        self.status_message.emit(f"Downloading {fname}…")

        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as resp:
                if resp.status_code != 200:
                    self.failed.emit(
                        f"HTTP {resp.status_code} — check repo id and filename (default branch)."
                    )
                    return

                if self._is_cancelled:
                    self.download_cancelled.emit()
                    return

                total = int(resp.headers.get("content-length") or 0)
                free = _free_bytes_on_filesystem(out_path.parent)

                if total > 0:
                    required = total + SAFETY_BUFFER_BYTES
                    if free < required:
                        self.insufficient_space_error.emit(required, free)
                        return
                else:
                    # Unknown size: still require at least the safety buffer to start.
                    if free < SAFETY_BUFFER_BYTES:
                        self.insufficient_space_error.emit(SAFETY_BUFFER_BYTES, free)
                        return
                    self.status_message.emit(
                        "File size unknown — free space was not fully verified in advance."
                    )

                if self._is_cancelled:
                    self.download_cancelled.emit()
                    return

                done = 0
                try:
                    with open(tmp_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 512):
                            if self._is_cancelled:
                                break
                            if not chunk:
                                continue
                            try:
                                f.write(chunk)
                            except OSError as e:
                                logger.exception("Write failed: %s", e)
                                _unlink_quiet(tmp_path)
                                self.failed.emit(f"Disk write failed: {e}")
                                return
                            done += len(chunk)
                            if total > 0:
                                pct = int(done * 100 / total)
                                self.progress_pct.emit(min(100, pct))
                            else:
                                self.progress_pct.emit(min(99, done // (1024 * 1024)))

                    if self._is_cancelled:
                        _unlink_quiet(tmp_path)
                        self.download_cancelled.emit()
                        return

                    try:
                        os.replace(tmp_path, out_path)
                    except OSError as e:
                        logger.exception("Could not finalize download: %s", e)
                        _unlink_quiet(tmp_path)
                        self.failed.emit(f"Could not move file into place: {e}")
                        return
                except OSError as e:
                    logger.exception("Download I/O error: %s", e)
                    _unlink_quiet(tmp_path)
                    self.failed.emit(str(e))
                    return

        except requests.RequestException as e:
            logger.exception("HF download request failed: %s", e)
            _unlink_quiet(tmp_path)
            self.failed.emit(str(e))
            return
        except Exception as e:
            logger.exception("HF download failed: %s", e)
            _unlink_quiet(tmp_path)
            self.failed.emit(str(e))
            return

        self.progress_pct.emit(100)
        self.finished_ok.emit(str(out_path))
        self.status_message.emit(f"Saved: {out_path.name}")
