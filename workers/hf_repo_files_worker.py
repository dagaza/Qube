"""Background fetch of .gguf filenames (and sizes) from a Hugging Face model repo (Hub API)."""

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
    """Lists repo files via huggingface_hub; filters to .gguf paths with sizes when available."""

    # list[tuple[str, int | None]] — path, size in bytes (None if unknown)
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, repo_id: str, revision: Optional[str] = None):
        super().__init__()
        self._repo_id = repo_id
        self._revision = revision

    def _emit_from_paths_with_sizes(
        self,
        api: object,
        repo: str,
        paths: list[str],
    ) -> bool:
        """Batch paths-info for .gguf files. Returns True if emitted."""
        if not paths:
            self.finished_ok.emit([])
            return True
        try:
            infos = api.get_paths_info(
                repo,
                paths,
                revision=self._revision,
                repo_type="model",
            )
        except Exception as e:
            logger.warning("get_paths_info failed: %s", e)
            return False
        from huggingface_hub.hf_api import RepoFile

        size_by_path: dict[str, int | None] = {}
        for info in infos:
            if isinstance(info, RepoFile):
                p = str(info.path)
                sz = info.size
                size_by_path[p] = int(sz) if sz is not None else None
        out: list[tuple[str, int | None]] = []
        for p in paths:
            out.append((p, size_by_path.get(p)))
        out.sort(key=lambda x: x[0].lower())
        self.finished_ok.emit(out)
        return True

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
            from huggingface_hub.hf_api import RepoFile

            api = HfApi()
            tree = api.list_repo_tree(
                repo,
                recursive=True,
                revision=self._revision,
                repo_type="model",
            )
            out: list[tuple[str, int | None]] = []
            for entry in tree:
                if isinstance(entry, RepoFile):
                    p = entry.path
                    if not str(p).lower().endswith(".gguf"):
                        continue
                    sz = entry.size
                    out.append((str(p), int(sz) if sz is not None else None))
            out.sort(key=lambda x: x[0].lower())
            if out and all(x[1] is None for x in out):
                paths_only = [p for p, _ in out]
                if self._emit_from_paths_with_sizes(api, repo, paths_only):
                    return
            self.finished_ok.emit(out)
            return
        except Exception as e:
            logger.warning("list_repo_tree failed, trying model_info: %s", e)

        try:
            if self.isInterruptionRequested():
                return
            from huggingface_hub import HfApi

            api = HfApi()
            info = api.model_info(
                repo,
                repo_type="model",
                revision=self._revision,
                files_metadata=True,
            )
            out_mi: list[tuple[str, int | None]] = []
            for s in info.siblings:
                rf = s.rfilename
                if not str(rf).lower().endswith(".gguf"):
                    continue
                sz = s.size
                out_mi.append((str(rf), int(sz) if sz is not None else None))
            out_mi.sort(key=lambda x: x[0].lower())
            if not out_mi:
                self.finished_ok.emit([])
                return
            if all(x[1] is None for x in out_mi):
                paths_only = [p for p, _ in out_mi]
                if self._emit_from_paths_with_sizes(api, repo, paths_only):
                    return
            self.finished_ok.emit(out_mi)
            return
        except Exception as e:
            logger.warning("model_info(files_metadata) failed, falling back to paths: %s", e)

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
        if not gguf:
            self.finished_ok.emit([])
            return
        if not self._emit_from_paths_with_sizes(api, repo, gguf):
            self.finished_ok.emit([(f, None) for f in gguf])
