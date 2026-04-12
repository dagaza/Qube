"""Background search on Hugging Face Hub for GGUF-tagged models."""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger("Qube.HFModelSearch")


def _model_id(m: Any) -> str:
    return str(getattr(m, "modelId", None) or getattr(m, "id", "") or "").strip()


def _is_gguf_model(m: Any) -> bool:
    tags = getattr(m, "tags", None) or []
    lowered = [str(t).lower() for t in tags]
    if "gguf" in lowered:
        return True
    mid = _model_id(m).lower()
    return "gguf" in mid


class HfModelSearchWorker(QThread):
    """
    Queries Hub for models and keeps those that look GGUF-related (tags or repo id).
    Emits list of dicts: {"repo_id": str, "title": str} where title is repo_id or card name.
    """

    finished_ok = pyqtSignal(list, int)  # models, request_seq
    failed = pyqtSignal(str, int)  # message, request_seq

    def __init__(self, query: str, request_seq: int, limit: int = 40):
        super().__init__()
        self._query = (query or "").strip()
        self._seq = int(request_seq)
        self._limit = max(5, min(100, int(limit)))

    def run(self) -> None:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            kwargs: dict = {"limit": 200, "full": True}
            if self._query:
                kwargs["search"] = self._query
            it = api.list_models(**kwargs)
        except Exception as e:
            logger.exception("HfApi.list_models failed: %s", e)
            self.failed.emit(str(e), self._seq)
            return

        out: list[dict] = []
        seen: set[str] = set()
        scanned = 0
        max_scan = 800
        try:
            for m in it:
                if scanned % 32 == 0 and self.isInterruptionRequested():
                    return
                scanned += 1
                if scanned > max_scan:
                    break
                if not _is_gguf_model(m):
                    continue
                rid = _model_id(m)
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                title = rid
                try:
                    card = getattr(m, "card_data", None)
                    if isinstance(card, dict):
                        t = card.get("model_name") or card.get("name")
                        if isinstance(t, str) and t.strip():
                            title = t.strip()
                except Exception:
                    pass
                out.append({"repo_id": rid, "title": title})
                if len(out) >= self._limit:
                    break
        except Exception as e:
            logger.exception("Hub search iteration failed: %s", e)
            self.failed.emit(str(e), self._seq)
            return

        self.finished_ok.emit(out, self._seq)
