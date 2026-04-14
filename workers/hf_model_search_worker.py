"""Background search on Hugging Face Hub for GGUF-tagged models."""

from __future__ import annotations

import logging
from datetime import datetime
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
    Emits list of dicts with repo/title plus lightweight metadata for list cards.
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
                description = ""
                capabilities: list[str] = []
                updated_at = ""
                try:
                    card = getattr(m, "card_data", None)
                    if isinstance(card, dict):
                        d = card.get("description") or card.get("summary")
                        if isinstance(d, str):
                            description = d.strip()
                        caps = card.get("capabilities")
                        if isinstance(caps, list):
                            capabilities = [str(c).strip() for c in caps if str(c).strip()]
                        elif isinstance(caps, str) and caps.strip():
                            capabilities = [caps.strip()]
                except Exception:
                    pass
                if not description:
                    try:
                        mid = getattr(m, "id", None)
                        if isinstance(mid, str):
                            description = mid.strip()
                    except Exception:
                        pass
                if not capabilities:
                    try:
                        tags = getattr(m, "tags", None) or []
                        tset = {str(t).lower() for t in tags}
                        if "text-generation" in tset:
                            capabilities.append("Text")
                        if "conversational" in tset:
                            capabilities.append("Chat")
                        if "vision" in tset:
                            capabilities.append("Vision")
                        if "tool-use" in tset or "tools" in tset:
                            capabilities.append("Tools")
                    except Exception:
                        pass
                try:
                    lm = getattr(m, "lastModified", None)
                    if isinstance(lm, datetime):
                        updated_at = lm.strftime("%Y-%m-%d")
                    elif lm is not None:
                        updated_at = str(lm).strip()
                except Exception:
                    pass
                out.append(
                    {
                        "repo_id": rid,
                        "title": title,
                        "description": description,
                        "capabilities": capabilities[:3],
                        "updated_at": updated_at,
                        "hf_pipeline_tag": str(getattr(m, "pipeline_tag", "") or ""),
                        "hf_tags": [str(t) for t in (getattr(m, "tags", None) or []) if str(t).strip()],
                    }
                )
                if len(out) >= self._limit:
                    break
        except Exception as e:
            logger.exception("Hub search iteration failed: %s", e)
            self.failed.emit(str(e), self._seq)
            return

        self.finished_ok.emit(out, self._seq)
