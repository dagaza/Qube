"""Background fetch of model metadata from Hugging Face Hub."""

from __future__ import annotations

import logging
import re
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger("Qube.HFModelMeta")


def _card_to_dict(card: Any) -> dict[str, Any]:
    if isinstance(card, dict):
        return card
    if card is None:
        return {}
    to_dict = getattr(card, "to_dict", None)
    if callable(to_dict):
        try:
            out = to_dict()
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    return {}


def _norm_tokens(values: list[Any]) -> list[str]:
    out: list[str] = []
    for v in values:
        s = str(v or "").strip()
        if s:
            out.append(s)
    return out


def _infer_params(card: dict[str, Any], tags: list[str], rid: str) -> str:
    for key in ("params", "parameter_count", "parameters", "model_size"):
        v = card.get(key)
        if isinstance(v, (int, float)) and v > 0:
            b = float(v)
            return f"{int(round(b))}B" if b >= 1 else f"{b:.1f}B"
        if isinstance(v, str) and v.strip():
            return v.strip().upper().replace(" ", "")
    hay = " ".join(tags + [rid]).lower()
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*b\b", hay)
    if m:
        return f"{m.group(1)}B".replace(".0B", "B")
    return "Unknown"


def _infer_arch(card: dict[str, Any], tags: list[str], rid: str) -> str:
    for key in ("architecture", "model_type"):
        v = card.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    known = ("gemma", "llama", "mistral", "qwen", "phi", "deepseek", "mixtral")
    hay = " ".join(tags + [rid]).lower()
    for k in known:
        if k in hay:
            return "gemma4" if k == "gemma" and "gemma-4" in hay else k
    return "Unknown"


def _infer_domain(card: dict[str, Any], tags: list[str]) -> str:
    v = card.get("domain")
    if isinstance(v, str) and v.strip():
        return v.strip()
    tl = [t.lower() for t in tags]
    if any(t in tl for t in ("vision", "multimodal", "image-text-to-text")):
        return "multimodal"
    if "audio" in tl:
        return "audio"
    if "text-generation" in tl:
        return "llm"
    return "llm"


def _infer_format(tags: list[str]) -> str:
    tl = [t.lower() for t in tags]
    if "gguf" in tl:
        return "GGUF"
    if "safetensors" in tl:
        return "safetensors"
    return "Unknown"


def _infer_capabilities(tags: list[str], card: dict[str, Any]) -> list[str]:
    tl = [t.lower() for t in tags]
    text_pool: list[str] = list(tl)
    for key in ("pipeline_tag", "library_name", "base_model", "tags", "model_name"):
        v = card.get(key)
        if isinstance(v, str) and v.strip():
            text_pool.append(v.lower())
        elif isinstance(v, list):
            text_pool.extend(str(x).lower() for x in v if str(x).strip())
    corpus = " ".join(text_pool)

    caps: list[str] = []

    def add_if(label: str, needles: tuple[str, ...]) -> None:
        if any(n in corpus for n in needles) and label not in caps:
            caps.append(label)

    add_if("Vision", ("vision", "image-text-to-text", "multimodal", "vl", "vlm"))
    add_if("Tool Use", ("tool-use", "tool use", "function-calling", "function calling", "tools"))
    add_if("Reasoning", ("reasoning", "chain-of-thought", "cot", "thinking"))
    add_if("Coding", ("code", "coding", "codegen"))
    add_if("TTS", ("text-to-speech", "text to speech", "tts"))
    add_if("STT", ("automatic-speech-recognition", "speech-to-text", "stt", "asr"))
    add_if("Audio", ("audio", "speech", "voice"))
    add_if("Multilingual", ("multilingual",))

    # Do not force "Text Generation" as a generic fallback; show only meaningful capabilities.
    return caps


class HfModelMetaWorker(QThread):
    """Fetches model metadata from Hub model_info/card_data."""

    finished_ok = pyqtSignal(str, dict)  # repo_id, metadata
    failed = pyqtSignal(str, str)  # repo_id, error message

    def __init__(self, repo_id: str):
        super().__init__()
        self._repo_id = (repo_id or "").strip()

    def run(self) -> None:
        repo = self._repo_id
        if not repo:
            self.failed.emit("", "Empty repository id.")
            return
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            # model_info() targets model repos; avoid extra kwargs that vary by client version.
            info = api.model_info(repo)
            if self.isInterruptionRequested():
                return
            card = _card_to_dict(getattr(info, "card_data", None))
            tags = _norm_tokens(getattr(info, "tags", None) or [])
            meta = {
                "params": _infer_params(card, tags, repo),
                "arch": _infer_arch(card, tags, repo),
                "domain": _infer_domain(card, tags),
                "format": _infer_format(tags),
                "capabilities": _infer_capabilities(tags, card),
            }
            self.finished_ok.emit(repo, meta)
        except Exception as e:
            logger.debug("model_info metadata fetch failed for %s: %s", repo, e)
            # Fallback: infer what we can from a lightweight list_models lookup.
            try:
                from huggingface_hub import HfApi

                api = HfApi()
                m = next(api.list_models(search=repo, full=True, limit=20), None)
                if m is not None:
                    rid = str(getattr(m, "modelId", None) or getattr(m, "id", "") or repo)
                    card = _card_to_dict(getattr(m, "card_data", None))
                    tags = _norm_tokens(getattr(m, "tags", None) or [])
                    meta = {
                        "params": _infer_params(card, tags, rid),
                        "arch": _infer_arch(card, tags, rid),
                        "domain": _infer_domain(card, tags),
                        "format": _infer_format(tags),
                        "capabilities": _infer_capabilities(tags, card),
                    }
                    self.finished_ok.emit(repo, meta)
                    return
            except Exception as sub_e:
                logger.debug("metadata fallback failed for %s: %s", repo, sub_e)
            self.failed.emit(repo, str(e))
