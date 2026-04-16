from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import openwakeword

from core.app_settings import (
    get_active_wakeword_id,
    get_wakeword_threshold_override,
    set_active_wakeword_id,
)

logger = logging.getLogger("Qube.WakewordManager")

_HIDDEN_WAKEWORDS = {"timer", "weather"}


@dataclass
class WakewordSpec:
    wakeword_id: str
    display_name: str
    source: str  # local
    path: str
    default_threshold: float
    recommended: bool
    cache_key: str = ""
    download_url: str = ""
    version: str = ""
    experimental: bool = False

    def threshold(self) -> float:
        override = get_wakeword_threshold_override(self.wakeword_id)
        return float(override) if override is not None else float(self.default_threshold)


class WakewordManager:
    def __init__(self, cache_dir: str = "models/wakeword"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._catalog: dict[str, WakewordSpec] = {}

    def refresh_catalog(self, include_remote: bool = True) -> dict[str, WakewordSpec]:
        _ = include_remote
        catalog = self._discover_local_catalog()
        self._catalog = catalog
        return dict(self._catalog)

    def list_recommended(self) -> list[WakewordSpec]:
        return sorted(
            [spec for spec in self._catalog.values() if spec.recommended],
            key=lambda s: s.display_name.lower(),
        )

    def list_community(self) -> list[WakewordSpec]:
        return sorted(
            [spec for spec in self._catalog.values() if not spec.recommended],
            key=lambda s: s.display_name.lower(),
        )

    def get_active_or_default(self) -> WakewordSpec | None:
        active_id = get_active_wakeword_id()
        if active_id and active_id in self._catalog:
            return self._catalog[active_id]
        recommended = self.list_recommended()
        if recommended:
            return next((w for w in recommended if "jarvis" in w.wakeword_id), recommended[0])
        return next(iter(self._catalog.values()), None)

    def get_by_id(self, wakeword_id: str) -> WakewordSpec | None:
        return self._catalog.get(str(wakeword_id or "").strip())

    def mark_active(self, wakeword_id: str) -> None:
        set_active_wakeword_id(wakeword_id)

    def ensure_model_available(self, spec: WakewordSpec) -> str:
        if os.path.isfile(spec.path):
            return spec.path
        raise FileNotFoundError(f"Wakeword model missing: {spec.path}")

    def _discover_local_catalog(self) -> dict[str, WakewordSpec]:
        out: dict[str, WakewordSpec] = {}
        try:
            pretrained_paths = openwakeword.get_pretrained_model_paths()
        except Exception as exc:
            logger.warning("Failed to discover bundled wakewords: %s", exc)
            pretrained_paths = []
        for path in pretrained_paths:
            stem = self._clean_stem(path)
            if self._is_hidden_wakeword(stem):
                continue
            display = self._display_name(stem)
            wakeword_id = stem.lower()
            out[wakeword_id] = WakewordSpec(
                wakeword_id=wakeword_id,
                display_name=display,
                source="local",
                path=path,
                default_threshold=0.5,
                recommended=True,
            )

        local_files = sorted(self.cache_dir.rglob("*.onnx")) + sorted(self.cache_dir.rglob("*.tflite"))
        for local_file in local_files:
            if not local_file.is_file():
                continue
            stem = self._clean_stem(str(local_file))
            if self._is_hidden_wakeword(stem):
                continue
            wakeword_id = stem.lower()
            if wakeword_id in out:
                continue
            parent_parts = {p.lower() for p in local_file.parts}
            is_community = ("community" in parent_parts) or ("experimental" in parent_parts) or ("en" in parent_parts)
            out[wakeword_id] = WakewordSpec(
                wakeword_id=wakeword_id,
                display_name=self._display_name(stem),
                source="local",
                path=str(local_file),
                default_threshold=0.5,
                recommended=not is_community and not stem.startswith("community_"),
                experimental=is_community or stem.startswith("community_"),
            )
        return out

    @staticmethod
    def _clean_stem(path_or_name: str) -> str:
        stem = Path(path_or_name).stem
        return stem.split("_v")[0].strip()

    @staticmethod
    def _display_name(stem: str) -> str:
        return " ".join(part.capitalize() for part in stem.replace("-", "_").split("_"))

    @staticmethod
    def _is_hidden_wakeword(stem: str) -> bool:
        tokenized = stem.replace("-", "_").lower().strip()
        return tokenized in _HIDDEN_WAKEWORDS

    @staticmethod
    def to_metadata_json(specs: dict[str, WakewordSpec]) -> dict[str, dict[str, Any]]:
        return {key: asdict(spec) for key, spec in specs.items()}
