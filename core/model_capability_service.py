from __future__ import annotations

from typing import Any

from core.model_capability_detection import (
    CAPABILITY_KEYS,
    ModelCapabilities,
    confidence_tier,
    detect_capabilities,
)
from core.system_capabilities_store import SystemCapabilitiesStore


class ModelCapabilityService:
    def __init__(self, store: SystemCapabilitiesStore | None = None):
        self.store = store or SystemCapabilitiesStore()

    def get_or_detect(
        self,
        model: dict[str, Any],
        *,
        force_refresh: bool = False,
    ) -> ModelCapabilities:
        model_id = str(model.get("id") or model.get("model_id") or "").strip()
        if not model_id:
            raise ValueError("Model id is required.")

        if not force_refresh:
            cached = self.store.get_cached_capabilities(model_id)
            if cached is not None:
                # Auto-heal cache across capability-schema expansions.
                if all(k in cached for k in ("reasoning", "tool_use", "vision", "tts", "stt", "coding", "audio")):
                    return ModelCapabilities.from_dict(cached)

        override = self.store.get_override(model_id)
        ground_truth = self.store.get_ground_truth(model_id)
        learned = self.store.load_learned_registry()
        curated = self.store.load_curated_registry()
        detected = detect_capabilities(
            model,
            user_overrides=override,
            ground_truth=ground_truth,
            learned_registry=learned,
            curated_registry=curated,
        )
        if not any(
            (
                detected.reasoning.value,
                detected.tool_use.value,
                detected.vision.value,
                detected.tts.value,
                detected.stt.value,
                detected.coding.value,
            )
        ):
            hf = model.get("huggingface") or {}
            self.store.append_missed_detection(
                {
                    "model_id": model_id,
                    "name": str(model.get("name") or ""),
                    "hf_tags": list(hf.get("tags") or []),
                    "pipeline_tag": str(hf.get("pipeline_tag") or ""),
                    "had_readme": bool(str(model.get("readme") or "").strip()),
                }
            )
        self.store.upsert_capabilities(model_id, detected.to_dict())
        return detected

    def on_model_loaded(self, model_id: str, runtime_capabilities: dict[str, bool]) -> ModelCapabilities:
        self.store.set_ground_truth(model_id, runtime_capabilities, source="runtime_detection")
        self.store.upsert_learned_capabilities(model_id, runtime_capabilities)
        return self.get_or_detect({"id": model_id}, force_refresh=True)

    def enrich_capabilities_after_load(self, model_id: str, model_data: dict[str, Any]) -> ModelCapabilities:
        payload = dict(model_data or {})
        payload["id"] = model_id
        # Re-run with richer post-load inputs (readme/runtime metadata) and persist.
        prev = self.get_capabilities(model_id)
        fresh = self.get_or_detect(payload, force_refresh=True)
        if prev is None:
            return fresh
        merged = self._merge_capabilities(prev, fresh)
        self.store.upsert_capabilities(model_id, merged.to_dict())
        return merged

    def set_override(self, model_id: str, overrides: dict[str, bool]) -> None:
        clean: dict[str, bool] = {}
        for k, v in overrides.items():
            key = str(k).strip()
            if key in CAPABILITY_KEYS:
                clean[key] = bool(v)
        if not clean:
            return
        self.store.set_override(model_id, clean)

    def get_override(self, model_id: str) -> dict[str, bool]:
        return self.store.get_override(model_id)

    def get_capabilities(self, model_id: str) -> ModelCapabilities | None:
        cached = self.store.get_cached_capabilities(model_id)
        if cached is None:
            return None
        return ModelCapabilities.from_dict(cached)

    def summarize_for_ui(self, caps: ModelCapabilities) -> dict[str, dict[str, Any]]:
        raw = caps.to_dict()
        out: dict[str, dict[str, Any]] = {}
        for key in CAPABILITY_KEYS:
            agg = raw[key]
            out[key] = {
                "value": bool(agg["value"]),
                "confidence": float(agg["confidence"]),
                "tier": confidence_tier(float(agg["confidence"])),
                "source": (agg["sources"][0]["source"] if agg["sources"] else "unknown"),
            }
        audio = raw.get("audio") or {}
        out["audio"] = {
            "value": bool(audio.get("value", False)),
            "confidence": float(audio.get("confidence", 0.0)),
            "tier": confidence_tier(float(audio.get("confidence", 0.0))),
            "source": (audio.get("sources", [{}])[0].get("source", "unknown") if audio.get("sources") else "unknown"),
        }
        return out

    @staticmethod
    def _merge_capabilities(old: ModelCapabilities, new: ModelCapabilities) -> ModelCapabilities:
        old_raw = old.to_dict()
        new_raw = new.to_dict()
        merged: dict[str, Any] = {}
        for key in ("reasoning", "tool_use", "vision", "tts", "stt", "coding", "audio"):
            o = old_raw.get(key) or {"value": False, "confidence": 0.0, "sources": []}
            n = new_raw.get(key) or {"value": False, "confidence": 0.0, "sources": []}
            if bool(o.get("value")) and not bool(n.get("value")):
                merged[key] = o
            elif bool(n.get("value")) and not bool(o.get("value")):
                merged[key] = n
            else:
                merged[key] = n if float(n.get("confidence", 0.0)) >= float(o.get("confidence", 0.0)) else o
        return ModelCapabilities.from_dict(merged)
