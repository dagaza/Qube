from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.model_capability_service import ModelCapabilityService


class OverridePayload(BaseModel):
    reasoning: bool | None = None
    tool_use: bool | None = None
    vision: bool | None = None
    tts: bool | None = None
    stt: bool | None = None
    coding: bool | None = None


class RuntimeCapabilitiesPayload(BaseModel):
    reasoning: bool | None = None
    tool_use: bool | None = None
    vision: bool | None = None
    tts: bool | None = None
    stt: bool | None = None
    coding: bool | None = None


class EnrichPayload(BaseModel):
    name: str | None = None
    description: str | None = None
    readme: str | None = None
    huggingface: dict[str, Any] | None = None
    files: dict[str, Any] | None = None


def create_model_capabilities_app(
    models_provider: Callable[[], list[dict[str, Any]]] | None = None,
    service: ModelCapabilityService | None = None,
) -> FastAPI:
    app = FastAPI(title="Qube Model Capabilities API")
    cap_service = service or ModelCapabilityService()

    def _provider() -> list[dict[str, Any]]:
        if models_provider is None:
            return []
        return models_provider()

    @app.get("/models")
    def list_models() -> list[dict[str, Any]]:
        models = _provider()
        out: list[dict[str, Any]] = []
        for model in models:
            mid = str(model.get("id") or model.get("model_id") or "").strip()
            if not mid:
                continue
            caps = cap_service.get_or_detect(model)
            out.append(
                {
                    "id": mid,
                    "capabilities": cap_service.summarize_for_ui(caps),
                }
            )
        return out

    @app.get("/models/{model_id}/capabilities")
    def get_model_capabilities(model_id: str, debug: bool = False) -> dict[str, Any]:
        model = next(
            (
                m
                for m in _provider()
                if str(m.get("id") or m.get("model_id") or "").strip() == model_id
            ),
            {"id": model_id},
        )
        caps = cap_service.get_or_detect(model)
        if debug:
            return caps.to_dict()
        return cap_service.summarize_for_ui(caps)

    @app.post("/models/{model_id}/override")
    def set_model_override(model_id: str, payload: OverridePayload) -> dict[str, Any]:
        overrides = payload.model_dump(exclude_none=True)
        if not overrides:
            raise HTTPException(status_code=400, detail="No override values provided.")
        cap_service.set_override(model_id, overrides)
        caps = cap_service.get_or_detect({"id": model_id}, force_refresh=True)
        return {
            "id": model_id,
            "override": cap_service.get_override(model_id),
            "capabilities": cap_service.summarize_for_ui(caps),
        }

    @app.post("/models/{model_id}/loaded")
    def on_model_loaded(model_id: str, payload: RuntimeCapabilitiesPayload) -> dict[str, Any]:
        runtime_caps = payload.model_dump(exclude_none=True)
        caps = cap_service.on_model_loaded(model_id, runtime_caps)
        return {"id": model_id, "capabilities": cap_service.summarize_for_ui(caps)}

    @app.post("/models/{model_id}/enrich")
    def enrich_model_capabilities(model_id: str, payload: EnrichPayload) -> dict[str, Any]:
        caps = cap_service.enrich_capabilities_after_load(model_id, payload.model_dump(exclude_none=True))
        return {"id": model_id, "capabilities": cap_service.summarize_for_ui(caps)}

    @app.get("/capabilities/export")
    def export_capabilities_bundle() -> dict[str, Any]:
        return cap_service.store.export_capabilities_bundle()

    return app
