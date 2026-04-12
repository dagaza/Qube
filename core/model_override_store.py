"""
Persistent per-model learned overrides from ablation self-heal (JSON on disk).

Does not change sampling or execution_policy — read by prompt_template_router for
prompt/stop/anchor adjustments only.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

OVERRIDE_PATH = os.path.expanduser("~/.qube/model_overrides.json")


@dataclass
class LearnedOverride:
    model_name: str
    force_execution_mode: Optional[str]
    enforcement_mode: Optional[str]
    strip_thinking: Optional[bool]
    extra_stop_tokens: list[str]
    enforce_assistant_anchor: bool


def load_overrides() -> dict[str, Any]:
    if not os.path.exists(OVERRIDE_PATH):
        return {}
    with open(OVERRIDE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_overrides(data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(OVERRIDE_PATH) or ".", exist_ok=True)
    with open(OVERRIDE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_override(model_name: str) -> Optional[LearnedOverride]:
    data = load_overrides()
    raw = data.get(model_name)
    if not isinstance(raw, dict):
        return None
    return LearnedOverride(
        model_name=str(raw.get("model_name", model_name)),
        force_execution_mode=raw.get("force_execution_mode"),
        enforcement_mode=raw.get("enforcement_mode"),
        strip_thinking=raw.get("strip_thinking"),
        extra_stop_tokens=list(raw.get("extra_stop_tokens") or []),
        enforce_assistant_anchor=bool(raw.get("enforce_assistant_anchor", False)),
    )


def store_override(override: LearnedOverride) -> None:
    data = load_overrides()
    data[override.model_name] = asdict(override)
    save_overrides(data)
