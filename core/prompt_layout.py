"""
Prompt layout resolution: where instructional text is placed in the message list.

Phase 1 (PR1): resolve + telemetry only — default ``system_ok``; no message-shape changes.
Curated family defaults and per-model overrides; no dynamic probing.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from core.model_capability_detection import match_pattern, normalize_model_id
from core.prompt_layout_store import get_override as get_layout_override

logger = logging.getLogger("Qube.PromptLayout")

PromptLayout = Literal["system_ok", "short_system", "flatten_user"]

VALID_PROMPT_LAYOUTS: tuple[PromptLayout, ...] = (
    "system_ok",
    "short_system",
    "flatten_user",
)

DEFAULT_PROMPT_LAYOUT: PromptLayout = "system_ok"


@dataclass(frozen=True)
class PromptLayoutResolution:
    layout: PromptLayout
    source: str
    degraded: bool
    evidence: str = ""


def normalize_prompt_layout(value: Any) -> Optional[PromptLayout]:
    s = str(value or "").strip().lower()
    if s in VALID_PROMPT_LAYOUTS:
        return s  # type: ignore[return-value]
    return None


def is_degraded_layout(layout: PromptLayout) -> bool:
    return layout in ("short_system", "flatten_user")


def _workspace_seed_registry_path() -> Path:
    return Path(__file__).resolve().parent.parent / "system_data" / "curated_registry.json"


def _user_registry_path() -> Path:
    return Path.home() / ".qube" / "system_data" / "curated_registry.json"


def _read_registry_file(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def load_curated_prompt_layout_registry() -> dict[str, Any]:
    """
    Load ``prompt_layout`` section from curated registry (user copy merged with seed).

    Returns ``{"exact": {...}, "patterns": [...]}``.
    """
    seed = _read_registry_file(_workspace_seed_registry_path())
    user = _read_registry_file(_user_registry_path())
    seed_pl = seed.get("prompt_layout") if isinstance(seed.get("prompt_layout"), dict) else {}
    user_pl = user.get("prompt_layout") if isinstance(user.get("prompt_layout"), dict) else {}

    exact: dict[str, str] = {}
    for src in (seed_pl, user_pl):
        if not isinstance(src, dict):
            continue
        for k, v in (src.get("exact") or {}).items():
            if not isinstance(v, str):
                if isinstance(v, dict) and "layout" in v:
                    lay = normalize_prompt_layout(v.get("layout"))
                else:
                    lay = None
            else:
                lay = normalize_prompt_layout(v)
            if lay:
                exact[str(k).strip().lower()] = lay

    patterns: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for src in (seed_pl, user_pl):
        if not isinstance(src, dict):
            continue
        for p in src.get("patterns") or []:
            if not isinstance(p, dict):
                continue
            match = str(p.get("match") or "").strip().lower()
            ptype = str(p.get("type") or "contains").strip().lower()
            lay = normalize_prompt_layout(p.get("layout"))
            if not match or not lay:
                continue
            key = (match, ptype)
            if key in seen:
                continue
            seen.add(key)
            patterns.append({"match": match, "type": ptype, "layout": lay})

    return {"exact": exact, "patterns": patterns}


def _identity_blob(
    *,
    model_id: str,
    model_display_name: str,
    model_path: str,
) -> str:
    parts = [
        str(model_id or "").strip().lower(),
        str(model_display_name or "").strip().lower(),
        os.path.basename(str(model_path or "").strip()).lower(),
    ]
    return " ".join(p for p in parts if p)


def _family_layout(ident: str) -> Optional[tuple[PromptLayout, str]]:
    """Conservative name/path heuristics when curated rules miss."""
    n = ident.lower()
    # Mistral 7B Instruct v0.x: weak/no system-role adherence in common GGUF builds.
    if "mistral" in n and "instruct" in n:
        if any(
            tok in n
            for tok in (
                "v0.1",
                "v0.2",
                "v0.3",
                "instruct-v0",
                "7b-instruct",
                "7b_instruct",
            )
        ):
            return "flatten_user", "family:mistral_instruct_v0"
    for token in ("alpaca", "vicuna", "wizard", "guanaco", "openbuddy"):
        if token in n:
            return "flatten_user", f"family:contains:{token}"
    for token in ("orca-mini", "tinyllama", "phi-1", "phi-2", "phi1", "phi2"):
        if token in n:
            return "short_system", f"family:contains:{token}"
    return None


def _curated_exact_layout(
    registry: dict[str, Any],
    *,
    model_id: str,
    normalized_model_id: str,
) -> Optional[tuple[PromptLayout, str]]:
    exact = registry.get("exact") or {}
    if not isinstance(exact, dict):
        return None
    candidates = [
        model_id.lower().strip(),
        normalized_model_id,
        model_id.lower().strip().split("/")[-1],
    ]
    for c in candidates:
        if c and c in exact:
            return exact[c], f"curated:exact:{c}"
    return None


def _curated_pattern_layout(
    registry: dict[str, Any],
    *,
    model_id: str,
    model_display_name: str,
    normalized_model_id: str,
) -> Optional[tuple[PromptLayout, str]]:
    patterns = registry.get("patterns") or []
    model_stub = {"name": model_display_name}
    raw_id = model_id.lower().strip()
    for p in patterns:
        if not isinstance(p, dict):
            continue
        lay = normalize_prompt_layout(p.get("layout"))
        if not lay:
            continue
        pat = {"match": p.get("match"), "type": p.get("type", "contains")}
        if match_pattern(raw_id, model_display_name, pat) or match_pattern(
            normalized_model_id, model_display_name, pat
        ) or match_pattern(raw_id.split("/")[-1], model_display_name, pat):
            ev = f"curated:pattern:{pat.get('type')}:{pat.get('match')}"
            return lay, ev
    return None


def resolve_prompt_layout(
    *,
    model_id: str = "",
    model_display_name: str = "",
    model_path: str = "",
    settings_override: Optional[str] = None,
    store_override: Optional[PromptLayout] = None,
    curated_registry: Optional[dict[str, Any]] = None,
) -> PromptLayoutResolution:
    """
    Resolve layout for a model. Priority:

    1. ``settings_override`` (when not ``auto``)
    2. ``store_override`` / ``prompt_layout_store`` for ``model_id``
    3. Curated exact
    4. Curated pattern
    5. Family heuristic
    6. ``system_ok`` default
    """
    mid = str(model_id or "").strip() or os.path.basename(str(model_path or "").strip())
    norm = normalize_model_id(mid) if mid else ""
    display = str(model_display_name or "").strip() or mid

    so = normalize_prompt_layout(settings_override) if settings_override not in (
        None,
        "",
        "auto",
    ) else None
    if so is not None:
        return PromptLayoutResolution(
            layout=so,
            source="settings",
            degraded=is_degraded_layout(so),
            evidence=f"settings={so}",
        )

    if store_override is None and mid:
        store_override = normalize_prompt_layout(get_layout_override(mid))
        if store_override is None and display:
            store_override = normalize_prompt_layout(get_layout_override(display))

    if store_override is not None:
        return PromptLayoutResolution(
            layout=store_override,
            source="user_override",
            degraded=is_degraded_layout(store_override),
            evidence=f"store:{mid or display}",
        )

    registry = curated_registry if curated_registry is not None else load_curated_prompt_layout_registry()

    exact_hit = _curated_exact_layout(registry, model_id=mid, normalized_model_id=norm)
    if exact_hit:
        lay, ev = exact_hit
        return PromptLayoutResolution(
            layout=lay,
            source="curated",
            degraded=is_degraded_layout(lay),
            evidence=ev,
        )

    pat_hit = _curated_pattern_layout(
        registry,
        model_id=mid,
        model_display_name=display,
        normalized_model_id=norm,
    )
    if pat_hit:
        lay, ev = pat_hit
        return PromptLayoutResolution(
            layout=lay,
            source="curated_pattern",
            degraded=is_degraded_layout(lay),
            evidence=ev,
        )

    ident = _identity_blob(model_id=mid, model_display_name=display, model_path=model_path)
    fam = _family_layout(ident)
    if fam:
        lay, ev = fam
        return PromptLayoutResolution(
            layout=lay,
            source="family",
            degraded=is_degraded_layout(lay),
            evidence=ev,
        )

    return PromptLayoutResolution(
        layout=DEFAULT_PROMPT_LAYOUT,
        source="default",
        degraded=False,
        evidence="default:system_ok",
    )
