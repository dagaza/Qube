"""
Merge explicit settings, inferred profile, and session overrides into a
deterministic preference policy for execution-layer transforms.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from core.app_settings import (
    get_profile_display_name,
    get_profile_locale,
    get_profile_units,
    get_profile_verbosity,
)
from core.user_profile import get_user_profile_store

logger = logging.getLogger("Qube.PreferencePolicy")

Provenance = Literal["explicit", "inferred", "session", "default"]

PRESENTATION_KEYS = (
    "units",
    "temperature",
    "wind_speed",
    "locale",
    "language",
    "verbosity",
    "display_name",
)

_WEATHER_HINTS = re.compile(
    r"\b(weather|forecast|temperature|wind|rain|snow|humidity|celsius|fahrenheit|"
    r"km/h|mph|today'?s?\s+weather)\b",
    re.I,
)


@dataclass(frozen=True)
class PreferenceField:
    value: str
    provenance: Provenance


@dataclass
class PreferencePolicy:
    """Resolved presentation preferences for one turn."""

    fields: dict[str, PreferenceField] = field(default_factory=dict)
    session_overrides: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> Optional[str]:
        pf = self.fields.get(key)
        return pf.value if pf else None

    def provenance_of(self, key: str) -> Provenance:
        pf = self.fields.get(key)
        return pf.provenance if pf else "default"

    def has_presentation_prefs(self) -> bool:
        return bool(self.fields)

    def units_system(self) -> Optional[str]:
        units = (self.get("units") or "").lower()
        if units in ("metric", "imperial"):
            return units
        temp = (self.get("temperature") or "").lower()
        if temp in ("celsius", "fahrenheit"):
            return "metric" if temp == "celsius" else "imperial"
        return None

    def compact_prompt_context(self, *, query: str = "", route: str = "") -> str:
        """Thin prose hint for ambiguity; max ~120 chars."""
        if not self.has_presentation_prefs():
            return ""
        parts: list[str] = []
        units = self.units_system()
        if units:
            parts.append(f"{units} units")
        name = self.get("display_name")
        if name:
            parts.append(f'call user "{name}"')
        verbosity = self.get("verbosity")
        if verbosity:
            parts.append(f"{verbosity} tone")
        if not parts:
            return ""
        route_u = str(route or "").upper()
        query_l = (query or "").lower()
        needs_hint = route_u in ("WEB", "INTERNET", "NONE") or _WEATHER_HINTS.search(query_l)
        if not needs_hint:
            return ""
        text = "User prefs: " + ", ".join(parts[:3]) + ". Apply silently."
        return text[:120]


def _explicit_fields() -> dict[str, PreferenceField]:
    out: dict[str, PreferenceField] = {}
    units = get_profile_units()
    if units:
        out["units"] = PreferenceField(str(units), "explicit")
        if units == "metric":
            out.setdefault("temperature", PreferenceField("celsius", "explicit"))
            out.setdefault("wind_speed", PreferenceField("kmh", "explicit"))
        elif units == "imperial":
            out.setdefault("temperature", PreferenceField("fahrenheit", "explicit"))
            out.setdefault("wind_speed", PreferenceField("mph", "explicit"))
    locale = get_profile_locale()
    if locale:
        out["locale"] = PreferenceField(str(locale), "explicit")
        lang = str(locale).split("-")[0]
        if lang:
            out.setdefault("language", PreferenceField(lang, "explicit"))
    verbosity = get_profile_verbosity()
    if verbosity:
        out["verbosity"] = PreferenceField(str(verbosity), "explicit")
    display_name = get_profile_display_name()
    if display_name:
        out["display_name"] = PreferenceField(str(display_name), "explicit")
    return out


def _inferred_fields() -> dict[str, PreferenceField]:
    out: dict[str, PreferenceField] = {}
    prefs = get_user_profile_store().get_inferred_preferences()
    for key, entry in prefs.items():
        if not isinstance(entry, dict):
            continue
        val = str(entry.get("value") or "").strip()
        if not val:
            continue
        out[str(key)] = PreferenceField(val, "inferred")
    return out


def resolve_preference_policy(
    session_overrides: Optional[dict[str, str]] = None,
) -> PreferencePolicy:
    """
    Merge precedence: session > explicit settings > inferred > default (omit).
    """
    merged: dict[str, PreferenceField] = {}
    merged.update(_inferred_fields())
    merged.update(_explicit_fields())
    overrides = dict(session_overrides or {})
    for key, val in overrides.items():
        v = str(val or "").strip()
        if v:
            merged[str(key)] = PreferenceField(v, "session")
    return PreferencePolicy(fields=merged, session_overrides=overrides)


def apply_tool_policy(
    query: str,
    policy: PreferencePolicy,
    *,
    tool: str = "internet",
) -> str:
    """Augment tool queries from presentation policy (execution layer)."""
    q = (query or "").strip()
    if not q or tool != "internet":
        return q
    units = policy.units_system()
    if not units:
        return q
    if not _WEATHER_HINTS.search(q):
        return q
    if units == "metric":
        augmented = f"{q} celsius km/h metric"
    else:
        augmented = f"{q} fahrenheit mph imperial"
    logger.info(
        "[PreferencePolicy] tool=%s units=%s query_augmented=true",
        tool,
        units,
    )
    return augmented


__all__ = [
    "PreferencePolicy",
    "PreferenceField",
    "Provenance",
    "PRESENTATION_KEYS",
    "resolve_preference_policy",
    "apply_tool_policy",
]
