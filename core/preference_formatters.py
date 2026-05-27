"""
Post-process tool output using presentation preference policy.
"""
from __future__ import annotations

import re
from typing import Any

from core.preference_policy import PreferencePolicy

_F_TO_C = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*F\b", re.I)
_MPH_TO_KMH = re.compile(r"(-?\d+(?:\.\d+)?)\s*mph\b", re.I)
_C_TO_F = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*C\b", re.I)
_KMH_TO_MPH = re.compile(r"(-?\d+(?:\.\d+)?)\s*km/h\b", re.I)


def _f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _append_metric(snippet: str) -> str:
    def repl_f(m: re.Match) -> str:
        val = float(m.group(1))
        c = _f_to_c(val)
        return f"{m.group(0)} ({c:.0f}°C)"

    def repl_mph(m: re.Match) -> str:
        val = float(m.group(1))
        kmh = val * 1.60934
        return f"{m.group(0)} ({kmh:.0f} km/h)"

    out = _F_TO_C.sub(repl_f, snippet)
    return _MPH_TO_KMH.sub(repl_mph, out)


def _append_imperial(snippet: str) -> str:
    def repl_c(m: re.Match) -> str:
        val = float(m.group(1))
        f = _c_to_f(val)
        return f"{m.group(0)} ({f:.0f}°F)"

    def repl_kmh(m: re.Match) -> str:
        val = float(m.group(1))
        mph = val / 1.60934
        return f"{m.group(0)} ({mph:.0f} mph)"

    out = _C_TO_F.sub(repl_c, snippet)
    return _KMH_TO_MPH.sub(repl_kmh, out)


def format_web_snippets(
    snippets: list[dict[str, Any]],
    policy: PreferencePolicy,
) -> list[dict[str, Any]]:
    """Conservatively append unit equivalents in parentheses."""
    units = policy.units_system()
    if not units or not snippets:
        return snippets
    out: list[dict[str, Any]] = []
    for item in snippets:
        row = dict(item)
        snippet = str(row.get("snippet") or "")
        title = str(row.get("title") or "")
        if units == "metric":
            row["snippet"] = _append_metric(snippet)
            row["title"] = _append_metric(title)
        else:
            row["snippet"] = _append_imperial(snippet)
            row["title"] = _append_imperial(title)
        out.append(row)
    return out


__all__ = ["format_web_snippets"]
