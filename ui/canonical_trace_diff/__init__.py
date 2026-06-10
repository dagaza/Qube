"""Canonical trace diff debugger UI (lazy imports — avoids PyQt6 at package import)."""
from __future__ import annotations

from typing import Any

__all__ = [
    "CanonicalTraceDiffView",
    "load_trace_pair",
    "load_scenario_run_pair_view",
    "open_canonical_trace_diff_window",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from ui.canonical_trace_diff import trace_diff_view as _mod

        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
