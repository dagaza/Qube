"""
Temporary GUI-thread probes for tools-pane collapse/expand (Phase A diagnostics).

Enable with ``QUBE_UI_PERF=1``. Optional animation baseline:

- ``QUBE_TOOLS_PANE_ANIM_MS`` — retained for experiments; production tools pane uses instant show/hide (Phase B).

Logs on logger ``Qube.UiPerf`` (stdout when root logging is configured).
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger("Qube.UiPerf")

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_UI_PERF_ENABLED: bool | None = None


def ui_perf_enabled() -> bool:
    global _UI_PERF_ENABLED
    if _UI_PERF_ENABLED is None:
        _UI_PERF_ENABLED = (
            os.environ.get("QUBE_UI_PERF", "").strip().lower() in _TRUTHY
        )
    return _UI_PERF_ENABLED


def tools_pane_anim_ms() -> int:
    """Nominal animation duration; 0 disables width animation (instant toggle)."""
    raw = os.environ.get("QUBE_TOOLS_PANE_ANIM_MS", "").strip()
    if not raw:
        return 350
    try:
        return max(0, int(raw))
    except ValueError:
        return 350


@dataclass
class ToolsPaneToggleProbe:
    target_expanded: bool
    animated: bool
    nominal_anim_ms: int
    started_at: float = field(default_factory=time.perf_counter)
    counters: dict[str, int] = field(default_factory=dict)
    durations_ms: dict[str, float] = field(default_factory=dict)
    anim_ticks_content: int = 0
    anim_ticks_frame: int = 0

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def add_ms(self, key: str, ms: float) -> None:
        self.durations_ms[key] = self.durations_ms.get(key, 0.0) + ms


_active: ToolsPaneToggleProbe | None = None


def active_tools_pane_probe() -> ToolsPaneToggleProbe | None:
    if not ui_perf_enabled():
        return None
    return _active


def begin_tools_pane_toggle(
    *,
    target_expanded: bool,
    animated: bool,
    nominal_anim_ms: int,
) -> ToolsPaneToggleProbe | None:
    global _active
    if not ui_perf_enabled():
        return None
    probe = ToolsPaneToggleProbe(
        target_expanded=target_expanded,
        animated=animated,
        nominal_anim_ms=nominal_anim_ms,
    )
    _active = probe
    logger.info(
        "tools_pane_toggle_begin expanded=%s animated=%s nominal_ms=%s",
        target_expanded,
        animated,
        nominal_anim_ms,
    )
    return probe


def ui_perf_count(key: str, n: int = 1) -> None:
    if not ui_perf_enabled():
        return
    probe = active_tools_pane_probe()
    if probe is not None:
        probe.bump(key, n)


@contextmanager
def ui_perf_measure(block: str) -> Iterator[None]:
    if not ui_perf_enabled():
        yield
        return
    probe = active_tools_pane_probe()
    if probe is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        probe.add_ms(block, (time.perf_counter() - t0) * 1000.0)


def record_tools_pane_anim_tick(which: str) -> None:
    if not ui_perf_enabled():
        return
    probe = active_tools_pane_probe()
    if probe is None:
        return
    if which == "content":
        probe.anim_ticks_content += 1
    elif which == "frame":
        probe.anim_ticks_frame += 1


def reset_ui_perf_probe_state() -> None:
    """Clear cached env and active probe (unit tests only)."""
    global _UI_PERF_ENABLED, _active
    _UI_PERF_ENABLED = None
    _active = None


def finish_tools_pane_toggle() -> None:
    global _active
    probe = _active
    if probe is None:
        return
    wall_ms = (time.perf_counter() - probe.started_at) * 1000.0
    payload: dict[str, Any] = {
        "event": "tools_pane_toggle_end",
        "target_expanded": probe.target_expanded,
        "animated": probe.animated,
        "nominal_anim_ms": probe.nominal_anim_ms,
        "wall_ms": round(wall_ms, 2),
        "anim_ticks_content": probe.anim_ticks_content,
        "anim_ticks_frame": probe.anim_ticks_frame,
        "counters": dict(sorted(probe.counters.items())),
        "durations_ms": {k: round(v, 2) for k, v in sorted(probe.durations_ms.items())},
    }
    if probe.animated and probe.nominal_anim_ms > 0:
        payload["anim_slip_ms"] = round(wall_ms - probe.nominal_anim_ms, 2)
    logger.info("tools_pane_toggle_summary %s", json.dumps(payload, sort_keys=True))
    _active = None