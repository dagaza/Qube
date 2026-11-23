"""
Temporary GUI-thread probes for tools-pane toggle and window resize (Phase A+ diagnostics).

Enable with ``QUBE_UI_PERF=1``. Optional animation baseline:

- ``QUBE_TOOLS_PANE_ANIM_MS`` — retained for experiments; production tools pane uses instant show/hide (Phase B).

Window resize: each grip-drag burst (coalesce idle → settle) logs ``window_resize_session_begin`` and
``window_resize_session_summary`` as one JSON line on logger ``Qube.UiPerf``.
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
    durations_max_ms: dict[str, float] = field(default_factory=dict)
    anim_ticks_content: int = 0
    anim_ticks_frame: int = 0

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def add_ms(self, key: str, ms: float) -> None:
        self.durations_ms[key] = self.durations_ms.get(key, 0.0) + ms
        prev = self.durations_max_ms.get(key, 0.0)
        if ms > prev:
            self.durations_max_ms[key] = ms


@dataclass
class WindowResizeSession:
    coalesce_interval_ms: int
    started_at: float = field(default_factory=time.perf_counter)
    main_resize_events: int = 0
    coalesce_timer_rearms: int = 0
    start_width: int | None = None
    start_height: int | None = None
    end_width: int | None = None
    end_height: int | None = None
    counters: dict[str, int] = field(default_factory=dict)
    durations_ms: dict[str, float] = field(default_factory=dict)
    durations_max_ms: dict[str, float] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def add_ms(self, key: str, ms: float) -> None:
        self.durations_ms[key] = self.durations_ms.get(key, 0.0) + ms
        prev = self.durations_max_ms.get(key, 0.0)
        if ms > prev:
            self.durations_max_ms[key] = ms


_active: ToolsPaneToggleProbe | None = None
_window_resize_active: WindowResizeSession | None = None


def _iter_perf_recorders() -> Iterator[ToolsPaneToggleProbe | WindowResizeSession]:
    if _active is not None:
        yield _active
    if _window_resize_active is not None:
        yield _window_resize_active


def active_tools_pane_probe() -> ToolsPaneToggleProbe | None:
    if not ui_perf_enabled():
        return None
    return _active


def active_window_resize_session() -> WindowResizeSession | None:
    if not ui_perf_enabled():
        return None
    return _window_resize_active


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


def begin_window_resize_session(*, coalesce_interval_ms: int) -> WindowResizeSession | None:
    global _window_resize_active
    if not ui_perf_enabled():
        return None
    session = WindowResizeSession(coalesce_interval_ms=coalesce_interval_ms)
    _window_resize_active = session
    logger.info(
        "window_resize_session_begin coalesce_ms=%s",
        coalesce_interval_ms,
    )
    return session


def record_window_resize_frame(*, width: int, height: int) -> None:
    session = active_window_resize_session()
    if session is None:
        return
    session.main_resize_events += 1
    if session.start_width is None:
        session.start_width = int(width)
        session.start_height = int(height)
    session.end_width = int(width)
    session.end_height = int(height)


def record_window_resize_coalesce_rearm() -> None:
    session = active_window_resize_session()
    if session is None:
        return
    session.coalesce_timer_rearms += 1


def ui_perf_count(key: str, n: int = 1) -> None:
    if not ui_perf_enabled():
        return
    for recorder in _iter_perf_recorders():
        recorder.bump(key, n)


@contextmanager
def ui_perf_measure(block: str) -> Iterator[None]:
    if not ui_perf_enabled():
        yield
        return
    recorders = list(_iter_perf_recorders())
    if not recorders:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - t0) * 1000.0
        for recorder in recorders:
            recorder.add_ms(block, elapsed)


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
    """Clear cached env and active probes (unit tests only)."""
    global _UI_PERF_ENABLED, _active, _window_resize_active
    _UI_PERF_ENABLED = None
    _active = None
    _window_resize_active = None


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
        "durations_max_ms": {
            k: round(v, 2) for k, v in sorted(probe.durations_max_ms.items())
        },
    }
    if probe.animated and probe.nominal_anim_ms > 0:
        payload["anim_slip_ms"] = round(wall_ms - probe.nominal_anim_ms, 2)
    logger.info("tools_pane_toggle_summary %s", json.dumps(payload, sort_keys=True))
    _active = None


def finish_window_resize_session() -> None:
    global _window_resize_active
    session = _window_resize_active
    if session is None:
        return
    wall_ms = (time.perf_counter() - session.started_at) * 1000.0
    payload: dict[str, Any] = {
        "event": "window_resize_session_end",
        "coalesce_interval_ms": session.coalesce_interval_ms,
        "session_wall_ms": round(wall_ms, 2),
        "main_resize_events": session.main_resize_events,
        "coalesce_timer_rearms": session.coalesce_timer_rearms,
        "counters": dict(sorted(session.counters.items())),
        "durations_ms": {k: round(v, 2) for k, v in sorted(session.durations_ms.items())},
        "durations_max_ms": {
            k: round(v, 2) for k, v in sorted(session.durations_max_ms.items())
        },
    }
    if session.start_width is not None and session.start_height is not None:
        payload["start_size"] = [session.start_width, session.start_height]
    if session.end_width is not None and session.end_height is not None:
        payload["end_size"] = [session.end_width, session.end_height]
        if session.start_width is not None and session.start_height is not None:
            payload["size_delta"] = [
                session.end_width - session.start_width,
                session.end_height - session.start_height,
            ]
    payload["time_to_settle_ms"] = round(
        wall_ms,
        2,
    )  # includes trailing coalesce interval after last resize event
    logger.info("window_resize_session_summary %s", json.dumps(payload, sort_keys=True))
    _window_resize_active = None