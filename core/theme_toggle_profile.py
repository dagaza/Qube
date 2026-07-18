"""Theme-toggle profiling, regression baselines, and log analysis.

Enable detailed step timings in the running app::

    QUBE_THEME_PROFILE=1 python main.py

Theme toggles apply the new QSS directly (no empty ``setStyleSheet("")`` pass).
To reproduce the legacy clear-then-apply path for regression checks::

    QUBE_THEME_PROFILE=1 QUBE_THEME_FORCE_STYLESHEET_CLEAR=1 python main.py

Logs are written to the ``Qube.ThemeProfile`` logger (INFO) in ``~/.qube/logs/qube.log``.

Startup snapshot (once per main window show, when profiling is enabled)::

    [ThemeProfile] startup app_version=... profile_session=... widget_count=...

After theme toggles, each hot-path block logs ``theme toggle total=...`` plus a
``context`` line (``widget_count``, ``qss_apply`` step, ``built_main_stages``, …).

Parse recent runs and compare to baselines::

    python tools/benchmark_theme_toggle.py --last 4
    python tools/benchmark_theme_toggle.py --check-regression

Scan for lazy-stage access footguns (``getattr(w, "settings_view")``, etc.)::

    python tools/audit_lazy_stage_footguns.py

Regression thresholds (override via env) apply to **hot-path** toggles only
(deferred batch entries are excluded)::

    QUBE_THEME_REGRESS_MAX_TOTAL_MS=900
    QUBE_THEME_REGRESS_MAX_QSS_APPLY_MS=700
    QUBE_THEME_REGRESS_MAX_WIDGET_COUNT=1800
    QUBE_THEME_REGRESS_MAX_BUILT_STAGES=1

Phase 2 baseline (232 conversation rows, Conversations only, lazy stages):
hot path ~673 ms, ``qss_apply`` ~533 ms, ``stage_0.refresh`` ~120 ms,
``widget_count`` ~1456, ``built_main_stages`` 1, ``deferred_stage_count`` 0.
See :data:`THEME_TOGGLE_PHASE2_BASELINE`.

Standard repro (manual benchmark before release or after UI refactors)::

    1. Fresh launch; stay on Conversations (do not open other nav pages).
    2. ``QUBE_THEME_PROFILE=1 python main.py``
    3. Toggle dark→light and light→dark once each.
    4. ``python tools/benchmark_theme_toggle.py --check-regression``
       (or grep ``ThemeProfile`` in ``~/.qube/logs/qube.log``)

User-facing outcome: theme toggle UI block ~6.5 s → ~0.67 s (~90% reduction).
Further micro-optimization is deferred unless users report pain.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Protocol

logger = logging.getLogger("Qube.ThemeProfile")

# Recorded 2026-07-18 on Phase 2 (lazy stages + skip-clear + deferred refresh).
THEME_TOGGLE_PHASE2_BASELINE: dict[str, int] = {
    "hot_path_ms": 673,
    "qss_apply_ms": 533,
    "stage_0_refresh_ms": 120,
    "widget_count": 1456,
    "built_main_stages": 1,
    "deferred_stage_count": 0,
    "conversation_rows": 232,
}

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_PROFILE_SESSION_ID: str | None = None

_THEME_TOGGLE_TOTAL_RE = re.compile(r"^theme toggle total=(?P<total>\d+)ms")
_THEME_CONTEXT_RE = re.compile(r"^context (?P<context>.+)$")
_THEME_STEP_RE = re.compile(
    r"^\s*(?P<name>\S+)=(?P<elapsed>\d+)ms\s+\((?P<pct>[\d.]+)%\)"
)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def is_theme_toggle_profile_enabled() -> bool:
    return _env_truthy("QUBE_THEME_PROFILE")


def is_theme_stylesheet_clear_forced() -> bool:
    """When true, ``MainWindow._toggle_theme`` clears app QSS before applying the new theme."""
    return _env_truthy("QUBE_THEME_FORCE_STYLESHEET_CLEAR")


def is_theme_stylesheet_clear_skipped() -> bool:
    """Default path: apply new QSS directly without an empty clear pass."""
    return not is_theme_stylesheet_clear_forced()


def get_theme_profile_app_version() -> str:
    try:
        from core.__version__ import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def get_theme_profile_session_id() -> str:
    """Stable id for correlating startup + toggle lines within one process."""
    global _PROFILE_SESSION_ID
    if _PROFILE_SESSION_ID is None:
        import uuid

        _PROFILE_SESSION_ID = uuid.uuid4().hex[:12]
    return _PROFILE_SESSION_ID


def enrich_theme_profile_context(context: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(context)
    enriched.setdefault("app_version", get_theme_profile_app_version())
    enriched.setdefault("profile_session", get_theme_profile_session_id())
    return enriched


class _SupportsAllWidgets(Protocol):
    def allWidgets(self) -> list[Any]: ...


def collect_application_widget_metrics(app: _SupportsAllWidgets | None) -> dict[str, int]:
    """Count live widgets to correlate global QSS cost with tree size."""
    if app is None:
        return {}
    widgets = app.allWidgets()
    return {
        "widget_count": len(widgets),
        "top_level_count": sum(1 for widget in widgets if widget.isWindow()),
        "visible_widget_count": sum(1 for widget in widgets if widget.isVisible()),
    }


def log_startup_widget_snapshot(
    app: _SupportsAllWidgets | None,
    *,
    built_main_stages: int,
    conversation_rows: int | None = None,
) -> None:
    """Log widget tree size once after the main window is shown (profiling only)."""
    if not is_theme_toggle_profile_enabled():
        return
    ctx: dict[str, Any] = {
        "built_main_stages": int(built_main_stages),
    }
    ctx.update(collect_application_widget_metrics(app))
    if conversation_rows is not None:
        ctx["conversation_rows"] = int(conversation_rows)
    ctx = enrich_theme_profile_context(ctx)
    logger.info(
        "[ThemeProfile] startup %s",
        " ".join(f"{key}={value}" for key, value in sorted(ctx.items())),
    )


@dataclass(frozen=True)
class ThemeProfileRegressionThresholds:
    max_total_ms: int = 900
    max_qss_apply_ms: int = 700
    max_widget_count: int = 1800
    max_built_stages: int = 1

    @classmethod
    def from_env(cls) -> ThemeProfileRegressionThresholds:
        return cls(
            max_total_ms=_env_int("QUBE_THEME_REGRESS_MAX_TOTAL_MS", 900),
            max_qss_apply_ms=_env_int("QUBE_THEME_REGRESS_MAX_QSS_APPLY_MS", 700),
            max_widget_count=_env_int("QUBE_THEME_REGRESS_MAX_WIDGET_COUNT", 1800),
            max_built_stages=_env_int("QUBE_THEME_REGRESS_MAX_BUILT_STAGES", 1),
        )


@dataclass
class ThemeProfileEntry:
    timestamp: str | None = None
    total_ms: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    steps: dict[str, int] = field(default_factory=dict)
    raw_lines: list[str] = field(default_factory=list)

    @property
    def is_deferred_batch(self) -> bool:
        return bool(self.context.get("deferred_batch"))

    @property
    def is_startup_snapshot(self) -> bool:
        return self.context.get("_kind") == "startup"

    @property
    def widget_count(self) -> int | None:
        value = self.context.get("widget_count")
        return int(value) if value is not None else None

    @property
    def built_main_stages(self) -> int | None:
        value = self.context.get("built_main_stages")
        return int(value) if value is not None else None

    @property
    def qss_apply_ms(self) -> int | None:
        for key in ("qss_apply", "qss_read_and_apply"):
            if key in self.steps:
                return self.steps[key]
        return None


@dataclass(frozen=True)
class ThemeProfileRegressionViolation:
    entry_index: int
    field: str
    actual: int
    limit: int
    message: str


def parse_theme_profile_context(text: str) -> dict[str, Any]:
    """Parse ``key=value`` tokens from a ThemeProfile context or startup line."""
    result: dict[str, Any] = {}
    for part in text.split():
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        if value.isdigit():
            result[key] = int(value)
        else:
            result[key] = value
    return result


def _extract_theme_profile_payload(line: str) -> str | None:
    marker = "[ThemeProfile]"
    idx = line.find(marker)
    if idx < 0:
        return None
    return line[idx + len(marker) :].strip()


def _line_timestamp(line: str) -> str | None:
    if line.startswith("[") and "]" in line:
        return line[1 : line.index("]")]
    return None


def parse_theme_profile_lines(lines: Iterable[str]) -> list[ThemeProfileEntry]:
    """Group consecutive ThemeProfile log lines into structured entries."""
    entries: list[ThemeProfileEntry] = []
    current: ThemeProfileEntry | None = None

    for line in lines:
        payload = _extract_theme_profile_payload(line)
        if payload is None:
            continue

        if payload.startswith("startup "):
            ctx = parse_theme_profile_context(payload[len("startup ") :])
            ctx["_kind"] = "startup"
            entries.append(
                ThemeProfileEntry(
                    timestamp=_line_timestamp(line),
                    context=ctx,
                    raw_lines=[line.rstrip("\n")],
                )
            )
            current = None
            continue

        total_match = _THEME_TOGGLE_TOTAL_RE.search(payload)
        if total_match:
            if current is not None:
                entries.append(current)
            current = ThemeProfileEntry(
                timestamp=_line_timestamp(line),
                total_ms=int(total_match.group("total")),
                raw_lines=[line.rstrip("\n")],
            )
            continue

        if current is None:
            continue

        current.raw_lines.append(line.rstrip("\n"))

        context_match = _THEME_CONTEXT_RE.search(payload)
        if context_match:
            current.context.update(parse_theme_profile_context(context_match.group("context")))
            continue

        step_match = _THEME_STEP_RE.search(payload)
        if step_match:
            current.steps[step_match.group("name")] = int(step_match.group("elapsed"))

    if current is not None:
        entries.append(current)
    return entries


def parse_theme_profile_log(text: str) -> list[ThemeProfileEntry]:
    return parse_theme_profile_lines(text.splitlines())


def filter_hot_path_toggle_entries(
    entries: list[ThemeProfileEntry],
) -> list[ThemeProfileEntry]:
    return [
        entry
        for entry in entries
        if not entry.is_deferred_batch and not entry.is_startup_snapshot and entry.total_ms > 0
    ]


def check_theme_profile_regression(
    entries: list[ThemeProfileEntry],
    thresholds: ThemeProfileRegressionThresholds | None = None,
) -> list[ThemeProfileRegressionViolation]:
    """Return violations for hot-path toggles exceeding default Phase 2 guardrails."""
    limits = thresholds or ThemeProfileRegressionThresholds.from_env()
    violations: list[ThemeProfileRegressionViolation] = []

    for index, entry in enumerate(filter_hot_path_toggle_entries(entries)):
        if entry.total_ms > limits.max_total_ms:
            violations.append(
                ThemeProfileRegressionViolation(
                    entry_index=index,
                    field="total_ms",
                    actual=entry.total_ms,
                    limit=limits.max_total_ms,
                    message=(
                        f"hot-path total {entry.total_ms}ms exceeds "
                        f"{limits.max_total_ms}ms"
                    ),
                )
            )

        qss_apply_ms = entry.qss_apply_ms
        if qss_apply_ms is not None and qss_apply_ms > limits.max_qss_apply_ms:
            violations.append(
                ThemeProfileRegressionViolation(
                    entry_index=index,
                    field="qss_apply_ms",
                    actual=qss_apply_ms,
                    limit=limits.max_qss_apply_ms,
                    message=(
                        f"qss_apply {qss_apply_ms}ms exceeds "
                        f"{limits.max_qss_apply_ms}ms"
                    ),
                )
            )

        widget_count = entry.widget_count
        if widget_count is not None and widget_count > limits.max_widget_count:
            violations.append(
                ThemeProfileRegressionViolation(
                    entry_index=index,
                    field="widget_count",
                    actual=widget_count,
                    limit=limits.max_widget_count,
                    message=(
                        f"widget_count {widget_count} exceeds "
                        f"{limits.max_widget_count}"
                    ),
                )
            )

        built_stages = entry.built_main_stages
        if built_stages is not None and built_stages > limits.max_built_stages:
            violations.append(
                ThemeProfileRegressionViolation(
                    entry_index=index,
                    field="built_main_stages",
                    actual=built_stages,
                    limit=limits.max_built_stages,
                    message=(
                        f"built_main_stages {built_stages} exceeds "
                        f"{limits.max_built_stages}"
                    ),
                )
            )

    return violations


def format_theme_profile_table(entries: list[ThemeProfileEntry]) -> str:
    """Render a compact text table for CLI inspection."""
    headers = (
        "when",
        "kind",
        "total",
        "qss_apply",
        "widgets",
        "built",
        "rows",
        "theme",
    )
    rows: list[list[str]] = []
    for entry in entries:
        kind = "startup" if entry.is_startup_snapshot else (
            "deferred" if entry.is_deferred_batch else "toggle"
        )
        rows.append(
            [
                entry.timestamp or "-",
                kind,
                str(entry.total_ms) if entry.total_ms else "-",
                str(entry.qss_apply_ms) if entry.qss_apply_ms is not None else "-",
                str(entry.widget_count) if entry.widget_count is not None else "-",
                str(entry.built_main_stages) if entry.built_main_stages is not None else "-",
                str(entry.context.get("conversation_rows", "-")),
                str(entry.context.get("target_theme", "-")),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    lines = [_fmt(list(headers)), _fmt(["-" * width for width in widths])]
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)


@dataclass
class ThemeToggleProfiler:
    """Collect per-step elapsed times for a single theme toggle."""

    enabled: bool = True
    _steps: list[tuple[str, int]] = field(default_factory=list)
    _total_ms: int = 0
    _timer: Any = field(default=None, init=False, repr=False)

    @classmethod
    def maybe_enabled(cls) -> ThemeToggleProfiler:
        return cls(enabled=is_theme_toggle_profile_enabled())

    def begin(self) -> None:
        if not self.enabled:
            return
        from PyQt6.QtCore import QElapsedTimer

        self._steps.clear()
        self._timer = QElapsedTimer()
        self._timer.start()

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        from PyQt6.QtCore import QElapsedTimer

        step_timer = QElapsedTimer()
        step_timer.start()
        try:
            yield
        finally:
            self._steps.append((name, step_timer.elapsed()))

    def finish(self, *, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if self._timer is not None:
            self._total_ms = self._timer.elapsed()
        else:
            self._total_ms = sum(elapsed for _, elapsed in self._steps)
        merged_context = enrich_theme_profile_context(context or {})
        self._log_summary(merged_context)
        return {
            "total_ms": self._total_ms,
            "steps": list(self._steps),
            "context": merged_context,
        }

    def _log_summary(self, context: dict[str, Any]) -> None:
        logger.info("[ThemeProfile] theme toggle total=%sms", self._total_ms)
        if context:
            ctx_parts = " ".join(f"{key}={value}" for key, value in sorted(context.items()))
            logger.info("[ThemeProfile] context %s", ctx_parts)
        for name, elapsed_ms in self._steps:
            pct = (elapsed_ms * 100.0 / self._total_ms) if self._total_ms else 0.0
            logger.info(
                "[ThemeProfile]   %s=%sms (%.1f%%)",
                name,
                elapsed_ms,
                pct,
            )
