#!/usr/bin/env python3
"""Shared runtime for Starfall Cursor hooks.

Every hook needs the same three things, and re-implementing them per hook is how
we ended up shipping the UTF-8 BOM parse bug five times. This module centralises:

  * ``read_payload()``   - BOM-tolerant stdin parsing (Cursor prepends a UTF-8 BOM
                           on Windows). Empty stdin -> ``{}``; non-empty but
                           unparseable -> ``PayloadError`` so the *caller* decides
                           whether to fail open or closed (guards fail closed).
  * ``write_debug()``    - gated, version-stamped diagnostics breadcrumb. Writes
                           nothing unless diagnostics are enabled, so it is safe to
                           leave calls in permanently (near-zero overhead when off).
  * ``emit()``           - standard hook result (JSON to stdout + exit 0).

Import from any hook via ``import common`` - a script's own directory is always on
``sys.path``, so the sibling import resolves without path juggling.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

RUNTIME_VERSION = "1.0.0"

HOOK_DIR = Path(__file__).resolve().parent
CURSOR_DIR = HOOK_DIR.parent
STARFALL_DIR = CURSOR_DIR / "starfall"
TRIGGER = CURSOR_DIR / ".starfall-mode"
LOCK = CURSOR_DIR / ".starfall-lock"
SETTINGS = STARFALL_DIR / "settings.json"
DEBUG_LOG = STARFALL_DIR / "hook-debug.log"

# Continuous work log: runs are delimited by "# Run NNN" sections rather than
# archived every run, so the engineering narrative stays in one file. The log is
# rolled into starfall-archive/ only on size/age/explicit triggers.
LOG = CURSOR_DIR / "starfall-log.md"
CONTEXT = CURSOR_DIR / "starfall-context.md"
ARCHIVE_DIR = CURSOR_DIR / "starfall-archive"
ARCHIVE_SENTINEL = CURSOR_DIR / ".starfall-archive-now"
MAX_LOG_BYTES = 512 * 1024
MAX_LOG_AGE_DAYS = 30


class PayloadError(ValueError):
    """Raised when stdin is non-empty but cannot be parsed as a JSON object."""


def read_payload() -> dict:
    """Parse hook stdin robustly.

    Returns ``{}`` for empty stdin. Raises :class:`PayloadError` when stdin is
    non-empty but not decodable/parseable to a JSON object, so guard hooks can
    fail closed instead of silently proceeding with ``{}``.
    """
    try:
        raw = sys.stdin.buffer.read()
    except Exception:
        try:
            raw = sys.stdin.read().encode("utf-8", "replace")
        except Exception as exc:  # pragma: no cover - stdin truly unavailable
            raise PayloadError(f"stdin unreadable: {exc}") from exc
    if not raw:
        return {}
    text = raw.decode("utf-8-sig", errors="replace")
    # Defensive: drop any stray bytes before the first JSON object.
    i = text.find("{")
    if i > 0:
        text = text[i:]
    try:
        obj = json.loads(text)
    except Exception as exc:
        raise PayloadError(str(exc)) from exc
    if not isinstance(obj, dict):
        raise PayloadError(f"payload is {type(obj).__name__}, expected object")
    return obj


def diagnostics_enabled() -> bool:
    """True when verbose hook diagnostics should be written.

    Precedence: ``STARFALL_DIAGNOSTICS`` env var (1/true/yes/on) overrides
    ``settings.json`` ``{"diagnostics": true}``; default is False.
    """
    env = os.environ.get("STARFALL_DIAGNOSTICS")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    try:
        cfg = json.loads(SETTINGS.read_text(encoding="utf-8")) if SETTINGS.exists() else {}
        return bool(cfg.get("diagnostics", False))
    except Exception:
        return False


def write_debug(event: str, **fields: object) -> None:
    """Append one version-stamped breadcrumb line - only when diagnostics are on.

    Writes to both the ``__file__``-anchored and cwd-anchored
    ``.cursor/starfall/hook-debug.log`` so a run is observable regardless of where
    the hook is invoked from. Never raises.
    """
    if not diagnostics_enabled():
        return
    parts = [f"{k}={v}" for k, v in fields.items()]
    rec = (
        datetime.now(timezone.utc).isoformat()
        + f"\tv{RUNTIME_VERSION}\tevent={event}\t"
        + "\t".join(parts)
        + "\n"
    )
    targets = {DEBUG_LOG.resolve(), (Path.cwd() / ".cursor" / "starfall" / "hook-debug.log").resolve()}
    for path in targets:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(rec)
        except Exception:
            pass


def emit(obj: dict) -> None:
    """Emit a hook result (JSON on stdout) and exit successfully."""
    print(json.dumps(obj))
    sys.exit(0)


# --- continuous work log / run management -----------------------------------
def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_logs_exist() -> None:
    """Create the log/context files (with a title header) if missing. Never raises."""
    try:
        CURSOR_DIR.mkdir(parents=True, exist_ok=True)
        if not LOG.exists():
            LOG.write_text(f"# Starfall Log\nStarted: {_iso()}\n\n", encoding="utf-8")
        if not CONTEXT.exists():
            CONTEXT.write_text(f"# Starfall Context\nStarted: {_iso()}\n\n", encoding="utf-8")
    except Exception:
        pass


def _count_runs(text: str) -> int:
    return len(re.findall(r"(?m)^# Run \d+", text))


def _log_age_days() -> float | None:
    try:
        first = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    for line in first[:5]:
        m = re.match(r"\s*Started:\s*(\S+)", line)
        if m:
            try:
                started = datetime.fromisoformat(m.group(1))
                if started.tzinfo is None:
                    started = started.replace(tzinfo=timezone.utc)
                return (datetime.now(timezone.utc) - started).total_seconds() / 86400.0
            except Exception:
                return None
    return None


def _rollover_reason() -> str:
    """Empty string => no rollover. Otherwise the reason (explicit/size/age)."""
    try:
        if ARCHIVE_SENTINEL.exists():
            return "explicit"
        if LOG.exists() and LOG.stat().st_size > MAX_LOG_BYTES:
            return "size"
        age = _log_age_days()
        if age is not None and age > MAX_LOG_AGE_DAYS:
            return "age"
    except Exception:
        pass
    return ""


def _rollover(stamp: str) -> None:
    try:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    for f, kind in ((LOG, "log"), (CONTEXT, "context")):
        try:
            if f.exists() and f.stat().st_size > 0:
                f.rename(ARCHIVE_DIR / f"{kind}-{stamp}.md")
        except Exception:
            pass
    try:
        ARCHIVE_SENTINEL.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def start_new_run() -> int:
    """Begin a fresh run in the continuous log.

    Called once when a run first arms (trigger transitions absent -> present).
    Rolls the log into the archive only when size/age/explicit triggers say so,
    then appends a ``# Run NNN`` section header to the log and context and returns
    the run number. Never raises (logging must not break prompt submission).
    """
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        reason = _rollover_reason()
        if reason:
            _rollover(stamp)
        ensure_logs_exist()
        text = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
        n = _count_runs(text) + 1
        iso = _iso()
        header = f"\n# Run {n:03d} - {iso}"
        if reason:
            header += f" (log rolled over: {reason})"
        header += "\n\n"
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(header)
        with CONTEXT.open("a", encoding="utf-8") as fh:
            fh.write(f"\n## Run {n:03d} - {iso}\n\n")
        return n
    except Exception:
        return 0
