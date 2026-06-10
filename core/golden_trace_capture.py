"""
Golden baseline capture for canonical LLM execution traces (regression / drift detection).

Enable with GOLDEN_TRACE_CAPTURE_MODE=1 to persist ONE complete trace per process
to debug/golden_traces/{timestamp}.json under the install root.

Provider-agnostic; pairs with core.canonical_trace_diff for comparisons.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.canonical_trace_diff import CanonicalTrace, build_trace_fingerprints, coerce_canonical_trace
from core.canonical_request import CanonicalRequest, CanonicalRequestExporter
from core.paths import install_root

logger = logging.getLogger("Qube.NativeLLM.Debug")

_capture_lock = threading.Lock()
_capture_completed = False
_last_capture_path: Optional[Path] = None


def golden_trace_capture_mode_enabled() -> bool:
    return os.environ.get("GOLDEN_TRACE_CAPTURE_MODE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def golden_traces_dir() -> Path:
    path = install_root() / "debug" / "golden_traces"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp_filename() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{ts}.json"


def build_golden_trace(
    *,
    request: CanonicalRequest | dict[str, Any],
    prompt: str,
    output: str,
    metadata: Optional[dict[str, Any]] = None,
) -> CanonicalTrace:
    if isinstance(request, CanonicalRequest):
        canonical_request = request
    else:
        canonical_request = CanonicalRequestExporter.export_canonical_request(
            dict(request or {})
        )
    prompt_text = str(prompt or "")
    output_text = str(output or "")
    return CanonicalTrace(
        request=canonical_request,
        prompt=prompt_text,
        output=output_text,
        metadata=dict(metadata or {}),
        fingerprints=build_trace_fingerprints(
            canonical_request,
            prompt_text,
            output_text,
        ),
    )


def save_golden_trace(trace: CanonicalTrace | dict[str, Any], path: Path) -> Path:
    data = coerce_canonical_trace(trace).to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_golden_trace(path: str | Path) -> CanonicalTrace:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("golden trace file must contain a JSON object")
    return coerce_canonical_trace(raw)


def maybe_capture_golden_trace(
    trace: CanonicalTrace | dict[str, Any],
    *,
    traces_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Persist one golden trace when GOLDEN_TRACE_CAPTURE_MODE is enabled.

    Subsequent calls in the same process are no-ops. Returns the written path
    on first capture, else None.
    """
    global _capture_completed, _last_capture_path
    if not golden_trace_capture_mode_enabled():
        return None
    with _capture_lock:
        if _capture_completed:
            return None
        try:
            dest = (traces_dir or golden_traces_dir()) / _timestamp_filename()
            save_golden_trace(trace, dest)
            _capture_completed = True
            _last_capture_path = dest
            logger.info("[GoldenTraceCapture] wrote baseline trace to %s", dest)
            return dest
        except Exception:
            logger.debug("[GoldenTraceCapture] capture failed", exc_info=True)
            return None


def last_golden_capture_path() -> Optional[Path]:
    return _last_capture_path


def reset_golden_trace_capture_for_tests() -> None:
    global _capture_completed, _last_capture_path
    with _capture_lock:
        _capture_completed = False
        _last_capture_path = None
