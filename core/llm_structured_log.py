"""Structured JSON telemetry for ~/.qube/logs/llm_debug.log (always on; no QUBE_LLM_DEBUG gate)."""
from __future__ import annotations

import json
import logging
from typing import Any

_logger = logging.getLogger("Qube.NativeLLM.Debug")


def structured_llm_log(event: str, payload: dict[str, Any] | None = None) -> None:
    """Emit one JSON line to Qube.NativeLLM.Debug (routed to ~/.qube/logs/llm_debug.log)."""
    body: dict[str, Any] = {"event": event}
    if payload:
        body.update(payload)
    _logger.info(json.dumps(body, ensure_ascii=False))
