"""Sidecar telemetry brain — summarize and health rules."""
from __future__ import annotations

from core.sidecar_telemetry import SidecarTelemetryBrain
from core.sidecar_types import SidecarTask


def test_summarize_empty_is_idle():
    brain = SidecarTelemetryBrain()
    summary = brain.summarize()
    assert summary["total_invocations"] == 0
    assert "health" in summary


def test_record_and_rewrite_turn_metrics():
    brain = SidecarTelemetryBrain()
    brain.set_runtime_state(model_loaded=True)
    brain.record(SidecarTask.query_rewrite, ok=True, latency_ms=42.0, foreground=True)
    brain.record(SidecarTask.query_rewrite, ok=False, latency_ms=10.0, foreground=True, reason="timeout")
    brain.record_turn(rewrite_attempted=True, rewrite_applied=True, rewrite_confidence=0.82)
    summary = brain.summarize()
    assert summary["total_invocations"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["rewrite"]["applied"] == 1
    assert summary["foreground"]["attempts"] == 2
