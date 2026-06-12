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


def test_queue_wait_telemetry():
    brain = SidecarTelemetryBrain()
    brain.record(
        SidecarTask.query_rewrite,
        ok=True,
        latency_ms=120.0,
        wait_ms=800.0,
        foreground=True,
    )
    summary = brain.summarize()
    assert summary["foreground"]["p95_wait_ms"] == 800.0


def test_digest_compression_telemetry():
    brain = SidecarTelemetryBrain()
    brain.record_turn(
        digest_memory_attempted=False,
        digest_memory_applied=False,
        digest_memory_chars_before=1200,
        digest_memory_chars_after=1200,
        digest_memory_skip_reason="below_threshold",
    )
    brain.record_turn(
        digest_memory_attempted=True,
        digest_memory_applied=True,
        digest_memory_chars_before=6000,
        digest_memory_chars_after=900,
    )
    summary = brain.summarize()
    assert summary["digest"]["memory_skipped_below_threshold"] == 1
    assert summary["digest"]["memory_applied"] == 1
    assert summary["digest"]["memory_avg_chars_after"] == 900.0


if __name__ == "__main__":
    test_summarize_empty_is_idle()
    test_record_and_rewrite_turn_metrics()
    test_queue_wait_telemetry()
    test_digest_compression_telemetry()
    print("ok")
