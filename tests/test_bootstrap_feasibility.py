"""Tests for bootstrap disk + memory feasibility."""

from __future__ import annotations

from core.bootstrap_feasibility import (
    BootstrapBlockReason,
    BootstrapSessionAssessment,
    assess_model_feasibility,
    build_session_assessment,
    models_blocked_for_session,
    selected_session_feasible,
)
from core.bootstrap_hf_metadata import BootstrapSizeSource, ResolvedBootstrapSize
from core.bootstrap_manifest import BOOTSTRAP_MODELS, BootstrapModelId, default_selection, locked_recommended_ids
from core.hardware_capability_profile import HardwareCapabilityProfile, HardwareTier


def _compact_profile() -> HardwareCapabilityProfile:
    return HardwareCapabilityProfile(
        total_ram_gb=8.0,
        total_vram_gb=0.0,
        cpu_cores=4,
        gpu_name=None,
        gpu_backend="cpu",
        tier=HardwareTier.COMPACT,
    )


def _assessment(
    *,
    profile: HardwareCapabilityProfile | None = None,
    size_overrides: dict[BootstrapModelId, int] | None = None,
) -> BootstrapSessionAssessment:
    resolved = {
        model_id: ResolvedBootstrapSize(
            model_id=model_id,
            size_bytes=(size_overrides or {}).get(model_id, BOOTSTRAP_MODELS[model_id].size_bytes),
            source=BootstrapSizeSource.HUGGINGFACE,
            detail="test",
        )
        for model_id in BootstrapModelId
    }
    return build_session_assessment(resolved=resolved, profile=profile or _compact_profile())


def test_main_llm_blocked_when_exceeds_inference_budget():
    assessment = _assessment(
        profile=HardwareCapabilityProfile(
            total_ram_gb=16.0,
            total_vram_gb=0.0,
            cpu_cores=8,
            gpu_name=None,
            gpu_backend="cpu",
            tier=HardwareTier.STANDARD,
        ),
        size_overrides={
            BootstrapModelId.LLM_QWEN35_9B: int(12 * 1024**3),
        },
    )
    selected = {BootstrapModelId.SIDECAR_QWEN17}
    fit = assess_model_feasibility(BootstrapModelId.LLM_QWEN35_9B, selected, assessment)

    assert fit.block_reason is BootstrapBlockReason.MEMORY
    assert "not enough memory" in fit.message.lower()


def test_disk_block_when_adding_model_exceeds_budget(monkeypatch):
    assessment = _assessment(
        profile=HardwareCapabilityProfile(
            total_ram_gb=64.0,
            total_vram_gb=24.0,
            cpu_cores=16,
            gpu_name="Test GPU",
            gpu_backend="cuda",
            tier=HardwareTier.ENTHUSIAST,
        )
    )
    selected = {BootstrapModelId.SIDECAR_QWEN17}
    huge = BootstrapModelId.LLM_NEMOTRON_NANO

    monkeypatch.setattr("core.bootstrap_selection.available_disk_bytes", lambda: 1)
    fit = assess_model_feasibility(huge, selected, assessment)

    assert fit.block_reason is BootstrapBlockReason.DISK
    assert "disk space" in fit.message.lower()


def test_models_blocked_for_session_skips_already_selected():
    assessment = _assessment()
    selected = {BootstrapModelId.SIDECAR_QWEN17}
    blocked = models_blocked_for_session(
        selected,
        {BootstrapModelId.SIDECAR_QWEN17, BootstrapModelId.LLM_QWEN35_9B},
        assessment,
    )
    assert BootstrapModelId.SIDECAR_QWEN17 not in blocked


def test_feasible_recommended_falls_back_to_gemma_when_qwen_too_large(monkeypatch):
    from core.bootstrap_feasibility import feasible_recommended_selection

    assessment = _assessment(
        profile=HardwareCapabilityProfile(
            total_ram_gb=16.0,
            total_vram_gb=0.0,
            cpu_cores=8,
            gpu_name=None,
            gpu_backend="cpu",
            tier=HardwareTier.STANDARD,
        )
    )
    monkeypatch.setattr("core.bootstrap_selection.available_disk_bytes", lambda: int(8 * 1024**3))

    feasible = feasible_recommended_selection(assessment)

    assert locked_recommended_ids().issubset(feasible)
    assert BootstrapModelId.LLM_QWEN35_9B not in feasible
    assert BootstrapModelId.LLM_GEMMA4_E4B in feasible


def test_feasible_selection_skips_models_that_exceed_disk(monkeypatch):
    from core.bootstrap_feasibility import feasible_selection_from_preset
    from core.bootstrap_manifest import RECOMMENDED_ORDER, locked_recommended_ids

    assessment = _assessment(
        profile=HardwareCapabilityProfile(
            total_ram_gb=16.0,
            total_vram_gb=0.0,
            cpu_cores=8,
            gpu_name=None,
            gpu_backend="cpu",
            tier=HardwareTier.STANDARD,
        )
    )
    preset = default_selection(advanced=False)
    monkeypatch.setattr("core.bootstrap_selection.available_disk_bytes", lambda: int(8 * 1024**3))

    feasible = feasible_selection_from_preset(
        preset,
        assessment,
        order=RECOMMENDED_ORDER,
        locked_ids=locked_recommended_ids(),
    )

    assert locked_recommended_ids().issubset(feasible)
    assert BootstrapModelId.LLM_QWEN35_9B not in feasible


def test_can_proceed_requires_disk_and_memory(monkeypatch):
    from core.bootstrap_feasibility import can_proceed_with_selection

    assessment = _assessment()
    monkeypatch.setattr("core.bootstrap_selection.available_disk_bytes", lambda: int(64 * 1024**3))
    core = {BootstrapModelId.SIDECAR_QWEN17}
    ok, _ = can_proceed_with_selection(core, assessment)
    assert ok


def test_feasible_recommended_core_only_when_disk_tight(monkeypatch):
    from core.bootstrap_feasibility import can_proceed_with_selection, feasible_recommended_selection

    assessment = _assessment()
    monkeypatch.setattr("core.bootstrap_selection.available_disk_bytes", lambda: int(1.9 * 1024**3))

    feasible = feasible_recommended_selection(assessment)

    assert feasible == set(locked_recommended_ids())
    ok, message = can_proceed_with_selection(feasible, assessment)
    assert not ok
    assert "disk space" in message.lower()


def test_summarize_blocked_models_splits_disk_and_memory():
    from core.bootstrap_feasibility import BootstrapModelFeasibility, summarize_blocked_models

    blocked = {
        BootstrapModelId.LLM_QWEN35_9B: BootstrapModelFeasibility(
            model_id=BootstrapModelId.LLM_QWEN35_9B,
            disk_ok=False,
            memory_ok=True,
            block_reason=BootstrapBlockReason.DISK,
            message="Not enough disk space to add this download (needs 6.95 GB more; headroom 5.83 GB).",
        )
    }
    summary = summarize_blocked_models(blocked)
    assert "not enough free disk space" in summary.lower()
    assert "Qwen 3.5 9B Q6" in summary
    assert "memory" not in summary.lower()


def test_assess_model_feasibility_advisory_memory_when_not_enforced():
    assessment = _assessment(
        profile=HardwareCapabilityProfile(
            total_ram_gb=8.0,
            total_vram_gb=0.0,
            cpu_cores=4,
            gpu_name=None,
            gpu_backend="cpu",
            tier=HardwareTier.COMPACT,
        ),
    )
    selected = {BootstrapModelId.SIDECAR_QWEN17}
    fit = assess_model_feasibility(
        BootstrapModelId.LLM_NEMOTRON_NANO,
        selected,
        assessment,
        enforce_memory=False,
    )

    assert fit.block_reason is BootstrapBlockReason.NONE
    assert not fit.memory_ok
    assert "not enough memory" in fit.message.lower()


def test_can_proceed_allows_memory_oversized_selection_when_not_enforced(monkeypatch):
    from core.bootstrap_feasibility import can_proceed_with_selection

    assessment = _assessment(
        profile=HardwareCapabilityProfile(
            total_ram_gb=8.0,
            total_vram_gb=0.0,
            cpu_cores=4,
            gpu_name=None,
            gpu_backend="cpu",
            tier=HardwareTier.COMPACT,
        ),
    )
    monkeypatch.setattr("core.bootstrap_selection.available_disk_bytes", lambda: int(64 * 1024**3))
    selected = {
        BootstrapModelId.SIDECAR_QWEN17,
        BootstrapModelId.LLM_NEMOTRON_NANO,
    }
    ok, message = can_proceed_with_selection(selected, assessment, enforce_memory=False)
    assert ok
    assert message == ""


def test_selected_session_feasible_flags_oversized_main_llm():
    assessment = _assessment(
        profile=HardwareCapabilityProfile(
            total_ram_gb=8.0,
            total_vram_gb=0.0,
            cpu_cores=4,
            gpu_name=None,
            gpu_backend="cpu",
            tier=HardwareTier.COMPACT,
        ),
        size_overrides={
            BootstrapModelId.LLM_QWEN35_9B: int(10 * 1024**3),
        },
    )
    selected = {
        BootstrapModelId.SIDECAR_QWEN17,
        BootstrapModelId.LLM_QWEN35_9B,
    }
    ok, message = selected_session_feasible(selected, assessment)
    assert not ok
    assert "Qwen" in message or "qwen" in message.lower()
