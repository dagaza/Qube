"""Disk + memory feasibility for first-run bootstrap selections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from core.bootstrap_manifest import (
    BOOTSTRAP_MODELS,
    BootstrapModelId,
    MAIN_LLM_GROUP,
    SIDECAR_GROUP,
    format_byte_size,
    normalize_selection,
)
from core.bootstrap_selection import (
    budget_headroom_bytes,
    required_bytes_for,
    selection_within_budget,
    total_selected_bytes,
)
from core.bootstrap_hf_metadata import BootstrapSizeSource, ResolvedBootstrapSize
from core.hardware_capability_profile import HardwareCapabilityProfile, detect_hardware_capability_profile

_RUNTIME_OVERHEAD_FACTOR = 1.15
_MIN_RAM_GB_FOR_CORE_STACK = 6.0


class BootstrapBlockReason(StrEnum):
    NONE = "none"
    DISK = "disk"
    MEMORY = "memory"


@dataclass(frozen=True)
class BootstrapModelFeasibility:
    model_id: BootstrapModelId
    disk_ok: bool
    memory_ok: bool
    block_reason: BootstrapBlockReason
    message: str = ""


@dataclass(frozen=True)
class BootstrapSessionAssessment:
    profile: HardwareCapabilityProfile
    resolved_sizes: dict[BootstrapModelId, ResolvedBootstrapSize]
    hf_verified_count: int
    estimate_count: int

    @property
    def size_bytes(self) -> dict[BootstrapModelId, int]:
        return {mid: entry.size_bytes for mid, entry in self.resolved_sizes.items()}

    def hardware_summary(self) -> str:
        return (
            f"Your system: {self.profile.summary_label} ({self.profile.tier_label} tier). "
            f"Estimated inference headroom ~{self.profile.inference_budget_gb:.1f} GB."
        )

    def size_source_summary(self) -> str:
        if self.estimate_count == 0:
            return "Download sizes verified with Hugging Face."
        if self.hf_verified_count == 0:
            return (
                "Could not reach Hugging Face — using offline size estimates. "
                "Connect to the internet for exact download sizes."
            )
        return (
            f"{self.hf_verified_count} of {len(self.resolved_sizes)} download sizes verified "
            f"with Hugging Face; {self.estimate_count} use offline estimates."
        )


def build_session_assessment(
    *,
    resolved: dict[BootstrapModelId, ResolvedBootstrapSize] | None = None,
    profile: HardwareCapabilityProfile | None = None,
) -> BootstrapSessionAssessment:
    sizes = resolved or {}
    if not sizes:
        from core.bootstrap_hf_metadata import resolve_all_bootstrap_sizes

        sizes = resolve_all_bootstrap_sizes()
    prof = profile or detect_hardware_capability_profile()
    hf_count = sum(1 for e in sizes.values() if e.source is BootstrapSizeSource.HUGGINGFACE)
    est_count = len(sizes) - hf_count
    return BootstrapSessionAssessment(
        profile=prof,
        resolved_sizes=sizes,
        hf_verified_count=hf_count,
        estimate_count=est_count,
    )


def _memory_load_gb(model_id: BootstrapModelId, size_bytes: int) -> float:
    return (float(size_bytes) / (1024.0**3)) * _RUNTIME_OVERHEAD_FACTOR


def assess_memory_fit(
    model_id: BootstrapModelId,
    size_bytes: int,
    profile: HardwareCapabilityProfile,
    *,
    size_entry: ResolvedBootstrapSize | None = None,
) -> tuple[bool, str]:
    load_gb = _memory_load_gb(model_id, size_bytes)
    source_note = ""
    if size_entry is not None:
        if size_entry.source is BootstrapSizeSource.HUGGINGFACE:
            source_note = f"Hugging Face reports {format_byte_size(size_bytes)} for this file. "
        else:
            source_note = f"Using estimated download size {format_byte_size(size_bytes)}. "

    if model_id in MAIN_LLM_GROUP:
        budget = profile.inference_budget_gb
        if budget <= 0:
            return True, ""
        if load_gb > budget:
            return (
                False,
                f"Not enough memory for this system. {source_note}"
                f"This main model needs ~{load_gb:.1f} GB runtime memory; "
                f"your system has ~{budget:.1f} GB headroom ({profile.summary_label}).",
            )
        if load_gb > budget * 0.9:
            return (
                True,
                f"{source_note}May run slowly — uses ~{load_gb:.1f} GB of your "
                f"~{budget:.1f} GB inference budget.",
            )
        return True, ""

    if model_id in SIDECAR_GROUP:
        if profile.total_ram_gb > 0 and profile.total_ram_gb < _MIN_RAM_GB_FOR_CORE_STACK:
            return (
                False,
                f"Not enough memory for this system. {source_note}"
                f"Sidecar needs a system with at least "
                f"{_MIN_RAM_GB_FOR_CORE_STACK:.0f} GB RAM (detected "
                f"{profile.total_ram_gb:.0f} GB).",
            )
        if profile.total_ram_gb > 0 and load_gb > profile.total_ram_gb * 0.35:
            return (
                False,
                f"Not enough memory for this system. {source_note}"
                f"Sidecar load (~{load_gb:.1f} GB) is heavy for "
                f"{profile.total_ram_gb:.0f} GB RAM.",
            )
        return True, ""

    if profile.total_ram_gb > 0 and load_gb > profile.total_ram_gb * 0.2:
        return (
            False,
            f"Not enough memory for this system. {source_note}"
            f"May not fit comfortably in {profile.total_ram_gb:.0f} GB RAM.",
        )
    return True, ""


def assess_model_feasibility(
    model_id: BootstrapModelId,
    selected: set[BootstrapModelId],
    assessment: BootstrapSessionAssessment,
) -> BootstrapModelFeasibility:
    sizes = assessment.size_bytes
    size_entry = assessment.resolved_sizes.get(model_id)
    size_bytes = sizes.get(model_id, BOOTSTRAP_MODELS[model_id].size_bytes)

    memory_ok, memory_msg = assess_memory_fit(
        model_id,
        size_bytes,
        assessment.profile,
        size_entry=size_entry,
    )

    disk_ok = True
    disk_msg = ""
    if model_id not in selected:
        trial = set(selected) | {model_id}
        disk_ok = selection_within_budget(trial, sizes=sizes)
        if not disk_ok:
            headroom = budget_headroom_bytes(selected, sizes=sizes)
            need = required_bytes_for(trial, sizes=sizes) - required_bytes_for(selected, sizes=sizes)
            disk_msg = (
                f"Not enough disk space to add this download "
                f"(needs {format_byte_size(need)} more; "
                f"headroom {format_byte_size(max(0, headroom))})."
            )

    if not memory_ok:
        return BootstrapModelFeasibility(
            model_id=model_id,
            disk_ok=disk_ok,
            memory_ok=False,
            block_reason=BootstrapBlockReason.MEMORY,
            message=memory_msg,
        )
    if not disk_ok:
        return BootstrapModelFeasibility(
            model_id=model_id,
            disk_ok=False,
            memory_ok=True,
            block_reason=BootstrapBlockReason.DISK,
            message=disk_msg,
        )
    return BootstrapModelFeasibility(
        model_id=model_id,
        disk_ok=True,
        memory_ok=True,
        block_reason=BootstrapBlockReason.NONE,
        message=memory_msg,
    )


def models_blocked_for_session(
    selected: set[BootstrapModelId],
    candidates: set[BootstrapModelId],
    assessment: BootstrapSessionAssessment,
) -> dict[BootstrapModelId, BootstrapModelFeasibility]:
    blocked: dict[BootstrapModelId, BootstrapModelFeasibility] = {}
    for model_id in candidates:
        if model_id in selected:
            continue
        fit = assess_model_feasibility(model_id, selected, assessment)
        if fit.block_reason is not BootstrapBlockReason.NONE:
            blocked[model_id] = fit
    return blocked


def summarize_blocked_models(
    blocked: dict[BootstrapModelId, BootstrapModelFeasibility],
) -> str:
    """Build a disk/memory-specific banner for models that cannot be added."""
    disk_labels = [
        BOOTSTRAP_MODELS[mid].label
        for mid, fit in blocked.items()
        if fit.block_reason is BootstrapBlockReason.DISK
    ]
    memory_labels = [
        BOOTSTRAP_MODELS[mid].label
        for mid, fit in blocked.items()
        if fit.block_reason is BootstrapBlockReason.MEMORY
    ]
    parts: list[str] = []
    if len(disk_labels) == 1:
        parts.append(
            f"{disk_labels[0]} cannot be added - not enough free disk space with your "
            "current selection."
        )
    elif disk_labels:
        parts.append(
            f"{len(disk_labels)} models cannot be added - not enough free disk space: "
            f"{', '.join(disk_labels)}."
        )
    if len(memory_labels) == 1:
        parts.append(
            f"{memory_labels[0]} cannot be added - not enough memory for this system."
        )
    elif memory_labels:
        parts.append(
            f"{len(memory_labels)} models cannot be added - not enough memory: "
            f"{', '.join(memory_labels)}."
        )
    if not parts:
        return ""
    return " ".join(parts) + " See row details."


_MAIN_LLM_PREFERENCE: tuple[BootstrapModelId, ...] = (
    BootstrapModelId.LLM_QWEN35_9B,
    BootstrapModelId.LLM_GEMMA4_E4B,
    BootstrapModelId.LLM_NEMOTRON_NANO,
)


def feasible_recommended_selection(
    assessment: BootstrapSessionAssessment,
    *,
    preset: set[BootstrapModelId] | None = None,
    locked_ids: set[BootstrapModelId] | None = None,
) -> set[BootstrapModelId]:
    """Build feasible recommended set: core, best fitting main LLM, then optional voice."""
    from core.bootstrap_manifest import default_selection, locked_recommended_ids

    target_preset = preset or default_selection(advanced=False)
    locked = set(locked_ids or locked_recommended_ids())
    sizes = assessment.size_bytes
    selected = normalize_selection(set(locked))

    for main_id in _MAIN_LLM_PREFERENCE:
        if main_id == BootstrapModelId.LLM_QWEN35_9B and main_id not in target_preset:
            continue
        if main_id in selected:
            break
        fit = assess_model_feasibility(main_id, selected, assessment)
        if fit.block_reason is not BootstrapBlockReason.NONE:
            continue
        trial = normalize_selection(selected | {main_id})
        if selection_within_budget(trial, sizes=sizes):
            selected = trial
            break

    for opt_id in (BootstrapModelId.WHISPER_SMALL, BootstrapModelId.KOKORO_TTS):
        if opt_id not in target_preset or opt_id in selected:
            continue
        fit = assess_model_feasibility(opt_id, selected, assessment)
        if fit.block_reason is not BootstrapBlockReason.NONE:
            continue
        trial = normalize_selection(selected | {opt_id})
        if selection_within_budget(trial, sizes=sizes):
            selected = trial

    return normalize_selection(selected)


def feasible_selection_from_preset(
    preset: set[BootstrapModelId],
    assessment: BootstrapSessionAssessment,
    *,
    order: tuple[BootstrapModelId, ...],
    locked_ids: set[BootstrapModelId] | None = None,
) -> set[BootstrapModelId]:
    """Return the largest feasible subset of preset, adding models in display order."""
    sizes = assessment.size_bytes
    locked = set(locked_ids or ())
    selected: set[BootstrapModelId] = set()
    for model_id in order:
        if model_id not in preset:
            continue
        if model_id in locked:
            selected.add(model_id)
    selected = normalize_selection(selected)

    for model_id in order:
        if model_id not in preset or model_id in selected or model_id in locked:
            continue
        fit = assess_model_feasibility(model_id, selected, assessment)
        if fit.block_reason is not BootstrapBlockReason.NONE:
            continue
        trial = normalize_selection(selected | {model_id})
        if selection_within_budget(trial, sizes=sizes):
            selected = trial
    return normalize_selection(selected)


def can_proceed_with_selection(
    selected: set[BootstrapModelId],
    assessment: BootstrapSessionAssessment,
) -> tuple[bool, str]:
    if not selected:
        return False, "Select at least one model to download."
    if not selection_within_budget(selected, sizes=assessment.size_bytes):
        return False, (
            "Not enough free disk space for the current selection. "
            "Deselect models or free disk space."
        )
    memory_ok, memory_message = selected_session_feasible(selected, assessment)
    if not memory_ok:
        return False, memory_message
    return True, ""


def selected_session_feasible(
    selected: set[BootstrapModelId],
    assessment: BootstrapSessionAssessment,
) -> tuple[bool, str]:
    problems: list[str] = []
    for model_id in selected:
        fit = assess_model_feasibility(model_id, selected, assessment)
        if not fit.memory_ok:
            problems.append(f"{BOOTSTRAP_MODELS[model_id].label}: {fit.message}")
    if problems:
        return False, "\n".join(problems)
    return True, ""


def selected_totals_label(
    selected: set[BootstrapModelId],
    assessment: BootstrapSessionAssessment,
) -> str:
    return format_byte_size(total_selected_bytes(selected, sizes=assessment.size_bytes))
