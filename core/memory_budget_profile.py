"""Soft memory experience labels for Model Manager downloads (no binary VRAM fail)."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from enum import Enum

from core.gpu_layers_cap import detect_gpu_vram_bytes


class MemoryBudgetKind(str, Enum):
    UNKNOWN = "unknown"
    DEDICATED_VRAM = "dedicated_vram"
    UNIFIED = "unified"


@dataclass(frozen=True)
class MemoryBudgetProfile:
    kind: MemoryBudgetKind
    budget_bytes: int


@dataclass(frozen=True)
class FileMemoryExperience:
    experience: str
    short_label: str
    detail: str
    style: str  # best | caution | neutral | unknown


def _is_apple_unified() -> bool:
    if not sys.platform.startswith("darwin"):
        return False
    return platform.machine().lower() in ("arm64", "aarch64")


def detect_memory_budget_profile() -> MemoryBudgetProfile:
    vram_b = int(detect_gpu_vram_bytes() or 0)
    if vram_b <= 0:
        return MemoryBudgetProfile(kind=MemoryBudgetKind.UNKNOWN, budget_bytes=0)
    if _is_apple_unified():
        return MemoryBudgetProfile(kind=MemoryBudgetKind.UNIFIED, budget_bytes=vram_b)
    return MemoryBudgetProfile(kind=MemoryBudgetKind.DEDICATED_VRAM, budget_bytes=vram_b)


def _fmt_gib(n: int) -> str:
    return f"{(max(0, int(n)) / (1024**3)):.2f} GB"


def experience_for_download(
    file_bytes: int,
    profile: MemoryBudgetProfile | None = None,
) -> FileMemoryExperience:
    profile = profile or detect_memory_budget_profile()
    fb = max(0, int(file_bytes))
    budget = max(0, int(profile.budget_bytes))

    if fb <= 0:
        return FileMemoryExperience(
            experience="unknown_size",
            short_label="System: Unknown size",
            detail="Download size could not be determined for this file.",
            style="unknown",
        )

    if budget <= 0 or profile.kind == MemoryBudgetKind.UNKNOWN:
        return FileMemoryExperience(
            experience="unknown_budget",
            short_label=f"System: File {_fmt_gib(fb)}",
            detail="GPU memory could not be detected. Many systems use shared RAM for larger models.",
            style="unknown",
        )

    ratio = fb / float(budget) if budget > 0 else 999.0

    if profile.kind == MemoryBudgetKind.UNIFIED:
        if ratio <= 1.0:
            short = f"System: Unified memory · {_fmt_gib(fb)} / {_fmt_gib(budget)}"
            detail = "Best responsiveness on unified memory for a file this size."
            style = "best"
        elif ratio <= 1.35:
            short = f"System: Unified memory · {_fmt_gib(fb)} (may run slower)"
            detail = "Larger than the typical unified-memory budget; may run slower but can still work."
            style = "caution"
        else:
            short = f"System: Unified memory · {_fmt_gib(fb)} (heavy)"
            detail = "May run slower on unified memory; consider a smaller quantization if load fails."
            style = "caution"
        return FileMemoryExperience(
            experience="unified",
            short_label=short,
            detail=detail,
            style=style,
        )

    # Dedicated VRAM
    if ratio <= 1.0:
        return FileMemoryExperience(
            experience="best_responsiveness",
            short_label=f"System: Best responsiveness ({_fmt_gib(fb)} / {_fmt_gib(budget)})",
            detail="Likely fits dedicated VRAM with good responsiveness.",
            style="best",
        )
    if ratio <= 1.25:
        return FileMemoryExperience(
            experience="may_run_slower",
            short_label=f"System: May run slower ({_fmt_gib(fb)} / {_fmt_gib(budget)})",
            detail="Slightly above detected VRAM; partial GPU offload or CPU layers may still work.",
            style="caution",
        )
    return FileMemoryExperience(
        experience="may_need_shared_memory",
        short_label=f"System: May need shared memory ({_fmt_gib(fb)} / {_fmt_gib(budget)})",
        detail=(
            "Above dedicated VRAM on paper — may use shared system memory on laptops "
            "and APUs; expect slower loads, not a hard failure."
        ),
        style="caution",
    )
