"""Detect a practical inference capability profile for Model Manager recommendations."""

from __future__ import annotations

import glob
import platform
import re
import sys
from dataclasses import dataclass
from enum import Enum

from core.gpu_layers_cap import detect_gpu_vram_bytes, gpu_memory_kind


class HardwareTier(str, Enum):
    """User-facing capability tier for local GGUF inference."""

    COMPACT = "compact"
    STANDARD = "standard"
    PERFORMANCE = "performance"
    ENTHUSIAST = "enthusiast"


_TIER_LABELS: dict[HardwareTier, str] = {
    HardwareTier.COMPACT: "Compact",
    HardwareTier.STANDARD: "Standard",
    HardwareTier.PERFORMANCE: "Performance",
    HardwareTier.ENTHUSIAST: "Enthusiast",
}


@dataclass(frozen=True)
class HardwareCapabilityProfile:
    total_ram_gb: float
    total_vram_gb: float
    cpu_cores: int
    gpu_name: str | None
    gpu_backend: str
    tier: HardwareTier

    @property
    def tier_label(self) -> str:
        return _TIER_LABELS.get(self.tier, self.tier.value.title())

    @property
    def has_detected_gpu(self) -> bool:
        return self.total_vram_gb > 0.0

    @property
    def inference_budget_gb(self) -> float:
        """Conservative GB budget for a single Q4-class GGUF load."""
        if self.gpu_backend in ("apple_unified", "amd_unified"):
            return max(0.0, self.total_vram_gb * 0.85)
        if self.total_vram_gb > 0:
            return max(0.0, self.total_vram_gb * 0.85)
        return max(0.0, self.total_ram_gb * 0.55)

    @property
    def summary_label(self) -> str:
        parts: list[str] = []
        if self.total_vram_gb > 0:
            if self.gpu_backend in ("apple_unified", "amd_unified"):
                parts.append(f"{self.total_vram_gb:.1f} GB unified GPU budget")
            else:
                parts.append(f"{self.total_vram_gb:.1f} GB VRAM")
        if self.total_ram_gb > 0:
            parts.append(f"{self.total_ram_gb:.0f} GB RAM")
        if self.cpu_cores > 0:
            parts.append(f"{self.cpu_cores} cores")
        return " · ".join(parts) if parts else "Unknown hardware"


def _bytes_to_gb(n: int) -> float:
    return max(0.0, float(n) / (1024.0**3))


def _detect_ram_gb() -> float:
    try:
        import psutil

        return _bytes_to_gb(int(psutil.virtual_memory().total))
    except Exception:
        return 0.0


def _detect_cpu_cores() -> int:
    try:
        import os

        return max(1, int(os.cpu_count() or 1))
    except Exception:
        return 1


def _linux_amdgpu_marketing_name() -> str | None:
    if not sys.platform.startswith("linux"):
        return None
    for path in glob.glob("/sys/class/drm/card*/device/product_name"):
        try:
            with open(path, encoding="utf-8") as f:
                name = f.read().strip()
        except OSError:
            continue
        if name:
            return name
    return None


def _detect_gpu_name() -> str | None:
    try:
        import pynvml

        try:
            pynvml.nvmlInit()
        except Exception as e:
            if "already" not in str(e).lower() and "initialized" not in str(e).lower():
                raise
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        raw = pynvml.nvmlDeviceGetName(handle)
        name = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        return name.strip() or None
    except Exception:
        pass

    if sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64"):
        return "Apple Silicon"

    if gpu_memory_kind() == "amd_unified":
        return _linux_amdgpu_marketing_name() or "AMD APU (unified memory)"

    return None


def _detect_gpu_backend() -> str:
    kind = gpu_memory_kind()
    if kind == "apple_unified":
        return "apple_unified"
    if kind == "amd_unified":
        return "amd_unified"
    if kind == "nvidia":
        return "nvidia"
    if kind == "amd_discrete":
        return "amd"
    if kind != "none":
        return "discrete"
    return "none"


def classify_hardware_tier(*, vram_gb: float, ram_gb: float) -> HardwareTier:
    if vram_gb > 0:
        if vram_gb < 6:
            return HardwareTier.COMPACT
        if vram_gb < 12:
            return HardwareTier.STANDARD
        if vram_gb < 24:
            return HardwareTier.PERFORMANCE
        return HardwareTier.ENTHUSIAST

    if ram_gb >= 64:
        return HardwareTier.PERFORMANCE
    if ram_gb >= 32:
        return HardwareTier.STANDARD
    return HardwareTier.COMPACT


def detect_hardware_capability_profile() -> HardwareCapabilityProfile:
    vram_bytes = int(detect_gpu_vram_bytes() or 0)
    vram_gb = _bytes_to_gb(vram_bytes)
    ram_gb = _detect_ram_gb()
    return HardwareCapabilityProfile(
        total_ram_gb=ram_gb,
        total_vram_gb=vram_gb,
        cpu_cores=_detect_cpu_cores(),
        gpu_name=_detect_gpu_name(),
        gpu_backend=_detect_gpu_backend(),
        tier=classify_hardware_tier(vram_gb=vram_gb, ram_gb=ram_gb),
    )


def format_tier_detail(profile: HardwareCapabilityProfile) -> str:
    if profile.gpu_backend == "apple_unified":
        gpu = profile.gpu_name or "Apple Silicon (unified memory)"
    elif profile.gpu_backend == "amd_unified":
        gpu = profile.gpu_name or "AMD APU (unified memory)"
    else:
        gpu = profile.gpu_name or "No GPU detected"
    return (
        f"{profile.tier_label} tier — {profile.summary_label}. "
        f"GPU: {gpu}. Estimated single-model budget ~{profile.inference_budget_gb:.1f} GB (Q4-class)."
    )
