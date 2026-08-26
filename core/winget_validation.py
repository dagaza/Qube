"""WinGet / Defender-safe startup mode for CUDA Windows builds."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("Qube.WinGetValidation")

# Post-install grace for packaged CUDA builds (WinGet step 08 launches soon after install).
_GRACE_SECONDS = 20 * 60
_INSTALL_TS_NAMES = (".qube-install-ts",)
_EXPLICIT_TRUTHY = frozenset({"1", "true", "yes", "on"})
_EXPLICIT_FALSY = frozenset({"0", "false", "no", "off"})
_SMOKE_RESULT_NAME = ".winget-validation-smoke.json"


def _read_windows_variant() -> str:
    env = os.environ.get("QUBE_WINDOWS_VARIANT", "").strip().lower()
    if env:
        return env
    if not getattr(sys, "frozen", False):
        return ""
    exe_dir = Path(sys.executable).resolve().parent
    for candidate in (
        exe_dir / ".qube-windows-variant",
        exe_dir / "_internal" / ".qube-windows-variant",
    ):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip().lower()
    return ""


def _explicit_env_requested() -> bool | None:
    raw = os.environ.get("QUBE_WINGET_VALIDATION", "").strip().lower()
    if not raw:
        return None
    if raw in _EXPLICIT_FALSY:
        return False
    return raw in _EXPLICIT_TRUTHY


def _install_grace_active() -> bool:
    if _read_windows_variant() != "cuda":
        return False
    if not getattr(sys, "frozen", False):
        return False
    exe_dir = Path(sys.executable).resolve().parent
    now = time.time()
    for name in _INSTALL_TS_NAMES:
        for candidate in (exe_dir / name, exe_dir / "_internal" / name):
            if not candidate.is_file():
                continue
            age = now - candidate.stat().st_mtime
            if age < _GRACE_SECONDS:
                logger.info(
                    "WinGet validation install grace active (%.0fs since install marker)",
                    age,
                )
                return True
    return False


def is_winget_validation_mode() -> bool:
    """True when llama.cpp / CUDA backend loads must be deferred for package validation."""
    explicit = _explicit_env_requested()
    if explicit is not None:
        return explicit
    return _install_grace_active()


def configure_winget_validation_mode(args: Any | None = None) -> None:
    """Sync ``QUBE_WINGET_VALIDATION`` from ``--winget-validation`` CLI flag."""
    if args is not None and getattr(args, "winget_validation", False):
        os.environ["QUBE_WINGET_VALIDATION"] = "1"


def apply_winget_validation_bootstrap_shortcut() -> bool:
    """
    Skip first-run consent and use mock bootstrap downloads.

    Returns True when the shortcut was applied (fresh install under validation mode).
    """
    if not is_winget_validation_mode():
        return False
    from core.bootstrap_selection import is_bootstrap_completed, save_bootstrap_selection

    if is_bootstrap_completed():
        return False
    save_bootstrap_selection(set())
    os.environ["QUBE_BOOTSTRAP_MOCK_DOWNLOAD"] = "1"
    logger.info(
        "WinGet validation mode: shell bootstrap (no models, mock downloads)."
    )
    return True


def validation_hardware_profile_stub() -> dict[str, Any]:
    """Hardware telemetry stub that avoids NVML / GPU probes during validation."""
    return {
        "gpu_memory_kind": "none",
        "gpu_memory_kind_label": "Deferred (WinGet validation mode)",
        "is_unified_gpu_memory": False,
        "vram_budget_bytes": 0,
        "vram_budget_gb": 0.0,
        "max_safe_n_gpu_layers": 0,
    }


def smoke_result_path() -> Path:
    from core.paths import user_data_root

    return user_data_root() / _SMOKE_RESULT_NAME


def write_smoke_result(*, boot_complete: bool = True) -> None:
    """Record whether ``llama_cpp`` was imported during a validation-mode boot."""
    from core.llama_cpp_import import llama_import_was_attempted

    payload = {
        "boot_complete": bool(boot_complete),
        "llama_import_attempted": bool(llama_import_was_attempted()),
        "ok": not llama_import_was_attempted(),
        "validation_mode": True,
    }
    path = smoke_result_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("WinGet validation smoke result written to %s", path)


def reset_winget_validation_state_for_tests() -> None:
    """Clear env override (unit tests only)."""
    os.environ.pop("QUBE_WINGET_VALIDATION", None)
