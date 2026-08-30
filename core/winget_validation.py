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
_BOOT_STATE_NAME = ".winget-validation-boot-state.json"
_BOOT_TRACE_NAME = ".winget-validation-boot-trace.jsonl"


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


def is_winget_smoke_validation() -> bool:
    """True for explicit CI / sandbox runs (--winget-validation or QUBE_WINGET_VALIDATION=1)."""
    return _explicit_env_requested() is True


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
    if is_winget_smoke_validation():
        record_boot_state("validation_mode_configured")


def apply_winget_validation_bootstrap_shortcut() -> bool:
    """
    Skip first-run consent and use mock bootstrap downloads.

    Returns True when the shortcut was applied (explicit WinGet smoke / CI only).
    """
    if not is_winget_smoke_validation():
        return False
    from core.bootstrap_selection import is_bootstrap_completed, save_bootstrap_selection

    if is_bootstrap_completed():
        record_boot_state("bootstrap_shortcut_skipped", reason="already_completed")
        return False
    save_bootstrap_selection(set())
    os.environ["QUBE_BOOTSTRAP_MOCK_DOWNLOAD"] = "1"
    from core.bootstrap_selection import is_bootstrap_completed

    if not is_bootstrap_completed():
        logger.warning(
            "WinGet validation bootstrap shortcut did not persist completion; "
            "consent bypass still active via validation mode."
        )
    logger.info(
        "WinGet validation mode: shell bootstrap (no models, mock downloads)."
    )
    record_boot_state("bootstrap_shortcut_applied")
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


def boot_state_path() -> Path:
    from core.paths import user_data_root

    return user_data_root() / _BOOT_STATE_NAME


def boot_trace_path() -> Path:
    from core.paths import user_data_root

    return user_data_root() / _BOOT_TRACE_NAME


def record_boot_state(state: str, **extra: Any) -> None:
    """Write diagnostic boot progress for CI when explicit smoke validation is active."""
    if not is_winget_smoke_validation():
        return
    payload: dict[str, Any] = {
        "state": state,
        "timestamp": time.time(),
    }
    payload.update(extra)
    line = json.dumps(payload, sort_keys=True) + "\n"
    root = boot_state_path().parent
    root.mkdir(parents=True, exist_ok=True)
    boot_state_path().write_text(line, encoding="utf-8")
    with boot_trace_path().open("a", encoding="utf-8") as trace_file:
        trace_file.write(line)
    if extra:
        logger.info("WinGet validation boot state: %s (%s)", state, extra)
    else:
        logger.info("WinGet validation boot state: %s", state)


def log_validation_startup_summary(
    *,
    shortcut_applied: bool,
    needs_consent: bool,
) -> None:
    """Log and persist post-shortcut bootstrap/consent decisions for CI diagnosis."""
    if not is_winget_smoke_validation():
        return
    from core.bootstrap_selection import is_bootstrap_completed

    validation_mode = is_winget_smoke_validation()
    bootstrap_completed = is_bootstrap_completed()
    logger.info(
        "Validation smoke startup: validation=%s shortcut=%s "
        "bootstrap_completed=%s needs_consent=%s",
        validation_mode,
        shortcut_applied,
        bootstrap_completed,
        needs_consent,
    )
    record_boot_state(
        "consent_decision",
        validation_mode=validation_mode,
        shortcut_applied=shortcut_applied,
        bootstrap_completed=bootstrap_completed,
        needs_consent=needs_consent,
    )


def _write_smoke_payload(payload: dict[str, Any]) -> None:
    path = smoke_result_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("WinGet validation smoke result written to %s", path)


def write_smoke_result(*, boot_complete: bool = True) -> None:
    """Record whether ``llama_cpp`` was imported during a validation-mode boot."""
    from core.llama_cpp_import import llama_import_was_attempted

    llama_attempted = llama_import_was_attempted()
    payload = {
        "boot_complete": bool(boot_complete),
        "llama_import_attempted": bool(llama_attempted),
        "ok": not llama_attempted,
        "stage": "boot_complete",
        "validation_mode": True,
    }
    _write_smoke_payload(payload)
    record_boot_state("boot_complete")


def write_smoke_failure(*, stage: str, error: str) -> None:
    """Record a validation-mode boot failure for CI (non-modal path)."""
    if not is_winget_smoke_validation():
        return
    from core.llama_cpp_import import llama_import_was_attempted

    payload = {
        "boot_complete": False,
        "error": str(error),
        "llama_import_attempted": bool(llama_import_was_attempted()),
        "ok": False,
        "stage": str(stage),
        "validation_mode": True,
    }
    _write_smoke_payload(payload)
    record_boot_state("boot_failed", stage=stage, error=str(error))


def reset_winget_validation_state_for_tests() -> None:
    """Clear env override (unit tests only)."""
    os.environ.pop("QUBE_WINGET_VALIDATION", None)
