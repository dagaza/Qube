"""WinGet / Defender-safe startup mode for CUDA Windows builds."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("Qube.WinGetValidation")

# Post-install grace for packaged CUDA builds (WinGet step 08 launches soon after install).
_GRACE_SECONDS = 20 * 60
_INSTALL_TS_NAMES = (".qube-install-ts",)
_EXPLICIT_TRUTHY = frozenset({"1", "true", "yes", "on"})
_EXPLICIT_FALSY = frozenset({"0", "false", "no", "off"})
_SMOKE_RESULT_NAME = ".winget-validation-smoke.json"
_BOOT_STATE_NAME = ".winget-validation-boot-state.json"
_BOOT_TRACE_NAME = ".winget-validation-boot-trace.jsonl"
_GRACE_TRACE_NAME = ".winget-validation-grace.jsonl"

ValidationModeLabel = Literal["smoke", "install_grace"]


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


def is_winget_install_grace() -> bool:
    """True during post-install grace without explicit smoke validation flags."""
    if _explicit_env_requested() is not None:
        return False
    return _install_grace_active()


def is_winget_validation_mode() -> bool:
    """True when llama.cpp / CUDA backend loads must be deferred for package validation."""
    explicit = _explicit_env_requested()
    if explicit is not None:
        return explicit
    return _install_grace_active()


def validation_mode_label() -> ValidationModeLabel | None:
    """Return the active WinGet validation path for diagnostics payloads."""
    if is_winget_smoke_validation():
        return "smoke"
    if is_winget_install_grace():
        return "install_grace"
    return None


def validation_diagnostics_active() -> bool:
    """True when validation boot diagnostics should be written to disk."""
    return validation_mode_label() is not None


def configure_winget_validation_mode(args: Any | None = None) -> None:
    """Sync ``QUBE_WINGET_VALIDATION`` from ``--winget-validation`` CLI flag."""
    if args is not None and getattr(args, "winget_validation", False):
        os.environ["QUBE_WINGET_VALIDATION"] = "1"
    if validation_diagnostics_active():
        record_boot_state("validation_mode_configured")


def apply_winget_validation_bootstrap_shortcut() -> bool:
    """
    Skip first-run consent and use a shell bootstrap selection.

    Smoke mode also enables mock bootstrap downloads for CI.
    Install grace uses the same consent bypass without mock downloads.
    """
    mode = validation_mode_label()
    if mode is None:
        return False
    from core.bootstrap_selection import is_bootstrap_completed, save_bootstrap_selection

    if is_bootstrap_completed():
        record_boot_state("bootstrap_shortcut_skipped", reason="already_completed")
        return False
    save_bootstrap_selection(set())
    if mode == "smoke":
        os.environ["QUBE_BOOTSTRAP_MOCK_DOWNLOAD"] = "1"
    from core.bootstrap_selection import is_bootstrap_completed

    if not is_bootstrap_completed():
        logger.warning(
            "WinGet validation bootstrap shortcut did not persist completion; "
            "consent bypass still active via validation mode."
        )
    logger.info(
        "WinGet validation mode (%s): shell bootstrap (no models%s).",
        mode,
        ", mock downloads" if mode == "smoke" else "",
    )
    record_boot_state("bootstrap_shortcut_applied", mock_downloads=mode == "smoke")
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


def grace_trace_path() -> Path:
    from core.paths import user_data_root

    return user_data_root() / _GRACE_TRACE_NAME


def record_boot_state(state: str, **extra: Any) -> None:
    """Write diagnostic boot progress for CI and WinGet install-grace launches."""
    mode = validation_mode_label()
    if mode is None:
        return
    payload: dict[str, Any] = {
        "mode": mode,
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
    if mode == "install_grace":
        with grace_trace_path().open("a", encoding="utf-8") as grace_file:
            grace_file.write(line)
    if extra:
        logger.info("WinGet validation boot state: %s (%s)", state, extra)
    else:
        logger.info("WinGet validation boot state: %s", state)


def record_validation_entry(*, stage: str = "process_entry", **extra: Any) -> None:
    """Early breadcrumb before heavy imports when validation diagnostics are active."""
    if not validation_diagnostics_active():
        return
    record_boot_state(
        stage,
        argv=list(sys.argv),
        executable=str(getattr(sys, "executable", "")),
        **extra,
    )


def log_validation_startup_summary(
    *,
    shortcut_applied: bool,
    needs_consent: bool,
) -> None:
    """Log and persist post-shortcut bootstrap/consent decisions for CI diagnosis."""
    mode = validation_mode_label()
    if mode is None:
        return
    from core.bootstrap_selection import is_bootstrap_completed

    bootstrap_completed = is_bootstrap_completed()
    logger.info(
        "Validation startup (%s): shortcut=%s bootstrap_completed=%s needs_consent=%s",
        mode,
        shortcut_applied,
        bootstrap_completed,
        needs_consent,
    )
    record_boot_state(
        "consent_decision",
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
    mode = validation_mode_label()
    if mode is None:
        return
    from core.llama_cpp_import import llama_import_was_attempted

    llama_attempted = llama_import_was_attempted()
    payload = {
        "boot_complete": bool(boot_complete),
        "llama_import_attempted": bool(llama_attempted),
        "mode": mode,
        "ok": not llama_attempted,
        "stage": "boot_complete",
        "validation_mode": True,
    }
    _write_smoke_payload(payload)
    record_boot_state("boot_complete")


def write_smoke_failure(*, stage: str, error: str) -> None:
    """Record a validation-mode boot failure for CI (non-modal path)."""
    mode = validation_mode_label()
    if mode is None:
        return
    from core.llama_cpp_import import llama_import_was_attempted

    payload = {
        "boot_complete": False,
        "error": str(error),
        "llama_import_attempted": bool(llama_import_was_attempted()),
        "mode": mode,
        "ok": False,
        "stage": str(stage),
        "validation_mode": True,
    }
    _write_smoke_payload(payload)
    record_boot_state("boot_failed", stage=stage, error=str(error))


def reset_winget_validation_state_for_tests() -> None:
    """Clear env override (unit tests only)."""
    os.environ.pop("QUBE_WINGET_VALIDATION", None)
