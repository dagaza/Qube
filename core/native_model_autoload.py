"""Startup autoload handling when a saved native .gguf path is missing on disk."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from core.app_settings import (
    get_internal_model_path,
    missing_gguf_shards,
    resolve_internal_model_path,
    set_internal_model_path,
)
from core.notification_types import NotificationEvent, NotificationSeverity
from core.system_capabilities_store import SystemCapabilitiesStore

logger = logging.getLogger("Qube.NativeModelAutoload")

ACTION_OPEN_MODELS = "open_models"
ACTION_OPEN_MODELS_REDOWNLOAD = "open_models_redownload"
_REDOWNLOAD_SEP = "\x1e"


@dataclass(frozen=True)
class NativeModelRefreshOutcome:
    """Result of attempting to refresh the native model from settings."""

    attempted: bool
    missing_display_name: str = ""
    cleared_stale_path: bool = False
    redownload_repo_id: str = ""
    redownload_filename: str = ""
    notify_user: bool = False
    missing_shards: bool = False


def encode_redownload_action_id(repo_id: str, filename: str) -> str:
    repo = str(repo_id or "").strip()
    name = Path(str(filename or "").strip()).name
    return f"{ACTION_OPEN_MODELS_REDOWNLOAD}{_REDOWNLOAD_SEP}{repo}{_REDOWNLOAD_SEP}{name}"


def decode_redownload_action_id(action_id: str) -> tuple[str, str] | None:
    if not str(action_id or "").startswith(ACTION_OPEN_MODELS_REDOWNLOAD):
        return None
    parts = str(action_id).split(_REDOWNLOAD_SEP, 2)
    if len(parts) != 3:
        return None
    repo, filename = parts[1].strip(), Path(parts[2].strip()).name
    if not repo or not filename:
        return None
    return repo, filename


def lookup_hf_redownload_target(path: str) -> tuple[str, str] | None:
    """Return ``(repo_id, gguf_basename)`` when Hub provenance is known for ``path``."""
    resolved = resolve_internal_model_path(str(path or ""))
    if not resolved:
        return None
    store = SystemCapabilitiesStore()
    repo = store.get_model_hf_provenance(resolved)
    basename = os.path.basename(resolved)
    if repo:
        return repo, basename
    for stored_path, stored_repo in store.load_model_hf_provenance_map().items():
        if os.path.basename(stored_path) == basename and stored_repo:
            return stored_repo, basename
    return None


def clear_stale_native_model_path(path: str) -> bool:
    """Clear saved native model path and drop provenance when the file is gone."""
    resolved = resolve_internal_model_path(str(path or ""))
    if not resolved:
        current = resolve_internal_model_path(get_internal_model_path())
        if not current:
            return False
        resolved = current
    set_internal_model_path("")
    store = SystemCapabilitiesStore()
    store.remove_model_hf_provenance(resolved)
    logger.info("Cleared stale native model path (%s).", os.path.basename(resolved))
    return True


def missing_autoload_model_notification(
    *,
    display_name: str,
    redownload_repo_id: str = "",
    redownload_filename: str = "",
    missing_shards: bool = False,
) -> NotificationEvent:
    label = display_name.strip() or "Saved model"
    if missing_shards:
        body = (
            f'"{label}" is incomplete on disk (missing GGUF shard files). '
            "Download all parts again or choose another model."
        )
    else:
        body = (
            f'"{label}" is no longer on disk. '
            "Select another model or download it again from Model Manager."
        )
    can_redownload = bool(redownload_repo_id.strip() and redownload_filename.strip())
    if can_redownload:
        return NotificationEvent(
            title="Saved model missing",
            body=body,
            severity=NotificationSeverity.WARNING,
            category="system",
            action_label="Re-download",
            action_id=encode_redownload_action_id(
                redownload_repo_id,
                redownload_filename,
            ),
            auto_dismiss_ms=0,
            dedupe_key="missing_autoload_native_model",
            rate_limit_key="missing_autoload_native_model",
            rate_limit_sec=60.0,
            tray_bump=True,
            icon_name="fa5s.cube",
        )
    return NotificationEvent(
        title="Saved model missing",
        body=body,
        severity=NotificationSeverity.WARNING,
        category="system",
        action_label="Open Model Manager",
        action_id=ACTION_OPEN_MODELS,
        auto_dismiss_ms=0,
        dedupe_key="missing_autoload_native_model",
        rate_limit_key="missing_autoload_native_model",
        rate_limit_sec=60.0,
        tray_bump=True,
        icon_name="fa5s.cube",
    )


def evaluate_native_model_refresh(
    path: str,
    *,
    autoload: bool = False,
) -> NativeModelRefreshOutcome:
    """Check ``path`` before native load; clear stale settings when absent."""
    resolved = resolve_internal_model_path(str(path or ""))
    if not resolved:
        return NativeModelRefreshOutcome(attempted=bool(autoload and path))

    display_name = os.path.basename(resolved)
    missing_file = not os.path.isfile(resolved)
    missing_shards = False if missing_file else bool(missing_gguf_shards(resolved))

    if not missing_file and not missing_shards:
        return NativeModelRefreshOutcome(attempted=bool(autoload))

    redownload = lookup_hf_redownload_target(resolved)
    cleared = clear_stale_native_model_path(resolved)
    if redownload:
        repo_id, filename = redownload
    else:
        repo_id, filename = "", display_name

    return NativeModelRefreshOutcome(
        attempted=True,
        missing_display_name=display_name,
        cleared_stale_path=cleared,
        redownload_repo_id=repo_id,
        redownload_filename=filename,
        notify_user=autoload,
        missing_shards=missing_shards,
    )
