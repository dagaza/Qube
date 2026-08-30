"""Persist and apply first-run bootstrap choices."""

from __future__ import annotations

import json
import logging
import os
import shutil

from core.app_settings import (
    get_settings_store,
    set_internal_model_path,
    set_sidecar_enabled,
    set_sidecar_model_path,
)
from core.auxiliary_cognition import bundled_default_path as cognition_default_path
from core.bootstrap_manifest import (
    BOOTSTRAP_MODELS,
    BootstrapModelId,
    default_selection,
    format_byte_size,
)
from core.paths import models_root
from workers.model_download_worker import SAFETY_BUFFER_BYTES

logger = logging.getLogger("Qube.Bootstrap")

KEY_COMPLETED = "qube.bootstrap.completed"
KEY_SELECTED = "qube.bootstrap.selectedModels"
KEY_VOICE_IN = "qube.bootstrap.voiceInputDefault"
KEY_VOICE_OUT = "qube.bootstrap.voiceOutputDefault"


def available_disk_bytes() -> int:
    try:
        return int(shutil.disk_usage(models_root()).free)
    except OSError:
        return 0


def format_available_disk() -> str:
    return format_byte_size(available_disk_bytes())


def is_bootstrap_completed() -> bool:
    return bool(get_settings_store().get(KEY_COMPLETED, False))


def _deserialize_selected(raw: str) -> set[BootstrapModelId]:
    out: set[BootstrapModelId] = set()
    if not raw:
        return out
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return out
    if not isinstance(items, list):
        return out
    for item in items:
        try:
            out.add(BootstrapModelId(str(item)))
        except ValueError:
            continue
    return out


def _serialize_selected(selected: set[BootstrapModelId]) -> str:
    return json.dumps(sorted(m.value for m in selected))


def get_selected_model_ids() -> set[BootstrapModelId]:
    raw = get_settings_store().get(KEY_SELECTED, "")
    return _deserialize_selected(str(raw or ""))


def get_voice_input_default() -> bool:
    return bool(get_settings_store().get(KEY_VOICE_IN, True))


def get_voice_output_default() -> bool:
    return bool(get_settings_store().get(KEY_VOICE_OUT, True))


def maybe_seed_bootstrap_selection_for_existing_install() -> None:
    """Pre-fill stored selection from on-disk models without skipping consent.

    Dev installs and upgrade paths often already have GGUF assets under
    ``models_root()``. Seeding ``KEY_SELECTED`` lets the consent dialog reflect
    what is installed, but the user must still confirm via
    :func:`save_bootstrap_selection` before splash downloads run.
    """
    if is_bootstrap_completed() or get_selected_model_ids():
        return
    from core.bootstrap_download import infer_installed_selection

    inferred = infer_installed_selection()
    if not inferred:
        return
    store = get_settings_store()
    store.set(KEY_SELECTED, _serialize_selected(inferred))
    store.set(KEY_VOICE_IN, BootstrapModelId.WHISPER_SMALL in inferred)
    store.set(KEY_VOICE_OUT, BootstrapModelId.KOKORO_TTS in inferred)
    logger.info(
        "Seeded bootstrap selection from installed models (consent still required): %s",
        _serialize_selected(inferred),
    )


def save_bootstrap_selection(selected: set[BootstrapModelId]) -> None:
    store = get_settings_store()
    store.set(KEY_SELECTED, _serialize_selected(selected))
    store.set(KEY_VOICE_IN, BootstrapModelId.WHISPER_SMALL in selected)
    store.set(KEY_VOICE_OUT, BootstrapModelId.KOKORO_TTS in selected)
    store.set(KEY_COMPLETED, True)
    apply_bootstrap_selection(selected)
    logger.info("Bootstrap selection saved: %s", _serialize_selected(selected))


def apply_bootstrap_selection(selected: set[BootstrapModelId]) -> None:
    """Map bootstrap choices onto runtime settings paths and flags."""
    if BootstrapModelId.SIDECAR_QWEN17 in selected or BootstrapModelId.SIDECAR_QWEN05 in selected:
        set_sidecar_enabled(True)
        if BootstrapModelId.SIDECAR_QWEN05 in selected:
            path = _resolved_download_path(BootstrapModelId.SIDECAR_QWEN05)
            if path:
                set_sidecar_model_path(path)
        else:
            path = _resolved_download_path(BootstrapModelId.SIDECAR_QWEN17)
            if path and os.path.isfile(path):
                set_sidecar_model_path(path)
            elif os.path.isfile(cognition_default_path()):
                set_sidecar_model_path(cognition_default_path())
    else:
        set_sidecar_enabled(False)

    main_llm = None
    for mid in (
        BootstrapModelId.LLM_QWEN35_9B,
        BootstrapModelId.LLM_GEMMA4_E4B,
        BootstrapModelId.LLM_NEMOTRON_NANO,
    ):
        if mid in selected:
            main_llm = mid
            break
    if main_llm is not None:
        path = _resolved_download_path(main_llm)
        if path and os.path.isfile(path):
            set_internal_model_path(path)


def _resolved_download_path(model_id: BootstrapModelId) -> str:
    from core.bootstrap_download import resolve_model_destination

    dest = resolve_model_destination(model_id)
    return str(dest) if dest is not None else ""


def total_selected_bytes(
    selected: set[BootstrapModelId],
    *,
    sizes: dict[BootstrapModelId, int] | None = None,
) -> int:
    if sizes:
        return sum(sizes.get(mid, BOOTSTRAP_MODELS[mid].size_bytes) for mid in selected if mid in BOOTSTRAP_MODELS)
    return sum(BOOTSTRAP_MODELS[mid].size_bytes for mid in selected if mid in BOOTSTRAP_MODELS)


def required_bytes_for(
    selected: set[BootstrapModelId],
    *,
    sizes: dict[BootstrapModelId, int] | None = None,
) -> int:
    return total_selected_bytes(selected, sizes=sizes) + SAFETY_BUFFER_BYTES


def budget_headroom_bytes(
    selected: set[BootstrapModelId],
    *,
    sizes: dict[BootstrapModelId, int] | None = None,
) -> int:
    return available_disk_bytes() - required_bytes_for(selected, sizes=sizes)


def selection_within_budget(
    selected: set[BootstrapModelId],
    *,
    sizes: dict[BootstrapModelId, int] | None = None,
) -> bool:
    return budget_headroom_bytes(selected, sizes=sizes) >= 0


def can_add_model(
    selected: set[BootstrapModelId],
    model_id: BootstrapModelId,
    *,
    sizes: dict[BootstrapModelId, int] | None = None,
) -> bool:
    if model_id in selected:
        return True
    trial = set(selected)
    trial.add(model_id)
    from core.bootstrap_manifest import normalize_selection

    trial = normalize_selection(trial)
    return selection_within_budget(trial, sizes=sizes)


def models_blocked_by_disk(
    selected: set[BootstrapModelId],
    candidates: set[BootstrapModelId],
    *,
    sizes: dict[BootstrapModelId, int] | None = None,
) -> set[BootstrapModelId]:
    blocked: set[BootstrapModelId] = set()
    for model_id in candidates:
        if model_id in selected:
            continue
        if not can_add_model(selected, model_id, sizes=sizes):
            blocked.add(model_id)
    return blocked


def preflight_download(
    selected: set[BootstrapModelId],
    *,
    sizes: dict[BootstrapModelId, int] | None = None,
) -> tuple[bool, str]:
    if not selected:
        return True, ""
    required = required_bytes_for(selected, sizes=sizes)
    free = available_disk_bytes()
    if free < required:
        return (
            False,
            f"Need {format_byte_size(required)} free disk space "
            f"({format_byte_size(total_selected_bytes(selected, sizes=sizes))} downloads + "
            f"{format_byte_size(SAFETY_BUFFER_BYTES)} safety buffer); "
            f"only {format_byte_size(free)} available.",
        )
    return True, ""


def maybe_reset_stale_shell_bootstrap_completion() -> bool:
    """Clear bootstrap.completed left by CUDA install-grace WinGet smoke (no models on disk)."""
    if not is_bootstrap_completed():
        return False
    if get_selected_model_ids():
        return False
    from core.bootstrap_download import infer_installed_selection, model_is_present

    if infer_installed_selection():
        return False
    if any(model_is_present(mid) for mid in BootstrapModelId):
        return False
    store = get_settings_store()
    store.set(KEY_COMPLETED, False)
    store.set(KEY_SELECTED, "")
    logger.warning(
        "Reset stale bootstrap completion (no selected models and no bootstrap assets on disk)."
    )
    return True


def should_show_bootstrap_consent() -> bool:
    from core.winget_validation import is_winget_smoke_validation

    maybe_reset_stale_shell_bootstrap_completion()
    if is_winget_smoke_validation():
        return False
    maybe_seed_bootstrap_selection_for_existing_install()
    return not is_bootstrap_completed()


def effective_bootstrap_selection() -> set[BootstrapModelId]:
    from core.winget_validation import is_winget_smoke_validation

    if is_winget_smoke_validation():
        return set()
    if is_bootstrap_completed():
        return get_selected_model_ids()
    selected = get_selected_model_ids()
    if selected:
        return selected
    return default_selection(advanced=False)


def apply_downloaded_bootstrap_model(model_id: BootstrapModelId) -> None:
    """Apply runtime settings after a single on-demand bootstrap download."""
    from core.bootstrap_download import resolve_model_destination

    if model_id == BootstrapModelId.WHISPER_SMALL:
        return
    if model_id == BootstrapModelId.KOKORO_TTS:
        return
    if model_id in {BootstrapModelId.SIDECAR_QWEN17, BootstrapModelId.SIDECAR_QWEN05}:
        dest = resolve_model_destination(model_id)
        if dest is not None and dest.is_file():
            set_sidecar_enabled(True)
            set_sidecar_model_path(str(dest))
        return
    if model_id in {
        BootstrapModelId.LLM_QWEN35_9B,
        BootstrapModelId.LLM_GEMMA4_E4B,
        BootstrapModelId.LLM_NEMOTRON_NANO,
    }:
        path = _resolved_download_path(model_id)
        if path and os.path.isfile(path):
            set_internal_model_path(path)
        return
    if model_id == BootstrapModelId.SEARCH_PRESET_BALANCED:
        from core.bootstrap_search_models import embedding_preset_cached_on_disk
        from core.embedding_models import mark_embedding_preset_available
        from core.embedding_modes import DEFAULT_MODE

        if embedding_preset_cached_on_disk(DEFAULT_MODE):
            mark_embedding_preset_available(DEFAULT_MODE)
        return
