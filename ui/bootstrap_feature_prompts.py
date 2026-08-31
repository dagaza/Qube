"""Confirm-and-download prompts when a feature needs a missing bootstrap model."""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtWidgets import QWidget

from core.bootstrap_download import model_is_present
from core.bootstrap_manifest import BOOTSTRAP_MODELS, BootstrapModelId, format_byte_size
from core.bootstrap_selection import apply_downloaded_bootstrap_model
from ui.components.prestige_dialog import PrestigeDialog
from workers.bootstrap_model_download_worker import BootstrapModelDownloadWorker

logger = logging.getLogger("Qube.UI.BootstrapFeaturePrompts")

MAIN_LLM_REQUIRED_TITLE = "Model required"
MAIN_LLM_REQUIRED_BODY = (
    "Chat needs a local conversational model.\n\n"
    "Open Model Manager to download a model that fits your hardware, "
    "then select it from the toolbar."
)


def _open_model_manager_from_parent(parent: QWidget) -> None:
    window = parent.window()
    if window is None:
        return
    if hasattr(window, "_open_model_manager_page"):
        window._open_model_manager_page()
    elif hasattr(window, "_on_notification_action"):
        window._on_notification_action("open_models")

def format_bootstrap_model_confirm_body(
    model_id: BootstrapModelId,
    *,
    feature_label: str,
) -> str:
    spec = BOOTSTRAP_MODELS[model_id]
    size = format_byte_size(spec.size_bytes)
    return (
        f"{feature_label} needs {spec.label} (~{size} download).\n\n"
        f"Source: {spec.source_display}\n\n"
        "Download now and enable this feature?"
    )


def ensure_bootstrap_model_downloaded(
    parent: QWidget,
    model_id: BootstrapModelId,
    *,
    feature_label: str,
    is_dark: bool | None = None,
) -> bool:
    """Return True when the model is on disk (after optional download)."""
    if model_is_present(model_id):
        return True

    if is_dark is None:
        is_dark = getattr(parent.window(), "_is_dark_theme", True)

    confirm = PrestigeDialog(
        parent,
        f"Download {BOOTSTRAP_MODELS[model_id].label}?",
        format_bootstrap_model_confirm_body(model_id, feature_label=feature_label),
        is_dark=is_dark,
        tone="danger",
        dialog_width=460,
    )
    if not confirm.exec():
        return False

    worker = BootstrapModelDownloadWorker(model_id)
    progress_dlg = PrestigeDialog(
        parent,
        f"Downloading {BOOTSTRAP_MODELS[model_id].label}",
        "Working… this may take a while on first download.",
        is_dark=is_dark,
        show_cancel=False,
    )
    progress_dlg.show()

    result: dict[str, Any] = {"ok": False, "err": None}

    def _on_ok(used_mock: bool) -> None:
        if used_mock:
            result["err"] = (
                "Mock download finished but no files were written. "
                "Unset QUBE_BOOTSTRAP_MOCK_DOWNLOAD or use QUBE_BOOTSTRAP_REAL_DOWNLOAD=1."
            )
        else:
            result["ok"] = True

    def _on_failed(err: str) -> None:
        result["err"] = err or "Download failed."

    worker.finished_ok.connect(_on_ok)
    worker.failed.connect(_on_failed)
    worker.start()
    worker.wait()

    try:
        progress_dlg.accept()
    except Exception:
        pass

    if not result["ok"]:
        PrestigeDialog(
            parent,
            "Download failed",
            str(result["err"] or "Model download failed."),
            is_dark=is_dark,
            tone="danger",
        ).exec()
        return False

    apply_downloaded_bootstrap_model(model_id)
    _reload_runtime_for_model(parent, model_id)
    return model_is_present(model_id)


def _reload_runtime_for_model(parent: QWidget, model_id: BootstrapModelId) -> None:
    window = parent.window()
    workers = getattr(window, "workers", None) or {}
    settings_view = getattr(window, "_settings_view", None)

    if model_id == BootstrapModelId.WHISPER_SMALL:
        stt = workers.get("stt")
        if stt is not None and hasattr(stt, "reload_from_settings"):
            try:
                stt.reload_from_settings()
            except Exception:
                logger.debug("STT reload after download failed", exc_info=True)
        if settings_view is not None and hasattr(settings_view, "_reload_stt_from_settings"):
            settings_view._reload_stt_from_settings()
        return

    if model_id == BootstrapModelId.KOKORO_TTS:
        if settings_view is not None and hasattr(settings_view, "_reload_tts_from_settings"):
            settings_view._reload_tts_from_settings()
        tts = workers.get("tts")
        if tts is not None:
            try:
                from core.tts_models import resolve_boot_tts_path

                tts.load_voice(resolve_boot_tts_path())
            except Exception:
                logger.debug("TTS reload after download failed", exc_info=True)
        return

    if model_id in {BootstrapModelId.SIDECAR_QWEN17, BootstrapModelId.SIDECAR_QWEN05}:
        sidecar_worker = workers.get("sidecar_worker")
        if sidecar_worker is not None and hasattr(sidecar_worker, "reload_from_settings"):
            try:
                sidecar_worker.reload_from_settings()
            except Exception:
                logger.debug("Sidecar reload after download failed", exc_info=True)
        if settings_view is not None and hasattr(settings_view, "_reload_sidecar_from_settings"):
            settings_view._reload_sidecar_from_settings()
        return

    if model_id in {
        BootstrapModelId.LLM_QWEN35_9B,
        BootstrapModelId.LLM_GEMMA4_E4B,
        BootstrapModelId.LLM_NEMOTRON_NANO,
    }:
        llm = workers.get("llm")
        if llm is not None and hasattr(llm, "refresh_native_model_from_settings"):
            try:
                llm.refresh_native_model_from_settings()
            except Exception:
                logger.debug("Native LLM reload after download failed", exc_info=True)
        if settings_view is not None and hasattr(settings_view, "_sync_active_native_label"):
            settings_view._sync_active_native_label()


def ensure_search_models_for_feature(
    parent: QWidget,
    *,
    feature_label: str,
    is_dark: bool | None = None,
) -> bool:
    from core.embedding_models import embedding_model_available, probe_embedding_preset_available

    if embedding_model_available():
        return True

    if is_dark is None:
        is_dark = getattr(parent.window(), "_is_dark_theme", True)

    confirm = PrestigeDialog(
        parent,
        "Prepare search models?",
        (
            f"{feature_label} needs the active Fast/Balanced/Power search preset.\n\n"
            "Download and load it now? (Requires internet on first use.)"
        ),
        is_dark=is_dark,
        tone="danger",
        dialog_width=420,
    )
    if not confirm.exec():
        return False

    from ui.views.settings.handlers.bootstrap_downloads import EmbeddingWarmupWorker

    worker = EmbeddingWarmupWorker()
    progress_dlg = PrestigeDialog(
        parent,
        "Preparing search models",
        "Downloading and loading the active search preset…",
        is_dark=is_dark,
        show_cancel=False,
    )
    progress_dlg.show()

    result: dict[str, Any] = {"ok": False, "err": None}

    def _on_ok() -> None:
        result["ok"] = True

    def _on_failed(message: str) -> None:
        result["err"] = message

    worker.finished_ok.connect(_on_ok)
    worker.failed.connect(_on_failed)
    worker.start()
    worker.wait()

    try:
        progress_dlg.accept()
    except Exception:
        pass

    if not result["ok"]:
        PrestigeDialog(
            parent,
            "Search models not ready",
            str(result["err"] or "Could not prepare search models."),
            is_dark=is_dark,
            tone="danger",
        ).exec()
        return False

    window = parent.window()
    workers = getattr(window, "workers", None) or {}
    settings_view = getattr(window, "_settings_view", None)
    if settings_view is not None and hasattr(settings_view, "_reload_embedder_from_settings"):
        settings_view._reload_embedder_from_settings()
        if hasattr(settings_view, "embedding_model_changed"):
            settings_view.embedding_model_changed.emit()
    llm = workers.get("llm")
    if llm is not None:
        try:
            from rag.embedder import EmbeddingModel

            embedder = EmbeddingModel()
            llm.embedder = embedder
            workers["embedder"] = embedder
            from workers.intent_router import EmbeddingCache

            llm.embedding_cache = EmbeddingCache(embedder)
        except Exception:
            logger.debug("Embedder reload after search warmup failed", exc_info=True)
    return probe_embedding_preset_available()


def main_llm_model_available() -> bool:
    import os

    from core.app_settings import get_engine_mode, get_internal_model_path, resolve_internal_model_path

    if get_engine_mode() != "internal":
        return True
    path = resolve_internal_model_path(get_internal_model_path() or "")
    return bool(path and os.path.isfile(path))


def ensure_main_llm_for_chat(parent: QWidget, *, is_dark: bool | None = None) -> bool:
    """Return True when a conversational model is ready; otherwise prompt for Model Manager."""
    if main_llm_model_available():
        return True

    if is_dark is None:
        is_dark = getattr(parent.window(), "_is_dark_theme", True)

    confirm = PrestigeDialog(
        parent,
        MAIN_LLM_REQUIRED_TITLE,
        MAIN_LLM_REQUIRED_BODY,
        is_dark=is_dark,
        tone="danger",
        dialog_width=460,
        confirm_text="Open Model Manager",
    )
    if confirm.exec():
        _open_model_manager_from_parent(parent)
    return False
