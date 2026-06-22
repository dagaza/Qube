"""Settings handler mixin: KnowledgeHandlersMixin (embedding model + triggers)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem

from core.app_settings import (
    get_advanced_embedding_unlocked,
    get_embedding_mode,
    set_advanced_embedding_unlocked,
    set_embedding_model_path,
    set_embedding_mode,
)
from core.bootstrap_search_models import (
    format_embedding_mode_switch_confirm_body,
    format_search_preset_download_failure,
)
from core.embedding_modes import get_mode_spec, list_mode_specs, normalize_mode_id
from core.embedding_models import (
    get_embedding_models_dir,
    gguf_override_available,
    list_selectable_embedding_models,
    preset_embedder_ready,
    resolve_active_gguf_path,
    validate_embedding_model_path,
)
from ui.views.settings.handlers.bootstrap_downloads import EmbeddingWarmupWorker
from ui.components.prestige_dialog import PrestigeDialog

logger = logging.getLogger("Qube.UI.Settings")

EMBEDDING_ENTRY_DELETABLE_ROLE = int(Qt.ItemDataRole.UserRole) + 3


class KnowledgeHandlersMixin:
    """Embedding model loader and related Knowledge settings behavior."""

    def _on_advanced_embedding_toggled(self, checked: bool) -> None:
        if checked:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            dlg = PrestigeDialog(
                self.window(),
                "Advanced embedding settings",
                "Custom embedding models are for expert use only.\n\n"
                "Models must be .gguf files placed in the embedding folder. "
                "Using a custom model reprocesses your library and memories.\n\nContinue?",
                is_dark=is_dark,
                tone="danger",
                dialog_width=450,
            )
            if not dlg.exec():
                self.advanced_embedding_toggle.blockSignals(True)
                self.advanced_embedding_toggle.setChecked(False)
                self.advanced_embedding_toggle.blockSignals(False)
                return
        set_advanced_embedding_unlocked(bool(checked))
        if hasattr(self, "advanced_embedding_panel"):
            self.advanced_embedding_panel.setVisible(bool(checked))

    def _build_embedding_mode_menu(self) -> None:
        if not hasattr(self, "embedding_mode_selector"):
            return
        items = [
            (f"{spec.label} — {spec.short_description}", spec.mode_id)
            for spec in list_mode_specs()
        ]
        self._build_prestige_menu(
            self.embedding_mode_selector,
            items,
            self._on_embedding_mode_selected,
        )
        self._sync_embedding_mode_selector()

    def _sync_embedding_mode_selector(self) -> None:
        if not hasattr(self, "embedding_mode_selector"):
            return
        spec = get_mode_spec(get_embedding_mode())
        self.embedding_mode_selector.setText(spec.label)
        if hasattr(self, "embedding_mode_description"):
            self.embedding_mode_description.setText(spec.short_description)

    def _on_embedding_mode_selected(self, mode_id: str) -> None:
        mode_id = normalize_mode_id(str(mode_id or ""))
        previous_mode = normalize_mode_id(get_embedding_mode())
        if mode_id == previous_mode and not resolve_active_gguf_path():
            self._sync_embedding_mode_selector()
            return

        spec = get_mode_spec(mode_id)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self.window(),
            f"Switch to {spec.label}?",
            format_embedding_mode_switch_confirm_body(mode_id),
            is_dark=is_dark,
            tone="danger",
            dialog_width=460,
        )
        if not dlg.exec():
            self._sync_embedding_mode_selector()
            return

        needs_download = (
            not gguf_override_available() and not preset_embedder_ready(mode_id=mode_id)
        )
        if needs_download:
            self._download_preset_then_switch(mode_id, previous_mode)
        else:
            self._commit_embedding_mode_switch(mode_id, previous_mode)

    def _commit_embedding_mode_switch(self, mode_id: str, previous_mode: str) -> None:
        set_embedding_mode(mode_id)
        self._sync_embedding_mode_selector()
        self.embedding_mode_change_requested.emit(mode_id, previous_mode)

    def _download_preset_then_switch(self, mode_id: str, previous_mode: str) -> None:
        if getattr(self, "_embedding_mode_warmup_worker", None) is not None:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Search models busy",
                "Search model download is already in progress.",
                is_dark=is_dark,
            ).exec()
            self._sync_embedding_mode_selector()
            return

        spec = get_mode_spec(mode_id)
        win = self.window()
        detail = f"Downloading {spec.label} search model…"
        if hasattr(win, "begin_background_progress"):
            win.begin_background_progress(detail)
        if hasattr(win, "update_status"):
            win.update_status(detail)

        worker = EmbeddingWarmupWorker(mode_id=mode_id)
        self._embedding_mode_warmup_worker = worker
        is_dark = getattr(self.window(), "_is_dark_theme", True)

        def _finish_download_ui() -> None:
            if hasattr(win, "finish_background_progress"):
                win.finish_background_progress()
            if hasattr(win, "update_status"):
                win.update_status("Idle", force=True)

        def _on_ok() -> None:
            self._embedding_mode_warmup_worker = None
            _finish_download_ui()
            self._commit_embedding_mode_switch(mode_id, previous_mode)

        def _on_failed(err: str) -> None:
            self._embedding_mode_warmup_worker = None
            _finish_download_ui()
            self._sync_embedding_mode_selector()
            PrestigeDialog(
                self.window(),
                "Search model not ready",
                str(err or format_search_preset_download_failure(mode_id)),
                is_dark=is_dark,
                tone="danger",
            ).exec()

        worker.finished_ok.connect(_on_ok)
        worker.failed.connect(_on_failed)
        worker.start()

    def _sync_embedding_models_dir_label(self) -> None:
        if hasattr(self, "embedding_dir_label"):
            self.embedding_dir_label.setText(get_embedding_models_dir())

    def _refresh_embedding_gguf_list(self) -> None:
        if not hasattr(self, "embedding_gguf_list"):
            return
        self.embedding_gguf_list.clear()
        active = resolve_active_gguf_path()
        try:
            active_norm = str(Path(active).resolve()) if active else ""
        except OSError:
            active_norm = active or ""

        for entry in list_selectable_embedding_models():
            item = QListWidgetItem(entry.display_name)
            item.setData(Qt.ItemDataRole.UserRole, entry.path)
            item.setData(EMBEDDING_ENTRY_DELETABLE_ROLE, entry.is_deletable)
            self.embedding_gguf_list.addItem(item)
            try:
                if active_norm and str(Path(entry.path).resolve()) == active_norm:
                    self.embedding_gguf_list.setCurrentItem(item)
            except OSError:
                if entry.path == active:
                    self.embedding_gguf_list.setCurrentItem(item)

    def _sync_active_embedding_label(self) -> None:
        if not hasattr(self, "active_embedding_model_lbl"):
            return
        gguf = resolve_active_gguf_path()
        if gguf and os.path.isfile(gguf):
            self.active_embedding_model_lbl.setText(
                f"{os.path.basename(gguf)} (custom GGUF override active)"
            )
            return
        spec = get_mode_spec(get_embedding_mode())
        self.active_embedding_model_lbl.setText(
            f"{spec.label} preset ({spec.fastembed_model})"
        )

    def _on_refresh_embedding_gguf_clicked(self) -> None:
        self._sync_embedding_models_dir_label()
        self._refresh_embedding_gguf_list()
        self._sync_active_embedding_label()
        self._sync_embedding_mode_selector()

    def _reload_embedder_from_settings(self) -> None:
        self.embedding_model_changed.emit()

    def _apply_selected_embedding_gguf(self) -> None:
        item = self.embedding_gguf_list.currentItem()
        if not item:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "No model",
                "Select an embedding model from the list.",
                is_dark=is_dark,
            ).exec()
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        ok, msg = validate_embedding_model_path(path)
        if not ok:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Invalid embedding model",
                msg or "That file cannot be used as the embedding model.",
                is_dark=is_dark,
            ).exec()
            return

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self.window(),
            "Use custom embedding model?",
            "Switching will reprocess your library and memories. "
            "This can take from a few minutes to several hours for large libraries. "
            "Progress appears in the banner below the top bar and on the Library page.\n\n"
            "Continue?",
            is_dark=is_dark,
            tone="danger",
            dialog_width=420,
        )
        if not dlg.exec():
            return

        set_embedding_model_path(path)
        self._sync_active_embedding_label()
        self._sync_embedding_mode_selector()
        self._reload_embedder_from_settings()

    def _delete_selected_embedding_gguf(self) -> None:
        item = self.embedding_gguf_list.currentItem()
        if not item:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "No model",
                "Select an embedding model to delete.",
                is_dark=is_dark,
            ).exec()
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not item.data(EMBEDDING_ENTRY_DELETABLE_ROLE):
            return
        if not path or not os.path.isfile(path):
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Missing file",
                "That file is not available on disk.",
                is_dark=is_dark,
            ).exec()
            return
        name = os.path.basename(path)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self.window(),
            "Delete embedding model",
            f'Permanently delete "{name}" from models/embedding/? This cannot be undone.',
            is_dark=is_dark,
        )
        if not dlg.exec():
            return
        try:
            os.remove(path)
        except OSError as e:
            logger.error("Failed to delete embedding GGUF %s: %s", path, e)
            PrestigeDialog(
                self.window(),
                "Delete failed",
                str(e),
                is_dark=is_dark,
            ).exec()
            return

        active = resolve_active_gguf_path()
        try:
            active_resolved = str(Path(active).resolve()) if active else ""
            deleted_resolved = str(Path(path).resolve())
            was_active = bool(active_resolved and active_resolved == deleted_resolved)
        except OSError:
            was_active = active == path
        if was_active:
            set_embedding_model_path("")
            self._reload_embedder_from_settings()

        self._sync_active_embedding_label()
        self._refresh_embedding_gguf_list()
        self._sync_embedding_mode_selector()
