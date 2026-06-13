"""Settings handler mixin: KnowledgeHandlersMixin (embedding model + triggers)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem

from core.app_settings import (
    get_advanced_embedding_unlocked,
    set_advanced_embedding_unlocked,
    set_embedding_model_path,
)
from core.embedding_models import (
    get_embedding_models_dir,
    is_protected_embedding_model,
    list_selectable_embedding_models,
    resolve_active_embedding_path,
    validate_embedding_model_path,
)
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
                "Swapping the embedding model affects RAG, memory search, and routing "
                "centroids. Models must be GGUF files placed in the embedding folder.\n\n"
                "If the new model outputs a different vector size, your LanceDB index "
                "will be reset automatically — re-ingest library documents afterward.\n\n"
                "The bundled Nomic Embed v1.5 default cannot be deleted.\n\nContinue?",
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

    def _sync_embedding_models_dir_label(self) -> None:
        if hasattr(self, "embedding_dir_label"):
            self.embedding_dir_label.setText(get_embedding_models_dir())

    def _refresh_embedding_gguf_list(self) -> None:
        if not hasattr(self, "embedding_gguf_list"):
            return
        self.embedding_gguf_list.clear()
        active = resolve_active_embedding_path()
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
        path = resolve_active_embedding_path()
        if not path or not os.path.isfile(path):
            self.active_embedding_model_lbl.setText("— (bundled default missing)")
            return
        base = os.path.basename(path)
        if is_protected_embedding_model(path):
            self.active_embedding_model_lbl.setText(f"{base} (bundled default)")
        else:
            self.active_embedding_model_lbl.setText(f"{base} (custom)")

    def _on_refresh_embedding_gguf_clicked(self) -> None:
        self._sync_embedding_models_dir_label()
        self._refresh_embedding_gguf_list()
        self._sync_active_embedding_label()

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
        if is_protected_embedding_model(path):
            set_embedding_model_path("")
        else:
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
            set_embedding_model_path(path)
        self._sync_active_embedding_label()
        self._reload_embedder_from_settings()

    def _reset_embedding_to_default(self) -> None:
        set_embedding_model_path("")
        self._refresh_embedding_gguf_list()
        self._sync_active_embedding_label()
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
        if is_protected_embedding_model(path):
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Protected model",
                "The bundled Nomic Embed v1.5 default cannot be deleted. Use Reset to "
                "default to stop using a custom embedding model.",
                is_dark=is_dark,
            ).exec()
            return
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

        active = resolve_active_embedding_path()
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
