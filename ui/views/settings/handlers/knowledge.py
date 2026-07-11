"""Settings handler mixin: KnowledgeHandlersMixin (embedding model + triggers)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QLineEdit, QListWidgetItem

from core.app_settings import (
    KEY_DEEP_RESEARCH_ENABLED,
    KEY_EXTERNAL_KNOWLEDGE_V2_ENABLED,
    KEY_INTERNAL_CORPUS_KNOWLEDGE_ENABLED,
    KEY_KNOWLEDGE_SOURCE_PREFERENCES,
    KEY_KNOWLEDGE_PROVIDER_CREDENTIALS,
    KEY_RESEARCH_MAP_ENABLED,
    get_advanced_embedding_unlocked,
    get_embedding_mode,
    get_knowledge_source_preferences,
    set_advanced_embedding_unlocked,
    set_deep_research_enabled,
    set_embedding_model_path,
    set_embedding_mode,
    set_external_knowledge_v2_enabled,
    set_internal_corpus_knowledge_enabled,
    set_knowledge_source_preferences,
    set_research_map_enabled,
)
from core.knowledge.connectors.base import list_connector_types
from core.knowledge.credentials import (
    clear_provider_api_key,
    env_override_active,
    set_provider_api_key,
)
from core.knowledge.provider_status import record_provider_credential_test
from core.knowledge.provider_credential_test import test_provider_credential
from core.knowledge.provider_credentials import (
    get_provider_credential_spec,
    provider_id_for_adapter,
)
from core.knowledge.source_preferences import set_adapter_enabled
from ui.components.provider_credential_dialog import open_provider_credential_dialog
from ui.views.settings.sections.knowledge_provider_credentials import sync_provider_credential_rows
from ui.views.settings.sections.knowledge_provider_status import sync_provider_status_panel
from ui.views.settings.sections.knowledge_sources import (
    sync_knowledge_source_checkboxes,
    sync_live_source_rows,
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


class ProviderCredentialTestWorker(QThread):
    """Background probe for Settings → Provider credentials Test button."""

    finished = pyqtSignal(str, bool, str)

    def __init__(
        self,
        *,
        provider_id: str,
        override_secret: str | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._provider_id = provider_id
        self._override_secret = override_secret

    def run(self) -> None:
        result = test_provider_credential(
            self._provider_id,
            override_secret=self._override_secret,
        )
        self.finished.emit(self._provider_id, result.ok, result.message)


class KnowledgeHandlersMixin:
    """Embedding model loader and related Knowledge settings behavior."""

    def _emit_external_settings_changed(self, *keys: str) -> None:
        if hasattr(self, "external_settings_reloaded"):
            self.external_settings_reloaded.emit(set(keys))

    def _on_external_knowledge_v2_toggled(self, checked: bool) -> None:
        set_external_knowledge_v2_enabled(checked)
        self._emit_external_settings_changed(KEY_EXTERNAL_KNOWLEDGE_V2_ENABLED)

    def _on_internal_corpus_knowledge_toggled(self, checked: bool) -> None:
        set_internal_corpus_knowledge_enabled(checked)
        self._emit_external_settings_changed(KEY_INTERNAL_CORPUS_KNOWLEDGE_ENABLED)

    def _on_research_map_toggled(self, checked: bool) -> None:
        set_research_map_enabled(checked)
        self._emit_external_settings_changed(KEY_RESEARCH_MAP_ENABLED)

    def _on_deep_research_enabled_toggled(self, checked: bool) -> None:
        set_deep_research_enabled(checked)
        self._emit_external_settings_changed(KEY_DEEP_RESEARCH_ENABLED)

    def _on_knowledge_source_toggled(
        self,
        service_id: str,
        adapter_id: str,
        checked: bool,
    ) -> None:
        prefs = set_adapter_enabled(
            get_knowledge_source_preferences(),
            service_id=service_id,
            adapter_id=adapter_id,
            enabled=checked,
        )
        set_knowledge_source_preferences(prefs)
        sync_live_source_rows(self)
        self._emit_external_settings_changed(KEY_KNOWLEDGE_SOURCE_PREFERENCES)
        if checked:
            self._maybe_nudge_key_required_source(adapter_id)

    def _on_live_source_configure_clicked(self, adapter_id: str) -> None:
        provider_id = provider_id_for_adapter(adapter_id)
        if not provider_id:
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        open_provider_credential_dialog(
            self,
            provider_id,
            is_dark=is_dark,
            parent=self.window(),
        )

    def _maybe_nudge_key_required_source(self, adapter_id: str) -> None:
        """One-time nudge when enabling a key-required source without a key."""
        from core.knowledge.adapters.catalog import get_adapter_entry
        from core.knowledge.credentials import resolve_credential
        from core.knowledge.source_access_summary import summarize_source_access

        entry = get_adapter_entry(adapter_id)
        if entry is None:
            return
        summary = summarize_source_access(entry)
        if summary.badge != "key_required":
            return

        provider_id = summary.provider_id
        if not provider_id:
            return
        cred = resolve_credential(provider_id)
        if cred.secret:
            return

        spec = get_provider_credential_spec(provider_id)
        label = spec.label if spec is not None else entry.label
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self.window(),
            f"{label} needs an API key",
            f"{label} requires an API key to return results.\n\nConfigure now?",
            is_dark=is_dark,
            confirm_text="CONFIGURE",
            cancel_text="NOT NOW",
            dialog_width=460,
        )
        if dlg.exec():
            open_provider_credential_dialog(
                self,
                provider_id,
                is_dark=is_dark,
                parent=self.window(),
            )

    def _on_knowledge_setup_callout_dismiss(self) -> None:
        self.knowledge_setup_callout_dismissed = True
        shell = getattr(self, "knowledge_setup_callout_shell", None)
        if shell is not None:
            shell.setVisible(False)

    def _provider_credential_field(self, provider_id: str) -> QLineEdit | None:
        fields = getattr(self, "knowledge_provider_key_fields", None)
        if not isinstance(fields, dict):
            return None
        return fields.get(provider_id)

    def _on_provider_credential_editing_finished(self, provider_id: str) -> None:
        if env_override_active(provider_id):
            sync_provider_credential_rows(self)
            sync_live_source_rows(self)
            return
        field = self._provider_credential_field(provider_id)
        if field is None:
            return
        text = field.text().strip()
        if not text:
            return
        set_provider_api_key(provider_id, text)
        field.clear()
        sync_provider_credential_rows(self)
        sync_live_source_rows(self)
        sync_provider_status_panel(self)
        self._emit_external_settings_changed(KEY_KNOWLEDGE_PROVIDER_CREDENTIALS)

    def _on_provider_credential_clear(self, provider_id: str) -> None:
        clear_provider_api_key(provider_id)
        field = self._provider_credential_field(provider_id)
        if field is not None:
            field.clear()
        sync_provider_credential_rows(self)
        sync_live_source_rows(self)
        sync_provider_status_panel(self)
        self._emit_external_settings_changed(KEY_KNOWLEDGE_PROVIDER_CREDENTIALS)

    def _on_provider_credential_signup(self, provider_id: str) -> None:
        spec = get_provider_credential_spec(provider_id)
        if spec is None or not spec.signup_url:
            return
        QDesktopServices.openUrl(QUrl(spec.signup_url))

    def _on_provider_credential_test(self, provider_id: str) -> None:
        if getattr(self, "_provider_credential_test_worker", None) is not None:
            return
        field = self._provider_credential_field(provider_id)
        override = field.text().strip() if field is not None else ""
        if override:
            set_provider_api_key(provider_id, override)
            if field is not None:
                field.clear()
            sync_provider_credential_rows(self)
            sync_live_source_rows(self)
            self._emit_external_settings_changed(KEY_KNOWLEDGE_PROVIDER_CREDENTIALS)

        worker = ProviderCredentialTestWorker(
            provider_id=provider_id,
            override_secret=None,
        )
        self._provider_credential_test_worker = worker
        is_dark = getattr(self.window(), "_is_dark_theme", True)

        def _finish(pid: str, ok: bool, message: str) -> None:
            self._provider_credential_test_worker = None
            record_provider_credential_test(pid, ok=ok, message=message)
            sync_provider_credential_rows(self)
            sync_live_source_rows(self)
            sync_provider_status_panel(self)
            PrestigeDialog(
                self.window(),
                "Connection test succeeded" if ok else "Connection test failed",
                message,
                is_dark=is_dark,
                tone="default" if ok else "danger",
            ).exec()

        worker.finished.connect(_finish)
        worker.start()

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

    def _build_custom_source_connector_menu(self) -> None:
        if not hasattr(self, "custom_source_connector_selector"):
            return
        items = [(connector_id, connector_id) for connector_id in list_connector_types()]
        self._build_prestige_menu(
            self.custom_source_connector_selector,
            items,
            self._on_custom_source_connector_selected,
        )
        self._sync_custom_source_connector_selector()

    def _on_custom_source_connector_selected(self, connector_id: str) -> None:
        self._custom_source_connector_id = str(connector_id or "rest_json")

    def _sync_custom_source_connector_selector(self) -> None:
        if not hasattr(self, "custom_source_connector_selector"):
            return
        from ui.views.settings.widgets import refit_settings_selector_width

        connector_id = getattr(self, "_custom_source_connector_id", "rest_json")
        self.custom_source_connector_selector.setText(connector_id)
        refit_settings_selector_width(self.custom_source_connector_selector)

    def _sync_embedding_mode_selector(self) -> None:
        if not hasattr(self, "embedding_mode_selector"):
            return
        spec = get_mode_spec(get_embedding_mode())
        self.embedding_mode_selector.setText(spec.label)
        if hasattr(self, "embedding_mode_description"):
            self.embedding_mode_description.setText(spec.short_description)
        if hasattr(self, "_sync_bootstrap_download_visibility"):
            self._sync_bootstrap_download_visibility()

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
            message = str(err or "").strip()
            if not message:
                message = format_search_preset_download_failure(
                    mode_id,
                    during_mode_switch=True,
                )
            PrestigeDialog(
                self.window(),
                "Search model not ready",
                message,
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

    def _save_knowledge_preset(self) -> None:
        from ui.views.settings.sections.knowledge_presets import save_preset_from_host

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        try:
            save_preset_from_host(self)
        except Exception as exc:
            PrestigeDialog(self.window(), "Preset error", str(exc), is_dark=is_dark).exec()

    def _delete_knowledge_preset(self) -> None:
        from ui.views.settings.sections.knowledge_presets import delete_selected_preset_from_host

        delete_selected_preset_from_host(self)

    def _explain_knowledge_preset(self) -> None:
        from ui.views.settings.sections.knowledge_presets import explain_selected_preset_from_host

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        text = explain_selected_preset_from_host(self)
        if not text:
            PrestigeDialog(
                self.window(),
                "Explain preset",
                "Select a preset in the table first.",
                is_dark=is_dark,
            ).exec()
            return
        PrestigeDialog(
            self.window(),
            "Explain preset",
            text,
            is_dark=is_dark,
            dialog_width=520,
        ).exec()

    def _build_retrieval_profile_menu(self) -> None:
        if not hasattr(self, "retrieval_profile_selector"):
            return
        from core.knowledge.retrieval_profiles import list_profile_specs

        items = [
            (f"{spec.label} — {spec.short_description}", spec.id)
            for spec in list_profile_specs()
        ]
        self._build_prestige_menu(
            self.retrieval_profile_selector,
            items,
            self._on_retrieval_profile_selected,
        )
        self._sync_retrieval_profile_selector()

    def _on_retrieval_profile_selected(self, profile_id: str) -> None:
        from core.app_settings import set_retrieval_profile

        set_retrieval_profile(str(profile_id))

    def _sync_retrieval_profile_selector(self) -> None:
        if not hasattr(self, "retrieval_profile_selector"):
            return
        from core.app_settings import get_retrieval_profile
        from core.knowledge.retrieval_profiles import get_profile_spec
        from ui.views.settings.widgets import refit_settings_selector_width

        spec = get_profile_spec(get_retrieval_profile())
        self.retrieval_profile_selector.setText(spec.label)
        if hasattr(self, "retrieval_profile_description"):
            self.retrieval_profile_description.setText(spec.short_description)
        refit_settings_selector_width(self.retrieval_profile_selector)

    def _save_custom_source(self) -> None:
        from ui.views.settings.sections.knowledge_custom_sources import save_custom_source_from_host

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        try:
            save_custom_source_from_host(self)
            from ui.views.settings.sections.knowledge_presets import (
                _refresh_preset_sources_hint,
            )

            _refresh_preset_sources_hint(self)
        except Exception as exc:
            PrestigeDialog(self.window(), "Source error", str(exc), is_dark=is_dark).exec()

    def _test_custom_source(self) -> None:
        from ui.views.settings.sections.knowledge_custom_sources import test_custom_source_from_host

        test_custom_source_from_host(self)

    def _delete_custom_source(self) -> None:
        from ui.views.settings.sections.knowledge_custom_sources import delete_custom_source_from_host

        delete_custom_source_from_host(self)

    def _on_open_custom_sources_settings_clicked(self) -> None:
        self.select_settings_section("knowledge", anchor="knowledge_custom_sources")

    def _on_open_my_knowledge_settings_clicked(self) -> None:
        self.select_settings_section("knowledge", anchor="knowledge_presets")

    def _refresh_retrieval_trace(self) -> None:
        panel = getattr(self, "retrieval_trace_panel", None)
        if panel is not None:
            panel.refresh()

    def _export_knowledge_pack(self) -> None:
        from core.knowledge.knowledge_pack import export_knowledge_pack_to_file
        from core.paths import user_data_root

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        path = user_data_root() / "knowledge-pack.json"
        export_knowledge_pack_to_file(path)
        PrestigeDialog(
            self.window(),
            "Knowledge pack exported",
            f"Saved to {path}",
            is_dark=is_dark,
        ).exec()

    def _import_knowledge_pack(self) -> None:
        from core.knowledge.knowledge_pack import import_knowledge_pack_from_file
        from core.paths import user_data_root

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        path = user_data_root() / "knowledge-pack.json"
        if not path.is_file():
            PrestigeDialog(
                self.window(),
                "Import failed",
                f"No pack found at {path}",
                is_dark=is_dark,
            ).exec()
            return
        summary = import_knowledge_pack_from_file(path)
        PrestigeDialog(
            self.window(),
            "Knowledge pack imported",
            str(summary),
            is_dark=is_dark,
        ).exec()
        self._refresh_retrieval_trace()
